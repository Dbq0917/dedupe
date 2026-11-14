"""
客户主数据高精确度合并示例：双阈值 + 人工复核队列 + 强制翻案。

业务原则：宁可少并，不能错并（precision 优先）。

三个机制：
1. 双阈值：score >= auto_threshold 自动并；
   review_threshold <= score < auto_threshold 进人工复核队列；
   score < review_threshold 直接不并。
2. 阈值标定：用一小批人工标注过的验证对，选出满足 precision
   目标的最低阈值（choose_threshold），不要凭感觉拍阈值。
3. 人工翻案两条路：
   - 软反馈：人工判定经 mark_pairs 回灌训练（错并对标 distinct、
     漏并对标 match），下一轮模型自动避开同类错误；
   - 硬翻案：never_merge / always_merge 覆盖表，聚类后确定性执行，
     保证翻案必生效，不受模型分数波动影响。

运行演示：python examples/customer_master_matching.py
"""

from __future__ import annotations

import itertools
import warnings
from collections.abc import Callable, Iterable, Sequence

import dedupe

# ---------------------------------------------------------------------------
# 阈值标定
# ---------------------------------------------------------------------------


def choose_threshold(
    scores: Sequence[float],
    labels: Sequence[int],
    precision_target: float = 0.99,
) -> tuple[float, float, float] | None:
    """在人工标注的验证对上扫描阈值。

    返回 (threshold, precision, recall)：满足 precision >= precision_target
    的阈值中 recall 最高的那个。没有任何阈值达标时返回 None，
    调用方应把自动合并阈值抬到 1.0 以上（即全部转人工）。
    """
    best: tuple[float, float, float] | None = None
    for candidate in sorted(set(scores)):
        tp = sum(1 for s, y in zip(scores, labels) if s >= candidate and y == 1)
        fp = sum(1 for s, y in zip(scores, labels) if s >= candidate and y == 0)
        fn = sum(1 for s, y in zip(scores, labels) if s < candidate and y == 1)
        if tp + fp == 0:
            continue
        precision = tp / (tp + fp)
        recall = tp / (tp + fn) if tp + fn else 0.0
        if precision >= precision_target and (best is None or recall > best[2]):
            best = (float(candidate), precision, recall)
    return best


# ---------------------------------------------------------------------------
# 双阈值分带
# ---------------------------------------------------------------------------


def split_bands(scores, auto_threshold: float, review_threshold: float):
    """把模型打分的候选对分成 自动并 / 人工复核 两档，其余忽略。"""
    auto = scores[scores["score"] >= auto_threshold]
    review = scores[
        (scores["score"] >= review_threshold) & (scores["score"] < auto_threshold)
    ]
    return auto, review


def review_queue(review_scores, data: dict) -> list[dict]:
    """生成人工复核队列（含记录原文，方便直接接前端或导出 Excel）。"""
    queue = []
    for pair, score in zip(review_scores["pairs"], review_scores["score"]):
        id_a, id_b = pair
        if hasattr(id_a, "item"):
            id_a, id_b = id_a.item(), id_b.item()
        queue.append(
            {
                "id_a": id_a,
                "id_b": id_b,
                "score": round(float(score), 4),
                "record_a": data[id_a],
                "record_b": data[id_b],
            }
        )
    queue.sort(key=lambda row: row["score"], reverse=True)
    return queue


# ---------------------------------------------------------------------------
# 硬翻案：覆盖表
# ---------------------------------------------------------------------------


def apply_overrides(
    clusters: Iterable[Sequence],
    never_merge: Iterable[tuple] = (),
    always_merge: Iterable[tuple] = (),
    priority: Callable | None = None,
) -> list[tuple]:
    """在聚类结果上确定性地执行人工裁定。

    always_merge：指定的 id 必须同组（并查集合并）。
    never_merge：指定的 id 对不得同组；冲突时把 priority 较低的一方
    踢成单例。priority(id) 返回值越小越可信（默认按 id 字符串序）。
    """
    parent: dict = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        parent[find(a)] = find(b)

    for cluster in clusters:
        first, *rest = cluster
        for member in rest:
            union(first, member)
    for a, b in always_merge:
        union(a, b)

    rank = priority or (lambda x: str(x))
    never = [tuple(pair) for pair in never_merge]

    changed = True
    while changed:  # 踢人可能解开其它冲突，迭代到无冲突为止
        changed = False
        for a, b in never:
            if a != b and find(a) == find(b):
                evict = max((a, b), key=rank)
                parent[evict] = evict  # 踢成单例
                changed = True

    groups: dict = {}
    for member in parent:
        groups.setdefault(find(member), []).append(member)
    return [tuple(members) for members in groups.values()]


# ---------------------------------------------------------------------------
# 训练（含错并反馈回灌）
# ---------------------------------------------------------------------------

VARIABLES = [
    dedupe.variables.ShortString("name"),
    dedupe.variables.ShortString("city", has_missing=True),
    # 关键教训：只有 name+city 时，"阿里巴巴 vs 阿里妈妈"这类硬对
    # 在特征层面不可分，任何阈值都做不到不错并。地址/电话/税号等
    # 区分度高的字段才是提高自动合并覆盖率的根本手段。
    dedupe.variables.ShortString("address", has_missing=True),
]


def build_matcher(data: dict, feedback: dict | None = None) -> dedupe.Dedupe:
    """feedback 即人工裁定：{"match": [(rec, rec)], "distinct": [(rec, rec)]}。

    错并的对放 distinct，漏并的对放 match——这是"软翻案"，
    让模型下一轮自己避开同类错误。
    """
    matcher = dedupe.Dedupe(VARIABLES, num_cores=1)
    if feedback:
        # 注意顺序：在 prepare_training 之前 mark_pairs，
        # 作为已有训练数据进入主动学习器；之后调用则要求
        # 记录必须已被采样进学习器，否则报错。
        matcher.mark_pairs(feedback)
    matcher.prepare_training(data, sample_size=500)
    matcher.train()
    return matcher


# ---------------------------------------------------------------------------
# 演示
# ---------------------------------------------------------------------------


def _demo_data() -> dict:
    rows = [
        # (source, name, city, address) —— 三个来源，同一家公司写法不同
        ("erp", "阿里巴巴集团控股有限公司", "杭州", "余杭区文一西路969号"),
        ("crm", "阿里巴巴（中国）网络技术有限公司", "杭州", "杭州市文一西路969号"),
        ("web", "Alibaba Group Holding Limited", "Hangzhou", "969 Wenyi West Road"),
        ("erp", "腾讯科技（深圳）有限公司", "深圳", "南山区科技园腾讯大厦"),
        ("crm", "深圳市腾讯计算机系统有限公司", "深圳", "南山区科技园"),
        ("web", "Tencent Holdings Ltd.", "Shenzhen", "Tencent Bldg, Nanshan"),
        ("erp", "北京京东世纪贸易有限公司", "北京", "亦庄科创十一街18号"),
        ("crm", "京东集团股份有限公司", "北京", "北京市亦庄科创十一街"),
        ("web", "JD.com, Inc.", "Beijing", "18 Kechuang 11 St, Yizhuang"),
        # 名字相近但确实是不同公司 —— 历史上被错并的一对
        ("erp", "杭州阿里妈妈软件服务有限公司", "杭州", "西湖区文三路478号"),
        ("crm", "阿里妈妈（杭州）科技有限公司", "杭州", "杭州市文三路478号"),
        ("web", "杭州阿里巴巴广告有限公司", "杭州", "文三路478号（西湖区）"),
    ]
    data = {}
    for i, (source, name, city, address) in enumerate(rows):
        data[f"{source}_{i:03d}"] = {
            "name": name,
            "city": city,
            "address": address,
            "source": source,
        }
    return data


# 真实的公司分组（演示用的"标准答案"）
TRUE_CLUSTERS = [
    ["erp_000", "crm_001", "web_002"],  # 阿里巴巴集团
    ["erp_003", "crm_004", "web_005"],  # 腾讯
    ["erp_006", "crm_007", "web_008"],  # 京东
    ["erp_009", "crm_010", "web_011"],  # 阿里妈妈系（独立公司，历史上被错并）
]


def _ground_truth_pairs() -> tuple[list, list]:
    """由标准答案生成 同公司对 / 跨公司对（id 形式）。"""
    matches, distincts = [], []
    for i, cluster_a in enumerate(TRUE_CLUSTERS):
        matches.extend(itertools.combinations(cluster_a, 2))
        for cluster_b in TRUE_CLUSTERS[i + 1 :]:
            distincts.extend(itertools.product(cluster_a, cluster_b))
    return matches, distincts


def main() -> None:
    warnings.filterwarnings("ignore", category=UserWarning)
    data = _demo_data()
    match_ids, distinct_ids = _ground_truth_pairs()

    # 人工标注：留出 1/3 做验证集（模型没见过的对），其余回灌训练。
    # 关键：错并对（阿里妈妈系 vs 阿里巴巴系）必须标 distinct 回灌，
    # 否则模型学不到"名字像但不是一家"的边界。
    train_match_ids = match_ids[1::2]
    val_match_ids = match_ids[::2]
    train_distinct_ids = [p for i, p in enumerate(distinct_ids) if i % 3 != 0]
    val_distinct_ids = distinct_ids[::3]
    feedback = {
        "match": [(data[a], data[b]) for a, b in train_match_ids],
        "distinct": [(data[a], data[b]) for a, b in train_distinct_ids],
    }
    matcher = build_matcher(data, feedback)

    # --- 阈值标定：用人工标注的验证对选阈值，而不是拍脑袋 ---
    validation = (
        [(a, b, 1) for a, b in val_match_ids]
        + [(a, b, 0) for a, b in val_distinct_ids]
    )
    val_pairs = [((a, data[a]), (b, data[b])) for a, b, _ in validation]
    val_scores = matcher.score(iter(val_pairs))
    result = choose_threshold(
        [float(s) for s in val_scores["score"]],
        [label for _, _, label in validation],
        precision_target=0.99,
    )
    if result is None:
        auto_threshold = 1.1  # 没有阈值达标：全部转人工，不自动并
        print("验证集上没有满足 precision 目标的阈值，自动合并已关闭")
    else:
        auto_threshold, precision, recall = result
        print(
            f"标定结果: auto_threshold={auto_threshold:.3f} "
            f"(precision={precision:.3f}, recall={recall:.3f})"
        )
    # 复核带下沿按复核人力定：越低漏并越少、人工量越大。
    # 注意：被 blocking 规则挡掉的候选对根本不会被打分，
    # 漏并只能依靠 train(recall=...) 的 blocking 覆盖率来兜底。
    review_threshold = 0.05

    # --- 分带 ---
    all_scores = matcher.score(matcher.pairs(data))
    auto, review = split_bands(all_scores, auto_threshold, review_threshold)

    # --- 自动合并 + 硬翻案 ---
    clusters = matcher.cluster(auto, threshold=min(auto_threshold, 1.0))
    cluster_ids = [ids for ids, _ in clusters]
    final = apply_overrides(
        cluster_ids,
        never_merge=[("erp_000", "erp_009")],  # 人工裁定：永不合并，必生效
    )

    print("\n自动合并结果:")
    for group in final:
        if len(group) > 1:
            names = [data[i]["name"] for i in group]
            print(f"  {[str(i) for i in group]} -> {names}")

    queue = review_queue(review, data)
    print(f"\n人工复核队列 ({len(queue)} 对):")
    for row in queue:
        print(
            f"  {row['score']:.3f}  {row['record_a']['name']}  <=>  "
            f"{row['record_b']['name']}"
        )

    # 复核结论落库后：mark_pairs 回灌 + 更新 never_merge/always_merge 表，
    # 并 write_training 持久化，下一轮训练自动生效。
    with open("training.json", "w") as f:
        matcher.write_training(f)
    print("\n训练数据已持久化到 training.json")


if __name__ == "__main__":
    main()
