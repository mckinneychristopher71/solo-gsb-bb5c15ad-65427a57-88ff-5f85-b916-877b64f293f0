"""共用标签模板分组（临时预览，不改写审查记录）。

把每条**历史审查记录**视为一个产品，复用其当前配方（两版记录取标签
核对对象，即新版配方；单版取现版）的**最终过敏原集合**：

- 用户选择 2～12 个**时效正常**的产品，并为每个产品设置最多允许的
  附加声明数量（该产品共用模板上除自身过敏原外，最多额外标注多少种）；
- 一组产品的**共用标签模板**定义为组内各产品过敏原集合的**并集**，
  因此某产品在该组模板下的附加声明 = 模板并集 − 该产品自身过敏原；
- 仅当组内**每个**产品的附加声明数都不超过各自上限时，该组才可成组；
- 用**子集动态规划**求**模板数量最少**的完整分组（每个产品恰好归入
  一组，单产品组恒可成组，故完整分组必然存在）；
- 模板数并列时，组内成员按产品 ID 升序、组间按各组产品 ID 序列做
  字典序比较（等价于以各组最小产品 ID 排序后逐组比较），取唯一的
  最小分组，保证结果确定。

记录缺失（不存在 / 已删除）或处于**需复核**状态（旧记录无指纹、相关
原料映射已变更）时禁止参与，并逐条指出原因。整个模块只读：不写入
字典，也不改写任何审查记录。
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from . import db

MIN_PRODUCTS = 2
MAX_PRODUCTS = 12

# 需复核原因 → 展示文案（与历史记录列表的口径一致）
_RECHECK_REASONS = {
    "legacy": "旧记录无字典指纹",
    "stale": "相关原料的过敏原映射已变更",
}


class LabelGroupError(ValueError):
    """标签模板分组请求无法完成（映射为 HTTP 400）。

    problems 逐条列出禁止参与的记录及原因，供页面定位。
    """

    def __init__(self, message: str, problems: Optional[List[dict]] = None):
        super().__init__(message)
        self.problems = problems or []


def _blocked_message(reason: Optional[str]) -> str:
    base = _RECHECK_REASONS.get(reason or "", "字典时效状态未知")
    return f"需复核（{base}），请先完成字典时效复核"


def _current_version(review: dict) -> dict:
    """记录当前配方（标签核对对象：有新版取新版，否则取现版）的版本结果。"""
    result = review.get("result") or {}
    return result.get("version_b") or result.get("version_a") or {}


def _product_of(review: dict) -> dict:
    """从审查记录提取产品：ID、名称、取值版本与最终过敏原集合。"""
    result = review.get("result") or {}
    return {
        "id": review["id"],
        "name": review["name"],
        "source_version": result.get("label_check_version")
        or review.get("label_a") or "现版配方",
        "allergens": sorted(_current_version(review).get("allergens", [])),
    }


def list_label_group_products() -> List[dict]:
    """候选产品清单：每条历史记录的当前配方最终过敏原集合与参与资格。"""
    products = []
    for summary in db.list_reviews():
        review = db.get_review(summary["id"])
        needs = bool(review.get("needs_recheck"))
        reason = review.get("recheck_reason")
        products.append({
            **_product_of(review),
            "created_at": review["created_at"],
            "has_two_versions": bool(review.get("text_b")),
            "eligible": not needs,
            "blocked_reason": _blocked_message(reason) if needs else None,
        })
    products.sort(key=lambda p: p["id"])
    return products


def build_grouping(products: List[dict],
                   max_extra: Dict[int, int]) -> dict:
    """对给定产品集合求模板数量最少的完整分组（子集动态规划）。

    products 为 ``[{id, name, allergens, ...}]``（按 ID 升序处理）；
    max_extra 为 ``{产品ID: 附加声明上限}``。一组产品的模板为组内过敏原
    并集，某成员的附加声明数 = 并集大小 − 该成员自身过敏原数（并集恒
    包含自身集合），不超过各自上限时该组才可成组。

    模板数并列时按组内及组间产品 ID 序列取唯一最小分组：组内成员按 ID
    升序，组间按成员 ID 元组做字典序比较（首元素不同时即按各组最小 ID
    排序，首元素相同时继续比较后续 ID）。
    """
    products = sorted(products, key=lambda p: p["id"])
    n = len(products)
    ids = [p["id"] for p in products]
    names = [p["name"] for p in products]
    sets = [set(p.get("allergens", [])) for p in products]
    union_all = sorted(set().union(*sets)) if sets else []
    limits = [int(max_extra[pid]) for pid in ids]

    size = 1 << n
    bit_of = {a: 1 << i for i, a in enumerate(union_all)}
    allergen_bits = [
        sum((bit_of[a] for a in sets[i]), 0) if sets[i] else 0
        for i in range(n)
    ]

    # 预计算每个子集的过敏原并集、成员 ID 元组与成组可行性：
    # 仅当组内每个成员的附加声明数（并集大小 − 自身过敏原数）都不超限。
    group_union = [0] * size
    member_ids: List[Tuple[int, ...]] = [()] * size
    feasible = [False] * size
    for mask in range(1, size):
        low = mask & -mask
        i = low.bit_length() - 1
        rest = mask ^ low
        group_union[mask] = group_union[rest] | allergen_bits[i]
        member_ids[mask] = tuple(sorted(
            ids[j] for j in range(n) if mask & (1 << j)))
        union_size = group_union[mask].bit_count()
        feasible[mask] = all(
            union_size - len(sets[j]) <= limits[j]
            for j in range(n) if mask & (1 << j))

    # 子集 DP：dp[mask] = 覆盖 mask 的最优完整分组（成员 ID 元组的元组，
    # 已规范化为组间按 ID 序列字典序升序）；比较口径为先比模板（组）数，
    # 再比规范化分组的字典序。只枚举含 mask 最小位的末组，每种划分恰好
    # 生成一次（该组携带最小元素，天然排在最前，其余分组已规范化）。
    dp: List[Optional[Tuple[Tuple[int, ...], ...]]] = [None] * size
    dp[0] = ()
    for mask in range(1, size):
        first = mask & -mask
        rest_all = mask ^ first
        sub = rest_all
        best: Optional[Tuple[Tuple[int, ...], ...]] = None
        while True:
            group_mask = mask ^ sub
            if feasible[group_mask]:
                cand_rest = dp[sub]
                if cand_rest is not None:
                    cand = (member_ids[group_mask],) + cand_rest
                    if best is None or (len(cand), cand) < (len(best), best):
                        best = cand
            if sub == 0:
                break
            sub = (sub - 1) & rest_all
        dp[mask] = best

    partition = dp[size - 1]
    # 单产品组恒可行，任何选择集都存在完整分组
    assert partition is not None, "完整分组必然存在"

    index = {pid: i for i, pid in enumerate(ids)}
    groups = []
    for gi, members in enumerate(partition, start=1):
        idxs = [index[pid] for pid in members]
        template_bits = 0
        for i in idxs:
            template_bits |= allergen_bits[i]
        template_set = {union_all[b] for b in range(len(union_all))
                        if template_bits & (1 << b)}
        template = sorted(template_set)
        members_out = []
        for i in idxs:
            extra = sorted(template_set - sets[i])
            members_out.append({
                "id": ids[i],
                "name": names[i],
                "source_version": products[i].get("source_version"),
                "allergens": sorted(sets[i]),
                "extra_allergens": extra,
                "extra_count": len(extra),
                "max_extra": limits[i],
                "within_limit": len(extra) <= limits[i],
            })
        groups.append({
            "index": gi,
            "template": template,
            "template_count": len(template),
            "members": members_out,
        })

    return {
        "products": [{
            "id": ids[i],
            "name": names[i],
            "source_version": products[i].get("source_version"),
            "allergens": sorted(sets[i]),
            "max_extra": limits[i],
        } for i in range(n)],
        "allergen_union": union_all,
        "group_count": len(groups),
        "groups": groups,
    }


def _coerce_limit(review_id, value) -> int:
    """附加声明上限校验：必须是不小于 0 的整数（拒绝浮点 / 非数值）。"""
    if isinstance(value, bool) or not isinstance(value, int):
        raise LabelGroupError(
            f"产品 #{review_id} 的最多附加声明数量必须是不小于 0 的整数"
            f"（当前值：{value!r}）")
    if value < 0:
        raise LabelGroupError(
            f"产品 #{review_id} 的最多附加声明数量不能为负数（当前值：{value}）")
    return value


def build_label_group_preview(items: List[dict]) -> dict:
    """标签模板分组预览的完整编排：校验记录资格后求最少模板分组（只读）。

    items 为 ``[{"review_id": 1, "max_extra": 0}, ...]``。记录缺失或处于
    需复核状态时抛出 :class:`LabelGroupError`，problems 逐条给出原因。
    """
    raw = list(items or [])
    ids = [it.get("review_id") for it in raw]
    if not MIN_PRODUCTS <= len(ids) <= MAX_PRODUCTS:
        raise LabelGroupError(
            f"请选择 {MIN_PRODUCTS}～{MAX_PRODUCTS} 个产品（当前 {len(ids)} 个）")
    if len(set(ids)) != len(ids):
        raise LabelGroupError("同一产品不能重复选择")

    # 先逐条校验附加声明上限，再校验记录资格，避免无效数字被静默接受
    limits: Dict[int, int] = {}
    for it in raw:
        rid = it.get("review_id")
        limits[rid] = _coerce_limit(rid, it.get("max_extra"))

    products, problems = [], []
    for rid in ids:
        try:
            rid_int = int(rid)
        except (TypeError, ValueError):
            problems.append({"id": rid, "name": None, "reason": "invalid",
                             "message": "记录 ID 无效，禁止参与分组"})
            continue
        review = db.get_review(rid_int)
        if review is None:
            problems.append({"id": rid_int, "name": None, "reason": "missing",
                             "message": "审查记录不存在或已删除，禁止参与分组"})
            continue
        if review.get("needs_recheck"):
            reason = review.get("recheck_reason")
            problems.append({
                "id": review["id"], "name": review["name"],
                "reason": reason or "unknown",
                "message": f"记录{_blocked_message(reason)}，禁止参与分组",
            })
            continue
        products.append(_product_of(review))
    if problems:
        raise LabelGroupError(
            f"所选产品中有 {len(problems)} 条记录禁止参与分组，请调整后重试",
            problems=problems)
    return build_grouping(products, limits)
