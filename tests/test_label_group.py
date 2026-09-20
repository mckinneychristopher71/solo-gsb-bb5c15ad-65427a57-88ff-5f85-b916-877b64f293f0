"""共用标签模板分组测试（不依赖 FastAPI，走 db + review + label_group 全链路）。

覆盖：
- 复用审查记录当前配方（两版取标签核对对象）的最终过敏原集合；
- 组模板为过敏原并集、附加声明 = 并集 − 自身过敏原及上限成组约束；
- 子集动态规划求模板数量最少的完整分组（对照暴力枚举）；
- 并列时按组内及组间产品 ID 序列取唯一结果；
- 缺失 / 需复核记录逐条拒绝；数量 / 重复 / 上限校验；全程只读。
"""
import itertools
import os
import tempfile
import unittest

_DB_DIR = tempfile.mkdtemp(prefix="allergen-label-group-")
os.environ["DB_PATH"] = os.path.join(_DB_DIR, "label_group.db")

from app import db  # noqa: E402
from app.label_group import (LabelGroupError,  # noqa: E402
                             build_grouping, build_label_group_preview,
                             list_label_group_products)
from app.recheck import build_recheck  # noqa: E402
from app.review import analyze  # noqa: E402

DICT_TEXT = """\
小麦粉：麸质谷物
乳清粉：乳
奶粉：乳
大豆卵磷脂：大豆
鸡蛋：蛋类
椰子粉：坚果
芝麻：芝麻
食盐：
"""
# A：麸质谷物、乳、大豆、蛋类；B：麸质谷物、坚果、蛋类、芝麻；C：麸质谷物、蛋类
TEXT_A = "配料表：小麦粉，奶粉（乳清粉），大豆卵磷脂，鸡蛋，食盐"
TEXT_B = "配料表：小麦粉，椰子粉，鸡蛋，芝麻，食盐"
TEXT_C = "配料表：小麦粉，鸡蛋，食盐"


def make_review(text_a, text_b=None, name="分组测试", with_fingerprint=True):
    result, fp = analyze(text_a=text_a, text_b=text_b, label_a="现版配方",
                         label_b="新版配方", declared=[])
    return db.create_review(
        name=name, declared=[], text_a=text_a, text_b=text_b,
        label_a="现版配方", label_b="新版配方", result=result,
        dict_fingerprint=fp if with_fingerprint else None)


def brute_force_grouping(products, max_extra):
    """暴力枚举全部集合划分，返回 (最少模板数, 规范化最小分组)。

    规范化分组：组内 ID 升序、组间按成员 ID 元组字典序升序。
    """
    ids = [p["id"] for p in products]
    sets = {p["id"]: set(p["allergens"]) for p in products}

    def partitions(els):
        if not els:
            yield []
            return
        first = els[0]
        for rest in partitions(els[1:]):
            for i in range(len(rest)):
                yield rest[:i] + [rest[i] + [first]] + rest[i + 1:]
            yield [[first]] + rest

    best = None
    for p in partitions(list(range(len(ids)))):
        id_groups = [tuple(sorted(ids[j] for j in g)) for g in p]
        ok = True
        for g in id_groups:
            union = set()
            for pid in g:
                union |= sets[pid]
            for pid in g:
                if len(union - sets[pid]) > max_extra[pid]:
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            continue
        key = tuple(sorted(id_groups))
        if best is None or (len(key), key) < (len(best), best):
            best = key
    return len(best), best


class GroupingCalcTest(unittest.TestCase):
    """纯计算：并集模板、附加声明上限、子集 DP 与并列取舍（不触碰数据库）。"""

    def test_zero_limit_forces_singletons(self):
        products = [
            {"id": 1, "name": "甲", "allergens": ["乳", "大豆"]},
            {"id": 2, "name": "乙", "allergens": ["乳"]},
            {"id": 3, "name": "丙", "allergens": ["蛋类"]},
        ]
        r = build_grouping(products, {1: 0, 2: 0, 3: 0})
        # 上限 0 只允许过敏原集合相同（或含于）的产品成组，三者两两不兼容
        self.assertEqual(r["group_count"], 3)
        self.assertEqual([[m["id"] for m in g["members"]] for g in r["groups"]],
                         [[1], [2], [3]])

    def test_subset_superset_share_template_at_zero_limit(self):
        # 乙（乳）是甲（乳、大豆）的子集：同组时乙有 1 个附加声明，
        # 甲 0 个；故上限 0 时不能成组，乙给 1 后可共用 1 个模板。
        products = [
            {"id": 1, "name": "甲", "allergens": ["乳", "大豆"]},
            {"id": 2, "name": "乙", "allergens": ["乳"]},
        ]
        r = build_grouping(products, {1: 0, 2: 0})
        self.assertEqual(r["group_count"], 2)
        r = build_grouping(products, {1: 0, 2: 1})
        self.assertEqual(r["group_count"], 1)
        [g] = r["groups"]
        self.assertEqual(g["template"], ["乳", "大豆"])
        by_id = {m["id"]: m for m in g["members"]}
        self.assertEqual(by_id[1]["extra_allergens"], [])
        self.assertEqual(by_id[2]["extra_allergens"], ["大豆"])
        self.assertTrue(all(m["within_limit"] for m in g["members"]))

    def test_template_is_allergen_union(self):
        products = [
            {"id": 1, "name": "甲", "allergens": ["乳"]},
            {"id": 2, "name": "乙", "allergens": ["蛋类"]},
        ]
        r = build_grouping(products, {1: 1, 2: 1})
        [g] = r["groups"]
        self.assertEqual(g["template"], ["乳", "蛋类"])
        self.assertEqual(r["allergen_union"], ["乳", "蛋类"])

    def test_minimum_template_count_matches_brute_force(self):
        products = [
            {"id": 3, "name": "P3", "allergens": ["乳", "大豆"]},
            {"id": 11, "name": "P11", "allergens": ["乳"]},
            {"id": 17, "name": "P17", "allergens": ["大豆", "蛋类"]},
            {"id": 23, "name": "P23", "allergens": []},
            {"id": 31, "name": "P31", "allergens": ["乳", "蛋类", "坚果"]},
        ]
        limits = {3: 1, 11: 1, 17: 0, 23: 2, 31: 0}
        r = build_grouping(products, limits)
        count, partition = brute_force_grouping(products, limits)
        self.assertEqual(r["group_count"], count)
        got = tuple(tuple(m["id"] for m in g["members"]) for g in r["groups"])
        self.assertEqual(got, partition)
        # 每组确实可成组，模板即并集，附加声明数不超限
        for g in r["groups"]:
            self.assertEqual(g["template_count"], len(g["template"]))
            self.assertTrue(g["members"])
            for m in g["members"]:
                self.assertEqual(
                    m["extra_count"],
                    len(set(g["template"]) - set(m["allergens"])))
                self.assertLessEqual(m["extra_count"], m["max_extra"])
                self.assertEqual(
                    m["extra_allergens"],
                    sorted(set(g["template"]) - set(m["allergens"])))

    def test_tie_break_intra_and_inter_group(self):
        # 1={乳,大豆}、2={乳,坚果} 互不相容（上限 0，差 1 个过敏原）；
        # 3={乳} 只能额外声明 1 种，因此可并入 1 或并入 2，但不能三组并一
        # （并集 {乳,大豆,坚果} 会让 1 多出「坚果」）。最少 2 组，两个最优
        # 划分并列：((1,),(2,3)) 与 ((1,3),(2,))。组内成员按 ID 升序后，
        # 组间按成员 ID 序列做字典序比较：首组 (1,) 是 (1,3) 的前缀更小，
        # 故唯一结果为 ((1,),(2,3))。
        products = [
            {"id": 1, "name": "P1", "allergens": ["乳", "大豆"]},
            {"id": 2, "name": "P2", "allergens": ["乳", "坚果"]},
            {"id": 3, "name": "P3", "allergens": ["乳"]},
        ]
        limits = {1: 0, 2: 0, 3: 1}
        r = build_grouping(products, limits)
        self.assertEqual(r["group_count"], 2)
        got = tuple(tuple(m["id"] for m in g["members"]) for g in r["groups"])
        _, expect = brute_force_grouping(products, limits)
        self.assertEqual(got, expect)
        self.assertEqual(got, ((1,), (2, 3)))

    def test_tie_break_groups_sorted_by_smallest_member(self):
        # 各产品过敏原互不相容（上限 0），只能全部单产品成组；
        # 组间顺序按各组产品 ID 序列（首元素即最小组 ID）升序。
        products = [
            {"id": 9, "name": "P9", "allergens": ["乳"]},
            {"id": 2, "name": "P2", "allergens": ["大豆"]},
            {"id": 5, "name": "P5", "allergens": ["蛋类"]},
        ]
        r = build_grouping(products, {9: 0, 2: 0, 5: 0})
        self.assertEqual(
            [[m["id"] for m in g["members"]] for g in r["groups"]],
            [[2], [5], [9]])

    def test_all_same_allergens_one_group(self):
        products = [{"id": i, "name": f"P{i}", "allergens": ["乳"]}
                    for i in (8, 1, 4)]
        r = build_grouping(products, {i: 0 for i in (8, 1, 4)})
        self.assertEqual(r["group_count"], 1)
        self.assertEqual(
            [m["id"] for m in r["groups"][0]["members"]], [1, 4, 8])
        for m in r["groups"][0]["members"]:
            self.assertEqual(m["extra_count"], 0)


class PreviewFlowTest(unittest.TestCase):
    """全链路：候选清单、资格校验、分组结果与只读性。"""

    def setUp(self):
        # 每个用例使用独立数据库，互不污染
        db.DB_PATH = os.path.join(
            _DB_DIR, f"label_group_{self.id().rsplit('.', 1)[-1]}.db")
        db.init_db()
        db.bulk_import(DICT_TEXT)

    def _fp(self, rid):
        with db._connect() as conn:
            return conn.execute(
                "SELECT dict_fingerprint FROM reviews WHERE id=?",
                (rid,)).fetchone()[0]

    def test_current_allergens_from_review(self):
        rid_a = make_review(TEXT_A, name="单版")
        rid_ab = make_review(TEXT_A, TEXT_B, name="两版")
        products = {p["id"]: p for p in list_label_group_products()}
        self.assertEqual(products[rid_a]["allergens"],
                         sorted(["麸质谷物", "乳", "大豆", "蛋类"]))
        self.assertEqual(products[rid_a]["source_version"], "现版配方")
        # 两版记录取标签核对对象（新版配方）的最终过敏原集合
        self.assertEqual(products[rid_ab]["allergens"],
                         sorted(["麸质谷物", "坚果", "蛋类", "芝麻"]))
        self.assertEqual(products[rid_ab]["source_version"], "新版配方")
        self.assertTrue(products[rid_a]["eligible"])
        self.assertIsNone(products[rid_a]["blocked_reason"])
        # 清单按产品 ID 升序
        ids = [p["id"] for p in list_label_group_products()]
        self.assertEqual(ids, sorted(ids))

    def test_preview_end_to_end_and_readonly(self):
        rid1 = make_review(TEXT_A, name="产品甲")
        rid2 = make_review(TEXT_B, name="产品乙")
        rid3 = make_review(TEXT_C, name="产品丙")
        before = {rid: db.get_review(rid) for rid in (rid1, rid2, rid3)}
        items = [
            {"review_id": rid3, "max_extra": 0},
            {"review_id": rid1, "max_extra": 2},
            {"review_id": rid2, "max_extra": 2},
        ]
        r = build_label_group_preview(items)
        products = [
            {"id": rid1, "name": "产品甲",
             "allergens": ["麸质谷物", "乳", "大豆", "蛋类"]},
            {"id": rid2, "name": "产品乙",
             "allergens": ["麸质谷物", "坚果", "蛋类", "芝麻"]},
            {"id": rid3, "name": "产品丙", "allergens": ["麸质谷物", "蛋类"]},
        ]
        limits = {rid1: 2, rid2: 2, rid3: 0}
        count, partition = brute_force_grouping(products, limits)
        self.assertEqual(r["group_count"], count)
        got = tuple(tuple(m["id"] for m in g["members"]) for g in r["groups"])
        self.assertEqual(got, partition)
        # products 按 ID 升序回显，含每产品上限
        self.assertEqual([p["id"] for p in r["products"]],
                         sorted([rid1, rid2, rid3]))
        self.assertEqual({p["id"]: p["max_extra"] for p in r["products"]},
                         limits)
        # 只读：记录内容、指纹、时效状态与记录数均不变
        for rid, old in before.items():
            self.assertEqual(db.get_review(rid), old)
            self.assertEqual(self._fp(rid), old["dict_fingerprint"])
        self.assertEqual(len(db.list_reviews()), 3)

    def test_missing_record_rejected(self):
        rid = make_review(TEXT_A)
        with self.assertRaises(LabelGroupError) as ctx:
            build_label_group_preview(
                [{"review_id": rid, "max_extra": 0},
                 {"review_id": 999999, "max_extra": 0}])
        [p] = ctx.exception.problems
        self.assertEqual(p["reason"], "missing")
        self.assertEqual(p["id"], 999999)
        self.assertIn("不存在", p["message"])

    def test_stale_record_rejected_until_rechecked(self):
        rid1 = make_review(TEXT_A, name="产品甲")
        rid2 = make_review(TEXT_C, name="产品丙")
        db.upsert_entry("乳清粉", ["乳", "花生"])  # 相关原料映射变更 → 需复核
        self.assertTrue(db.get_review(rid1)["needs_recheck"])
        with self.assertRaises(LabelGroupError) as ctx:
            build_label_group_preview(
                [{"review_id": rid1, "max_extra": 0},
                 {"review_id": rid2, "max_extra": 0}])
        [p] = ctx.exception.problems
        self.assertEqual(p["reason"], "stale")
        self.assertEqual(p["id"], rid1)
        self.assertIn("需复核", p["message"])
        # 候选清单同步标记禁止参与并给出原因
        products = {p["id"]: p for p in list_label_group_products()}
        self.assertFalse(products[rid1]["eligible"])
        self.assertIn("需复核", products[rid1]["blocked_reason"])
        self.assertTrue(products[rid2]["eligible"])
        # 完成字典时效复核后恢复可参与
        outcome = build_recheck(db.get_review(rid1))
        db.update_review_result(rid1, outcome["new_result"],
                                outcome["fingerprint"])
        self.assertFalse(db.get_review(rid1)["needs_recheck"])
        r = build_label_group_preview(
            [{"review_id": rid1, "max_extra": 0},
             {"review_id": rid2, "max_extra": 0}])
        self.assertEqual(r["group_count"], 2)

    def test_legacy_record_rejected(self):
        rid1 = make_review(TEXT_A, with_fingerprint=False)
        rid2 = make_review(TEXT_C)
        with self.assertRaises(LabelGroupError) as ctx:
            build_label_group_preview(
                [{"review_id": rid1, "max_extra": 0},
                 {"review_id": rid2, "max_extra": 0}])
        [p] = ctx.exception.problems
        self.assertEqual(p["reason"], "legacy")
        self.assertIn("指纹", p["message"])

    def test_selection_count_and_duplicates(self):
        rid = make_review(TEXT_A)
        with self.assertRaises(LabelGroupError):
            build_label_group_preview([{"review_id": rid, "max_extra": 0}])
        with self.assertRaises(LabelGroupError):
            build_label_group_preview(
                [{"review_id": i, "max_extra": 0} for i in range(13)])
        rid2 = make_review(TEXT_C)
        with self.assertRaises(LabelGroupError) as ctx:
            build_label_group_preview(
                [{"review_id": rid, "max_extra": 0},
                 {"review_id": rid, "max_extra": 0},
                 {"review_id": rid2, "max_extra": 0}])
        self.assertIn("重复", str(ctx.exception))

    def test_max_extra_validation(self):
        rid1 = make_review(TEXT_A)
        rid2 = make_review(TEXT_C)
        with self.assertRaises(LabelGroupError):
            build_label_group_preview(
                [{"review_id": rid1, "max_extra": -1},
                 {"review_id": rid2, "max_extra": 0}])
        with self.assertRaises(LabelGroupError):
            build_label_group_preview(
                [{"review_id": rid1, "max_extra": 1.5},
                 {"review_id": rid2, "max_extra": 0}])
        # 合法上限可以成组（丙是甲的过敏原子集，甲给 0、丙给 2 可共用）
        r = build_label_group_preview(
            [{"review_id": rid1, "max_extra": 0},
             {"review_id": rid2, "max_extra": 2}])
        self.assertEqual(r["group_count"], 1)
        [g] = r["groups"]
        by_id = {m["id"]: m for m in g["members"]}
        self.assertEqual(by_id[rid2]["extra_count"], 2)
        self.assertEqual(by_id[rid2]["extra_allergens"], ["乳", "大豆"])


if __name__ == "__main__":
    unittest.main()
