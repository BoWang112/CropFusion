# -*- coding: utf-8 -*-
"""
generate_fewshot_pool_v2.py
========================

用途
----
从本文已经构建的病虫害知识库与 QA 候选生成逻辑中，自动筛选并生成
“六类任务 × 每类 30 条 = 180 条”的动态少样本示例池 fewshot_pool.json。

设计目标
--------
1. 复用 qa_benchmark_v4.py 的真实知识证据与六类任务生成逻辑；
2. 与最终问答评测集 qa_test.json 在“问题”和“gold evidence”层面严格隔离；
3. 每类任务均衡抽取 30 条；
4. 尽量提高作物、病虫害和证据来源的多样性；
5. 保存 evidence_ids、source_ids、derived_relations、gold_graph_evidence，便于后续审计与复现；
6. 可选使用 BGE 对 few-shot 与测试集做高语义相似度过滤；
7. 输出自动审计报告，最终仍建议人工审核后将 review_status 改为 approved。

默认输出
--------
data/fewshot_pool.json
data/fewshot_pool_audit.json

依赖
----
本脚本默认与以下文件放在同一项目中：
    config.py
    qa_benchmark_v4.py
    evidence_prompting_v3.py       # 仅 --semantic_audit 时需要
    modules/kg.py

正式论文实验建议
----------------
推荐先冻结 qa_test.json，再运行本脚本生成 few-shot pool。
这样可以明确保证：
    Question_fewshot ∩ Question_test = ∅
    Evidence_fewshot ∩ Evidence_test = ∅

如果启用 --semantic_audit，还会过滤同任务下与测试问题 BGE cosine
similarity >= 阈值（默认 0.94）的候选问题。

示例
----
基础生成：
    python generate_fewshot_pool_v2.py

指定每类数量：
    python generate_fewshot_pool_v2.py --per_task 30

开启语义泄漏过滤：
    python generate_fewshot_pool_v2.py --semantic_audit --semantic_threshold 0.94

正式模式，要求测试集必须存在：
    python generate_fewshot_pool_v2.py --require_eval --semantic_audit
"""

import argparse
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

# ------------------------------------------------------------
# 项目内模块
# ------------------------------------------------------------

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

# 项目当前把 QA benchmark v4 的生成逻辑实现在 generate_qa_test.py 中；
# 若存在独立 qa_benchmark_v4.py 则优先使用，否则回退到 generate_qa_test.py。
try:
    import qa_benchmark_v4 as qav4
except ImportError:
    try:
        import generate_qa_test as qav4
    except ImportError as e:
        raise ImportError(
            "无法导入 qa_benchmark_v4.py 或 generate_qa_test.py。"
            "请确认存在其一。"
        ) from e

try:
    from config import EXTRACT_JSON, EVAL_FILE
except ImportError as e:
    raise ImportError(
        "无法导入 config.py 中的 EXTRACT_JSON / EVAL_FILE。"
    ) from e


# ============================================================
# 0. 全局默认值
# ============================================================

RANDOM_SEED = 2026
DEFAULT_PER_TASK = 30
EXPECTED_TOTAL = 180

DEFAULT_OUTPUT = "data/fewshot_pool.json"
DEFAULT_AUDIT_OUTPUT = "data/fewshot_pool_audit.json"

# 每个 (crop, pest) 在同一任务中优先最多选多少条。
# 如果因此不足 30 条，会自动进入第二阶段放宽限制。
DEFAULT_MAX_PER_PEST_PER_TASK = 2

# 候选生成倍率。qa_benchmark_v4 原本默认 2；
# few-shot 为提高多样性，可以适当增大。
DEFAULT_CANDIDATE_MULTIPLIER = 6

VALID_TASKS = list(qav4.TASKS)


# ============================================================
# 1. 基础工具
# ============================================================

def normalize_question(q: str) -> str:
    """
    与 qa_benchmark_v4 的问题规范化逻辑兼容。
    """
    if hasattr(qav4, "norm_question"):
        return qav4.norm_question(q)
    q = str(q or "").strip().lower()
    return re.sub(r"[\s，。、；：？！“”‘’\"',.!?;:（）()【】\[\]]+", "", q)


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: str) -> None:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    with open(path_obj, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def evidence_id_set(item: Dict[str, Any]) -> Set[str]:
    """
    兼容 qa_benchmark_v4 的 gold_evidence_ids。
    """
    ids = item.get("gold_evidence_ids") or item.get("evidence_ids") or []
    return {str(x) for x in ids if str(x).strip()}


def graph_evidence_key(edge: Dict[str, Any]) -> Tuple[str, str, str]:
    """用 (head, rel, tail) 标识真实 KG 关系。"""
    return (
        str(edge.get("head") or "").strip(),
        str(edge.get("rel") or edge.get("relation") or "").strip(),
        str(edge.get("tail") or "").strip(),
    )


def graph_evidence_set(item: Dict[str, Any]) -> Set[Tuple[str, str, str]]:
    edges = item.get("gold_graph_evidence") or []
    out = set()
    for edge in edges:
        if not isinstance(edge, dict):
            continue
        key = graph_evidence_key(edge)
        if all(key):
            out.add(key)
    return out


def format_graph_gold_for_fewshot(edges: Sequence[Dict[str, Any]]) -> str:
    """将真实 KG gold evidence 转成 few-shot 中可读的证据文本。"""
    lines = []
    for i, edge in enumerate(edges, start=1):
        head = str(edge.get("head") or "").strip()
        rel = str(edge.get("rel") or edge.get("relation") or "").strip()
        tail = str(edge.get("tail") or "").strip()
        if head and rel and tail:
            lines.append(f"[G{i}] {head} -{rel}-> {tail}")
    return "\n".join(lines)


def task_name(item: Dict[str, Any]) -> str:
    t = str(item.get("task") or "").strip()
    # 兼容旧名称
    aliases = {
        "相似病虫害区分": "相似病虫害鉴别",
        "相似案例区分": "相似病虫害鉴别",
        "多条件推理": "多条件综合推理",
        "发生条件判断": "条件判断",
        "病虫害类别": "病虫害识别",
    }
    return aliases.get(t, t)


# ============================================================
# 2. 读取并解析最终评测集
# ============================================================

def load_eval_constraints(
    eval_file: str,
    require_eval: bool = False,
) -> Dict[str, Any]:
    """
    从最终评测集抽取必须避开的：
    - normalized questions
    - gold text evidence IDs
    - gold graph evidence edges
    - task-wise questions
    """
    path = Path(eval_file)

    if not path.exists():
        if require_eval:
            raise FileNotFoundError(
                f"未找到最终评测集：{eval_file}\n"
                "正式论文模式下建议先冻结 qa_test.json，再生成 few-shot pool。"
            )

        print(
            f"[WARN] 未找到评测集 {eval_file}。\n"
            "       当前无法保证 few-shot 与最终测试集完全独立。\n"
            "       正式实验建议先冻结 qa_test.json，再重新运行本脚本。"
        )
        return {
            "exists": False,
            "items": [],
            "question_norms": set(),
            "evidence_ids": set(),
            "graph_edges": set(),
            "questions_by_task": defaultdict(list),
        }

    data = load_json(str(path))
    if not isinstance(data, list):
        raise ValueError("评测集 JSON 顶层必须是 list。")

    q_norms: Set[str] = set()
    eids: Set[str] = set()
    graph_edges: Set[Tuple[str, str, str]] = set()
    by_task: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for item in data:
        q = str(item.get("question") or "").strip()
        if q:
            q_norms.add(normalize_question(q))
        eids.update(evidence_id_set(item))
        graph_edges.update(graph_evidence_set(item))
        by_task[task_name(item)].append(item)

    print(
        f"[Eval] 已加载测试集：{len(data)} 条；"
        f"唯一问题={len(q_norms)}；"
        f"gold text evidence IDs={len(eids)}；"
        f"gold KG edges={len(graph_edges)}"
    )

    return {
        "exists": True,
        "items": data,
        "question_norms": q_norms,
        "evidence_ids": eids,
        "graph_edges": graph_edges,
        "questions_by_task": by_task,
    }

# ============================================================
# 3. 从 qa_benchmark_v4 生成候选
# ============================================================

def build_clean_candidates(candidate_multiplier: int) -> Dict[str, List[Dict[str, Any]]]:
    """
    直接复用 qa_benchmark_v4：
        EXTRACT_JSON
            -> build_pest_catalog
            -> generate_candidates
            -> quality_check
            -> dedupe

    通过临时修改 qav4.MAX_CANDIDATES_PER_ITEM_PER_TASK
    来增加候选数量。
    """
    random.seed(RANDOM_SEED)

    with open(EXTRACT_JSON, "r", encoding="utf-8") as f:
        recs = json.load(f)

    recs = [
        r for r in recs
        if (r.get("pest") or "").strip()
        and (r.get("pest") or "").strip().lower() != "nan"
        and (r.get("crop") or "").strip()
    ]

    catalog = qav4.build_pest_catalog(recs)

    print(f"[Data] 原始记录：{len(recs)}")
    print(f"[Data] (crop, pest) 素材条目：{len(catalog)}")

    old_multiplier = getattr(qav4, "MAX_CANDIDATES_PER_ITEM_PER_TASK", 2)
    qav4.MAX_CANDIDATES_PER_ITEM_PER_TASK = candidate_multiplier

    try:
        # qa_benchmark_v4.generate_candidates 需要 KG
        kg = qav4.CropPestKG(qav4.NODES_CSV, qav4.RELS_CSV)
        clean = qav4.generate_candidates(kg, catalog)
    finally:
        qav4.MAX_CANDIDATES_PER_ITEM_PER_TASK = old_multiplier

    for t in VALID_TASKS:
        print(f"[Candidate] {t}: {len(clean.get(t, []))}")

    return clean


# ============================================================
# 4. 严格避开评测集的问题与 evidence
# ============================================================

def filter_eval_overlap(
    candidates_by_task: Dict[str, List[Dict[str, Any]]],
    eval_constraints: Dict[str, Any],
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Any]]:
    """
    严格过滤：
    1. question 完全规范化重复；
    2. gold text evidence ID 与 eval 相交；
    3. 真实 gold KG edge 与 eval 相交。
    """
    eval_q = eval_constraints["question_norms"]
    eval_e = eval_constraints["evidence_ids"]
    eval_g = eval_constraints.get("graph_edges", set())

    filtered: Dict[str, List[Dict[str, Any]]] = {}
    report = {}

    for task in VALID_TASKS:
        kept = []
        counters = Counter()

        for qa in candidates_by_task.get(task, []):
            nq = normalize_question(qa.get("question", ""))
            eids = evidence_id_set(qa)
            gedges = graph_evidence_set(qa)

            if nq and nq in eval_q:
                counters["exact_question_overlap"] += 1
                continue

            if eids and (eids & eval_e):
                counters["text_evidence_overlap"] += 1
                continue

            if gedges and (gedges & eval_g):
                counters["graph_evidence_overlap"] += 1
                continue

            kept.append(qa)

        filtered[task] = kept
        report[task] = {
            "before": len(candidates_by_task.get(task, [])),
            "after_filter": len(kept),
            "removed_exact_question_overlap": counters["exact_question_overlap"],
            "removed_text_evidence_overlap": counters["text_evidence_overlap"],
            "removed_graph_evidence_overlap": counters["graph_evidence_overlap"],
        }

        print(
            f"[Leakage filter] {task}: "
            f"{report[task]['before']} -> {report[task]['after_filter']}; "
            f"Q={report[task]['removed_exact_question_overlap']}, "
            f"Text-E={report[task]['removed_text_evidence_overlap']}, "
            f"Graph-E={report[task]['removed_graph_evidence_overlap']}"
        )

    return filtered, report

# ============================================================
# 5. 可选：BGE 高语义相似度过滤
# ============================================================

def semantic_filter_against_eval(
    candidates_by_task: Dict[str, List[Dict[str, Any]]],
    eval_constraints: Dict[str, Any],
    threshold: float = 0.94,
    bge_model: Optional[str] = None,
    device: Optional[str] = None,
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Any]]:
    """
    对同一 task：
        candidate question vs eval questions
    计算最大 cosine similarity。

    若 max similarity >= threshold，则删除该候选。
    这样可进一步减少“同一道题换一种说法”的泄漏风险。

    仅当测试集存在时执行。
    """
    if not eval_constraints["exists"]:
        print("[WARN] 无测试集，跳过 BGE 语义泄漏过滤。")
        return candidates_by_task, {}

    try:
        from evidence_prompting_v3 import (
            BGEEmbedder,
            DEFAULT_BGE_MODEL,
        )
    except ImportError as e:
        raise ImportError(
            "启用 --semantic_audit 需要 evidence_prompting_v3.py "
            "与本脚本位于同一项目目录。"
        ) from e

    model_name = bge_model or DEFAULT_BGE_MODEL
    embedder = BGEEmbedder(model_name=model_name, device=device)

    filtered: Dict[str, List[Dict[str, Any]]] = {}
    report = {}

    for task in VALID_TASKS:
        cands = candidates_by_task.get(task, [])
        eval_items = eval_constraints["questions_by_task"].get(task, [])

        if not cands or not eval_items:
            filtered[task] = cands
            report[task] = {
                "before": len(cands),
                "after_semantic_filter": len(cands),
                "removed_high_similarity": 0,
                "threshold": threshold,
            }
            continue

        cand_questions = [x["question"] for x in cands]
        eval_questions = [x["question"] for x in eval_items]

        cand_vec = embedder.encode(cand_questions)
        eval_vec = embedder.encode(eval_questions)

        # vectors 已 L2 normalize
        sim = cand_vec @ eval_vec.T
        max_scores, max_idx = sim.max(dim=1)

        kept = []
        removed_examples = []

        for i, qa in enumerate(cands):
            score = float(max_scores[i].item())
            if score >= threshold:
                j = int(max_idx[i].item())
                removed_examples.append({
                    "candidate_question": qa["question"],
                    "eval_question": eval_questions[j],
                    "similarity": round(score, 6),
                })
                continue
            qa = dict(qa)
            qa.setdefault("metadata", {})
            qa["metadata"]["max_eval_semantic_similarity"] = round(score, 6)
            kept.append(qa)

        filtered[task] = kept
        report[task] = {
            "before": len(cands),
            "after_semantic_filter": len(kept),
            "removed_high_similarity": len(removed_examples),
            "threshold": threshold,
            "removed_examples_top20": sorted(
                removed_examples,
                key=lambda x: x["similarity"],
                reverse=True,
            )[:20],
        }

        print(
            f"[Semantic filter] {task}: {len(cands)} -> {len(kept)}; "
            f"removed={len(removed_examples)} @ threshold={threshold}"
        )

    report["_model"] = model_name
    return filtered, report


# ============================================================
# 6. few-shot 格式转换
# ============================================================

def qa_to_fewshot(qa: Dict[str, Any], idx: int) -> Dict[str, Any]:
    """
    将 QA benchmark v4 候选转成 evidence_prompting_v3.py
    可直接读取的 few-shot 格式。
    """
    task = task_name(qa)
    evidence_texts = list(qa.get("gold_evidence_texts") or [])
    evidence_ids = list(qa.get("gold_evidence_ids") or [])
    source_ids = list(qa.get("source_ids") or [])
    graph_gold = list(qa.get("gold_graph_evidence") or [])
    derived_relations = list(qa.get("derived_relations") or [])

    if len(evidence_texts) == 1:
        text_block = f"[E1] {evidence_texts[0]}"
    elif evidence_texts:
        text_block = "\n".join(
            f"[E{i}] {txt}"
            for i, txt in enumerate(evidence_texts, start=1)
        )
    else:
        text_block = ""

    graph_block = format_graph_gold_for_fewshot(graph_gold)

    blocks = []
    if text_block:
        blocks.append("文本证据：\n" + text_block)
    if graph_block:
        blocks.append("图谱证据：\n" + graph_block)

    evidence = "\n".join(blocks).strip()
    if not evidence:
        evidence = "（缺少支持证据）"

    fs_id = f"fs_{idx:04d}"
    metadata = qa.get("metadata") or {}

    item = {
        "id": fs_id,
        "task": task,
        "crop": qa.get("crop", ""),
        "pest": qa.get("gold_pest", ""),
        "question": qa.get("question", ""),
        "evidence": evidence,
        "evidence_ids": evidence_ids,
        "source_ids": source_ids,
        "derived_relations": derived_relations,
        "gold_graph_evidence": graph_gold,
        "answer": qa.get("reference_answer") or qa.get("answer", ""),
        "review_status": "pending",
        "origin_qa_id": qa.get("id", ""),
        "benchmark_source_version": "v4",
    }

    for key in [
        "contrast_pest",
        "contrast_pest_full",
        "similar_relation",
        "evidence_coherence",
        "condition_effect",
        "control_categories",
        "max_eval_semantic_similarity",
    ]:
        if key in metadata:
            item[key] = metadata[key]

    return item

# ============================================================
# 7. 多样性优先均衡抽样
# ============================================================

def diverse_sample_task(
    candidates: Sequence[Dict[str, Any]],
    target: int,
    max_per_pest: int,
    global_used_questions: Set[str],
) -> List[Dict[str, Any]]:
    """
    目标：
    - 优先覆盖更多不同 (crop, pest)
    - 尽量避免同一病虫害在一个 task 中占太多
    - 问题全局不重复

    第一阶段：
        每个 (crop, pest) 最多 max_per_pest 条；
    第二阶段：
        若仍不足 target，放宽该限制继续补齐。
    """
    items = list(candidates)
    random.shuffle(items)

    # 分组
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for qa in items:
        key = (
            str(qa.get("crop") or ""),
            str(qa.get("gold_pest") or ""),
        )
        groups[key].append(qa)

    group_keys = list(groups.keys())
    random.shuffle(group_keys)

    selected: List[Dict[str, Any]] = []
    local_count = Counter()
    used_origin_ids = set()

    # round-robin，优先跨 pest 多样性
    progressed = True
    while len(selected) < target and progressed:
        progressed = False

        for key in group_keys:
            if len(selected) >= target:
                break
            if local_count[key] >= max_per_pest:
                continue

            bucket = groups[key]
            while bucket:
                qa = bucket.pop()
                nq = normalize_question(qa.get("question", ""))

                if nq in global_used_questions:
                    continue

                origin_id = qa.get("id")
                if origin_id and origin_id in used_origin_ids:
                    continue

                selected.append(qa)
                global_used_questions.add(nq)
                if origin_id:
                    used_origin_ids.add(origin_id)
                local_count[key] += 1
                progressed = True
                break

    # 第二阶段：如果因 max_per_pest 不足，放宽限制
    if len(selected) < target:
        leftovers = []
        for bucket in groups.values():
            leftovers.extend(bucket)
        random.shuffle(leftovers)

        for qa in leftovers:
            if len(selected) >= target:
                break

            nq = normalize_question(qa.get("question", ""))
            if nq in global_used_questions:
                continue

            origin_id = qa.get("id")
            if origin_id and origin_id in used_origin_ids:
                continue

            selected.append(qa)
            global_used_questions.add(nq)
            if origin_id:
                used_origin_ids.add(origin_id)

    return selected


def balanced_diverse_sample(
    candidates_by_task: Dict[str, List[Dict[str, Any]]],
    per_task: int,
    max_per_pest: int,
) -> List[Dict[str, Any]]:
    """
    六类严格均衡抽样。
    """
    selected_raw: List[Dict[str, Any]] = []
    global_used_questions: Set[str] = set()

    shortages = {}

    for task in VALID_TASKS:
        candidates = candidates_by_task.get(task, [])

        picked = diverse_sample_task(
            candidates=candidates,
            target=per_task,
            max_per_pest=max_per_pest,
            global_used_questions=global_used_questions,
        )

        if len(picked) < per_task:
            shortages[task] = {
                "available_after_filter": len(candidates),
                "selected": len(picked),
                "needed": per_task,
                "shortage": per_task - len(picked),
            }

        selected_raw.extend(picked)
        print(f"[Select] {task}: {len(picked)}/{per_task}")

    if shortages:
        lines = [
            "无法严格生成六类均衡 few-shot pool："
        ]
        for task, info in shortages.items():
            lines.append(
                f"  - {task}: 过滤后候选 {info['available_after_filter']}，"
                f"实际选中 {info['selected']}，目标 {info['needed']}，"
                f"缺 {info['shortage']}"
            )
        lines.append(
            "\n建议：\n"
            "1. 增加 --candidate_multiplier；\n"
            "2. 检查 qa_benchmark_v4 中该任务可用真实证据是否不足；\n"
            "3. 对“相似病虫害鉴别”重点检查图谱相似关系数量；\n"
            "4. 不建议用其他任务补齐该类。"
        )
        raise RuntimeError("\n".join(lines))

    # 固定任务内数量后整体打乱
    random.shuffle(selected_raw)

    # 转为 few-shot 格式并编号
    fewshot = [
        qa_to_fewshot(qa, idx=i)
        for i, qa in enumerate(selected_raw, start=1)
    ]

    return fewshot


# ============================================================
# 8. 最终审计
# ============================================================

def audit_final_pool(
    pool: Sequence[Dict[str, Any]],
    eval_constraints: Dict[str, Any],
    per_task: int,
) -> Dict[str, Any]:
    task_cnt = Counter(task_name(x) for x in pool)
    crop_cnt = Counter(x.get("crop", "") for x in pool)
    pest_cnt = Counter(
        (x.get("crop", ""), x.get("pest", ""))
        for x in pool
    )

    q_norms = [normalize_question(x.get("question", "")) for x in pool]
    duplicate_questions = len(q_norms) - len(set(q_norms))

    fs_evidence = set()
    fs_graph_edges = set()
    evidence_reuse_counter = Counter()
    graph_reuse_counter = Counter()

    for x in pool:
        for eid in x.get("evidence_ids", []):
            eid = str(eid)
            fs_evidence.add(eid)
            evidence_reuse_counter[eid] += 1

        for edge in x.get("gold_graph_evidence", []) or []:
            if isinstance(edge, dict):
                key = graph_evidence_key(edge)
                if all(key):
                    fs_graph_edges.add(key)
                    graph_reuse_counter[key] += 1

    eval_q = eval_constraints["question_norms"]
    eval_e = eval_constraints["evidence_ids"]
    eval_g = eval_constraints.get("graph_edges", set())

    q_overlap = [
        x["id"]
        for x in pool
        if normalize_question(x.get("question", "")) in eval_q
    ]
    e_overlap = sorted(fs_evidence & eval_e)
    g_overlap = sorted(fs_graph_edges & eval_g)

    repeated_evidence = {
        eid: n for eid, n in evidence_reuse_counter.items() if n > 1
    }
    repeated_graph = {
        " | ".join(key): n for key, n in graph_reuse_counter.items() if n > 1
    }

    task_balance_ok = all(
        task_cnt.get(t, 0) == per_task
        for t in VALID_TASKS
    )

    required_fields = {
        "id", "task", "question", "evidence", "answer",
        "evidence_ids", "source_ids",
        "derived_relations", "gold_graph_evidence",
        "review_status",
    }
    missing_fields = []
    special_task_issues = []

    for x in pool:
        missing = sorted(required_fields - set(x.keys()))
        if missing:
            missing_fields.append({
                "id": x.get("id"),
                "missing": missing,
            })

        task = task_name(x)

        if task == "相似病虫害鉴别":
            # 同作物兜底样本（similar_relation == "同作物兜底"）允许无真实 KG 边；
            # 只有既无 KG 边又未标记兜底来源的才视为异常。
            sim_rel = x.get("similar_relation", "")
            if not x.get("gold_graph_evidence") and sim_rel != "同作物兜底":
                special_task_issues.append({
                    "id": x.get("id"),
                    "issue": "differential_without_real_kg_edge",
                })
            if len(x.get("evidence_ids", [])) < 2:
                special_task_issues.append({
                    "id": x.get("id"),
                    "issue": "differential_with_less_than_2_text_evidence",
                })

        if task == "多条件综合推理":
            if len(x.get("evidence_ids", [])) < 2:
                special_task_issues.append({
                    "id": x.get("id"),
                    "issue": "multi_with_less_than_2_evidence",
                })
            if x.get("evidence_coherence") not in {"same_record", "same_source"}:
                special_task_issues.append({
                    "id": x.get("id"),
                    "issue": f"multi_bad_coherence:{x.get('evidence_coherence')}",
                })

    report = {
        "generator_version": "v2",
        "source_benchmark_version": "v4",
        "total": len(pool),
        "expected_total": per_task * len(VALID_TASKS),
        "task_counts": dict(task_cnt),
        "task_balance_ok": task_balance_ok,
        "num_crops": len([x for x in crop_cnt if x]),
        "top_crops": crop_cnt.most_common(20),
        "num_crop_pest_pairs": len([x for x in pest_cnt if any(x)]),
        "duplicate_question_count": duplicate_questions,

        "eval_exact_question_overlap_count": len(q_overlap),
        "eval_exact_question_overlap_ids": q_overlap[:50],

        "eval_text_evidence_overlap_count": len(e_overlap),
        "eval_text_evidence_overlap_ids": e_overlap[:100],

        "eval_graph_evidence_overlap_count": len(g_overlap),
        "eval_graph_evidence_overlap_edges": [
            {"head": x[0], "rel": x[1], "tail": x[2]}
            for x in g_overlap[:100]
        ],

        "unique_text_evidence_ids": len(fs_evidence),
        "unique_graph_edges": len(fs_graph_edges),

        "reused_text_evidence_id_count": len(repeated_evidence),
        "reused_text_evidence_top50": sorted(
            repeated_evidence.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:50],

        "reused_graph_edge_count": len(repeated_graph),
        "reused_graph_edge_top50": sorted(
            repeated_graph.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:50],

        "missing_required_fields": missing_fields[:100],
        "special_task_issues": special_task_issues[:100],
        "all_review_pending": all(
            x.get("review_status") == "pending"
            for x in pool
        ),
    }

    if len(pool) != per_task * len(VALID_TASKS):
        raise RuntimeError(
            f"最终 few-shot 数量为 {len(pool)}，"
            f"目标应为 {per_task * len(VALID_TASKS)}。"
        )

    if not task_balance_ok:
        raise RuntimeError(
            f"六类任务数量不均衡：{dict(task_cnt)}"
        )

    if duplicate_questions > 0:
        raise RuntimeError(
            f"few-shot pool 内仍存在 {duplicate_questions} 条重复问题。"
        )

    if q_overlap:
        raise RuntimeError(
            f"few-shot 与 eval 存在 {len(q_overlap)} 条完全重复问题。"
        )

    if e_overlap:
        raise RuntimeError(
            f"few-shot 与 eval 存在 {len(e_overlap)} 个 gold text evidence ID 重叠。"
        )

    if g_overlap:
        raise RuntimeError(
            f"few-shot 与 eval 存在 {len(g_overlap)} 条 gold KG edge 重叠。"
        )

    if missing_fields:
        raise RuntimeError(
            f"few-shot pool 存在必需字段缺失：{missing_fields[:5]}"
        )

    if special_task_issues:
        raise RuntimeError(
            "few-shot pool 存在不符合 QA benchmark v4 约束的任务样本："
            f"{special_task_issues[:5]}"
        )

    return report

# ============================================================
# 9. 人工审核辅助输出
# ============================================================

def write_review_tsv(pool: Sequence[Dict[str, Any]], output_json: str) -> str:
    """输出人工审核 TSV。"""
    out_path = str(Path(output_json).with_suffix(".review.tsv"))

    headers = [
        "id",
        "task",
        "crop",
        "pest",
        "contrast_pest",
        "question",
        "evidence",
        "answer",
        "evidence_ids",
        "source_ids",
        "derived_relations",
        "gold_graph_evidence",
        "evidence_coherence",
        "condition_effect",
        "control_categories",
        "review_status",
        "review_comment",
    ]

    def clean_cell(v: Any) -> str:
        if isinstance(v, (list, dict)):
            v = json.dumps(v, ensure_ascii=False)
        return str(v or "").replace("\t", " ").replace("\r", " ").replace("\n", " ")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\t".join(headers) + "\n")
        for x in pool:
            row = dict(x)
            row["review_comment"] = ""
            f.write(
                "\t".join(clean_cell(row.get(h, "")) for h in headers)
                + "\n"
            )

    return out_path

# ============================================================
# 10. CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="生成 180 条动态少样本示例池（六类任务均衡）"
    )

    parser.add_argument(
        "--eval_file",
        type=str,
        default=EVAL_FILE,
        help="最终评测集 qa_test.json 路径，用于严格去泄漏。",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT,
        help="few-shot JSON 输出路径。",
    )
    parser.add_argument(
        "--audit_output",
        type=str,
        default=DEFAULT_AUDIT_OUTPUT,
        help="自动审计报告输出路径。",
    )
    parser.add_argument(
        "--per_task",
        type=int,
        default=DEFAULT_PER_TASK,
        help="每类 few-shot 数量，论文当前建议为 30。",
    )
    parser.add_argument(
        "--max_per_pest",
        type=int,
        default=DEFAULT_MAX_PER_PEST_PER_TASK,
        help="同一任务下每个 (crop,pest) 优先最多抽取多少条。",
    )
    parser.add_argument(
        "--candidate_multiplier",
        type=int,
        default=DEFAULT_CANDIDATE_MULTIPLIER,
        help="每个素材条目每任务生成候选次数，默认 4。",
    )
    parser.add_argument(
        "--require_eval",
        action="store_true",
        help="正式模式：若 qa_test.json 不存在则直接报错。",
    )
    parser.add_argument(
        "--semantic_audit",
        action="store_true",
        help="启用 BGE 高语义相似度过滤。",
    )
    parser.add_argument(
        "--semantic_threshold",
        type=float,
        default=0.94,
        help="few-shot vs eval 同任务问题的 BGE cosine 过滤阈值。",
    )
    parser.add_argument(
        "--bge_model",
        type=str,
        default=None,
        help="BGE 模型名；不填则使用 evidence_prompting_v3.py 默认值。",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="BGE 设备，如 cuda / cpu；不填自动判断。",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help="随机种子。",
    )

    return parser.parse_args()


# ============================================================
# 11. main
# ============================================================

def main():
    args = parse_args()

    global RANDOM_SEED
    RANDOM_SEED = args.seed
    random.seed(RANDOM_SEED)

    if args.per_task <= 0:
        raise ValueError("--per_task 必须 > 0")
    if args.max_per_pest <= 0:
        raise ValueError("--max_per_pest 必须 > 0")
    if args.candidate_multiplier <= 0:
        raise ValueError("--candidate_multiplier 必须 > 0")
    if not (0.0 < args.semantic_threshold <= 1.0):
        raise ValueError("--semantic_threshold 必须在 (0,1] 内")

    target_total = args.per_task * len(VALID_TASKS)

    print("=" * 68)
    print("Generate Dynamic Few-shot Pool")
    print("=" * 68)
    print(f"任务数：{len(VALID_TASKS)}")
    print(f"每类目标：{args.per_task}")
    print(f"总目标：{target_total}")
    print(f"随机种子：{RANDOM_SEED}")
    print(f"测试集：{args.eval_file}")
    print(f"输出：{args.output}")
    print(f"语义过滤：{'ON' if args.semantic_audit else 'OFF'}")
    print("=" * 68)

    # 1) 加载最终测试集约束
    eval_constraints = load_eval_constraints(
        eval_file=args.eval_file,
        require_eval=args.require_eval,
    )

    # 2) 从知识库生成经过 qav4 自动 QC 的候选
    clean = build_clean_candidates(
        candidate_multiplier=args.candidate_multiplier
    )

    # 3) 删除与最终评测集的问题/evidence 重叠
    filtered, exact_report = filter_eval_overlap(
        candidates_by_task=clean,
        eval_constraints=eval_constraints,
    )

    # 4) 可选 BGE 语义相似度泄漏过滤
    semantic_report = {}
    if args.semantic_audit:
        filtered, semantic_report = semantic_filter_against_eval(
            candidates_by_task=filtered,
            eval_constraints=eval_constraints,
            threshold=args.semantic_threshold,
            bge_model=args.bge_model,
            device=args.device,
        )

    print("\n[Available after leakage filtering]")
    for t in VALID_TASKS:
        print(f"  {t}: {len(filtered.get(t, []))}")

    # 5) 六类各抽 per_task 条，优先保持 crop/pest 多样性
    fewshot_pool = balanced_diverse_sample(
        candidates_by_task=filtered,
        per_task=args.per_task,
        max_per_pest=args.max_per_pest,
    )

    # 6) 最终硬审计
    final_report = audit_final_pool(
        pool=fewshot_pool,
        eval_constraints=eval_constraints,
        per_task=args.per_task,
    )

    # 7) 保存
    save_json(fewshot_pool, args.output)

    audit_report = {
        "generator": "generate_fewshot_pool_v2.py",
        "seed": RANDOM_SEED,
        "per_task": args.per_task,
        "target_total": target_total,
        "candidate_multiplier": args.candidate_multiplier,
        "max_per_pest": args.max_per_pest,
        "eval_file": args.eval_file,
        "eval_exists": eval_constraints["exists"],
        "semantic_audit_enabled": args.semantic_audit,
        "semantic_threshold": args.semantic_threshold if args.semantic_audit else None,
        "exact_and_evidence_filter": exact_report,
        "semantic_filter": semantic_report,
        "final_audit": final_report,
        "manual_review_note": (
            "当前所有 review_status 均为 pending。"
            "正式论文实验前建议由人工逐条核验 question-evidence-answer 一致性，"
            "审核通过后再冻结 fewshot_pool.json。"
        ),
    }
    save_json(audit_report, args.audit_output)

    review_tsv = write_review_tsv(fewshot_pool, args.output)

    # 8) 控制台统计
    task_cnt = Counter(x["task"] for x in fewshot_pool)

    print("\n" + "=" * 68)
    print("Few-shot pool 生成完成")
    print("=" * 68)
    print(f"总数：{len(fewshot_pool)}")
    for t in VALID_TASKS:
        print(f"  {t}: {task_cnt.get(t, 0)}")

    print(f"\n唯一 evidence IDs：{final_report['unique_text_evidence_ids']}")
    print(f"与 eval 问题完全重叠：{final_report['eval_exact_question_overlap_count']}")
    print(f"与 eval text evidence 重叠：{final_report['eval_text_evidence_overlap_count']}")
    print(f"与 eval KG edge 重叠：{final_report['eval_graph_evidence_overlap_count']}")
    print(f"池内重复问题：{final_report['duplicate_question_count']}")
    print(f"作物覆盖数：{final_report['num_crops']}")
    print(f"(crop,pest) 覆盖数：{final_report['num_crop_pest_pairs']}")

    print("\n输出文件：")
    print(f"  few-shot pool : {args.output}")
    print(f"  audit report  : {args.audit_output}")
    print(f"  review TSV    : {review_tsv}")

    print(
        "\n[IMPORTANT] 当前只是自动生成候选并通过程序质检。"
        "正式实验前请人工审核，尤其检查：\n"
        "  1. 问题是否自然且无答案泄漏；\n"
        "  2. answer 中每个确定事实是否都由 evidence 支持；\n"
        "  3. 症状、作物、受害部位是否一致；\n"
        "  4. 相似病虫害鉴别是否真的具有鉴别点；\n"
        "  5. 防治措施是否包含 evidence 外的药剂/剂量；\n"
        "  6. 多条件综合推理是否至少包含两个互补证据，且来自同一 record/source；\n"
        "  7. 相似病虫害鉴别是否具有真实 KG 相似关系，而非仅因为属于同一作物。"
    )
    print("=" * 68)


if __name__ == "__main__":
    main()
