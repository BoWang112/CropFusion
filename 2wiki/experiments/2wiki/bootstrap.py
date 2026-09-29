# -*- coding: utf-8 -*-
"""
bootstrap.py —— paired bootstrap 置信区间
==========================================
从 8 个 baseline 中找 Answer F1 最高的，计算 Ours-Full 与它的 delta，
输出 Mean Delta 与 95% CI（N_BOOTSTRAP=1000, SEED=42）。
"""
import sys
import os
import json
import random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import PRED, normalize_answer, f1_score
import numpy as np

N_BOOTSTRAP = 1000
SEED = 42

METHOD_FILES = {
    'LLM-No Retrieval': 'llm_no_retrieval.jsonl',
    'NaiveRAG': 'naive_rag.jsonl',
    'Hybrid RAG': 'hybrid_rag.jsonl',
    'GraphRAG': 'graphrag.jsonl',
    'LightRAG': 'lightrag.jsonl',
    'G-Retriever': 'g_retriever.jsonl',
    'PathRAG': 'pathrag.jsonl',
    'SG-RAG': 'sg_rag.jsonl',
    'Ours-Full': 'ours_full.jsonl',   # run_method.py: run_ours_full（四路检索 + 两级融合 + 动态 few-shot）
}


def per_question_f1(rows):
    """返回按 question id 排序的 per-question F1 列表（顺序一致）"""
    d = {}
    for r in rows:
        d[r['id']] = f1_score(normalize_answer(r['prediction']),
                              normalize_answer(r['gold_answer']))
    return d


def main():
    # 加载每个方法的 per-question F1（对齐到公共 question id 集）
    all_d = {}
    common_ids = None
    for name, f in METHOD_FILES.items():
        rows = [json.loads(l) for l in open(os.path.join(PRED, f), encoding='utf-8')]
        d = per_question_f1(rows)
        all_d[name] = d
        ids = set(d.keys())
        common_ids = ids if common_ids is None else (common_ids & ids)
    common_ids = sorted(common_ids)
    print(f'共同 question 数: {len(common_ids)}')

    f1_mat = {}
    for name, d in all_d.items():
        f1_mat[name] = np.array([d[i] for i in common_ids])

    ours = f1_mat['Ours-Full']
    # 找 best baseline（8 个非 Ours 中 F1 最高）
    baselines = [n for n in METHOD_FILES if n != 'Ours-Full']
    base_f1 = {n: float(f1_mat[n].mean()) for n in baselines}
    best_base = max(base_f1, key=base_f1.get)
    print(f'Best baseline: {best_base} (F1={base_f1[best_base]:.4f})')
    print(f'Ours-Full F1: {float(ours.mean()):.4f}')
    delta_obs = float(ours.mean()) - base_f1[best_base]
    print(f'Delta F1 = {delta_obs:.4f}')

    # paired bootstrap
    rng = np.random.default_rng(SEED)
    n = len(common_ids)
    deltas = []
    for _ in range(N_BOOTSTRAP):
        idx = rng.integers(0, n, size=n)
        d = float(ours[idx].mean() - f1_mat[best_base][idx].mean())
        deltas.append(d)
    deltas = np.array(deltas)
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    print(f'\nMean Delta = {deltas.mean():.4f}')
    print(f'95% CI = [{lo:.4f}, {hi:.4f}]')
    print(f'CI 不包含 0（显著）: {not (lo <= 0 <= hi)}')
    return {'best_baseline': best_base, 'best_baseline_f1': base_f1[best_base],
            'ours_f1': float(ours.mean()), 'delta': delta_obs,
            'ci': [round(lo, 4), round(hi, 4)],
            'significant': not (lo <= 0 <= hi)}


if __name__ == '__main__':
    main()
