# -*- coding: utf-8 -*-
"""
supplementary_metrics.py —— 补充指标计算
==========================================
补齐论文表3/表6 缺失的数据：
1. 基线（naive/hybrid/graph-only）的 P@5/R@5/NDCG@5（与 Ours 同口径，多证据）
2. 图重排序精确率：图谱证据中 gold 相关三元组占比（用于解释图重排序价值）
3. 实体定位率：图谱锚点命中 gold_pest 的比例

输出：data/results/supplementary_metrics.json
"""
import sys
import os
import json
import math
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from modules.kg import CropPestKG
from modules.retriever import Embedder, VectorIndex, BM25Index, TextBlockStore, Retriever
from modules.llm import LLMClient
from config import NODES_CSV, RELS_CSV, INDEX_DIR, EVAL_FILE, RESULT_DIR


def ndcg5(relevance):
    dcg = sum(h / math.log2(i + 2) for i, h in enumerate(relevance[:5]))
    ideal = sorted(relevance, reverse=True)[:5]
    idcg = sum(h / math.log2(i + 2) for i, h in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def main():
    kg = CropPestKG(NODES_CSV, RELS_CSV)
    store = TextBlockStore.load(os.path.join(INDEX_DIR, 'text_blocks.json'))
    with open(os.path.join(INDEX_DIR, 'vector_doc_ids.json'), encoding='utf-8') as f:
        doc_ids = json.load(f)
    vindex = VectorIndex.load(os.path.join(INDEX_DIR, 'vector.faiss'), doc_ids)
    bindex = BM25Index.load(os.path.join(INDEX_DIR, 'bm25.pkl'))
    llm = LLMClient()
    r = Retriever(kg, store, vindex, bindex, Embedder(), llm_client=llm)
    qas = json.load(open(EVAL_FILE, encoding='utf-8'))

    # 多证据标准块（每个病虫害的所有文本块）
    pest_blocks = defaultdict(set)
    for b in store.blocks:
        if b.get('entity_name'):
            pest_blocks[b['entity_name']].add(b['id'])

    variants = ['full', 'naive-text', 'hybrid-text', 'graph-only',
                '-no-text-rrf', '-no-graph-rerank', '-no-semantic-graph',
                '-no-entity-graph']
    results = {}

    for variant in variants:
        acc_p = acc_r = acc_f = acc_nd = 0.0
        graph_prec_total = 0
        graph_prec_hit = 0
        anchor_total = 0
        anchor_hit = 0
        n = 0
        for qa in qas:
            pest = qa.get('gold_pest')
            if not pest:
                continue
            te, ge, det = r.get_evidence(qa['question'], variant=variant)
            n += 1

            # 1) 文本检索 P@5/R@5/NDCG@5（多证据）
            std_ids = pest_blocks.get(pest, set())
            top_ids = [e['id'] for e in te[:5]]
            hits = [1 if did in std_ids else 0 for did in top_ids]
            nh = sum(hits)
            denom = min(5, len(std_ids))
            acc_p += nh / 5
            acc_r += nh / denom if denom else 0
            acc_f += 2 * nh / (5 + denom) if nh else 0
            acc_nd += ndcg5(hits)

            # 2) 图谱证据精确率：gold 相关的三元组占比
            if ge:
                gold_related = sum(
                    1 for g in ge
                    if pest in (g.get('head') or '') or pest in (g.get('tail') or '')
                    or pest in (g.get('anchor') or '')
                )
                graph_prec_hit += gold_related
                graph_prec_total += len(ge)

            # 3) 实体定位率：图谱锚点命中 gold 的比例
            anchors = {g.get('anchor') for g in ge if g.get('anchor')}
            if anchors:
                anchor_total += 1
                if pest in anchors:
                    anchor_hit += 1

        m = max(1, n)
        results[variant] = {
            'n': m,
            'P5': acc_p / m,
            'R5': acc_r / m,
            'F1@5': acc_f / m,
            'NDCG5': acc_nd / m,
            'graph_precision': graph_prec_hit / graph_prec_total if graph_prec_total else 0.0,
            'graph_precision_edges': graph_prec_total,
            'entity_loc_rate': anchor_hit / anchor_total if anchor_total else 0.0,
            'entity_loc_n': anchor_total,
        }
        print(f'{variant:<20} P5={results[variant]["P5"]:.3f} '
              f'R5={results[variant]["R5"]:.3f} '
              f'NDCG5={results[variant]["NDCG5"]:.3f} '
              f'GPrec={results[variant]["graph_precision"]:.3f} '
              f'ELoc={results[variant]["entity_loc_rate"]:.3f}', flush=True)

    os.makedirs(RESULT_DIR, exist_ok=True)
    with open(os.path.join(RESULT_DIR, 'supplementary_metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f'\n已保存: {os.path.join(RESULT_DIR, "supplementary_metrics.json")}')


if __name__ == '__main__':
    main()
