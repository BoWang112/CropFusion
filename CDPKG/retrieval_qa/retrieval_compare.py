# -*- coding: utf-8 -*-
"""
retrieval_compare.py —— 检索对比 + 消融的完整证据指标
=======================================================
对每个变体计算：
  - 文本证据排序质量：Evidence-Top1 / Top3 / Top5 / Avg-Rank
  - 图谱证据命中率：图谱证据中 head/tail/anchor 命中 gold_pest 的比例

输出：data/results/retrieval_compare.json
"""
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from modules.kg import CropPestKG
from modules.retriever import Embedder, VectorIndex, BM25Index, TextBlockStore, Retriever
from modules.llm import LLMClient
from config import NODES_CSV, RELS_CSV, INDEX_DIR, EVAL_FILE, RESULT_DIR

VARIANTS = [
    'full', '-no-text-rrf', '-no-graph-rerank', '-no-semantic-graph',
    '-no-entity-graph', '-no-evidence-prompt', '-no-fewshot', '-no-diag-constraints',
    'naive-text', 'hybrid-text', 'graph-only',
]


def load_pipeline():
    kg = CropPestKG(NODES_CSV, RELS_CSV)
    store = TextBlockStore.load(os.path.join(INDEX_DIR, 'text_blocks.json'))
    with open(os.path.join(INDEX_DIR, 'vector_doc_ids.json'), encoding='utf-8') as f:
        doc_ids = json.load(f)
    vindex = VectorIndex.load(os.path.join(INDEX_DIR, 'vector.faiss'), doc_ids)
    bindex = BM25Index.load(os.path.join(INDEX_DIR, 'bm25.pkl'))
    llm = LLMClient()
    r = Retriever(kg, store, vindex, bindex, Embedder(), llm_client=llm)
    return r


def main():
    r = load_pipeline()
    qas = json.load(open(EVAL_FILE, encoding='utf-8'))
    results = {}
    for variant in VARIANTS:
        t1 = t3 = t5 = 0
        ranks = []
        ghit = 0
        n = 0
        for qa in qas:
            pest = qa.get('gold_pest')
            if not pest:
                continue
            te, ge, det = r.get_evidence(qa['question'], variant=variant)
            # 文本证据排序
            rank = None
            for i, e in enumerate(te, start=1):
                if e.get('entity') == pest:
                    rank = i
                    break
            if rank:
                n += 1
                ranks.append(rank)
                if rank == 1:
                    t1 += 1
                if rank <= 3:
                    t3 += 1
                if rank <= 5:
                    t5 += 1
            else:
                n += 1
                ranks.append(11)  # 未召回
            # 图谱证据命中
            if any(pest in (g.get('head') or '') or pest in (g.get('tail') or '')
                   or pest in (g.get('anchor') or '') for g in ge):
                ghit += 1
        m = max(1, n)
        results[variant] = {
            'text_top1': t1 / m,
            'text_top3': t3 / m,
            'text_top5': t5 / m,
            'text_avg_rank': sum(ranks) / len(ranks) if ranks else 0,
            'graph_hit': ghit / m,
            'n': m,
        }
        print(f'{variant:<22} T1={results[variant]["text_top1"]:.3f} '
              f'T3={results[variant]["text_top3"]:.3f} '
              f'AvgRank={results[variant]["text_avg_rank"]:.2f} '
              f'GraphHit={results[variant]["graph_hit"]:.3f}', flush=True)

    os.makedirs(RESULT_DIR, exist_ok=True)
    with open(os.path.join(RESULT_DIR, 'retrieval_compare.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f'\n已保存: {os.path.join(RESULT_DIR, "retrieval_compare.json")}')


if __name__ == '__main__':
    main()
