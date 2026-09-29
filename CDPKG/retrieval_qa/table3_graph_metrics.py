# -*- coding: utf-8 -*-
"""
table3_graph_metrics.py —— Table 3 图通道指标统一计算
======================================================
对全部检索方法计算三个图通道指标（与参数扫描同口径）：
  - Graph-Hit@3    : gold_pest 出现在前 3 条图谱证据 head/tail/anchor 中
  - Graph-Recall@5 : gold 图谱邻域被前 5 条图谱证据覆盖的比例
  - Anchor-MRR     : gold_pest 在锚点序列中的排名倒数

方法：full / graph-only / GraphRAG-Local / G-Retriever / LightRAG / PathRAG / SG-RAG
全 480 题，统一 pipeline（QueryParser 缓存共享）。
"""
import sys
import os
import json
import math

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import NODES_CSV, RELS_CSV, INDEX_DIR, EVAL_FILE, RESULT_DIR, LLM
from modules.kg import CropPestKG
from modules.retriever import Embedder, VectorIndex, BM25Index, TextBlockStore, Retriever
from modules.llm import LLMClient


def gold_graph_neighbors(kg, pest):
    """gold_pest 的图谱直接邻居名 + 自身。Graph-Recall 的分子分母依据。"""
    nid = kg.find_node(pest)
    if nid is None:
        return {pest}
    nb = {pest}
    for tgt, rel in kg.neighbors(nid):
        tname = kg.node_name(tgt)
        if tname and tname.lower() != 'nan':
            nb.add(tname)
    return nb


def load_pipeline():
    kg = CropPestKG(NODES_CSV, RELS_CSV)
    store = TextBlockStore.load(os.path.join(INDEX_DIR, 'text_blocks.json'))
    with open(os.path.join(INDEX_DIR, 'vector_doc_ids.json'), encoding='utf-8') as f:
        doc_ids = json.load(f)
    vindex = VectorIndex.load(os.path.join(INDEX_DIR, 'vector.faiss'), doc_ids)
    bindex = BM25Index.load(os.path.join(INDEX_DIR, 'bm25.pkl'))
    llm = LLMClient(LLM)
    r = Retriever(kg, store, vindex, bindex, Embedder(), llm_client=llm)
    return r, kg, llm


def load_engines(retriever, kg, llm):
    """按 run_eval_resumable 的加载方式实例化各引擎"""
    from graphrag.search import GraphRAGEngine
    from graphrag.summarize import load_summaries, load_communities
    from graphrag.gretriever import GRetrieverEngine
    from graphrag.lightrag import LightRAGEngine
    from graphrag.pathrag import PathRAGEngine
    from graphrag.sgrag import SGRAGEngine
    from config import GRAPHRAG_DATA, GRAPHRAG

    engines = {}
    summaries = load_summaries(os.path.join(GRAPHRAG_DATA, 'community_summaries.json'))
    comm_map, _ = load_communities(os.path.join(GRAPHRAG_DATA, 'communities.json'))
    community_of = {}
    for cid, members in comm_map.items():
        for m in members:
            community_of[m] = cid
    engines['GraphRAG-Local'] = GraphRAGEngine(
        kg, retriever, llm, summaries=summaries, community_of=community_of,
        summary_top_k=GRAPHRAG['summary_top_k'])
    engines['G-Retriever'] = GRetrieverEngine(kg, retriever)
    engines['LightRAG'] = LightRAGEngine(kg, retriever)
    engines['PathRAG'] = PathRAGEngine(kg, retriever)
    engines['SG-RAG'] = SGRAGEngine(kg, retriever)
    return engines


def get_graph_evidence(method, retriever, engine, question):
    """返回 (text_ev, graph_ev)，统一入口"""
    if method == 'full':
        te, ge, _ = retriever.get_evidence(question, variant='full')
        return te, ge
    if method == 'graph-only':
        te, ge, _ = retriever.get_evidence(question, variant='graph-only')
        return te, ge
    # 图RAG引擎
    if method == 'GraphRAG-Local':
        te, ge, _ = engine.local_search(question)
        return te, ge
    te, ge, _ = engine.search(question)
    return te, ge


def main():
    r, kg, llm = load_pipeline()
    engines = load_engines(r, kg, llm)
    methods = ['full', 'graph-only', 'GraphRAG-Local', 'G-Retriever',
               'LightRAG', 'PathRAG', 'SG-RAG']
    qas = json.load(open(EVAL_FILE, encoding='utf-8'))

    results = {}
    for method in methods:
        engine = engines.get(method)
        hit3 = recall5 = mrr = 0.0
        n = 0
        n_recall = 0
        n_mrr = 0
        for qa in qas:
            pest = qa.get('gold_pest')
            if not pest:
                continue
            te, ge = get_graph_evidence(method, r, engine, qa['question'])
            n += 1
            # Graph-Hit@3
            top3_names = set()
            for g in ge[:3]:
                top3_names.add(g.get('head') or '')
                top3_names.add(g.get('tail') or '')
                top3_names.add(g.get('anchor') or '')
            if pest in top3_names:
                hit3 += 1.0
            # Graph-Recall@5（gold 邻域覆盖）
            gnb = gold_graph_neighbors(kg, pest)
            if gnb and ge:
                ev5 = set()
                for g in ge[:5]:
                    ev5.add(g.get('head') or '')
                    ev5.add(g.get('tail') or '')
                recall5 += len(ev5 & gnb) / len(gnb)
                n_recall += 1
            # Anchor-MRR
            anchors = [g.get('anchor') for g in ge if g.get('anchor')]
            if anchors:
                for i, a in enumerate(anchors):
                    if a == pest:
                        mrr += 1.0 / (i + 1)
                        break
                else:
                    mrr += 0.0
                n_mrr += 1
        results[method] = {
            'n': n,
            'Graph-Hit@3': hit3 / n if n else 0.0,
            'Graph-Recall@5': recall5 / n_recall if n_recall else 0.0,
            'Anchor-MRR': mrr / n_mrr if n_mrr else 0.0,
            'mrr_n': n_mrr,
        }
        print(f'{method:<14} Hit@3={results[method]["Graph-Hit@3"]:.4f}  '
              f'Recall@5={results[method]["Graph-Recall@5"]:.4f}  '
              f'MRR={results[method]["Anchor-MRR"]:.4f}  (mrr_n={n_mrr})', flush=True)

    os.makedirs(RESULT_DIR, exist_ok=True)
    with open(os.path.join(RESULT_DIR, 'table3_graph_metrics.json'), 'w',
              encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print(f'\n完成，写入 data/results/table3_graph_metrics.json')


if __name__ == '__main__':
    main()
