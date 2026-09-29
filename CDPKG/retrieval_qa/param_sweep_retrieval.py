# -*- coding: utf-8 -*-
"""
param_sweep_retrieval.py —— 检索参数敏感性扫描（单变量，只算检索指标）
========================================================================
对每个参数、每个取值跑完整检索（get_evidence, variant='full'），
计算多证据口径的 P@3 / R@3 / F1@3 / NDCG@3 及图谱证据精确率。

单变量设计：一次只改一个参数，其余保持默认值（论文标准做法）。

用法：
  .venv-re/Scripts/python.exe -X utf8 param_sweep_retrieval.py \
      --param graph_hop --values 1 2 3 4 \
      [--subset N] [--out data/results/param_sweep.json]
"""
import sys
import os
import json
import math

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from config import NODES_CSV, RELS_CSV, INDEX_DIR, EVAL_FILE, RESULT_DIR
from modules.kg import CropPestKG
from modules.retriever import Embedder, VectorIndex, BM25Index, TextBlockStore, Retriever
from modules.llm import LLMClient

# 默认参数（config.RETRIEVAL）
DEFAULT = dict(config.RETRIEVAL)

# 各参数扫描取值（单变量）
SWEEP = {
    'graph_hop': [1, 2, 3, 4],
    'rrf_k': [30, 60, 90, 120],
    'text_top_k': [5, 10, 15, 20],
    'graph_top_k': [2, 3, 5, 8],
}


def ndcg_k(relevance, k=3):
    dcg = sum(h / math.log2(i + 2) for i, h in enumerate(relevance[:k]))
    ideal = sorted(relevance, reverse=True)[:k]
    idcg = sum(h / math.log2(i + 2) for i, h in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def gold_graph_neighbors(kg, pest):
    """gold_pest 的图谱邻域（症状有/为害/发病于/防治方法有 等直接邻居名）。
    用于 Graph-Recall@K：衡量"回答该问题所需信息"在前 K 条图谱证据中的覆盖度。"""
    nid = kg.find_node(pest)
    if nid is None:
        return set()
    nb = set()
    for tgt, rel in kg.neighbors(nid):
        tname = kg.node_name(tgt)
        if tname and tname.lower() != 'nan':
            nb.add(tname)
        # 也把直接关系（如 发病于作物→水稻）本身算入
    # 加入 gold 自身
    nb.add(pest)
    return nb


def load_pipeline():
    kg = CropPestKG(NODES_CSV, RELS_CSV)
    store = TextBlockStore.load(os.path.join(INDEX_DIR, 'text_blocks.json'))
    with open(os.path.join(INDEX_DIR, 'vector_doc_ids.json'), encoding='utf-8') as f:
        doc_ids = json.load(f)
    vindex = VectorIndex.load(os.path.join(INDEX_DIR, 'vector.faiss'), doc_ids)
    bindex = BM25Index.load(os.path.join(INDEX_DIR, 'bm25.pkl'))
    r = Retriever(kg, store, vindex, bindex, Embedder(), llm_client=LLMClient())
    return r, store


def run_sweep(param, values, r, store, qas, subset=0):
    """对 param 的每个取值跑全套指标。返回 {value: metrics}
    r 复用同一实例（_parse_cache 保留），避免每个参数值重复调用 QueryParser。"""
    if subset and subset > 0:
        qas = qas[:subset]

    # 多证据标准块
    pest_blocks = {}
    for b in store.blocks:
        if b.get('entity_name'):
            pest_blocks.setdefault(b['entity_name'], set()).add(b['id'])

    out = {}
    for val in values:
        # 覆盖参数（其余保持默认）
        cfg = dict(DEFAULT)
        cfg[param] = val
        config.RETRIEVAL.clear()
        config.RETRIEVAL.update(cfg)

        acc_p = acc_r = acc_f = acc_nd = 0.0
        gprec_hit = gprec_total = 0
        loc_hit = loc_total = 0
        g_recall = {k: [] for k in (1, 3, 5)}   # Graph-Recall@K（gold 图谱邻域在前 K 条图谱证据中覆盖度）
        g_hit = {k: [] for k in (1, 3, 5)}       # Graph-Hit@K（gold_pest 出现在前 K 条图谱证据中）
        anchor_mrr = []                          # Anchor-MRR（gold_pest 在图谱证据锚点中的排名）
        coverage_sum = []                        # 全图谱证据对 gold 邻域的覆盖比例（hop 敏感）
        n = 0
        for qa in qas:
            pest = qa.get('gold_pest')
            if not pest:
                continue
            te, ge, det = r.get_evidence(qa['question'], variant='full')
            n += 1
            # 文本多证据 @3
            std_ids = pest_blocks.get(pest, set())
            top_ids = [e['id'] for e in te[:3]]
            hits = [1 if did in std_ids else 0 for did in top_ids]
            nh = sum(hits)
            denom = min(3, len(std_ids))
            acc_p += nh / 3
            acc_r += nh / denom if denom else 0
            acc_f += 2 * nh / (3 + denom) if nh else 0
            acc_nd += ndcg_k(hits)
            # 图谱证据精确率
            if ge:
                gold_related = sum(
                    1 for g in ge
                    if pest in (g.get('head') or '') or pest in (g.get('tail') or '')
                    or pest in (g.get('anchor') or '')
                )
                gprec_hit += gold_related
                gprec_total += len(ge)
            # 实体定位率（锚点命中 gold）
            anchors = [g.get('anchor') for g in ge if g.get('anchor')]
            if anchors:
                loc_total += 1
                if pest in anchors:
                    loc_hit += 1
                # Anchor-MRR：gold_pest 在锚点序列中的首个位置
                for i, a in enumerate(anchors):
                    if a == pest:
                        anchor_mrr.append(1.0 / (i + 1))
                        break
                else:
                    anchor_mrr.append(0.0)
            # Graph-Recall@K：gold 图谱邻域名在前 K 条图谱证据三元组中出现比例
            # （对 hop 不敏感：hop 改变子图扩展深度而非前 K 条展示）
            # graph_coverage：全部图谱证据对 gold 邻域的覆盖比例（hop 敏感，
            #   hop 越大覆盖越全；代价是图谱精确率/边数增加）
            gnb = gold_graph_neighbors(r.kg, pest)
            if gnb and ge:
                for K in (1, 3, 5):
                    ev_names = set()
                    for g in ge[:K]:
                        ev_names.add(g.get('head') or '')
                        ev_names.add(g.get('tail') or '')
                    hit = len(ev_names & gnb)
                    g_recall[K].append(hit / len(gnb))
                    # Graph-Hit@K：gold_pest 自身出现在前 K 条图谱证据中
                    g_hit[K].append(1 if pest in ev_names else 0)
                all_names = set()
                for g in ge:
                    all_names.add(g.get('head') or '')
                    all_names.add(g.get('tail') or '')
                coverage_sum.append(len(all_names & gnb) / len(gnb))

        m = max(1, n)
        out[str(val)] = {
            'n': m,
            'P3': acc_p / m,
            'R3': acc_r / m,
            'F1_3': acc_f / m,
            'NDCG3': acc_nd / m,
            'graph_precision': gprec_hit / gprec_total if gprec_total else 0.0,
            'graph_precision_edges': gprec_total,
            'entity_loc_rate': loc_hit / loc_total if loc_total else 0.0,
            'entity_loc_n': loc_total,
            'g_recall1': sum(g_recall[1]) / len(g_recall[1]) if g_recall[1] else 0.0,
            'g_recall3': sum(g_recall[3]) / len(g_recall[3]) if g_recall[3] else 0.0,
            'g_recall5': sum(g_recall[5]) / len(g_recall[5]) if g_recall[5] else 0.0,
            'g_hit1': sum(g_hit[1]) / len(g_hit[1]) if g_hit[1] else 0.0,
            'g_hit3': sum(g_hit[3]) / len(g_hit[3]) if g_hit[3] else 0.0,
            'g_hit5': sum(g_hit[5]) / len(g_hit[5]) if g_hit[5] else 0.0,
            'anchor_mrr': sum(anchor_mrr) / len(anchor_mrr) if anchor_mrr else 0.0,
            'graph_coverage': sum(coverage_sum) / len(coverage_sum) if coverage_sum else 0.0,
        }
        print(f'[{param}={val}]  P3={out[str(val)]["P3"]:.4f}  '
              f'R3={out[str(val)]["R3"]:.4f}  '
              f'F1={out[str(val)]["F1_3"]:.4f}  '
              f'NDCG={out[str(val)]["NDCG3"]:.4f}  '
              f'GPrec={out[str(val)]["graph_precision"]:.4f}  '
              f'ELoc={out[str(val)]["entity_loc_rate"]:.4f}  '
              f'GHit@1={out[str(val)]["g_hit1"]:.4f}  '
              f'GHit@3={out[str(val)]["g_hit3"]:.4f}  '
              f'GHit@5={out[str(val)]["g_hit5"]:.4f}  '
              f'GR@5={out[str(val)]["g_recall5"]:.4f}  '
              f'GCov={out[str(val)]["graph_coverage"]:.4f}  '
              f'MRR={out[str(val)]["anchor_mrr"]:.4f}  '
              f'(edges={out[str(val)]["graph_precision_edges"]}, n={m})', flush=True)

    # 恢复默认
    config.RETRIEVAL.clear()
    config.RETRIEVAL.update(DEFAULT)
    return out


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--param', default='graph_hop',
                    choices=list(SWEEP.keys()) + ['all'],
                    help='扫描参数（all=全参数单变量扫描）')
    ap.add_argument('--values', nargs='*', type=int, default=None,
                    help='取值列表（默认用 SWEEP 定义）')
    ap.add_argument('--subset', type=int, default=0, help='仅前 N 条（0=全部）')
    ap.add_argument('--out', default=os.path.join(RESULT_DIR, 'param_sweep.json'))
    args = ap.parse_args()

    if args.param == 'all':
        all_results = {}
        # pipeline 只建一次，全部参数共享 QueryParser 缓存
        r, store = load_pipeline()
        qas = json.load(open(EVAL_FILE, encoding='utf-8'))
        for p in SWEEP:
            print(f'\n========== 扫描参数 {p} ==========', flush=True)
            all_results[p] = run_sweep(p, SWEEP[p], r, store, qas, subset=args.subset)
        os.makedirs(RESULT_DIR, exist_ok=True)
        with open(args.out, 'w', encoding='utf-8') as f:
            json.dump({'default': DEFAULT, 'sweep': all_results}, f,
                      ensure_ascii=False, indent=1)
        print(f'\n全参数扫描完成，结果写入 {args.out}')
    else:
        values = args.values or SWEEP[args.param]
        r, store = load_pipeline()
        qas = json.load(open(EVAL_FILE, encoding='utf-8'))
        res = run_sweep(args.param, values, r, store, qas, subset=args.subset)
        os.makedirs(RESULT_DIR, exist_ok=True)
        with open(args.out, 'w', encoding='utf-8') as f:
            json.dump({'param': args.param, 'default': DEFAULT, 'results': res},
                      f, ensure_ascii=False, indent=1)
        print(f'\n完成，结果写入 {args.out}')
