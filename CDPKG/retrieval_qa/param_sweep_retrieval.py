# -*- coding: utf-8 -*-
"""
param_sweep_retrieval.py —— 检索参数敏感性扫描（单变量，只算检索指标）
========================================================================
对每个参数、每个取值跑完整检索（get_evidence, variant='full'），
计算统一原子证据口径的 P@3 / R@3 / F1@3 / NDCG@3 及图谱证据精确率。

检索指标本身不使用任务标签；需要构建提示词的 param_sweep_gen.py
统一调用自动任务分类，不读取测试集真实任务标签。

单变量设计：一次只改一个参数，其余保持默认值（论文标准做法）。

用法：
  .venv-re/Scripts/python.exe -X utf8 param_sweep_retrieval.py \
      --param graph_hop --values 1 2 3 4 \
      [--subset N] [--out data/results/param_sweep.json]
"""
import sys
import os
import json
import importlib.util

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from config import NODES_CSV, RELS_CSV, INDEX_DIR, EVAL_FILE, RESULT_DIR, EXTRACT_JSON
from config import get_llm_config, LLM_PROVIDERS
from modules.kg import CropPestKG
from modules.retriever import Embedder, VectorIndex, BM25Index, TextBlockStore, Retriever
from modules.llm import LLMClient

# 复用 3_evaluate.py 的统一原子证据指标（文件名以数字开头，用 importlib 加载）
_EVAL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '3_evaluate.py')
_spec = importlib.util.spec_from_file_location('cdpkg_eval3', _EVAL_PATH)
eval3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(eval3)

# 默认参数（config.RETRIEVAL）
DEFAULT = dict(config.RETRIEVAL)

# 各参数扫描取值（单变量）
SWEEP = {
    'graph_rerank_lambda': [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    'graph_hop': [1, 2, 3, 4],
    'semantic_top_k': [5, 10, 20, 30, 40],
    'entity_top_k': [5, 10, 20, 30, 40],
    'rrf_k': [30, 60, 90, 120],
    'text_top_k': [3, 5, 10, 15, 20],
    'graph_top_k': [2, 3, 5, 8, 10],
}

# 二维组合扫描：图扩展跳数 × 候选锚点数量
TWO_D_SWEEP = {
    'hop_x_anchor': (
        ('graph_hop', 'semantic_top_k'),
        [(h, a) for h in (1, 2, 3, 4) for a in (5, 10, 20, 30, 40)],
    ),
}


def gold_graph_neighbors(kg, pest):
    """gold_pest 的图谱邻域（症状有/为害/发病于/防治方法有 等直接邻居名）。
    用于 Graph-Recall@K：衡量"回答该问题所需信息"在前 K 条图谱证据中的覆盖度。"""
    nid = kg.find_pest(pest)
    if nid is None:
        return set()
    nb = set()
    for tgt, rel in kg.neighbors(nid):
        tname = kg.node_name(tgt)
        if tname and tname.lower() != 'nan':
            nb.add(tname)
    # 加入 gold 自身
    nb.add(pest)
    return nb


def load_pipeline(provider=None):
    kg = CropPestKG(NODES_CSV, RELS_CSV)
    store = TextBlockStore.load(os.path.join(INDEX_DIR, 'text_blocks.json'))
    with open(os.path.join(INDEX_DIR, 'vector_doc_ids.json'), encoding='utf-8') as f:
        doc_ids = json.load(f)
    vindex = VectorIndex.load(os.path.join(INDEX_DIR, 'vector.faiss'), doc_ids)
    bindex = BM25Index.load(os.path.join(INDEX_DIR, 'bm25.pkl'))
    r = Retriever(kg, store, vindex, bindex, Embedder(),
                  llm_client=LLMClient(get_llm_config(provider)))
    return r, store


def _param_label(param, val):
    if isinstance(param, (tuple, list)):
        return ','.join(f'{p}={v}' for p, v in zip(param, val))
    return f'{param}={val}'


def _value_key(param, val):
    if isinstance(param, (tuple, list)):
        return ','.join(str(v) for v in val)
    return str(val)


def run_sweep(param, values, r, store, qas, subset=0):
    """对单参数或二维参数组合跑全套指标。返回 {value_combo: metrics}

    param 为 str 时做单变量扫描；为 tuple 时同时设置多个参数（二维组合），
    其余参数保持 DEFAULT 不变。
    """
    if subset and subset > 0:
        qas = qas[:subset]

    # 统一 Gold 文本证据映射（与 3_evaluate.py 完全一致）
    blocks_by_entity, seg_index = eval3.build_text_gold_index(store, EXTRACT_JSON)

    out = {}
    for val in values:
        # 覆盖参数（其余保持默认）
        cfg = dict(DEFAULT)
        if isinstance(param, (tuple, list)):
            for p, v in zip(param, val):
                cfg[p] = v
        else:
            cfg[param] = val
        config.RETRIEVAL.clear()
        config.RETRIEVAL.update(cfg)
        key = _value_key(param, val)
        label = _param_label(param, val)

        acc_p = acc_r = acc_nd = 0.0
        rank_sum = 0.0
        n_eval = 0
        gprec_hit = gprec_total = 0
        loc_hit = loc_total = 0
        g_recall = {k: [] for k in (1, 3, 5)}   # Graph-Recall@K（gold 图谱邻域在前 K 条图谱证据中覆盖度）
        g_hit = {k: [] for k in (1, 3, 5)}       # Graph-Hit@K（gold_pest 出现在前 K 条图谱证据中）
        anchor_mrr = []                          # Anchor-MRR（gold_pest 在图谱证据锚点中的排名）
        coverage_sum = []                        # 全图谱证据对 gold 邻域的覆盖比例（hop 敏感）
        n = 0
        retrieval_failures = 0
        for qa in qas:
            pest = qa.get('gold_pest')
            if not pest:
                continue
            try:
                te, ge, det = r.get_evidence(qa['question'], variant='full')
            except Exception as e:
                # 检索失败按零分参与固定样本量统计，不缩小 n_eval
                retrieval_failures += 1
                print(f'  [{label}] 检索失败 {qa.get("id")}: {str(e)[:100]}',
                      flush=True)
                te, ge, det = [], [], {}
            n += 1

            # 统一原子证据指标（文本块 + 三元组；路径/子图已展开为三元组）
            ranked = eval3._build_ranked_atoms(
                te, ge, 'Ours-Retrieval',
                query=qa['question'], embedder=r.embedder)
            gold_text_atoms, gold_triples = \
                eval3._gold_for_atom_eval(
                    qa, blocks_by_entity, seg_index)
            atom_score = eval3._score_atom_ranking(
                ranked, gold_text_atoms, gold_triples, 3)
            if atom_score is not None:
                acc_p += atom_score['p']
                acc_r += atom_score['r']
                acc_nd += atom_score['ndcg']
                rank_sum += (atom_score['first_hit']
                             if atom_score['first_hit'] is not None else 4)
                n_eval += 1
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
        me = max(1, n_eval)
        p3 = acc_p / me
        r3 = acc_r / me
        f1_3 = (2 * p3 * r3 / (p3 + r3)) if (p3 + r3) > 0 else 0.0
        out[key] = {
            'n': m,
            'n_eval': n_eval,
            'retrieval_failures': retrieval_failures,
            'P3': p3,
            'R3': r3,
            'F1_3': f1_3,
            'NDCG3': acc_nd / me,
            'avg_rank': rank_sum / me if n_eval else 0.0,
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
        print(f'[{label}]  P3={out[key]["P3"]:.4f}  '
              f'R3={out[key]["R3"]:.4f}  '
              f'F1={out[key]["F1_3"]:.4f}  '
              f'NDCG={out[key]["NDCG3"]:.4f}  '
              f'GPrec={out[key]["graph_precision"]:.4f}  '
              f'ELoc={out[key]["entity_loc_rate"]:.4f}  '
              f'GHit@1={out[key]["g_hit1"]:.4f}  '
              f'GHit@3={out[key]["g_hit3"]:.4f}  '
              f'GHit@5={out[key]["g_hit5"]:.4f}  '
              f'GR@5={out[key]["g_recall5"]:.4f}  '
              f'GCov={out[key]["graph_coverage"]:.4f}  '
              f'MRR={out[key]["anchor_mrr"]:.4f}  '
              f'(edges={out[key]["graph_precision_edges"]}, '
              f'n={m}, n_eval={n_eval})', flush=True)

    # 恢复默认
    config.RETRIEVAL.clear()
    config.RETRIEVAL.update(DEFAULT)
    return out


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--param', default='graph_hop',
                    choices=list(SWEEP.keys()) + list(TWO_D_SWEEP.keys()) + ['all'],
                    help='扫描参数（all=全部单变量+二维组合扫描）')
    ap.add_argument('--values', nargs='*', type=float, default=None,
                    help='取值列表（默认用 SWEEP 定义）')
    ap.add_argument('--subset', type=int, default=0, help='仅前 N 条（0=全部）')
    ap.add_argument('--provider', default='deepseek',
                    choices=sorted(LLM_PROVIDERS),
                    help='LLM provider（模型 ID 由对应环境变量传入）')
    ap.add_argument('--out', default=os.path.join(RESULT_DIR, 'param_sweep.json'))
    args = ap.parse_args()

    if args.param == 'all':
        all_results = {}
        # pipeline 只建一次，全部参数共享 QueryParser 缓存
        r, store = load_pipeline(args.provider)
        qas = json.load(open(EVAL_FILE, encoding='utf-8'))
        for p in SWEEP:
            print(f'\n========== 扫描参数 {p} ==========', flush=True)
            all_results[p] = run_sweep(p, SWEEP[p], r, store, qas, subset=args.subset)
        for name, (params, values) in TWO_D_SWEEP.items():
            print(f'\n========== 扫描参数 {name} ({params[0]} × {params[1]}) ==========',
                  flush=True)
            all_results[name] = run_sweep(params, values, r, store, qas,
                                          subset=args.subset)
        os.makedirs(RESULT_DIR, exist_ok=True)
        with open(args.out, 'w', encoding='utf-8') as f:
            json.dump({'default': DEFAULT, 'sweep': all_results}, f,
                      ensure_ascii=False, indent=1)
        print(f'\n全参数扫描完成，结果写入 {args.out}')
    elif args.param in TWO_D_SWEEP:
        params, values = TWO_D_SWEEP[args.param]
        r, store = load_pipeline(args.provider)
        qas = json.load(open(EVAL_FILE, encoding='utf-8'))
        res = run_sweep(params, values, r, store, qas, subset=args.subset)
        os.makedirs(RESULT_DIR, exist_ok=True)
        with open(args.out, 'w', encoding='utf-8') as f:
            json.dump({'param': args.param, 'params': list(params),
                       'default': DEFAULT, 'results': res},
                      f, ensure_ascii=False, indent=1)
        print(f'\n完成，结果写入 {args.out}')
    else:
        values = args.values or SWEEP[args.param]
        if args.param == 'graph_rerank_lambda':
            values = [float(v) for v in values]
        else:
            values = [int(v) for v in values]
        r, store = load_pipeline(args.provider)
        qas = json.load(open(EVAL_FILE, encoding='utf-8'))
        res = run_sweep(args.param, values, r, store, qas, subset=args.subset)
        os.makedirs(RESULT_DIR, exist_ok=True)
        with open(args.out, 'w', encoding='utf-8') as f:
            json.dump({'param': args.param, 'default': DEFAULT, 'results': res},
                      f, ensure_ascii=False, indent=1)
        print(f'\n完成，结果写入 {args.out}')
