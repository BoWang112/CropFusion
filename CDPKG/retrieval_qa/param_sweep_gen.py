# -*- coding: utf-8 -*-
"""
param_sweep_gen.py —— 生成参数敏感性扫描（子集 + LLM 生成）
================================================================
对 graph_rerank_lambda / graph_hop / semantic_top_k / entity_top_k /
text_top_k / graph_top_k / fewshot_top_k 单变量扫描：
每个取值跑 50 题子集，完整生成（DeepSeek）并算 7 项指标。

- 提示：full 管线（与最终方法一致），仅覆盖被扫描参数
- 任务分类：统一自动分类，不读取测试集真实任务标签
- 生成：三次独立生成 + 核心结论一致性投票 + 证据覆盖度选择
- 单变量：一次只变一个参数，其余默认
- 同一子集内可比（固定前 N 条）

用法：
  .venv-re/Scripts/python.exe -X utf8 param_sweep_gen.py \
      --param text_top_k --values 5 10 15 --subset 50
"""
import sys
import os
import json
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from config import NODES_CSV, RELS_CSV, INDEX_DIR, EVAL_FILE, RESULT_DIR
from config import get_llm_config, LLM_PROVIDERS
from modules.kg import CropPestKG
from modules.retriever import Embedder, VectorIndex, BM25Index, TextBlockStore, Retriever
from modules.llm import LLMClient
from modules.task_classifier import classify_task
from modules.prompts import (FewShotSelector, BGEEmbedder, load_fewshot_pool,
                             DEFAULT_FEWSHOT_FILE, build_prompt_with_dynamic_fewshot)
from run_eval_resumable import (compute_bleu_n, compute_bleu4, compute_rouge_l,
                                compute_rouge_12, compute_bertscore_batch)

DEFAULT = dict(config.RETRIEVAL)
DEFAULT_FEWSHOT = DEFAULT.get('fewshot_top_k', 3)

SWEEP_GEN = {
    'graph_rerank_lambda': [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    'graph_hop': [1, 2, 3, 4],
    'semantic_top_k': [5, 10, 20, 30, 40],
    'entity_top_k': [5, 10, 20, 30, 40],
    'text_top_k': [3, 5, 10, 15, 20, 25],
    'graph_top_k': list(range(1, 11)),
    'fewshot_top_k': [0, 1, 2, 3, 4, 5],
}

# 二维组合扫描：图扩展跳数 × 候选锚点数量
TWO_D_SWEEP_GEN = {
    'hop_x_anchor': (
        ('graph_hop', 'semantic_top_k'),
        [(h, a) for h in (1, 2, 3, 4) for a in (5, 10, 20, 30, 40)],
    ),
}


def load_pipeline(provider=None):
    kg = CropPestKG(NODES_CSV, RELS_CSV)
    store = TextBlockStore.load(os.path.join(INDEX_DIR, 'text_blocks.json'))
    with open(os.path.join(INDEX_DIR, 'vector_doc_ids.json'), encoding='utf-8') as f:
        doc_ids = json.load(f)
    vindex = VectorIndex.load(os.path.join(INDEX_DIR, 'vector.faiss'), doc_ids)
    bindex = BM25Index.load(os.path.join(INDEX_DIR, 'bm25.pkl'))
    llm = LLMClient(get_llm_config(provider))
    r = Retriever(kg, store, vindex, bindex, Embedder(), llm_client=llm)
    return r, kg, llm, store


def _param_label(param, val):
    if isinstance(param, (tuple, list)):
        return ','.join(f'{p}={v}' for p, v in zip(param, val))
    return f'{param}={val}'


def _value_key(param, val):
    if isinstance(param, (tuple, list)):
        return ','.join(str(v) for v in val)
    return str(val)


def run_sweep(param, values, r, kg, llm, store, qas):
    """对单参数或二维参数组合跑完整生成。

    param 为 tuple 时同时设置多个参数（二维组合），其余参数保持默认。
    """
    pool = load_fewshot_pool(DEFAULT_FEWSHOT_FILE)
    selector = FewShotSelector(pool, BGEEmbedder())
    out = {}
    for val in values:
        # 覆盖参数（其余默认）
        cfg = dict(DEFAULT)
        fs_top_k = DEFAULT_FEWSHOT
        if isinstance(param, (tuple, list)):
            for p, v in zip(param, val):
                cfg[p] = v
                if p == 'fewshot_top_k':
                    fs_top_k = v
        elif param == 'fewshot_top_k':
            fs_top_k = val
            cfg['fewshot_top_k'] = val
        else:
            cfg[param] = val
        key = _value_key(param, val)
        label = _param_label(param, val)
        config.RETRIEVAL.clear()
        config.RETRIEVAL.update(cfg)

        acc = {k: [] for k in ('bleu1', 'bleu2', 'bleu4', 'rouge1', 'rouge2',
                               'rouge_l', 'bertscore')}
        refs, hyps = [], []
        gen_methods = Counter()
        gen_errors = 0
        for qa in qas:
            q = qa['question']
            # 自动任务分类，不读取 qa['task'] 真实标签
            task, _task_method = classify_task(q, llm)
            te, ge, det = r.get_evidence(q, variant='full')
            prompt = build_prompt_with_dynamic_fewshot(
                question=q, task_type=task, text_evidence=te, graph_evidence=ge,
                selector=selector, top_k=fs_top_k, mode='concise',
                include_citations=False)
            # 三次独立生成 + 核心结论一致性投票 + 证据覆盖度选择
            generation = llm.complete_with_consistency_details(prompt)
            ans = generation.get('answer', '')
            gen_methods[generation.get('method')] += 1
            if generation.get('error'):
                gen_errors += 1
            ref = qa.get('answer', '')
            refs.append(ref)
            hyps.append(ans)
            acc['bleu1'].append(compute_bleu_n(ref, ans, 1))
            acc['bleu2'].append(compute_bleu_n(ref, ans, 2))
            acc['bleu4'].append(compute_bleu4(ref, ans))
            rl = compute_rouge_l(ref, ans)
            r1, r2 = compute_rouge_12(ref, ans)
            acc['rouge_l'].append(rl)
            acc['rouge1'].append(r1)
            acc['rouge2'].append(r2)
        bf = compute_bertscore_batch(refs, hyps)
        acc['bertscore'] = bf

        import statistics
        out[key] = {k: statistics.mean(v) for k, v in acc.items()}
        out[key]['generation_methods'] = dict(gen_methods)
        out[key]['generation_errors'] = gen_errors
        out[key]['provider'] = llm.cfg.get('provider')
        out[key]['model'] = llm.cfg.get('model')
        out[key]['generation_params'] = {
            'temperature': llm.cfg.get('temperature', 0.2),
            'top_p': llm.cfg.get('top_p', 0.9),
            'max_tokens': llm.cfg.get('max_tokens', 2048),
            'num_samples': llm.cfg.get('num_samples', 3),
        }
        print(f'[{label}]  BLEU-1={out[key]["bleu1"]:.4f}  '
              f'BLEU-2={out[key]["bleu2"]:.4f}  '
              f'BLEU-4={out[key]["bleu4"]:.4f}  '
              f'ROUGE-1={out[key]["rouge1"]:.4f}  '
              f'ROUGE-2={out[key]["rouge2"]:.4f}  '
              f'ROUGE-L={out[key]["rouge_l"]:.4f}  '
              f'BERTScore={out[key]["bertscore"]:.4f}', flush=True)

    config.RETRIEVAL.clear()
    config.RETRIEVAL.update(DEFAULT)
    return out


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--param', default='text_top_k',
                    choices=list(SWEEP_GEN.keys()) + list(TWO_D_SWEEP_GEN.keys()))
    ap.add_argument('--values', nargs='*', type=float, default=None,
                    help='取值列表（默认用 SWEEP_GEN；二维组合使用内置取值）')
    ap.add_argument('--subset', type=int, default=50, help='子集大小')
    ap.add_argument('--provider', default='deepseek',
                    choices=sorted(LLM_PROVIDERS),
                    help='LLM provider（模型 ID 由对应环境变量传入）')
    ap.add_argument('--out', default=os.path.join(RESULT_DIR, 'param_sweep_gen.json'))
    args = ap.parse_args()

    r, kg, llm, store = load_pipeline(args.provider)
    qas = json.load(open(EVAL_FILE, encoding='utf-8'))[: args.subset]

    if args.param in TWO_D_SWEEP_GEN:
        params, values = TWO_D_SWEEP_GEN[args.param]
        print(f'二维扫描 {params[0]} × {params[1]}，组合数 {len(values)}，'
              f'子集 {len(qas)} 题', flush=True)
        res = run_sweep(params, values, r, kg, llm, store, qas)
        os.makedirs(RESULT_DIR, exist_ok=True)
        with open(args.out, 'w', encoding='utf-8') as f:
            json.dump({'param': args.param, 'params': list(params),
                       'subset': args.subset,
                       'default': {**DEFAULT, 'fewshot_top_k': DEFAULT_FEWSHOT},
                       'results': res}, f, ensure_ascii=False, indent=1)
        print(f'\n完成，结果写入 {args.out}')
    else:
        values = args.values or SWEEP_GEN[args.param]
        if args.param == 'graph_rerank_lambda':
            values = [float(v) for v in values]
        else:
            values = [int(v) for v in values]
        print(f'扫描 {args.param}，取值 {values}，子集 {len(qas)} 题', flush=True)
        res = run_sweep(args.param, values, r, kg, llm, store, qas)
        os.makedirs(RESULT_DIR, exist_ok=True)
        with open(args.out, 'w', encoding='utf-8') as f:
            json.dump({'param': args.param, 'subset': args.subset,
                       'default': {**DEFAULT, 'fewshot_top_k': DEFAULT_FEWSHOT},
                       'results': res}, f, ensure_ascii=False, indent=1)
        print(f'\n完成，结果写入 {args.out}')
