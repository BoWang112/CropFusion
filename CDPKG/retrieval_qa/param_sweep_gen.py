# -*- coding: utf-8 -*-
"""
param_sweep_gen.py —— 生成参数敏感性扫描（子集 + LLM 生成）
================================================================
对 text_top_k / graph_top_k / fewshot_top_k 单变量扫描：
每个取值跑 50 题子集，完整生成（DeepSeek）并算 7 项指标。

- 提示：full 管线（与最终方法一致），仅覆盖被扫描参数
- 单变量：一次只变一个参数，其余默认
- 同一子集内可比（固定前 N 条）

用法：
  .venv-re/Scripts/python.exe -X utf8 param_sweep_gen.py \
      --param text_top_k --values 5 10 15 --subset 50
"""
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from config import NODES_CSV, RELS_CSV, INDEX_DIR, EVAL_FILE, RESULT_DIR
from modules.kg import CropPestKG
from modules.retriever import Embedder, VectorIndex, BM25Index, TextBlockStore, Retriever
from modules.llm import LLMClient
from modules.prompts import (FewShotSelector, BGEEmbedder, load_fewshot_pool,
                             DEFAULT_FEWSHOT_FILE, build_prompt_with_dynamic_fewshot)
from run_eval_resumable import (compute_bleu_n, compute_bleu4, compute_rouge_l,
                                compute_rouge_12, compute_bertscore_batch)

DEFAULT = dict(config.RETRIEVAL)
DEFAULT_FEWSHOT = 3

SWEEP_GEN = {
    'text_top_k': [5, 10, 15, 20, 25],
    'graph_top_k': list(range(1, 11)),
    'fewshot_top_k': list(range(1, 6)),
}


def load_pipeline():
    kg = CropPestKG(NODES_CSV, RELS_CSV)
    store = TextBlockStore.load(os.path.join(INDEX_DIR, 'text_blocks.json'))
    with open(os.path.join(INDEX_DIR, 'vector_doc_ids.json'), encoding='utf-8') as f:
        doc_ids = json.load(f)
    vindex = VectorIndex.load(os.path.join(INDEX_DIR, 'vector.faiss'), doc_ids)
    bindex = BM25Index.load(os.path.join(INDEX_DIR, 'bm25.pkl'))
    llm = LLMClient()
    r = Retriever(kg, store, vindex, bindex, Embedder(), llm_client=llm)
    return r, kg, llm, store


def run_sweep(param, values, r, kg, llm, store, qas):
    """对 param 的每个取值跑完整生成，返回 {value: {metrics}}"""
    pool = load_fewshot_pool(DEFAULT_FEWSHOT_FILE)
    selector = FewShotSelector(pool, BGEEmbedder())
    out = {}
    for val in values:
        # 覆盖参数（其余默认）
        cfg = dict(DEFAULT)
        if param == 'fewshot_top_k':
            fs_top_k = val
            cfg['text_top_k'] = DEFAULT['text_top_k']
            cfg['graph_top_k'] = DEFAULT['graph_top_k']
        else:
            fs_top_k = DEFAULT_FEWSHOT
            cfg[param] = val
        config.RETRIEVAL.clear()
        config.RETRIEVAL.update(cfg)

        acc = {k: [] for k in ('bleu1', 'bleu2', 'bleu4', 'rouge1', 'rouge2',
                               'rouge_l', 'bertscore')}
        refs, hyps = [], []
        for qa in qas:
            q = qa['question']
            task = qa.get('task', '病虫害识别')
            te, ge, det = r.get_evidence(q, variant='full')
            prompt = build_prompt_with_dynamic_fewshot(
                question=q, task_type=task, text_evidence=te, graph_evidence=ge,
                selector=selector, top_k=fs_top_k, mode='concise',
                include_citations=False)
            ans = llm.complete(prompt)
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
        out[str(val)] = {k: statistics.mean(v) for k, v in acc.items()}
        print(f'[{param}={val}]  BLEU-1={out[str(val)]["bleu1"]:.4f}  '
              f'BLEU-2={out[str(val)]["bleu2"]:.4f}  '
              f'BLEU-4={out[str(val)]["bleu4"]:.4f}  '
              f'ROUGE-1={out[str(val)]["rouge1"]:.4f}  '
              f'ROUGE-2={out[str(val)]["rouge2"]:.4f}  '
              f'ROUGE-L={out[str(val)]["rouge_l"]:.4f}  '
              f'BERTScore={out[str(val)]["bertscore"]:.4f}', flush=True)

    config.RETRIEVAL.clear()
    config.RETRIEVAL.update(DEFAULT)
    return out


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--param', default='text_top_k',
                    choices=list(SWEEP_GEN.keys()))
    ap.add_argument('--values', nargs='*', type=int, default=None,
                    help='取值列表（默认用 SWEEP_GEN 定义）')
    ap.add_argument('--subset', type=int, default=50, help='子集大小')
    ap.add_argument('--out', default=os.path.join(RESULT_DIR, 'param_sweep_gen.json'))
    args = ap.parse_args()

    values = args.values or SWEEP_GEN[args.param]
    r, kg, llm, store = load_pipeline()
    qas = json.load(open(EVAL_FILE, encoding='utf-8'))[: args.subset]
    print(f'扫描 {args.param}，取值 {values}，子集 {len(qas)} 题', flush=True)
    res = run_sweep(args.param, values, r, kg, llm, store, qas)
    os.makedirs(RESULT_DIR, exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump({'param': args.param, 'subset': args.subset,
                   'default': {**DEFAULT, 'fewshot_top_k': DEFAULT_FEWSHOT},
                   'results': res}, f, ensure_ascii=False, indent=1)
    print(f'\n完成，结果写入 {args.out}')
