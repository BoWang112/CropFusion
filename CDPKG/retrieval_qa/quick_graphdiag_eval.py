# -*- coding: utf-8 -*-
"""
quick_graphdiag_eval.py —— graph_diag 快速生成评测
====================================================
只跑 EVAL_FILE 中 diag_mode=graph_diag 的多条件综合推理题，
对比 full 与关键消融的生成指标，验证防治边截断修复是否拉开差距。

用法：
  BGE_MODEL=../models/bge-large-zh-v1.5 ../.venv-re/Scripts/python.exe -X utf8 -u quick_graphdiag_eval.py [--n 40] [--variants full,-no-semantic-graph,-no-entity-graph,-no-text-rrf]
"""
import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import INDEX_DIR, EVAL_FILE  # noqa: E402
from modules.kg import CropPestKG  # noqa: E402
from modules.retriever import Embedder, VectorIndex, BM25Index, TextBlockStore, Retriever  # noqa: E402
from modules.prompts import FewShotSelector, BGEEmbedder, build_prompt_with_dynamic_fewshot  # noqa: E402
from modules.prompts import load_fewshot_pool, DEFAULT_FEWSHOT_FILE  # noqa: E402
from modules.llm import LLMClient  # noqa: E402
from config import NODES_CSV, RELS_CSV  # noqa: E402


def compute_bleu_n(ref, hyp, n):
    try:
        from nltk.translate.bleu_score import sentence_bleu
        from nltk.translate.bleu_score import SmoothingFunction
        cc = SmoothingFunction().method4
        return sentence_bleu([list(ref)], list(hyp),
                             weights=tuple(1.0 / n for _ in range(n)),
                             smoothing_function=cc)
    except Exception:
        return 0.0


def compute_bleu4(ref, hyp):
    return compute_bleu_n(ref, hyp, 4)


def compute_rouge_l(ref, hyp):
    try:
        from rouge_score import rouge_scorer
        from rouge_score.rouge_scorer import tokenizers
        class CharTokenizer(tokenizers.Tokenizer):
            def tokenize(self, text):
                return list(text.replace(' ', ''))
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'],
                                          use_stemmer=False, tokenizer=CharTokenizer())
        s = scorer.score(ref, hyp)
        return s['rouge1'].fmeasure, s['rouge2'].fmeasure, s['rougeL'].fmeasure
    except Exception:
        return 0.0, 0.0, 0.0


def compute_bertscore_batch(refs, hyps):
    try:
        import warnings
        import bert_score
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            P, R, F = bert_score.score(hyps, refs, lang='zh',
                                       model_type='bert-base-chinese', verbose=False)
        return [float(f) for f in F]
    except Exception:
        return [0.0] * len(refs)


def load_pipeline():
    kg = CropPestKG(NODES_CSV, RELS_CSV)
    store = TextBlockStore.load(os.path.join(INDEX_DIR, 'text_blocks.json'))
    with open(os.path.join(INDEX_DIR, 'vector_doc_ids.json'), encoding='utf-8') as f:
        doc_ids = json.load(f)
    vindex = VectorIndex.load(os.path.join(INDEX_DIR, 'vector.faiss'), doc_ids)
    bindex = BM25Index.load(os.path.join(INDEX_DIR, 'bm25.pkl'))
    llm = LLMClient()
    retriever = Retriever(kg, store, vindex, bindex, Embedder(), llm_client=llm)
    return retriever, kg, llm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=40)
    ap.add_argument('--variants', default='full,-no-semantic-graph,-no-entity-graph,-no-text-rrf')
    args = ap.parse_args()
    variants = [v.strip() for v in args.variants.split(',')]

    qas = json.load(open(EVAL_FILE, encoding='utf-8'))
    diag = [q for q in qas if (q.get('metadata') or {}).get('diag_mode') == 'graph_diag']
    diag = diag[:args.n]
    print(f'图谱综合诊断题: {len(diag)}（full vs {" vs ".join(variants[1:])}）', flush=True)

    retriever, kg, llm = load_pipeline()
    pool = load_fewshot_pool(DEFAULT_FEWSHOT_FILE)
    selector = FewShotSelector(pool, BGEEmbedder())

    results = {v: [] for v in variants}
    for i, qa in enumerate(diag):
        question = qa['question']
        task = qa.get('task', '病虫害识别')
        for v in variants:
            text_ev, graph_ev, _ = retriever.get_evidence(question, variant=v)
            prompt = build_prompt_with_dynamic_fewshot(
                question=question, task_type=task,
                text_evidence=text_ev, graph_evidence=graph_ev,
                selector=selector, top_k=3, mode='concise',
                include_citations=False,
            )
            answer = llm.complete(prompt)
            results[v].append({'question': question, 'ref': qa.get('answer', ''),
                               'answer': answer, 'gold_pest': qa.get('gold_pest')})
        if (i + 1) % 5 == 0:
            print(f'  已评测 {i+1}/{len(diag)}', flush=True)

    # 指标
    sums = {}
    for v in variants:
        rows = results[v]
        refs = [r['ref'] for r in rows]
        hyps = [r['answer'] for r in rows]
        b1 = sum(compute_bleu_n(ref, hyp, 1) for ref, hyp in zip(refs, hyps)) / len(refs)
        b2 = sum(compute_bleu_n(ref, hyp, 2) for ref, hyp in zip(refs, hyps)) / len(refs)
        b4 = sum(compute_bleu4(ref, hyp) for ref, hyp in zip(refs, hyps)) / len(refs)
        r1s, r2s, rls = [], [], []
        for ref, hyp in zip(refs, hyps):
            r1, r2, rl = compute_rouge_l(ref, hyp)
            r1s.append(r1); r2s.append(r2); rls.append(rl)
        bs = compute_bertscore_batch(refs, hyps)
        m_r1 = sum(r1s) / len(r1s); m_r2 = sum(r2s) / len(r2s); m_rl = sum(rls) / len(rls)
        m_bs = sum(bs) / len(bs)
        gscore = 0.2 * b4 + 0.2 * (m_r1 + m_r2 + m_rl) / 3 + 0.6 * m_bs
        sums[v] = dict(b1=b1, b2=b2, b4=b4, r1=m_r1, r2=m_r2, rl=m_rl, bs=m_bs, g=gscore)
        print(f'\n[{v}] n={len(rows)}')
        print(f'  BLEU-1={b1:.4f}  BLEU-2={b2:.4f}  BLEU-4={b4:.4f}')
        print(f'  ROUGE-1={m_r1:.4f}  ROUGE-2={m_r2:.4f}  ROUGE-L={m_rl:.4f}')
        print(f'  BERTScore={m_bs:.4f}  Gscore={gscore:.4f}', flush=True)

    # 落盘
    os.makedirs(os.path.join(os.path.dirname(EVAL_FILE), 'results'), exist_ok=True)
    with open(os.path.join(os.path.dirname(EVAL_FILE), 'results', 'quick_graphdiag_eval.json'),
              'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=1)

    # 相对 full 的差异
    print('\n===== 相对 full 的差异 =====')
    f = sums['full']
    for v in variants[1:]:
        s = sums[v]
        for k, name in [('b1', 'BLEU-1'), ('b4', 'BLEU-4'), ('r2', 'ROUGE-2'), ('bs', 'BERTScore')]:
            rel = (f[k] - s[k]) / s[k] * 100
            print(f'  {v} vs full  {name}: {rel:+.2f}%')


if __name__ == '__main__':
    main()
