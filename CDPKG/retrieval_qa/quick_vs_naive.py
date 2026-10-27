# -*- coding: utf-8 -*-
"""
quick_vs_naive.py —— 快速验证：图谱综合诊断题的 full vs naive-text
=============================================================
只跑 EVAL_FILE 中 diag_mode=graph_diag 的多条件综合推理题，
对比 full 与 naive-text 两个变体的生成指标，判断做法 A 能否拉开差距。

用法：
  ../.venv-re/Scripts/python.exe -X utf8 quick_vs_naive.py [--n 55]
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import INDEX_DIR, EVAL_FILE  # noqa: E402
from modules.kg import CropPestKG  # noqa: E402
from modules.retriever import Embedder, VectorIndex, BM25Index, TextBlockStore, Retriever  # noqa: E402
from modules.prompts import FewShotSelector, BGEEmbedder, build_prompt_with_dynamic_fewshot  # noqa: E402
from modules.prompts import load_fewshot_pool, DEFAULT_FEWSHOT_FILE  # noqa: E402
from modules.llm import LLMClient  # noqa: E402
from config import NODES_CSV, RELS_CSV  # noqa: E402


def compute_bleu4(ref, hyp):
    try:
        from nltk.translate.bleu_score import sentence_bleu
        from nltk.translate.bleu_score import SmoothingFunction
        cc = SmoothingFunction().method4
        return sentence_bleu([list(ref)], list(hyp), weights=(0.25, 0.25, 0.25, 0.25),
                             smoothing_function=cc)
    except Exception:
        return 0.0


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


def compute_rouge_l(ref, hyp):
    """ROUGE-1/2/L：字符级 tokenizer（中文无空格，默认 tokenizer 会把整句当一个 token，
    导致 ROUGE 恒为 0。字符级切分是中文评测的常规做法，与 BLEU 的字符级对齐。）"""
    try:
        from rouge_score import rouge_scorer
        from rouge_score.rouge_scorer import tokenizers
        class CharTokenizer(tokenizers.Tokenizer):
            def tokenize(self, text):
                return list(text.replace(' ', ''))
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False,
                                          tokenizer=CharTokenizer())
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
    return retriever, kg, store, llm


def main():
    n_limit = int(sys.argv[sys.argv.index('--n') + 1]) if '--n' in sys.argv else 0
    qas = json.load(open(EVAL_FILE, encoding='utf-8'))
    diag = [q for q in qas if (q.get('metadata') or {}).get('diag_mode') == 'graph_diag']
    print(f'图谱综合诊断题: {len(diag)}')
    if n_limit:
        diag = diag[:n_limit]
    print(f'本次评测: {len(diag)} 条（full vs naive-text）\n')

    retriever, kg, store, llm = load_pipeline()
    pool = load_fewshot_pool(DEFAULT_FEWSHOT_FILE)
    selector = FewShotSelector(pool, BGEEmbedder())

    variants = ['full', 'naive-text']
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
                               'answer': answer, 'gold_pest': qa.get('gold_pest'),
                               'missing_sites': (qa.get('metadata') or {}).get('missing_sites', []),
                               'control_nodes': (qa.get('metadata') or {}).get('control_nodes', [])})
        if (i + 1) % 5 == 0:
            print(f'  已评测 {i+1}/{len(diag)}', flush=True)

    # 指标
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
        m_r1 = sum(r1s) / len(r1s)
        m_r2 = sum(r2s) / len(r2s)
        m_rl = sum(rls) / len(rls)
        m_bs = sum(bs) / len(bs)
        gscore = 0.2 * b4 + 0.2 * (m_r1 + m_r2 + m_rl) / 3 + 0.6 * m_bs
        # 保存逐条结果
        for i, row in enumerate(rows):
            row.update({'bleu1': compute_bleu_n(refs[i], hyps[i], 1),
                        'bleu2': compute_bleu_n(refs[i], hyps[i], 2),
                        'bleu4': compute_bleu4(refs[i], hyps[i]),
                        'rouge1': r1s[i], 'rouge2': r2s[i], 'rouge_l': rls[i],
                        'bertscore': bs[i]})
        print(f'\n[{v}] n={len(rows)}')
        print(f'  BLEU-1={b1:.4f}  BLEU-2={b2:.4f}  BLEU-4={b4:.4f}')
        print(f'  ROUGE-1={m_r1:.4f}  ROUGE-2={m_r2:.4f}  ROUGE-L={m_rl:.4f}')
        print(f'  BERTScore={m_bs:.4f}  Gscore={gscore:.4f}')

    # 落盘逐条结果
    os.makedirs(os.path.join(os.path.dirname(EVAL_FILE), 'results'), exist_ok=True)
    with open(os.path.join(os.path.dirname(EVAL_FILE), 'results', 'quick_vs_naive.json'),
              'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=1)

    # 差异
    f = results['full']; n = results['naive-text']
    for name, key in [('BLEU-1', 'b1'), ('BLEU-4', 'b4'), ('BERTScore', 'bs')]:
        pass
    # 逐条差异（防 copy）
    full_win = naive_win = tie = 0
    for fr, nr in zip(f, n):
        fb = compute_bleu4(fr['ref'], fr['answer'])
        nb = compute_bleu4(nr['ref'], nr['answer'])
        if fb > nb + 0.005: full_win += 1
        elif nb > fb + 0.005: naive_win += 1
        else: tie += 1
    print(f'\n逐条 BLEU-4 对比: full 领先 {full_win}, naive 领先 {naive_win}, 持平 {tie}')


if __name__ == '__main__':
    main()
