# -*- coding: utf-8 -*-
"""
3_evaluate.py —— 评测脚本
====================================
在自动生成的评测基准集（data/qa_test.json）上评测：

- 检索性能：Precision@k / Recall@k / F1@k / NDCG@k（k 由 --k 指定，默认 3）
- 问答性能：BLEU-4 / ROUGE-L / BERTScore-F1

评测任务可配置：--metrics retrieval / qa / both；--subset N 控制规模。
"""
import os
import sys
import json
import math
import argparse
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import INDEX_DIR, EVAL_FILE, RESULT_DIR  # noqa: E402
from modules.kg import CropPestKG  # noqa: E402
from modules.retriever import (  # noqa: E402
    Embedder, VectorIndex, BM25Index, TextBlockStore, Retriever,
)
from modules.llm import LLMClient  # noqa: E402

from config import NODES_CSV, RELS_CSV  # noqa: E402


def load_pipeline():
    kg = CropPestKG(NODES_CSV, RELS_CSV)
    store = TextBlockStore.load(os.path.join(INDEX_DIR, 'text_blocks.json'))
    with open(os.path.join(INDEX_DIR, 'vector_doc_ids.json'), encoding='utf-8') as f:
        doc_ids = json.load(f)
    vindex = VectorIndex.load(os.path.join(INDEX_DIR, 'vector.faiss'), doc_ids)
    bindex = BM25Index.load(os.path.join(INDEX_DIR, 'bm25.pkl'))
    embedder = Embedder()
    llm = LLMClient()
    retriever = Retriever(kg, store, vindex, bindex, embedder, llm_client=llm)
    return retriever, kg, store


def ndcg(relevance, k=5):
    """NDCG@k：relevance 为按序排列的 0/1 列表，标准 NDCG 公式（log2 折扣）"""
    rel = relevance[:k]
    dcg = sum(r / math.log2(i + 2) for i, r in enumerate(rel))
    ideal = sorted(relevance, reverse=True)[:k]
    idcg = sum(r / math.log2(i + 2) for i, r in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def eval_retrieval(retriever, kg, store, qas, k=5):
    """检索性能：对每个 QA，标准证据 = 与答案 pest 相关的文本块/图证据。"""
    pest_blocks = {}
    for b in store.blocks:
        if b.get('entity_name'):
            pest_blocks.setdefault(b['entity_name'], b['id'])
    stats = defaultdict(float)
    n = 0
    for qa in qas:
        # 兼容新旧格式：优先 gold_pest（新评测集），回退 pest
        pest = qa.get('gold_pest') or qa.get('pest')
        if not pest:
            continue
        n += 1
        # 标准证据 id 集合
        std_ids = {pest_blocks[pest]} if pest in pest_blocks else set()
        # 取检索 Top-k（文本层融合结果）
        fused = retriever.text_fusion(qa['question'], top_k=20)
        top_ids = [did for did, _ in fused[:k]]
        hits = [1 if did in std_ids else 0 for did in top_ids]
        # Precision@k
        stats['p'] += sum(hits) / k
        stats['r'] += sum(hits) / max(1, len(std_ids))
        stats['f1'] += (2 * sum(hits) / (k + len(std_ids))) if std_ids else 0
        stats['ndcg'] += ndcg(hits, k)
        # 证据排序质量：gold 文本块是否排第 1 / 前 3 / 前 5
        if std_ids:
            gold_id = next(iter(std_ids))
            if gold_id in top_ids:
                rank = top_ids.index(gold_id) + 1
                stats['top1'] += (rank == 1)
                stats['top3'] += (rank <= 3)
                stats['top5'] += (rank <= 5)
                stats['rank_sum'] += rank
                stats['rank_cnt'] += 1
            else:
                stats['rank_cnt'] += 1
                stats['rank_sum'] += (k + 1)  # 未召回记作超出 Top-k
    m = max(1, n)
    out = {kk: vv / m for kk, vv in stats.items() if kk not in ('rank_sum', 'rank_cnt')}
    if stats['rank_cnt'] > 0:
        out['avg_rank'] = stats['rank_sum'] / stats['rank_cnt']
    return out, n


# ---------- 问答指标 ----------
def compute_bleu4(ref, hyp):
    try:
        from nltk.translate.bleu_score import sentence_bleu
        from nltk.translate.bleu_score import SmoothingFunction
        ref_tok = list(ref)
        hyp_tok = list(hyp)
        cc = SmoothingFunction().method4
        return sentence_bleu([ref_tok], hyp_tok, weights=(0.25, 0.25, 0.25, 0.25),
                             smoothing_function=cc)
    except Exception:
        return 0.0


def compute_rouge_l(ref, hyp):
    try:
        from rouge_score import rouge_scorer
        from rouge_score.rouge_scorer import tokenizers
        class CharTokenizer(tokenizers.Tokenizer):
            def tokenize(self, text):
                return list(text.replace(' ', ''))
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False,
                                          tokenizer=CharTokenizer())
        s = scorer.score(ref, hyp)
        return s['rougeL'].fmeasure
    except Exception:
        return 0.0


def compute_bertscore_batch(refs, hyps):
    """批量 BERTScore（单次加载模型，抑制警告，大幅提速）"""
    try:
        import warnings
        import bert_score
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            P, R, F = bert_score.score(hyps, refs, lang='zh',
                                       model_type='bert-base-chinese',
                                       verbose=False)
        return [float(f) for f in F]
    except Exception:
        return [0.0] * len(refs)


def eval_qa(retriever, kg, store, llm, qas, subset=None, save=True, concise=False):
    """端到端问答：检索→提示→生成→BLEU/ROUGE/BERTScore
    concise=True 用精简提示模板（模型只输出结论，与参考答案可比）
    性能优化：BLEU/ROUGE 逐条算，BERTScore 批量算（单次模型加载）。"""
    if subset:
        qas = qas[:subset]
    rows = []
    agg = defaultdict(float)
    n = 0
    for i, qa in enumerate(qas):
        question = qa['question']
        text_ev, graph_ev, detail = retriever.get_evidence(question)
        fewshots = select_fewshot(question, qa.get('task', '病虫害识别'), top_k=3)
        prompt = build_evidence_prompt(question, text_ev, graph_ev,
                                       learning_goal=qa.get('task', '病虫害识别'),
                                       concise=concise)
        full = f"{format_fewshot(fewshots)}\n\n{prompt}"
        # 评测用单次生成（自一致性主要面向产品稳定性，评测指标对单次生成更可复现）
        answer = llm.complete(full)
        ref = qa.get('answer', '')
        b4 = compute_bleu4(ref, answer)
        rl = compute_rouge_l(ref, answer)
        agg['bleu'] += b4
        agg['rouge'] += rl
        n += 1
        rows.append({'question': question, 'task': qa.get('task'), 'ref': ref,
                     'answer': answer, 'bleu': b4, 'rouge': rl, 'bertscore': None})
        if (i + 1) % 20 == 0:
            print(f"  已评测 {i + 1}/{len(qas)}")

    # 批量计算 BERTScore（单次模型加载）
    print("  批量计算 BERTScore...")
    bf_scores = compute_bertscore_batch([r['ref'] for r in rows],
                                        [r['answer'] for r in rows])
    for r, bf in zip(rows, bf_scores):
        r['bertscore'] = bf
        agg['bertscore'] += bf

    m = max(1, n)
    res = {'bleu4': agg['bleu'] / m, 'rouge_l': agg['rouge'] / m,
           'bertscore': agg['bertscore'] / m, 'n': n}
    if save:
        os.makedirs(RESULT_DIR, exist_ok=True)
        with open(os.path.join(RESULT_DIR, 'qa_results.json'), 'w', encoding='utf-8') as f:
            json.dump({'metrics': res, 'rows': rows}, f, ensure_ascii=False, indent=1)
    return res, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--metrics', default='both', choices=['retrieval', 'qa', 'both'])
    ap.add_argument('--subset', type=int, default=0, help='仅评测前 N 条（0=全部）')
    ap.add_argument('--k', type=int, default=3, help='检索指标 @k（默认 3）')
    ap.add_argument('--concise', action='store_true',
                    help='精简回答模式：模型只输出结论与要点（与参考答案可比）')
    args = ap.parse_args()

    print('加载检索管线...')
    retriever, kg, store = load_pipeline()
    qas = json.load(open(EVAL_FILE, encoding='utf-8'))
    print(f'评测集：{len(qas)} 组\n')

    if args.metrics in ('retrieval', 'both'):
        print('=' * 60)
        print('检索性能评测')
        res, n = eval_retrieval(retriever, kg, store, qas, k=args.k)
        print(f'  (n={n})')
        kk_ = args.k
        label_map = {'p': f'P@{kk_}', 'r': f'R@{kk_}', 'f1': f'F1@{kk_}', 'ndcg': f'NDCG@{kk_}',
                     'top1': 'Evidence-Top1', 'top3': 'Evidence-Top3',
                     'top5': 'Evidence-Top5', 'avg_rank': 'Avg-Rank'}
        for kk, vv in res.items():
            lbl = label_map.get(kk, kk.upper())
            print(f'  {lbl:>15} = {vv:.4f}')

    if args.metrics in ('qa', 'both'):
        print('=' * 60)
        mode = '精简回答模式' if args.concise else '完整推理模式'
        print(f'问答性能评测（{mode}）')
        llm = LLMClient()
        if llm.mock_mode:
            print('  ⚠ 当前为模拟 LLM（未配置 API Key），结果仅用于链路验证。')
        subset = args.subset or None
        res, rows = eval_qa(retriever, kg, store, llm, qas, subset=subset,
                            concise=args.concise)
        print(f'  (n={res["n"]})')
        print(f'  BLEU-4     = {res["bleu4"]:.4f}')
        print(f'  ROUGE-L    = {res["rouge_l"]:.4f}')
        print(f'  BERTScore  = {res["bertscore"]:.4f}')

    print('=' * 60)
    print('评测完成。')


if __name__ == '__main__':
    main()
