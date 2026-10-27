# -*- coding: utf-8 -*-
"""
run_eval_zeroshot.py —— DeepSeek 零样本直接回答基线（无检索）
================================================================
与 run_eval_resumable.py 的 full 变体对齐，唯一区别是提示词不含任何检索证据
（无文本证据、无图谱证据、无少样本），验证"检索系统相对裸 LLM 的贡献"。

- 提示词：任务模板 + 输出规则（concise，与 Ours 保持一致），问题直接提问
- 任务类型：由 modules/task_classifier.py 从问题文本自动判断，
  不读取 qa['task']（测试集真实任务标签）
- 指标：与 run_eval_resumable.py 完全一致（BLEU-1/2/4、ROUGE-1/2/L、BERTScore）
- 进度文件：data/results/qa_progress_zeroshot_v2.json（新版缓存，避免旧结果跳过）

用法：
  .venv-re/Scripts/python.exe -X utf8 run_eval_zeroshot.py [--subset N]
"""
import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import EVAL_FILE, RESULT_DIR  # noqa: E402
from config import get_llm_config, LLM_PROVIDERS  # noqa: E402
from modules.prompts import TASK_INSTRUCTIONS_CONCISE, get_output_rules  # noqa: E402
from modules.task_classifier import classify_task  # noqa: E402
from modules.llm import LLMClient  # noqa: E402

CHUNK = 20
CACHE_VERSION = 'v2'
PROVIDER = 'deepseek'
PROGRESS = os.path.join(RESULT_DIR, f'qa_progress_zeroshot_{CACHE_VERSION}.json')
SUMMARY = os.path.join(RESULT_DIR, f'qa_summary_zeroshot_{CACHE_VERSION}.json')


# ---------- 指标（与 run_eval_resumable.py 保持一致） ----------
def compute_bleu4(ref, hyp):
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        ref_tok = list(ref); hyp_tok = list(hyp)
        cc = SmoothingFunction().method4
        return sentence_bleu([ref_tok], hyp_tok, weights=(0.25, 0.25, 0.25, 0.25),
                             smoothing_function=cc)
    except Exception:
        return 0.0


def compute_bleu_n(ref, hyp, n):
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        ref_tok = list(ref); hyp_tok = list(hyp)
        cc = SmoothingFunction().method4
        weights = tuple(1.0 / n for _ in range(n))
        return sentence_bleu([ref_tok], hyp_tok, weights=weights, smoothing_function=cc)
    except Exception:
        return 0.0


def compute_rouge_l(ref, hyp):
    try:
        from rouge_score import rouge_scorer
        from rouge_score.rouge_scorer import tokenizers
        class CharTokenizer(tokenizers.Tokenizer):
            def tokenize(self, text):
                return list(text.replace(' ', ''))
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False, tokenizer=CharTokenizer())
        s = scorer.score(ref, hyp)
        return s['rougeL'].fmeasure
    except Exception:
        return 0.0


def compute_rouge_12(ref, hyp):
    try:
        from rouge_score import rouge_scorer
        from rouge_score.rouge_scorer import tokenizers
        class CharTokenizer(tokenizers.Tokenizer):
            def tokenize(self, text):
                return list(text.replace(' ', ''))
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2'], use_stemmer=False,
                                          tokenizer=CharTokenizer())
        s = scorer.score(ref, hyp)
        return s['rouge1'].fmeasure, s['rouge2'].fmeasure
    except Exception:
        return 0.0, 0.0


def compute_bertscore_batch(refs, hyps):
    try:
        import warnings
        import bert_score
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            P, R, F = bert_score.score(hyps, refs, lang='zh', model_type='bert-base-chinese',
                                       verbose=False)
        return [float(f) for f in F]
    except Exception:
        return [0.0] * len(refs)


def build_zeroshot_prompt(question, task):
    """零样本提示：任务模板（concise）+ 输出规则 + 问题。无任何检索证据。"""
    parts = []
    parts.append("你是一个农业病虫害问答助手，请直接回答用户问题。")
    if task in TASK_INSTRUCTIONS_CONCISE:
        parts.append(TASK_INSTRUCTIONS_CONCISE[task].strip())
    parts.append(get_output_rules('concise', include_citations=False).strip())
    parts.append(f"用户问题：{question}")
    return "\n\n".join(parts)


def main():
    os.makedirs(RESULT_DIR, exist_ok=True)
    done = {}
    if os.path.exists(PROGRESS):
        with open(PROGRESS, encoding='utf-8') as f:
            done = json.load(f)
        print(f'已恢复进度：{len(done)} 条完成')
    qas = json.load(open(EVAL_FILE, encoding='utf-8'))
    if SUBSET and SUBSET > 0:
        qas = qas[:SUBSET]
    todo = [qa for qa in qas if qa['question'] not in done]
    print(f'总 {len(qas)} 条，待评测 {len(todo)} 条')

    llm = LLMClient(get_llm_config(PROVIDER))
    t_start = time.time()
    for i, qa in enumerate(todo):
        question = qa['question']
        # 任务类型由问题文本自动判断，不读取 qa['task'] 真实标签
        task, task_method = classify_task(question, llm)
        prompt = build_zeroshot_prompt(question, task)
        # 三次独立生成 + 核心结论一致性投票 + 证据覆盖度选择
        # （零样本无检索证据，覆盖度退化为 0，仍保留完整选择过程）
        generation = llm.complete_with_consistency_details(prompt)
        answer = generation.get('answer', '')
        if generation.get('method') == 'failed':
            print(f'  ⚠ LLM 生成失败，本条保留待重试：{question[:30]}...',
                  flush=True)
            continue
        ref = qa.get('answer', '')
        b1 = compute_bleu_n(ref, answer, 1)
        b2 = compute_bleu_n(ref, answer, 2)
        b4 = compute_bleu4(ref, answer)
        rl = compute_rouge_l(ref, answer)
        r1, r2 = compute_rouge_12(ref, answer)
        done[question] = {
            'task': task, 'task_method': task_method, 'ref': ref, 'answer': answer,
            'candidates': generation.get('candidates', []),
            'votes': generation.get('votes', []),
            'coverage': generation.get('coverage', []),
            'agreement': generation.get('agreement', []),
            'core_conclusions': generation.get('core_conclusions', []),
            'valid_indices': generation.get('valid_indices', []),
            'selected_index': generation.get('selected_index'),
            'generation_method': generation.get('method'),
            'generation_error': generation.get('error'),
            'provider': generation.get('provider'),
            'model': generation.get('model'),
            'temperature': generation.get('temperature'),
            'top_p': generation.get('top_p'),
            'max_tokens': generation.get('max_tokens'),
            'bleu1': b1, 'bleu2': b2, 'bleu': b4, 'rouge': rl,
            'rouge1': r1, 'rouge2': r2,
            'bertscore': None, 'gold_pest': qa.get('gold_pest'),
        }
        if (i + 1) % CHUNK == 0:
            with open(PROGRESS, 'w', encoding='utf-8') as f:
                json.dump(done, f, ensure_ascii=False)
            elapsed = time.time() - t_start
            per = elapsed / (i + 1)
            remaining = per * (len(todo) - i - 1) / 60
            print(f'  已评测 {len(done)}/{len(qas)}（本次 {i+1}/{len(todo)}），'
                  f'预计剩余 {remaining:.0f} 分钟', flush=True)

    with open(PROGRESS, 'w', encoding='utf-8') as f:
        json.dump(done, f, ensure_ascii=False)

    if not done:
        print('没有成功完成的条目，未写入汇总。'
              '请检查 API Key / 模型 ID 配置后重试。', flush=True)
        return

    print('批量计算 BERTScore...', flush=True)
    items = list(done.items())
    refs = [v['ref'] for _, v in items]
    answers = [v['answer'] for _, v in items]
    bf_scores = compute_bertscore_batch(refs, answers)
    for (_q, v), bf in zip(items, bf_scores):
        v['bertscore'] = bf
    with open(PROGRESS, 'w', encoding='utf-8') as f:
        json.dump(done, f, ensure_ascii=False)

    import statistics
    m = lambda k: statistics.mean([v.get(k, 0.0) for _, v in items])
    summary = {
        'n': len(items), 'variant': 'zeroshot',
        'bleu1': m('bleu1'), 'bleu2': m('bleu2'), 'bleu4': m('bleu'),
        'rouge1': m('rouge1'), 'rouge2': m('rouge2'), 'rouge_l': m('rouge'),
        'bertscore': m('bertscore'),
    }
    with open(SUMMARY, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print('=' * 60)
    print(f'零样本评测完成 (n={summary["n"]})')
    for k in ('bleu1', 'bleu2', 'bleu4', 'rouge1', 'rouge2', 'rouge_l', 'bertscore'):
        print(f'  {k:<9} = {summary[k]:.4f}')


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--subset', type=int, default=0, help='仅评测前 N 条（0=全部）')
    ap.add_argument('--provider', default='deepseek',
                    choices=sorted(LLM_PROVIDERS),
                    help='LLM provider（模型 ID 由对应环境变量传入）')
    args = ap.parse_args()
    SUBSET = args.subset
    PROVIDER = args.provider
    main()
