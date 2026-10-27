# -*- coding: utf-8 -*-
"""
run_eval_resumable.py —— 可续跑的端到端评测
=============================================
分块处理 480 条评测，每块完成后落盘，支持断点续跑：
- 已完成条目的结果保存在 data/results/qa_progress_v2_<variant>.json
  （PIPELINE_VERSION='v2'，与旧版缓存隔离，避免旧流程结果被直接跳过）
- 中途中断后重新运行会跳过已完成条目
- BERTScore 全部收集后批量计算

评测流程：
- 任务类型由统一的 modules/task_classifier.py 从问题文本自动分类
  （不读取 qa['task'] 真实标签）；无 API Key 时降级为规则分类。
- 生成采用三次独立采样 + 核心结论一致性投票 + 证据覆盖度选择
  （modules/llm.py）。
- 每条结果保存 task/task_method、三条候选回答、投票、覆盖度、
  核心结论、最终选中下标与最终答案，便于审计。

适配 prompts.py v2（FewShotSelector + build_prompt_with_dynamic_fewshot）。

用法：
  .venv-re/Scripts/python.exe -X utf8 run_eval_resumable.py
"""
import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import INDEX_DIR, EVAL_FILE, RETRIEVAL  # noqa: E402
from modules.kg import CropPestKG  # noqa: E402
from modules.retriever import Embedder, VectorIndex, BM25Index, TextBlockStore, Retriever  # noqa: E402
from modules.prompts import FewShotSelector, BGEEmbedder, build_prompt_with_dynamic_fewshot  # noqa: E402
from modules.prompts import load_fewshot_pool, DEFAULT_FEWSHOT_FILE  # noqa: E402
from modules.task_classifier import classify_task  # noqa: E402
from modules.llm import LLMClient  # noqa: E402
from config import NODES_CSV, RELS_CSV, RESULT_DIR  # noqa: E402
from config import get_llm_config, LLM_PROVIDERS  # noqa: E402
from config import EVAL_VARIANTS, EVALUATION_PIPELINE_VERSION  # noqa: E402

VARIANT = 'full'   # 由 --variant 覆盖（进度/结果文件按变体隔离）
SUBSET = 0         # 由 --subset 覆盖（快速验证用）
PROVIDER = 'deepseek'   # 由 --provider 覆盖（shennong = 神农大模型）
DATASET = 'local'   # 由 --dataset 覆盖（local=480题, global=全局聚合题）
CHUNK = 20   # 每 20 条落盘一次
# 新版评测流程版本号：进度/汇总文件带 v2 前缀，避免旧缓存（旧提示词/单次生成/
# 真实任务标签）导致新流程被直接跳过。
PIPELINE_VERSION = EVALUATION_PIPELINE_VERSION


def progress_file():
    # 不同 LLM provider 用不同文件名，避免覆盖（如 deepseek 与 shennong 结果并存）
    tag = '' if PROVIDER == 'deepseek' else f'_{PROVIDER}'
    dtag = '' if DATASET == 'local' else f'_{DATASET}'
    return os.path.join(
        RESULT_DIR, f'qa_progress_{PIPELINE_VERSION}_{VARIANT}{dtag}{tag}.json')


def summary_file():
    tag = '' if PROVIDER == 'deepseek' else f'_{PROVIDER}'
    dtag = '' if DATASET == 'local' else f'_{DATASET}'
    return os.path.join(
        RESULT_DIR, f'qa_summary_{PIPELINE_VERSION}_{VARIANT}{dtag}{tag}.json')


# ---------- 问答指标（与 3_evaluate.py 保持一致，独立内联避免循环导入） ----------
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


def compute_bleu_n(ref, hyp, n):
    """BLEU-n（n=1/2/3），用均匀权重，方便与 BLEU-4 对照。"""
    try:
        from nltk.translate.bleu_score import sentence_bleu
        from nltk.translate.bleu_score import SmoothingFunction
        ref_tok = list(ref)
        hyp_tok = list(hyp)
        cc = SmoothingFunction().method4
        weights = tuple(1.0 / n for _ in range(n))
        return sentence_bleu([ref_tok], hyp_tok, weights=weights,
                             smoothing_function=cc)
    except Exception:
        return 0.0


def compute_rouge_l(ref, hyp):
    """ROUGE-L：字符级 tokenizer（中文无空格，默认 tokenizer 会把整句当一个 token，
    导致 ROUGE 恒为 0。字符级切分是中文评测的常规做法，与 BLEU 的字符级对齐。）"""
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


def compute_rouge_12(ref, hyp):
    """ROUGE-1 / ROUGE-2（字符级 tokenizer，精确率 P 的 fmeasure 便于 BLEU/ROUGE 对齐）"""
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


def compute_gscore(bleu, rouge1, rouge2, rouge_l, bertscore, weights=None):
    """
    Gscore 综合评分（参照 CEA 2025 论文：BLEU+BERTScore+ROUGE+chrF 加权合成）。
    默认权重：BLEU 0.2 / ROUGE 0.2 / BERTScore 0.6（语义主导）。
    """
    if weights is None:
        weights = {'bleu': 0.2, 'rouge': 0.2, 'bertscore': 0.6}
    rouge_avg = (rouge1 + rouge2 + rouge_l) / 3.0
    return (weights['bleu'] * bleu
            + weights['rouge'] * rouge_avg
            + weights['bertscore'] * bertscore)


def compute_bertscore_batch(refs, hyps):
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


def load_pipeline():
    kg = CropPestKG(NODES_CSV, RELS_CSV)
    store = TextBlockStore.load(os.path.join(INDEX_DIR, 'text_blocks.json'))
    with open(os.path.join(INDEX_DIR, 'vector_doc_ids.json'), encoding='utf-8') as f:
        doc_ids = json.load(f)
    vindex = VectorIndex.load(os.path.join(INDEX_DIR, 'vector.faiss'), doc_ids)
    bindex = BM25Index.load(os.path.join(INDEX_DIR, 'bm25.pkl'))
    embedder = Embedder()
    llm = LLMClient(get_llm_config(PROVIDER))
    retriever = Retriever(kg, store, vindex, bindex, embedder, llm_client=llm)
    return retriever, kg, store, llm


def load_graphrag_engine(retriever, kg, llm):
    """加载 GraphRAG 引擎（社区摘要 + 社区归属）。

    GraphRAG-Local 不依赖社区摘要；缺失摘要/社区文件时以空列表/空映射运行，
    避免正式对比入口因预处理文件缺失直接失败。
    """
    from graphrag.summarize import load_summaries, load_communities  # noqa: E402
    from graphrag.search import GraphRAGEngine  # noqa: E402
    from config import GRAPHRAG_DATA, GRAPHRAG  # noqa: E402
    try:
        summaries = load_summaries(
            os.path.join(GRAPHRAG_DATA, 'community_summaries.json'))
    except Exception:
        summaries = []
    try:
        comm_map, _ = load_communities(
            os.path.join(GRAPHRAG_DATA, 'communities.json'))
    except Exception:
        comm_map = {}
    community_of = {}
    for cid, members in comm_map.items():
        for m in members:
            community_of[m] = cid
    return GraphRAGEngine(kg, retriever, llm, summaries=summaries,
                          community_of=community_of,
                          summary_top_k=GRAPHRAG['summary_top_k'])


def load_gretriever_engine(retriever, kg):
    """加载 G-Retriever 基线引擎（嵌入打分简化版）"""
    from graphrag.gretriever import GRetrieverEngine  # noqa: E402
    return GRetrieverEngine(kg, retriever)


def load_lightrag_engine(retriever, kg):
    """加载 LightRAG 基线引擎（双层检索，预构建图谱适配版）"""
    from graphrag.lightrag import LightRAGEngine  # noqa: E402
    return LightRAGEngine(kg, retriever)


def load_pathrag_engine(retriever, kg):
    """加载 PathRAG 基线引擎（关系路径检索+剪枝，论文适配版）"""
    from graphrag.pathrag import PathRAGEngine  # noqa: E402
    return PathRAGEngine(kg, retriever)


def load_sgrag_engine(retriever, kg):
    """加载 SG-RAG 基线引擎（嵌入子图匹配，论文适配版）"""
    from graphrag.sgrag import SGRAGEngine  # noqa: E402
    return SGRAGEngine(kg, retriever)


def load_eval_set():
    """按数据集加载评测 QA（local=qa_test.json, global=qa_global.json）"""
    from config import GLOBAL_QA_FILE  # noqa: E402
    path = EVAL_FILE if DATASET == 'local' else GLOBAL_QA_FILE
    return json.load(open(path, encoding='utf-8'))


def build_fewshot_selector():
    """加载 180 条示例池并用 BGE 构建选择器"""
    pool = load_fewshot_pool(DEFAULT_FEWSHOT_FILE)
    bge = BGEEmbedder()
    selector = FewShotSelector(pool, bge)
    return selector


def build_variant_prompt(question, task, text_ev, graph_ev, selector, variant):
    """
    按变体构建提示词：
    - 提示侧消融：
        -no-evidence-prompt : 移除证据驱动提示（仅拼接证据到问题，基础 RAG 提示）
        -no-fewshot         : 移除动态少样本示例
        -no-diag-constraints: 移除结构化诊断要求（仅保留任务与证据）
    - ours-retrieval：检索 + concise 输出规则（无证据驱动模板/fewshot/诊断约束），
      提示结构与 zero-shot 对齐，隔离"纯检索"贡献。
    - 其他变体走标准 build_prompt_with_dynamic_fewshot。
    """
    # Ours-Retrieval：任务指令(concise) + 输出规则(concise) + 证据 + 问题。
    # 与零样本提示同构（无证据驱动模板、无 fewshot、无诊断约束），
    # 只额外加入检索证据，从而把"检索贡献"从提示工程中隔离出来。
    if variant == 'ours-retrieval':
        from modules.prompts import TASK_INSTRUCTIONS_CONCISE, get_output_rules
        parts = []
        parts.append("你是一个农业病虫害问答助手，请直接回答用户问题。")
        if task in TASK_INSTRUCTIONS_CONCISE:
            parts.append(TASK_INSTRUCTIONS_CONCISE[task].strip())
        parts.append(get_output_rules('concise', include_citations=False).strip())
        if text_ev:
            parts.append("检索到的文本：\n" + "\n".join(
                f"- {e['text']}" for e in text_ev[:10]))
        if graph_ev:
            parts.append("图谱关系：\n" + "\n".join(
                f"- {g['head']} -{g['rel']}-> {g['tail']}" for g in graph_ev[:20]))
        parts.append(f"用户问题：{question}")
        return "\n\n".join(parts)
    # 全局聚合题：不用六类任务模板，直接证据 + 问题
    if task == 'global_summary':
        parts = [f"用户问题：{question}"]
        if text_ev:
            parts.append("检索到的文本：\n" + "\n".join(
                f"- {e['text']}" for e in text_ev[:10]))
        if graph_ev:
            parts.append("图谱关系：\n" + "\n".join(
                f"- {g['head']} -{g['rel']}-> {g['tail']}" for g in graph_ev[:20]))
        parts.append("请依据以上信息，用简洁中文综合回答（结论 + 关键要点，总长不超过 200 字）。")
        return "\n\n".join(parts)
    # 基础 RAG 提示：证据直接拼接到问题，无角色/约束/诊断要求
    if variant == '-no-evidence-prompt':
        parts = [f"用户问题：{question}"]
        if text_ev:
            parts.append("检索到的文本：\n" + "\n".join(
                f"- {e['text']}" for e in text_ev[:10]))
        if graph_ev:
            parts.append("图谱关系：\n" + "\n".join(
                f"- {g['head']} -{g['rel']}-> {g['tail']}" for g in graph_ev[:20]))
        parts.append("请根据以上信息回答。")
        return "\n\n".join(parts)

    # 移除诊断约束：构建提示但把任务指令换成空（只保留任务+证据）
    if variant == '-no-diag-constraints':
        from modules.prompts import build_evidence_prompt
        prompt = build_evidence_prompt(
            question=question,
            text_evidence=text_ev,
            graph_evidence=graph_ev,
            task_type=task,
            fewshot_examples=selector.select(
                question, task, top_k=RETRIEVAL.get('fewshot_top_k', 3)),
            mode='concise',
            include_citations=False,
        )
        # 去掉"组织要求"部分（诊断约束所在）
        marker = '【当前任务的组织要求】'
        if marker in prompt:
            prompt = prompt.split(marker)[0].rstrip() + "\n\n请直接给出最终回答。"
        return prompt

    # 移除少样本：直接构建无示例的提示词
    if variant == '-no-fewshot':
        from modules.prompts import build_evidence_prompt
        return build_evidence_prompt(
            question=question,
            text_evidence=text_ev,
            graph_evidence=graph_ev,
            task_type=task,
            fewshot_examples=[],
            mode='concise',
            include_citations=False,
        )

    # 标准完整提示
    return build_prompt_with_dynamic_fewshot(
        question=question,
        task_type=task,
        text_evidence=text_ev,
        graph_evidence=graph_ev,
        selector=selector,
        top_k=RETRIEVAL.get('fewshot_top_k', 3),
        mode='concise',
        include_citations=False,
    )


def main():
    os.makedirs(RESULT_DIR, exist_ok=True)
    # 加载已完成的进度
    done = {}
    if os.path.exists(progress_file()):
        with open(progress_file(), encoding='utf-8') as f:
            done = json.load(f)
        print(f'已恢复进度：{len(done)} 条完成')
    qas = load_eval_set()
    if SUBSET and SUBSET > 0:
        qas = qas[:SUBSET]
    todo = [qa for qa in qas if qa['question'] not in done]
    print(f'总 {len(qas)} 条，待评测 {len(todo)} 条（dataset={DATASET}）')

    retriever, kg, store, llm = load_pipeline()
    selector = build_fewshot_selector()
    # GraphRAG 变体加载引擎
    engine = None
    if VARIANT in ('grag-local', 'grag-global'):
        engine = load_graphrag_engine(retriever, kg, llm)
    elif VARIANT == 'g-retriever':
        engine = load_gretriever_engine(retriever, kg)
    elif VARIANT == 'lightrag':
        engine = load_lightrag_engine(retriever, kg)
    elif VARIANT == 'pathrag':
        engine = load_pathrag_engine(retriever, kg)
    elif VARIANT == 'sg-rag':
        engine = load_sgrag_engine(retriever, kg)
    print(f'管线 + few-shot 选择器加载完成（variant={VARIANT}），开始评测...')

    t_start = time.time()
    for i, qa in enumerate(todo):
        question = qa['question']
        # 任务类型由 LLM 从问题文本自动判断，不读取 qa['task']（测试集真实标签）。
        # 全局聚合题按数据集类型直接使用 global_summary 模板。
        if DATASET == 'global':
            task, task_method = 'global_summary', 'dataset'
        else:
            task, task_method = classify_task(question, llm)
        if VARIANT == 'grag-local':
            text_ev, graph_ev, detail = engine.local_search(question)
            prompt = build_variant_prompt(question, task, text_ev, graph_ev,
                                          selector, 'graph-only')
        elif VARIANT in ('g-retriever', 'lightrag', 'pathrag', 'sg-rag'):
            text_ev, graph_ev, detail = engine.search(question)
            prompt = build_variant_prompt(question, task, text_ev, graph_ev,
                                          selector, 'graph-only')
        elif VARIANT == 'grag-global':
            text_ev, graph_ev, ganswer, detail = engine.global_search(question)
            # 全局题直接使用 GraphRAG 综合回答（不走六类任务模板）
            answer = ganswer
            # 引擎内部已完成综合生成，这里只记录单候选，保持结果字段统一
            generation = {
                'answer': answer,
                'candidates': [answer],
                'valid_indices': [0],
                'core_conclusions': [answer],
                'votes': [1],
                'coverage': [0.0],
                'agreement': [0.0],
                'selected_index': 0,
                'method': 'engine',
                'error': None,
            }
        else:
            # ours-retrieval：完整检索（与 full 相同路径，含 _keyword_target_pests
            # 目标锚点增强），提示侧退化为 concise 输出规则 + 证据，隔离检索贡献。
            ev_variant = 'full' if VARIANT == 'ours-retrieval' else VARIANT
            text_ev, graph_ev, detail = retriever.get_evidence(question, variant=ev_variant)
            prompt = build_variant_prompt(question, task, text_ev, graph_ev, selector, VARIANT)
        if VARIANT != 'grag-global':
            # 三次独立生成 + 核心结论一致性投票 + 证据覆盖度选择（论文 3.4（4））
            generation = llm.complete_with_consistency_details(prompt)
            answer = generation.get('answer', '')
            if generation.get('method') == 'failed':
                # 三次生成全部失败：不写入进度，下次运行会重试该题
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
            # 生成过程审计信息：三条候选、投票、覆盖度、最终选择
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
        # 每 CHUNK 条落盘
        if (i + 1) % CHUNK == 0:
            with open(progress_file(), 'w', encoding='utf-8') as f:
                json.dump(done, f, ensure_ascii=False)
            elapsed = time.time() - t_start
            per = elapsed / (i + 1)
            remaining = per * (len(todo) - i - 1) / 60
            print(f'  已评测 {len(done)}/{len(qas)}（本次 {i+1}/{len(todo)}），'
                  f'预计剩余 {remaining:.0f} 分钟', flush=True)

    # 全部完成，落盘
    with open(progress_file(), 'w', encoding='utf-8') as f:
        json.dump(done, f, ensure_ascii=False)

    if not done:
        print('没有成功完成的条目，未写入汇总。'
              '请检查 API Key / 模型 ID 配置后重试。', flush=True)
        return

    # 批量 BERTScore
    print('批量计算 BERTScore...', flush=True)
    items = list(done.items())
    refs = [v['ref'] for _, v in items]
    answers = [v['answer'] for _, v in items]
    bf_scores = compute_bertscore_batch(refs, answers)
    for (_q, v), bf in zip(items, bf_scores):
        v['bertscore'] = bf
    with open(progress_file(), 'w', encoding='utf-8') as f:
        json.dump(done, f, ensure_ascii=False)

    # 汇总指标
    import statistics
    bleu1 = [v.get('bleu1', 0.0) for _, v in items]
    bleu2 = [v.get('bleu2', 0.0) for _, v in items]
    bleu4 = [v['bleu'] for _, v in items]
    rouge = [v['rouge'] for _, v in items]
    rouge1 = [v.get('rouge1', 0.0) for _, v in items]
    rouge2 = [v.get('rouge2', 0.0) for _, v in items]
    bs = [v['bertscore'] for _, v in items]
    m_b1 = statistics.mean(bleu1)
    m_b2 = statistics.mean(bleu2)
    m_b4 = statistics.mean(bleu4)
    m_r1 = statistics.mean(rouge1)
    m_r2 = statistics.mean(rouge2)
    m_rl = statistics.mean(rouge)
    m_bs = statistics.mean(bs)
    gscore = compute_gscore(m_b4, m_r1, m_r2, m_rl, m_bs)
    summary = {
        'n': len(items),
        'variant': VARIANT,
        'bleu1': m_b1,
        'bleu2': m_b2,
        'bleu4': m_b4,
        'rouge1': m_r1,
        'rouge2': m_r2,
        'rouge_l': m_rl,
        'bertscore': m_bs,
        'gscore': gscore,
    }
    with open(summary_file(), 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print('=' * 60)
    print(f'评测完成 (variant={VARIANT}, n={summary["n"]})')
    print(f'  BLEU-1     = {summary["bleu1"]:.4f}')
    print(f'  BLEU-2     = {summary["bleu2"]:.4f}')
    print(f'  BLEU-4     = {summary["bleu4"]:.4f}')
    print(f'  ROUGE-1    = {summary["rouge1"]:.4f}')
    print(f'  ROUGE-2    = {summary["rouge2"]:.4f}')
    print(f'  ROUGE-L    = {summary["rouge_l"]:.4f}')
    print(f'  BERTScore  = {summary["bertscore"]:.4f}')
    print(f'  Gscore     = {summary["gscore"]:.4f}')


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', default='full',
                    choices=EVAL_VARIANTS + ['grag-global'],
                    help='检索/提示变体（消融/基线/图RAG方法）')
    ap.add_argument('--subset', type=int, default=0,
                    help='仅评测前 N 条（0=全部），用于快速验证')
    ap.add_argument('--provider', default='deepseek',
                    choices=sorted(LLM_PROVIDERS),
                    help='LLM provider（模型 ID 由对应环境变量传入）')
    ap.add_argument('--dataset', default='local',
                    choices=['local', 'global'],
                    help='评测数据集：local=qa_test.json(480), global=qa_global.json')
    args = ap.parse_args()
    VARIANT = args.variant
    SUBSET = args.subset
    PROVIDER = args.provider
    DATASET = args.dataset
    main()
