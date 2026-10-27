# -*- coding: utf-8 -*-
"""
validate_protocol.py —— 正式实验前协议检查（14 项）
=====================================================
任一检查失败 → 禁止运行正式实验。

泄漏防护是"代码级保证"：
- 检索/构图只用 context 字段（retrieve 与 generate 读不到 gold 字段）
- gold answer/supporting_facts/evidences/type 只出现在结果文件与评测
- 本文件做静态审计 + 数据完整性 + 参数一致性检查
"""
import sys
import os
import json
import re

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (DATA, SAMPLE_FILE, FEWSHOT_FILE,
                    RANDOM_SEED, TEST_TYPES, TEST_PER_TYPE, FEWSHOT_PER_TYPE,
                    MAX_EVIDENCE_UNITS, MAX_EVIDENCE_TOKENS,
                    LLM_TEMPERATURE, LLM_TOP_P, LLM_MAX_TOKENS,
                    normalize_type, estimate_tokens)
import numpy as np

FAILURES = []


def record(name, ok, detail=''):
    status = 'PASS' if ok else 'FAIL'
    print(f'  [{status}] {name}' + (f' — {detail}' if detail else ''))
    if not ok:
        FAILURES.append(name)


def main():
    print('========== Protocol Validation ==========')

    # 1. test sample 数量（由 TEST_PER_TYPE × 类别数 决定）
    samples = [json.loads(l) for l in open(SAMPLE_FILE, encoding='utf-8')]
    n_test = TEST_PER_TYPE * len(TEST_TYPES)
    record(f'1. test sample 数量 == {n_test}', len(samples) == n_test, str(len(samples)))

    # 2. 每个 type 样本数一致
    from collections import Counter
    cnt = Counter(normalize_type(s['type']) for s in samples)
    record(f'2. 四种 type 各 {TEST_PER_TYPE}',
           all(cnt[t] == TEST_PER_TYPE for t in TEST_TYPES), str(dict(cnt)))

    # 3. few-shot 池（由 FEWSHOT_PER_TYPE × 类别数 决定）
    fs = [json.loads(l) for l in open(FEWSHOT_FILE, encoding='utf-8')]
    fcnt = Counter(normalize_type(f['type']) for f in fs)
    n_fs = FEWSHOT_PER_TYPE * len(TEST_TYPES)
    record(f'3. few-shot 池 {n_fs} 条、各 {FEWSHOT_PER_TYPE}',
           len(fs) == n_fs and all(fcnt[t] == FEWSHOT_PER_TYPE for t in TEST_TYPES),
           str(dict(fcnt)))

    # 4. 测试集与 few-shot 无 id 冲突
    sids = {s['id'] for s in samples}
    fids = {f['id'] for f in fs}
    record('4. test 与 few-shot 无 id 冲突', not (sids & fids))

    # 5. 每个 question 有 candidate context（title + sentences 对齐）
    ctx_ok = all(
        len(s['context']['title']) == len(s['context']['sentences'])
        for s in samples + fs)
    record('5. context title/sentences 对齐', ctx_ok)

    # 6. gold answer 非空
    record('6. gold answer 非空', all(s.get('answer') for s in samples))

    # 7. 静态审计：检索/生成逻辑不读 gold 字段。
    #    gold_answer 只出现在结果文件的"输出组装"处（run_one 末尾 / main 输出），
    #    检索与 prompt 构建（run_* 函数）不得引用 gold 字段。
    BASE = os.path.dirname(os.path.abspath(__file__))
    gold_leaks = ['gold_answer', 'supporting_facts', "'evidences'", '"evidences"']
    all_found = []

    # 7a. run_method.py
    src = open(os.path.join(BASE, 'run_method.py'), encoding='utf-8').read()
    run_one_idx = src.find('def run_one')
    retrieve_src = src[:run_one_idx] if run_one_idx > 0 else src
    all_found += [w for w in gold_leaks if w in retrieve_src]

    # 7b. run_ours_ablation.py（最终 Ours-Full 方法所在文件）
    ab_src = open(os.path.join(BASE, 'run_ours_ablation.py'), encoding='utf-8').read()
    main_idx = ab_src.find('def main(')
    ab_retrieve = ab_src[:main_idx] if main_idx > 0 else ab_src
    all_found += [w for w in gold_leaks if w in ab_retrieve]

    found = list(dict.fromkeys(all_found))
    record('7. 检索/生成代码不含 gold 字段引用（代码级防泄漏，含 run_ours_ablation.py）',
           not found, ('发现: ' + ','.join(found)) if found else '')

    # 8. 所有 RAG evidence <= 8 units（实测预测文件，若有）
    pred_dir = os.path.join(DATA, '..', 'outputs', 'predictions')
    if os.path.isdir(pred_dir):
        files = [f for f in os.listdir(pred_dir) if f.endswith('.jsonl')]
        bad = 0
        for f in files:
            for line in open(os.path.join(pred_dir, f), encoding='utf-8'):
                r = json.loads(line)
                if len(r.get('retrieved_evidence', [])) > MAX_EVIDENCE_UNITS:
                    bad += 1
        record('8. evidence <= 8 units（预测文件）', bad == 0, f'{len(files)} files, {bad} violations')
    else:
        record('8. evidence <= 8 units（预测文件）', True, 'no predictions yet (deferred)')

    # 9. LLM 参数一致
    record('9. LLM 生成参数一致 (temp=0.2, top_p=0.9, max=64)',
           LLM_TEMPERATURE == 0.2 and LLM_TOP_P == 0.9 and LLM_MAX_TOKENS == 64)

    # 10. random seed == 42
    record('10. random seed == 42', RANDOM_SEED == 42)

    # 11. 检索参数固定
    from common import DENSE_TOP_K, BM25_TOP_K, RRF_K, GRAPH_HOPS, FEWSHOT_TOP_K
    record('11. 检索参数固定 (K=20/20, RRF=60, hop=2, fewshot=3)',
           DENSE_TOP_K == 20 and BM25_TOP_K == 20 and RRF_K == 60
           and GRAPH_HOPS == 2 and FEWSHOT_TOP_K == 3)

    # 12. sentence corpus 一致性（每 question 句子数 vs 索引）
    sentences = [json.loads(l) for l in open(os.path.join(DATA, 'sentences.jsonl'), encoding='utf-8')]
    lookup = json.load(open(os.path.join(DATA, 'sentence_lookup.json'), encoding='utf-8'))
    q2s = lookup['q_sent_idx']
    corpus_ok = all(
        len(q2s.get(s['id'], [])) > 0 for s in samples)
    record('12. 所有 test question 有句子索引', corpus_ok)

    # 13. embedding 形状（question 数 = test + few-shot）
    se = np.load(os.path.join(DATA, 'sentence_embeddings.npy'))
    qe = np.load(os.path.join(DATA, 'question_embeddings.npy'))
    n_q = len(samples) + len(fs)
    record(f'13. embedding 形状 (sent x 768, {n_q} x 768)',
           se.shape[1] == 768 and qe.shape == (n_q, 768),
           f'sent={se.shape}, q={qe.shape}')

    # 14. graph 构建完整（每个 question 都有图）
    graphs = json.load(open(os.path.join(DATA, 'graphs.json'), encoding='utf-8'))
    record(f'14. {len(samples)} 个 question 都有 local graph',
           len(graphs) == len(samples) and all(len(g['nodes']) == 10 for g in graphs.values()),
           f'graphs={len(graphs)}')

    print()
    if FAILURES:
        print(f'========== FAIL: {len(FAILURES)} 项检查未通过 ==========')
        print('禁止运行正式实验。')
        sys.exit(1)
    else:
        print('========== ALL CHECKS PASSED ==========')


if __name__ == '__main__':
    main()
