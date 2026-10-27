# -*- coding: utf-8 -*-
"""
evaluate.py —— 评测（HotpotQA/2Wiki 官方指标）
================================================
- Answer EM / F1（normalize_answer + token overlap）
- Support EM / F1（支撑句集合匹配；图/路径证据映射回 (title, sent_id)）
- Joint EM / F1（Joint EM = Answer EM ∧ Support EM；Joint F1 = (Answer F1 + SP F1)/2）
- 分任务统计（4 type × Answer EM/F1）
- token 统计（读 API usage 字段）

输出：
  outputs/tables/table_overall.csv
  outputs/tables/table_taskwise.csv
  outputs/tables/token_usage.csv
"""
import sys
import os
import json
import csv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (DATA, PRED, TABLES, SAMPLE_FILE,
                    normalize_answer, f1_score, normalize_type)
from collections import Counter

METHOD_FILES = {
    'LLM-No Retrieval': 'llm_no_retrieval.jsonl',
    'NaiveRAG': 'naive_rag.jsonl',
    'Hybrid RAG': 'hybrid_rag.jsonl',
    'GraphRAG': 'graphrag.jsonl',
    'LightRAG': 'lightrag.jsonl',
    'G-Retriever': 'g_retriever.jsonl',
    'PathRAG': 'pathrag.jsonl',
    'SG-RAG': 'sg_rag.jsonl',
    'Ours-Full': 'ours_full.jsonl',   # run_method.py: run_ours_full（四路检索 + 两级融合 + 动态 few-shot）
}
ORDER = list(METHOD_FILES.keys())

# gold supporting_facts（评测用，不进入检索/生成）
GOLD_SF = {}


def load_gold_sf():
    for line in open(SAMPLE_FILE, encoding='utf-8'):
        s = json.loads(line)
        sf = s.get('supporting_facts') or {}
        GOLD_SF[s['id']] = {
            'title': list(sf.get('title', [])),
            'sent_id': [int(x) for x in sf.get('sent_id', [])],
        }


def support_hit(ev, gold):
    """检索证据命中 gold supporting_facts（非去重计数，供消融脚本沿用）。
    - 文本证据：title + sentence_id 精确匹配
    - 图证据：title（或 node_titles 中任一）命中 + 任一 sentence_ids 落在 gold 对应 title 的 sent_id 中
    """
    gold_title_sent = {}   # title -> set(sent_id)
    for t, sid in zip(gold['title'], gold['sent_id']):
        gold_title_sent.setdefault(t, set()).add(sid)
    hit = 0
    total = len(ev)
    for e in ev:
        titles = e.get('node_titles') or [e.get('title')]
        sids = e.get('sentence_ids')
        if sids:  # 图/路径证据：sentence_ids 列表
            for t in titles:
                if t in gold_title_sent and (set(sids) & gold_title_sent[t]):
                    hit += 1
                    break
        else:     # 文本证据：单个 sentence_id
            if (e.get('title'), e.get('sentence_id')) in set(zip(gold['title'], gold['sent_id'])):
                hit += 1
    return hit, total


def support_metrics(ev, gold):
    """去重的 support P/R/F1（正式口径）。
    - Recall@8（R）：gold (title, sent_id) 对被 top-8 证据覆盖的比例（证据=≤8 units 预算）
    - Precision（P）：证据单位中命中 ≥1 个 gold 对的单位占比
    - F1：2PR/(P+R)
    图证据按 node_titles（或 title）+ sentence_ids 映射回 (title, sentence_id)。
    R/P 均在 gold 对上集合去重，保证 [0,1]。
    """
    gold_pairs = set(zip(gold['title'], gold['sent_id']))
    if not gold_pairs:
        return 0.0, 0.0, 0.0
    covered = set()          # 被覆盖的 gold 对
    hit_units = 0
    total_units = len(ev)
    for e in ev:
        titles = e.get('node_titles') or [e.get('title')]
        sids = e.get('sentence_ids')
        unit_hit = False
        if sids:             # 图/路径证据：title(s)+sentence_ids 列表
            for tt, sid in gold_pairs:
                if tt in titles and sid in sids:
                    covered.add((tt, sid))
                    unit_hit = True
        else:                # 文本证据：单个 sentence_id
            if (e.get('title'), e.get('sentence_id')) in gold_pairs:
                covered.add((e.get('title'), e.get('sentence_id')))
                unit_hit = True
        if unit_hit:
            hit_units += 1
    p = hit_units / total_units if total_units else 0.0
    r = len(covered) / len(gold_pairs)
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def load_preds():
    preds = {}
    for name, f in METHOD_FILES.items():
        p = os.path.join(PRED, f)
        if not os.path.exists(p):
            print(f'  [缺失] {f}')
            preds[name] = []
            continue
        rows = [json.loads(l) for l in open(p, encoding='utf-8')]
        preds[name] = rows
    return preds


def build_pred_sp(qid, ev, graphs):
    """把检索证据映射为"预测支撑句集合" pred_sp = {(title, sent_id), ...}。
    - 文本证据：单个 (title, sentence_id)
    - 图/路径证据：node_titles（或 title）中每个节点的全部 sentence_ids
      （路径记录只有并集，每节点句号从 graphs.json 恢复，保证映射正确）
    """
    node_sids = {n['title']: n.get('sentence_ids', [])
                 for n in graphs.get(qid, {}).get('nodes', [])}
    pred = set()
    for e in ev:
        sids = e.get('sentence_ids')
        if sids:                     # 图/路径证据
            titles = e.get('node_titles') or [e.get('title')]
            for t in titles:
                own = node_sids.get(t)
                if own:
                    pred.update((t, sid) for sid in own)
                else:
                    pred.update((t, sid) for sid in sids)
        else:                        # 文本证据：单句
            pred.add((e.get('title'), e.get('sentence_id')))
    return pred


def sp_metrics(pred_sp, gold_sp):
    """HotpotQA 风格支撑句指标：Support EM（集合精确匹配）+ Support F1。
    gold_sp: set of (title, sent_id)
    """
    gold_sp = set(gold_sp)
    if not gold_sp:
        return 0.0, 0.0, 0.0, 0.0
    common = pred_sp & gold_sp
    num_same = len(common)
    if num_same == 0 or not pred_sp:
        return 0.0, 0.0, 0.0, 0.0
    p = num_same / len(pred_sp)
    r = num_same / len(gold_sp)
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    em = 1 if pred_sp == gold_sp else 0
    return p, r, f, em


def eval_method(rows, graphs):
    """Answer EM / F1（论文 Table 1 主指标）。
    graphs 参数保留给未来 support 评测复用，当前只输出答案指标。
    """
    a_em = a_f1 = 0.0
    n = 0
    for r in rows:
        n += 1
        gn = normalize_answer(r['gold_answer'])
        pn = normalize_answer(r['prediction'])
        a_em += 1 if pn == gn else 0
        a_f1 += f1_score(pn, gn)
    m = max(1, n)
    return {'EM': a_em / m, 'F1': a_f1 / m, 'n': n}


def main():
    load_gold_sf()
    preds = load_preds()
    graphs = json.load(open(os.path.join(DATA, 'graphs.json'), encoding='utf-8'))

    # ---------- 总体 ----------
    overall = {}
    for name in ORDER:
        overall[name] = eval_method(preds[name], graphs)

    # token 统计
    tok = {}
    for name in ORDER:
        rows = preds[name]
        tot_in = sum(r.get('input_tokens', 0) for r in rows)
        tot_out = sum(r.get('output_tokens', 0) for r in rows)
        tok[name] = {
            'n': len(rows),
            'calls': len(rows),
            'in': tot_in,
            'out': tot_out,
            'total': tot_in + tot_out,
            'avg_in': tot_in / len(rows) if rows else 0,
            'avg_out': tot_out / len(rows) if rows else 0,
        }

    # ---------- 输出 overall ----------
    with open(os.path.join(TABLES, 'table_overall.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['Method', 'Answer_EM', 'Answer_F1',
                    'Avg_Input_Tokens', 'Avg_Output_Tokens', 'Avg_Latency'])
        for name in ORDER:
            o = overall[name]
            t = tok[name]
            lat = sum(r.get('latency_sec', 0) for r in preds[name]) / len(preds[name]) if preds[name] else 0
            w.writerow([name, f'{o["EM"]:.4f}', f'{o["F1"]:.4f}',
                        f'{t["avg_in"]:.1f}', f'{t["avg_out"]:.1f}', f'{lat:.3f}'])
    print('已保存 table_overall.csv')

    # ---------- 分任务 ----------
    types = ['comparison', 'inference', 'compositional', 'bridge-comparison']
    with open(os.path.join(TABLES, 'table_taskwise.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        hdr = ['Method']
        for t in types:
            hdr += [f'{t.title().replace("-", "_")}_EM',
                    f'{t.title().replace("-", "_")}_F1']
        w.writerow(hdr)
        for name in ORDER:
            row = [name]
            for t in types:
                sub = [r for r in preds[name] if normalize_type(r.get('gold_type', '')) == t]
                em = sum(1 for r in sub if normalize_answer(r['prediction']) == normalize_answer(r['gold_answer'])) / max(1, len(sub))
                f1s = sum(f1_score(normalize_answer(r['prediction']), normalize_answer(r['gold_answer'])) for r in sub) / max(1, len(sub))
                row += [f'{em:.4f}', f'{f1s:.4f}']
            w.writerow(row)
    print('已保存 table_taskwise.csv')

    # ---------- token ----------
    with open(os.path.join(TABLES, 'token_usage.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['Method', 'Num_Questions', 'Num_LLM_Calls', 'Input_Tokens',
                    'Output_Tokens', 'Total_Tokens', 'Avg_Input_Tokens', 'Avg_Output_Tokens'])
        for name in ORDER:
            t = tok[name]
            w.writerow([name, t['n'], t['calls'], t['in'], t['out'], t['total'],
                        f'{t["avg_in"]:.1f}', f'{t["avg_out"]:.1f}'])
    print('已保存 token_usage.csv')

    # 控制台摘要
    print()
    print(f'{"Method":<16} {"EM":>6} {"F1":>6}')
    for name in ORDER:
        o = overall[name]
        print(f'{name:<16} {o["EM"]*100:>5.1f}% {o["F1"]*100:>5.1f}%')


if __name__ == '__main__':
    main()
