# -*- coding: utf-8 -*-
"""
diagnose_ours.py —— Ours-Full 失败归因分析（只读，不修改任何实验）
==================================================================
诊断问题：
  A. 证据构成：graph vs text 单位数、token 占用
  B. 证据质量：gold sentence 覆盖率（各方法对比）
  C. 失败归因：证据覆盖但答错（生成层）vs 证据没覆盖（检索层）
  D. 锚点留存：graph_top 里是否保留了 question 中的显式实体锚点
  E. 分类型损失
"""
import sys
import os
import json
import re
from collections import Counter, defaultdict

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from common import (DATA, PRED, SAMPLE_FILE,
                    normalize_answer, f1_score, normalize_type)

PREDFILES = {
    'Ours-Full': 'ours_full.jsonl',
    'LightRAG': 'lightrag.jsonl',
    'SG-RAG': 'sg_rag.jsonl',
    'Hybrid RAG': 'hybrid_rag.jsonl',
    'GraphRAG': 'graphrag.jsonl',
    'NaiveRAG': 'naive_rag.jsonl',
}

# ---------- 数据 ----------
SAMPLES = [json.loads(l) for l in open(SAMPLE_FILE, encoding='utf-8')]
SID2ROW = {s['id']: s for s in SAMPLES}

# gold supporting facts
GOLD = {}
for s in SAMPLES:
    sf = s.get('supporting_facts') or {}
    GOLD[s['id']] = {
        'title': list(sf.get('title', [])),
        'sent_id': [int(x) for x in sf.get('sent_id', [])],
        'pairs': set(zip(sf.get('title', []), [int(x) for x in sf.get('sent_id', [])])),
    }


def load_preds(name):
    rows = [json.loads(l) for l in open(os.path.join(PRED, PREDFILES[name]), encoding='utf-8')]
    return {r['id']: r for r in rows}


PREDS = {n: load_preds(n) for n in PREDFILES}

# ---------- 覆盖率 ----------
def gold_covered(ev, gold):
    """ev 覆盖的 gold (title, sent_id) 对数"""
    cov = set()
    for e in ev:
        t = e.get('title')
        sids = e.get('sentence_ids')
        if sids:  # 图证据
            gs = gold['title'].count(t)
            # 匹配该 title 的 gold sent_ids
            for tt, sid in gold['pairs']:
                if tt == t and sid in sids:
                    cov.add((tt, sid))
        else:
            for tt, sid in gold['pairs']:
                if tt == e.get('title') and sid == e.get('sentence_id'):
                    cov.add((tt, sid))
    return cov


def f1_for(rows, qid):
    r = rows[qid]
    return f1_score(normalize_answer(r['prediction']), normalize_answer(r['gold_answer']))


print('=' * 72)
print('A. Ours-Full 证据构成（graph vs text）')
print('=' * 72)
of_rows = PREDS['Ours-Full']
comp = Counter()
for r in of_rows.values():
    kinds = [e['source'] for e in r['retrieved_evidence']]
    comp[('graph', kinds.count('graph'))] += 1
    comp[('text', kinds.count('text'))] += 1
g_dist = Counter()
t_dist = Counter()
for r in of_rows.values():
    kinds = [e['source'] for e in r['retrieved_evidence']]
    g_dist[kinds.count('graph')] += 1
    t_dist[kinds.count('text')] += 1
print(f'  graph 单位数分布: {dict(sorted(g_dist.items()))}')
print(f'  text  单位数分布: {dict(sorted(t_dist.items()))}')

# 平均 token（graph vs text，用 API 实际估算 by text len）
def approx_tok(s):
    return len(s.split())
g_tok = sum(approx_tok(e['text']) for r in of_rows.values() for e in r['retrieved_evidence'] if e['source'] == 'graph')
t_tok = sum(approx_tok(e['text']) for r in of_rows.values() for e in r['retrieved_evidence'] if e['source'] == 'text')
print(f'  平均 graph 证据 token/条: {g_tok/len(of_rows):.0f}')
print(f'  平均 text  证据 token/条: {t_tok/len(of_rows):.0f}')
n_text0 = sum(1 for r in of_rows.values() if sum(1 for e in r['retrieved_evidence'] if e['source']=='text') == 0)
print(f'  纯 graph（text=0 条）: {n_text0}/{len(of_rows)}')

print()
print('=' * 72)
print('B. gold sentence 覆盖率（各方法）')
print('=' * 72)
cov_stats = {}
for n in PREDFILES:
    covs = []
    for qid, r in PREDS[n].items():
        c = len(gold_covered(r.get('retrieved_evidence', []), GOLD[qid]))
        gt = len(GOLD[qid]['pairs'])
        covs.append(c / gt if gt else 0.0)
    cov_stats[n] = covs
    print(f'  {n:<12} 平均覆盖 {sum(covs)/len(covs)*100:5.1f}%   覆盖>0 占比 {sum(1 for c in covs if c>0)/len(covs)*100:5.1f}%')

print()
print('=' * 72)
print('C. 失败归因：Ours-Full 答错时，证据是否覆盖了 gold？')
print('=' * 72)
# 分桶
n_ev_cov = n_ev_miss = n_ok = 0
bad_cases = []
for qid in of_rows:
    r = of_rows[qid]
    cov = gold_covered(r.get('retrieved_evidence', []), GOLD[qid])
    f1 = f1_for(of_rows, qid)
    if f1 >= 0.999:
        n_ok += 1
        continue
    if cov:
        n_ev_cov += 1
    else:
        n_ev_miss += 1
    bad_cases.append(qid)
print(f'  答对: {n_ok}/{len(of_rows)}')
print(f'  答错但证据已覆盖 gold（生成层/证据次序问题）: {n_ev_cov}')
print(f'  答错且证据未覆盖 gold（检索层问题）: {n_ev_miss}')

# 同问题 LightRAG 的覆盖率
lr_better_cov = 0
for qid in bad_cases:
    if qid not in PREDS['LightRAG']:
        continue
    lc = gold_covered(PREDS['LightRAG'][qid].get('retrieved_evidence', []), GOLD[qid])
    oc = gold_covered(of_rows[qid].get('retrieved_evidence', []), GOLD[qid])
    if len(lc) > len(oc):
        lr_better_cov += 1
print(f'  其中 LightRAG 证据覆盖 > Ours-Full 的问题数: {lr_better_cov}/{len(bad_cases)}')

print()
print('=' * 72)
print('D. 显式实体锚点在 Ours-Full 图证据中的留存')
print('=' * 72)
def entity_anchors(question, titles):
    q_lower = question.lower()
    matches = []
    from difflib import SequenceMatcher
    for t in titles:
        tl = t.lower()
        if tl in q_lower:
            matches.append(t)
            continue
        m = re.match(r'^(.+?)\s*\(', t)
        if m and m.group(1).lower() in q_lower:
            matches.append(t)
            continue
        for token in re.findall(r"[\w']+", q_lower):
            if len(token) > 3 and SequenceMatcher(None, tl, token).ratio() > 0.9:
                matches.append(t)
                break
    return list(dict.fromkeys(matches))

# 需要 question 的 title 集合 —— 从 graph 证据里恢复全部节点集（仅统计图证据标题）
retain = lost = no_ent = 0
lost_examples = []
for qid, r in of_rows.items():
    q = r['question']
    graph_titles = [e['title'] for e in r['retrieved_evidence'] if e['source'] == 'graph']
    all_titles = set(graph_titles)
    # 图证据之外的可能锚点：从全量图里找。用 graphs.json 恢复
    g = json.load(open(os.path.join(DATA, 'graphs.json'), encoding='utf-8'))
    all_node_titles = {n['title'] for n in g[qid]['nodes']}
    ents = entity_anchors(q, all_node_titles)
    if not ents:
        no_ent += 1
        continue
    kept = [t for t in ents if t in graph_titles]
    if kept:
        retain += 1
    else:
        lost += 1
        lost_examples.append((qid, q, ents))
n_q = len(of_rows)
print(f'  有显式实体锚点的问题: {n_q-no_ent}/{n_q}')
print(f'  锚点全部进入图证据 Top: {retain}/{n_q-no_ent}')
print(f'  锚点全部丢失（被 rerank 挤掉）: {lost}/{n_q-no_ent}')
for qid, q, ents in lost_examples[:5]:
    print(f'    丢失锚点样例 [{qid}]: {ents}')
    print(f'      Q: {q[:110]}')
    got = [e['title'] for e in of_rows[qid]['retrieved_evidence'] if e['source'] == 'graph']
    print(f'      实际图证据: {got}')

print()
print('=' * 72)
print('E. 分类型：Ours-Full vs LightRAG / SG-RAG F1 差值')
print('=' * 72)
types = ['comparison', 'inference', 'compositional', 'bridge-comparison']
for t in types:
    qids = [qid for qid in of_rows if normalize_type(GOLD[qid].get('type','')) or True]
# 从 sample 里拿 type
TYPE_OF = {s['id']: normalize_type(s['type']) for s in SAMPLES}
for t in types:
    qids = [qid for qid in of_rows if TYPE_OF.get(qid) == t]
    of_f1 = sum(f1_for(of_rows, q) for q in qids) / max(1, len(qids))
    lr_f1 = sum(f1_for(PREDS['LightRAG'], q) for q in qids) / max(1, len(qids))
    sg_f1 = sum(f1_for(PREDS['SG-RAG'], q) for q in qids) / max(1, len(qids))
    print(f'  {t:<18} Ours={of_f1:.3f}  LightRAG={lr_f1:.3f}  SG={sg_f1:.3f}   (n={len(qids)})')
