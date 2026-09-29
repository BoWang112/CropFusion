# -*- coding: utf-8 -*-
"""
eval_ablation.py —— 消融变体对比评测
=====================================
对每个变体计算 EM / F1 / Support / gold 覆盖率，与基线 Ours-Full 对比。
复用 evaluate.py 的 support_hit 与 gold 加载，保证指标口径一致。
"""
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evaluate                      # noqa: F401  (reuse support_hit / GOLD_SF)
from evaluate import GOLD_SF, support_hit
from common import PRED, normalize_answer, f1_score, normalize_type

evaluate.load_gold_sf()

VARIANTS = ['ours_full', 'anchor_retain', 'text_quota',
            'entity_first', 'sentence_ev', 'v2',
            'anchor_fs0', 'no_rerank', 'no_rerank_fs0',
            'pure_graph8', 'sentence_gated8',
            'graph8_bfs', 'pure_graph8_fs0', 'pure_graph8_fs_type',
            'pure_graph8_fs_type5', 'pure_graph8_fs_type10']
DISPLAY = {
    'ours_full': 'Ours-Full (baseline)',
    'anchor_retain': 'Fix1 anchor_retain',
    'text_quota': 'Fix2 text_quota',
    'entity_first': 'Fix3 entity_first',
    'sentence_ev': 'Fix4 sentence_ev',
    'v2': 'All fixes (v2)',
    'anchor_fs0': 'anchor_retain+noFS',
    'no_rerank': 'anchor_retain+noRerank',
    'no_rerank_fs0': 'noRerank+noFS',
    'pure_graph8': 'pure_graph8',
    'sentence_gated8': 'sentence_gated8',
    'graph8_bfs': 'graph8_bfs',
    'pure_graph8_fs0': 'pure_graph8+noFS',
    'pure_graph8_fs_type': 'pure_graph8+FSbyType',
    'pure_graph8_fs_type5': 'FSbyType5',
    'pure_graph8_fs_type10': 'FSbyType10',
}


def load_rows(name):
    p = os.path.join(PRED, f'{name}.jsonl')
    if not os.path.exists(p):
        return []
    return [json.loads(l) for l in open(p, encoding='utf-8')]


def gold_pairs(qid):
    gold = GOLD_SF[qid]
    return set(zip(gold['title'], gold['sent_id']))


def metrics(rows):
    em = f1s = sp = sr = sf1 = 0.0
    cov = 0.0
    n = len(rows)
    for r in rows:
        gn = normalize_answer(r['gold_answer'])
        pn = normalize_answer(r['prediction'])
        em += 1 if pn == gn else 0
        f1s += f1_score(pn, gn)
        gold = GOLD_SF.get(r['id'], {'title': [], 'sent_id': []})
        hit, total = support_hit(r.get('retrieved_evidence', []), gold)
        if total:
            gt = len(set(zip(gold['title'], gold['sent_id'])))
            p = hit / total
            rec = hit / gt if gt else 0
            f = 2 * p * rec / (p + rec) if (p + rec) else 0
            sp += p
            sr += rec
            sf1 += f
        # gold 覆盖率
        gp = gold_pairs(r['id'])
        c = set()
        for e in r.get('retrieved_evidence', []):
            t = e.get('title')
            sids = e.get('sentence_ids')
            if sids:
                for tt, sid in gp:
                    if tt == t and sid in sids:
                        c.add((tt, sid))
            else:
                for tt, sid in gp:
                    if tt == t and sid == e.get('sentence_id'):
                        c.add((tt, sid))
        cov += len(c) / max(1, len(gp))
    m = max(1, n)
    return {'EM': em / m, 'F1': f1s / m, 'SP': sp / m, 'SR': sr / m,
            'SF1': sf1 / m, 'COV': cov / m, 'n': n}


def main():
    print(f'{"variant":<20} {"EM":>6} {"F1":>6} {"SP":>6} {"SR":>6} '
          f'{"SF1":>6} {"COV":>6} {"n":>4}')
    all_m = {}
    for name in VARIANTS:
        rows = load_rows(name)
        m = metrics(rows) if rows else {}
        all_m[name] = m
        if rows:
            print(f'{DISPLAY[name]:<20} {m["EM"]*100:>5.1f}% {m["F1"]*100:>5.1f}% '
                  f'{m["SP"]*100:>5.1f}% {m["SR"]*100:>5.1f}% '
                  f'{m["SF1"]*100:>5.1f}% {m["COV"]*100:>5.1f}% {m["n"]:>4}')
        else:
            print(f'{DISPLAY[name]:<20} (无预测)')

    base = all_m['ours_full']
    print()
    print('相对基线 Δ (percentage points):')
    print(f'{"variant":<20} {"ΔEM":>6} {"ΔF1":>6} {"ΔSF1":>6} {"ΔCOV":>6}')
    for name in VARIANTS[1:]:
        m = all_m[name]
        if not m:
            continue
        print(f'{DISPLAY[name]:<20} '
              f'{(m["EM"]-base["EM"])*100:>+5.1f} {(m["F1"]-base["F1"])*100:>+5.1f} '
              f'{(m["SF1"]-base["SF1"])*100:>+5.1f} '
              f'{(m["COV"]-base["COV"])*100:>+5.1f}')

    # 分类型（基线 + 最优变体）
    print()
    types = ['comparison', 'inference', 'compositional', 'bridge-comparison']
    best = max(VARIANTS[1:], key=lambda n: all_m[n]['F1'] if all_m[n] else 0)
    print(f'分类型 F1（baseline vs 最优变体 {best}）:')
    rows_of = load_rows('ours_full')
    rows_bf = load_rows(best)
    from collections import defaultdict
    t_of, t_bf = defaultdict(list), defaultdict(list)
    for r in rows_of:
        t_of[r.get('gold_type', '')].append(r)
    for r in rows_bf:
        t_bf[r.get('gold_type', '')].append(r)
    for t in types:
        def f1s(rs):
            if not rs:
                return 0.0
            return sum(f1_score(normalize_answer(r['prediction']),
                                normalize_answer(r['gold_answer'])) for r in rs) / len(rs)
        print(f'  {t:<18} Ours={f1s(t_of[t]):.3f}  {best}={f1s(t_bf[t]):.3f}')


if __name__ == '__main__':
    main()
