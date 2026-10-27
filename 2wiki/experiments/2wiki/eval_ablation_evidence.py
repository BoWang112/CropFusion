# -*- coding: utf-8 -*-
"""
eval_ablation_evidence.py —— 消融变体证据构成对比
================================================
统计每个变体的 graph/text 单位数分布、token 占比、显式实体锚点留存率。
"""
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import PRED

VARIANTS = ['ours_full', 'anchor_retain', 'text_quota',
            'entity_first', 'sentence_ev', 'v2',
            'anchor_fs0', 'no_rerank', 'no_rerank_fs0',
            'pure_graph8', 'sentence_gated8']
DISPLAY = {
    'ours_full': 'Ours-Full(baseline)',
    'anchor_retain': 'Fix1 anchor_retain',
    'text_quota': 'Fix2 text_quota',
    'entity_first': 'Fix3 entity_first',
    'sentence_ev': 'Fix4 sentence_ev',
    'v2': 'All (v2)',
    'anchor_fs0': 'anchor_retain+noFS',
    'no_rerank': 'anchor_retain+noRerank',
    'no_rerank_fs0': 'noRerank+noFS',
    'pure_graph8': 'pure_graph8',
    'sentence_gated8': 'sentence_gated8',
}


def load(name):
    p = os.path.join(PRED, f'{name}.jsonl')
    if not os.path.exists(p):
        return []
    return [json.loads(l) for l in open(p, encoding='utf-8')]


def approx_tok(s):
    return len(s.split())


def is_text(e):
    """文本证据 source 为 dense/bm25/rrf；图内句子(graph_sent)也算句子"""
    return e.get('source') in ('dense', 'bm25', 'rrf', 'graph_sent')


def is_graph(e):
    return e.get('source') == 'graph'


def main():
    print(f'{"variant":<22} {"g5/t3":>6} {"g3/t5":>6} {"t=0":>5} '
          f'{"avg_g":>5} {"avg_t":>5} {"g_tok%":>6} {"ev_units":>8}')
    for name in VARIANTS:
        rows = load(name)
        if not rows:
            print(f'{DISPLAY[name]:<22} (无预测)')
            continue
        n = len(rows)
        g5t3 = sum(1 for r in rows
                   if sum(1 for e in r['retrieved_evidence'] if is_graph(e)) == 5
                   and sum(1 for e in r['retrieved_evidence'] if is_text(e)) == 3)
        g3t5 = sum(1 for r in rows
                   if sum(1 for e in r['retrieved_evidence'] if is_graph(e)) == 3
                   and sum(1 for e in r['retrieved_evidence'] if is_text(e)) == 5)
        t0 = sum(1 for r in rows if sum(1 for e in r['retrieved_evidence'] if is_text(e)) == 0)
        g_cnt = [sum(1 for e in r['retrieved_evidence'] if is_graph(e)) for r in rows]
        t_cnt = [sum(1 for e in r['retrieved_evidence'] if is_text(e)) for r in rows]
        gt = sum(approx_tok(e['text']) for r in rows for e in r['retrieved_evidence'] if is_graph(e))
        tt = sum(approx_tok(e['text']) for r in rows for e in r['retrieved_evidence'] if is_text(e))
        units = [len(r['retrieved_evidence']) for r in rows]
        print(f'{DISPLAY[name]:<22} {g5t3:>6} {g3t5:>6} {t0:>5} '
              f'{sum(g_cnt)/n:>5.1f} {sum(t_cnt)/n:>5.1f} '
              f'{gt/(gt+tt)*100:>5.0f}% {sum(units)/n:>8.2f}')


if __name__ == '__main__':
    main()
