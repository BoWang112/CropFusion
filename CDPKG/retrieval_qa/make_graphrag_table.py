# -*- coding: utf-8 -*-
"""
make_graphrag_table.py —— GraphRAG 对比实验汇总表（表 A/B/C）
================================================================
表A：现有 480 局部题 —— GraphRAG-Local vs full vs graph-only
表B：新增 89 全局题 —— GraphRAG-Global vs full
表C：能力-查询类型矩阵

读取 data/results/qa_summary_*.json，输出对比表。
"""
import os
import json

RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'results')
METRICS = [
    ('bleu1', 'BLEU-1'), ('bleu2', 'BLEU-2'), ('bleu4', 'BLEU-4'),
    ('rouge1', 'ROUGE-1'), ('rouge2', 'ROUGE-2'), ('rouge_l', 'ROUGE-L'),
    ('bertscore', 'BERTScore'),
]


def load(variant, dataset='local'):
    tag = '' if dataset == 'local' else '_global'
    p = os.path.join(RESULT_DIR, f'qa_summary_{variant}{tag}.json')
    if os.path.exists(p):
        with open(p, encoding='utf-8') as f:
            return json.load(f)
    return None


def pct(base, other):
    return (base - other) / other * 100 if other else 0.0


def print_table(title, rows, full_name, dataset):
    """rows: list[(name, summary_dict)]；full_name 为基准"""
    print(f'\n=== {title} ===')
    print(f'{"方法":<20}' + ''.join(f'{l:>11}' for _, l in METRICS))
    base = load(full_name, dataset)
    if base:
        print(f'{full_name:<20}' + ''.join(
            f'{base[m]:>11.4f}' for m, _ in METRICS))
    for name, s in rows:
        line = f'{name:<20}'
        for m, l in METRICS:
            if s and m in s:
                rel = pct(base[m], s[m]) if base else 0
                mark = '✓' if rel >= 2 else (' ' if rel > 0 else '↓')
                line += f'{s[m]:>7.4f}({rel:+5.1f}%{mark})'
            else:
                line += f'{"-":>11}'
        print(line)


def main():
    # 表A：480 局部题
    full = load('full', 'local')
    gl = load('grag-local', 'local')
    go = load('graph-only', 'local')
    print_table('表A：480 局部题（GraphRAG-Local vs full vs graph-only）',
                [('GraphRAG-Local', gl), ('graph-only', go)], 'full', 'local')

    # 表B：89 全局题
    full_g = load('full', 'global')
    gg = load('grag-global', 'global')
    print_table('表B：89 全局题（GraphRAG-Global vs full）',
                [('GraphRAG-Global', gg)], 'full', 'global')

    # 表C：能力矩阵
    print('\n=== 表C：能力-查询类型矩阵（相对提升%） ===')
    print(f'{"查询类型":<16}{"评测集":<8}{"full":>12}{"GraphRAG":>14}')
    rows = []
    if full and gl:
        b4 = full['bleu4']; g = gl['bleu4']
        rows.append(('局部精确', '480', f'{b4:.4f}', f'{g:.4f} ({pct(b4,g):+.1f}%)'))
    if full_g and gg:
        b4 = full_g['bleu4']; g = gg['bleu4']
        rows.append(('全局聚合', '89', f'{b4:.4f}', f'{g:.4f} ({pct(b4,g):+.1f}%)'))
    for r in rows:
        print(f'{r[0]:<16}{r[1]:<8}{r[2]:>12}{r[3]:>14}')


if __name__ == '__main__':
    main()
