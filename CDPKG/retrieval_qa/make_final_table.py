# -*- coding: utf-8 -*-
"""
make_final_table.py —— 从 qa_summary_*.json 汇总最终论文表格
================================================================
读取 data/results/qa_summary_<variant>.json（重跑完成后生成），
输出：
  1) 对比实验表（full vs naive-text / hybrid-text / graph-only）
  2) 消融实验表（full vs 7 个消融）
每行含 8 个生成指标，并标注相对 full 的相对提升%。

用法：
  ../.venv-re/Scripts/python.exe -X utf8 make_final_table.py
"""
import os
import json

RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'results')

METRICS = [
    ('bleu1', 'BLEU-1'), ('bleu2', 'BLEU-2'), ('bleu4', 'BLEU-4'),
    ('rouge1', 'ROUGE-1'), ('rouge2', 'ROUGE-2'), ('rouge_l', 'ROUGE-L'),
    ('bertscore', 'BERTScore'), ('gscore', 'Gscore'),
]

VARIANTS_ORDER = [
    'full',
    '-no-text-rrf', '-no-graph-rerank', '-no-semantic-graph', '-no-entity-graph',
    '-no-evidence-prompt', '-no-fewshot', '-no-diag-constraints',
    'naive-text', 'hybrid-text', 'graph-only',
]


def load_summaries():
    out = {}
    for v in VARIANTS_ORDER:
        p = os.path.join(RESULT_DIR, f'qa_summary_{v}.json')
        if os.path.exists(p):
            with open(p, encoding='utf-8') as f:
                out[v] = json.load(f)
    return out


def pct(full_val, other_val):
    """相对提升 %（full 相对 other）"""
    if other_val == 0:
        return float('inf')
    return (full_val - other_val) / other_val * 100


def print_row(name, vals, full_vals, width=13):
    cells = [name.ljust(18)]
    for i, (key, _label) in enumerate(METRICS):
        v = vals.get(key, 0.0)
        if full_vals is not None and full_vals.get(key):
            rel = pct(full_vals[key], v)
            mark = '✓' if rel >= 2.0 else (' ' if rel >= 0 else '↓')
            cells.append(f'{v:.4f} ({rel:+.1f}%{mark})'.ljust(width + 8))
        else:
            cells.append(f'{v:.4f}'.ljust(width))
    print(' | '.join(cells).rstrip(' |'))


def main():
    s = load_summaries()
    missing = [v for v in VARIANTS_ORDER if v not in s]
    if missing:
        print(f'[警告] 缺少变体摘要: {missing}')
        print('重跑尚未完成，以下是已有变体：\n')

    full = s.get('full', {})
    header = '方法'.ljust(18) + ' | ' + ' | '.join(
        label.ljust(21) for _k, label in METRICS)
    print('=' * len(header))
    print('对比实验（full vs 基线）—— 括号内为 full 相对基线的提升%，✓=≥2%')
    print('=' * len(header))
    print(header)
    print('-' * len(header))
    for v in ['full', 'naive-text', 'hybrid-text', 'graph-only']:
        if v not in s:
            continue
        full_vals = full if v != 'full' else None
        print_row(v, s[v], full_vals)
    print()

    print('=' * len(header))
    print('消融实验（full vs 消融）—— 括号内为 full 相对消融的提升%，✓=≥2%')
    print('=' * len(header))
    print(header)
    print('-' * len(header))
    for v in ['full', '-no-text-rrf', '-no-graph-rerank', '-no-semantic-graph',
              '-no-entity-graph', '-no-evidence-prompt', '-no-fewshot',
              '-no-diag-constraints']:
        if v not in s:
            continue
        full_vals = full if v != 'full' else None
        print_row(v, s[v], full_vals)

    # 汇总达标检查
    print()
    print('=' * 70)
    print('达标检查（要求 ≥2%）')
    print('=' * 70)
    if full:
        for group_name, group in [
            ('对比基线', ['naive-text', 'hybrid-text', 'graph-only']),
            ('检索消融', ['-no-text-rrf', '-no-graph-rerank', '-no-semantic-graph', '-no-entity-graph']),
            ('提示消融', ['-no-evidence-prompt', '-no-fewshot', '-no-diag-constraints']),
        ]:
            print(f'\n[{group_name}]')
            for v in group:
                if v not in s:
                    print(f'  {v}: 缺失')
                    continue
                stats = []
                for key, label in METRICS:
                    if key not in s[v] or not s[v][key]:
                        continue
                    rel = pct(full[key], s[v][key])
                    stats.append((label, rel))
                over = [l for l, r in stats if r >= 2.0]
                under = [f'{l}={r:+.1f}%' for l, r in stats if r < 2.0]
                print(f'  {v}: 过线 {len(over)}/{len(stats)}  {", ".join(over) or "无"}')
                if under:
                    print(f'      未过线: {", ".join(under)}')


if __name__ == '__main__':
    main()
