# -*- coding: utf-8 -*-
"""
plot_results.py —— 结果表 → 图表
====================================================
输入：一张含 Overall / Task-wise 两个 sheet 的 xlsx（见 --xlsx）
  Figure 1 (Overall)    : 分组柱状图（每方法 2 柱 = Answer EM / Answer F1）
  Figure 2 (Task-wise)  : 4 张独立折线图（每任务一张，EM/F1 两线，图例左上）
输出：300 DPI PNG（见 --out）

用法：
  python plot_results.py --xlsx <结果表.xlsx> [--out <输出目录>]
"""
import os
import argparse
import openpyxl
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(BASE)))   # 仓库根目录

XLSX = os.path.join(ROOT, 'results.xlsx')   # 默认值，可用 --xlsx 覆盖
OUT = os.path.join(ROOT, 'charts')          # 默认值，可用 --out 覆盖

# ---------- 主题 ----------
INK = '#0b0b0b'
MUTED = '#52514e'
GRID = '#e2e1da'
EM_C = '#2a78d6'    # Answer EM 蓝
F1_C = '#eb6834'    # Answer F1 橙
TASK_C = ['#2a78d6', '#eb6834', '#1baf7a', '#4a3aa7']

plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.edgecolor': MUTED,
    'text.color': INK,
    'axes.labelcolor': INK,
    'xtick.color': MUTED,
    'ytick.color': MUTED,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Microsoft YaHei', 'SimHei'],
    'axes.unicode_minus': False,
    'font.size': 10,
    'axes.labelsize': 11,
    'legend.fontsize': 8.5,
    'axes.linewidth': 0.8,
})


def load_data():
    wb = openpyxl.load_workbook(XLSX, data_only=True)
    ov = [r for r in wb['Overall'].iter_rows(values_only=True) if r[0]]
    tw = [r for r in wb['Task-wise'].iter_rows(values_only=True) if r[0]]
    methods = [str(r[0]) for r in ov[1:]]
    em = [float(r[1]) for r in ov[1:]]
    f1 = [float(r[2]) for r in ov[1:]]
    # task-wise 列：0 Method, 1-2 Comp, 3-4 Inf, 5-6 Comp*, 7-8 Bridge（EM,F1 交替）
    tasks_em = {
        'Comparison':       [float(r[1]) for r in tw[1:]],
        'Inference':        [float(r[3]) for r in tw[1:]],
        'Compositional':    [float(r[5]) for r in tw[1:]],
        'Bridge-comparison': [float(r[7]) for r in tw[1:]],
    }
    tasks_f1 = {
        'Comparison':       [float(r[2]) for r in tw[1:]],
        'Inference':        [float(r[4]) for r in tw[1:]],
        'Compositional':    [float(r[6]) for r in tw[1:]],
        'Bridge-comparison': [float(r[8]) for r in tw[1:]],
    }
    return methods, em, f1, tasks_em, tasks_f1


def figure1(methods, em, f1):
    """纯分组柱状图（全英文）：每方法两根柱（Answer EM / Answer F1），图例左上"""
    fig, ax = plt.subplots(figsize=(8.6, 4.8), dpi=300)
    x = np.arange(len(methods))
    bw = 0.34
    for off, vals, c, lab in [(-bw / 2, em, EM_C, 'Answer EM'),
                              (+bw / 2, f1, F1_C, 'Answer F1')]:
        ax.bar(x + off, vals, width=bw, color=c, alpha=0.9, label=lab, zorder=2)
        for xi, yi in zip(x, vals):
            ax.annotate(f'{yi:.1f}', (xi + off, yi), textcoords='offset points',
                        xytext=(0, 2), ha='center', fontsize=7.2, color=c,
                        bbox=dict(facecolor='white', edgecolor='none',
                                  alpha=0.85, pad=0.15))
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=26, ha='right', fontsize=8)
    ax.set_ylim(0, 92)
    ax.set_ylabel('Score (%)')
    ax.set_xlabel('Method')
    ax.grid(axis='y', color=GRID, linewidth=0.7, zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='upper left', frameon=False)
    fig.tight_layout()
    p = os.path.join(OUT, 'fig1_overall_bars.png')
    fig.savefig(p, bbox_inches='tight')
    plt.close(fig)
    print('saved', p)


def figure2(methods, tasks_em, tasks_f1):
    """4 张独立折线图（全英文）：每任务一张，Answer EM / F1 两线，
    数值标签带白底（不与线/彼此重叠），图例统一左上角"""
    for tname in tasks_f1.keys():
        fig, ax = plt.subplots(figsize=(8.6, 4.8), dpi=300)
        x = np.arange(len(methods))
        em_vals, f1_vals = tasks_em[tname], tasks_f1[tname]
        ax.plot(x, em_vals, color=EM_C, marker='o', markersize=5.2,
                linewidth=1.8, label='Answer EM', zorder=3)
        ax.plot(x, f1_vals, color=F1_C, marker='o', markersize=5.2,
                linewidth=1.8, label='Answer F1', zorder=3)
        # EM 标签在点下方，F1 标签在点上方，白底避免压线/互叠
        for xi, yi in zip(x, em_vals):
            ax.annotate(f'{yi:.1f}', (xi, yi), textcoords='offset points',
                        xytext=(0, -4), ha='center', fontsize=7, color=EM_C,
                        bbox=dict(facecolor='white', edgecolor='none',
                                  alpha=0.9, pad=0.1))
        for xi, yi in zip(x, f1_vals):
            ax.annotate(f'{yi:.1f}', (xi, yi), textcoords='offset points',
                        xytext=(0, 4), ha='center', fontsize=7, color=F1_C,
                        bbox=dict(facecolor='white', edgecolor='none',
                                  alpha=0.9, pad=0.1))
        ax.set_title(tname, fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=26, ha='right', fontsize=8)
        ax.set_ylim(0, 100)
        ax.set_ylabel('Score (%)')
        ax.set_xlabel('Method')
        ax.grid(axis='y', color=GRID, linewidth=0.7, zorder=0)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(loc='upper left', frameon=True, fontsize=8.5,
                  facecolor='white', edgecolor='none')
        fig.tight_layout()
        fname = tname.lower().replace(' ', '_').replace('-', '_')
        p = os.path.join(OUT, f'fig2_{fname}.png')
        fig.savefig(p, bbox_inches='tight')
        plt.close(fig)
        print('saved', p)


def lighten(hexc, f=0.45):
    """把颜色变浅（混白）供 EM 柱使用"""
    h = hexc.lstrip('#')
    r, g, b = [int(h[i:i + 2], 16) for i in (0, 2, 4)]
    return f'#{int(r + (255 - r) * f):02x}{int(g + (255 - g) * f):02x}{int(b + (255 - b) * f):02x}'


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--xlsx', default=XLSX,
                    help='结果表 xlsx 路径（需含 Overall / Task-wise 两个 sheet）')
    ap.add_argument('--out', default=OUT, help='图表输出目录')
    args = ap.parse_args()
    XLSX, OUT = args.xlsx, args.out
    os.makedirs(OUT, exist_ok=True)

    methods, em, f1, tasks_em, tasks_f1 = load_data()
    print('methods:', methods)
    figure1(methods, em, f1)
    figure2(methods, tasks_em, tasks_f1)
