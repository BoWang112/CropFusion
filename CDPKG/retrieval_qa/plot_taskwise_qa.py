# -*- coding: utf-8 -*-
"""
plot_taskwise_qa.py —— 六类任务问答指标折线图
==============================================
读取 taskwise_qa_six_tables.xlsx（6 个 sheet = 六类任务），
每个任务画一张折线图：
  - 横轴：7 个指标（BLEU-1/2/4, ROUGE-1/2/L, BERTScore-F1）
  - 纵轴：分数（%）
  - 8 条基线线（验证过的分类色板）+ 本文方法 Ours-Full（主墨色黑粗线）
输出到独立文件夹 taskwise_charts/ 下的 6 张 300DPI PNG。
"""
import os
import sys
import openpyxl
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# 工作目录：本脚本所在目录（retrieval_qa/），xlsx 和输出都在 CDPKG-LLM 根目录
BASE = os.path.dirname(os.path.abspath(__file__))   # .../CDPKG/retrieval_qa
ROOT = os.path.dirname(os.path.dirname(BASE))        # .../CDPKG-LLM
XLSX = os.path.join(ROOT, 'taskwise_qa_six_tables.xlsx')
OUT = os.path.join(ROOT, 'taskwise_charts')
os.makedirs(OUT, exist_ok=True)

# 指标（横轴顺序）
METRICS = ['BLEU-1', 'BLEU-2', 'BLEU-4', 'ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'BERTScore-F1']

# 验证过的分类色板（light 模式，8 槽）—— 8 条基线
PALETTE = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100',
           '#e87ba4', '#008300', '#4a3aa7', '#e34948']
# 本文方法：主墨色黑（强强调，区别于分类色）
OURS_COLOR = '#0b0b0b'
OURS_LABEL = 'Ours-Full'

# 表面与墨色（light）
SURFACE = '#ffffff'
INK = '#0b0b0b'
MUTED = '#52514e'
GRID = '#e1e0d9'

plt.rcParams.update({
    'figure.facecolor': SURFACE,
    'axes.facecolor': SURFACE,
    'axes.edgecolor': MUTED,
    'text.color': INK,
    'axes.labelcolor': INK,
    'xtick.color': MUTED,
    'ytick.color': MUTED,
    'font.family': 'sans-serif',
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'legend.fontsize': 8.5,
    'axes.linewidth': 0.8,
})


def load_sheet(ws):
    """解析一个 sheet：返回 (title, methods[], data_matrix)"""
    rows = list(ws.iter_rows(values_only=True))
    title = rows[0][0] or ws.title
    # 跳过标题行（第0行标题、第1行空、第2行表头），数据从第3行开始
    methods = []
    data = []
    for r in rows[3:]:
        if r[0] is None:
            continue
        methods.append(str(r[0]).strip())
        data.append([float(x) for x in r[1:8]])
    return title, methods, np.array(data)


def plot_one(title, methods, data, fname):
    n_series = len(methods)
    x = np.arange(len(METRICS))
    fig, ax = plt.subplots(figsize=(9, 5.4), dpi=300)
    fig.subplots_adjust(left=0.10, right=0.97, top=0.97, bottom=0.10)

    # 网格：细、隐性（只画水平线）
    ax.yaxis.grid(True, color=GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # 8 条基线：分类色板薄线
    for i in range(n_series):
        is_ours = methods[i] == OURS_LABEL
        if is_ours:
            continue
        ax.plot(x, data[i], color=PALETTE[i % 8], linewidth=1.4,
                marker='o', markersize=4, zorder=3, label=methods[i])

    # 本文方法：黑粗线 + 大标记，最后画在最上层
    ours_idx = methods.index(OURS_LABEL) if OURS_LABEL in methods else -1
    if ours_idx >= 0:
        ax.plot(x, data[ours_idx], color=OURS_COLOR, linewidth=2.8,
                marker='D', markersize=7, zorder=5, label=OURS_LABEL)

    # 坐标轴：y 轴按数据范围自适应，放大曲线间差异
    ax.set_xticks(x)
    ax.set_xticklabels(METRICS)
    ax.set_xlabel('Evaluation metric')
    ax.set_ylabel('Score (%)')
    lo = np.min(data) - 3.0
    hi = np.max(data) + 3.0
    ax.set_ylim(max(0, lo), hi)

    # 图例放图内左上角双列（压缩高度，减少遮挡）
    ax.legend(loc='upper left', ncol=2, frameon=True, framealpha=0.92,
              edgecolor=MUTED, borderpad=0.5, labelspacing=0.3,
              columnspacing=0.8, handlelength=1.8, fontsize=8)

    fig.savefig(os.path.join(OUT, fname), dpi=300,
                facecolor=SURFACE, bbox_inches='tight')
    plt.close(fig)
    print(f'  已保存: {fname}')


def main():
    wb = openpyxl.load_workbook(XLSX, data_only=True)
    # 任务标题 → 文件名（英文名）
    title_map = {
        'Identification': '01_identification',
        'Symptom Analysis': '02_symptom_analysis',
        'Conditional Judgment': '03_conditional_judgment',
        'Prevention Control': '04_prevention_control',
        'Similar Identification': '05_similar_identification',
        'Multi-condition Reasoning': '06_multi_condition_reasoning',
    }
    print(f'输出目录: {OUT}')
    for ws in wb.worksheets:
        title, methods, data = load_sheet(ws)
        fname = title_map.get(ws.title, f'{len(title_map):02d}_{ws.title}') + '.png'
        print(f'\n绘制: {title}  ({len(methods)} 方法)')
        plot_one(title, methods, data, fname)
    print('\n全部完成')


if __name__ == '__main__':
    main()
