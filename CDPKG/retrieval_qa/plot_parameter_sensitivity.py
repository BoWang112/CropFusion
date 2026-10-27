# -*- coding: utf-8 -*-
"""
plot_parameter_sensitivity.py —— 四组参数敏感性图
====================================================
(a) 图谱扩展跳数 × 候选锚点数 → 热力图（1x3，F1@3/NDCG@3/BERTScore）
(b) 图谱重排序权重 λ → 折线图
(c) 文本/图谱证据数量 → 折线图（text vs graph 对照，1x3）
(d) 动态少样本数量 → 折线图

输出到 CDPKG-LLM/parameter_sensitivity_charts/
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(BASE))
XLSX = os.path.join(ROOT, 'parameter_sensitivity_four_tables.xlsx')
OUT = os.path.join(ROOT, 'parameter_sensitivity_charts')
os.makedirs(OUT, exist_ok=True)

# 分类色板（light，8 槽，验证过）
CAT = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300']
# 本文方法强调色（主墨黑）
OURS = '#0b0b0b'
# 最优值标注色（status-critical，与分类色区分）
OPT = '#c0392b'
# Sequential blue（热力图）
SEQ = LinearSegmentedColormap.from_list(
    'seq_blue', ['#ffffff', '#cde2fb', '#86b6ef', '#3987e5', '#1c5cab', '#104281'])
SURFACE = '#ffffff'
INK = '#0b0b0b'
MUTED = '#52514e'
GRID = '#e1e0d9'

plt.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE,
    'axes.edgecolor': MUTED, 'text.color': INK, 'axes.labelcolor': INK,
    'xtick.color': MUTED, 'ytick.color': MUTED,
    'font.family': 'sans-serif', 'font.size': 10,
    'axes.titlesize': 11.5, 'axes.labelsize': 11, 'legend.fontsize': 9,
    'axes.linewidth': 0.8,
})


def style_ax(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


# ---------------- (a) 3D surface ----------------
def plot_surface(rows):
    """(a) h × Ka 联合效应 → 单张 3D surface（NDCG@3 检索质量主指标）"""
    hops = sorted({r[0] for r in rows})
    kas = sorted({r[1] for r in rows})
    Z = np.zeros((len(hops), len(kas)))
    for r in rows:
        Z[hops.index(r[0]), kas.index(r[1])] = r[3]  # NDCG@3
    X, Y = np.meshgrid(kas, hops)
    fig = plt.figure(figsize=(9, 6.4), dpi=300)
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap=SEQ, edgecolor='none',
                           linewidth=0, antialiased=True, alpha=0.95)
    # 最优点标注（h=2, Ka=20）
    bi, bj = int(np.argmax(Z) / len(kas)), np.argmax(Z) % len(kas)
    ax.scatter(kas[bj], hops[bi], Z[bi, bj] + 0.2, color=OPT, s=60,
               depthshade=False, zorder=5)
    ax.text(kas[bj], hops[bi], Z[bi, bj] + 1.0, f'Optimal\n(h={hops[bi]}, $K_a$={kas[bj]})',
            ha='center', fontsize=9, fontweight='bold', color=OPT)
    ax.set_xlabel('Candidate anchors $K_a$', fontsize=11, labelpad=8)
    ax.set_ylabel('Graph hops $h$', fontsize=11, labelpad=8)
    ax.set_xticks(kas)
    ax.set_yticks(hops)
    ax.view_init(elev=25, azim=-135)
    fig.colorbar(surf, ax=ax, shrink=0.6, aspect=12, pad=0.08,
                 label='NDCG@3 (%)')
    fig.savefig(os.path.join(OUT, 'a_graph_expansion_surface.png'), dpi=300,
                facecolor=SURFACE, bbox_inches='tight')
    plt.close(fig)
    print('  已保存: a_graph_expansion_surface.png')


# ---------------- (b)(d) 通用折线 ----------------
def plot_lines(rows, x_label, x_vals, metrics, title, fname, x_idx=0,
               markers=None, optimal_x=None, opt_label=None):
    # rows: [(x, m1, m2, ...), ...]
    if markers is None:
        markers = ['o', 's', 'D', '^', 'v', 'P']
    fig, ax = plt.subplots(figsize=(8.5, 5.6), dpi=300)
    fig.subplots_adjust(left=0.11, right=0.97, top=0.97, bottom=0.18)
    ax.yaxis.grid(True, color=GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    style_ax(ax)
    xs = [r[x_idx] for r in rows]
    for mi, (lab, idx) in enumerate(metrics):
        ys = [r[idx] for r in rows]
        c = CAT[mi % len(CAT)]
        ax.plot(xs, ys, color=c, linewidth=1.8, marker=markers[mi],
                markersize=6, zorder=3, label=lab)
        # 数值标注：白色描边避免与线/网格重叠
        for xi, yi in zip(xs, ys):
            ax.annotate(f'{yi:.1f}', (xi, yi), textcoords='offset points',
                        xytext=(0, 8), ha='center', fontsize=7.5, color=INK,
                        path_effects=[pe.withStroke(linewidth=2.5,
                                                    foreground=SURFACE)],
                        zorder=4)
    # 最优参数竖线标注（框放在 x 轴下方图外，避免遮挡折点数值）
    if optimal_x is not None:
        ax.axvline(optimal_x, color=OPT, linewidth=1.4, linestyle='--',
                   zorder=2, alpha=0.85)
        ax.annotate(opt_label or f'Optimal: {optimal_x}',
                    (optimal_x, 0), xycoords=('data', 'axes fraction'),
                    xytext=(0, -18), textcoords='offset points',
                    ha='center', fontsize=10, fontweight='bold', color=OPT,
                    bbox=dict(boxstyle='round,pad=0.3', fc=SURFACE,
                              ec=OPT, lw=1.2),
                    zorder=5)
    ax.set_xlabel(x_label)
    ax.set_ylabel('Score (%)')
    lo = max(0, min(min(r[idx] for r in rows) for _, idx in metrics) - 3.0)
    hi = max(max(r[idx] for r in rows) for _, idx in metrics) + 3.0
    ax.set_ylim(lo, hi)
    # 图例放图内右侧中部（数据在 60-79 之间有空白带，正好落空区）
    ax.legend(loc='center right', ncol=1, frameon=True, framealpha=0.85,
              edgecolor=MUTED, borderpad=0.5, labelspacing=0.35)
    fig.subplots_adjust(right=0.97)
    fig.savefig(os.path.join(OUT, fname), dpi=300, facecolor=SURFACE,
                bbox_inches='tight')
    plt.close(fig)
    print(f'  已保存: {fname}')


# ---------------- (c) 文本 vs 图谱证据（单张折线图） ----------------
def plot_evidence(rows):
    text_rows = [r for r in rows if r[0] == 'Text evidence']
    graph_rows = [r for r in rows if r[0] == 'Graph evidence']
    metrics = [('BLEU-4 (%)', 2, 'BLEU-4'), ('ROUGE-L (%)', 3, 'ROUGE-L'),
               ('BERTScore-F1 (%)', 4, 'BERTScore-F1')]
    fig, ax = plt.subplots(figsize=(9, 5.8), dpi=300)
    fig.subplots_adjust(left=0.11, right=0.97, top=0.97, bottom=0.20)
    ax.yaxis.grid(True, color=GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    style_ax(ax)
    markers = ['o', 's', 'D']
    allv = []
    # 文本证据：3 指标实线（蓝系）
    for mi, (lab, idx, short) in enumerate(metrics):
        tx = [r[1] for r in text_rows]; ty = [r[idx] for r in text_rows]
        allv += ty
        ax.plot(tx, ty, color=CAT[mi], linewidth=1.8, marker=markers[mi],
                markersize=6, zorder=3, label=f'{short} (Text)')
        for xi, yi in zip(tx, ty):
            ax.annotate(f'{yi:.1f}', (xi, yi), textcoords='offset points',
                        xytext=(0, 8), ha='center', fontsize=7, color=CAT[mi],
                        path_effects=[pe.withStroke(linewidth=2.2,
                                                    foreground=SURFACE)],
                        zorder=4)
    # 图谱证据：3 指标虚线（灰系，区分 text/graph）
    for mi, (lab, idx, short) in enumerate(metrics):
        gx = [r[1] for r in graph_rows]; gy = [r[idx] for r in graph_rows]
        allv += gy
        ax.plot(gx, gy, color=CAT[mi], linewidth=1.8, marker=markers[mi],
                markersize=6, linestyle='--', zorder=3,
                label=f'{short} (Graph)')
        for xi, yi in zip(gx, gy):
            ax.annotate(f'{yi:.1f}', (xi, yi), textcoords='offset points',
                        xytext=(0, -13), ha='center', fontsize=7, color=CAT[mi],
                        path_effects=[pe.withStroke(linewidth=2.2,
                                                    foreground=SURFACE)],
                        zorder=4)
    # 最优位置竖线：text 最优/ graph 最优（框放图外下方）
    t_opt = text_rows[np.argmax([r[2] for r in text_rows])][1]
    g_opt = graph_rows[np.argmax([r[2] for r in graph_rows])][1]
    ax.axvline(t_opt, color=CAT[0], linewidth=1.2, linestyle='--', zorder=2, alpha=0.6)
    ax.axvline(g_opt, color=CAT[3], linewidth=1.2, linestyle=':', zorder=2, alpha=0.6)
    ax.annotate(f'T*={t_opt:.0f}', (t_opt, 0), xycoords=('data', 'axes fraction'),
                xytext=(0, -16), textcoords='offset points', ha='center',
                fontsize=9.5, fontweight='bold', color=CAT[0],
                bbox=dict(boxstyle='round,pad=0.3', fc=SURFACE, ec=CAT[0], lw=1.2),
                zorder=5)
    ax.annotate(f'G*={g_opt:.0f}', (g_opt, 0), xycoords=('data', 'axes fraction'),
                xytext=(0, -28), textcoords='offset points', ha='center',
                fontsize=9.5, fontweight='bold', color=CAT[3],
                bbox=dict(boxstyle='round,pad=0.3', fc=SURFACE, ec=CAT[3], lw=1.2),
                zorder=5)
    ax.set_xlabel('Number of evidence')
    ax.set_ylabel('Score (%)')
    ax.set_ylim(max(0, min(allv) - 3.0), max(allv) + 3.0)
    # 图例放图内右侧中部（双列，落在 BLEU-4 与 BERTScore 之间的空白带）
    ax.legend(loc='center right', ncol=2, frameon=True, framealpha=0.85,
              edgecolor=MUTED, fontsize=8, borderpad=0.4, labelspacing=0.25,
              handlelength=1.6, columnspacing=0.8)
    fig.subplots_adjust(right=0.97)
    fig.savefig(os.path.join(OUT, 'c_evidence_numbers.png'), dpi=300,
                facecolor=SURFACE, bbox_inches='tight')
    plt.close(fig)
    print('  已保存: c_evidence_numbers.png')


def main():
    import openpyxl
    wb = openpyxl.load_workbook(XLSX, data_only=True)

    # (a) Graph expansion
    ws = wb['Graph expansion']
    rows = [r for r in ws.iter_rows(values_only=True)
            if r[0] is not None and isinstance(r[0], (int, float))]
    print('绘制 (a) 3D surface...')
    plot_surface(rows)

    # (b) Graph reranking
    ws = wb['Graph reranking']
    rows = [r for r in ws.iter_rows(values_only=True)
            if r[0] is not None and isinstance(r[0], (int, float))]
    print('绘制 (b) 折线图...')
    plot_lines(rows, 'Re-ranking weight $\\lambda$', None,
               [('F1@3 (%)', 1), ('NDCG@3 (%)', 2), ('BLEU-4 (%)', 3),
                ('ROUGE-L (%)', 4), ('BERTScore-F1 (%)', 5)],
               '(b) Effect of graph re-ranking weight $\\lambda$',
               'b_graph_reranking_lambda.png', optimal_x=0.4,
               opt_label='Optimal $\\lambda$ = 0.4')

    # (c) Evidence numbers（第一列是文本类型，非数字，用第二列判断）
    ws = wb['Evidence numbers']
    rows = [r for r in ws.iter_rows(values_only=True)
            if r[1] is not None and isinstance(r[1], (int, float))]
    print('绘制 (c) 折线图...')
    plot_evidence(rows)

    # (d) Few-shot number
    ws = wb['Few-shot number']
    rows = [r for r in ws.iter_rows(values_only=True)
            if r[0] is not None and isinstance(r[0], (int, float))]
    print('绘制 (d) 折线图...')
    plot_lines(rows, 'Number of few-shot examples $K_f$', None,
               [('BLEU-4 (%)', 1), ('ROUGE-L (%)', 2), ('BERTScore-F1 (%)', 3)],
               '(d) Effect of dynamic few-shot example number',
               'd_fewshot_number.png', optimal_x=3,
               opt_label='Optimal $K_f$ = 3')
    print('\n全部完成 ->', OUT)


if __name__ == '__main__':
    main()
