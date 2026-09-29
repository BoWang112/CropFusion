# -*- coding: utf-8 -*-
"""
run_all_variants.py —— 串行跑消融 + 基线对比
==============================================
依次对每个 variant 运行 run_eval_resumable.py（全量 480 条，真实 LLM）。
每个变体独立进度/结果文件；已完成的可跳过（断点续跑）。

运行方式（后台）：
  BGE_MODEL=../models/bge-large-zh-v1.5 \
    .venv-re/Scripts/python.exe -X utf8 -u run_all_variants.py

产出（每个变体）：
  data/results/qa_progress_<variant>.json   （逐条结果）
  data/results/qa_summary_<variant>.json    （指标汇总）
"""
import os
import sys
import json
import subprocess
import time

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON = os.path.join(THIS_DIR, '..', '.venv-re', 'Scripts', 'python.exe')
SCRIPT = os.path.join(THIS_DIR, 'run_eval_resumable.py')
RESULT_DIR = os.path.join(THIS_DIR, 'data', 'results')

# 变体顺序：full（完整方法，已跑过，会跳过）→ 7 个消融模块 → 3 个基线。
# 消融模块（对应论文表6）：
#   检索侧：-no-text-rrf / -no-graph-rerank / -no-semantic-graph / -no-entity-graph
#   提示侧：-no-evidence-prompt / -no-fewshot / -no-diag-constraints
# 基线：naive-text / hybrid-text / graph-only
VARIANTS = [
    'full',
    # 检索侧消融
    '-no-text-rrf',
    '-no-graph-rerank',
    '-no-semantic-graph',
    '-no-entity-graph',
    # 提示侧消融
    '-no-evidence-prompt',
    '-no-fewshot',
    '-no-diag-constraints',
    # 基线对比
    'naive-text',
    'hybrid-text',
    'graph-only',
]

TOTAL_EXPECTED = 480


def variant_done(variant):
    """该变体是否已全部评测完成（进度文件含 480 条且已算完 BERTScore）"""
    pf = os.path.join(RESULT_DIR, f'qa_progress_{variant}.json')
    if not os.path.exists(pf):
        return False
    try:
        with open(pf, encoding='utf-8') as f:
            done = json.load(f)
    except Exception:
        return False
    if len(done) < TOTAL_EXPECTED:
        return False
    # BERTScore 是否算完
    if any(v.get('bertscore') is None for v in done.values()):
        return False
    return True


def main():
    os.makedirs(RESULT_DIR, exist_ok=True)
    for i, variant in enumerate(VARIANTS, start=1):
        if variant_done(variant):
            print(f'[{i}/{len(VARIANTS)}] {variant}: 已完成，跳过', flush=True)
            continue
        print(f'[{i}/{len(VARIANTS)}] 开始 {variant} ...', flush=True)
        t0 = time.time()
        cmd = [PYTHON, '-X', 'utf8', '-u', SCRIPT,
               f'--variant={variant}']
        proc = subprocess.run(cmd, cwd=THIS_DIR)
        if proc.returncode != 0:
            print(f'  !! {variant} 运行异常 (exit={proc.returncode})，继续下一个', flush=True)
        else:
            elapsed = (time.time() - t0) / 60
            print(f'  {variant} 完成，耗时 {elapsed:.0f} 分钟', flush=True)


if __name__ == '__main__':
    main()
