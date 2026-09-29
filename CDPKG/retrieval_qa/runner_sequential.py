# -*- coding: utf-8 -*-
"""runner_sequential.py —— 无人值守串行评测监控器
====================================================
- 检测当前变体进度文件是否完成（480 条 + BERTScore 非空）
- 完成后启动下一个变体
- 全部 9 个变体完成后退出

用法（后台）：
  .venv-re/Scripts/python.exe -X utf8 -u runner_sequential.py
"""
import os
import sys
import time
import subprocess
import json

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON = os.path.join(THIS_DIR, '..', '.venv-re', 'Scripts', 'python.exe')
SCRIPT = os.path.join(THIS_DIR, 'run_eval_resumable.py')
RESULT_DIR = os.path.join(THIS_DIR, 'data', 'results')
MODEL = os.path.join(THIS_DIR, '..', 'models', 'bge-large-zh-v1.5')

VARIANTS = [
    'full',
    '-no-text-rrf',
    '-no-graph-rerank',
    '-no-semantic-graph',
    '-no-entity-graph',
    '-no-evidence-prompt',
    '-no-fewshot',
    '-no-diag-constraints',
    'graph-only',
]
TOTAL = 480


def variant_done(variant):
    """进度文件含 480 条且 BERTScore 全部算完"""
    pf = os.path.join(RESULT_DIR, f'qa_progress_{variant}.json')
    if not os.path.exists(pf):
        return False
    try:
        with open(pf, encoding='utf-8') as f:
            done = json.load(f)
    except Exception:
        return False
    if len(done) < TOTAL:
        return False
    if any(v.get('bertscore') is None for v in done.values()):
        return False
    return True


def progress_count(variant):
    pf = os.path.join(RESULT_DIR, f'qa_progress_{variant}.json')
    if not os.path.exists(pf):
        return 0
    try:
        with open(pf, encoding='utf-8') as f:
            return len(json.load(f))
    except Exception:
        return 0


def main():
    for variant in VARIANTS:
        if variant_done(variant):
            print(f'[{variant}] 已完成，跳过', flush=True)
            continue
        print(f'[{variant}] 开始（当前进度 {progress_count(variant)}/{TOTAL}）...', flush=True)
        env = dict(os.environ)
        env['BGE_MODEL'] = MODEL
        proc = subprocess.Popen(
            [PYTHON, '-X', 'utf8', '-u', SCRIPT, f'--variant={variant}'],
            cwd=THIS_DIR, env=env,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        # 轮询等待该变体完成
        while True:
            time.sleep(60)
            if proc.poll() is not None:
                break
            # 进程还活着但长时间无进展则打印提示
            print(f'  [{variant}] 进度 {progress_count(variant)}/{TOTAL}', flush=True)
        if variant_done(variant):
            print(f'  [{variant}] 完成', flush=True)
        else:
            print(f'  [{variant}] 异常退出 (code={proc.returncode})', flush=True)
    print('ALL_DONE', flush=True)


if __name__ == '__main__':
    main()
