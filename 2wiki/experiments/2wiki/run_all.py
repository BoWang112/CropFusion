# -*- coding: utf-8 -*-
"""
run_all.py —— 9 种方法全量运行
================================
按顺序运行 9 种方法（断点续跑：已有结果跳过，--force 重跑）。
用法：
  python run_all.py [--limit N] [--force]
"""
import sys
import os
import subprocess
import argparse

METHODS = ['llm_no_retrieval', 'naive_rag', 'hybrid_rag', 'graphrag',
           'lightrag', 'g_retriever', 'pathrag', 'sg_rag', 'ours_full']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=0, help='每方法仅跑前 N 条')
    ap.add_argument('--force', action='store_true', help='强制重跑')
    ap.add_argument('--method', default=None, help='只跑指定方法（默认全部）')
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    py = sys.executable  # 当前解释器（.venv-re/Scripts/python.exe）

    targets = [args.method] if args.method else METHODS
    for m in targets:
        cmd = [py, '-X', 'utf8', os.path.join(here, 'run_method.py'),
               '--method', m]
        if args.limit:
            cmd += ['--limit', str(args.limit)]
        if args.force:
            cmd += ['--force']
        print(f'\n========== {m} ==========', flush=True)
        subprocess.run(cmd)


if __name__ == '__main__':
    main()
