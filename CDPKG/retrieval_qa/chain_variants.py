# -*- coding: utf-8 -*-
"""chain_variants.py —— 串行跑完 9 个图谱侧变体（单进程，避免重复实例冲突）"""
import os
import sys
import time

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON = os.path.join(THIS_DIR, '..', '.venv-re', 'Scripts', 'python.exe')
SCRIPT = os.path.join(THIS_DIR, 'run_eval_resumable.py')

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

def main():
    # 确保没有其他评测进程残留
    import subprocess
    for i, variant in enumerate(VARIANTS, start=1):
        print(f'[{i}/{len(VARIANTS)}] 开始 {variant} ...', flush=True)
        t0 = time.time()
        env = dict(os.environ)
        env['BGE_MODEL'] = os.path.join(THIS_DIR, '..', 'models', 'bge-large-zh-v1.5')
        cmd = [PYTHON, '-X', 'utf8', '-u', SCRIPT, f'--variant={variant}']
        proc = subprocess.run(cmd, cwd=THIS_DIR, env=env)
        elapsed = (time.time() - t0) / 60
        if proc.returncode != 0:
            print(f'  !! {variant} 异常 (exit={proc.returncode})', flush=True)
        else:
            print(f'  {variant} 完成，耗时 {elapsed:.0f} 分钟', flush=True)
    print('全部变体完成', flush=True)

if __name__ == '__main__':
    main()
