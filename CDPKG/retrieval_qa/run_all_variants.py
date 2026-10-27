# -*- coding: utf-8 -*-
"""
run_all_variants.py —— 串行跑完整模型、七项消融和全部问答基线
================================================================
统一使用 run_eval_resumable.py 的新版评估协议：

- 完整模型：full
- 七项消融：-no-text-rrf / -no-graph-rerank / -no-semantic-graph /
  -no-entity-graph / -no-evidence-prompt / -no-fewshot / -no-diag-constraints
- 问答基线：naive-text / hybrid-text / graph-only / grag-local /
  lightrag / g-retriever / pathrag / sg-rag

所有变体使用同一数据集、同一 LLM provider 与同一生成参数；
消融只关闭指定模块。进度/汇总文件统一为 v2 命名，避免混用旧结果：

  data/results/qa_progress_v2_<variant>[_global][_provider].json
  data/results/qa_summary_v2_<variant>[_global][_provider].json

运行方式：
  .venv-re/Scripts/python.exe -X utf8 -u run_all_variants.py
  .venv-re/Scripts/python.exe -X utf8 -u run_all_variants.py --provider shennong
"""
import os
import sys
import json
import argparse
import subprocess
import time

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)

from config import (  # noqa: E402
    RESULT_DIR, EVAL_FILE, GLOBAL_QA_FILE, FORMAL_VARIANTS,
    EVALUATION_PIPELINE_VERSION, LLM_PROVIDERS,
)

PYTHON = os.path.join(THIS_DIR, '..', '.venv-re', 'Scripts', 'python.exe')
SCRIPT = os.path.join(THIS_DIR, 'run_eval_resumable.py')

VARIANTS = list(FORMAL_VARIANTS)


def _tags(provider, dataset):
    tag = '' if provider == 'deepseek' else f'_{provider}'
    dtag = '' if dataset == 'local' else f'_{dataset}'
    return dtag, tag


def progress_path(variant, provider, dataset):
    dtag, tag = _tags(provider, dataset)
    return os.path.join(
        RESULT_DIR,
        f'qa_progress_{EVALUATION_PIPELINE_VERSION}_{variant}{dtag}{tag}.json')


def expected_total(dataset):
    path = EVAL_FILE if dataset == 'local' else GLOBAL_QA_FILE
    with open(path, encoding='utf-8') as f:
        return len(json.load(f))


def variant_done(variant, provider, dataset, total_expected):
    """该变体是否已完成全部条目并算完 BERTScore。"""
    pf = progress_path(variant, provider, dataset)
    if not os.path.exists(pf):
        return False
    try:
        with open(pf, encoding='utf-8') as f:
            done = json.load(f)
    except Exception:
        return False
    if len(done) < total_expected:
        return False
    if any(v.get('bertscore') is None for v in done.values()):
        return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--provider', default='deepseek',
                    choices=sorted(LLM_PROVIDERS),
                    help='LLM provider（模型 ID 由对应环境变量传入）')
    ap.add_argument('--dataset', default='local', choices=['local', 'global'])
    args = ap.parse_args()

    os.makedirs(RESULT_DIR, exist_ok=True)
    total_expected = expected_total(args.dataset)
    print(f'数据集={args.dataset}（{total_expected} 条），'
          f'provider={args.provider}，变体数={len(VARIANTS)}', flush=True)

    for i, variant in enumerate(VARIANTS, start=1):
        if variant_done(variant, args.provider, args.dataset, total_expected):
            print(f'[{i}/{len(VARIANTS)}] {variant}: 已完成，跳过', flush=True)
            continue
        print(f'[{i}/{len(VARIANTS)}] 开始 {variant} ...', flush=True)
        t0 = time.time()
        cmd = [PYTHON, '-X', 'utf8', '-u', SCRIPT,
               f'--variant={variant}',
               f'--provider={args.provider}',
               f'--dataset={args.dataset}']
        proc = subprocess.run(cmd, cwd=THIS_DIR)
        if proc.returncode != 0:
            print(f'  !! {variant} 运行异常 (exit={proc.returncode})，'
                  f'继续下一个', flush=True)
        else:
            elapsed = (time.time() - t0) / 60
            print(f'  {variant} 完成，耗时 {elapsed:.0f} 分钟', flush=True)


if __name__ == '__main__':
    main()
