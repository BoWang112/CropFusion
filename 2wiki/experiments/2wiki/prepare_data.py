# -*- coding: utf-8 -*-
"""
prepare_data.py —— 2Wiki 数据准备
=================================
- 从 development split 按 type 分层抽样测试样本（每类 TEST_PER_TYPE 条，seed=42）
- 从 train split 独立抽样 40 条 few-shot 池（每类 10，seed=42）
- 保存 sample.jsonl / sample_ids.json / fewshot_pool.jsonl

禁止数据泄漏：answer/supporting_facts/evidences/type 字段保留在原文件，
但检索与生成过程不读它们（本脚本只做抽样与序列化）。
"""
import sys
import os
import json
import random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (DATA, SAMPLE_FILE, IDS_FILE, FEWSHOT_FILE,
                    VAL_PARQUET, TRAIN_PARQUET, RANDOM_SEED,
                    TEST_TYPES, TEST_PER_TYPE, FEWSHOT_PER_TYPE,
                    normalize_type)


def load_parquet(path):
    import pandas as pd
    df = pd.read_parquet(path)
    rows = []
    for _, r in df.iterrows():
        rows.append({
            'id': str(r['id']),
            'question': str(r['question']),
            'answer': str(r['answer']),
            'type': normalize_type(r['type']),
            'evidences': r['evidences'],
            'supporting_facts': r['supporting_facts'],
            'context': r['context'],
        })
    return rows


def serialize_row(r):
    """序列化为 jsonl 行（numpy 数组转 list）"""
    import numpy as np
    sf = r['supporting_facts']
    ev = r['evidences']
    out = {
        'id': r['id'],
        'question': r['question'],
        'answer': r['answer'],
        'type': r['type'],
        'evidences': None,
        'supporting_facts': None,
        'context': {
            'title': list(r['context']['title']),
            'sentences': [list(s) for s in r['context']['sentences']],
        },
    }
    # evidences / supporting_facts 序列化（供评测与分析，不进入检索/生成）
    if ev is not None:
        out['evidences'] = [list(e) if hasattr(e, '__iter__') else e for e in ev]
    if sf is not None:
        out['supporting_facts'] = {
            'title': list(sf['title']),
            'sent_id': [int(x) for x in sf['sent_id']],
        }
    return out


def main():
    print('加载数据...')
    val = load_parquet(VAL_PARQUET)
    train = load_parquet(TRAIN_PARQUET)
    print(f'  val={len(val)}, train={len(train)}')

    # 按 type 分组
    def group_by_type(rows):
        g = {}
        for r in rows:
            g.setdefault(r['type'], []).append(r)
        return g

    val_g = group_by_type(val)
    train_g = group_by_type(train)

    # ---------- 测试样本（每类 TEST_PER_TYPE 条） ----------
    rng = random.Random(RANDOM_SEED)
    sample = []
    for t in TEST_TYPES:
        pool = val_g.get(t, [])
        if len(pool) < TEST_PER_TYPE:
            raise RuntimeError(f'type {t} 在 val 中仅 {len(pool)} 条，不足 {TEST_PER_TYPE}')
        chosen = rng.sample(pool, TEST_PER_TYPE)
        sample.extend(chosen)
    # 固定顺序：先按 type 打散，避免同 type 连续
    rng.shuffle(sample)

    with open(SAMPLE_FILE, 'w', encoding='utf-8') as f:
        for r in sample:
            f.write(json.dumps(serialize_row(r), ensure_ascii=False) + '\n')
    with open(IDS_FILE, 'w', encoding='utf-8') as f:
        json.dump([r['id'] for r in sample], f, ensure_ascii=False, indent=1)

    # ---------- 40 条 few-shot 池 ----------
    rng2 = random.Random(RANDOM_SEED)
    fs = []
    for t in TEST_TYPES:
        pool = train_g.get(t, [])
        if len(pool) < FEWSHOT_PER_TYPE:
            raise RuntimeError(f'type {t} 在 train 中仅 {len(pool)} 条，不足 {FEWSHOT_PER_TYPE}')
        chosen = rng2.sample(pool, FEWSHOT_PER_TYPE)
        fs.extend(chosen)
    rng2.shuffle(fs)

    with open(FEWSHOT_FILE, 'w', encoding='utf-8') as f:
        for r in fs:
            f.write(json.dumps(serialize_row(r), ensure_ascii=False) + '\n')

    # 统计
    from collections import Counter
    st = Counter(normalize_type(r['type']) for r in sample)
    ft = Counter(normalize_type(r['type']) for r in fs)
    print(f'测试样本: {len(sample)} 条')
    print(f'  type 分布: {dict(st)}')
    print(f'few-shot: {len(fs)} 条')
    print(f'  type 分布: {dict(ft)}')
    # 无 id 冲突
    sample_ids = {r['id'] for r in sample}
    fs_ids = {r['id'] for r in fs}
    assert not (sample_ids & fs_ids), '测试集与 few-shot 池存在 id 冲突！'
    print('id 冲突检查: PASS')
    print(f'已保存 {SAMPLE_FILE}')
    print(f'已保存 {IDS_FILE}')
    print(f'已保存 {FEWSHOT_FILE}')


if __name__ == '__main__':
    main()
