# -*- coding: utf-8 -*-
"""
generate_global_qa.py —— GraphRAG Phase 5：图谱驱动全局聚合题生成
==================================================================
从知识图谱统计事实自动生成"全局类"评测题（跨病虫害聚合查询），
供 GraphRAG Global Search 对比评测使用。

模板（每模板按作物聚合）：
  T1 传播方式:  "{crop}相关病害常见的传播方式有哪些？"
  T2 化学防治:  "{crop}类作物病虫害可采用的化学防治措施有哪些？"
  T3 农业防治:  "{crop}类作物病虫害可采用的农业防治措施有哪些？"
  T4 发病时期:  "{crop}的主要病害通常在哪些时期发生？"
  T5 危害部位:  "{crop}的病虫害主要危害哪些部位？"
  T6 典型症状:  "{crop}常见病害有哪些典型症状？"

参考答案 = 图谱统计事实去重列表（可对齐 n-gram 指标）。

产出：data/qa_global.json（约 90 条）
"""
import os
import sys
import json
import random
import hashlib
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import NODES_CSV, RELS_CSV, GLOBAL_QA_FILE  # noqa: E402
from modules.kg import CropPestKG  # noqa: E402

RANDOM_SEED = 2026
TOTAL_TARGET = 90
MAX_ITEMS = 10
MIN_VALUES = 3  # 每个 crop 在该关系上至少 3 个去重值才生成

# 模板定义：(模板id, 模板函数, 关系标签, 节点类型过滤)
TEMPLATES = {
    'spread': {
        'label': '传播方式',
        'q': lambda c: f'{c}相关病害常见的传播方式有哪些？',
        'rel': '传播',
        'tail_ents': {'病原体', '其他因素'},
        'ans_prefix': f'主要包括：',
        'spread_sources': True,  # 取传播边的 head（源头：灰飞/蚜虫/病土等）
    },
    'chem_ctrl': {
        'label': '化学防治',
        'q': lambda c: f'{c}类作物病虫害可采用的化学防治措施有哪些？',
        'rel': '防治方法有',
        'tail_ents': {'化学防治'},
        'ans_prefix': '化学防治措施主要包括：',
    },
    'agri_ctrl': {
        'label': '农业防治',
        'q': lambda c: f'{c}类作物病虫害可采用的农业防治措施有哪些？',
        'rel': '防治方法有',
        'tail_ents': {'农业防治'},
        'ans_prefix': '农业防治措施主要包括：',
    },
    'period': {
        'label': '发病时期',
        'q': lambda c: f'{c}的主要病害通常在哪些时期发生？',
        'rel': '发病于作物',
        'tail_ents': {'发病时期'},
        'ans_prefix': '主要包括：',
    },
    'site': {
        'label': '危害部位',
        'q': lambda c: f'{c}的病虫害主要危害哪些部位？',
        'rel': '为害',
        'tail_ents': {'发病部位'},
        'ans_prefix': '主要包括：',
    },
    'symptom': {
        'label': '典型症状',
        'q': lambda c: f'{c}常见病害有哪些典型症状？',
        'rel': '症状有',
        'tail_ents': {'具体症状'},
        'ans_prefix': '主要包括：',
    },
}


def _normalize(s):
    return s.replace(' ', '').strip()


def stable_id(*parts):
    raw = '||'.join(str(p) for p in parts)
    return 'gqa_' + hashlib.sha1(raw.encode('utf-8')).hexdigest()[:16]


def collect_crop_values(kg, crop_nid, rel, tail_ents, spread_sources=False):
    """收集某作物及其病虫害在某关系上的去重 tail 值"""
    values = []
    # 作物出边（如 作物-发病于作物? 实际作物是头）
    for tgt, r in kg.neighbors(crop_nid, rel):
        if kg.node_entity(tgt) in tail_ents:
            values.append(kg.node_name(tgt))
    # 作物关联的病虫害：水稻 -影响-> 稻瘟病（出边）
    pest_ids = []
    for tgt, r in kg.neighbors(crop_nid):
        if kg.node_entity(tgt) in ('病害', '虫害', '害虫'):
            pest_ids.append(tgt)
    # 补充：通过 in_neighbors（如 稻瘟病 -发病于作物-> 抽穗期 的反向）
    for src, r in kg.in_neighbors(crop_nid):
        if kg.node_entity(src) in ('病害', '虫害', '害虫'):
            pest_ids.append(src)
    # 部分关系是 病虫害→X，需先找到该作物的病虫害
    for pid in pest_ids:
        for tgt, r in kg.neighbors(pid, rel):
            if kg.node_entity(tgt) in tail_ents:
                values.append(kg.node_name(tgt))
    # 传播方式特殊：传播方式源节点很少（灰飞/蚜虫/病土等，约9个），
    # 它们不直接连到作物，从图谱所有"传播"边取 head（源节点）
    if spread_sources:
        spread_heads = []
        for u, v, d in kg.graph.edges(data=True):
            if d.get('reltype') == '传播' and kg.node_entity(u) in ('传播方式', '其他因素'):
                spread_heads.append(kg.node_name(u))
        values = list(dict.fromkeys(spread_heads))
    return list(dict.fromkeys(v for v in values if v and v.lower() != 'nan'))


def make_global_qa(kg):
    crops = []
    for nid in kg.entities_of_type('作物'):
        name = kg.node_name(nid)
        if name and name.lower() != 'nan':
            crops.append((nid, name))
    print(f'作物数: {len(crops)}')

    random.seed(RANDOM_SEED)
    qas = []
    # 每个模板收集候选（crop + 值）
    for tid, tpl in TEMPLATES.items():
        candidates = []
        for crop_nid, crop_name in crops:
            values = collect_crop_values(kg, crop_nid, tpl['rel'], tpl['tail_ents'],
                                         spread_sources=tpl.get('spread_sources', False))
            if len(values) >= MIN_VALUES:
                candidates.append((crop_name, values))
        # 传播方式源很少（约9个），用全部作物共享同一批源，降低 MIN_VALUES
        if tid == 'spread' and not candidates:
            all_spread = []
            for crop_nid, crop_name in crops:
                values = collect_crop_values(kg, crop_nid, tpl['rel'], tpl['tail_ents'],
                                             spread_sources=True)
                if values:
                    all_spread.append((crop_name, values))
            candidates = all_spread
        random.shuffle(candidates)
        # 每模板目标数 = TOTAL_TARGET / 6
        per = TOTAL_TARGET // len(TEMPLATES)
        for crop_name, values in candidates[:per]:
            question = tpl['q'](crop_name)
            items = list(dict.fromkeys(values))[:MAX_ITEMS]
            answer = tpl['ans_prefix'] + '、'.join(items) + '。'
            qas.append({
                'id': stable_id(tid, crop_name, question),
                'task': 'global_summary',
                'crop': crop_name,
                'template': tid,
                'question': question,
                'answer': answer,
                'reference_answer': answer,
                'gold_crop': crop_name,
                'gold_evidence_ids': [],
                'gold_graph_evidence': [],
                'source_relations': [tpl['rel']],
                'metadata': {'template': tid, 'label': tpl['label'],
                             'n_items': len(items)},
                'benchmark_version': 'graphrag_global_v1',
            })
        print(f'  [{tid}] {tpl["label"]}: 候选 {len(candidates)}, 生成 {min(per, len(candidates))}')

    # 去重（按问题）
    seen = set()
    final = []
    for qa in qas:
        k = _normalize(qa['question'])
        if k in seen:
            continue
        seen.add(k)
        final.append(qa)
    random.shuffle(final)
    final = final[:TOTAL_TARGET]
    return final


def main():
    kg = CropPestKG(NODES_CSV, RELS_CSV)
    qas = make_global_qa(kg)
    os.makedirs(os.path.dirname(GLOBAL_QA_FILE), exist_ok=True)
    with open(GLOBAL_QA_FILE, 'w', encoding='utf-8') as f:
        json.dump(qas, f, ensure_ascii=False, indent=1)
    print(f'\n生成 {len(qas)} 条全局题 → {GLOBAL_QA_FILE}')
    from collections import Counter
    tpl_cnt = Counter(q['template'] for q in qas)
    print('模板分布:', dict(tpl_cnt))
    # 抽查
    for qa in qas[:3]:
        print(f'\n  Q: {qa["question"]}')
        print(f'  A: {qa["answer"][:100]}')


if __name__ == '__main__':
    main()
