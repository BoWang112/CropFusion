# -*- coding: utf-8 -*-
"""
graphrag/summarize.py —— 社区摘要生成（DeepSeek）
===================================================
对每个社区，从其成员节点的结构化统计生成一段知识摘要，落盘 JSON。
支持断点续跑（resume）。

产出：data/graphrag/community_summaries.json
  [{community_id, size, has_crop, crops, pests, topics,
    keywords, summary, raw}]
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graphrag.prompts import (COMMUNITY_SUMMARY_SYSTEM,
                              COMMUNITY_SUMMARY_TEMPLATE)

# 关系 → 中文标签（用于结构化统计）
REL_LABELS = {
    '为害': '危害部位', '发病于': '发病时期', '传播': '传播方式',
    '症状有': '典型症状', '防治方法有': '防治要点',
    '胁迫有': '诱发因素', '致使': '诱发因素',
}


def _build_community_profile(kg, community):
    """把社区成员（节点 id 集合）转成结构化统计（供摘要模板）"""
    crops, pests = [], []
    sites, periods, spread, controls = set(), set(), set(), set()
    symptoms = set()
    ctrl_ents = {'农业防治', '化学防治', '物理防治', '生物防治', '其他因素'}
    site_ents = {'发病部位'}
    period_ents = {'发病时期'}
    spread_ents = {'传播方式'}
    sym_ents = {'具体症状'}
    entity_dist = {}

    for nid in community:
        ent = kg.node_entity(nid)
        entity_dist[ent] = entity_dist.get(ent, 0) + 1
        if ent == '作物':
            crops.append(kg.node_name(nid))
        elif ent in ('病害', '虫害', '害虫'):
            pests.append(kg.node_name(nid))
        # 出边统计
        for tgt, rel in kg.neighbors(nid):
            tgt_ent = kg.node_entity(tgt)
            tname = kg.node_name(tgt)
            if tgt_ent in site_ents and rel == '为害':
                sites.add(tname)
            elif tgt_ent in period_ents:
                periods.add(tname)
            elif tgt_ent in spread_ents:
                spread.add(tname)
            elif tgt_ent in sym_ents:
                symptoms.add(tname)

    # 防治要点：每个病虫害最多取 5 条（避免药名清单爆炸）
    controls = set()
    for nid in community:
        if kg.node_entity(nid) in ('病害', '虫害', '害虫'):
            for tgt, rel in kg.neighbors(nid, '防治方法有'):
                if kg.node_entity(tgt) in ctrl_ents:
                    controls.add(kg.node_name(tgt))
                    if len(controls) >= 40:
                        break
            if len(controls) >= 40:
                break

    def fmt(items, maxn=12):
        items = [x for x in items if x]
        return '、'.join(list(dict.fromkeys(items))[:maxn]) or '（无）'

    return {
        'crops': fmt(crops, 10),
        'pests': fmt(pests, 20),
        'sites': fmt(sites, 10),
        'periods': fmt(periods, 8),
        'spread': fmt(spread, 8),
        'controls': fmt(controls, 15),
        'symptoms': fmt(symptoms, 10),
        'entity_dist': '、'.join(f'{k}{v}' for k, v in
                                 sorted(entity_dist.items(), key=lambda x: -x[1])[:8]),
    }


def summarize_one_community(kg, community, llm, max_tokens=600):
    """对单个社区生成摘要，返回 dict"""
    prof = _build_community_profile(kg, community)
    prompt = COMMUNITY_SUMMARY_TEMPLATE.format(**prof)
    raw = llm.complete(prompt, system=COMMUNITY_SUMMARY_SYSTEM,
                       temperature=0.2, max_tokens=max_tokens)
    # 提取关键词行
    keywords = []
    for ln in (raw or '').split('\n'):
        ln = ln.strip()
        if ln.startswith('关键词：') or ln.startswith('关键词:'):
            keywords = [k.strip() for k in ln.split('：', 1)[-1].split('、')
                        if k.strip()]
            break
    return {
        'summary': raw,
        'keywords': keywords,
        'profile': prof,
    }


def summarize_communities(kg, communities, llm, out_path,
                          resume=True, subset=0):
    """
    批量生成社区摘要。
    subset>0 时只处理前 subset 个（验证用）。
    resume=True 跳过 out_path 中已存在的 community_id。
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    existing = {}
    if resume and os.path.exists(out_path):
        try:
            with open(out_path, encoding='utf-8') as f:
                existing = {d['community_id']: d for d in json.load(f)}
        except Exception:
            existing = {}

    items = list(communities)
    if subset > 0:
        items = items[:subset]

    results = dict(existing)
    for i, comm in enumerate(items):
        cid = i
        if cid in results and results[cid].get('summary'):
            print(f'  C{cid} 已存在，跳过', flush=True)
            continue
        # 社区 id 取排序位置
        result = summarize_one_community(kg, comm, llm)
        crops = [kg.node_name(n) for n in comm if kg.node_entity(n) == '作物']
        pests = [kg.node_name(n) for n in comm
                 if kg.node_entity(n) in ('病害', '虫害', '害虫')]
        results[cid] = {
            'community_id': cid,
            'size': len(comm),
            'has_crop': len(crops) > 0,
            'crops': crops,
            'pests': pests[:20],
            'topics': [],
            'keywords': result['keywords'],
            'summary': result['summary'],
        }
        # 每 5 个社区落盘一次
        if (i + 1) % 5 == 0:
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(list(results.values()), f, ensure_ascii=False, indent=1)
            print(f'  已落盘 {len(results)} 条', flush=True)

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(list(results.values()), f, ensure_ascii=False, indent=1)
    return list(results.values())


def load_summaries(path):
    """加载社区摘要 JSON"""
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def load_communities(path):
    """加载社区划分 JSON（community_id → member set）"""
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    out = {}
    for c in data['communities']:
        out[c['community_id']] = set(c['members'])
    return out, data['communities']
