# -*- coding: utf-8 -*-
"""
graphrag/community.py —— 社区检测（GraphRAG 对比实验 Phase 1-2）
==================================================================
构建加权无向投影图，并用 Leiden（cdlib）检测社区结构。

设计要点（论文对应：社区检测构建全局索引）：
- 知识图谱是有向图，社区检测在无向投影上进行
- "防治方法有"边中 69% 是防治节点互相串联的噪声链（ctrl→ctrl），
  对这些边大幅降权（0.05），病虫害→防治保留部分信号（0.5），
  诊断/为害/条件类边权重 1.0，本体类（属/拉丁名称）降权 0.1
- 社区按作物天然成簇（实测 Louvain modularity ~0.61），
  社区是 GraphRAG Global Search 的检索单元
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import networkx as nx

# 防治类实体类型（投影时这些节点参与但边降权）
CTRL_ENTS = {'农业防治', '化学防治', '物理防治', '生物防治', '其他因素'}
# 诊断/为害/条件类关系（主聚类信号，权重 1.0）
DIAG_RELS = {'为害', '症状有', '发病于', '处于', '影响', '直接引发',
             '传播', '胁迫有', '致使', '越冬情况', '世代情况',
             '易感时间', '易感区域', '发病于作物'}
# 本体类关系（权重 0.1）
ONT_RELS = {'属', '拉丁名称', '分类地位', '归类于', '类似虫害有', '知识来源于'}


def build_projection(kg, ctrl_ctrl_weight=0.05, pest_ctrl_weight=0.5):
    """
    从 CropPestKG 构建加权无向投影图。

    kg: modules/kg.py 的 CropPestKG（networkx DiGraph 在 kg.graph）
    返回 nx.Graph，边带 'weight' 属性。
    """
    g = nx.Graph()
    for u, v, data in kg.graph.edges(data=True):
        rel = data.get('reltype', '')
        u_ent = kg.node_entity(u)
        v_ent = kg.node_entity(v)
        # 防治→防治噪声链
        if rel == '防治方法有' and u_ent in CTRL_ENTS and v_ent in CTRL_ENTS:
            w = ctrl_ctrl_weight
        # 病虫害→防治
        elif rel == '防治方法有':
            w = pest_ctrl_weight
        # 本体类
        elif rel in ONT_RELS:
            w = 0.1
        # 诊断/为害/条件类
        else:
            w = 1.0
        if g.has_edge(u, v):
            g[u][v]['weight'] += w
        else:
            g.add_edge(u, v, weight=w)
    return g


def detect_communities(kg, method='leiden', min_size=10, seed=42):
    """
    Leiden 社区检测（cdlib），networkx Louvain 兜底。

    返回 list[set[node_id]]，仅保留 size >= min_size 的社区。
    """
    proj = build_projection(kg)
    communities = None
    if method == 'leiden':
        try:
            from cdlib import algorithms
            comms = algorithms.leiden(proj, weights='weight', seed=seed)
            communities = [set(c) for c in comms.communities]
        except Exception as e:
            print(f'[WARN] Leiden 失败（{e}），改用 Louvain', flush=True)
            communities = None
    if communities is None:
        communities = list(nx.community.louvain_communities(
            proj, seed=seed, weight='weight'))
    # 过滤小社区
    out = [set(c) for c in communities if len(c) >= min_size]
    # 未过滤的节点归入最近社区（保持全集覆盖）
    covered = set().union(*out) if out else set()
    leftover = set(proj.nodes()) - covered
    if leftover and out:
        for nid in leftover:
            # 找邻居最多的社区
            best = None
            best_cnt = 0
            for ci, comm in enumerate(out):
                cnt = sum(1 for nb in proj.neighbors(nid) if nb in comm)
                if cnt > best_cnt:
                    best_cnt = cnt
                    best = ci
            if best is not None:
                out[best].add(nid)
    return out, proj


def community_stats(kg, communities):
    """每社区统计：规模、作物数、病虫害数、实体类型分布"""
    stats = []
    for ci, comm in enumerate(communities):
        crops = [kg.node_name(n) for n in comm if kg.node_entity(n) == '作物']
        pests = [kg.node_name(n) for n in comm
                 if kg.node_entity(n) in ('病害', '虫害', '害虫')]
        ents = {}
        for n in comm:
            e = kg.node_entity(n)
            ents[e] = ents.get(e, 0) + 1
        stats.append({
            'community_id': ci,
            'size': len(comm),
            'has_crop': len(crops) > 0,
            'crops': crops,
            'pests': pests,
            'entity_dist': ents,
        })
    return stats


def save_communities(path, communities, kg, meta=None):
    """把社区及其统计落盘 JSON"""
    import json
    stats = community_stats(kg, communities)
    payload = {
        'meta': meta or {},
        'communities': [
            {'community_id': s['community_id'],
             'size': s['size'],
             'has_crop': s['has_crop'],
             'crops': s['crops'],
             'pests': s['pests'],
             'entity_dist': s['entity_dist'],
             'members': sorted(list(comm))}
            for s, comm in zip(stats, communities)
        ],
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    return payload
