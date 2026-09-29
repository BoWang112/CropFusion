# -*- coding: utf-8 -*-
"""
build_communities.py —— GraphRAG Phase 2：构建加权投影 + Leiden 社区检测
==========================================================================
用法：
  ../.venv-re/Scripts/python.exe -X utf8 build_communities.py [--method leiden|--method louvain]
产出：
  data/graphrag/communities.json（社区划分 + 统计）
"""
import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import NODES_CSV, RELS_CSV, GRAPHRAG, GRAPHRAG_DATA
from modules.kg import CropPestKG
from graphrag.community import (build_projection, detect_communities,
                                community_stats, save_communities)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--method', default=GRAPHRAG['community_method'],
                    choices=['leiden', 'louvain'])
    args = ap.parse_args()

    kg = CropPestKG(NODES_CSV, RELS_CSV)
    print(f'图谱节点: {kg.graph.number_of_nodes()}, 边: {kg.graph.number_of_edges()}')

    # 投影统计
    proj = build_projection(kg)
    print(f'无向投影: 节点 {proj.number_of_nodes()}, 边 {proj.number_of_edges()}')
    # 边权分布
    weights = [d['weight'] for _, _, d in proj.edges(data=True)]
    print(f'边权分布: min={min(weights):.2f} max={max(weights):.2f} '
          f'<1.0 占比 {sum(1 for w in weights if w < 1.0)/len(weights)*100:.1f}%')

    # 社区检测
    communities, _ = detect_communities(kg, method=args.method,
                                        min_size=GRAPHRAG['community_min_size'])
    print(f'社区数（>= {GRAPHRAG["community_min_size"]} 节点）: {len(communities)}')
    stats = community_stats(kg, communities)
    has_crop = sum(1 for s in stats if s['has_crop'])
    print(f'含作物的社区: {has_crop}/{len(stats)}')

    # 每社区构成
    print('\n社区构成（前 15 个）:')
    for s in stats[:15]:
        crop_str = ','.join(s['crops'][:5]) if s['crops'] else '(无作物)'
        pest_str = ','.join(s['pests'][:6]) if s['pests'] else '(无病虫害)'
        print(f'  C{s["community_id"]}: size={s["size"]} 作物[{crop_str}] '
              f'病虫害[{pest_str}]')

    # 落盘
    out_path = os.path.join(GRAPHRAG_DATA, 'communities.json')
    save_communities(out_path, communities, kg, meta={
        'method': args.method,
        'min_size': GRAPHRAG['community_min_size'],
        'projection_ctrl_ctrl_weight': GRAPHRAG['projection_ctrl_ctrl_weight'],
        'projection_pest_ctrl_weight': GRAPHRAG['projection_pest_ctrl_weight'],
    })
    print(f'\n已保存: {out_path}')


if __name__ == '__main__':
    main()
