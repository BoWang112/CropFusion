# -*- coding: utf-8 -*-
"""
build_summaries.py —— GraphRAG Phase 3：社区摘要生成（DeepSeek）
==================================================================
用法：
  BGE_MODEL=../models/bge-large-zh-v1.5 ../.venv-re/Scripts/python.exe -X utf8 \
    build_summaries.py [--subset 3]   # subset>0 先验证，再全量
产出：
  data/graphrag/community_summaries.json
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (NODES_CSV, RELS_CSV, GRAPHRAG, GRAPHRAG_DATA,
                    get_llm_config)
from modules.kg import CropPestKG
from modules.llm import LLMClient
from graphrag.community import detect_communities
from graphrag.summarize import summarize_communities, load_summaries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--subset', type=int, default=0,
                    help='仅处理前 N 个社区（0=全部），用于验证')
    args = ap.parse_args()

    kg = CropPestKG(NODES_CSV, RELS_CSV)
    communities, _ = detect_communities(
        kg, method=GRAPHRAG['community_method'],
        min_size=GRAPHRAG['community_min_size'],
        ctrl_ctrl_weight=GRAPHRAG['projection_ctrl_ctrl_weight'],
        pest_ctrl_weight=GRAPHRAG['projection_pest_ctrl_weight'])
    print(f'社区数: {len(communities)}')

    llm = LLMClient(get_llm_config())
    out_path = os.path.join(GRAPHRAG_DATA, 'community_summaries.json')
    summaries = summarize_communities(kg, communities, llm, out_path,
                                      resume=True, subset=args.subset)
    print(f'完成: {len(summaries)} 条摘要 → {out_path}')

    # 展示前 3 条摘要
    if args.subset and args.subset > 0:
        print('\n=== 样例摘要 ===')
        for s in summaries[:3]:
            print(f'\n--- C{s["community_id"]} (size={s["size"]}) ---')
            print(s['summary'][:300])


if __name__ == '__main__':
    main()
