# -*- coding: utf-8 -*-
"""
2_demo_cli.py —— 交互式问答命令行
====================================
加载索引 → 输入问题 → 四路检索 + 两级融合 → 证据驱动提示 → LLM 生成回答。
支持 -e 展示检索证据详情。

用法：
  python 2_demo_cli.py                # 交互模式
  python 2_demo_cli.py "水稻叶片有梭形病斑"   # 单问题
  python 2_demo_cli.py -e "水稻稻瘟病怎么防治"  # 展示证据
"""
import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import INDEX_DIR  # noqa: E402
from modules.kg import CropPestKG  # noqa: E402
from modules.retriever import (  # noqa: E402
    Embedder, VectorIndex, BM25Index, TextBlockStore, Retriever,
)
from modules.prompts import build_evidence_prompt, select_fewshot, format_fewshot  # noqa: E402
from modules.llm import LLMClient  # noqa: E402

from config import NODES_CSV, RELS_CSV  # noqa: E402


def load_pipeline():
    """加载 kg + 索引 + retriever + llm，返回可直接问答的对象"""
    print('加载图结构...')
    kg = CropPestKG(NODES_CSV, RELS_CSV)
    print('加载文本块...')
    store = TextBlockStore.load(os.path.join(INDEX_DIR, 'text_blocks.json'))
    print('加载向量索引...')
    with open(os.path.join(INDEX_DIR, 'vector_doc_ids.json'), encoding='utf-8') as f:
        doc_ids = json.load(f)
    vindex = VectorIndex.load(os.path.join(INDEX_DIR, 'vector.faiss'), doc_ids)
    print('加载 BM25 索引...')
    bindex = BM25Index.load(os.path.join(INDEX_DIR, 'bm25.pkl'))
    print('加载嵌入模型（BGE-large-zh）...')
    embedder = Embedder()
    llm = LLMClient()
    retriever = Retriever(kg, store, vindex, bindex, embedder, llm_client=llm)
    return retriever, llm, kg


def qa(retriever, llm, question, show_evidence=False):
    """一次完整问答：检索 → 融合 → 提示 → 生成"""
    text_ev, graph_ev, detail = retriever.get_evidence(question)
    if show_evidence:
        print('\n' + '=' * 60)
        print(f'【问题】{question}')
        print(f'\n【文本证据 Top{len(text_ev)}】')
        for i, e in enumerate(text_ev):
            print(f"  E{i+1} [{e['entity_type']}/{e['entity']}] {e['text'][:80]}...")
        print(f'\n【图谱证据 Top{len(graph_ev)}】')
        for i, g in enumerate(graph_ev[:15]):
            print(f"  G{i+1} {g['head']} -{g['rel']}-> {g['tail']}  ({g['tail_type']})")
        print(f'\n【检索详情】')
        print(f"  文本融合候选 {len(detail['fused_text'])}，实体锚点 {detail['entity_anchors'][:5]}，"
              f"语义锚点 {detail['semantic_anchors'][:5]}")
        print(f"  候选图：节点 {detail['graph_nodes_count']}，边 {detail['graph_edges_count']}")
        print('=' * 60)

    # 构建提示词 + 少样本示例
    task_type = '病虫害识别'
    fewshots = select_fewshot(question, task_type, top_k=3)
    fewshot_block = format_fewshot(fewshots)
    prompt = build_evidence_prompt(question, text_ev, graph_ev, learning_goal=task_type)
    full_prompt = f"以下为参考示例，帮助理解回答方式：\n{fewshot_block}\n\n{prompt}"

    # 生成（模拟/真实）
    answer = llm.complete_with_consistency(full_prompt)
    return answer


def main():
    ap = argparse.ArgumentParser(description='CDPKG-LLM 交互式问答')
    ap.add_argument('question', nargs='?', help='问题（缺省进入交互模式）')
    ap.add_argument('-e', '--evidence', action='store_true', help='展示检索证据')
    args = ap.parse_args()

    retriever, llm, kg = load_pipeline()
    print('CDPKG-LLM 就绪。' + ('（模拟 LLM）' if llm.mock_mode else f"（LLM: {llm.cfg['model']}）"))

    if args.question:
        ans = qa(retriever, llm, args.question, args.evidence)
        print('\n' + '=' * 60)
        print('【回答】')
        print(ans)
        return

    print('\n输入问题开始问答（输入 exit 退出）\n')
    while True:
        try:
            q = input('> ').strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            continue
        if q.lower() in ('exit', 'quit', '退出'):
            break
        ans = qa(retriever, llm, q, args.evidence)
        print('\n' + '=' * 60)
        print('【回答】')
        print(ans + '\n')


if __name__ == '__main__':
    main()
