# -*- coding: utf-8 -*-
"""
verify_walkthrough_candidates.py —— 论文示例候选的稳定性验证
================================================================
对候选列表里的每条题目，连续多次跑 get_evidence（variant='full'），
要求每次图谱证据锚点都含 gold_pest（graph_hit 稳定 True），且文本 top3 也命中。
通过筛选的才是可复现的论文示例。
"""
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import NODES_CSV, RELS_CSV, INDEX_DIR, EVAL_FILE, RESULT_DIR, LLM
from modules.kg import CropPestKG
from modules.retriever import Embedder, VectorIndex, BM25Index, TextBlockStore, Retriever
from modules.llm import LLMClient


def load_pipeline():
    kg = CropPestKG(NODES_CSV, RELS_CSV)
    store = TextBlockStore.load(os.path.join(INDEX_DIR, 'text_blocks.json'))
    with open(os.path.join(INDEX_DIR, 'vector_doc_ids.json'), encoding='utf-8') as f:
        doc_ids = json.load(f)
    vindex = VectorIndex.load(os.path.join(INDEX_DIR, 'vector.faiss'), doc_ids)
    bindex = BM25Index.load(os.path.join(INDEX_DIR, 'bm25.pkl'))
    r = Retriever(kg, store, vindex, bindex, Embedder(), llm_client=LLMClient(LLM))
    return r, kg


def main():
    r, kg = load_pipeline()
    cands = json.load(open(os.path.join(RESULT_DIR, 'retrieval_walkthrough',
                                        'best_candidates.json'), encoding='utf-8'))
    qas = json.load(open(EVAL_FILE, encoding='utf-8'))
    qmap = {}
    for qa in qas:
        key = (qa.get('gold_pest'), qa['question'][:20])
        qmap.setdefault(key, qa['question'])

    stable = []
    for c in cands:
        gold = c['gold_pest']
        question = c['question']
        # 连续跑 5 次，全部要求图谱锚点含 gold + 文本 top3 命中
        hits = 0
        for _ in range(5):
            te, ge, det = r.get_evidence(question, variant='full')
            anchors = {g.get('anchor') for g in ge if g.get('anchor')}
            g_hit = gold in anchors
            t_hit = any(e['entity'] == gold for e in te[:3])
            if g_hit and t_hit:
                hits += 1
        if hits == 5:
            stable.append({
                'gold': gold, 'question': question,
                'graph_evidence_count': len(ge),
                'text_top3': [e['entity'] for e in te[:3]],
            })
            print(f'稳定命中: {gold:<8} ge={len(ge)}  text_top3={[e["entity"] for e in te[:3]]}',
                  flush=True)

    print(f'\n稳定候选: {len(stable)} / {len(cands)}')
    with open(os.path.join(RESULT_DIR, 'retrieval_walkthrough',
                           'stable_candidates.json'), 'w', encoding='utf-8') as f:
        json.dump(stable, f, ensure_ascii=False, indent=1)
    print('保存: stable_candidates.json')


if __name__ == '__main__':
    main()
