# -*- coding: utf-8 -*-
"""
find_best_walkthrough.py —— 挑选论文示例 + 保存完整检索过程
==============================================================
1. 保存"甘蓝病毒病"的完整检索过程（四路信号 + RRF + 图谱 + 答案）
2. 扫描全部 480 题，找"四路检索 + 两级融合都发挥关键作用"的最佳示例：
   - 四路检索（向量/BM25/语义图/实体图）都返回有效信号
   - RRF 融合后文本 top-3 命中 gold_pest
   - 图谱锚点含 gold_pest
   - 且各路径信号有"分歧"（非单一路径独挑大梁）
3. 输出到 data/results/retrieval_walkthrough/ 下
"""
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import NODES_CSV, RELS_CSV, INDEX_DIR, EVAL_FILE, RESULT_DIR, LLM
from modules.kg import CropPestKG
from modules.retriever import (Embedder, VectorIndex, BM25Index, TextBlockStore,
                               Retriever, RETRIEVAL)
from modules.llm import LLMClient

OUT = os.path.join(RESULT_DIR, 'retrieval_walkthrough')
os.makedirs(OUT, exist_ok=True)


def load_pipeline():
    kg = CropPestKG(NODES_CSV, RELS_CSV)
    store = TextBlockStore.load(os.path.join(INDEX_DIR, 'text_blocks.json'))
    with open(os.path.join(INDEX_DIR, 'vector_doc_ids.json'), encoding='utf-8') as f:
        doc_ids = json.load(f)
    vindex = VectorIndex.load(os.path.join(INDEX_DIR, 'vector.faiss'), doc_ids)
    bindex = BM25Index.load(os.path.join(INDEX_DIR, 'bm25.pkl'))
    llm = LLMClient(LLM)
    r = Retriever(kg, store, vindex, bindex, Embedder(), llm_client=llm)
    return r, kg


def capture(question, gold, task):
    """记录一条问题四路检索 + 融合的完整过程"""
    rec = {'question': question, 'gold_pest': gold, 'task': task}
    # 1) 四路检索
    vec = r.vector_search(question)
    kw = r.keyword_search(question)
    s_anchors, s_cand = r.semantic_graph_search(question)
    e_ents, e_anchors, e_cand, e_paths = r.entity_graph_search(question)
    rec['paths'] = {
        'vector_top1': r.block_by_id[vec[0][0]]['entity_name'] if vec else None,
        'vector_score': round(vec[0][1], 3) if vec else None,
        'bm25_top1': r.block_by_id[kw[0][0]]['entity_name'] if kw else None,
        'bm25_score': round(kw[0][1], 3) if kw else None,
        'semantic_anchors': [kg.node_name(a) for a in s_anchors][:8],
        'semantic_n': len(s_anchors),
        'entity_anchors': [kg.node_name(a) for a in e_anchors][:8],
        'entity_n': len(e_anchors),
        'entity_entities': e_ents[:5],
    }
    # 2) 目标锚点推断
    target = r._find_target_pest(question)
    kw_targets = r._keyword_target_pests(question)
    rec['target_anchor'] = {
        'exact_match': list(target) if target else [],
        'bm25_infer': list(kw_targets) if kw_targets else [],
    }
    # 3) RRF 融合（向量+BM25）
    fused = r.rrf_fusion([vec, kw], k=RETRIEVAL['rrf_k'])
    rec['rrf_top5'] = [
        {'id': did, 'entity': r.block_by_id[did]['entity_name'],
         'score': round(s, 4)}
        for did, s in fused[:5]
    ]
    rec['rrf_top3_hit_gold'] = any(
        r.block_by_id[did]['entity_name'] == gold for did, _ in fused[:3])
    # 4) 完整证据
    te, ge, det = r.get_evidence(question, variant='full')
    rec['text_evidence_top3'] = [e['entity'] for e in te[:3]]
    rec['text_hit_gold'] = any(e['entity'] == gold for e in te[:3])
    anchors = [g.get('anchor') for g in ge if g.get('anchor')]
    rec['graph_anchors'] = anchors
    rec['graph_hit_gold'] = gold in anchors
    rec['graph_evidence_count'] = len(ge)
    rec['graph_first_5'] = [
        {'head': g['head'], 'rel': g['rel'], 'tail': g['tail']}
        for g in ge[:5]
    ]
    return rec


if __name__ == '__main__':
    global r, kg
    r, kg = load_pipeline()
    qas = json.load(open(EVAL_FILE, encoding='utf-8'))

    # ---- 1. 甘蓝病毒病 ----
    target_q = next(x for x in qas if x.get('gold_pest') == '甘蓝病毒病')
    rec = capture(target_q['question'], '甘蓝病毒病', target_q.get('task'))
    # 补上主实验答案
    full = json.load(open(os.path.join(RESULT_DIR, 'qa_progress_full.json'), encoding='utf-8'))
    for qq, vv in full.items():
        if '老叶背面有黑色坏死斑' in qq:
            rec['final_answer'] = vv['answer']
            rec['bleu4'] = vv.get('bleu')
            break
    with open(os.path.join(OUT, 'walkthrough_甘蓝病毒病.json'), 'w', encoding='utf-8') as f:
        json.dump(rec, f, ensure_ascii=False, indent=1)
    print('已保存: walkthrough_甘蓝病毒病.json')
    print('  文本top3命中:', rec['text_hit_gold'], '| 图谱锚点命中:', rec['graph_hit_gold'])

    # ---- 2. 扫描找"四路+两级全发挥作用"的例子 ----
    best = []
    for qa in qas:
        gold = qa.get('gold_pest')
        if not gold:
            continue
        rec = capture(qa['question'], gold, qa.get('task'))
        # 判定：四路都有信号 + RRF top3 命中 + 图谱锚点命中 + 路径有分歧
        vec_ok = rec['paths']['vector_top1'] is not None
        bm25_ok = rec['paths']['bm25_top1'] is not None
        sem_ok = rec['paths']['semantic_n'] > 0
        ent_ok = rec['paths']['entity_n'] > 0
        all4 = vec_ok and bm25_ok and sem_ok and ent_ok
        dual_hit = rec['text_hit_gold'] and rec['graph_hit_gold']
        # 分歧度：向量top1 ≠ gold 但 BM25 top1 == gold（体现多源互补）
        divergence = (rec['paths']['vector_top1'] != gold
                      and rec['paths']['bm25_top1'] == gold)
        if all4 and dual_hit and divergence:
            best.append((gold, rec))
    print(f'\n找到"四路检索+两级融合全发挥作用且路径分歧"的候选: {len(best)} 个')
    for gold, rec in best[:8]:
        print(f'  {gold:<8} 向量top1={rec["paths"]["vector_top1"]}  '
              f'BM25top1={rec["paths"]["bm25_top1"]}  '
              f'te命中={rec["text_hit_gold"]} ge命中={rec["graph_hit_gold"]}')

    # 保存全部候选
    with open(os.path.join(OUT, 'best_candidates.json'), 'w', encoding='utf-8') as f:
        json.dump([{'gold': g, **rec} for g, rec in best], f,
                  ensure_ascii=False, indent=1)
    print(f'\n全部候选已保存: best_candidates.json ({len(best)} 条)')
