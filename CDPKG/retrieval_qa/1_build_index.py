# -*- coding: utf-8 -*-
"""
1_build_index.py —— 构建检索索引
==================================
从知识图谱 CSV + 抽取结果 JSON 构建：
  - 图结构（Graph View）：networkx 图
  - 文本块（Textual View）：实体级文本块
  - 向量索引：BGE-large-zh 编码 + FAISS
  - BM25 倒排索引（Lexical View）：jieba 分词

产物保存在 data/index/ 下，供 2_demo_cli.py / 3_evaluate.py 复用。

对应论文 3.3.1 Multi-view knowledge representation。
"""
import os
import sys
import json
import re
import pickle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (  # noqa: E402
    NODES_CSV, RELS_CSV, EXTRACT_JSON, INDEX_DIR, RETRIEVAL,
)
from modules.kg import CropPestKG
from modules.retriever import (
    Embedder, VectorIndex, BM25Index, TextBlockStore, tokenize, normalize_text,
)

# 防治类实体（文本块生成时保留原文，但作为关系检索对象的权重低）
CONTROL_ENTS = {'农业防治', '化学防治', '物理防治', '生物防治'}
# 排除"其他因素"这类低信息量实体作为文本块主体
LOW_INFO_ENTS = {'其他因素', '专业书籍', '权威网站', '分类地位', '拉丁文名称'}


def build_text_blocks(kg, extract_json=EXTRACT_JSON):
    """从抽取结果构建实体级文本块（论文 3.3.1 文本视图）。

    每条知识库记录（crop-pest）为一个单元，汇总其 segments 的
    文本、实体、三元组，形成以病虫害为中心的文本块。
    文本块同时挂载来源与实体 id，便于向量/图证据互相索引。
    """
    recs = json.load(open(extract_json, encoding='utf-8'))
    blocks = []
    bid = 0
    # 记录每个 (crop, pest) 的所有 segments，作为块内容
    for rec in recs:
        crop = (rec.get('crop') or '').strip()
        pest = (rec.get('pest') or '').strip()
        domain = rec.get('domain', '病害')
        if not pest:
            continue
        # 收集本记录的全部段文本（去重，限制长度避免超长块）
        seg_texts = []
        for seg in rec.get('segments', []):
            t = (seg.get('text') or '').strip()
            if t and t not in seg_texts:
                seg_texts.append(t)
        if not seg_texts:
            continue
        content = '\n'.join(seg_texts[:15])
        # 实体 id 关联
        pest_id = kg.find_node(pest)
        crop_id = kg.find_node(crop) if crop else None
        pest_ent = kg.node_entity(pest_id) if pest_id else domain
        # 块 id：统一用 b_<index>
        bid += 1
        blocks.append({
            'id': f'b{bid}',
            'text': content,
            'entity_id': pest_id,
            'entity_name': pest,
            'entity_type': pest_ent,
            'crop': crop,
            'source': f'{domain}知识库·{pest}',
            'meta': {'domain': domain, 'crop': crop},
        })
    return TextBlockStore(blocks)


def main():
    os.makedirs(INDEX_DIR, exist_ok=True)
    print('=' * 60)
    print('1/4  加载知识图谱图结构')
    kg = CropPestKG(NODES_CSV, RELS_CSV)
    print(f'  节点 {kg.graph.number_of_nodes()}，边 {kg.graph.number_of_edges()}')

    print('2/4  构建实体级文本块')
    store = build_text_blocks(kg)
    store_path = os.path.join(INDEX_DIR, 'text_blocks.json')
    store.save(store_path)
    print(f'  文本块 {len(store.blocks)} 条 -> {store_path}')

    print('3/4  构建向量索引（BGE-large-zh + FAISS）')
    embedder = Embedder()
    texts = [b['text'] for b in store.blocks]
    vecs = embedder.encode(texts, normalize=True, batch_size=32)
    dim = vecs.shape[1]
    doc_ids = [b['id'] for b in store.blocks]
    vindex = VectorIndex(dim, vecs, doc_ids)
    vindex_path = os.path.join(INDEX_DIR, 'vector.faiss')
    vindex.save(vindex_path)
    with open(os.path.join(INDEX_DIR, 'vector_doc_ids.json'), 'w', encoding='utf-8') as f:
        json.dump(doc_ids, f, ensure_ascii=False)
    print(f'  向量维度 {dim}，样本 {len(doc_ids)} -> {vindex_path}')

    print('4/4  构建 BM25 倒排索引')
    corpus = [tokenize(t) for t in texts]
    bindex = BM25Index(corpus, doc_ids)
    bindex_path = os.path.join(INDEX_DIR, 'bm25.pkl')
    bindex.save(bindex_path)
    print(f'  文档 {len(corpus)} 篇 -> {bindex_path}')

    print('5/6  预计算病虫害文本块向量（语义驱动图检索加速）')
    import numpy as np
    pest_nodes = kg.entities_of_type('病害') + kg.entities_of_type('虫害')
    pest_block_id = {b['entity_id']: b['id'] for b in store.blocks
                     if b.get('entity_id') in set(pest_nodes)}
    pids = list(pest_block_id.values())
    ptexts = [b['text'] for b in store.blocks if b['id'] in set(pids)]
    pvecs = embedder.encode(ptexts, batch_size=32, normalize=True)
    np.save(os.path.join(INDEX_DIR, 'pest_vecs.npy'), pvecs)
    with open(os.path.join(INDEX_DIR, 'pest_vec_ids.json'), 'w', encoding='utf-8') as f:
        json.dump(pids, f)
    print(f'  病虫害块 {len(pids)} 条 -> pest_vecs.npy')

    print('6/6  预热三元组边向量缓存（图重排序加速）')
    # 预编码全图 expand_rels 内的边，供图层重排序复用，避免评测时逐边编码
    expand_rels = {'为害', '症状有', '发病于', '处于', '影响', '直接引发',
                   '传播', '越冬情况', '世代情况', '易感时间', '易感区域',
                   '致使', '胁迫有', '防治方法有'}
    evidence_entities = {
        '作物', '病害', '虫害', '病原体', '害虫', '具体症状', '发病部位',
        '发病时期', '气候因素', '环境因素', '栽培因素', '其他因素',
        '世代情况', '越冬情况', '传播方式', '初侵染源', '易感时间',
        '易感地区', '农业防治', '化学防治', '物理防治', '生物防治',
    }
    ctrl_ents = {'农业防治', '化学防治', '物理防治', '生物防治', '其他因素'}
    tri_texts = []
    for s, e, d in kg.graph.edges(data=True):
        rel = d.get('reltype')
        if rel not in expand_rels:
            continue
        se, ee = kg.node_entity(s), kg.node_entity(e)
        if ee in ctrl_ents and se not in ctrl_ents:
            continue
        if ee not in evidence_entities:
            continue
        tri_texts.append(f'{kg.node_name(s)} {rel} {kg.node_name(e)}')
    if tri_texts:
        tri_vecs = embedder.encode(tri_texts, batch_size=256, normalize=True)
        cache = {t: v for t, v in zip(tri_texts, tri_vecs)}
        with open(os.path.join(INDEX_DIR, 'tri_cache.pkl'), 'wb') as f:
            pickle.dump(cache, f)
        print(f'  三元组边缓存 {len(cache)} 条 -> tri_cache.pkl')
    else:
        print('  无可用三元组边')

    print('=' * 60)
    print('索引构建完成。')


if __name__ == '__main__':
    main()
