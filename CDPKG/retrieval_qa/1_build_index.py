# -*- coding: utf-8 -*-
"""
1_build_index.py —— 构建检索索引
==================================
从知识图谱 CSV + 抽取结果 JSON 构建：
  - 图结构（Graph View）：networkx 图
  - 文本片段（Textual View）：按知识库 segment 语义边界构建
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
import hashlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (  # noqa: E402
    NODES_CSV, RELS_CSV, EXTRACT_JSON, INDEX_DIR, RETRIEVAL,
)
from modules.kg import CropPestKG
from modules.retriever import (
    Embedder, VectorIndex, BM25Index, TextBlockStore, tokenize, normalize_text,
)

# 防治类实体（文本片段生成时保留原文，但作为关系检索对象的权重低）
CONTROL_ENTS = {'农业防治', '化学防治', '物理防治', '生物防治'}
# 排除"其他因素"这类低信息量实体作为文本片段主体
LOW_INFO_ENTS = {'其他因素', '专业书籍', '权威网站', '分类地位', '拉丁文名称'}


def _normalize_space(text):
    return re.sub(r'\s+', ' ', text or '').strip()


def _strip_template(text):
    """去除知识库“作物：x，病虫害：y，”模板前缀（与 QA 生成器一致）。"""
    text = _normalize_space(text)
    if not text:
        return ''
    for _ in range(3):
        old = text
        text = re.sub(r'^作物[:：]\s*[^，,。；;]*[，,]?\s*', '', text)
        text = re.sub(r'^病虫害[:：]\s*[^，,。；;]*[，,]?\s*', '', text)
        if text == old:
            break
    return text.strip()


def _stable_id(*parts, prefix='seg'):
    raw = '||'.join(_normalize_space(p) for p in parts if p is not None)
    return f'{prefix}_{hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]}'


def build_text_blocks(kg, extract_json=EXTRACT_JSON):
    """按知识库 segment 语义边界构建实体关联的文本片段。

    旧实现把每条 (crop, pest) 记录前 15 个 segment 拼成一个实体级大块，
    导致一个块包含多个事实、Gold segment 无法精确对应。现在：
      - 每个 segment 成为一个独立文本片段（segment 本身即知识库语义单元）；
      - 片段 ID 优先使用原始 segment ID，否则用
        stable_id(crop, pest, stripped_text, prefix='seg') 重建，
        与 generate_qa_test.py 的 gold_evidence_ids 使用同一规则；
      - 保留 entity_id/entity_name/entity_type/crop/source 及
        record_index/segment_index/segment_id/原始文本映射。
    """
    recs = json.load(open(extract_json, encoding='utf-8'))
    blocks = []
    used_ids = set()
    block_by_id = {}
    for rec_idx, rec in enumerate(recs):
        crop = (rec.get('crop') or '').strip()
        pest = (rec.get('pest') or '').strip()
        domain = rec.get('domain', '病害')
        pest_ent = domain if domain in ('病害', '虫害') else None
        if not pest:
            continue
        pest_id = kg.find_node(pest, entity=pest_ent)
        if pest_id is None:
            # 领域字段与图谱类型不一致时，退回病虫害类型集合定位，
            # 但仍优先使用上面的同类型匹配，避免同名不同类型误链。
            pest_id = kg.find_pest(pest)
        pest_ent = kg.node_entity(pest_id) if pest_id else (pest_ent or domain)
        record_source = rec.get('source_id') or rec.get('source') or \
            f'{domain}:{rec.get("row", rec_idx)}'

        for seg_idx, seg in enumerate(rec.get('segments') or []):
            raw_text = (seg.get('text') or '').strip()
            stripped = _strip_template(raw_text)
            # 与 QA 生成器一致：过短 segment 不作为证据单元
            if len(stripped) < 10:
                continue

            raw_id = ''
            for key in ('id', 'segment_id', 'chunk_id', 'evidence_id',
                        'text_id', 'uid'):
                value = seg.get(key)
                if value is not None and str(value).strip():
                    raw_id = str(value).strip()
                    break
            seg_id = raw_id or _stable_id(crop, pest, stripped, prefix='seg')
            # 同一 crop/pest/text 的稳定 ID 相同：合并来源信息，不重复建块
            if seg_id in used_ids:
                existing = block_by_id.get(seg_id)
                if existing is not None:
                    meta = existing.setdefault('meta', {})
                    sources = meta.setdefault('source_ids', [])
                    if record_source and record_source not in sources:
                        sources.append(record_source)
                    indices = meta.setdefault('record_indices', [])
                    if rec_idx not in indices:
                        indices.append(rec_idx)
                continue
            used_ids.add(seg_id)

            block = {
                'id': seg_id,
                'text': raw_text or stripped,
                'entity_id': pest_id,
                'entity_name': pest,
                'entity_type': pest_ent,
                'crop': crop,
                'source': f'{domain}知识库·{pest}',
                'meta': {
                    'domain': domain,
                    'crop': crop,
                    'record_index': rec_idx,
                    'record_indices': [rec_idx],
                    'record_row': rec.get('row'),
                    'segment_index': seg_idx,
                    'segment_id': seg_id,
                    'stripped_text': stripped,
                    'source_id': record_source,
                    'source_ids': [record_source] if record_source else [],
                },
            }
            blocks.append(block)
            block_by_id[seg_id] = block
    return TextBlockStore(blocks)


def main():
    os.makedirs(INDEX_DIR, exist_ok=True)
    print('=' * 60)
    print('1/4  加载知识图谱图结构')
    kg = CropPestKG(NODES_CSV, RELS_CSV)
    print(f'  节点 {kg.graph.number_of_nodes()}，边 {kg.graph.number_of_edges()}')

    print('2/4  构建语义片段文本块')
    store = build_text_blocks(kg)
    store_path = os.path.join(INDEX_DIR, 'text_blocks.json')
    store.save(store_path)
    print(f'  文本片段 {len(store.blocks)} 条 -> {store_path}')

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
    # 每个病虫害实体用一个代表片段（首片段）做语义锚点向量；文本检索本身
    # 覆盖该病虫害的全部片段。使用 setdefault 保证代表片段选择确定。
    pest_block_id = {}
    for b in store.blocks:
        if b.get('entity_id') in set(pest_nodes):
            pest_block_id.setdefault(b['entity_id'], b['id'])
    pids = list(pest_block_id.values())
    block_by_id = {b['id']: b for b in store.blocks}
    # 必须按 pids 顺序取文本，保证 pest_vecs 与 pest_vec_ids 严格对齐
    ptexts = [block_by_id[pid]['text'] for pid in pids if pid in block_by_id]
    pids = [pid for pid in pids if pid in block_by_id]
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
    for s, e, rel, _data in kg.iter_edges():
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
