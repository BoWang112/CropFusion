# -*- coding: utf-8 -*-
"""
3_evaluate.py —— 评测脚本
====================================
在自动生成的评测基准集（data/qa_test.json）上评测：

- 检索性能：P@k / R@k / F1@k / NDCG@k（k 由 --k 指定，默认 3）
  * 统一原子证据：文本片段（segment 级文本块）、三元组、关系路径、
    候选子图全部展开为带稳定 ID/来源的原子证据；路径/子图展开为
    三元组原子并保留 path_id/subgraph_id。
  * 文本证据保留 RRF 排名，图谱证据保留图谱重排序排名；两路分别
    映射为 1/(60+通道内排名) 后合并排序，同分按与方法名无关的
    原子 ID 打破平局，不使用额外的 BGE 重排序。Gold 文本按
    segment_id / 原文包含匹配，Gold 图谱按 (head, rel, tail) 精确匹配，
    重复 Gold 事实只统计一次。
  * 所有方法使用同一 evaluable 问题集合与同一套指标；F1 由汇总 P/R
    计算，检索失败按零分计入，不静默缩小样本量。
- 问答性能：BLEU-4 / ROUGE-L / BERTScore-F1
  统一使用自动任务分类（不读取 qa['task']）、动态 Few-shot 和三次生成
  一致性聚合（核心结论投票 + 证据覆盖度选择）；正式评估固定 concise
  提示模式，与 run_eval_resumable.py 保持一致；结果写入独立版本文件
  data/results/qa_results_v2.json，不覆盖历史结果。

评测任务可配置：--metrics retrieval / qa / both；--subset N 控制规模。
统一多方法检索评估：--metrics retrieval --methods all
（NaiveRAG / Hybrid RAG / GraphRAG / LightRAG / G-Retriever / PathRAG /
 SG-RAG / Ours-Retrieval，全部使用原子证据口径）。
"""
import os
import re
import sys
import json
import math
import hashlib
import argparse
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import INDEX_DIR, EVAL_FILE, RESULT_DIR, EXTRACT_JSON, RETRIEVAL  # noqa: E402
from config import get_llm_config, LLM_PROVIDERS  # noqa: E402
from modules.kg import CropPestKG  # noqa: E402
from modules.retriever import (  # noqa: E402
    Embedder, VectorIndex, BM25Index, TextBlockStore, Retriever,
)
from modules.llm import LLMClient  # noqa: E402
from modules.prompts import (  # noqa: E402
    FewShotSelector, BGEEmbedder, load_fewshot_pool, DEFAULT_FEWSHOT_FILE,
    build_prompt_with_dynamic_fewshot,
)
from modules.task_classifier import classify_task  # noqa: E402

from config import NODES_CSV, RELS_CSV  # noqa: E402


# 问答评测结果独立版本文件，避免覆盖历史 qa_results.json
QA_RESULTS_VERSION = 'v2'


def load_pipeline(provider=None):
    kg = CropPestKG(NODES_CSV, RELS_CSV)
    store = TextBlockStore.load(os.path.join(INDEX_DIR, 'text_blocks.json'))
    with open(os.path.join(INDEX_DIR, 'vector_doc_ids.json'), encoding='utf-8') as f:
        doc_ids = json.load(f)
    vindex = VectorIndex.load(os.path.join(INDEX_DIR, 'vector.faiss'), doc_ids)
    bindex = BM25Index.load(os.path.join(INDEX_DIR, 'bm25.pkl'))
    embedder = Embedder()
    llm = LLMClient(get_llm_config(provider))
    retriever = Retriever(kg, store, vindex, bindex, embedder, llm_client=llm)
    return retriever, kg, store


def _normalize_space(text):
    return re.sub(r'\s+', ' ', text or '').strip()


def _match_norm(text):
    """去掉空白，用于 gold 证据文本与文本块内容的包含判断。"""
    return re.sub(r'\s+', '', str(text or ''))


def _strip_template(text):
    """去除知识库“作物：x，病虫害：y，”模板前缀（与 generate_qa_test.py 一致）。"""
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


def _stable_id(*parts, prefix='ev'):
    """与 generate_qa_test.py:stable_id 相同的稳定 hash ID 构造。"""
    raw = '||'.join(_normalize_space(p) for p in parts if p is not None)
    return f'{prefix}_{hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]}'


def _triple_key(item):
    return (str(item.get('head') or '').strip(),
            str(item.get('rel') or '').strip(),
            str(item.get('tail') or '').strip())


def _triple_id(head, rel, tail):
    """图谱 gold 边 ID：kg_<sha1(head||rel||tail)[:16]>（与生成器一致）。"""
    return _stable_id(head, rel, tail, prefix='kg')


def _blocks_by_entity(store):
    blocks = defaultdict(list)
    for b in store.blocks:
        name = b.get('entity_name')
        if name:
            blocks[name].append(b)
    return blocks


def build_text_gold_index(store, extract_json=EXTRACT_JSON):
    """建立 gold 文本证据 ID（seg_*）→ 检索文本片段 ID 的映射。

    新版索引按 segment 语义边界构建文本片段，片段 ID 直接使用
    gold_evidence_ids 的稳定 seg_* ID；因此优先做 ID 直接映射。
    兼容旧索引时回退：
      1) 抽取结果 JSON 的 segment 原文，用与生成器相同的
         stable_id(crop, pest, strip_template(seg.text), prefix='seg') 重建 ID；
         若该 segment 原文出现在文本片段中，则精确映射到该片段；
      2) 文本包含匹配（gold_evidence_texts 出现在片段中）兜底；
      3) 未收录/被截断的 segment 不映射到实体块（exact=False, block_id=None）。

    返回 (blocks_by_entity, seg_index)；seg_index[sid] =
        {'text', 'block_id', 'exact', 'pest'}
    """
    blocks_by_entity = _blocks_by_entity(store)
    all_block_ids = {b['id'] for b in store.blocks}
    seg_index = {}
    try:
        with open(extract_json, encoding='utf-8') as f:
            records = json.load(f)
    except Exception:
        records = []

    for rec in records:
        crop = (rec.get('crop') or '').strip()
        pest = (rec.get('pest') or '').strip()
        if not pest:
            continue
        candidates = blocks_by_entity.get(pest, [])
        for seg in rec.get('segments') or []:
            text = _strip_template(seg.get('text') or '')
            if len(text) < 10:
                continue
            raw_id = ''
            for key in ('id', 'segment_id', 'chunk_id', 'evidence_id',
                        'text_id', 'uid'):
                value = seg.get(key)
                if value is not None and str(value).strip():
                    raw_id = str(value).strip()
                    break
            sid = raw_id or _stable_id(crop, pest, text, prefix='seg')
            # 新版索引：片段 ID 与 gold segment ID 一致，直接映射
            exact_block = sid if sid in all_block_ids else None
            if exact_block is None:
                for b in candidates:
                    if text and text in (b.get('text') or ''):
                        exact_block = b['id']
                        break
            # 只记录“原文确实出现在文本片段中”的精确映射；
            # 未收录/被截断的 segment 不映射到实体块，避免错误命中。
            seg_index[sid] = {
                'text': text,
                'block_id': exact_block,
                'exact': exact_block is not None,
                'pest': pest,
            }
    return blocks_by_entity, seg_index


def _gold_text_atoms(qa, blocks_by_entity, seg_index):
    """把一条 QA 的 gold 文本证据转换为原子证据集合。

    每个原子记录其文本、可映射的文本块集合、是否精确映射到块内容。
    """
    ids = qa.get('gold_evidence_ids') or []
    texts = qa.get('gold_evidence_texts') or []
    pest = qa.get('gold_pest') or ''
    pest_block_objs = blocks_by_entity.get(pest, [])

    atoms = []
    stats = {
        'ids': len(ids),
        'ids_in_index': 0,
        'ids_exact': 0,
        'ids_truncated': 0,
        'ids_unmapped': 0,
        'atoms': 0,
        'atoms_exact': 0,
        'no_block': False,
    }

    if ids:
        for i, sid in enumerate(ids):
            gold_text = _normalize_space(texts[i]) if i < len(texts) else ''
            entry = seg_index.get(sid)
            if entry:
                stats['ids_in_index'] += 1
                text = entry['text'] or gold_text
                if entry['exact']:
                    stats['ids_exact'] += 1
                    blocks = {entry['block_id']}
                else:
                    # segment 原文未被文本块收录（或被截断）：不使用实体块兜底，
                    # 该原子仍计入召回率分母，但不能被错误计为已覆盖。
                    stats['ids_truncated'] += 1
                    blocks = set()
                exact = entry['exact']
            else:
                stats['ids_unmapped'] += 1
                text = gold_text
                # 仅在 gold 证据文本确实出现在某文本块中时建立映射
                blocks = {
                    b['id'] for b in blocks_by_entity.get(pest, [])
                    if gold_text and _match_norm(gold_text) in _match_norm(b.get('text'))
                }
                exact = bool(blocks)
            atoms.append({'text': text, 'gold_text': gold_text,
                          'blocks': blocks, 'exact': exact,
                          'segment_id': sid})
    elif texts:
        # 兼容只有 gold_evidence_texts、没有 ID 的评测集
        for gold_text in texts:
            gold_text = _normalize_space(gold_text)
            blocks = {
                b['id'] for b in blocks_by_entity.get(pest, [])
                if gold_text and _match_norm(gold_text) in _match_norm(b.get('text'))
            }
            atoms.append({'text': gold_text, 'gold_text': gold_text,
                          'blocks': blocks, 'exact': bool(blocks),
                          'segment_id': None})

    # 同一病虫害可能有多个文本块（重复记录）：gold 证据文本实际出现在哪些块中，
    # 这些块都应视为相关块。这里仍是“文本精确包含”匹配，不是实体块兜底。
    for atom in atoms:
        needle = _match_norm(atom['gold_text'] or atom['text'])
        if not needle:
            continue
        for b in pest_block_objs:
            if needle in _match_norm(b.get('text')):
                atom['blocks'].add(b['id'])

    # 去重：同一 Gold 事实（相同证据原文）只统计一次；不同 segment ID
    # 但文本相同的重复事实合并为一个原子，保留全部 segment ID 与文本块。
    unique = []
    by_fact = {}
    for atom in atoms:
        fact = _match_norm(atom.get('gold_text') or atom.get('text'))
        seg = atom.get('segment_id')
        if fact:
            key = ('text', fact)
        elif seg:
            key = ('seg', seg)
        else:
            key = ('raw', atom.get('text', ''))
        existing = by_fact.get(key)
        if existing is None:
            atom['segment_ids'] = {seg} if seg else set()
            by_fact[key] = atom
            unique.append(atom)
        else:
            existing['blocks'] |= atom.get('blocks', set())
            existing['exact'] = existing.get('exact') or atom.get('exact')
            if seg:
                existing.setdefault('segment_ids', set()).add(seg)
    atoms = unique

    gold_blocks = set()
    for atom in atoms:
        gold_blocks |= atom['blocks']
    stats['atoms'] = len(atoms)
    stats['atoms_exact'] = sum(1 for a in atoms if a['exact'])
    # no_block 表示“有 gold 文本证据但没有任何文本块可精确承接”；
    # 这些题目不再用实体块兜底，会如实计入未命中。
    stats['no_block'] = bool(atoms) and not gold_blocks
    return atoms, gold_blocks, stats


def ndcg(relevance, k=5, n_ideal=None):
    """NDCG@k：relevance 为按序排列的 0/1 列表。

    n_ideal 为该查询**完整 Gold 相关事实数量**（文本 Gold 事实 + 图谱
    Gold 三元组；重复事实已去重），IDCG 取前 min(k, n_ideal) 个理想位置。
    这样未召回全部 gold 证据时 NDCG 会相应降低，而不是只看返回列表。
    """
    rel = relevance[:k]
    dcg = sum(r / math.log2(i + 2) for i, r in enumerate(rel))
    if n_ideal is None:
        ideal = sorted(relevance, reverse=True)[:k]
        idcg = sum(r / math.log2(i + 2) for i, r in enumerate(ideal))
    else:
        idcg = sum(1 / math.log2(i + 2) for i in range(min(k, max(0, int(n_ideal)))))
    return dcg / idcg if idcg > 0 else 0.0


def _text_relevant_blocks(atoms, blocks):
    """当 gold 文本块映射为空时，用 gold 证据文本在完整文本块集合中
    确定相关块集合。

    该集合同时用于 _is_hit 判定和 NDCG 的 IDCG，避免出现
    “文本匹配命中 → P/R>0，但 IDCG=0 → NDCG=0”的口径不一致。
    """
    needles = []
    for atom in atoms:
        needle = _match_norm(atom.get('gold_text') or atom.get('text'))
        if needle and needle not in needles:
            needles.append(needle)
    if not needles:
        return set()
    relevant = set()
    for block in blocks:
        block_text = _match_norm(block.get('text'))
        if not block_text:
            continue
        if any(needle in block_text for needle in needles):
            bid = block.get('id')
            if bid:
                relevant.add(bid)
    return relevant


def _gold_graph_triples(qa, diagnostics):
    """解析 gold 图谱证据，并校验 triple_id 映射。"""
    triples = set()
    for item in qa.get('gold_graph_evidence') or []:
        head, rel, tail = _triple_key(item)
        if not (head and tail):
            continue
        triples.add((head, rel, tail))
        tid = item.get('triple_id')
        diagnostics['graph_gold_triples'] += 1
        if not tid:
            diagnostics['graph_triple_id_missing'] += 1
        elif tid == _triple_id(head, rel, tail):
            diagnostics['graph_triple_id_valid'] += 1
        else:
            diagnostics['graph_triple_id_invalid'] += 1
    return triples




def eval_retrieval(retriever, kg, store, qas, k=3, extract_json=EXTRACT_JSON,
                   llm=None):
    """统一协议：Ours-Full 检索评估（委托 eval_retrieval_methods）。

    返回的整体 P/R/F1/NDCG 与所有对比方法完全一致；text_*/graph_* 仅为
    同一原子排序上的通道诊断，不再另用一套实体块/子图评估逻辑。
    """
    results, diag = eval_retrieval_methods(
        retriever, kg, store, qas, ['full'], k=k,
        extract_json=extract_json, llm=llm)
    m = results['Ours-Full']
    out = {
        'p': m['p'], 'r': m['r'], 'f1': m['f1'], 'ndcg': m['ndcg'],
        'text_p': m['text_p'], 'text_r': m['text_r'],
        'text_f1': m['text_f1'], 'text_ndcg': m['text_ndcg'],
        'graph_p': m['graph_p'], 'graph_r': m['graph_r'],
        'graph_f1': m['graph_f1'], 'graph_ndcg': m['graph_ndcg'],
        'top1': m['top1'], 'top3': m['top3'], 'top5': m['top5'],
        'avg_rank': m['avg_rank'],
        'n': m['n'], 'failures': m.get('failures', 0),
        'n_text': diag.get('text_questions', 0),
        'n_graph': diag.get('graph_questions', 0),
    }
    return out, m['n'], diag


# ============================================================
# 统一原子证据评估（文本块 / 三元组 / 路径 / 子图 → 原子证据）
# ============================================================

RETRIEVAL_METHOD_SPECS = {
    'naive-rag': ('NaiveRAG', 'naive-text'),
    'hybrid-rag': ('Hybrid RAG', 'hybrid-text'),
    'graphrag': ('GraphRAG', 'grag-local'),
    'lightrag': ('LightRAG', 'lightrag'),
    'g-retriever': ('G-Retriever', 'g-retriever'),
    'pathrag': ('PathRAG', 'pathrag'),
    'sg-rag': ('SG-RAG', 'sg-rag'),
    'ours-retrieval': ('Ours-Retrieval', 'full'),
    'graph-only': ('Graph-only', 'graph-only'),
    'full': ('Ours-Full', 'full'),
}
ALL_RETRIEVAL_METHODS = [
    'naive-rag', 'hybrid-rag', 'graphrag', 'lightrag',
    'g-retriever', 'pathrag', 'sg-rag', 'ours-retrieval',
]


def resolve_retrieval_methods(names):
    """把 CLI 方法名解析为内部 method key 列表。"""
    if not names or 'all' in names:
        return list(ALL_RETRIEVAL_METHODS)
    out = []
    for name in names:
        key = str(name).strip().lower()
        if key not in RETRIEVAL_METHOD_SPECS:
            raise ValueError(
                f'未知检索方法：{name!r}。可选：all, '
                f'{", ".join(RETRIEVAL_METHOD_SPECS)}')
        out.append(key)
    return list(dict.fromkeys(out))


UNIFIED_RANK_K = 60   # 两路通道统一排序分数：1/(60+通道内排名)


def _text_atom(item, method_label):
    content = item.get('text') or ''
    bid = item.get('id')
    meta = item.get('meta') or {}
    segment_id = item.get('segment_id') or meta.get('segment_id')
    source_id = item.get('source_id') or meta.get('source_id')
    stable = segment_id or bid
    atom_id = f'text::{stable}' if stable else \
        f'text::{hashlib.sha1(_match_norm(content).encode("utf-8")).hexdigest()[:16]}'
    refs = [x for x in (segment_id, bid, source_id) if x]
    return {
        'id': atom_id, 'type': 'text', 'source': method_label,
        'evidence_source': item.get('source'),
        'channel': 'text', 'origin': 'text',
        'block_id': bid, 'segment_id': segment_id,
        'source_id': source_id, 'content': content,
        'entity': item.get('entity'), 'entity_type': item.get('entity_type'),
        'rrf_rank': item.get('rrf_rank'),
        'rrf_score': (float(item['score'])
                      if item.get('score') is not None else None),
        'refs': refs,
    }


def _triple_atom(item, method_label, origin='triple', parent=None):
    head, rel, tail = _triple_key(item)
    if not (head or tail):
        return None
    tid = item.get('triple_id') or _triple_id(head, rel, tail)
    refs = [tid]
    if parent:
        refs.append(parent)
    atom = {
        'id': f'triple::{tid}', 'type': 'triple', 'source': method_label,
        'evidence_source': item.get('source'),
        'channel': 'graph', 'origin': origin, 'triple': (head, rel, tail),
        'content': f'{head} -{rel}-> {tail}', 'refs': refs,
        'relation': rel,
    }
    if parent:
        atom['parent'] = parent
    for key in ('path_id', 'subgraph_id', 'subgraph_rank', 'triple_rank',
                'graph_rank'):
        if item.get(key) is not None:
            atom[key] = item[key]
    path_ids = item.get('path_ids')
    if path_ids:
        atom['path_ids'] = list(path_ids)
        for pid in path_ids:
            if pid and pid not in refs:
                refs.append(pid)
    if item.get('triple_score') is not None:
        atom['triple_score'] = float(item['triple_score'])
    if item.get('score') is not None:
        atom['retrieval_score'] = float(item['score'])
    return atom


def _edge_sort_key(edge):
    """三元组/边对象的稳定排序键（仅用于无序集合的展开）。"""
    if isinstance(edge, dict):
        return _triple_key(edge)
    if isinstance(edge, (list, tuple)) and len(edge) >= 3:
        return (str(edge[0]), str(edge[1]), str(edge[2]))
    return (str(edge),)


def _ordered_edges(edges):
    """保留列表顺序；集合按稳定键排序，避免评测结果依赖 set 迭代顺序。"""
    if isinstance(edges, (set, frozenset)):
        return sorted(edges, key=_edge_sort_key)
    return list(edges)


def _expand_graph_item(item, method_label, group_index):
    """把路径/子图/三元组统一展开为三元组原子。

    展开时保留路径 ID、子图 ID、关系类型与原始排名；同一子图/路径内
    的边按上游给定顺序展开，不再按实体名称重排。
    """
    atoms = []
    edges = item.get('edges')
    path = item.get('path')
    inherited = {k: item.get(k) for k in
                 ('subgraph_id', 'subgraph_rank', 'path_id', 'graph_rank',
                  'source', 'triple_score')}
    if edges or path:
        is_path = bool(path) and not edges
        raw_edges = path if is_path else edges
        parent = (item.get('path_id') if is_path
                  else item.get('subgraph_id')) or \
            (f'path::{group_index}' if is_path else f'subgraph::{group_index}')
        origin = 'path' if is_path else 'subgraph'
        for idx, edge in enumerate(_ordered_edges(raw_edges), start=1):
            if isinstance(edge, dict):
                edge_item = dict(edge)
            else:
                # 统一约定 tuple = (head, rel, tail)
                edge_item = {'head': edge[0], 'rel': edge[1], 'tail': edge[2]}
            for key, value in inherited.items():
                if value is not None:
                    edge_item.setdefault(key, value)
            default_triple_rank = (item.get('triple_rank')
                                   if len(raw_edges) == 1 else idx)
            edge_item.setdefault('triple_rank', default_triple_rank)
            if item.get('score') is not None:
                edge_item.setdefault('score', item['score'])
            atom = _triple_atom(edge_item, method_label, origin, parent)
            if atom:
                atoms.append(atom)
        return atoms
    origin = ('path' if item.get('path_id')
              else ('subgraph' if item.get('subgraph_id') else 'triple'))
    parent = item.get('path_id') or item.get('subgraph_id')
    atom = _triple_atom(item, method_label, origin, parent)
    return [atom] if atom else []


def _rank_atoms_by_query(atoms, query=None, embedder=None):
    """按统一通道排名分数排序原子证据。

    文本与图谱原子分别保留各自通道内排名，再统一映射为
    score = 1/(UNIFIED_RANK_K + channel_rank)。相同分数使用固定、
    与方法名称无关的原子 ID 打破平局。query/embedder 仅为接口兼容。
    """
    if not atoms:
        return []
    ordered = sorted(
        atoms,
        key=lambda a: (-float(a.get('unified_score') or 0.0), a['id']),
    )
    for rank, atom in enumerate(ordered, start=1):
        atom['rank'] = rank
        atom['rank_score'] = float(atom.get('unified_score') or 0.0)
    return ordered


def _merge_atom_sources(existing, duplicate):
    """合并重复原子（同 ID）的来源/引用/父证据信息。"""
    srcs = existing.setdefault('duplicate_sources', [])
    for src in (duplicate.get('source'), duplicate.get('evidence_source')):
        if src and src not in srcs:
            srcs.append(src)
    refs = existing.setdefault('refs', [])
    for ref in duplicate.get('refs') or []:
        if ref not in refs:
            refs.append(ref)
    parents = existing.setdefault('parents', [])
    for parent in (duplicate.get('parent'), duplicate.get('subgraph_id'),
                   duplicate.get('path_id')):
        if parent and parent not in parents:
            parents.append(parent)
    path_ids = existing.setdefault('path_ids', [])
    for pid in duplicate.get('path_ids') or []:
        if pid and pid not in path_ids:
            path_ids.append(pid)
    for key in ('subgraph_rank', 'triple_rank', 'graph_rank',
                'triple_score', 'rrf_rank', 'rrf_score'):
        if existing.get(key) is None and duplicate.get(key) is not None:
            existing[key] = duplicate[key]
    return existing


def _build_ranked_atoms(text_evidence, graph_evidence, method_label,
                        query=None, embedder=None):
    """把文本片段、三元组、路径、候选子图展开为统一排序的原子证据。

    - 文本证据保留 RRF 排名；
    - 图谱证据保留图谱重排序排名（子图排名 -> 三元组排名）；
    - 两路分别映射为 1/(60+通道内排名)，再合并排序；
    - 同分用与方法名无关的原子 ID 打破平局；
    - 合并后按原子 ID 去重，保留全部来源/路径/子图信息。
    """
    atoms = []
    text_rank = 0
    for item in text_evidence or []:
        text_rank += 1
        atom = _text_atom(item, method_label)
        channel_rank = int(item.get('rrf_rank') or text_rank)
        atom['channel_rank'] = channel_rank
        atom['original_rank'] = channel_rank
        atom['unified_score'] = 1.0 / (UNIFIED_RANK_K + channel_rank)
        atoms.append(atom)

    graph_rank = 0
    for group_index, item in enumerate(graph_evidence or []):
        expanded = _expand_graph_item(item, method_label, group_index)
        for atom in expanded:
            # 优先使用上游传递的真实图谱通道排名；缺失时按展开顺序补齐。
            orig = atom.get('graph_rank')
            if orig is not None and len(expanded) == 1:
                channel_rank = int(orig)
                graph_rank = max(graph_rank, channel_rank)
            else:
                graph_rank += 1
                channel_rank = graph_rank
            atom['channel_rank'] = channel_rank
            atom.setdefault('original_rank', channel_rank)
            atom['unified_score'] = 1.0 / (UNIFIED_RANK_K + channel_rank)
            atoms.append(atom)

    # 先按统一分数排序，再按原子 ID 去重：保留排名最高的一条，
    # 并把后续重复原子的来源信息合并进来。
    atoms.sort(key=lambda a: (-float(a.get('unified_score') or 0.0), a['id']))
    unique = []
    by_id = {}
    for atom in atoms:
        existing = by_id.get(atom['id'])
        if existing is None:
            by_id[atom['id']] = atom
            unique.append(atom)
        else:
            _merge_atom_sources(existing, atom)
    return _rank_atoms_by_query(unique, query, embedder)


def _gold_for_atom_eval(qa, blocks_by_entity, seg_index):
    """返回 (gold text atoms, gold triples)，与原子评估共用同一 Gold 口径。"""
    atoms, _gold_blocks, _stats = _gold_text_atoms(
        qa, blocks_by_entity, seg_index)
    triples = _gold_graph_triples(qa, defaultdict(int))
    return atoms, triples


def _prepare_gold_sets(qas, store, extract_json=EXTRACT_JSON):
    """一次性准备所有方法的共同 Gold 集合与 ID 映射诊断。

    返回 (evaluable, diagnostics)。evaluable 只包含至少有一条 Gold 证据
    （文本原子或图谱三元组）的问题；所有检索方法必须在这个相同集合上
    评估，保证样本量一致。evaluable 元素为
    (qa, gold_text_atoms, gold_triples)。
    """
    blocks_by_entity, seg_index = build_text_gold_index(store, extract_json)
    evaluable = []
    diag = defaultdict(int)
    diag['text_gold_ids'] = 0
    diag['text_ids_in_index'] = 0
    diag['text_ids_exact'] = 0
    diag['text_ids_truncated'] = 0
    diag['text_ids_unmapped'] = 0
    diag['text_atoms'] = 0
    diag['text_atoms_exact'] = 0
    diag['text_no_block_questions'] = 0
    diag['text_text_matched_questions'] = 0
    diag['text_questions'] = 0
    diag['graph_gold_triples'] = 0
    diag['graph_triple_id_valid'] = 0
    diag['graph_triple_id_missing'] = 0
    diag['graph_triple_id_invalid'] = 0
    diag['graph_questions'] = 0
    diag['questions_total'] = 0

    for qa in qas:
        diag['questions_total'] += 1
        atoms, gold_blocks, stats = _gold_text_atoms(
            qa, blocks_by_entity, seg_index)
        diag['text_gold_ids'] += stats['ids']
        diag['text_ids_in_index'] += stats['ids_in_index']
        diag['text_ids_exact'] += stats['ids_exact']
        diag['text_ids_truncated'] += stats['ids_truncated']
        diag['text_ids_unmapped'] += stats['ids_unmapped']
        diag['text_atoms'] += stats['atoms']
        diag['text_atoms_exact'] += stats['atoms_exact']
        if stats['no_block']:
            diag['text_no_block_questions'] += 1

        # 仅用于诊断：gold 文本在完整文本块中的匹配情况。
        # 不再用该集合做“文本块相关即命中”的兜底。
        if not gold_blocks and _text_relevant_blocks(atoms, store.blocks):
            diag['text_text_matched_questions'] += 1

        triples = _gold_graph_triples(qa, diag)
        if atoms:
            diag['text_questions'] += 1
        if triples:
            diag['graph_questions'] += 1
        if atoms or triples:
            evaluable.append((qa, atoms, triples))
    return evaluable, dict(diag)


def _score_method_atoms(atoms, gold_text_atoms, gold_triples, k):
    """同一原子排序上的整体 + 文本/图谱通道诊断指标。"""
    overall = _score_atom_ranking(
        atoms, gold_text_atoms, gold_triples, k)
    if overall is None:
        return None

    text_atoms = [a for a in atoms if a['type'] == 'text']
    text_score = _score_atom_ranking(
        text_atoms, gold_text_atoms, (), k)
    graph_atoms = [a for a in atoms if a['type'] == 'triple']
    graph_score = _score_atom_ranking(
        graph_atoms, (), gold_triples, k)
    return {
        'overall': overall,
        'text': text_score,
        'graph': graph_score,
    }


def _atom_matches_text(atom, gold_text_atoms):
    """返回该文本原子覆盖的 gold 文本原子下标集合。

    命中判定只依据可追溯的 Gold 对应关系：
      1) 原子与 gold 的 segment_id 完全相同（新版 segment 级索引）；
      2) gold 证据原文确实出现在原子内容中（文本包含匹配）。
    不使用“文本块 ID 属于 gold 相关块集合”这一块级兜底，避免仅凭
    文本块相关就判定 Gold 原子命中。
    """
    matched = set()
    atom_seg = atom.get('segment_id')
    content = _match_norm(atom.get('content'))
    for idx, gold in enumerate(gold_text_atoms):
        gold_segs = gold.get('segment_ids') or set()
        gold_seg = gold.get('segment_id')
        if gold_seg:
            gold_segs = set(gold_segs) | {gold_seg}
        if atom_seg and atom_seg in gold_segs:
            matched.add(idx)
            continue
        needle = _match_norm(gold.get('gold_text') or gold.get('text'))
        if needle and needle in content:
            matched.add(idx)
    return matched


def _atom_hit(atom, gold_text_atoms, gold_triples):
    """单条原子证据是否命中 Gold。"""
    if atom['type'] == 'text':
        idxs = _atom_matches_text(atom, gold_text_atoms)
        return bool(idxs), idxs
    key = atom.get('triple')
    return key in gold_triples, None


def _score_atom_ranking(ranked, gold_text_atoms, gold_triples, k):
    """原子证据级 P@k / R@k / F1@k / NDCG@k。"""
    total_gold = len(gold_text_atoms) + len(gold_triples)
    if total_gold == 0:
        return None
    ordered = list(ranked or [])
    top = ordered[:k]
    hits = []
    covered_text = set()
    covered_triples = set()
    for atom in top:
        hit, text_idxs = _atom_hit(atom, gold_text_atoms, gold_triples)
        new_text = {i for i in (text_idxs or set())
                    if i >= 0 and i not in covered_text}
        new_triple = (hit and atom['type'] != 'text'
                      and atom.get('triple') not in covered_triples)
        # 只有覆盖了新的 Gold 事实才计为一次有效命中；
        # 同一 Gold 事实被多条原子重复命中时只统计一次。
        counted = bool(new_text or new_triple)
        hits.append(1 if counted else 0)
        covered_text |= new_text
        if new_triple:
            covered_triples.add(atom.get('triple'))

    precision = sum(hits) / k
    recall = (len(covered_text) + len(covered_triples)) / total_gold
    f1 = (2 * precision * recall / (precision + recall)
          if precision + recall > 0 else 0.0)
    n_ideal = min(k, total_gold)
    nd = ndcg(hits, k, n_ideal=n_ideal)
    first_hit = next(
        (i + 1 for i, atom in enumerate(ordered)
         if _atom_hit(atom, gold_text_atoms, gold_triples)[0]),
        None,
    )
    return {
        'p': precision, 'r': recall, 'f1': f1, 'ndcg': nd,
        'first_hit': first_hit,
    }


def _build_retrieval_engines(retriever, kg, llm, internal_names):
    """按需加载基线引擎；未请求的方法不初始化。"""
    engines = {}
    names = set(internal_names)
    if 'grag-local' in names or 'grag-global' in names:
        from graphrag.search import GraphRAGEngine
        from graphrag.summarize import load_summaries, load_communities
        from config import GRAPHRAG_DATA, GRAPHRAG
        try:
            summaries = load_summaries(
                os.path.join(GRAPHRAG_DATA, 'community_summaries.json'))
        except Exception:
            summaries = []
        try:
            comm_map, _ = load_communities(
                os.path.join(GRAPHRAG_DATA, 'communities.json'))
        except Exception:
            comm_map = {}
        community_of = {}
        for cid, members in comm_map.items():
            for member in members:
                community_of[member] = cid
        engines['grag-local'] = GraphRAGEngine(
            kg, retriever, llm, summaries=summaries,
            community_of=community_of,
            summary_top_k=GRAPHRAG['summary_top_k'])
        engines['grag-global'] = engines['grag-local']
    if 'g-retriever' in names:
        from graphrag.gretriever import GRetrieverEngine
        engines['g-retriever'] = GRetrieverEngine(kg, retriever)
    if 'lightrag' in names:
        from graphrag.lightrag import LightRAGEngine
        engines['lightrag'] = LightRAGEngine(kg, retriever)
    if 'pathrag' in names:
        from graphrag.pathrag import PathRAGEngine
        engines['pathrag'] = PathRAGEngine(kg, retriever)
    if 'sg-rag' in names:
        from graphrag.sgrag import SGRAGEngine
        engines['sg-rag'] = SGRAGEngine(kg, retriever)
    return engines


def _run_retrieval_method(internal_name, question, retriever, engines):
    if internal_name in ('naive-text', 'hybrid-text', 'graph-only', 'full'):
        return retriever.get_evidence(question, variant=internal_name)
    if internal_name == 'grag-local':
        return engines['grag-local'].local_search(question)
    if internal_name == 'grag-global':
        text_ev, graph_ev, _answer, detail = \
            engines['grag-global'].global_search(question)
        return text_ev, graph_ev, detail
    if internal_name in ('g-retriever', 'lightrag', 'pathrag', 'sg-rag'):
        return engines[internal_name].search(question)
    raise ValueError(f'未知内部检索方法：{internal_name!r}')


def eval_retrieval_methods(retriever, kg, store, qas, methods,
                           k=3, extract_json=EXTRACT_JSON, llm=None):
    """所有检索方法使用同一数据、证据粒度和指标口径的统一评估入口。

    每条检索结果展开为原子证据：
      - 文本片段 → text 原子（稳定 segment/block ID + 来源）；
      - 三元组 → triple 原子（稳定 kg hash）；
      - 关系路径 / 候选子图 → 展开为多条 triple 原子，保留
        path_id/subgraph_id/关系类型/原始排名。
    文本证据保留 RRF 排名，图谱证据保留图谱重排序排名；两路分别映射为
    1/(60+通道内排名) 后统一合并排序，同分按与方法名无关的原子 ID
    打破平局。合并后按原子 ID 去重并保留全部来源信息。
    P@k/R@k/F1@k/NDCG@k 全部在原子级计算，F1 由汇总 P/R 计算。

    所有方法使用同一个 evaluable 问题集合；检索失败计入 failures 并按
    零分参与汇总，绝不静默缩小样本量后再与其它方法比较。
    """
    if llm is None:
        llm = LLMClient(get_llm_config())
    if store.blocks and not all(
            (b.get('meta') or {}).get('segment_id') for b in store.blocks):
        print('  ⚠ 当前索引不是新版“segment 语义片段”索引，'
              '文本原子可能仍是实体级大块；请先运行 1_build_index.py。',
              flush=True)
    evaluable, diag = _prepare_gold_sets(qas, store, extract_json)
    n_total = len(evaluable)

    internal_names = [RETRIEVAL_METHOD_SPECS[m][1] for m in methods]
    engines = _build_retrieval_engines(retriever, kg, llm, internal_names)

    results = {}
    for method in methods:
        label, internal = RETRIEVAL_METHOD_SPECS[method]
        stats = defaultdict(float)
        text_n = graph_n = 0
        failures = 0
        atoms_total = 0

        for qi, (qa, gold_text_atoms, gold_triples) in enumerate(evaluable):
            try:
                text_ev, graph_ev, _detail = _run_retrieval_method(
                    internal, qa['question'], retriever, engines)
            except Exception as e:
                failures += 1
                print(f'  [{label}] 检索失败 {qa.get("id")}: {str(e)[:100]}',
                      flush=True)
                text_ev, graph_ev = [], []

            try:
                ranked = _build_ranked_atoms(
                    text_ev, graph_ev, label,
                    query=qa['question'], embedder=None)
            except Exception as e:
                failures += 1
                print(f'  [{label}] 原子证据构建失败 {qa.get("id")}: '
                      f'{str(e)[:100]}', flush=True)
                ranked = []
            atoms_total += len(ranked)
            scores = _score_method_atoms(
                ranked, gold_text_atoms, gold_triples, k)
            if scores is None:
                # evaluable 已保证至少一条 Gold；正常不会发生。
                failures += 1
                continue

            overall = scores['overall']
            stats['p'] += overall['p']
            stats['r'] += overall['r']
            stats['ndcg'] += overall['ndcg']
            first_hit = overall['first_hit']
            stats['rank_sum'] += (first_hit if first_hit is not None else k + 1)
            stats['top1'] += 1 if first_hit is not None and first_hit <= 1 else 0
            stats['top3'] += 1 if first_hit is not None and first_hit <= 3 else 0
            stats['top5'] += 1 if first_hit is not None and first_hit <= 5 else 0

            text_score = scores['text']
            if text_score is not None:
                text_n += 1
                stats['text_p'] += text_score['p']
                stats['text_r'] += text_score['r']
                stats['text_ndcg'] += text_score['ndcg']

            graph_score = scores['graph']
            if graph_score is not None:
                graph_n += 1
                stats['graph_p'] += graph_score['p']
                stats['graph_r'] += graph_score['r']
                stats['graph_ndcg'] += graph_score['ndcg']

            if n_total and (qi + 1) % 50 == 0:
                print(f'  [{label}] 已评测 {qi + 1}/{n_total}', flush=True)

        m = max(1, n_total)
        p = stats['p'] / m
        r = stats['r'] / m
        mt = max(1, text_n)
        mg = max(1, graph_n)
        text_p = stats['text_p'] / mt
        text_r = stats['text_r'] / mt
        graph_p = stats['graph_p'] / mg
        graph_r = stats['graph_r'] / mg
        results[label] = {
            'n': n_total,
            'failures': failures,
            'p': p,
            'r': r,
            'f1': (2 * p * r / (p + r) if p + r > 0 else 0.0),
            'ndcg': stats['ndcg'] / m,
            'text_p': text_p,
            'text_r': text_r,
            'text_f1': (2 * text_p * text_r / (text_p + text_r)
                        if text_p + text_r > 0 else 0.0),
            'text_ndcg': stats['text_ndcg'] / mt,
            'graph_p': graph_p,
            'graph_r': graph_r,
            'graph_f1': (2 * graph_p * graph_r / (graph_p + graph_r)
                         if graph_p + graph_r > 0 else 0.0),
            'graph_ndcg': stats['graph_ndcg'] / mg,
            'top1': stats['top1'] / m,
            'top3': stats['top3'] / m,
            'top5': stats['top5'] / m,
            'avg_rank': stats['rank_sum'] / m if n_total else 0.0,
            'avg_atoms': atoms_total / m if n_total else 0.0,
            'n_text': text_n,
            'n_graph': graph_n,
        }

        print(f'  [{label}] P@{k}={results[label]["p"]:.4f}  '
              f'R@{k}={results[label]["r"]:.4f}  '
              f'F1@{k}={results[label]["f1"]:.4f}  '
              f'NDCG@{k}={results[label]["ndcg"]:.4f}  '
              f'(n={n_total}, failures={failures})', flush=True)

    return results, diag


# ---------- 问答指标 ----------
def compute_bleu4(ref, hyp):
    try:
        from nltk.translate.bleu_score import sentence_bleu
        from nltk.translate.bleu_score import SmoothingFunction
        ref_tok = list(ref)
        hyp_tok = list(hyp)
        cc = SmoothingFunction().method4
        return sentence_bleu([ref_tok], hyp_tok, weights=(0.25, 0.25, 0.25, 0.25),
                             smoothing_function=cc)
    except Exception:
        return 0.0


def compute_rouge_l(ref, hyp):
    try:
        from rouge_score import rouge_scorer
        from rouge_score.rouge_scorer import tokenizers
        class CharTokenizer(tokenizers.Tokenizer):
            def tokenize(self, text):
                return list(text.replace(' ', ''))
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False,
                                          tokenizer=CharTokenizer())
        s = scorer.score(ref, hyp)
        return s['rougeL'].fmeasure
    except Exception:
        return 0.0


def compute_bertscore_batch(refs, hyps):
    """批量 BERTScore（单次加载模型，抑制警告，大幅提速）"""
    try:
        import warnings
        import bert_score
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            P, R, F = bert_score.score(hyps, refs, lang='zh',
                                       model_type='bert-base-chinese',
                                       verbose=False)
        return [float(f) for f in F]
    except Exception:
        return [0.0] * len(refs)


_FEWSHOT_SELECTOR = None


def build_fewshot_selector():
    """加载示例池并用 BGE 构建动态 Few-shot 选择器（只构建一次）。"""
    global _FEWSHOT_SELECTOR
    if _FEWSHOT_SELECTOR is None:
        pool = load_fewshot_pool(DEFAULT_FEWSHOT_FILE)
        _FEWSHOT_SELECTOR = FewShotSelector(pool, BGEEmbedder())
    return _FEWSHOT_SELECTOR


def eval_qa(retriever, kg, store, llm, qas, subset=None, save=True, concise=False):
    """端到端问答：自动任务分类 → 检索 → 动态 Few-shot → 提示 → 三次生成。

    - 任务类型由 modules/task_classifier 从问题文本自动判断，不读取 qa['task']；
    - 动态 Few-shot 与 run_eval_resumable.py 使用同一 FewShotSelector；
    - 生成使用三次独立采样 + 核心结论一致性投票 + 证据覆盖度选择；
    - 正式评估统一使用 concise 提示模式，与 run_eval_resumable.py 的
      full 变体保持一致；concise 参数保留用于接口兼容。
    性能优化：BLEU/ROUGE 逐条算，BERTScore 批量算（单次模型加载）。
    """
    if subset:
        qas = qas[:subset]
    selector = build_fewshot_selector()
    # 正式评估固定 concise：避免 3_evaluate 用 full、run_eval_resumable 用
    # concise 导致两套指标不可比。concise 参数保留但不再切换提示模式。
    _ = concise
    mode = 'concise'
    rows = []
    agg = defaultdict(float)
    n = 0
    for i, qa in enumerate(qas):
        question = qa['question']
        task, task_method = classify_task(question, llm)
        text_ev, graph_ev, detail = retriever.get_evidence(question)
        prompt = build_prompt_with_dynamic_fewshot(
            question=question,
            task_type=task,
            text_evidence=text_ev,
            graph_evidence=graph_ev,
            selector=selector,
            top_k=RETRIEVAL.get('fewshot_top_k', 3),
            mode=mode,
            include_citations=False,
        )
        generation = llm.complete_with_consistency_details(prompt)
        answer = generation.get('answer', '')
        if generation.get('method') == 'failed':
            print(f"  ⚠ LLM 生成失败，跳过：{question[:30]}...")
            continue
        ref = qa.get('answer', '')
        b4 = compute_bleu4(ref, answer)
        rl = compute_rouge_l(ref, answer)
        agg['bleu'] += b4
        agg['rouge'] += rl
        n += 1
        rows.append({
            'question': question, 'task': task, 'task_method': task_method,
            'ref': ref, 'answer': answer, 'bleu': b4, 'rouge': rl,
            'bertscore': None,
            # 生成审计信息（与 run_eval_resumable.py 口径一致）
            'candidates': generation.get('candidates', []),
            'votes': generation.get('votes', []),
            'coverage': generation.get('coverage', []),
            'agreement': generation.get('agreement', []),
            'core_conclusions': generation.get('core_conclusions', []),
            'valid_indices': generation.get('valid_indices', []),
            'selected_index': generation.get('selected_index'),
            'generation_method': generation.get('method'),
            'generation_error': generation.get('error'),
            'provider': generation.get('provider'),
            'model': generation.get('model'),
            'temperature': generation.get('temperature'),
            'top_p': generation.get('top_p'),
            'max_tokens': generation.get('max_tokens'),
        })
        if (i + 1) % 20 == 0:
            print(f"  已评测 {i + 1}/{len(qas)}")

    # 批量计算 BERTScore（单次模型加载）
    print("  批量计算 BERTScore...")
    bf_scores = compute_bertscore_batch([r['ref'] for r in rows],
                                        [r['answer'] for r in rows])
    for r, bf in zip(rows, bf_scores):
        r['bertscore'] = bf
        agg['bertscore'] += bf

    m = max(1, n)
    res = {'bleu4': agg['bleu'] / m, 'rouge_l': agg['rouge'] / m,
           'bertscore': agg['bertscore'] / m, 'n': n}
    if save:
        os.makedirs(RESULT_DIR, exist_ok=True)
        out_path = os.path.join(RESULT_DIR, f'qa_results_{QA_RESULTS_VERSION}.json')
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump({'metrics': res, 'rows': rows}, f, ensure_ascii=False, indent=1)
    return res, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--metrics', default='both', choices=['retrieval', 'qa', 'both'])
    ap.add_argument('--subset', type=int, default=0, help='仅评测前 N 条（0=全部）')
    ap.add_argument('--k', type=int, default=3, help='检索指标 @k（默认 3）')
    ap.add_argument('--provider', default='deepseek',
                    choices=sorted(LLM_PROVIDERS),
                    help='LLM provider 配置（模型 ID 由对应环境变量传入）')
    ap.add_argument('--methods', nargs='*', default=['full'],
                    help='检索方法：all 或 ' +
                         ', '.join(RETRIEVAL_METHOD_SPECS) +
                         '（默认 full，使用详细通道指标；多方法/all 使用统一原子证据指标）')
    ap.add_argument('--concise', action='store_true',
                    help='兼容参数：正式评估已固定使用 concise 提示模式')
    args = ap.parse_args()

    print('加载检索管线...')
    retriever, kg, store = load_pipeline(args.provider)
    qas = json.load(open(EVAL_FILE, encoding='utf-8'))
    print(f'评测集：{len(qas)} 组\n')

    if args.metrics in ('retrieval', 'both'):
        print('=' * 60)
        retrieval_qas = qas[:args.subset] if args.subset else qas
        methods = resolve_retrieval_methods(args.methods)
        # 单个 full/ours-retrieval：保留详细文本/图谱通道指标；
        # 多方法或 all：使用统一原子证据评估入口。
        detailed = (len(methods) == 1
                    and methods[0] in ('full', 'ours-retrieval'))
        if detailed:
            print('检索性能评测（文本 + 图谱原子 Gold 证据）')
            res, n, diag = eval_retrieval(
                retriever, kg, store, retrieval_qas, k=args.k)
            kk_ = args.k
            print(f'  (n_all={n}, n_text={res["n_text"]}, '
                  f'n_graph={res["n_graph"]})')
            if res.get('failures'):
                print(f'  ⚠ 检索失败 {res["failures"]} 条（按零分计入，'
                      f'样本量保持一致）')
            rows = [
                (f'P@{kk_} (all)', res['p']),
                (f'R@{kk_} (all)', res['r']),
                (f'F1@{kk_} (all)', res['f1']),
                (f'NDCG@{kk_} (all)', res['ndcg']),
                (f'Text-P@{kk_}', res['text_p']),
                (f'Text-R@{kk_}', res['text_r']),
                (f'Text-F1@{kk_}', res['text_f1']),
                (f'Text-NDCG@{kk_}', res['text_ndcg']),
                ('Evidence-Top1', res['top1']),
                ('Evidence-Top3', res['top3']),
                ('Evidence-Top5', res['top5']),
                ('Avg-Rank', res['avg_rank']),
                (f'Graph-P@{kk_}', res['graph_p']),
                (f'Graph-R@{kk_}', res['graph_r']),
                (f'Graph-F1@{kk_}', res['graph_f1']),
                (f'Graph-NDCG@{kk_}', res['graph_ndcg']),
            ]
            for label, value in rows:
                print(f'  {label:>17} = {value:.4f}')

            print('  证据 ID 映射检查：')
            print(f"    文本 gold IDs: {diag['text_gold_ids']}；"
                  f"抽取结果命中 {diag['text_ids_in_index']}；"
                  f"精确映射到块 {diag['text_ids_exact']}；"
                  f"截断未映射 {diag['text_ids_truncated']}；"
                  f"未映射 {diag['text_ids_unmapped']}")
            print(f"    文本 gold 原子: {diag['text_atoms']}"
                  f"（精确 {diag['text_atoms_exact']}）；"
                  f"无映射文本块问题数 {diag['text_no_block_questions']}；"
                  f"文本匹配补充相关块问题数 "
                  f"{diag['text_text_matched_questions']}")
            print(f"    图谱 gold 三元组: {diag['graph_gold_triples']}；"
                  f"triple_id 校验通过 {diag['graph_triple_id_valid']}，"
                  f"缺失 {diag['graph_triple_id_missing']}，"
                  f"不匹配 {diag['graph_triple_id_invalid']}")
            if diag['text_ids_unmapped']:
                print(f"    ⚠ 有 {diag['text_ids_unmapped']} 个 gold 文本 ID "
                      f"未在抽取结果中找到，已按文本包含匹配；"
                      f"仍无法匹配的不再使用实体块兜底。")
            if diag['graph_triple_id_invalid']:
                print(f"    ⚠ 有 {diag['graph_triple_id_invalid']} 个 gold "
                      f"图谱 triple_id 与 (head, rel, tail) 不一致，"
                      f"请检查评测集。")
        else:
            labels = [RETRIEVAL_METHOD_SPECS[m][0] for m in methods]
            print('统一原子证据评估：' + ', '.join(labels))
            method_llm = LLMClient(get_llm_config(args.provider))
            method_results, _method_diag = eval_retrieval_methods(
                retriever, kg, store, retrieval_qas, methods,
                k=args.k, llm=method_llm)
            print(f'  {"Method":<16} {"P@k":>8} {"R@k":>8} '
                  f'{"F1@k":>8} {"NDCG@k":>8} {"AvgRank":>8} '
                  f'{"n":>5} {"fail":>5}')
            for label, m in method_results.items():
                print(f'  {label:<16} {m["p"]:>8.4f} {m["r"]:>8.4f} '
                      f'{m["f1"]:>8.4f} {m["ndcg"]:>8.4f} '
                      f'{m["avg_rank"]:>8.4f} {m["n"]:>5d} '
                      f'{m.get("failures", 0):>5d}')

    if args.metrics in ('qa', 'both'):
        print('=' * 60)
        print('问答性能评测（正式评估：concise 提示，与 run_eval_resumable 一致）')
        llm = LLMClient(get_llm_config(args.provider))
        if llm.mock_mode:
            print('  ⚠ 当前为模拟 LLM（未配置 API Key），结果仅用于链路验证。')
        subset = args.subset or None
        res, rows = eval_qa(retriever, kg, store, llm, qas, subset=subset,
                            concise=args.concise)
        print(f'  (n={res["n"]})')
        print(f'  BLEU-4     = {res["bleu4"]:.4f}')
        print(f'  ROUGE-L    = {res["rouge_l"]:.4f}')
        print(f'  BERTScore  = {res["bertscore"]:.4f}')

    print('=' * 60)
    print('评测完成。')


if __name__ == '__main__':
    main()
