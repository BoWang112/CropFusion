# -*- coding: utf-8 -*-
"""
run_method.py —— 单方法运行器（9 种方法）
=========================================
用法：
  python run_method.py --method llm_no_retrieval --limit 10   # 冒烟测试
  python run_method.py --method ours_full                     # 全量

方法：
  llm_no_retrieval / naive_rag / hybrid_rag / graphrag / lightrag /
  g_retriever / pathrag / sg_rag / ours_full

公平性保证：
- 同一 LLM（DeepSeek deepseek-chat）、同一生成参数
- 同一 sentence corpus / embedding / question embedding
- 同一 evidence 预算（≤8 units, ≤1500 tokens）
- LLM 缓存（hash(model+method+question_id+prompt+gen_cfg)）
- 断点续跑（读取已有预测文件，method+id 存在则跳过）
"""
import sys
import os
import json
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (DATA, PRED, SAMPLE_FILE, FEWSHOT_FILE,
                    DENSE_TOP_K, BM25_TOP_K, RRF_K,
                    SEMANTIC_ANCHOR_TOP_K, GRAPH_HOPS, FEWSHOT_TOP_K,
                    GRAPH_TOP_K, MAX_EVIDENCE_UNITS, MAX_EVIDENCE_TOKENS,
                    SEMANTIC_WEIGHT, STRUCTURAL_WEIGHT,
                    LLM_MODEL, LLM_TEMPERATURE, LLM_TOP_P, LLM_MAX_TOKENS,
                    complete, usage_reset, usage_snapshot,
                    build_base_prompt, build_evidence_prompt,
                    normalize_type, estimate_tokens)
import numpy as np

# 加载索引
SENTENCES = [json.loads(l) for l in open(os.path.join(DATA, 'sentences.jsonl'), encoding='utf-8')]
SENT_EMB = np.load(os.path.join(DATA, 'sentence_embeddings.npy'))
Q_EMB = np.load(os.path.join(DATA, 'question_embeddings.npy'))
LOOKUP = json.load(open(os.path.join(DATA, 'sentence_lookup.json'), encoding='utf-8'))
GRAPHS = json.load(open(os.path.join(DATA, 'graphs.json'), encoding='utf-8'))
Q_ORDER = LOOKUP['question_order']
Q_TYPE = LOOKUP['question_type']
Q2SENT = LOOKUP['q_sent_idx']

# question embedding 按 Q_ORDER 对齐
QID2EMB = {qid: Q_EMB[i] for i, qid in enumerate(Q_ORDER)}
# sentence 的 question_id 前缀索引
SENT_BY_Q = {}
for i, s in enumerate(SENTENCES):
    SENT_BY_Q.setdefault(s['question_id'], []).append((i, s))

# few-shot 池
FEWSHOT = [json.loads(l) for l in open(FEWSHOT_FILE, encoding='utf-8')]
FEWSHOT_QIDS = [q['id'] for q in FEWSHOT]

# 测试样本
TEST_ROWS = [json.loads(l) for l in open(SAMPLE_FILE, encoding='utf-8')]

GEN_CFG = {'temperature': LLM_TEMPERATURE, 'top_p': LLM_TOP_P,
           'max_tokens': LLM_MAX_TOKENS}

# 预测文件名映射
METHOD_FILES = {
    'llm_no_retrieval': 'llm_no_retrieval.jsonl',
    'naive_rag': 'naive_rag.jsonl',
    'hybrid_rag': 'hybrid_rag.jsonl',
    'graphrag': 'graphrag.jsonl',
    'lightrag': 'lightrag.jsonl',
    'g_retriever': 'g_retriever.jsonl',
    'pathrag': 'pathrag.jsonl',
    'sg_rag': 'sg_rag.jsonl',
    'ours_full': 'ours_full.jsonl',
}


# ================= 检索基础设施 =================

def get_question_emb(question):
    """question embedding（统一）"""
    return QID2EMB.get(_current_qid, None)


# ---------- dense retrieval ----------
def dense_search(qid, question, top_k=DENSE_TOP_K):
    """BGE 余弦相似度检索该 question 的 candidate sentences"""
    qv = QID2EMB[qid]
    idxs = Q2SENT[qid]
    if not idxs:
        return []
    embs = SENT_EMB[idxs]
    scores = embs @ qv  # 余弦（单位向量）
    order = np.argsort(-scores)
    return [(idxs[j], float(scores[j])) for j in order[:top_k]]


# ---------- BM25 retrieval ----------
_bm25_cache = {}


def get_bm25(qid):
    """按 question 过滤句子，构建 BM25 索引（仅该 question 的 candidate context）"""
    if qid in _bm25_cache:
        return _bm25_cache[qid]
    from rank_bm25 import BM25Okapi
    idxs = Q2SENT[qid]
    corpus = [SENTENCES[i]['text'].split() for i in idxs]
    bm = BM25Okapi(corpus, k1=1.5, b=0.75)
    _bm25_cache[qid] = (bm, idxs)
    return _bm25_cache[qid]


def bm25_search(qid, question, top_k=BM25_TOP_K):
    bm, idxs = get_bm25(qid)
    scores = bm.get_scores(question.split())
    order = np.argsort(-scores)
    return [(idxs[j], float(scores[j])) for j in order[:top_k]]


# ---------- RRF ----------
def rrf_fusion(lists, k=RRF_K):
    import math
    scores = {}
    for ranked in lists:
        for pos, (doc_id, _s) in enumerate(ranked):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + pos + 1)
    return sorted(scores.items(), key=lambda x: -x[1])


# ---------- 图检索基础设施 ----------
def get_graph(qid):
    return GRAPHS[qid]


def graph_adjacency(qid):
    """返回 {title: [neighbor titles]}（无向）"""
    g = get_graph(qid)
    adj = {n['title']: set() for n in g['nodes']}
    for e in g['edges']:
        adj.setdefault(e['source'], set()).add(e['target'])
        adj.setdefault(e['target'], set()).add(e['source'])
    return adj


def node_emb_matrix(qid):
    g = get_graph(qid)
    return np.array(g['node_embeddings']), [n['title'] for n in g['nodes']]


def node_text(qid):
    g = get_graph(qid)
    return {n['title']: n['paragraph_text'] for n in g['nodes']}


def semantic_anchors(qid, question, top_k=SEMANTIC_ANCHOR_TOP_K):
    """节点 embedding 余弦 → Top-K 语义锚点"""
    qv = QID2EMB[qid]
    embs, titles = node_emb_matrix(qid)
    scores = embs @ qv
    order = np.argsort(-scores)
    return [(titles[j], float(scores[j])) for j in order[:top_k]]


def entity_anchors(qid, question):
    """question 中显式匹配 candidate title 的实体（exact/case-insensitive/normalized/fuzzy）"""
    g = get_graph(qid)
    titles = [n['title'] for n in g['nodes']]
    q_lower = question.lower()
    matches = []
    from difflib import SequenceMatcher
    for t in titles:
        tl = t.lower()
        if tl in q_lower:
            matches.append(t)
            continue
        # normalized：去括号消歧
        import re
        m = re.match(r'^(.+?)\s*\(', t)
        if m and m.group(1).lower() in q_lower:
            matches.append(t)
            continue
        # fuzzy：标题出现在问题中的短语相似
        for token in re.findall(r"[\w']+", q_lower):
            if len(token) > 3 and SequenceMatcher(None, tl, token).ratio() > 0.9:
                matches.append(t)
                break
    return list(dict.fromkeys(matches))


def bfs_expand(qid, anchors, hops=GRAPH_HOPS):
    """从锚点 BFS 扩展，返回 {title: {neighbor: count}} 的 2-hop 子图"""
    adj = graph_adjacency(qid)
    frontier = set(anchors)
    visited = set()
    result_nodes = set(anchors)
    for _ in range(hops):
        nxt = set()
        for n in frontier:
            if n in adj:
                for nb in adj[n]:
                    if nb not in visited:
                        result_nodes.add(nb)
                        nxt.add(nb)
        visited |= frontier
        frontier = nxt
    return result_nodes


# ================= evidence 预算控制 =================

def evidence_units_to_text(units):
    """units: list of str（证据文本）→ 截断到预算，返回 (text_list, 是否截断)"""
    out = []
    total_tokens = 0
    for u in units:
        ntok = estimate_tokens(u)
        if len(out) >= MAX_EVIDENCE_UNITS or total_tokens + ntok > MAX_EVIDENCE_TOKENS:
            break
        out.append(u)
        total_tokens += ntok
    return out


# ================= 9 种方法 =================

# --- Method 1: LLM-No Retrieval ---
def run_llm_no_retrieval(qid, question):
    prompt = build_base_prompt(question)
    text, itok, otok = complete(LLM_MODEL, 'llm_no_retrieval', qid, prompt, GEN_CFG)
    return {'prediction': text, 'retrieved_evidence': [], 'prompt': prompt,
            'input_tokens': itok, 'output_tokens': otok}


# --- Method 2: NaiveRAG（仅 Dense）---
def run_naive_rag(qid, question):
    hits = dense_search(qid, question)
    units = [SENTENCES[i]['text'] for i, _ in hits]
    units = evidence_units_to_text(units)
    prompt = build_evidence_prompt(question, units)
    text, itok, otok = complete(LLM_MODEL, 'naive_rag', qid, prompt, GEN_CFG)
    ev = [{'evidence_id': SENTENCES[i]['evidence_id'],
           'title': SENTENCES[i]['title'],
           'sentence_id': SENTENCES[i]['sentence_id'],
           'text': SENTENCES[i]['text'], 'source': 'dense'}
          for i, _ in hits[:len(units)]]
    return {'prediction': text, 'retrieved_evidence': ev, 'prompt': prompt,
            'input_tokens': itok, 'output_tokens': otok}


# --- Method 3: Hybrid RAG（Dense + BM25 + RRF）---
def run_hybrid_rag(qid, question):
    dense = dense_search(qid, question)
    bm25 = bm25_search(qid, question)
    fused = rrf_fusion([dense, bm25])
    # 记录来源
    dense_set = {i for i, _ in dense}
    bm25_set = {i for i, _ in bm25}
    units = [SENTENCES[i]['text'] for i, _ in fused]
    units = evidence_units_to_text(units)
    prompt = build_evidence_prompt(question, units)
    text, itok, otok = complete(LLM_MODEL, 'hybrid_rag', qid, prompt, GEN_CFG)
    ev = []
    for i, _ in fused[:len(units)]:
        src = 'dense' if i in dense_set else ('bm25' if i in bm25_set else 'rrf')
        ev.append({'evidence_id': SENTENCES[i]['evidence_id'],
                   'title': SENTENCES[i]['title'],
                   'sentence_id': SENTENCES[i]['sentence_id'],
                   'text': SENTENCES[i]['text'], 'source': src})
    return {'prediction': text, 'retrieved_evidence': ev, 'prompt': prompt,
            'input_tokens': itok, 'output_tokens': otok}


# ================= 图方法公共：图上下文线性化 =================

def graph_evidence_lines(qid, anchors, expanded_nodes):
    """把子图节点组织成自然语言 evidence。返回 [(title, line, sentence_ids), ...]
    sentence_ids: 该节点下实际句子 id 列表（供 Support 评测追踪，不改证据内容）"""
    node_txt = node_text(qid)
    adj = graph_adjacency(qid)
    g = get_graph(qid)
    node_sent_ids = {n['title']: n.get('sentence_ids', []) for n in g['nodes']}
    pairs = []
    for t in expanded_nodes:
        txt = node_txt.get(t, '')
        if not txt:
            continue
        nbs = adj.get(t, set())
        nb_line = ', '.join(sorted(nbs)[:4]) if nbs else 'none'
        pairs.append((t, f'[{t}] (links to: {nb_line}) {txt}',
                      node_sent_ids.get(t, [])))
    return pairs


def truncate_pairs(pairs):
    """统一证据预算截断 pairs=[(title, line, sids), ...] → 截断后"""
    out = []
    total_tok = 0
    for title, line, sids in pairs:
        ntok = estimate_tokens(line)
        if len(out) >= MAX_EVIDENCE_UNITS or total_tok + ntok > MAX_EVIDENCE_TOKENS:
            break
        out.append((title, line, sids))
        total_tok += ntok
    return out


def graph_ev_to_records(qid, pairs):
    """图 evidence pairs → 结果记录（每节点一条，带 sentence_ids 供 Support 评测）。
    记录数与进入 prompt 的 units 一一对应（≤8），不逐句展开。"""
    g = get_graph(qid)
    node_sent_ids = {n['title']: n.get('sentence_ids', []) for n in g['nodes']}
    ev = []
    for title, _line, sids in pairs:
        ev.append({'evidence_id': f'graph::{title}',
                   'title': title,
                   'sentence_id': sids[0] if sids else -1,
                   'sentence_ids': list(sids),
                   'text': node_text(qid).get(title, ''), 'source': 'graph'})
    return ev


def run_graphrag(qid, question):
    """GraphRAG：语义定位节点 → 邻域扩展 → 图上下文线性化"""
    anchors = [t for t, _ in semantic_anchors(qid, question, top_k=SEMANTIC_ANCHOR_TOP_K)]
    expanded = bfs_expand(qid, anchors, hops=GRAPH_HOPS)
    pairs = truncate_pairs(graph_evidence_lines(qid, anchors, expanded))
    units = [line for _, line, _ in pairs]
    prompt = build_evidence_prompt(question, units)
    text, itok, otok = complete(LLM_MODEL, 'graphrag', qid, prompt, GEN_CFG)
    ev = graph_ev_to_records(qid, pairs)
    return {'prediction': text, 'retrieved_evidence': ev, 'prompt': prompt,
            'input_tokens': itok, 'output_tokens': otok}


def run_lightrag(qid, question):
    """LightRAG：低层实体锚点子图 + 高层结构关联 + 相关文本"""
    # 低层：显式实体锚点
    ent = entity_anchors(qid, question)
    # 高层：语义锚点补充
    sem = [t for t, _ in semantic_anchors(qid, question, top_k=SEMANTIC_ANCHOR_TOP_K)]
    anchors = list(dict.fromkeys(ent + sem))
    expanded = bfs_expand(qid, anchors, hops=GRAPH_HOPS)
    pairs = truncate_pairs(graph_evidence_lines(qid, anchors, expanded))
    units = [line for _, line, _ in pairs]
    prompt = build_evidence_prompt(question, units)
    text, itok, otok = complete(LLM_MODEL, 'lightrag', qid, prompt, GEN_CFG)
    ev = graph_ev_to_records(qid, pairs)
    return {'prediction': text, 'retrieved_evidence': ev, 'prompt': prompt,
            'input_tokens': itok, 'output_tokens': otok}


def run_g_retriever(qid, question):
    """G-Retriever：问题感知的图节点相关性打分 → 紧凑子图 → 线性化"""
    qv = QID2EMB[qid]
    embs, titles = node_emb_matrix(qid)
    scores = embs @ qv
    order = np.argsort(-scores)[:GRAPH_TOP_K]
    anchors = [titles[j] for j in order]
    expanded = bfs_expand(qid, anchors, hops=GRAPH_HOPS)
    pairs = truncate_pairs(graph_evidence_lines(qid, anchors, expanded))
    units = [line for _, line, _ in pairs]
    prompt = build_evidence_prompt(question, units)
    text, itok, otok = complete(LLM_MODEL, 'g_retriever', qid, prompt, GEN_CFG)
    ev = graph_ev_to_records(qid, pairs)
    return {'prediction': text, 'retrieved_evidence': ev, 'prompt': prompt,
            'input_tokens': itok, 'output_tokens': otok}


def run_pathrag(qid, question):
    """PathRAG：查询相关锚点 → 枚举候选路径 → 按相关度排序剪枝 → 路径线性化"""
    qv = QID2EMB[qid]
    adj = graph_adjacency(qid)
    # 锚点：显式实体 + 语义
    ent = entity_anchors(qid, question)
    sem = [t for t, _ in semantic_anchors(qid, question, top_k=SEMANTIC_ANCHOR_TOP_K)]
    anchors = list(dict.fromkeys(ent + sem))
    if not anchors:
        anchors = [n['title'] for n in get_graph(qid)['nodes']]
    # 枚举 2-hop 路径（真实存在于图中）
    paths = []
    for a in anchors:
        if a not in adj:
            continue
        for b in adj[a]:
            paths.append((a, b))
            for c in adj[b]:
                if c != a:
                    paths.append((a, b, c))
    # 路径打分：query 与路径节点文本相关度 + 长度奖励
    node_txt = node_text(qid)
    def path_score(p):
        s = 0.0
        for t in p:
            e = node_emb(qid, t)
            if e is not None:
                s += float(np.dot(QID2EMB[qid], e))
        s += 0.2 * len(p)
        return s
    scored = sorted(paths, key=path_score, reverse=True)
    # 剪枝：去重覆盖相同节点集合的路径，保留 Top-K
    selected = []
    seen_tail = set()
    for p in scored:
        key = frozenset(p)
        if key in seen_tail:
            continue
        seen_tail.add(key)
        selected.append(p)
        if len(selected) >= GRAPH_TOP_K:
            break
    pairs = []
    g = get_graph(qid)
    node_sent_ids = {n['title']: n.get('sentence_ids', []) for n in g['nodes']}
    for p in selected:
        chain = ' -> '.join(p)
        txt = '; '.join(node_txt.get(t, '')[:200] for t in p)
        sids = sorted(set(sid for t in p for sid in node_sent_ids.get(t, [])))
        pairs.append((chain, f'[Path: {chain}] {txt}', sids))
    pairs = truncate_pairs(pairs)
    units = [line for _, line, _ in pairs]
    prompt = build_evidence_prompt(question, units)
    text, itok, otok = complete(LLM_MODEL, 'pathrag', qid, prompt, GEN_CFG)
    ev = []
    for i, (chain, _line, sids) in enumerate(pairs):
        nodes = [n.strip() for n in chain.split(' -> ')] if chain else []
        ev.append({'evidence_id': f'path::{i}',
                   'title': nodes[0] if nodes else '',
                   'node_titles': nodes,          # 路径全部节点（供 Support 评测按任意节点匹配）
                   'sentence_id': sids[0] if sids else -1,
                   'sentence_ids': list(sids),
                   'text': chain, 'source': 'path'})
    return {'prediction': text, 'retrieved_evidence': ev, 'prompt': prompt,
            'input_tokens': itok, 'output_tokens': otok}


def node_emb(qid, title):
    """节点 embedding 查询辅助"""
    embs, titles = node_emb_matrix(qid)
    if title in titles:
        return embs[titles.index(title)]
    return None


def run_sg_rag(qid, question):
    """SG-RAG：抽取 query 关键实体/条件 → 候选节点 → 多条件子图匹配"""
    ent = entity_anchors(qid, question)
    sem = [t for t, _ in semantic_anchors(qid, question, top_k=SEMANTIC_ANCHOR_TOP_K)]
    cands = list(dict.fromkeys(ent + sem))
    if not cands:
        cands = [n['title'] for n in get_graph(qid)['nodes']]
    # 多条件匹配：覆盖 query 中多个显式条件的节点优先
    import re
    q_lower = question.lower()
    stop = {'what', 'where', 'when', 'which', 'whose', 'how', 'who', 'the',
            'and', 'that', 'with', 'from', 'this', 'was', 'were', 'did',
            'have', 'been', 'has', 'of', 'in', 'on', 'to', 'for'}
    cond_terms = [w for w in re.findall(r"[\w']+", q_lower)
                  if len(w) > 4 and w.lower() not in stop]
    def match_score(t):
        tl = t.lower()
        s = 1.0
        if t in ent:
            s += 2.0
        txt = node_text(qid).get(t, '').lower()
        s += sum(1 for w in cond_terms if w in txt) * 0.1
        return s
    cands.sort(key=lambda t: -match_score(t))
    top = cands[:GRAPH_TOP_K]
    expanded = bfs_expand(qid, top, hops=GRAPH_HOPS)
    pairs = truncate_pairs(graph_evidence_lines(qid, top, expanded))
    units = [line for _, line, _ in pairs]
    prompt = build_evidence_prompt(question, units)
    text, itok, otok = complete(LLM_MODEL, 'sg_rag', qid, prompt, GEN_CFG)
    ev = graph_ev_to_records(qid, pairs)
    return {'prediction': text, 'retrieved_evidence': ev, 'prompt': prompt,
            'input_tokens': itok, 'output_tokens': otok}


# ================= Ours-Full =================

def run_ours_full(qid, question):
    """Ours-Full：Dense + BM25 + RRF + 语义图 + 实体图 + 图rerank + 融合 + 动态few-shot + 证据驱动生成"""
    # 1) 文本分支
    dense = dense_search(qid, question)
    bm25 = bm25_search(qid, question)
    text_ranked = rrf_fusion([dense, bm25])
    dense_set = {i for i, _ in dense}
    bm25_set = {i for i, _ in bm25}

    # 2) 图谱分支：语义锚点 + 实体锚点
    sem_anchors = [t for t, _ in semantic_anchors(qid, question, top_k=SEMANTIC_ANCHOR_TOP_K)]
    ent_anchors = entity_anchors(qid, question)
    graph_anchors = list(dict.fromkeys(sem_anchors + ent_anchors))

    # 3) 图谱候选（锚点 2-hop 扩展）
    expanded = bfs_expand(qid, graph_anchors, hops=GRAPH_HOPS)

    # 4) 图 rerank：Semantic + Structural 综合得分
    def graph_score(t):
        sem_score = float(np.dot(QID2EMB[qid], node_emb(qid, t))) if node_emb(qid, t) is not None else 0.0
        adj = graph_adjacency(qid)
        nbs = adj.get(t, set())
        # structural：连接相关锚点数量 + 邻居数 + 是否覆盖 query 锚点
        struct_score = 0.0
        if nbs:
            rel_nb = sum(1 for nb in nbs if nb in set(graph_anchors))
            struct_score = 0.5 * (rel_nb / len(nbs)) + 0.5 * min(1.0, len(nbs) / 4.0)
        return SEMANTIC_WEIGHT * sem_score + STRUCTURAL_WEIGHT * struct_score
    reranked_anchors = sorted(expanded, key=graph_score, reverse=True)
    graph_top = reranked_anchors[:GRAPH_TOP_K]

    # 5) 图证据辅助（node_txt/adj 供融合阶段使用）
    node_txt = node_text(qid)
    adj = graph_adjacency(qid)

    # 6) 动态 few-shot：question embedding 与 40 个 fewshot question 余弦 → Top-3
    qv = QID2EMB[qid]
    fs_emb = np.array([QID2EMB[fq] for fq in FEWSHOT_QIDS])
    fs_scores = fs_emb @ qv
    fs_order = np.argsort(-fs_scores)[:FEWSHOT_TOP_K]
    fs_map = {q['id']: q for q in FEWSHOT}
    fs_examples = []
    for j in fs_order:
        fq = fs_map[FEWSHOT_QIDS[j]]
        # few-shot example 的 evidence：用该 fewshot question 的 dense top 句
        fq_hits = dense_search(fq['id'], fq['question'])
        fs_ev = ' '.join(SENTENCES[i]['text'] for i, _ in fq_hits[:3])
        fs_examples.append({'question': fq['question'], 'evidence': fs_ev,
                            'answer': fq['answer']})

    # 7) 文本 + 图谱证据融合（去重、排序、预算）
    # 图证据：线性化文本喂 LLM，但记录用节点 title
    fused_pool = []
    for t in graph_top:
        txt = node_txt.get(t, '')
        nbs = ', '.join(sorted(adj.get(t, set()))[:4]) or 'none'
        gl = f'[Graph: {t} links to {nbs}] {txt}'
        fused_pool.append(('graph', gl, t))   # rec = title
    for i, _ in text_ranked:
        fused_pool.append(('text', SENTENCES[i]['text'], i))  # 存数字索引
    # 统一预算截断（≤8 units, ≤1500 tokens）
    fused_units = []
    fused_records = []
    total_tok = 0
    for kind, unit_text, rec in fused_pool:
        ntok = estimate_tokens(unit_text)
        if len(fused_units) >= MAX_EVIDENCE_UNITS or total_tok + ntok > MAX_EVIDENCE_TOKENS:
            break
        fused_units.append(unit_text)
        fused_records.append((kind, rec))
        total_tok += ntok

    prompt = build_evidence_prompt(question, fused_units, fewshot_examples=fs_examples)
    text, itok, otok = complete(LLM_MODEL, 'ours_full', qid, prompt, GEN_CFG)

    ev = []
    g_all = get_graph(qid)
    node_sent_ids = {n['title']: n.get('sentence_ids', []) for n in g_all['nodes']}
    for kind, rec in fused_records:
        if kind == 'graph':
            sids = node_sent_ids.get(rec, [])
            ev.append({'evidence_id': f'graph::{rec}',
                       'title': rec,
                       'sentence_id': sids[0] if sids else -1,
                       'sentence_ids': list(sids),
                       'text': node_txt.get(rec, ''), 'source': 'graph'})
        else:
            i = rec  # sentence index
            src = 'dense' if i in dense_set else ('bm25' if i in bm25_set else 'rrf')
            ev.append({'evidence_id': SENTENCES[i]['evidence_id'],
                       'title': SENTENCES[i]['title'],
                       'sentence_id': SENTENCES[i]['sentence_id'],
                       'text': SENTENCES[i]['text'], 'source': src})
    return {'prediction': text, 'retrieved_evidence': ev, 'prompt': prompt,
            'input_tokens': itok, 'output_tokens': otok}


# ================= 分发 =================

RUNNERS = {
    'llm_no_retrieval': run_llm_no_retrieval,
    'naive_rag': run_naive_rag,
    'hybrid_rag': run_hybrid_rag,
    'graphrag': run_graphrag,
    'lightrag': run_lightrag,
    'g_retriever': run_g_retriever,
    'pathrag': run_pathrag,
    'sg_rag': run_sg_rag,
    'ours_full': run_ours_full,
}

_current_qid = None


def run_one(method, qa):
    """跑单条，返回结果 dict"""
    global _current_qid
    _current_qid = qa['id']
    t0 = time.time()
    res = RUNNERS[method](qa['id'], qa['question'])
    # 组装输出
    out = {
        'id': qa['id'],
        'method': method,
        'question': qa['question'],
        'gold_answer': qa['answer'],
        'gold_type': normalize_type(qa['type']),
        'prediction': res['prediction'],
        'retrieved_evidence': res['retrieved_evidence'],
        'input_tokens': res['input_tokens'],
        'output_tokens': res['output_tokens'],
        'latency_sec': round(time.time() - t0, 3),
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--method', required=True, choices=list(METHOD_FILES.keys()))
    ap.add_argument('--limit', type=int, default=0, help='仅跑前 N 条（0=全部）')
    ap.add_argument('--force', action='store_true', help='强制重跑（忽略断点）')
    ap.add_argument('--start', type=int, default=0, help='从第 N 条开始')
    args = ap.parse_args()

    method = args.method
    outfile = os.path.join(PRED, METHOD_FILES[method])
    done = {}
    if os.path.exists(outfile) and not args.force:
        for line in open(outfile, encoding='utf-8'):
            try:
                r = json.loads(line)
                done[r['id']] = r
            except Exception:
                pass
        print(f'恢复进度: {len(done)} 条已完成')

    todo = [qa for qa in TEST_ROWS
            if qa['id'] not in done and qa['id'] not in {}]
    if args.start:
        todo = todo[args.start:]
    if args.limit:
        todo = todo[:args.limit]
    print(f'方法={method}, 待跑 {len(todo)} 条')

    usage_reset()
    results = []
    for i, qa in enumerate(todo):
        try:
            out = run_one(method, qa)
        except Exception as e:
            print(f'  [{i}] {qa["id"]} 失败: {str(e)[:80]}', flush=True)
            continue
        results.append(out)
        # 立即落盘
        with open(outfile, 'a', encoding='utf-8') as f:
            f.write(json.dumps(out, ensure_ascii=False) + '\n')
        if (i + 1) % 5 == 0 or i + 1 == len(todo):
            print(f'  完成 {i+1}/{len(todo)} 条', flush=True)

    us = usage_snapshot()
    print(f'完成: {method} {len(results)} 条 (调用 {us["calls"]} 次, '
          f'输入 {us["input_tokens"]} tokens, 输出 {us["output_tokens"]} tokens)')
    print(f'结果已追加到 {outfile}')


if __name__ == '__main__':
    main()
