# -*- coding: utf-8 -*-
"""
run_ours_ablation.py —— Ours-Full 消融变体（4 个独立修复 + 全合版 v2）
========================================================================
每个变体 = 基线 Ours-Full + 单点修复（其余完全一致，含动态 few-shot）：
  anchor_retain : 图 rerank 后强制保留显式实体锚点（修复 1）
  text_quota    : 融合预算给文本配额（图证据 ≤3，其余填精准句子）（修复 2）
  entity_first  : 锚点排序 entity 优先于 semantic（修复 3）
  sentence_ev   : 图证据句子化（节点内取与问题最相关的 top-2 句）（修复 4）
  v2            : 以上四项全合
  anchor_fs0    : anchor_retain + 去掉动态 few-shot
  no_rerank     : anchor_retain + 不做图 rerank（锚点按 BFS 顺序直进，接近 LightRAG 配方）
  no_rerank_fs0 : no_rerank + 去掉动态 few-shot
  pure_graph8   : 实体锚点强制 + 语义排序 + 8 个图节点全部进证据（无文本，few-shot 保留）
  sentence_gated8: 实体锚点强制 + 语义排序 + 图内句子全局 top-8（few-shot 保留）
  graph8_bfs    : 实体锚点强制 + BFS 顺序 8 节点（结构优先于语义排序）
  pure_graph8_fs0: pure_graph8 去掉 few-shot
  pure_graph8_fs_type: pure_graph8 + few-shot 按题型（同 type 池内选 top-3）
  pure_graph8_fs_type5: 同上，同 type 池内选 top-5
  pure_graph8_fs_type10: 同上，同 type 池内全选（top-10）

输出：outputs/predictions/{variant}.jsonl（与 run_method.py 同格式，可被 evaluate 复用）
LLM 缓存：method 名用 variant 名，互不冲突。
"""
import sys
import os
import json
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (PRED, DENSE_TOP_K, BM25_TOP_K, RRF_K,
                    SEMANTIC_ANCHOR_TOP_K, GRAPH_HOPS, FEWSHOT_TOP_K, GRAPH_TOP_K,
                    MAX_EVIDENCE_UNITS, MAX_EVIDENCE_TOKENS,
                    SEMANTIC_WEIGHT, STRUCTURAL_WEIGHT,
                    LLM_MODEL, complete, usage_reset, usage_snapshot,
                    build_evidence_prompt, normalize_type, estimate_tokens)
import numpy as np

# 复用 run_method.py 的检索基础设施与数据加载
from run_method import (dense_search, bm25_search, rrf_fusion, semantic_anchors,
                        entity_anchors, bfs_expand, node_text, graph_adjacency,
                        node_emb, SENTENCES, SENT_EMB, QID2EMB, SENT_BY_Q,
                        FEWSHOT, FEWSHOT_QIDS, get_graph, TEST_ROWS, GEN_CFG)

VARIANTS = ['anchor_retain', 'text_quota', 'entity_first', 'sentence_ev', 'v2',
            'anchor_fs0', 'no_rerank', 'no_rerank_fs0',
            'pure_graph8', 'sentence_gated8',
            'graph8_bfs', 'pure_graph8_fs0', 'pure_graph8_fs_type',
            'pure_graph8_fs_type5', 'pure_graph8_fs_type10']


# ---------- 规则题型预测（只用问题文本，不用 gold type，满足协议） ----------
_COMP_VERBS = ['born first', 'born later', 'born earlier', 'first out of',
               'came out', 'released', 'lived longer', 'is older', 'older than',
               'is younger', 'younger than', 'died first', 'died later', 'died earlier',
               'same country', 'same nationality', 'originate', 'originated',
               'share the same nationality', 'the other']


def predict_type(question):
    """基于问题文本的题型规则预测（不读 gold 字段，保证协议合规）"""
    ql = question.lower()
    first = ql.split()[0] if ql.split() else ''
    if any(k in ql for k in _COMP_VERBS):
        if 'director' in ql or 'whose' in ql or 'has the' in ql:
            return 'bridge-comparison'
        return 'comparison'
    if (first in ('where', 'what', 'when', 'how')
            or any(k in ql for k in ['award', 'nationality', 'birthday', 'place of birth'])):
        return 'compositional'
    import re as _re
    q2 = _re.sub(r'^(who|which|what|when|where|how|whose|whom|are|is|do|did|was|were|does)\b\s*',
                 '', question)
    rels = (len(_re.findall(r'\bof\b', q2)) + len(_re.findall(r"'s\b", q2))
            + len(_re.findall(r'\b(that|whose|which)\b', q2.lower())))
    return 'compositional' if rels >= 2 else 'inference'


def node_sentences_scored(qid, title):
    """节点内句子按与问题的余弦相似度降序，返回 [(score, idx, sent_dict), ...]"""
    g = get_graph(qid)
    node_sids = {n['title']: n.get('sentence_ids', []) for n in g['nodes']}
    want = set(node_sids.get(title, []))
    cands = []
    for i, s in SENT_BY_Q.get(qid, []):
        if s['title'] == title and s['sentence_id'] in want:
            cands.append((float(SENT_EMB[i] @ QID2EMB[qid]), i, s))
    cands.sort(key=lambda x: -x[0])
    return cands


def ordered_expand(qid, anchors, hops=GRAPH_HOPS):
    """BFS 扩展，返回确定顺序的节点列表：锚点在前，然后 1-hop、2-hop 分层。
    不按语义/结构打分，仅按图结构顺序（接近 LightRAG 配方）。"""
    adj = graph_adjacency(qid)
    ordered = []
    seen = set()
    for a in anchors:
        if a not in seen:
            seen.add(a)
            ordered.append(a)
    frontier = list(seen)
    for _ in range(hops):
        nxt = []
        for n in frontier:
            for nb in sorted(adj.get(n, set())):
                if nb not in seen:
                    seen.add(nb)
                    ordered.append(nb)
                    nxt.append(nb)
        frontier = nxt
    return ordered


def node_top_sentences(qid, title, top_n=2):
    """节点内与问题最相关的 top_n 句（复用 node_sentences_scored）"""
    return [s for _sc, _i, s in node_sentences_scored(qid, title)[:top_n]]


def run_variant(qid, question, v, qa_type=''):
    # ---------- 1) 文本分支 ----------
    dense = dense_search(qid, question)
    bm25 = bm25_search(qid, question)
    text_ranked = rrf_fusion([dense, bm25])
    dense_set = {i for i, _ in dense}
    bm25_set = {i for i, _ in bm25}

    # ---------- 2) 图锚点 ----------
    sem_anchors = [t for t, _ in semantic_anchors(qid, question, top_k=SEMANTIC_ANCHOR_TOP_K)]
    ent_anchors = entity_anchors(qid, question)
    if v in ('entity_first', 'v2'):
        graph_anchors = list(dict.fromkeys(ent_anchors + sem_anchors))   # 修复 3
    else:
        graph_anchors = list(dict.fromkeys(sem_anchors + ent_anchors))

    # ---------- 3) BFS 扩展 + 4) 图 rerank / 锚点直进 ----------
    if v in ('no_rerank', 'no_rerank_fs0'):
        # 不做 rerank：锚点按 BFS 顺序直接进证据（实体锚点最前，与 anchor_retain 一致）
        expanded_ordered = ordered_expand(qid, graph_anchors, hops=GRAPH_HOPS)
        forced = [t for t in ent_anchors if t in expanded_ordered]
        rest = [t for t in expanded_ordered if t not in forced]
        graph_top = (forced + rest)[:GRAPH_TOP_K]
    elif v in ('graph8_bfs',):
        # 实体锚点强制 + BFS 顺序（结构优先），8 节点
        graph_anchors = list(dict.fromkeys(ent_anchors + sem_anchors))
        expanded_ordered = ordered_expand(qid, graph_anchors, hops=GRAPH_HOPS)
        forced = [t for t in ent_anchors if t in expanded_ordered]
        rest = [t for t in expanded_ordered if t not in forced]
        graph_top = (forced + rest)[:8]
    elif v in ('pure_graph8', 'sentence_gated8', 'pure_graph8_fs0', 'pure_graph8_fs_type',
               'pure_graph8_fs_type5', 'pure_graph8_fs_type10'):
        # 实体锚点强制 + 纯语义排序（无结构分）+ 更多图节点（≤8）
        graph_anchors = list(dict.fromkeys(ent_anchors + sem_anchors))
        expanded = bfs_expand(qid, graph_anchors, hops=GRAPH_HOPS)

        def sem_only(t):
            e = node_emb(qid, t)
            return float(np.dot(QID2EMB[qid], e)) if e is not None else -1.0

        ordered = sorted(expanded, key=sem_only, reverse=True)
        forced = [t for t in ent_anchors if t in set(expanded)]
        rest = [t for t in ordered if t not in forced]
        graph_top = (forced + rest)[:8]
    else:
        expanded = bfs_expand(qid, graph_anchors, hops=GRAPH_HOPS)

        def graph_score(t):
            sem_score = float(np.dot(QID2EMB[qid], node_emb(qid, t))) if node_emb(qid, t) is not None else 0.0
            nbs = graph_adjacency(qid).get(t, set())
            struct_score = 0.0
            if nbs:
                rel_nb = sum(1 for nb in nbs if nb in set(graph_anchors))
                struct_score = 0.5 * (rel_nb / len(nbs)) + 0.5 * min(1.0, len(nbs) / 4.0)
            return SEMANTIC_WEIGHT * sem_score + STRUCTURAL_WEIGHT * struct_score

        reranked_anchors = sorted(expanded, key=graph_score, reverse=True)

        # 修复 1：强制保留显式实体锚点（rerank 只能排先后，不能删除）
        if v in ('anchor_retain', 'v2', 'anchor_fs0'):
            expanded_set = set(expanded)
            forced = [t for t in ent_anchors if t in expanded_set]
            rest = [t for t in reranked_anchors if t not in forced]
            reranked_anchors = forced + rest

        graph_top = reranked_anchors[:GRAPH_TOP_K]

    node_txt = node_text(qid)
    adj = graph_adjacency(qid)
    use_sentence_ev = v in ('sentence_ev', 'v2')

    def graph_line(t):
        nbs = ', '.join(sorted(adj.get(t, set()))[:4]) or 'none'
        if use_sentence_ev:                      # 修复 4：句子化
            txt = ' '.join(s['text'] for s in node_top_sentences(qid, t, 2))
        else:
            txt = node_txt.get(t, '')
        return f'[Graph: {t} links to {nbs}] {txt}'

    # ---------- 5) 动态 few-shot（fs0 变体跳过；fs_type 按题型选） ----------
    fs_examples = []
    if v not in ('anchor_fs0', 'no_rerank_fs0', 'pure_graph8_fs0'):
        qv = QID2EMB[qid]
        if v in ('pure_graph8_fs_type', 'pure_graph8_fs_type5', 'pure_graph8_fs_type10'):
            # 规则预测题型（不用 gold type，协议合规）
            tgt = predict_type(question)
            fs_k = {'pure_graph8_fs_type': 3,
                    'pure_graph8_fs_type5': 5,
                    'pure_graph8_fs_type10': 10}[v]
            pool = [q for q in FEWSHOT if normalize_type(q.get('type', '')) == tgt]
            picked = sorted(pool, key=lambda q: float(QID2EMB[q['id']] @ qv),
                            reverse=True)[:fs_k]
        else:
            fs_emb = np.array([QID2EMB[fq] for fq in FEWSHOT_QIDS])
            fs_scores = fs_emb @ qv
            fs_order = np.argsort(-fs_scores)[:FEWSHOT_TOP_K]
            fs_map = {q['id']: q for q in FEWSHOT}
            picked = [fs_map[FEWSHOT_QIDS[j]] for j in fs_order]
        for fq in picked:
            fq_hits = dense_search(fq['id'], fq['question'])
            fs_ev = ' '.join(SENTENCES[i]['text'] for i, _ in fq_hits[:3])
            fs_examples.append({'question': fq['question'], 'evidence': fs_ev,
                                'answer': fq['answer']})

    # ---------- 6) 融合（修复 2：文本配额；pure_graph8/句子化走专门分支） ----------
    if v in ('pure_graph8', 'sentence_gated8', 'graph8_bfs',
             'pure_graph8_fs0', 'pure_graph8_fs_type',
             'pure_graph8_fs_type5', 'pure_graph8_fs_type10'):
        if v == 'sentence_gated8':
            # 图内句子：每节点取 top-2，全局按余弦取 top-8
            pool = []
            for t in graph_top:
                for sc, i, s in node_sentences_scored(qid, t)[:2]:
                    pool.append((sc, s))
            pool.sort(key=lambda x: -x[0])
            fused_pool = [('gsent', s['text'],
                           (s['title'], s['sentence_id'], s['text']))
                          for sc, s in pool[:8]]
        else:
            # 8 个图节点段落全进证据，无文本
            fused_pool = [('graph', graph_line(t), t) for t in graph_top]
    else:
        fused_pool = []
        graph_cap = 3 if v in ('text_quota', 'v2') else len(graph_top)
        for t in graph_top[:graph_cap]:
            fused_pool.append(('graph', graph_line(t), t))
        for i, _ in text_ranked:
            fused_pool.append(('text', SENTENCES[i]['text'], i))

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
    text, itok, otok = complete(LLM_MODEL, v, qid, prompt, GEN_CFG)

    # ---------- 7) 输出记录（与基线同格式） ----------
    g_all = get_graph(qid)
    node_sent_ids = {n['title']: n.get('sentence_ids', []) for n in g_all['nodes']}
    ev = []
    for kind, rec in fused_records:
        if kind == 'graph':
            sids = node_sent_ids.get(rec, [])
            ev.append({'evidence_id': f'graph::{rec}', 'title': rec,
                       'sentence_id': sids[0] if sids else -1,
                       'sentence_ids': list(sids), 'text': node_txt.get(rec, ''),
                       'source': 'graph'})
        elif kind == 'gsent':
            gtitle, gsid, gtxt = rec
            ev.append({'evidence_id': f'gsent::{gtitle}::{gsid}', 'title': gtitle,
                       'sentence_id': gsid, 'text': gtxt, 'source': 'graph_sent'})
        else:
            i = rec
            src = 'dense' if i in dense_set else ('bm25' if i in bm25_set else 'rrf')
            ev.append({'evidence_id': SENTENCES[i]['evidence_id'],
                       'title': SENTENCES[i]['title'],
                       'sentence_id': SENTENCES[i]['sentence_id'],
                       'text': SENTENCES[i]['text'], 'source': src})
    return {'prediction': text, 'retrieved_evidence': ev, 'prompt': prompt,
            'input_tokens': itok, 'output_tokens': otok}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', required=True, choices=VARIANTS)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()

    v = args.variant
    outfile = os.path.join(PRED, f'{v}.jsonl')
    done = {}
    if os.path.exists(outfile) and not args.force:
        for line in open(outfile, encoding='utf-8'):
            try:
                r = json.loads(line)
                done[r['id']] = r
            except Exception:
                pass
        print(f'恢复进度: {len(done)} 条已完成')

    todo = [qa for qa in TEST_ROWS if qa['id'] not in done]
    if args.start:
        todo = todo[args.start:]
    if args.limit:
        todo = todo[:args.limit]
    print(f'变体={v}, 待跑 {len(todo)} 条')

    usage_reset()
    results = []
    for i, qa in enumerate(todo):
        t0 = time.time()
        try:
            res = run_variant(qa['id'], qa['question'], v, qa.get('type', ''))
            out = {
                'id': qa['id'], 'method': v, 'question': qa['question'],
                'gold_answer': qa['answer'], 'gold_type': normalize_type(qa['type']),
                'prediction': res['prediction'],
                'retrieved_evidence': res['retrieved_evidence'],
                'input_tokens': res['input_tokens'],
                'output_tokens': res['output_tokens'],
                'latency_sec': round(time.time() - t0, 3),
            }
        except Exception as e:
            print(f'  [{i}] {qa["id"]} 失败: {str(e)[:100]}', flush=True)
            continue
        results.append(out)
        with open(outfile, 'a', encoding='utf-8') as f:
            f.write(json.dumps(out, ensure_ascii=False) + '\n')
        if (i + 1) % 5 == 0 or i + 1 == len(todo):
            print(f'  完成 {i+1}/{len(todo)} 条', flush=True)

    us = usage_snapshot()
    print(f'完成: {v} {len(results)} 条 (调用 {us["calls"]} 次, '
          f'输入 {us["input_tokens"]} tokens, 输出 {us["output_tokens"]} tokens)')
    print(f'结果已追加到 {outfile}')


if __name__ == '__main__':
    main()
