# -*- coding: utf-8 -*-
"""
graphrag/search.py —— GraphRAGEngine：Local Search + Global Search
====================================================================
论文对应：GraphRAG 对比实验（完整版）

Local Search：
  给定查询（病虫害名/症状），在图谱中找到锚点 → 子图扩展 → 三元组证据。
  复用 Retriever 的 _find_target_pest / _keyword_target_pests / _directed_expand。
  同时按 GraphRAG Local Search 的证据组织方式返回：
    - 实体证据（锚点实体的 id/名称/类型/查询相关度）；
    - 关系证据（子图内三元组，带 anchor/anchor_id/subgraph_id）；
    - 文本证据（锚点实体在索引中的语义片段，字段与 Retriever.get_evidence
      的文本证据完全一致）。

Global Search：
  给定全局聚合问题（如"水稻相关病害的传播方式"），
  先按社区摘要与查询的语义相似度选 top-k 社区，
  再 map（每社区提炼要点）+ reduce（综合成最终回答）。

检索证据结构与 Retriever.get_evidence 一致，可直接喂给
build_prompt_with_dynamic_fewshot。
"""
import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graphrag.prompts import (GLOBAL_MAP_SYSTEM, GLOBAL_MAP_TEMPLATE,
                              GLOBAL_REDUCE_SYSTEM, GLOBAL_REDUCE_TEMPLATE)


class GraphRAGEngine:
    """GraphRAG 检索引擎（Local + Global）"""

    def __init__(self, kg, retriever, llm, summaries=None,
                 community_of=None, summary_vecs=None,
                 summary_top_k=8, text_top_k=5):
        """
        kg: CropPestKG
        retriever: Retriever（复用其图检索原语）
        llm: LLMClient
        summaries: list[dict] 社区摘要（community_id/summary/crops/keywords）
        community_of: dict[node_id -> community_id]（用于强制锚点社区优先）
        summary_vecs: 预编码摘要向量（numpy，形状 [n_summaries, dim]）
        text_top_k: Local Search 返回的锚点文本片段数量
        """
        self.kg = kg
        self.retriever = retriever
        self.llm = llm
        self.summaries = summaries or []
        self.community_of = community_of or {}
        self.summary_vecs = summary_vecs
        self.summary_top_k = summary_top_k
        self.text_top_k = text_top_k
        # 实体/文本片段向量缓存（同一节点/片段只编码一次）
        self._node_vec_cache = {}
        self._block_vec_cache = {}

    # ---------- Local Search ----------
    def local_search(self, query, hop=2, per_graph_edges=25):
        """Local Search：锚点 → 子图扩展 → 实体/关系/文本证据。

        返回 (text_evidence, graph_evidence, detail)，与
        Retriever.get_evidence 的返回接口一致，可直接进入统一证据评估。
        """
        anchors = list(dict.fromkeys(self._find_anchors(query)))
        qv = self.retriever.embedder.encode_query(query) if anchors else None

        # 每个锚点独立扩展，同时记录“边 -> 锚点”的归属，供证据溯源
        anchor_edges = {}
        edges = set()
        nodes = set()
        for anchor in anchors:
            sub_nodes, sub_edges = self.retriever._directed_expand(
                [anchor], hop=hop)
            anchor_edges[anchor] = set(sub_edges)
            edges |= set(sub_edges)
            nodes |= set(sub_nodes)

        graph_evidence = self._edge_evidence(qv, anchors, anchor_edges,
                                             edges, per_graph_edges)
        entity_evidence = self._entity_evidence(anchors, qv)
        text_evidence = self._anchor_text_evidence(anchors, qv)

        detail = {
            'variant': 'grag-local',
            'graph_evidence_count': len(graph_evidence),
            'text_evidence_count': len(text_evidence),
            'entity_evidence_count': len(entity_evidence),
            'graph_nodes_count': len(nodes),
            'candidate_edges': len(edges),
            'selected_edges': len(graph_evidence),
            'anchors': [self.kg.node_name(a) for a in anchors],
            'anchor_ids': list(anchors),
            'entities': entity_evidence,
        }
        return text_evidence, graph_evidence, detail

    def _edge_evidence(self, qv, anchors, anchor_edges, edges,
                       per_graph_edges):
        """关系证据：子图内三元组按查询相关度排序，并保留锚点归属。"""
        if not edges or qv is None:
            return []
        edge_list = sorted(edges, key=self._edge_sort_key)
        vecs = self.retriever._encode_triples(edge_list)
        sims = vecs @ qv
        scored_edges = sorted(
            zip(edge_list, sims.tolist()),
            key=lambda item: (-item[1], self._edge_sort_key(item[0])))
        max_edges = per_graph_edges * max(1, len(anchors))
        scored_edges = scored_edges[:max_edges]

        anchor_of_edge = {}
        for anchor in anchors:
            for edge in anchor_edges.get(anchor, ()):
                anchor_of_edge.setdefault(edge, anchor)

        anchor_rank = {a: i + 1 for i, a in enumerate(anchors)}
        per_anchor_count = {}
        graph_evidence = []
        for graph_rank, ((src, dst, rel), sim) in enumerate(scored_edges, start=1):
            anchor_id = anchor_of_edge.get((src, dst, rel))
            anchor = self.kg.node_name(anchor_id) if anchor_id is not None \
                else (self.kg.node_name(anchors[0]) if anchors else None)
            per_anchor_count[anchor_id] = per_anchor_count.get(anchor_id, 0) + 1
            graph_evidence.append({
                'head': self.kg.node_name(src),
                'head_type': self.kg.node_entity(src),
                'rel': rel,
                'tail': self.kg.node_name(dst),
                'tail_type': self.kg.node_entity(dst),
                'anchor': anchor,
                'anchor_id': anchor_id,
                'score': round(float(sim), 6),
                'triple_score': round(float(sim), 6),
                'graph_rank': graph_rank,
                'subgraph_rank': anchor_rank.get(anchor_id),
                'triple_rank': per_anchor_count[anchor_id],
                'source': 'graphrag_local',
                'subgraph_id': (f'grag::{anchor_id}' if anchor_id is not None
                                else 'grag::global'),
            })
        return graph_evidence

    def _entity_evidence(self, anchors, qv):
        """实体证据：锚点实体的名称/类型/查询相关度（可追溯）。"""
        if not anchors or qv is None:
            return []
        vecs = self._node_vectors(anchors)
        sims = vecs @ qv
        out = []
        for nid, sim in zip(anchors, sims.tolist()):
            out.append({
                'id': nid,
                'name': self.kg.node_name(nid),
                'type': self.kg.node_entity(nid),
                'score': round(float(sim), 6),
            })
        return out

    def _anchor_text_evidence(self, anchors, qv):
        """文本证据：锚点实体在文本索引中的语义片段。

        字段与 Retriever.get_evidence 的 text_evidence 保持一致
        （id/text/score/entity/entity_type/source/segment_id/...）。
        """
        if not anchors or qv is None:
            return []
        anchor_set = set(anchors)
        blocks = [b for b in self.retriever.blocks.blocks
                  if b.get('entity_id') in anchor_set]
        if not blocks:
            return []
        uncached = [b['id'] for b in blocks
                    if b['id'] not in self._block_vec_cache]
        if uncached:
            by_id = {b['id']: b for b in blocks}
            vecs = self.retriever.embedder.encode(
                [by_id[bid]['text'] for bid in uncached], batch_size=64)
            for bid, vec in zip(uncached, vecs):
                self._block_vec_cache[bid] = vec
        mat = np.stack([self._block_vec_cache[b['id']] for b in blocks])
        sims = mat @ qv
        order = sorted(range(len(blocks)),
                       key=lambda i: (-float(sims[i]), blocks[i]['id']))
        out = []
        for rank, i in enumerate(order[:self.text_top_k], start=1):
            block = blocks[i]
            meta = block.get('meta') or {}
            out.append({
                'id': block['id'],
                'text': block['text'],
                'score': round(float(sims[i]), 6),
                'rrf_rank': rank,
                'source': block.get('source'),
                'retriever_source': 'graphrag_local',
                'entity': block.get('entity_name'),
                'entity_type': block.get('entity_type'),
                'segment_id': meta.get('segment_id'),
                'record_index': meta.get('record_index'),
                'segment_index': meta.get('segment_index'),
                'source_id': meta.get('source_id'),
                'meta': meta,
            })
        return out

    def _node_vectors(self, nids):
        import numpy as np
        uncached = [n for n in nids if n not in self._node_vec_cache]
        if uncached:
            texts = [self.kg.node_name(n) or '' for n in uncached]
            vecs = self.retriever.embedder.encode(texts, batch_size=64)
            for nid, vec in zip(uncached, vecs):
                self._node_vec_cache[nid] = vec
        return np.stack([self._node_vec_cache[n] for n in nids])

    def _edge_sort_key(self, edge):
        """三元组 (src, dst, rel) 的稳定排序键。"""
        src, dst, rel = edge
        return (self.kg.node_name(src) or '', rel or '',
                self.kg.node_name(dst) or '')

    def _find_anchors(self, query):
        """Local Search 锚点解析（优先级递减，结果稳定排序）"""
        # 1. 题干显式病虫害名
        target = self.retriever._find_target_pest(query)
        if target:
            return sorted(target,
                          key=lambda n: self.kg.node_name(n) or '')[:3]
        # 2. BM25 症状块推断
        kw = self.retriever._keyword_target_pests(query, top_k=2)
        if kw:
            return kw
        # 3. 语义图检索 top1
        anchors, _ = self.retriever.semantic_graph_search(query, top_k=1)
        return anchors[:1]

    # ---------- Global Search ----------
    def global_evidence(self, query, top_k=None):
        """按社区摘要与查询的语义相似度选 top-k 社区"""
        top_k = top_k or self.summary_top_k
        if not self.summaries:
            return []
        if self.summary_vecs is None:
            self._encode_summaries()
        qv = self.retriever.embedder.encode_query(query)
        scores = self.summary_vecs @ qv
        order = sorted(range(len(scores)), key=lambda i: -scores[i])
        # 强制：查询含作物名 → 该作物所在社区排最前
        crop_hit = None
        for cname in self._extract_crops(query):
            nid = self.kg.find_node(cname, entity='作物')
            if nid and nid in self.community_of:
                crop_hit = self.community_of[nid]
                break
        if crop_hit is not None:
            order = [crop_hit] + [i for i in order if i != crop_hit]
        return [self.summaries[i] for i in order[:top_k]]

    def _encode_summaries(self):
        """编码社区摘要向量"""
        texts = []
        for s in self.summaries:
            parts = []
            if s.get('crops'):
                parts.append('作物：' + '、'.join(s['crops']))
            if s.get('keywords'):
                parts.append('关键词：' + '、'.join(s['keywords']))
            parts.append(s.get('summary', ''))
            texts.append('\n'.join(parts))
        vecs = self.retriever.embedder.encode(texts, batch_size=16)
        self.summary_vecs = np.stack(vecs)

    def _extract_crops(self, query):
        """从查询中提取作物名（遍历图谱作物节点）"""
        out = []
        for name, nid in self.kg.iter_name_nodes():
            if (self.kg.node_entity(nid) == '作物' and name
                    and name in query):
                out.append(self.kg.norm_to_name.get(name, name))
        return list(dict.fromkeys(out))

    def global_search(self, query):
        """Global Search：map-reduce，返回 (text_ev, graph_ev, answer, detail)"""
        selected = self.global_evidence(query)
        # Map：每社区提炼要点
        points = []
        used_communities = []
        for s in selected:
            prompt = GLOBAL_MAP_TEMPLATE.format(summary=s.get('summary', ''), question=query)
            resp = self.llm.complete(prompt, system=GLOBAL_MAP_SYSTEM,
                                     temperature=0.2, max_tokens=300)
            resp = (resp or '').strip()
            if resp and '无相关事实' not in resp:
                points.append(f'【社区C{s.get("community_id")}】\n{resp}')
                used_communities.append(s.get('community_id'))
        # Reduce：综合
        if points:
            reduce_prompt = GLOBAL_REDUCE_TEMPLATE.format(
                points='\n\n'.join(points), question=query)
            answer = self.llm.complete(reduce_prompt, system=GLOBAL_REDUCE_SYSTEM,
                                       temperature=0.2, max_tokens=400)
        else:
            answer = '根据图谱社区摘要，未找到与该问题直接相关的信息。'
        # 组装证据（社区摘要块）
        graph_evidence = [{
            'head': f'社区C{s.get("community_id")}',
            'rel': '社区摘要',
            'tail': (s.get('summary') or '')[:200],
            'anchor': '、'.join((s.get('crops') or [])[:3]),
            'subgraph_id': f'grag-global::C{s.get("community_id")}',
        } for s in selected[:5]]
        detail = {
            'variant': 'grag-global',
            'graph_evidence_count': len(graph_evidence),
            'communities_used': used_communities,
            'selected_communities': [s.get('community_id') for s in selected],
            'map_points': points,
        }
        return [], graph_evidence, answer, detail
