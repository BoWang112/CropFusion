# -*- coding: utf-8 -*-
"""
graphrag/search.py —— GraphRAGEngine：Local Search + Global Search
====================================================================
论文对应：GraphRAG 对比实验（完整版）

Local Search：
  给定查询（病虫害名/症状），在图谱中找到锚点 → 子图扩展 → 三元组证据。
  复用 Retriever 的 _find_target_pest / _keyword_target_pests / directed_expand。

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
                 summary_top_k=8):
        """
        kg: CropPestKG
        retriever: Retriever（复用其图检索原语）
        llm: LLMClient
        summaries: list[dict] 社区摘要（community_id/summary/crops/keywords）
        community_of: dict[node_id -> community_id]（用于强制锚点社区优先）
        summary_vecs: 预编码摘要向量（numpy，形状 [n_summaries, dim]）
        """
        self.kg = kg
        self.retriever = retriever
        self.llm = llm
        self.summaries = summaries or []
        self.community_of = community_of or {}
        self.summary_vecs = summary_vecs
        self.summary_top_k = summary_top_k

    # ---------- Local Search ----------
    def local_search(self, query, hop=2):
        """Local Search：锚点 → 子图 → 三元组证据"""
        anchors = self._find_anchors(query)
        edges = set()
        nodes = set()
        anchor_names = []
        for a in anchors:
            n, e = self.retriever.directed_expand([a], hop=hop)
            if e:
                edges |= e
                nodes |= n
                anchor_names.append(self.kg.node_name(a))
        graph_evidence = []
        for src, dst, rel in edges:
            graph_evidence.append({
                'head': self.kg.node_name(src),
                'head_type': self.kg.node_entity(src),
                'rel': rel,
                'tail': self.kg.node_name(dst),
                'tail_type': self.kg.node_entity(dst),
                'anchor': self.kg.node_name(src) if src in anchors else (anchor_names[0] if anchor_names else None),
            })
        detail = {
            'variant': 'grag-local',
            'graph_evidence_count': len(graph_evidence),
            'graph_nodes_count': len(nodes),
            'anchors': anchor_names,
        }
        return [], graph_evidence, detail

    def _find_anchors(self, query):
        """Local Search 锚点解析（优先级递减）"""
        # 1. 题干显式病虫害名
        target = self.retriever._find_target_pest(query)
        if target:
            return list(target)[:3]
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
            nid = self.kg.find_node(cname)
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
        for name, nid in self.kg.norm_names:
            if self.kg.node_entity(nid) == '作物' and name and name in query:
                out.append(self.kg.norm_to_name.get(name, name))
        return out

    def global_search(self, query):
        """Global Search：map-reduce，返回 (text_ev, graph_ev, detail)"""
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
        } for s in selected[:5]]
        detail = {
            'variant': 'grag-global',
            'graph_evidence_count': len(graph_evidence),
            'communities_used': used_communities,
            'selected_communities': [s.get('community_id') for s in selected],
            'map_points': points,
        }
        return [], graph_evidence, answer, detail
