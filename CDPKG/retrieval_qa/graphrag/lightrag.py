# -*- coding: utf-8 -*-
"""
graphrag/lightrag.py —— LightRAG 基线（双层图检索，预构建图谱适配版）
======================================================================
论文对应：LightRAG (EMNLP 2025 Findings) —— 简单快速的图增强 RAG。

本基线（预构建图谱适配版）实现：
- 低层检索（Low-level）：显式实体链接 + 作物 + BM25 关键词 + 语义锚点，
  聚合具体实体子图；
- 高层检索（High-level）：按问题主题（防治/症状/部位/条件/时期/传播）
  在种子节点周围 2 跳内聚合跨实体关系边；
- 两层结果合并后，用“三元组-查询余弦相似度”统一排序并裁剪，
  输出问题相关的图谱证据。
"""
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class LightRAGEngine:
    """LightRAG 双层检索基线"""

    def __init__(self, kg, retriever, top_k=5, hop=2,
                 per_theme_edges=40, max_edges=80):
        self.kg = kg
        self.retriever = retriever
        self.top_k = top_k
        self.hop = hop
        self.per_theme_edges = per_theme_edges
        self.max_edges = max_edges
        self.theme_rels = {
            '防治': {'防治方法有'},
            '症状': {'症状有'},
            '部位': {'为害', '发病于'},
            '条件': {'处于', '胁迫有', '致使', '影响', '发病于'},
            '时期': {'发病于', '易感时间', '处于'},
            '传播': {'传播', '越冬情况'},
        }

    # ---------- 低层检索（实体级） ----------
    def _low_level(self, query):
        """显式实体 + 作物 + 关键词 + 语义锚点（多源融合）。"""
        anchors = []
        target = self.retriever._find_target_pest(query)
        if target:
            anchors.extend(sorted(target,
                                  key=lambda n: self.kg.node_name(n) or '')[:3])
        for name, nid in self.kg.norm_names:
            if self.kg.node_entity(nid) == '作物' and name and name in query:
                anchors.append(nid)
                for tgt, _ in self.kg.neighbors(nid):
                    if self.kg.node_entity(tgt) in ('病害', '虫害', '害虫'):
                        anchors.append(tgt)
                break
        anchors.extend(self.retriever._keyword_target_pests(query, top_k=2))
        sem_anchors, _ = self.retriever.semantic_graph_search(
            query, top_k=max(self.top_k, 3))
        anchors.extend(sem_anchors)
        anchors = [n for n in dict.fromkeys(anchors)
                   if self.kg.node_entity(n) in ('病害', '虫害', '害虫', '作物')]
        return anchors[:max(self.top_k, 5)]

    # ---------- 高层检索（主题级） ----------
    def _detect_theme(self, query):
        if any(k in query for k in ['症状', '表现', '危害']):
            return '症状'
        if any(k in query for k in ['部位', '哪里', '危害哪些']):
            return '部位'
        if any(k in query for k in ['条件', '温湿', '环境', '什么时候']):
            return '条件'
        if any(k in query for k in ['防治', '用药', '药剂', '怎么治', '防控']):
            return '防治'
        if any(k in query for k in ['时期', '阶段']):
            return '时期'
        if any(k in query for k in ['传播', '传染']):
            return '传播'
        return '防治'

    def _high_level(self, query, seed_nodes):
        """按主题在种子节点周围聚合跨实体关系边（2 跳）。"""
        theme = self._detect_theme(query)
        rels = self.theme_rels.get(theme, {'防治方法有'})
        edges = set()
        nodes = set(seed_nodes)
        frontier = set(seed_nodes)
        visited = set()
        for _ in range(self.hop):
            nxt = set()
            for nid in frontier:
                for tgt, rel in self.kg.neighbors(nid):
                    if rel in rels:
                        edges.add((nid, tgt, rel))
                        nodes.add(nid)
                        nodes.add(tgt)
                        nxt.add(tgt)
                for src, rel in self.kg.in_neighbors(nid):
                    if rel in rels:
                        edges.add((src, nid, rel))
                        nodes.add(src)
                        nodes.add(nid)
                        nxt.add(src)
            visited |= frontier
            frontier = nxt - visited

        # 用查询向量对高层边打分，保留主题内最相关的边
        edge_list = list(edges)
        scored = []
        if edge_list:
            qv = self.retriever.embedder.encode_query(query)
            vecs = self.retriever._encode_triples(edge_list)
            sims = vecs @ qv
            scored = sorted(zip(edge_list, sims.tolist()),
                            key=lambda x: (-x[1], x[0]))
            scored = scored[:self.per_theme_edges]
        return nodes, scored, theme

    # ---------- 主入口 ----------
    def search(self, query):
        """双层检索：低层实体子图 + 高层主题聚合，统一按查询相关性排序。"""
        low_anchors = self._low_level(query)
        edges = set()
        nodes = set(low_anchors)
        anchor_names = []
        for a in low_anchors:
            n, e = self.retriever._directed_expand([a], hop=self.hop)
            if e:
                edges |= set(e)
                nodes |= set(n)
                anchor_names.append(self.kg.node_name(a))

        hl_nodes, hl_scored, theme = self._high_level(query, low_anchors)
        nodes |= hl_nodes
        edges |= {e for e, _s in hl_scored}

        # 统一打分排序（低层 + 高层合并后）
        edge_list = list(edges)
        graph_evidence = []
        if edge_list:
            qv = self.retriever.embedder.encode_query(query)
            vecs = self.retriever._encode_triples(edge_list)
            sims = vecs @ qv
            ordered = sorted(zip(edge_list, sims.tolist()),
                             key=lambda x: (-x[1], x[0]))
            for triple_rank, ((src, dst, rel), sim) in enumerate(
                    ordered[:self.max_edges], start=1):
                anchor = (self.kg.node_name(src) if src in low_anchors
                          else (anchor_names[0] if anchor_names else '主题聚合'))
                graph_evidence.append({
                    'head': self.kg.node_name(src),
                    'head_type': self.kg.node_entity(src),
                    'rel': rel,
                    'tail': self.kg.node_name(dst),
                    'tail_type': self.kg.node_entity(dst),
                    'anchor': anchor,
                    'score': round(float(sim), 6),
                    'triple_score': round(float(sim), 6),
                    'graph_rank': triple_rank,
                    'subgraph_rank': 1,
                    'triple_rank': triple_rank,
                    'source': 'lightrag',
                    'subgraph_id': f'lightrag::{theme}',
                })
        detail = {
            'variant': 'lightrag',
            'graph_evidence_count': len(graph_evidence),
            'graph_nodes_count': len(nodes),
            'low_anchors': anchor_names,
            'high_level_edges': len(hl_scored),
            'theme': theme,
        }
        return [], graph_evidence, detail
