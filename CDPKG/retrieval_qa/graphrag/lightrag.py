# -*- coding: utf-8 -*-
"""
graphrag/lightrag.py —— LightRAG 基线（双层检索，预构建图谱适配版）
======================================================================
论文对应：LightRAG (EMNLP 2025 Findings) —— 简单快速的图增强 RAG。

LightRAG 原版：从语料自动构建图（实体/关系抽取 + 图结构双重去重），
检索时用"低层检索 + 高层检索"双层机制：
- 低层检索（Low-level）：精确匹配问题中的具体实体 → 检索实体级子图
- 高层检索（High-level）：主题/概念级聚合 → 检索跨实体全局子图
两者结果融合（配合文本块）喂给 LLM。

本基线（预构建图谱适配版）：
- 图是领域预构建知识图谱（非自动抽取）
- 低层检索：问题中具体实体（作物/病虫害）→ 实体子图
- 高层检索：按问题主题类型（防治/症状/部位/条件）聚合图谱知识
- 双层结果合并为图谱证据，无文本块（与 GraphRAG-Local 一致）
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class LightRAGEngine:
    """LightRAG 双层检索基线"""

    def __init__(self, kg, retriever, top_k=5, hop=2):
        self.kg = kg
        self.retriever = retriever
        self.top_k = top_k
        self.hop = hop
        # 主题类型 → 关系集（高层检索用）
        self.theme_rels = {
            '防治': {'防治方法有'},
            '症状': {'症状有'},
            '部位': {'为害', '发病于'},
            '条件': {'处于', '胁迫有', '致使', '影响', '发病于作物'},
            '时期': {'发病于作物'},
            '传播': {'传播'},
        }

    # ---------- 低层检索（实体级） ----------
    def _low_level(self, query):
        """精确匹配问题中的具体实体 → 实体子图"""
        anchors = []
        # 1. 病虫害名精确匹配
        target = self.retriever._find_target_pest(query)
        if target:
            anchors = list(target)[:3]
        else:
            # 2. 作物实体
            for name, nid in self.kg.norm_names:
                if self.kg.node_entity(nid) == '作物' and name in query:
                    anchors.append(nid)
                    break
            # 3. BM25 症状推断
            if not anchors:
                anchors = self.retriever._keyword_target_pests(query, top_k=2)
        return anchors

    # ---------- 高层检索（主题级） ----------
    def _high_level(self, query, seed_nodes):
        """按问题主题类型聚合图谱知识。

        以低层检索的锚点（seed_nodes）为中心，聚合这些节点及其
        一跳邻居的主题相关边（跨实体信息），避免全局扫描导致
        主题边爆炸稀释目标。
        """
        # 判断问题主题
        theme = '防治'  # 默认
        if any(k in query for k in ['症状', '表现', '危害']):
            theme = '症状'
        elif any(k in query for k in ['部位', '哪里', '危害哪些']):
            theme = '部位'
        elif any(k in query for k in ['条件', '温湿', '环境', '什么时候']):
            theme = '条件'
        elif any(k in query for k in ['防治', '用药', '药剂', '怎么治']):
            theme = '防治'
        elif any(k in query for k in ['时期', '阶段']):
            theme = '时期'
        elif any(k in query for k in ['传播', '传染']):
            theme = '传播'
        rels = self.theme_rels.get(theme, {'防治方法有'})
        # 以 seed_nodes 为中心扩展一跳，聚合主题边
        edges = set()
        nodes = set()
        frontier = set(seed_nodes)
        for nid in frontier:
            for tgt, rel in self.kg.neighbors(nid):
                if rel in rels:
                    edges.add((nid, tgt, rel))
                    nodes.add(nid)
                    nodes.add(tgt)
        return nodes, edges

    # ---------- 主入口 ----------
    def search(self, query):
        """双层检索：低层 + 高层子图合并"""
        # 低层：实体子图
        low_anchors = self._low_level(query)
        edges = set()
        nodes = set()
        anchor_names = []
        for a in low_anchors:
            n, e = self.retriever.directed_expand([a], hop=self.hop)
            if e:
                edges |= e
                nodes |= n
                anchor_names.append(self.kg.node_name(a))
        # 高层：主题聚合（以低层锚点为中心补充跨实体信息）
        hl_nodes, hl_edges = self._high_level(query, low_anchors)
        edges |= hl_edges
        nodes |= hl_nodes

        graph_evidence = []
        for src, dst, rel in edges:
            anchor = self.kg.node_name(src) if src in low_anchors \
                else (anchor_names[0] if anchor_names else '主题聚合')
            graph_evidence.append({
                'head': self.kg.node_name(src),
                'head_type': self.kg.node_entity(src),
                'rel': rel,
                'tail': self.kg.node_name(dst),
                'tail_type': self.kg.node_entity(dst),
                'anchor': anchor,
            })
        detail = {
            'variant': 'lightrag',
            'graph_evidence_count': len(graph_evidence),
            'graph_nodes_count': len(nodes),
            'low_anchors': anchor_names,
            'theme': None,
        }
        return [], graph_evidence, detail
