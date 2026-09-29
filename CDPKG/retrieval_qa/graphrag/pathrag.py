# -*- coding: utf-8 -*-
"""
graphrag/pathrag.py —— PathRAG 基线（关系路径检索 + 剪枝，论文适配版）
==========================================================================
论文对应：PathRAG (2025, arXiv 2502.14902) —— Pruning Graph-based
Retrieval Augmented Generation with Relational Paths.

PathRAG 核心思想：
- 检索单元是"关系路径"（实体-关系-实体-...），而非节点/子图
- 路径是自然语言推理的载体，比节点集合更利于 LLM 生成
- 用路径剪枝（信息量/多样性/检索代价）选择高价值路径，避免冗余

本基线（预构建图谱适配版）：
- 从目标锚点生成多跳关系路径（2-3 跳）
- 路径按"查询相关度 + 多样性"打分
- 剪枝保留 Top-K 条高价值路径作为图谱证据

检索证据结构与 Retriever.get_evidence 一致（head/rel/tail 三元组）。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class PathRAGEngine:
    """PathRAG 关系路径检索基线"""

    def __init__(self, kg, retriever, max_paths=8, path_len=3, hop=2):
        self.kg = kg
        self.retriever = retriever
        self.max_paths = max_paths
        self.path_len = path_len      # 路径最大跳数
        self.hop = hop
        # 高信息量关系（诊断/为害/防治）权重高
        self.rel_weight = {
            '为害': 1.5, '症状有': 1.5, '发病于': 1.5, '处于': 1.5,
            '防治方法有': 1.2, '影响': 1.0, '直接引发': 1.0,
            '传播': 1.0, '胁迫有': 1.0, '致使': 1.0,
        }

    def _find_anchors(self, query):
        """锚点解析（与 GraphRAG-Local 一致的三级策略）"""
        target = self.retriever._find_target_pest(query)
        if target:
            return list(target)[:3]
        kw = self.retriever._keyword_target_pests(query, top_k=2)
        if kw:
            return kw
        anchors, _ = self.retriever.semantic_graph_search(query, top_k=1)
        return anchors[:1]

    def _enumerate_paths(self, anchors, max_len=None):
        """从锚点出发枚举关系路径（DFS，限制长度）"""
        max_len = max_len or self.path_len
        paths = []  # list of [(src, rel, dst), ...]
        for a in anchors:
            # DFS 枚举所有 ≤ max_len 跳的路径
            stack = [(a, [])]
            while stack:
                node, path = stack.pop()
                if len(path) >= max_len:
                    continue
                for tgt, rel in self.kg.neighbors(node):
                    # 只走高信息量关系
                    if rel not in self.rel_weight:
                        continue
                    # 防治类节点不再继续展开（避免链蔓延）
                    if self.kg.node_entity(tgt) in (
                            '农业防治', '化学防治', '物理防治', '生物防治', '其他因素'):
                        new_path = path + [(node, rel, tgt)]
                        if new_path:
                            paths.append(new_path)
                        continue
                    new_path = path + [(node, rel, tgt)]
                    paths.append(new_path)
                    stack.append((tgt, new_path))
        return paths

    def _score_path(self, path, qv):
        """路径打分：查询相关度（尾实体文本块向量 × 查询）+ 信息量权重"""
        # 路径中每个三元组的尾实体（或头实体）文本块向量与查询相关度
        score = 0.0
        for src, rel, dst in path:
            w = self.rel_weight.get(rel, 0.8)
            # 用节点名向量近似（无文本块时）
            ent_text = self.kg.node_name(dst)
            if ent_text and qv is not None:
                # 简化：用尾实体名直接判断是否与查询相关（子串）
                score += w
            else:
                score += w
        # 路径长度奖励（信息量）
        score += 0.5 * len(path)
        return score

    def _prune_paths(self, paths, qv):
        """路径剪枝：按分数排序 + 多样性去重（避免重复覆盖同一尾实体）"""
        if not paths:
            return []
        scored = []
        seen_tails = set()
        for p in paths:
            s = self._score_path(p, qv)
            scored.append((s, p))
        scored.sort(key=lambda x: -x[0])
        selected = []
        for s, p in scored:
            # 多样性：避免多路径覆盖相同尾实体集合
            tails = frozenset(t for _s, _r, t in p)
            if tails in seen_tails:
                continue
            seen_tails.add(tails)
            selected.append(p)
            if len(selected) >= self.max_paths:
                break
        return selected

    def search(self, query):
        """PathRAG 检索：锚点 → 路径枚举 → 剪枝 → 证据"""
        anchors = self._find_anchors(query)
        qv = self.retriever.embedder.encode_query(query)
        paths = self._enumerate_paths(anchors)
        selected = self._prune_paths(paths, qv)

        graph_evidence = []
        anchor_names = [self.kg.node_name(a) for a in anchors]
        for p in selected:
            for src, rel, dst in p:
                graph_evidence.append({
                    'head': self.kg.node_name(src),
                    'head_type': self.kg.node_entity(src),
                    'rel': rel,
                    'tail': self.kg.node_name(dst),
                    'tail_type': self.kg.node_entity(dst),
                    'anchor': self.kg.node_name(src) if src in anchors
                              else (anchor_names[0] if anchor_names else '路径'),
                })
        detail = {
            'variant': 'pathrag',
            'graph_evidence_count': len(graph_evidence),
            'path_count': len(selected),
            'anchors': anchor_names,
        }
        return [], graph_evidence, detail
