# -*- coding: utf-8 -*-
"""
graphrag/pathrag.py —— PathRAG 基线（查询相关路径检索 + 流式剪枝）
====================================================================
论文对应：PathRAG (2025, arXiv 2502.14902) —— Pruning Graph-based
Retrieval Augmented Generation with Relational Paths.

本基线（预构建图谱适配版）实现：
1) 从锚点枚举 1~max_len 跳的真实关系路径（禁止环）；
2) 路径评分真正使用查询向量：路径线性化文本与查询向量的余弦相似度，
   并融合关系信息量权重与路径长度因素；
3) 流式剪枝：把路径分数归一化为“流量”，在节点容量约束下按边际流量
   （流量 × 新颖度 × 拥塞惩罚）选择路径，抑制冗余与噪声路径；
4) 被选路径展开为三元组证据，保留 path_id/score 便于审计。
"""
import os
import sys
import hashlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class PathRAGEngine:
    """PathRAG 关系路径检索基线"""

    def __init__(self, kg, retriever, max_paths=8, path_len=3, hop=2,
                 max_candidates=2000, node_capacity=2, min_flow=0.05):
        self.kg = kg
        self.retriever = retriever
        self.max_paths = max_paths
        self.path_len = path_len      # 路径最大跳数
        self.hop = hop
        self.max_candidates = max_candidates
        self.node_capacity = node_capacity
        self.min_flow = min_flow
        # 高信息量关系（诊断/为害/防治/物候）权重高；
        # 与 Retriever.expand_rels 保持一致，避免路径枚举漏掉有效关系。
        self.rel_weight = {
            '为害': 1.5, '症状有': 1.5, '发病于': 1.5, '处于': 1.5,
            '防治方法有': 1.2, '影响': 1.0, '直接引发': 1.0,
            '传播': 1.0, '胁迫有': 1.0, '致使': 1.0,
            '越冬情况': 1.0, '世代情况': 1.0,
            '易感时间': 1.0, '易感区域': 1.0,
        }
        # 路径终点实体类型：只作为路径末端，不再继续扩展
        self.terminal_entities = {
            '具体症状', '发病部位', '发病时期', '气候因素', '环境因素',
            '栽培因素', '其他因素', '世代情况', '越冬情况', '传播方式',
            '初侵染源', '易感时间', '易感地区', '农业防治', '化学防治',
            '物理防治', '生物防治', '类似虫害', '分类地位', '拉丁文名称',
            '专业书籍', '权威网站', '作物类型',
        }
        self._path_vec_cache = {}

    def _find_anchors(self, query):
        """锚点解析（与 GraphRAG-Local 一致的三级策略）"""
        target = self.retriever._find_target_pest(query)
        if target:
            return sorted(target,
                          key=lambda n: self.kg.node_name(n) or '')[:3]
        kw = self.retriever._keyword_target_pests(query, top_k=2)
        if kw:
            return kw
        anchors, _ = self.retriever.semantic_graph_search(query, top_k=1)
        return anchors[:1]

    def _enumerate_paths(self, anchors, max_len=None):
        """从锚点出发 DFS 枚举关系路径（限制长度、禁止环、限制候选数）。"""
        max_len = max_len or self.path_len
        paths = []
        seen = set()
        for anchor in anchors:
            stack = [(anchor, [], {anchor})]
            while stack and len(paths) < self.max_candidates:
                node, path, visited = stack.pop()
                if len(path) >= max_len:
                    continue
                for tgt, rel in self.kg.neighbors(node):
                    if rel not in self.rel_weight:
                        continue
                    if tgt in visited:
                        continue
                    new_path = path + [(node, rel, tgt)]
                    key = tuple(new_path)
                    if key not in seen:
                        seen.add(key)
                        paths.append(new_path)
                    if self.kg.node_entity(tgt) not in self.terminal_entities:
                        stack.append((tgt, new_path, visited | {tgt}))
        return paths

    def _path_text(self, path):
        """路径线性化文本（用于查询相关性编码）。"""
        return ' ; '.join(
            f'{self.kg.node_name(src)} {rel} {self.kg.node_name(dst)}'
            for src, rel, dst in path)

    def _score_paths(self, paths, qv):
        """批量计算路径分数：查询余弦 + 关系权重 + 长度因素。"""
        if not paths:
            return []
        import numpy as np
        texts = [self._path_text(p) for p in paths]
        uncached = [t for t in texts if t not in self._path_vec_cache]
        if uncached:
            vecs = self.retriever.embedder.encode(uncached, batch_size=64)
            for text, vec in zip(uncached, vecs):
                self._path_vec_cache[text] = vec
        if qv is not None:
            vecs = np.stack([self._path_vec_cache[t] for t in texts])
            sims = vecs @ qv
        else:
            sims = [0.0] * len(texts)

        scores = []
        max_rel_weight = max(self.rel_weight.values()) if self.rel_weight else 1.0
        for path, sim in zip(paths, sims):
            rel_score = sum(self.rel_weight.get(rel, 0.8)
                            for _s, rel, _d in path) / (len(path) * max_rel_weight)
            length_score = 1.0 / len(path)
            scores.append(
                0.7 * float(sim) + 0.2 * rel_score + 0.1 * length_score)
        return scores

    def _flow_prune(self, paths, scores):
        """流式剪枝：归一化分数作为流量，在节点容量约束下选路径。

        - 节点容量 = node_capacity，避免大量路径共享同一枢纽节点；
        - 拥塞惩罚 = 已用容量 / 总容量；
        - 边际流量 = 归一化流量 × (0.5 + 0.5 × 新节点比例) × (1 - 拥塞)；
        - 低于 min_flow 或节点超容量的路径被剪掉。
        """
        if not paths:
            return []
        lo, hi = min(scores), max(scores)
        span = hi - lo
        normalized = [1.0 if span <= 1e-12 else (s - lo) / span for s in scores]
        order = sorted(range(len(paths)), key=lambda i: -scores[i])

        from collections import defaultdict
        usage = defaultdict(int)
        selected = []
        seen_paths = set()
        for idx in order:
            path = paths[idx]
            nodes = {n for s, _r, d in path for n in (s, d)}
            if any(usage[n] >= self.node_capacity for n in nodes):
                continue
            # 以完整关系路径（含关系类型）去重：同一实体对上的不同关系
            # 路径必须分别保留，不能仅按终点集合合并。
            path_key = tuple(path)
            if path_key in seen_paths:
                continue
            congestion = sum(usage[n] for n in nodes) / max(
                1, len(nodes) * self.node_capacity)
            novelty = sum(1 for n in nodes if usage[n] == 0) / max(1, len(nodes))
            marginal = normalized[idx] * (0.5 + 0.5 * novelty) * (1.0 - congestion)
            if marginal < self.min_flow:
                continue
            seen_paths.add(path_key)
            selected.append((scores[idx], path))
            for n in nodes:
                usage[n] += 1
            if len(selected) >= self.max_paths:
                break
        return selected

    def search(self, query):
        """PathRAG 检索：锚点 → 路径枚举 → 查询相关评分 → 流式剪枝 → 证据。"""
        anchors = self._find_anchors(query)
        qv = self.retriever.embedder.encode_query(query)
        paths = self._enumerate_paths(anchors)
        scores = self._score_paths(paths, qv)
        selected = self._flow_prune(paths, scores)

        anchor_names = [self.kg.node_name(a) for a in anchors]
        graph_evidence = []
        seen_triples = set()
        graph_rank = 0
        for path_rank, (score, path) in enumerate(selected, start=1):
            path_id = 'path::' + hashlib.sha1(
                str(tuple(path)).encode('utf-8')).hexdigest()[:12]
            triple_rank = 0
            for src, rel, dst in path:
                key = (src, rel, dst)
                if key in seen_triples:
                    continue
                seen_triples.add(key)
                triple_rank += 1
                graph_rank += 1
                graph_evidence.append({
                    'head': self.kg.node_name(src),
                    'head_type': self.kg.node_entity(src),
                    'rel': rel,
                    'tail': self.kg.node_name(dst),
                    'tail_type': self.kg.node_entity(dst),
                    'anchor': self.kg.node_name(src) if src in anchors
                              else (anchor_names[0] if anchor_names else '路径'),
                    'path_id': path_id,
                    'path_rank': path_rank,
                    'graph_rank': graph_rank,
                    'triple_rank': triple_rank,
                    'score': round(float(score), 6),
                    'triple_score': round(float(score), 6),
                    'source': 'pathrag',
                })
        detail = {
            'variant': 'pathrag',
            'graph_evidence_count': len(graph_evidence),
            'path_count': len(selected),
            'candidate_paths': len(paths),
            'anchors': anchor_names,
            'path_scores': [round(float(s), 6) for s, _p in selected],
        }
        return [], graph_evidence, detail
