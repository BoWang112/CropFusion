# -*- coding: utf-8 -*-
"""
graphrag/gretriever.py —— G-Retriever 基线（PCST 子图选择）
================================================================
论文对应：G-Retriever (NeurIPS 2024) —— 问题感知的图检索。

G-Retriever 原版用可训练的 GNN 对节点/边打分，并以 Prize-Collecting
Steiner Tree (PCST) 选择与问题相关且结构连通的子图。

本基线（预构建图谱适配版）：
- BGE 嵌入计算“节点收益”（node prize）：问题向量与节点文本的余弦相似度，
  显式锚点额外加成；
- 三元组与问题的余弦相似度定义“边代价”：越相关代价越低；
- 贪心 PCST：从收益最高的锚点出发，每次加入
  gain = prize(新节点) - cost(边) 最大的相邻边，直到预算用尽或
  gain <= 0（保证子图连通、紧凑且问题相关）。
"""
import os
import sys
import hashlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class GRetrieverEngine:
    """G-Retriever 简化基线（问题感知 + PCST 子图选择）"""

    def __init__(self, kg, retriever, top_k=5, hop=2,
                 max_nodes=60, max_edges=100, anchor_bonus=0.5):
        self.kg = kg
        self.retriever = retriever
        self.top_k = top_k
        self.hop = hop
        self.max_nodes = max_nodes
        self.max_edges = max_edges
        self.anchor_bonus = anchor_bonus
        self._node_vec_cache = {}
        self._init_pest_vecs()

    def _init_pest_vecs(self):
        """预编码病虫害节点的文本块向量（问题感知锚点打分用）。"""
        self.pest_ids = []
        self.pest_nid = []
        self.pest_vecs = None
        pest_blocks = []
        seen_nids = set()
        for b in self.retriever.blocks.blocks:
            if b.get('entity_id') and b['entity_type'] in ('病害', '虫害'):
                if b['entity_id'] not in seen_nids:
                    seen_nids.add(b['entity_id'])
                    pest_blocks.append(b)
        if not pest_blocks:
            if self.retriever.pest_vecs is not None:
                self.pest_vecs = self.retriever.pest_vecs
                self.pest_ids = self.retriever.pest_vec_ids
                self.pest_nid = [self.retriever.block_by_id[i]['entity_id']
                                 for i in self.retriever.pest_vec_ids]
            return
        self.pest_ids = [b['id'] for b in pest_blocks]
        self.pest_nid = [b['entity_id'] for b in pest_blocks]
        texts = [b['text'] for b in pest_blocks]
        self.pest_vecs = self.retriever.embedder.encode(texts, batch_size=64)

    def _score_anchors(self, query):
        """问题感知锚点打分：查询向量与病虫害文本块向量的余弦相似度。"""
        if self.pest_vecs is None:
            return []
        qv = self.retriever.embedder.encode_query(query)
        scores = self.pest_vecs @ qv
        order = sorted(
            range(len(scores)),
            key=lambda i: (-float(scores[i]),
                           self.kg.node_name(self.pest_nid[i]) or ''),
        )
        anchors = []
        seen = set()
        for i in order:
            nid = self.pest_nid[i]
            if nid in seen:
                continue
            seen.add(nid)
            anchors.append(nid)
            if len(anchors) >= self.top_k:
                break
        return anchors

    # ---------- 节点收益 / 边代价 ----------
    def _node_vectors(self, nids):
        """节点文本向量（带缓存）。"""
        import numpy as np
        if not nids:
            return None
        uncached = [n for n in nids if n not in self._node_vec_cache]
        if uncached:
            texts = [self.kg.node_name(n) or '' for n in uncached]
            vecs = self.retriever.embedder.encode(texts, batch_size=64)
            for nid, vec in zip(uncached, vecs):
                self._node_vec_cache[nid] = vec
        return np.stack([self._node_vec_cache[n] for n in nids])

    def _pcst_select(self, query, anchors):
        """贪心 PCST：选择连通、紧凑、问题相关的子图。

        返回 (selected_nodes, selected_edges, node_prizes, edge_costs)。
        """
        qv = self.retriever.embedder.encode_query(query)
        candidate_nodes = set(anchors)
        candidate_edges = set()
        for anchor in anchors:
            nodes, edges = self.retriever._directed_expand([anchor], hop=self.hop)
            candidate_nodes |= set(nodes)
            candidate_edges |= set(edges)
        candidate_nodes = {n for n in candidate_nodes
                           if self.kg.node_entity(n) in self.retriever.evidence_entities}
        if not candidate_nodes:
            return set(), set(), {}, {}

        node_ids = list(candidate_nodes)
        node_vecs = self._node_vectors(node_ids)
        node_sims = node_vecs @ qv
        prizes = {}
        for nid, sim in zip(node_ids, node_sims):
            prize = max(0.0, float(sim))
            if nid in anchors:
                prize += self.anchor_bonus
            prizes[nid] = prize

        edge_list = [e for e in candidate_edges
                     if e[0] in candidate_nodes and e[1] in candidate_nodes]
        edge_list = sorted(edge_list, key=lambda e: (e[0], e[1], e[2]))
        if not edge_list:
            return set(), set(), prizes, {}
        edge_vecs = self.retriever._encode_triples(edge_list)
        edge_sims = edge_vecs @ qv
        costs = {}
        for edge, sim in zip(edge_list, edge_sims.tolist()):
            costs[edge] = max(0.1, 1.0 - float(sim))

        # 贪心 PCST：从收益最高锚点开始，加入 gain 最大的相邻边/节点
        roots = list(anchors) or list(node_ids)
        root = sorted(
            roots,
            key=lambda n: (-prizes.get(n, 0.0), self.kg.node_name(n) or ''),
        )[0]
        selected_nodes = {root}
        selected_edges = set()
        while len(selected_nodes) < self.max_nodes and len(selected_edges) < self.max_edges:
            best = None
            best_gain = 0.0
            for edge in edge_list:
                if edge in selected_edges:
                    continue
                u, v, _rel = edge
                if u in selected_nodes and v not in selected_nodes:
                    new_node = v
                elif v in selected_nodes and u not in selected_nodes:
                    new_node = u
                else:
                    continue
                gain = prizes.get(new_node, 0.0) - costs.get(edge, 1.0)
                if gain > best_gain:
                    best_gain = gain
                    best = (edge, new_node)
            if best is None:
                break
            edge, new_node = best
            selected_edges.add(edge)
            selected_nodes.add(new_node)

        # 兜底：若没有任何正收益边，至少保留根锚点的一跳邻域
        if not selected_edges:
            for edge in edge_list:
                if edge[0] == root or edge[1] == root:
                    selected_edges.add(edge)
                    selected_nodes.add(edge[0])
                    selected_nodes.add(edge[1])
                    if len(selected_edges) >= self.max_edges:
                        break
        # MultiDiGraph：同一实体对可能存在多条不同关系。PCST 生成树
        # 每个节点对只会加入一条边，这里补回已选节点之间的全部关系边，
        # 按边代价（越低越相关）稳定排序，受 max_edges 预算约束。
        if selected_nodes:
            extra_edges = [
                e for e in edge_list
                if e not in selected_edges
                and e[0] in selected_nodes and e[1] in selected_nodes
            ]
            extra_edges.sort(key=lambda e: (costs.get(e, 1.0), str(e)))
            for edge in extra_edges:
                if len(selected_edges) >= self.max_edges:
                    break
                selected_edges.add(edge)
        return selected_nodes, selected_edges, prizes, costs

    def search(self, query):
        """问题感知图检索：PCST 选子图 → 三元组证据。"""
        anchors = self._score_anchors(query)
        nodes, edges, prizes, costs = self._pcst_select(query, anchors)

        # 证据按边-查询相似度排序，保证返回顺序可复现且问题相关
        edge_list = list(edges)
        scored_edges = []
        if edge_list:
            qv = self.retriever.embedder.encode_query(query)
            edge_vecs = self.retriever._encode_triples(edge_list)
            sims = edge_vecs @ qv
            scored_edges = sorted(zip(edge_list, sims.tolist()),
                                  key=lambda x: (-x[1], x[0]))

        anchor_names = [self.kg.node_name(a) for a in anchors]
        pcst_id = 'pcst::' + hashlib.sha1(
            str(sorted(edges)).encode('utf-8')).hexdigest()[:12]
        graph_evidence = []
        for graph_rank, ((src, dst, rel), sim) in enumerate(scored_edges, start=1):
            graph_evidence.append({
                'head': self.kg.node_name(src),
                'head_type': self.kg.node_entity(src),
                'rel': rel,
                'tail': self.kg.node_name(dst),
                'tail_type': self.kg.node_entity(dst),
                'anchor': self.kg.node_name(src) if src in anchors
                          else (anchor_names[0] if anchor_names else None),
                'score': float(sim),
                'triple_score': float(sim),
                'graph_rank': graph_rank,
                'subgraph_rank': 1,
                'triple_rank': graph_rank,
                'source': 'g_retriever',
                'subgraph_id': pcst_id,
            })
        detail = {
            'variant': 'g-retriever',
            'graph_evidence_count': len(graph_evidence),
            'graph_nodes_count': len(nodes),
            'pcst_nodes': len(nodes),
            'pcst_edges': len(edges),
            'anchors': anchor_names,
            'node_prizes': {self.kg.node_name(k): round(v, 4)
                            for k, v in list(prizes.items())[:20]},
        }
        return [], graph_evidence, detail
