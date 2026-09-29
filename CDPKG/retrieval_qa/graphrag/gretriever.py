# -*- coding: utf-8 -*-
"""
graphrag/gretriever.py —— G-Retriever 基线（嵌入打分简化版）
================================================================
论文对应：G-Retriever (NeurIPS 2024) —— 问题感知的图检索。

G-Retriever 原版：微调一个问题感知的 GNN，对图中节点/边打分，
选择与问题相关的子图喂给 LLM。

本基线（简化版）：
- 用 BGE 嵌入对图谱中"病虫害实体节点"做问题感知打分
  （节点文本与问题向量的余弦相似度），替代可训练的 GNN
- 取 Top-K 相关节点为锚点，扩展子图，转成三元组证据
- 与 full 相比，它只走"图谱单通道"（无文本 RRF 融合），
  且锚点选择完全由问题-节点语义相关度驱动（问题感知）

检索证据结构与 Retriever.get_evidence 一致。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class GRetrieverEngine:
    """G-Retriever 简化基线（问题感知图检索）"""

    def __init__(self, kg, retriever, top_k=5, hop=2):
        """
        kg: CropPestKG
        retriever: Retriever（提供 pest_vecs/pest_block 预编码、directed_expand）
        top_k: 问题感知打分选 Top-K 锚点
        hop: 子图扩展跳数
        """
        self.kg = kg
        self.retriever = retriever
        self.top_k = top_k
        self.hop = hop
        # 预计算病虫害文本块向量（问题感知打分用）
        self._init_pest_vecs()

    def _init_pest_vecs(self):
        """预编码病虫害节点的文本块向量（问题感知打分用）

        不能只依赖 retriever.pest_vec_ids（索引构建时的旧列表，
        可能与当前图谱节点不完全对齐，导致部分病虫害被排除在打分候选外）。
        这里从当前 block_store 现算，保证覆盖全部有文本块的病虫害。
        """
        self.pest_ids = []
        self.pest_nid = []
        self.pest_vecs = None
        import numpy as np
        # 从当前 store 收集病虫害文本块
        pest_blocks = []
        seen_nids = set()
        for b in self.retriever.blocks.blocks:
            if b.get('entity_id') and b['entity_type'] in ('病害', '虫害'):
                if b['entity_id'] not in seen_nids:
                    seen_nids.add(b['entity_id'])
                    pest_blocks.append(b)
        if not pest_blocks:
            # 退化：用 retriever 预计算
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
        """问题感知打分：查询向量与每个病虫害节点文本块向量的余弦相似度"""
        if self.pest_vecs is None:
            return []
        qv = self.retriever.embedder.encode_query(query)
        scores = self.pest_vecs @ qv
        order = sorted(range(len(scores)), key=lambda i: -scores[i])
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

    def search(self, query):
        """问题感知图检索：打分选锚点 → 子图扩展 → 三元组证据"""
        anchors = self._score_anchors(query)
        edges = set()
        nodes = set()
        anchor_names = []
        for a in anchors:
            n, e = self.retriever.directed_expand([a], hop=self.hop)
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
                'anchor': self.kg.node_name(src) if src in anchors
                          else (anchor_names[0] if anchor_names else None),
            })
        detail = {
            'variant': 'g-retriever',
            'graph_evidence_count': len(graph_evidence),
            'graph_nodes_count': len(nodes),
            'anchors': anchor_names,
        }
        return [], graph_evidence, detail
