# -*- coding: utf-8 -*-
"""
kg.py —— 知识图谱图结构（Graph View）
=======================================
从 nodes.csv / relationships.csv 构建 networkx 有向图，
提供邻域扩展、最短路径、实体别名查询等图谱检索原语。

对应论文 3.3.1 (1) Graph view representation 与
3.3.4 语义/实体驱动图检索的图结构部分。
"""
import csv
import json
import os
import sys
import networkx as nx
from collections import defaultdict

# 使模块在作为独立脚本 / 包成员时都能定位到顶层 config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (  # noqa: E402
    NODES_CSV, RELS_CSV,
)


class CropPestKG:
    """作物病虫害知识图谱（内存图结构）"""

    def __init__(self, nodes_csv=None, rels_csv=None):
        nodes_csv = nodes_csv or NODES_CSV
        rels_csv = rels_csv or RELS_CSV
        self.graph = nx.DiGraph()
        self.name_to_id = {}            # 规范名称 -> node id（同名取第一个）
        self.alias_to_id = {}           # 别名 -> node id（构建时注册）
        self.node_by_id = {}
        self._load(nodes_csv, rels_csv)
        # 预计算的规范化名称列表（长优先，用于实体抽取加速）
        norm = self._normalize
        self.norm_names = sorted(
            [(norm(name), nid) for name, nid in self.name_to_id.items()
             if len(name) >= 2],
            key=lambda x: -len(x[0]),
        )
        # 规范化名称 -> 原始名称（字典融合时还原用）
        self.norm_to_name = {}
        for name, nid in self.name_to_id.items():
            nn = norm(name)
            if nn and nn not in self.norm_to_name:
                self.norm_to_name[nn] = name

    @staticmethod
    def _normalize(s):
        import re
        return re.sub(r'[\s，。、；：？！"“”‘’()《》·-]+', '', str(s or '').lower())

    # ---------- 加载 ----------
    def _load(self, nodes_csv, rels_csv):
        with open(nodes_csv, encoding='utf-8-sig', newline='') as f:
            for row in csv.DictReader(f):
                nid = row['id:ID']
                name = (row['name'] or '').strip()
                ent = (row['entity'] or '').strip()
                props = {}
                raw = row.get('props', '')
                if raw:
                    try:
                        props = json.loads(raw)
                    except Exception:
                        props = {}
                self.node_by_id[nid] = {
                    'id': nid, 'name': name, 'entity': ent, 'props': props,
                }
                self.graph.add_node(nid, name=name, entity=ent, props=props)
                if name and name not in self.name_to_id:
                    self.name_to_id[name] = nid
                # 注册别名
                for k in ('别名', '别名/俗名', '俗名'):
                    v = props.get(k)
                    if v:
                        for a in str(v).split(','):
                            a = a.strip()
                            if a and a not in self.alias_to_id:
                                self.alias_to_id[a] = nid

        with open(rels_csv, encoding='utf-8-sig', newline='') as f:
            for row in csv.DictReader(f):
                s, r, e = row[':START_ID'], row[':TYPE'], row[':END_ID']
                props = {}
                raw = row.get('props', '')
                if raw:
                    try:
                        props = json.loads(raw)
                    except Exception:
                        props = {}
                if self.graph.has_node(s) and self.graph.has_node(e):
                    self.graph.add_edge(s, e, reltype=r, props=props)

    # ---------- 基础查询 ----------
    def node_name(self, nid):
        n = self.node_by_id.get(nid)
        return n['name'] if n else nid

    def node_entity(self, nid):
        n = self.node_by_id.get(nid)
        return n['entity'] if n else None

    def node(self, nid):
        return self.node_by_id.get(nid)

    def find_node(self, name):
        """按规范名称 / 别名定位节点 id，返回 None 表示未找到"""
        if name in self.name_to_id:
            return self.name_to_id[name]
        return self.alias_to_id.get(name)

    def neighbors(self, nid, reltype=None):
        """出边邻接 (目标id, 关系类型)"""
        out = []
        for _, tgt, data in self.graph.out_edges(nid, data=True):
            if reltype is None or data.get('reltype') == reltype:
                out.append((tgt, data.get('reltype')))
        return out

    def in_neighbors(self, nid, reltype=None):
        out = []
        for src, _, data in self.graph.in_edges(nid, data=True):
            if reltype is None or data.get('reltype') == reltype:
                out.append((src, data.get('reltype')))
        return out

    def entities_of_type(self, ent_type):
        """返回某实体类型的全部节点 id 列表"""
        return [nid for nid, d in self.graph.nodes(data=True)
                if d.get('entity') == ent_type]

    # ---------- 图检索原语（对应 3.3.4） ----------
    def ego_subgraph(self, anchors, hop=2, exclude_entities=None):
        """锚点邻域扩展：对每个锚点取 hop 跳子图，合并为候选子图。

        返回子图内节点 id 集合、边集合（(src, dst, reltype)）。
        exclude_entities: 需排除的实体类型（如防治类），避免证据过度膨胀。
        """
        # BFS 精确计算 hop 跳邻域
        final_nodes = set(anchors)
        frontier = set(anchors)
        for _ in range(hop):
            nxt = set()
            for n in frontier:
                for tgt in self.graph.successors(n):
                    if tgt not in final_nodes:
                        final_nodes.add(tgt)
                        nxt.add(tgt)
                for src in self.graph.predecessors(n):
                    if src not in final_nodes:
                        final_nodes.add(src)
                        nxt.add(src)
            frontier = nxt
        # 排除指定实体类型
        if exclude_entities:
            final_nodes = {n for n in final_nodes
                           if self.node_entity(n) not in exclude_entities}
        edges = set()
        for u, v, d in self.graph.edges(data=True):
            if u in final_nodes and v in final_nodes:
                edges.add((u, v, d.get('reltype')))
        return final_nodes, edges

    def paths_between(self, anchors, max_len=3):
        """锚点之间实际存在的最短路径集合（路径长度 <= max_len）。
        返回 (src, dst, [node_ids]) 列表。
        """
        result = []
        anchors = list(anchors)
        for i in range(len(anchors)):
            for j in range(i + 1, len(anchors)):
                a, b = anchors[i], anchors[j]
                try:
                    for path in nx.all_simple_paths(self.graph, a, b, cutoff=max_len):
                        result.append((a, b, path))
                        break  # 每个节点对只取一条最短路径
                except (nx.NetworkXNoPath, nx.NetworkXError):
                    pass
        return result
