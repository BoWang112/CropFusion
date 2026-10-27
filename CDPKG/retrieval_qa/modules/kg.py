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


DEFAULT_TERMINAL_ENTITIES = {
    '具体症状', '发病部位', '发病时期', '气候因素', '环境因素', '栽培因素',
    '其他因素', '世代情况', '越冬情况', '传播方式', '初侵染源',
    '易感时间', '易感地区', '农业防治', '化学防治', '物理防治', '生物防治',
    '类似虫害', '分类地位', '拉丁文名称', '专业书籍', '权威网站', '作物类型',
}


class CropPestKG:
    """作物病虫害知识图谱（内存图结构）"""

    def __init__(self, nodes_csv=None, rels_csv=None):
        nodes_csv = nodes_csv or NODES_CSV
        rels_csv = rels_csv or RELS_CSV
        # 同一对实体之间可能存在多种关系（如 A-为害->B 与 A-影响->B），
        # 必须用多重有向图保存；关系类型作为边 key。
        self.graph = nx.MultiDiGraph()
        # 同名节点可能属于不同实体类型（如属性实体 vs 病虫害实体），
        # 因此保留“名称 -> 全部 node id”的完整索引；name_to_id 仍保留
        # 首个节点，兼容旧调用方。
        self.name_to_ids = {}           # 规范名称 -> [node id, ...]
        self.alias_to_ids = {}          # 别名 -> [node id, ...]
        self.name_to_id = {}            # 规范名称 -> node id（首个）
        self.alias_to_id = {}           # 别名 -> node id（首个）
        self.node_by_id = {}
        self._load(nodes_csv, rels_csv)
        # 预计算的规范化名称列表（长优先，用于实体抽取加速）。
        # 使用全部同名节点，避免只保留第一个类型而漏掉其它类型。
        norm = self._normalize
        pairs = []
        for name, ids in self.name_to_ids.items():
            if len(name) < 2:
                continue
            for nid in ids:
                pairs.append((norm(name), nid))
        self.norm_names = sorted(pairs, key=lambda x: -len(x[0]))
        # 规范化名称 -> 全部 node id（类型消歧前的候选集合）
        self.norm_to_ids = {}
        for name, ids in self.name_to_ids.items():
            nn = norm(name)
            if nn:
                self.norm_to_ids.setdefault(nn, []).extend(ids)
        # 规范化别名 -> 全部 node id（别名消歧兜底）
        self.norm_alias_to_ids = {}
        for alias, ids in self.alias_to_ids.items():
            nn = norm(alias)
            if nn:
                self.norm_alias_to_ids.setdefault(nn, []).extend(ids)
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
    @staticmethod
    def _merge_edge_props(old, new):
        """合并同一 (source, relation, target) 重复边的来源信息。

        - 来源ID/source_ids/sources 合并为去重列表；
        - 来源（中文标签串）按“、”拆分后去重合并；
        - 其它属性保留已存在的非空值。
        """
        if not new:
            return old
        for key, value in new.items():
            if key in ('来源ID', 'source_id', 'source_ids', 'sources'):
                existing = old.get(key, [])
                if not isinstance(existing, list):
                    existing = [existing] if existing else []
                values = value if isinstance(value, (list, tuple, set)) else [value]
                for item in values:
                    if item and item not in existing:
                        existing.append(item)
                old[key] = existing
            elif key == '来源':
                existing = old.get(key, '')
                labels = []
                for text in (existing, value):
                    if isinstance(text, (list, tuple, set)):
                        parts = text
                    else:
                        parts = str(text or '').split('、')
                    for part in parts:
                        part = str(part).strip()
                        if part and part not in labels:
                            labels.append(part)
                old[key] = '、'.join(labels)
            elif key not in old or old[key] in (None, '', [], {}):
                old[key] = value
        return old

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
                if name:
                    self.name_to_ids.setdefault(name, []).append(nid)
                    self.name_to_id.setdefault(name, nid)
                # 注册别名（同一别名可能对应多个不同类型节点）
                for k in ('别名', '别名/俗名', '俗名'):
                    v = props.get(k)
                    if v:
                        for a in str(v).split(','):
                            a = a.strip()
                            if a:
                                ids = self.alias_to_ids.setdefault(a, [])
                                if nid not in ids:
                                    ids.append(nid)
                                self.alias_to_id.setdefault(a, nid)

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
                    # 关系唯一标识 = (source_id, relation_type, target_id)；
                    # 同一三元组重复出现时合并来源，不重复建边。
                    edge_key = r
                    existing = self.graph.get_edge_data(s, e, key=edge_key)
                    if existing is not None:
                        merged = dict(existing.get('props') or {})
                        self._merge_edge_props(merged, props)
                        self.graph[s][e][edge_key]['props'] = merged
                    else:
                        self.graph.add_edge(s, e, key=edge_key,
                                            reltype=r, props=props)

    # ---------- 基础查询 ----------
    def node_name(self, nid):
        n = self.node_by_id.get(nid)
        return n['name'] if n else nid

    def node_entity(self, nid):
        n = self.node_by_id.get(nid)
        return n['entity'] if n else None

    def node(self, nid):
        return self.node_by_id.get(nid)

    @staticmethod
    def _id_sort_key(nid):
        """节点 id 的稳定排序键（CSV 中的 id 可能是数字字符串）。"""
        text = str(nid)
        try:
            return (0, int(text))
        except (TypeError, ValueError):
            return (1, text)

    @staticmethod
    def _context_tokens(context):
        if context is None:
            return []
        if isinstance(context, str):
            return [context.strip()] if context.strip() else []
        if isinstance(context, (list, tuple, set, frozenset)):
            return [str(c).strip() for c in context
                    if c is not None and str(c).strip()]
        return [str(context).strip()]

    def _node_context_text(self, nid):
        """节点可用于上下文匹配的文本（上下文键/领域/来源等属性）。"""
        node = self.node_by_id.get(nid) or {}
        props = node.get('props') or {}
        parts = []
        for key in ('上下文', '领域', '作物', '病虫害', '来源', '来源ID',
                    'source_ids', 'sources'):
            value = props.get(key)
            if isinstance(value, (list, tuple, set, frozenset)):
                parts.extend(str(v) for v in value if v)
            elif value:
                parts.append(str(value))
        return ' '.join(parts)

    def find_node(self, name, entity=None, context=None, aliases=True):
        """按规范名称 / 别名定位节点 id，支持实体类型与上下文消歧。

        参数：
          name    : 实体标准名或别名。
          entity  : 期望的实体类型（如 '病害'/'虫害'/'作物'）。给出时只在该
                    类型内匹配；若同名节点全部是其它类型则返回 None，避免
                    把“同名不同类型”的节点错误链接。
          context : 上下文线索（字符串或字符串序列），例如记录键
                    '病害:row12'、作物名或领域名。用于在同名同类型节点中
                    选择属性（上下文/领域/来源等）最匹配的节点。
          aliases : 是否同时查询别名表。

        返回匹配到的 node id；未找到返回 None。
        """
        if name is None:
            return None
        name = str(name).strip()
        if not name:
            return None

        candidates = []
        for nid in self.name_to_ids.get(name, []):
            if nid not in candidates:
                candidates.append(nid)
        if aliases:
            for nid in self.alias_to_ids.get(name, []):
                if nid not in candidates:
                    candidates.append(nid)
        if not candidates:
            # 规范化名称/别名兜底（去除空白/标点、统一大小写）
            norm_name = self._normalize(name)
            for nid in self.norm_to_ids.get(norm_name, []):
                if nid not in candidates:
                    candidates.append(nid)
            if aliases:
                for nid in self.norm_alias_to_ids.get(norm_name, []):
                    if nid not in candidates:
                        candidates.append(nid)
        if not candidates:
            return None

        if entity:
            typed = [nid for nid in candidates
                     if self.node_entity(nid) == entity]
            if not typed:
                return None
            candidates = typed

        tokens = self._context_tokens(context)
        if tokens:
            scored = []
            for nid in candidates:
                haystack = self._node_context_text(nid)
                score = sum(1 for token in tokens if token and token in haystack)
                scored.append((score, nid))
            best = max(score for score, _nid in scored)
            if best > 0:
                candidates = [nid for score, nid in scored if score == best]

        return sorted(candidates, key=self._id_sort_key)[0]

    def find_pest(self, name, context=None):
        """在病虫害相关实体类型中定位节点（病害/虫害/害虫）。"""
        for entity in ('病害', '虫害', '害虫'):
            nid = self.find_node(name, entity=entity, context=context)
            if nid is not None:
                return nid
        return None

    def iter_name_nodes(self):
        """遍历全部 (规范名称, node id) 对（含同名不同类型节点）。"""
        for name, ids in self.name_to_ids.items():
            for nid in ids:
                yield name, nid

    def neighbors(self, nid, reltype=None):
        """出边邻接 (目标id, 关系类型)，返回该实体对上的全部关系。"""
        out = []
        for tgt, rel, _data in self.iter_out_edges(nid):
            if reltype is None or rel == reltype:
                out.append((tgt, rel))
        return out

    def in_neighbors(self, nid, reltype=None):
        """入边邻接 (源id, 关系类型)，返回该实体对上的全部关系。"""
        out = []
        for src, rel, _data in self.iter_in_edges(nid):
            if reltype is None or rel == reltype:
                out.append((src, rel))
        return out

    def entities_of_type(self, ent_type):
        """返回某实体类型的全部节点 id 列表"""
        return [nid for nid, d in self.graph.nodes(data=True)
                if d.get('entity') == ent_type]

    # ---------- 多重图边遍历（统一 MultiDiGraph 访问方式） ----------
    def iter_out_edges(self, nid):
        """遍历节点出边，返回 (target_id, reltype, data)。"""
        if self.graph.is_multigraph():
            for _, tgt, key, data in self.graph.out_edges(
                    nid, keys=True, data=True):
                yield tgt, data.get('reltype', key), data
        else:
            for _, tgt, data in self.graph.out_edges(nid, data=True):
                yield tgt, data.get('reltype'), data

    def iter_in_edges(self, nid):
        """遍历节点入边，返回 (source_id, reltype, data)。"""
        if self.graph.is_multigraph():
            for src, _, key, data in self.graph.in_edges(
                    nid, keys=True, data=True):
                yield src, data.get('reltype', key), data
        else:
            for src, _, data in self.graph.in_edges(nid, data=True):
                yield src, data.get('reltype'), data

    def iter_edges(self):
        """遍历全图边，返回 (source_id, target_id, reltype, data)。"""
        if self.graph.is_multigraph():
            for u, v, key, data in self.graph.edges(keys=True, data=True):
                yield u, v, data.get('reltype', key), data
        else:
            for u, v, data in self.graph.edges(data=True):
                yield u, v, data.get('reltype'), data

    # ---------- 图检索原语（对应 3.3.4） ----------
    def _neighbors_by_direction(self, nid, direction='both', rel_types=None):
        """按方向/关系类型返回 (neighbor_id, reltype)。"""
        out = []
        if direction in ('out', 'both'):
            for tgt, rel in self.neighbors(nid):
                if rel_types is None or rel in rel_types:
                    out.append((tgt, rel))
        if direction in ('in', 'both'):
            for src, rel in self.in_neighbors(nid):
                if rel_types is None or rel in rel_types:
                    out.append((src, rel))
        return out

    def ego_subgraph(self, anchors, hop=2, exclude_entities=None,
                     terminal_entities=None, rel_types=None, direction='both'):
        """锚点邻域扩展：对每个锚点取 hop 跳有向子图，合并为候选子图。

        anchors: 锚点节点 id 列表。
        hop: 最大扩展跳数。
        exclude_entities: 需排除的实体类型（不加入最终子图，也不扩展）。
        terminal_entities: 只作为终止节点、不再继续扩展的实体类型；
            默认属性/防治/来源类节点（DEFAULT_TERMINAL_ENTITIES）。
        rel_types: 允许遍历的关系类型集合；None 表示全部。
        direction: 'out' 只沿出边，'in' 只沿入边，'both' 双向扩展。

        返回 (节点 id 集合, 边集合 [(src, dst, reltype)])。
        """
        if terminal_entities is None:
            terminal_entities = DEFAULT_TERMINAL_ENTITIES
        terminal_entities = set(terminal_entities)
        exclude_entities = set(exclude_entities or [])

        final_nodes = {
            a for a in anchors
            if a is not None and self.node_entity(a) not in exclude_entities
        }
        frontier = set(final_nodes)
        visited = set()
        for _ in range(max(0, hop)):
            nxt = set()
            for node in frontier:
                if node in visited:
                    continue
                if self.node_entity(node) in terminal_entities:
                    continue  # 属性节点作为终止节点，不继续扩散
                for tgt, _rel in self._neighbors_by_direction(
                        node, direction=direction, rel_types=rel_types):
                    if tgt == node:
                        continue
                    if self.node_entity(tgt) in exclude_entities:
                        continue
                    if tgt not in final_nodes:
                        final_nodes.add(tgt)
                    if self.node_entity(tgt) in terminal_entities:
                        continue
                    nxt.add(tgt)
            visited |= frontier
            frontier = nxt - visited

        edges = set()
        sub = self.graph.subgraph(final_nodes)
        if sub.is_multigraph():
            for u, v, key, data in sub.edges(keys=True, data=True):
                rel = data.get('reltype', key)
                if rel_types is not None and rel not in rel_types:
                    continue
                edges.add((u, v, rel))
        else:
            for u, v, data in sub.edges(data=True):
                rel = data.get('reltype')
                if rel_types is not None and rel not in rel_types:
                    continue
                edges.add((u, v, rel))
        return final_nodes, edges

    @staticmethod
    def _edge_variants(graph, u, v, rel_types=None):
        """返回 u→v（或无向视图中的相邻边）上满足关系约束的全部关系边。"""
        data = graph.get_edge_data(u, v) or {}
        variants = []
        if getattr(graph, 'is_multigraph', lambda: False)():
            for key, attrs in data.items():
                rel = attrs.get('reltype', key)
                if rel_types is None or rel in rel_types:
                    variants.append((u, v, rel))
        else:
            rel = data.get('reltype')
            # 普通图无 reltype 时无法判断，保守保留
            if rel_types is None or rel is None or rel in rel_types:
                variants.append((u, v, rel))
        variants.sort(key=lambda e: str(e[2]))
        return variants

    @classmethod
    def _path_uses_rel_types(cls, graph, path, rel_types):
        """路径上每一步都存在至少一条满足关系类型约束的真实边。"""
        if not rel_types:
            return True
        for u, v in zip(path, path[1:]):
            if not cls._edge_variants(graph, u, v, rel_types):
                return False
        return True

    def paths_between(self, anchors, max_len=3, rel_types=None,
                      max_paths_per_pair=1, directed=True):
        """多锚点之间的有向简单路径搜索（路径长度 <= max_len）。

        返回 [(anchor_a, anchor_b, [node_ids...], [rel_edges...])] 列表。
        - rel_edges 为 [(src, dst, reltype)]，恢复该节点路径上实际存在的
          全部有效关系；同一实体对存在多种关系时逐一保留；
        - rel_types: 只保留全部边关系都在该集合内的路径；
        - max_paths_per_pair: 每对锚点最多保留的**节点路径**数；每条保留
          节点路径会展开其全部关系组合（默认 1 条节点路径）；
        - directed: True 使用有向图；False 使用无向视图。
        """
        import itertools

        anchors = [a for a in dict.fromkeys(anchors or []) if a is not None]
        graph = self.graph if directed else self.graph.to_undirected()
        result = []
        for i in range(len(anchors)):
            for j in range(i + 1, len(anchors)):
                a, b = anchors[i], anchors[j]
                if a == b:
                    continue
                try:
                    generator = nx.all_simple_paths(
                        graph, a, b, cutoff=max_len)
                except (nx.NetworkXNoPath, nx.NetworkXError):
                    continue
                kept = 0
                # 惰性遍历，默认只保留前 max_paths_per_pair 条可行节点路径；
                # 每条节点路径展开其全部关系组合。
                for path in generator:
                    step_variants = [
                        self._edge_variants(graph, u, v, rel_types)
                        for u, v in zip(path, path[1:])
                    ]
                    if any(not variants for variants in step_variants):
                        continue
                    for combo in itertools.product(*step_variants):
                        result.append((a, b, list(path), list(combo)))
                    kept += 1
                    if kept >= max_paths_per_pair:
                        break
        return result
