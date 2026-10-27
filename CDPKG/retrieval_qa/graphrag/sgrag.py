# -*- coding: utf-8 -*-
"""
graphrag/sgrag.py —— SG-RAG 基线（多约束结构感知子图匹配）
================================================================
论文对应：SG-RAG (ACL 2026, CAU X-lab) —— Structure Guided
Retrieval-Augmented Generation for Factual Queries.

本基线（预构建图谱适配版）实现：
1) 多条件约束抽取：作物、部位、症状、时期、环境/栽培条件；
2) 结构感知子图匹配：检查候选病虫害子图中的
   `为害/发病于/症状有/处于/胁迫有/...` 边是否覆盖各约束；
3) 候选排序：约束覆盖度 + 问题-节点语义相似度 + 结构覆盖度；
4) 选中的病虫害子图按问题相关性裁剪边，输出三元组证据。
"""
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 约束类型 → 参与匹配的关系边
CONSTRAINT_RELS = {
    '作物': {'发病于', '影响', '为害'},
    '部位': {'为害', '发病于'},
    '症状': {'症状有'},
    '时期': {'发病于', '易感时间', '处于'},
    '条件': {'处于', '胁迫有', '致使', '影响', '发病于'},
}

SITE_ALIASES = {
    '叶片': ['叶片', '叶面', '叶缘', '叶尖', '叶脉', '叶鞘', '叶子', '叶'],
    '茎秆': ['茎秆', '茎部', '茎基', '茎节', '茎', '秆'],
    '穗部': ['穗部', '穗颈', '穗轴', '稻穗', '麦穗', '穗'],
    '根部': ['根部', '根系', '根颈', '根'],
    '果实': ['果实', '果面', '果梗', '果柄', '果'],
    '花器': ['花器', '花瓣', '花蕾', '花'],
    '种子': ['种子', '籽粒', '籽'],
    '幼苗': ['幼苗', '苗期', '苗'],
    '全株': ['全株', '植株'],
}

PERIOD_WORDS = ['苗期', '成株期', '花期', '抽穗', '幼苗期', '穗期',
                '分蘖', '灌浆', '青果期', '生长前期', '生长后期']
COND_WORDS = ['高温', '高湿', '低温', '阴雨', '干旱', '积水',
              '连作', '密植', '通风', '排水', '湿度', '温度']


class SGRAGEngine:
    """SG-RAG 多约束子图匹配基线"""

    def __init__(self, kg, retriever, top_k=5, hop=2, per_graph_edges=25,
                 match_threshold=0.5):
        self.kg = kg
        self.retriever = retriever
        self.top_k = top_k
        self.hop = hop
        self.per_graph_edges = per_graph_edges
        self.match_threshold = match_threshold
        self._node_vec_cache = {}

    # ---------- 约束抽取 ----------
    def _extract_constraints(self, query):
        """从查询抽取多条件约束（优先使用 LLM QueryParser，字典规则兜底）。"""
        parsed = {}
        if getattr(self.retriever, 'parser', None) is not None:
            try:
                parsed = self.retriever._cached_parse(query) or {}
            except Exception:
                parsed = {}
        constraints = {}

        crop = str(parsed.get('crop') or '').strip()
        if not crop:
            for name, nid in self.kg.norm_names:
                if self.kg.node_entity(nid) == '作物' and name and name in query:
                    crop = self.kg.norm_to_name.get(name, name)
                    break
        if crop:
            constraints['作物'] = crop

        site = str(parsed.get('site') or '').strip()
        if not site:
            for canonical, aliases in SITE_ALIASES.items():
                if any(a in query for a in aliases):
                    site = canonical
                    break
        if site:
            constraints['部位'] = site

        symptom_terms = []
        raw_terms = parsed.get('symptom_terms') or []
        if isinstance(raw_terms, str):
            raw_terms = [raw_terms]
        for term in raw_terms:
            term = str(term).strip()
            if term and term not in symptom_terms:
                symptom_terms.append(term)
        symptom = str(parsed.get('symptom') or '').strip()
        if symptom:
            for term in [symptom] + symptom_terms:
                if term and term not in symptom_terms:
                    symptom_terms.append(term)
        if symptom_terms:
            constraints['症状'] = symptom_terms

        period = ''
        for w in PERIOD_WORDS:
            if w in query:
                period = w
                break
        if period:
            constraints['时期'] = period

        condition = str(parsed.get('environment') or '').strip()
        if not condition:
            for w in COND_WORDS:
                if w in query:
                    condition = w
                    break
        if condition:
            constraints['条件'] = condition
        return constraints

    # ---------- 结构感知子图匹配 ----------
    def _rel_names(self, nid):
        out_rels = defaultdict(set)
        in_rels = defaultdict(set)
        for tgt, rel in self.kg.neighbors(nid):
            name = self.kg.node_name(tgt)
            if name and name.lower() != 'nan':
                out_rels[rel].add(name)
        for src, rel in self.kg.in_neighbors(nid):
            name = self.kg.node_name(src)
            if name and name.lower() != 'nan':
                in_rels[rel].add(name)
        return out_rels, in_rels

    @staticmethod
    def _any_contains(names, needle):
        needle = str(needle or '').strip()
        if not needle:
            return False
        return any(needle in name or name in needle for name in names)

    def _match_score(self, pest_nid, constraints):
        """返回 (约束匹配明细, 综合匹配分数)。"""
        pest_name = self.kg.node_name(pest_nid) or ''
        out_rels, in_rels = self._rel_names(pest_nid)
        matched = {}
        weights = {'作物': 0.25, '部位': 0.25, '症状': 0.30,
                   '时期': 0.10, '条件': 0.10}
        total = 0.0
        weight_sum = 0.0

        if '作物' in constraints:
            crop = constraints['作物']
            names = set(out_rels.get('发病于', set())) | set(in_rels.get('发病于', set()))
            names |= set(out_rels.get('影响', set())) | set(in_rels.get('影响', set()))
            hit = crop in pest_name or self._any_contains(names, crop)
            if not hit:
                crop_nid = self.kg.find_node(crop, entity='作物')
                if crop_nid:
                    hit = any(tgt == pest_nid
                              for tgt, _ in self.kg.neighbors(crop_nid))
            matched['作物'] = 1.0 if hit else 0.0
            total += weights['作物'] * matched['作物']
            weight_sum += weights['作物']

        if '部位' in constraints:
            site = constraints['部位']
            aliases = SITE_ALIASES.get(site, [site])
            names = set()
            for rel in CONSTRAINT_RELS['部位']:
                names |= out_rels.get(rel, set())
                names |= in_rels.get(rel, set())
            hit = any(self._any_contains(names, a) for a in aliases)
            matched['部位'] = 1.0 if hit else 0.0
            total += weights['部位'] * matched['部位']
            weight_sum += weights['部位']

        if '症状' in constraints:
            terms = constraints['症状']
            names = set(out_rels.get('症状有', set())) | set(in_rels.get('症状有', set()))
            hits = sum(1 for term in terms if self._any_contains(names, term))
            matched['症状'] = hits / max(1, len(terms))
            total += weights['症状'] * matched['症状']
            weight_sum += weights['症状']

        if '时期' in constraints:
            period = constraints['时期']
            names = set()
            for rel in CONSTRAINT_RELS['时期']:
                names |= out_rels.get(rel, set())
                names |= in_rels.get(rel, set())
            hit = self._any_contains(names, period) or period in pest_name
            matched['时期'] = 1.0 if hit else 0.0
            total += weights['时期'] * matched['时期']
            weight_sum += weights['时期']

        if '条件' in constraints:
            cond = constraints['条件']
            names = set()
            for rel in CONSTRAINT_RELS['条件']:
                names |= out_rels.get(rel, set())
                names |= in_rels.get(rel, set())
            hit = self._any_contains(names, cond)
            matched['条件'] = 1.0 if hit else 0.0
            total += weights['条件'] * matched['条件']
            weight_sum += weights['条件']

        score = total / weight_sum if weight_sum else 0.0
        return matched, score

    # ---------- 候选与排序 ----------
    def _node_vectors(self, nids):
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

    def _candidates(self, query):
        """候选病虫害：显式病虫害名 / 作物邻居 / 关键词锚点 / 语义锚点。"""
        found = list(self.retriever._find_target_pest(query))
        for name, nid in self.kg.norm_names:
            if self.kg.node_entity(nid) == '作物' and name and name in query:
                for tgt, _ in self.kg.neighbors(nid):
                    if self.kg.node_entity(tgt) in ('病害', '虫害', '害虫'):
                        found.append(tgt)
        found.extend(self.retriever._keyword_target_pests(query, top_k=3))
        anchors, _ = self.retriever.semantic_graph_search(
            query, top_k=max(self.top_k * 3, 10))
        found.extend(anchors)
        found = [n for n in dict.fromkeys(found)
                 if self.kg.node_entity(n) in ('病害', '虫害', '害虫')]
        found = sorted(found, key=lambda n: self.kg.node_name(n) or '')
        return found[:max(self.top_k * 6, 30)]

    def _rank_candidates(self, query, constraints, candidates):
        """候选排序：约束覆盖 + 语义相似度 + 结构覆盖度。"""
        if not candidates:
            return []
        qv = self.retriever.embedder.encode_query(query)
        vecs = self._node_vectors(candidates)
        sims = vecs @ qv if vecs is not None else [0.0] * len(candidates)
        ranked = []
        for nid, sim in zip(candidates, sims):
            matched, match_score = self._match_score(nid, constraints)
            out_rels, _in_rels = self._rel_names(nid)
            covered_rel_types = sum(
                1 for rels in CONSTRAINT_RELS.values()
                if any(rel in out_rels for rel in rels))
            structural = covered_rel_types / max(1, len(CONSTRAINT_RELS))
            score = (0.60 * match_score + 0.25 * float(sim)
                     + 0.15 * structural)
            ranked.append({
                'nid': nid, 'score': score, 'match_score': match_score,
                'semantic': float(sim), 'structural': structural,
                'matched': matched,
            })
        ranked.sort(key=lambda x: -x['score'])
        return ranked

    def search(self, query):
        """SG-RAG 检索：约束抽取 → 子图匹配 → 候选排序 → 证据。"""
        constraints = self._extract_constraints(query)
        candidates = self._candidates(query)

        if not candidates:
            anchors, _ = self.retriever.semantic_graph_search(
                query, top_k=self.top_k)
            candidates = anchors

        ranked = self._rank_candidates(query, constraints, candidates)
        strict = [r for r in ranked if r['match_score'] >= self.match_threshold]
        selected = (strict or ranked)[:self.top_k]

        qv = self.retriever.embedder.encode_query(query)
        graph_evidence = []
        anchor_names = []
        matched_names = []
        graph_rank = 0
        for subgraph_rank, item in enumerate(selected, start=1):
            nid = item['nid']
            anchor_names.append(self.kg.node_name(nid))
            matched_names.append({
                'pest': self.kg.node_name(nid),
                'matched': item['matched'],
                'score': round(item['score'], 4),
            })
            nodes, edges = self.retriever._directed_expand([nid], hop=self.hop)
            edge_list = list(edges)
            if not edge_list:
                continue
            edge_vecs = self.retriever._encode_triples(edge_list)
            sims = edge_vecs @ qv
            ordered = sorted(zip(edge_list, sims.tolist()),
                             key=lambda x: (-x[1], x[0]))
            for triple_rank, ((src, dst, rel), sim) in enumerate(
                    ordered[:self.per_graph_edges], start=1):
                graph_rank += 1
                graph_evidence.append({
                    'head': self.kg.node_name(src),
                    'head_type': self.kg.node_entity(src),
                    'rel': rel,
                    'tail': self.kg.node_name(dst),
                    'tail_type': self.kg.node_entity(dst),
                    'anchor': self.kg.node_name(nid),
                    'anchor_id': nid,
                    'score': round(float(sim), 6),
                    'triple_score': round(float(sim), 6),
                    'graph_rank': graph_rank,
                    'subgraph_rank': subgraph_rank,
                    'triple_rank': triple_rank,
                    'source': 'sg_rag',
                    'subgraph_id': f'sg::{nid}',
                })

        detail = {
            'variant': 'sg-rag',
            'graph_evidence_count': len(graph_evidence),
            'constraints': constraints,
            'matched_pests': matched_names,
            'n_candidates': len(candidates),
            'n_strict_matches': len(strict),
        }
        return [], graph_evidence, detail
