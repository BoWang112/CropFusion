# -*- coding: utf-8 -*-
"""
graphrag/sgrag.py —— SG-RAG 基线（嵌入子图匹配，论文适配版）
================================================================
论文对应：SG-RAG (ACL 2026, CAU X-lab) —— Structure Guided
Retrieval-Augmented Generation for Factual Queries.

SG-RAG 核心思想（Exact Retrieval Problem, ERP）：
- 事实性问题含多个约束，答案必须同时满足所有约束
- 把查询建模为"嵌入子图匹配"：查询 → 约束图 → 分解路径 →
  结构感知匹配 → 精确候选子图

本基线（预构建图谱适配版）：
- 约束提取：从查询提取 症状/部位/作物/时期 等约束条件
- 子图匹配：对候选病虫害，检查其图谱子图（为害/症状/发病于 边）
  是否覆盖全部约束 → 多约束精确匹配
- 精确检索：选同时满足所有约束的病虫害子图作为证据

这与 graph_diag 题（症状+部位+防治多约束）高度匹配，
正是 SG-RAG 论文提出的 ERP 场景。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 约束类型 → 匹配的关系边
CONSTRAINT_RELS = {
    '作物': None,          # 病虫害所属作物（通过影响/发病于作物）
    '部位': {'为害', '发病于'},
    '症状': {'症状有'},
    '时期': {'发病于作物'},
    '条件': {'处于', '胁迫有', '致使', '影响'},
}


class SGRAGEngine:
    """SG-RAG 嵌入子图匹配基线"""

    def __init__(self, kg, retriever, top_k=5, hop=2):
        self.kg = kg
        self.retriever = retriever
        self.top_k = top_k
        self.hop = hop

    # ---------- 约束提取 ----------
    def _extract_constraints(self, query):
        """从查询提取约束条件（作物/症状/部位/时期/条件）"""
        constraints = {}
        # 作物约束
        for name, nid in self.kg.norm_names:
            if self.kg.node_entity(nid) == '作物' and name and name in query:
                constraints['作物'] = self.kg.norm_to_name.get(name, name)
                break
        # 部位约束（SITE_ALIASES 风格）
        site_aliases = {
            '叶片': ['叶片', '叶面', '叶缘', '叶尖', '叶脉', '叶鞘', '叶子'],
            '茎秆': ['茎秆', '茎部', '茎基', '茎节', '秆'],
            '穗部': ['穗部', '穗颈', '穗轴', '稻穗', '麦穗', '穗'],
            '根部': ['根部', '根系', '根颈', '根'],
            '果实': ['果实', '果面', '果梗', '果柄', '果'],
            '花器': ['花器', '花瓣', '花蕾', '花'],
            '种子': ['种子', '籽粒', '籽'],
            '幼苗': ['幼苗', '苗期', '苗'],
            '全株': ['全株', '植株'],
        }
        for canonical, aliases in site_aliases.items():
            if any(a in query for a in aliases):
                constraints['部位'] = canonical
                break
        # 时期约束
        period_words = ['苗期', '成株期', '花期', '抽穗', '幼苗期', '穗期',
                        '分蘖', '灌浆', '青果期']
        for w in period_words:
            if w in query:
                constraints['时期'] = w
                break
        # 条件约束
        cond_words = ['高温', '高湿', '低温', '阴雨', '干旱', '积水',
                      '连作', '密植', '通风', '排水']
        cond_hit = [w for w in cond_words if w in query]
        if cond_hit:
            constraints['条件'] = cond_hit[0]
        return constraints

    # ---------- 子图匹配 ----------
    def _match_constraints(self, pest_nid, constraints):
        """检查病虫害节点的图谱子图是否满足全部约束（子图匹配）"""
        nid_ent = self.kg.node_entity(pest_nid)
        if nid_ent not in ('病害', '虫害', '害虫'):
            return False
        # 收集该病虫害的图谱邻域（为害/症状/发病于 等边）
        neighbors = {}
        for tgt, rel in self.kg.neighbors(pest_nid):
            tname = self.kg.node_name(tgt)
            if tname and tname.lower() != 'nan':
                neighbors.setdefault(rel, set()).add(tname)
        # 作物约束：病虫害名含作物名 或 作物-影响->病虫害
        if '作物' in constraints:
            crop = constraints['作物']
            if crop not in self.kg.node_name(pest_nid):
                # 检查作物是否关联
                crop_nid = self.kg.find_node(crop)
                if crop_nid is None:
                    return False
                linked = any(tgt == pest_nid for tgt, _ in self.kg.neighbors(crop_nid))
                if not linked:
                    return False
        # 部位约束
        if '部位' in constraints:
            site = constraints['部位']
            matched = False
            for rel in CONSTRAINT_RELS['部位']:
                if any(site in t for t in neighbors.get(rel, set())):
                    matched = True
                    break
            # 兜底：部位名含在病虫害文本/图谱中
            if not matched:
                for rel in CONSTRAINT_RELS['部位']:
                    if any(site in self.kg.node_name(pest_nid) for _ in [0]):
                        matched = True
                        break
            if not matched:
                return False
        # 症状约束
        if '症状' in constraints:
            sym = constraints['症状']
            if not any(sym in t for t in neighbors.get('症状有', set())):
                return False
        # 时期约束
        if '时期' in constraints:
            period = constraints['时期']
            if not any(period in t for t in neighbors.get('发病于作物', set())):
                # 宽松：时期名在病虫害名里
                if period not in self.kg.node_name(pest_nid):
                    return False
        return True

    # ---------- 候选生成 ----------
    def _candidates(self, query):
        """候选病虫害：从查询中提取的病虫害名或全部病虫害"""
        target = self.retriever._find_target_pest(query)
        if target:
            return list(target)
        # 否则从查询作物找关联病虫害
        for name, nid in self.kg.norm_names:
            if self.kg.node_entity(nid) == '作物' and name in query:
                pests = []
                for tgt, _ in self.kg.neighbors(nid):
                    if self.kg.node_entity(tgt) in ('病害', '虫害', '害虫'):
                        pests.append(tgt)
                if pests:
                    return pests
        return []

    def search(self, query):
        """SG-RAG 检索：约束提取 → 子图匹配 → 精确候选子图"""
        constraints = self._extract_constraints(query)
        candidates = self._candidates(query)
        if not candidates:
            # 无候选：语义图兜底
            anchors, _ = self.retriever.semantic_graph_search(query, top_k=self.top_k)
            candidates = anchors

        # 子图匹配：筛选满足全部约束的病虫害
        matched = []
        for pest_nid in candidates:
            if self._match_constraints(pest_nid, constraints):
                matched.append(pest_nid)

        # 无匹配时退回候选（放宽约束）
        if not matched:
            matched = candidates[:self.top_k]
        matched = matched[:self.top_k]

        # 组装匹配子图证据
        edges = set()
        nodes = set()
        anchor_names = []
        for m in matched:
            n, e = self.retriever.directed_expand([m], hop=self.hop)
            if e:
                edges |= e
                nodes |= n
                anchor_names.append(self.kg.node_name(m))

        graph_evidence = []
        for src, dst, rel in edges:
            graph_evidence.append({
                'head': self.kg.node_name(src),
                'head_type': self.kg.node_entity(src),
                'rel': rel,
                'tail': self.kg.node_name(dst),
                'tail_type': self.kg.node_entity(dst),
                'anchor': self.kg.node_name(src) if src in matched
                          else (anchor_names[0] if anchor_names else 'SG'),
            })
        detail = {
            'variant': 'sg-rag',
            'graph_evidence_count': len(graph_evidence),
            'constraints': constraints,
            'matched_pests': anchor_names,
            'n_candidates': len(candidates),
        }
        return [], graph_evidence, detail
