# -*- coding: utf-8 -*-
"""
retriever.py —— 四路并行检索 + 两级融合
=========================================
实现论文 3.3.2~3.3.5 的检索与融合策略：

- 向量检索（BGE-large-zh + FAISS 余弦相似度）
- 关键词检索（jieba 分词 + BM25）
- 语义驱动图检索（BGE 在病虫害实体节点上匹配锚点 → 邻域扩展）
- 实体驱动图检索（显式实体抽取 → 实体链接 → 邻域 + 锚点间路径）
- 文本层 RRF 融合  +  图层结构/语义一致性重排序

对应论文：
  3.3.2 Vector-based semantic retrieval
  3.3.3 Keyword-based retrieval with lexical matching
  3.3.4 (1) Semantic-driven graph retrieval / (2) Entity-driven graph retrieval
  3.3.5 Two-stage fusion strategy
"""
import os
import sys
import re
import json
import pickle
import hashlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (  # noqa: E402
    BGE_MODEL, RETRIEVAL, INDEX_DIR,
)


# ---------- 嵌入模型（BGE-large-zh，论文方案） ----------
class Embedder:
    """文本嵌入模型封装，惰性加载，支持查询/文档归一化"""

    def __init__(self, model_path=None, device=None):
        self.model_path = model_path or BGE_MODEL
        self.device = device
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            import torch
            dev = self.device or ('cuda' if torch.cuda.is_available() else 'cpu')
            self._model = SentenceTransformer(self.model_path, device=dev)
        return self._model

    def encode(self, texts, normalize=True, batch_size=64):
        """编码文本，返回 numpy 数组；normalize 后直接用于余弦相似度"""
        model = self._load()
        if isinstance(texts, str):
            texts = [texts]
        vecs = model.encode(texts, batch_size=batch_size, show_progress_bar=False,
                            normalize_embeddings=normalize)
        return vecs

    def encode_query(self, query, batch_size=16):
        return self.encode([query], batch_size=batch_size)[0]


# ---------- 中文分词与规范化 ----------
_TOKEN_PAT = re.compile(r'[一-龥]+|[a-zA-Z0-9]+')


def normalize_text(s):
    """去除空格、标点，统一为小写"""
    if not s:
        return ''
    s = re.sub(r'\s+', '', str(s))
    s = s.lower()
    return s


def tokenize(text):
    """jieba 分词 + 停用词过滤，返回词项列表"""
    import jieba
    jieba.initialize() if not jieba.dt else None
    toks = jieba.lcut(text)
    stops = set('的了是在与和及或又也而于并且一种等为就都还已经将把被对从向到，。；：？！、""''()《》【】 \t\n')
    out = []
    for t in toks:
        t = t.strip()
        if not t or t in stops:
            continue
        out.append(t)
    return out


# ---------- 倒排索引 / BM25 ----------
class BM25Index:
    """基于 rank_bm25 的关键词检索索引"""

    def __init__(self, corpus, doc_ids=None):
        """
        corpus: 分词后的文档列表（list of list[str]）
        doc_ids: 与 corpus 等长的文档 id 列表
        """
        from rank_bm25 import BM25Okapi
        self.corpus = corpus
        self.doc_ids = doc_ids
        self.bm25 = BM25Okapi(corpus)

    def search(self, query_tokens, top_k=20):
        scores = self.bm25.get_scores(query_tokens)
        order = sorted(range(len(scores)), key=lambda i: -scores[i])
        out = []
        for i in order:
            if scores[i] <= 0:
                continue
            did = self.doc_ids[i] if self.doc_ids else i
            out.append((did, float(scores[i])))
            if len(out) >= top_k:
                break
        return out

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump({'corpus': self.corpus, 'doc_ids': self.doc_ids}, f)

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as f:
            d = pickle.load(f)
        return cls(d['corpus'], d['doc_ids'])


# ---------- FAISS 向量索引 ----------
class VectorIndex:
    """FAISS 向量检索索引（余弦相似度，Inner Product + L2 归一化向量）"""

    def __init__(self, dim, vectors, doc_ids):
        import faiss
        self.dim = dim
        self.doc_ids = list(doc_ids)
        idx = faiss.IndexFlatIP(dim)
        idx.add(vectors.astype('float32'))
        self.index = idx

    def search(self, query_vec, top_k=20):
        scores, ids = self.index.search(query_vec.reshape(1, -1).astype('float32'), top_k)
        out = []
        for score, i in zip(scores[0], ids[0]):
            if i < 0:
                continue
            out.append((self.doc_ids[int(i)], float(score)))
        return out

    def save(self, path):
        import faiss
        faiss.write_index(self.index, path)

    @classmethod
    def load(cls, path, doc_ids):
        import faiss
        idx = faiss.read_index(path)
        inst = cls.__new__(cls)
        inst.index = idx
        inst.dim = idx.d
        inst.doc_ids = list(doc_ids)
        return inst


# ---------- 文本块（Textual View） ----------
class TextBlockStore:
    """实体级文本块集合（论文 3.3.1 文本视图）"""

    def __init__(self, blocks=None):
        self.blocks = blocks or []   # list[dict], 每个含 id,text,entity_id,entity_name,entity_type,source,meta

    def save(self, path):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.blocks, f, ensure_ascii=False, indent=1)

    @classmethod
    def load(cls, path):
        with open(path, encoding='utf-8') as f:
            return cls(json.load(f))


# ---------- 检索执行器 ----------
class Retriever:
    """组合四路检索 + 两级融合"""

    def __init__(self, kg, block_store, vector_index, bm25_index,
                 embedder=None, llm_client=None):
        self.kg = kg
        self.blocks = block_store
        self.vec = vector_index
        self.bm25 = bm25_index
        self.embedder = embedder or Embedder()
        # LLM 问题解析器（真实 API / 模拟降级）
        self.parser = None
        if llm_client is not None:
            from .query_parser import QueryParser
            self.parser = QueryParser(llm_client=llm_client, kg=kg)
        # 解析缓存：同一查询只解析一次（评测时大量重复查询）
        self._parse_cache = {}

        # 三元组边向量缓存（同一 (src,dst,rel) 只编码一次，提升评测性能）
        self._tri_cache = {}
        cache_path = os.path.join(INDEX_DIR, 'tri_cache.pkl')
        if os.path.exists(cache_path):
            try:
                import pickle
                with open(cache_path, 'rb') as f:
                    self._tri_cache = pickle.load(f)
            except Exception:
                self._tri_cache = {}
        # 文本块快速索引
        self.block_by_id = {b['id']: b for b in block_store.blocks}
        # 病虫害实体节点（语义驱动图检索目标集合）
        self.pest_nodes = [kg.entities_of_type('病害'), kg.entities_of_type('虫害')]
        self.pest_nodes = self.pest_nodes[0] + self.pest_nodes[1]
        self.pest_block_id = {}
        for b in block_store.blocks:
            if b.get('entity_id') and b['entity_id'] in set(self.pest_nodes):
                # 同一病虫害可能有多个语义片段；取首片段作为语义锚点代表
                self.pest_block_id.setdefault(b['entity_id'], b['id'])
        # 预计算的病虫害文本块向量（索引构建时生成，避免每次查询重编码）
        self.pest_vecs = None
        pest_vec_path = os.path.join(INDEX_DIR, 'pest_vecs.npy')
        pest_id_path = os.path.join(INDEX_DIR, 'pest_vec_ids.json')
        if os.path.exists(pest_vec_path):
            try:
                import numpy as np
                self.pest_vecs = np.load(pest_vec_path)
                with open(pest_id_path, encoding='utf-8') as f:
                    self.pest_vec_ids = json.load(f)
            except Exception:
                self.pest_vecs = None
        # 剔除对象实体（'为害' 关系要求对象为发病部位类节点）
        self.site_entities = {'发病部位', '具体症状', '发病时期', '世代情况', '越冬情况'}
        # 高信息量锚点类型（图检索锚点限定在这些类型，避免'叶片'、'病斑'等泛词
        # 作为锚点导致邻域爆炸或证据漂移）
        self.anchor_types = {'作物', '病害', '虫害', '病原体', '害虫'}
        # 候选子图仅保留这些实体类型的节点（过滤防治方法链等长尾）
        self.evidence_entities = {
            '作物', '病害', '虫害', '病原体', '害虫', '具体症状', '发病部位',
            '发病时期', '气候因素', '环境因素', '栽培因素', '其他因素',
            '世代情况', '越冬情况', '传播方式', '初侵染源', '易感时间',
            '易感地区', '农业防治', '化学防治', '物理防治', '生物防治',
        }
        # 图谱检索扩展关系白名单：诊断/症状/为害/条件等核心关系。
        # '防治方法有' 仅在病虫害节点一跳时保留（避免防治方法链蔓延）。
        self.expand_rels = {
            '为害', '症状有', '发病于', '处于', '影响', '直接引发',
            '传播', '越冬情况', '世代情况', '易感时间', '易感区域',
            '致使', '胁迫有', '防治方法有',
        }

    # ---------- 1. 向量检索 ----------
    def vector_search(self, query, top_k=None):
        qv = self.embedder.encode_query(query)
        return self.vec.search(qv, top_k or RETRIEVAL['vector_top_k'])

    # ---------- 2. 关键词检索（BM25） ----------
    def keyword_search(self, query, top_k=None):
        toks = tokenize(query)
        if not toks:
            return []
        return self.bm25.search(toks, top_k or RETRIEVAL['keyword_top_k'])

    # ---------- 3. 语义驱动图检索 ----------
    def semantic_graph_search(self, query, top_k=None, hop=None):
        """语义驱动图检索：在病虫害实体节点上做语义匹配，每个候选锚点独立成子图。
        返回 (anchors, [候选子图边集合列表])，供图层重排序逐个子图打分。"""
        top_k = top_k or RETRIEVAL['semantic_top_k']
        hop = hop or RETRIEVAL['graph_hop']
        qv = self.embedder.encode_query(query)
        # 使用预计算的病虫害文本块向量（避免每次查询重编码）
        if self.pest_vecs is not None:
            vecs = self.pest_vecs
            ids = self.pest_vec_ids
        else:
            pest_blocks = [self.block_by_id[self.pest_block_id[n]]
                           for n in self.pest_nodes if n in self.pest_block_id]
            if not pest_blocks:
                return [], []
            ids = [b['id'] for b in pest_blocks]
            vecs = self.embedder.encode([self.block_by_id[i]['text'] for i in ids], batch_size=64)
        scores = vecs @ qv
        order = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_k]
        anchors = []
        for i in order:
            bid = ids[i]
            nid = self.block_by_id[bid]['entity_id']
            anchors.append(nid)
        # 每个锚点独立子图（有向出边扩展）
        candidates = []
        for a in anchors:
            nodes, edges = self._directed_expand([a], hop=hop)
            candidates.append(edges)
        return anchors, candidates

    # ---------- 4. 实体驱动图检索 ----------
    def _find_target_pest(self, query):
        """从查询字符串中直接匹配图谱病虫害名（作物+病害/虫害全名）。
        查询如"黄瓜发生霜霉病后怎么防治"→ 图谱节点"黄瓜霜霉病"是其子串。
        返回图谱节点 id 集合（病害/虫害/害虫类型）。"""
        qn = normalize_text(query)
        hits = set()
        for name, nid in self.kg.norm_names:
            ent = self.kg.node_entity(nid)
            if ent not in ('病害', '虫害', '害虫'):
                continue
            if name and name in qn:
                hits.add(nid)
        return hits

    def _keyword_target_pests(self, query, top_k=2):
        """基于关键词（BM25）的病虫害锚点推断。

        语义驱动图检索的词汇信号增强：把 BM25 命中的文本块映射到其所属
        病虫害实体，取 BM25 分数最高的前 top_k 个病虫害。

        动机：综合诊断类问题（症状+受害部位->病害）题干不含病虫害名，
        纯 BGE 向量对目标病害锚点 top1 命中仅约 30%，而 BM25 对含
        具体症状（病斑形态/尺寸）的文本块定位极强（实测 top1 ~97%）。
        关键词命中的症状块归属该病虫害实体块，可稳定推断目标病害，
        使 full 的目标防治边/部位边可靠进入提示词。

        与 _find_target_pest（显式病虫害名匹配，实体驱动图检索）互补：
        _find_target_pest 处理题干含病虫害名的问题；
        本方法处理题干只含症状/部位的问题（graph_diag）。
        """
        toks = tokenize(query)
        if not toks:
            return []
        kw = self.bm25.search(toks, top_k=20)
        scores = {}
        for did, s in kw:
            b = self.block_by_id.get(did)
            if not b or not b.get('entity_id'):
                continue
            eid = b['entity_id']
            if self.kg.node_entity(eid) not in ('病害', '虫害', '害虫'):
                continue
            if s > scores.get(eid, 0.0):
                scores[eid] = s
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        return [eid for eid, _s in ranked[:top_k]]

    def _fuzzy_link(self, name, etype=None):
        """模糊实体链接：口语/俗名 → 图谱规范节点。
        策略（按优先级）：
        1) 规范名包含口语名（如 '小麦条锈病' 包含 '小麦'）
        2) 口语名包含规范名（如 '稻瘟病' 是 '水稻稻瘟病' 的子串）
        3) 共享子串（如 '土豆' vs '马铃薯' 通过字符重叠）
        仅匹配实体类型一致（或类型为空）的节点。"""
        from difflib import SequenceMatcher
        if not name or len(name) < 2:
            return None
        best = None
        best_score = 0.0
        for cand_name, nid in self.kg.iter_name_nodes():
            cand_ent = self.kg.node_entity(nid)
            if cand_ent not in self.anchor_types:
                continue
            # 实体类型一致性校验（etype 存在时）
            if etype and etype in self.anchor_types and cand_ent != etype:
                continue
            # 子串匹配
            if cand_name in name or name in cand_name:
                score = min(len(cand_name), len(name)) / max(len(cand_name), len(name))
            else:
                # 字符重叠度（处理 土豆/马铃薯 这类）
                common = sum(1 for c in name if c in cand_name)
                score = common / max(len(name), len(cand_name))
            if score > 0.6 and score > best_score:
                best_score = score
                best = (cand_name, nid, cand_ent)
        return best

    def _extract_entities(self, query):
        """从查询中抽取显式实体（实体驱动图检索入口）。
        优先用 LLM 解析（论文 3.3.4(2) 的实体抽取提示模板），
        无 LLM 时降级为字典精确匹配。
        仅保留高信息量锚点类型（作物/病害/虫害/病原体/害虫），
        过滤'叶片'、'病斑'等泛词，避免邻域爆炸。"""
        if self.parser is not None:
            parsed = self._cached_parse(query)
            ents = []
            for e in parsed.get('entities', []):
                name = e.get('name', '')
                etype = e.get('type', '')
                # 类型已知时必须在同类型节点中定位，避免同名不同类型的
                # 属性实体/病虫害实体被错误链接。
                nid = self.kg.find_node(
                    name, entity=etype if etype in self.anchor_types else None)
                if nid and self.kg.node_entity(nid) in self.anchor_types:
                    ents.append({'name': name, 'id': nid, 'entity': self.kg.node_entity(nid)})
                    continue
                # 模糊链接：口语/俗名 → 图谱规范名（实体链接兜底）
                linked = self._fuzzy_link(name, etype)
                if linked:
                    ents.append({'name': linked[0], 'id': linked[1],
                                 'entity': linked[2]})
                elif etype in self.anchor_types:
                    ents.append({'name': name, 'id': None, 'entity': etype})
            return ents
        # 字典精确匹配降级
        qn = normalize_text(query)
        names = []
        for name, nid in self.kg.norm_names:
            if name in qn:
                names.append((name, nid))
        used = []
        found = []
        for name, nid in names:
            if any(name in u for u in used):
                continue
            used.append(name)
            ent = self.kg.node_entity(nid)
            if ent in self.anchor_types:
                found.append({'name': self.kg.norm_to_name.get(name, name),
                              'id': nid, 'entity': ent})
        return found

    def entity_graph_search(self, query, hop=None, max_len=None):
        hop = hop or RETRIEVAL['graph_hop']
        max_len = max_len or RETRIEVAL['path_max_len']
        ents = self._extract_entities(query)
        # 图扩展锚点：病虫害类实体正常扩展；作物类仅沿'症状有'取症状，
        # 避免沿共享部位节点拉入无关病虫害
        expand_anchors = [e['id'] for e in ents
                          if e['id'] and self.kg.node_entity(e['id']) in ('病害', '虫害')]
        crop_anchors = [e['id'] for e in ents
                        if e['id'] and self.kg.node_entity(e['id']) == '作物']
        anchors = list(expand_anchors)
        candidates = []
        for a in expand_anchors:
            nodes, edges = self._directed_expand([a], hop=hop)
            candidates.append(edges)
        # 作物锚点：仅沿'症状有'取症状边（作物-症状有->症状），补充诊断线索
        for c in crop_anchors:
            sym_edges = set()
            for tgt, rel in self.kg.neighbors(c, '症状有'):
                sym_edges.add((c, tgt, rel))
            if sym_edges:
                candidates.append(sym_edges)
                anchors.append(c)
        if not anchors and not candidates:
            return [], [], [], []
        # 锚点间路径补充：paths_between 返回真实关系边（MultiDiGraph 中
        # 同一实体对的不同关系会分别保留），并附带稳定 path_id。
        paths = self.kg.paths_between(list(anchors), max_len=max_len)
        enriched_paths = []
        for item in paths:
            if len(item) >= 4:
                a, b, node_path, edge_path = item[:4]
            else:
                a, b, node_path = item[:3]
                edge_path = []
            path_edges = set(edge_path)
            if path_edges:
                # 优先加入包含起点 a 的候选子图；没有则新建路径子图，
                # 避免路径证据因候选对齐丢失。
                target_idx = None
                for i, cand in enumerate(candidates):
                    if any(e[0] == a for e in cand):
                        target_idx = i
                        break
                if target_idx is None:
                    candidates.append(set(path_edges))
                    anchors.append(a)
                else:
                    candidates[target_idx] = candidates[target_idx] | path_edges
            path_id = self._path_id(a, b, edge_path)
            enriched_paths.append((a, b, node_path, edge_path, path_id))
        return [e['name'] for e in ents], list(anchors), candidates, enriched_paths

    def _directed_expand(self, anchors, hop=2):
        """从锚点沿出边单向扩展（保留与锚点直接相关的属性/症状/防治知识）。

        关键约束：
        - 关系白名单 self.expand_rels
        - 仅 高价值节点（作物/病虫害/病原体/害虫）作为扩展前沿继续展开；
          症状、部位、条件等作为终点节点（叶节点），防止经共享枢纽
          （如'叶片'）把无关实体拉入子图
        - '防治方法有' 仅在锚点一跳时纳入（防治节点不再继续展开，
          避免 药剂→防治方法有→药剂 噪声链蔓延）

        返回 (节点集合, 边集合)。
        """
        expandable_types = {'作物', '病害', '虫害', '病原体', '害虫'}
        ctrl_ents = {'农业防治', '化学防治', '物理防治', '生物防治', '其他因素'}
        final_nodes = set(anchors)
        frontier = set(anchors)
        for depth in range(hop):
            nxt = set()
            for n in frontier:
                is_anchor = n in anchors
                for tgt, rel, _data in self.kg.iter_out_edges(n):
                    if rel not in self.expand_rels:
                        continue
                    tgt_ent = self.kg.node_entity(tgt)
                    # 防治类节点：仅在锚点一跳时纳入，不再从防治节点继续展开
                    if tgt_ent in ctrl_ents:
                        if is_anchor:
                            final_nodes.add(tgt)
                        continue
                    # 非锚点节点的目标实体类型限制
                    if tgt_ent not in self.evidence_entities:
                        continue
                    if tgt not in final_nodes:
                        final_nodes.add(tgt)
                        # 仅高价值节点进入下一跳前沿
                        if tgt_ent in expandable_types:
                            nxt.add(tgt)
            frontier = nxt
        edges = set()
        for u, v, rel, _data in self.kg.iter_edges():
            if u in final_nodes and v in final_nodes and rel in self.expand_rels:
                edges.add((u, v, rel))
        return final_nodes, edges

    # 公开别名（供 graphrag/ 包复用，无 _ 私有语义）
    directed_expand = _directed_expand

    # ---------- 文本层融合：RRF ----------
    @staticmethod
    def rrf_fusion(lists, k=60):
        """lists: list of (doc_id, rank_score) 列表；返回按 RRF 得分降序的 (doc_id, score)"""
        import math
        scores = {}
        for ranked in lists:
            for pos, (did, _score) in enumerate(ranked):
                scores[did] = scores.get(did, 0.0) + 1.0 / (k + pos + 1)
        return sorted(scores.items(), key=lambda x: -x[1])

    # 症状判别词：病斑形态/颜色/质地等强判别特征（症状类任务排序改进）
    SYMPTOM_BOOST_WORDS = [
        '病斑', '坏死', '霉层', '褪绿', '变褐', '斑点', '枯死', '腐烂',
        '水渍', '凹陷', '溃疡', '穿孔', '畸形', '卷曲', '萎蔫', '枯萎',
        '圆形', '椭圆形', '梭形', '轮纹', '晕圈', '边缘', '中央',
        '褐色', '黑色', '白色', '黄色', '灰色', '红色', '绿色', '紫色',
        '斑', '粉状', '霉', '小点', '条斑',
    ]

    def _extract_symptom_terms(self, query):
        """提取查询中的症状判别词。
        优先用 LLM 解析结果（更精准的判别词），无 LLM 时用硬编码词表。"""
        if self.parser is not None:
            parsed = self._cached_parse(query)
            terms = parsed.get('symptom_terms') or []
            if terms:
                return terms
        return [w for w in self.SYMPTOM_BOOST_WORDS if w in query]

    def _cached_parse(self, query):
        """带缓存的查询解析"""
        if query not in self._parse_cache:
            self._parse_cache[query] = self.parser.parse(query)
        return self._parse_cache[query]

    def _apply_symptom_boost(self, results, symptom_terms):
        """
        对检索结果加权并重排：包含症状判别词的文本块排名提前。
        返回按加权分数降序的新列表（供 RRF 使用，RRF 只看排名）。
        """
        if not symptom_terms:
            return results
        boosted = []
        for did, score in results:
            b = self.block_by_id.get(did)
            w = 0.0
            if b and b.get('text'):
                hits = sum(1 for wd in symptom_terms if wd in b['text'])
                if hits > 0:
                    w = 0.15 * hits
            boosted.append((did, score + w))
        # 按加权分数重排（同分保持原顺序）
        boosted.sort(key=lambda x: -x[1])
        return boosted

    def text_fusion(self, query, top_k=None, apply_symptom_boost=False):
        """向量 + 关键词 → 标准 RRF → 返回 (doc_id, rrf_score) 列表。

        默认只做标准 RRF。症状判别词加权属于额外启发式，仅在
        apply_symptom_boost=True 时启用，避免与正式 RRF 评分口径混用。
        """
        vec_res = self.vector_search(query)
        kw_res = self.keyword_search(query)
        if apply_symptom_boost:
            sym_terms = self._extract_symptom_terms(query)
            vec_res = self._apply_symptom_boost(vec_res, sym_terms)
            kw_res = self._apply_symptom_boost(kw_res, sym_terms)
        fused = self.rrf_fusion([vec_res, kw_res], k=RETRIEVAL['rrf_k'])
        return fused[: (top_k or RETRIEVAL['text_top_k'])]

    # ---------- 图层融合：重排序 ----------
    def _subgraph_compactness(self, edges, anchor_id=None, anchor_ids=None,
                              rel_types=None, max_len=None, max_paths=2000):
        """候选子图的结构紧凑性（有效关系路径集合）。

        以锚点为根，沿**有向有效关系**（rel_types，默认 expand_rels）枚举
        长度不超过 max_len（默认 path_max_len=3）的简单关系路径；
        结构紧凑性 = 有效路径长度倒数的平均值 mean(1/len(path))。

        与“锚点到其它节点的最短距离均值”不同，这里对**每一条有效关系
        路径**计数：同一对节点之间若存在多条不同的有效路径，它们都参与
        平均，从而更严格地惩罚依赖长路径/绕行路径才连通的松散子图。
        - 关系方向严格遵守有向扩展方向；
        - anchor_ids 支持多锚点，分别枚举各锚点出发的路径（锚点之间的
          连通路径也会被计入）；
        - 属性/防治/症状等节点只作为路径终点，不再继续扩展，避免经共享
          枢纽节点产生组合爆炸。
        - max_paths 为路径枚举上限（默认 2000），防止高密度子图组合爆炸；
          邻接表按稳定顺序展开，保证同一子图得分可复现。
        """
        from collections import defaultdict

        if rel_types is None:
            rel_types = getattr(self, 'expand_rels', None)
        if max_len is None:
            max_len = RETRIEVAL.get('path_max_len', 3)
        terminal_entities = {
            '具体症状', '发病部位', '发病时期', '气候因素', '环境因素',
            '栽培因素', '其他因素', '世代情况', '越冬情况', '传播方式',
            '初侵染源', '易感时间', '易感地区', '农业防治', '化学防治',
            '物理防治', '生物防治', '类似虫害', '分类地位', '拉丁文名称',
            '专业书籍', '权威网站', '作物类型',
        }

        nodes = set()
        out_adj = defaultdict(set)
        for src, dst, rel in edges:
            if src is None or dst is None:
                continue
            nodes.add(src)
            nodes.add(dst)
            if rel_types is not None and rel not in rel_types:
                continue
            out_adj[src].add((dst, rel))
        if len(nodes) <= 1:
            return 0.0

        # 锚点集合：兼容单锚点与多锚点两种调用方式
        raw_roots = []
        if anchor_ids is not None:
            if isinstance(anchor_ids, (str, int)):
                raw_roots = [anchor_ids]
            else:
                raw_roots = list(anchor_ids)
        elif anchor_id is not None:
            raw_roots = [anchor_id]
        roots = []
        for root in raw_roots:
            if root in nodes and root not in roots:
                roots.append(root)
        if not roots:
            # 未提供有效锚点：选择出度最高的节点作为根（同分按稳定 id）
            best_degree = max(len(out_adj.get(n, ())) for n in nodes)
            best_nodes = [n for n in nodes
                          if len(out_adj.get(n, ())) == best_degree]
            roots = [min(best_nodes, key=self._node_sort_key)]

        def _edge_sort_key(item):
            nxt, rel = item
            return (self.kg.node_name(nxt) or '', rel or '',
                    self._node_sort_key(nxt))

        path_lengths = []
        truncated = False
        for root in roots:
            # (当前节点, 已走节点序列, 已访问集合)
            stack = [(root, (), frozenset((root,)))]
            while stack and len(path_lengths) < max_paths:
                node, path, visited = stack.pop()
                if len(path) >= max_len:
                    continue
                if self.kg.node_entity(node) in terminal_entities:
                    continue
                for nxt, _rel in sorted(out_adj.get(node, ()),
                                        key=_edge_sort_key):
                    if nxt in visited:
                        continue
                    new_path = path + (nxt,)
                    path_lengths.append(len(new_path))
                    if len(path_lengths) >= max_paths:
                        truncated = True
                        break
                    stack.append((nxt, new_path, visited | {nxt}))
            if truncated:
                break

        if not path_lengths:
            return 0.0
        return sum(1.0 / length for length in path_lengths) / len(path_lengths)

    def _node_sort_key(self, nid):
        """节点 id 的稳定排序键（CSV id 可能是数字字符串）。"""
        text = str(nid)
        try:
            return (0, int(text))
        except (TypeError, ValueError):
            return (1, text)

    def graph_rerank(self, query, candidates, anchors=None, top_k=None,
                     per_graph_edges=25, lambda_=None):
        """
        对候选子图按结构紧凑性 + 语义相关性重排序（论文 3.3.5 (2)）。

        candidates: list[ set[(src_id, dst_id, reltype)] ]，每个元素是一个候选子图。
        anchors: 与 candidates 对齐的锚点节点 id 列表（作为证据标签）。

        评分：
          结构紧凑性 = 子图内以锚点为根的最短关系路径长度倒数的平均值
          语义相关性 = 子图内三元组与查询向量余弦相似度的平均值
          综合得分   = λ × 结构紧凑性 + (1-λ) × 语义相关性
        λ 默认取 RETRIEVAL['graph_rerank_lambda']（默认 0.4），可通过参数覆盖。
        保留原有的边裁剪逻辑（诊断边优先、防治意图预留槽位）。
        """
        if not candidates:
            return []
        anchors = anchors or [None] * len(candidates)
        if lambda_ is None:
            lambda_ = RETRIEVAL.get('graph_rerank_lambda', 0.4)
        lambda_ = min(1.0, max(0.0, float(lambda_)))

        # 诊断类关系（症状/为害/条件）优先保留；防治类边按意图处理
        diag_rels = {'为害', '症状有', '发病于', '处于', '影响', '直接引发',
                     '传播', '越冬情况', '世代情况', '易感时间', '易感区域',
                     '致使', '胁迫有'}
        ctrl_rels = {'防治方法有'}
        qv = self.embedder.encode_query(query)
        ctrl_intent = self._has_control_intent(query)
        scored = []

        for idx, edges in enumerate(candidates):
            if not edges:
                continue
            edge_list = list(edges)
            # 语义相关性：三元组线性化文本与查询向量的平均余弦相似度
            tri_vecs = self._encode_triples(edge_list)
            sem_scores = tri_vecs @ qv
            sem_mean = float(sem_scores.mean()) if len(sem_scores) else 0.0

            raw_anchor = anchors[idx] if idx < len(anchors) else None
            if isinstance(raw_anchor, (list, tuple, set, frozenset)):
                anchor_ids = list(raw_anchor)
            elif raw_anchor is None:
                anchor_ids = []
            else:
                anchor_ids = [raw_anchor]
            anchor_id = anchor_ids[0] if anchor_ids else None
            compactness = self._subgraph_compactness(
                edge_list, anchor_id=anchor_id, anchor_ids=anchor_ids,
                rel_types=self.expand_rels,
                max_len=RETRIEVAL.get('path_max_len', 3))
            score = lambda_ * compactness + (1.0 - lambda_) * sem_mean

            # 边裁剪：按语义相似度排序（同分按边文本稳定排序），
            # 诊断边优先；防治意图时为防治边留槽位。
            def _edge_rank_key(pair):
                edge, sim = pair
                return (-sim, self.kg.node_name(edge[0]) or '',
                        edge[2] or '', self.kg.node_name(edge[1]) or '')

            ranked = sorted(zip(edge_list, sem_scores.tolist()),
                            key=_edge_rank_key)
            diag_edges = [e for e, _s in ranked if e[2] in diag_rels]
            ctrl_edges = [e for e, _s in ranked if e[2] in ctrl_rels]
            ctrl_budget = (per_graph_edges // 2) if ctrl_intent else 0
            kept = set(diag_edges[:per_graph_edges - ctrl_budget])
            if not ctrl_intent:
                others_edges = [e for e, _s in ranked
                                if e[2] not in diag_rels and e[2] not in ctrl_rels]
                if len(kept) < per_graph_edges - ctrl_budget:
                    kept |= set(others_edges[:per_graph_edges - ctrl_budget - len(kept)])
            elif len(kept) < per_graph_edges - ctrl_budget:
                kept |= set(ctrl_edges[:per_graph_edges - ctrl_budget - len(kept)])
            if ctrl_budget and ctrl_edges:
                kept |= set(ctrl_edges[:ctrl_budget])
            if ctrl_intent:
                kept |= set(ctrl_edges)

            # 子图内部三元组严格按查询相关性排序（同分按稳定三元组键）；
            # 保留每条三元组的相关性分数，供统一原子证据排序使用。
            kept_ranked = [(e, s) for e, s in ranked if e in kept]
            ordered_edges = [e for e, _s in kept_ranked]
            triple_scores = {e: float(s) for e, s in kept_ranked}
            kept_nodes = {x for e in kept for x in (e[0], e[1])}
            anchor_names = [self._node_name(self.kg, a) for a in anchor_ids
                            if a is not None]
            anchor_name = '、'.join([n for n in anchor_names if n]) or None
            scored.append({
                'score': score, 'nodes': kept_nodes, 'edges': kept,
                'ordered_edges': ordered_edges, 'triple_scores': triple_scores,
                'subgraph_id': self._subgraph_id(kept),
                'subgraph_score': score,
                'sem': sem_mean, 'struct': compactness, 'lambda': lambda_,
                'anchor': anchor_name, 'anchor_id': anchor_id,
                'anchor_ids': anchor_ids,
            })

        scored.sort(key=lambda x: (-x['score'], x['anchor'] or ''))
        ranked_subgraphs = scored[: (top_k or RETRIEVAL['graph_top_k'])]
        for rank, item in enumerate(ranked_subgraphs, start=1):
            item['subgraph_rank'] = rank
        return ranked_subgraphs

    @staticmethod
    def _has_control_intent(query):
        """查询是否含明确防治意图（综合诊断类问题会问"如何防治/用什么药"）"""
        if not query:
            return False
        ctrl_kw = ("防治", "用药", "施药", "药剂", "防控", "防除", "怎么治", "治疗")
        return any(k in query for k in ctrl_kw)

    @staticmethod
    def _node_name(kg, nid):
        """图节点名称（含空值保护）"""
        if nid is None:
            return None
        return kg.node_name(nid)

    def _triple_id(self, src, dst, rel):
        """稳定三元组 ID（与 3_evaluate.py 的 kg_<sha1> 规则一致）。"""
        head = self.kg.node_name(src) or str(src)
        tail = self.kg.node_name(dst) or str(dst)
        raw = '||'.join(str(x) for x in (head, rel or '', tail))
        return 'kg_' + hashlib.sha1(raw.encode('utf-8')).hexdigest()[:16]

    def _edge_sort_key(self, edge):
        src, dst, rel = edge
        return (self.kg.node_name(src) or '', rel or '',
                self.kg.node_name(dst) or '')

    def _subgraph_id(self, edges):
        """候选子图内容稳定 ID（与查询排名无关）。"""
        ids = sorted(self._triple_id(s, d, r) for s, d, r in edges)
        raw = '||'.join(ids)
        return 'subgraph::' + hashlib.sha1(
            raw.encode('utf-8')).hexdigest()[:16]

    def _path_id(self, anchor_a, anchor_b, edge_path):
        """多锚点关系路径稳定 ID（由真实关系边序列决定）。"""
        raw = '||'.join(f'{s}->{r}->{d}' for s, d, r in edge_path)
        if not raw:
            raw = f'{anchor_a}->{anchor_b}'
        return 'path::' + hashlib.sha1(raw.encode('utf-8')).hexdigest()[:16]

    def _rank_edges_by_query(self, edges, qv):
        """按三元组-查询余弦对边排序；返回 (有序边列表, 分数映射)。"""
        if not edges or qv is None:
            return [], {}
        edge_list = list(edges)
        vecs = self._encode_triples(edge_list)
        sims = vecs @ qv
        ranked = sorted(
            zip(edge_list, sims.tolist()),
            key=lambda pair: (-float(pair[1]), self._edge_sort_key(pair[0])),
        )
        return [e for e, _s in ranked], {e: float(s) for e, s in ranked}

    def _encode_triples(self, edges):
        """批量编码三元组边文本（带缓存）。返回与 edges 等长的向量数组。"""
        import numpy as np
        texts = [f"{self.kg.node_name(s)} {rel} {self.kg.node_name(d)}"
                 for s, d, rel in edges]
        uncached = [t for t in texts if t not in self._tri_cache]
        if uncached:
            new_vecs = self.embedder.encode(uncached, batch_size=256)
            for t, v in zip(uncached, new_vecs):
                self._tri_cache[t] = v
        vecs = np.stack([self._tri_cache[t] for t in texts]) if texts else np.zeros((0, 1024))
        return vecs

    # ---------- 证据组装 ----------
    def get_evidence(self, query, variant='full', use_extra_heuristics=False):
        """
        完整检索 + 融合流程，返回 {text_evidence, graph_evidence, detail}。

        use_extra_heuristics=False（正式评估默认）：
          文本层只用标准 RRF；图层只用标准 graph_rerank 评分。
        use_extra_heuristics=True：
          额外启用症状词加权、关键词目标锚点推断、目标锚点提升/强制保留；
          这些启发式与标准 RRF/图重排序分离，避免影响正式评分口径。

        variant 控制检索路径（用于论文表6消融 + 基线对比）：
          full                : 四路检索 + 两级融合（完整方法）
          -no-text-rrf        : 移除文本层RRF（向量/关键词结果直接拼接排序）
          -no-graph-rerank    : 移除图层重排序（候选子图按初始顺序）
          -no-semantic-graph  : 移除语义驱动图检索
          -no-entity-graph    : 移除实体驱动图检索
          naive-text          : 仅向量检索（NaiveRAG 基线）
          hybrid-text         : 向量 + 关键词 RRF（Hybrid RAG 基线）
          graph-only          : 仅图谱证据（GraphRAG 风格基线）
        """
        # ---- 文本证据 ----
        if variant == 'naive-text':
            # 仅向量检索
            vec_res = self.vector_search(query)
            fused = vec_res[:RETRIEVAL['text_top_k']]
        elif variant == 'hybrid-text':
            # 向量 + 关键词，RRF 融合
            fused = self.rrf_fusion(
                [self.vector_search(query), self.keyword_search(query)],
                k=RETRIEVAL['rrf_k'],
            )[:RETRIEVAL['text_top_k']]
        elif variant == '-no-text-rrf':
            # 无 RRF：向量结果排前面，关键词结果补充
            vec_res = self.vector_search(query)
            kw_res = self.keyword_search(query)
            seen = set()
            fused = []
            for did, s in vec_res + kw_res:
                if did not in seen:
                    seen.add(did)
                    fused.append((did, s))
                if len(fused) >= RETRIEVAL['text_top_k']:
                    break
        elif variant == 'graph-only':
            # GraphRAG 风格基线：仅图谱证据，无文本证据
            fused = []
        else:
            # 标准文本融合；症状加权仅在额外启发式模式启用
            fused = self.text_fusion(
                query, top_k=RETRIEVAL['text_top_k'],
                apply_symptom_boost=use_extra_heuristics)

        text_evidence = []
        for rrf_rank, (did, score) in enumerate(fused, start=1):
            b = self.block_by_id.get(did)
            if b:
                meta = b.get('meta') or {}
                text_evidence.append({
                    'id': did, 'text': b['text'], 'score': round(score, 4),
                    'rrf_rank': rrf_rank, 'rrf_score': float(score),
                    'channel': 'text',
                    'entity': b.get('entity_name'), 'entity_type': b.get('entity_type'),
                    'source': b.get('source'),
                    'segment_id': meta.get('segment_id'),
                    'record_index': meta.get('record_index'),
                    'segment_index': meta.get('segment_index'),
                    'source_id': meta.get('source_id'),
                    'meta': meta,
                })

        # ---- 图谱证据 ----
        graph_evidence = []
        total_nodes = set()
        if variant not in ('naive-text', 'hybrid-text'):
            # graph-only: 仅图谱证据（语义+实体驱动图检索，无文本融合）
            use_semantic = variant not in ('-no-semantic-graph',)
            use_entity = variant not in ('-no-entity-graph',)
            s_anchors, s_candidates = (self.semantic_graph_search(query)
                                       if use_semantic else ([], []))
            _ents, _anchors, e_candidates, e_paths = (
                self.entity_graph_search(query)
                if use_entity else ([], [], [], []))
            all_candidates = s_candidates + e_candidates
            all_anchors = s_anchors + _anchors
            # 关系路径 -> 边 的稳定 path_id 映射（供证据溯源与原子评估）
            path_ids_by_edge = {}
            for item in e_paths:
                if len(item) >= 5:
                    _a, _b, _node_path, edge_path, path_id = item[:5]
                elif len(item) >= 4:
                    _a, _b, _node_path, edge_path = item[:4]
                    path_id = self._path_id(_a, _b, edge_path)
                else:
                    continue
                for edge in edge_path:
                    ids = path_ids_by_edge.setdefault(edge, [])
                    if path_id not in ids:
                        ids.append(path_id)
            # 目标锚点：查询中显式提到的病虫害图谱节点（实体驱动检索的增强）。
            # 只对未移除实体图检索的变体生效：-no-entity-graph 应失去
            # 基于实体匹配的目标锚点，否则该消融会被目标锚点提升绕过。
            target_anchor_ids = self._find_target_pest(query) if use_entity else set()
            if use_entity and _anchors:
                target_anchor_ids |= {a for a in _anchors
                                      if self.kg.node_entity(a) in ('病害', '虫害', '害虫')}
            # 关键词锚点推断（语义驱动图检索的词汇信号增强）：综合诊断类问题
            # （graph_diag，题干只含症状/部位、无病虫害名）纯向量对目标病害
            # top1 命中仅 ~30%，BM25 对具体症状文本块定位极强，可稳定推断
            # 目标病害。仅对 full 生效：-no-text-rrf（无文本 RRF）与
            # -no-semantic-graph（无语义图检索）都应失去这条信号。
            if use_extra_heuristics and variant == 'full':
                target_anchor_ids |= set(self._keyword_target_pests(query))
            existing_anchor_set = set(all_anchors)
            for ta in target_anchor_ids:
                if ta not in existing_anchor_set:
                    nodes, edges = self._directed_expand([ta], hop=RETRIEVAL['graph_hop'])
                    if edges:
                        all_candidates.append(edges)
                        all_anchors.append(ta)
                        existing_anchor_set.add(ta)
            # 目标锚点提升：把目标锚点子图排到最前，避免错误候选锚点
            # （同作物的其他病虫害）把目标病害的防治边/部位边挤出展示范围。
            # 仅对 full 生效（提升本身是一种重排序）；-no-graph-rerank 保持
            # 语义图检索的原始锚点顺序，从而体现"图层重排序"的消融差异。
            if use_extra_heuristics and target_anchor_ids and variant != '-no-graph-rerank':
                ordered = []
                seen_idx = set()
                for idx, a in enumerate(all_anchors):
                    if a in target_anchor_ids and idx not in seen_idx:
                        ordered.append(idx)
                        seen_idx.add(idx)
                for idx in range(len(all_anchors)):
                    if idx not in seen_idx:
                        ordered.append(idx)
                all_candidates = [all_candidates[i] for i in ordered]
                all_anchors = [all_anchors[i] for i in ordered]

            if variant == '-no-graph-rerank':
                # 无图层重排序：按原始子图顺序；子图内三元组仍按查询相关性排序
                reranked = []
                qv = self.embedder.encode_query(query)
                for idx, cand in enumerate(all_candidates[:RETRIEVAL['graph_top_k']]):
                    if not cand:
                        continue
                    ordered_edges, triple_scores = self._rank_edges_by_query(
                        cand, qv)
                    nodes = {x for e in cand for x in (e[0], e[1])}
                    anchor_id = all_anchors[idx] if idx < len(all_anchors) else None
                    reranked.append({
                        'score': 0.0, 'nodes': nodes, 'edges': cand,
                        'ordered_edges': ordered_edges,
                        'triple_scores': triple_scores,
                        'subgraph_id': self._subgraph_id(cand),
                        'subgraph_score': 0.0,
                        'subgraph_rank': len(reranked) + 1,
                        'anchor': self._node_name(self.kg, anchor_id),
                        'anchor_id': anchor_id,
                        'anchor_ids': [anchor_id] if anchor_id is not None else [],
                    })
            else:
                reranked = self.graph_rerank(query, all_candidates, anchors=all_anchors,
                                             top_k=RETRIEVAL['graph_top_k'])
            # 目标锚点子图强制保留：即使 graph_rerank 按语义得分排序把它们挤出 Top-K，
            # 也强制加入展示（目标病害是查询明确指认的，其防治/部位边是检索必需信息）。
            # 仅对 full（完整方法）生效；-no-graph-rerank 不做强制保留，
            # 否则移除重排序后仍能靠强制保留拿到目标锚点边，消融就不敏感了。
            if use_extra_heuristics and target_anchor_ids and variant != '-no-graph-rerank':
                def _is_target_subgraph(cand):
                    ids = cand.get('anchor_ids')
                    if ids:
                        return any(a in target_anchor_ids for a in ids)
                    anchor_id = cand.get('anchor_id')
                    return anchor_id in target_anchor_ids

                target_scored = [c for c in reranked if _is_target_subgraph(c)]
                others = [c for c in reranked if c not in target_scored]
                reranked = target_scored + others
            # 仅展示 Top-K 个子图。
            # 子图按重排序得分排序；子图内三元组按查询相关性排序。
            # graph_rank 为图谱通道内的全局展示排名（子图排名 -> 三元组排名），
            # 供 3_evaluate.py 计算统一通道排名字段。
            reranked_sorted = reranked[:RETRIEVAL['graph_top_k']]
            graph_rank = 0
            source_label = ('graph_expand' if variant == '-no-graph-rerank'
                            else 'graph_rerank')
            for presentation_rank, cand in enumerate(reranked_sorted, start=1):
                anchor = cand.get('anchor')
                anchor_id = cand.get('anchor_id')
                if not cand.get('ordered_edges'):
                    qv = self.embedder.encode_query(query)
                    ordered_edges, triple_scores = self._rank_edges_by_query(
                        cand.get('edges') or set(), qv)
                    cand['ordered_edges'] = ordered_edges
                    cand['triple_scores'] = triple_scores
                else:
                    ordered_edges = cand['ordered_edges']
                    triple_scores = cand.get('triple_scores') or {}
                subgraph_rank = cand.get('subgraph_rank', presentation_rank)
                evidence_items = self._edges_to_evidence(
                    ordered_edges, triple_scores=triple_scores,
                    subgraph_rank=subgraph_rank,
                    subgraph_score=cand.get('score', 0.0),
                    subgraph_id=cand.get('subgraph_id'),
                    anchor=anchor, anchor_id=anchor_id,
                    source=source_label,
                    path_ids=path_ids_by_edge,
                )
                for t in evidence_items:
                    graph_rank += 1
                    t['graph_rank'] = graph_rank
                    graph_evidence.append(t)
                total_nodes |= cand['nodes']

        detail = {
            'fused_text': fused,
            'graph_evidence_count': len(graph_evidence),
            'graph_nodes_count': len(total_nodes),
            'variant': variant,
            'use_extra_heuristics': use_extra_heuristics,
        }
        return text_evidence, graph_evidence, detail

    def _edges_to_evidence(self, ordered_edges, triple_scores=None,
                           subgraph_rank=None, subgraph_score=None,
                           subgraph_id=None, anchor=None, anchor_id=None,
                           source='graph_rerank', path_ids=None):
        """有序三元组边 → 图谱证据（保留子图/三元组排名与溯源信息）。

        ordered_edges 必须已按子图内查询相关性排序；本方法不再按实体名称
        重排，从而保证图证据顺序与图谱重排序结果一致。
        """
        out = []
        triple_scores = triple_scores or {}
        path_ids = path_ids or {}
        for triple_rank, (src, dst, rel) in enumerate(ordered_edges, start=1):
            edge = (src, dst, rel)
            item = {
                'head': self.kg.node_name(src),
                'head_type': self.kg.node_entity(src),
                'rel': rel,
                'tail': self.kg.node_name(dst),
                'tail_type': self.kg.node_entity(dst),
                'triple_id': self._triple_id(src, dst, rel),
                'anchor': anchor,
                'anchor_id': anchor_id,
                'subgraph_id': subgraph_id,
                'subgraph_rank': subgraph_rank,
                'subgraph_score': (round(float(subgraph_score), 6)
                                   if subgraph_score is not None else None),
                'triple_rank': triple_rank,
                'triple_score': round(float(triple_scores.get(edge, 0.0)), 6),
                'source': source,
                'channel': 'graph',
            }
            ids = path_ids.get(edge)
            if ids:
                item['path_ids'] = list(ids)
            out.append(item)
        return out
