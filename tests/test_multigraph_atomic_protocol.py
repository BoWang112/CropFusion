# -*- coding: utf-8 -*-
"""CropFusion 多重图与统一原子证据协议单元测试。

覆盖：
- MultiDiGraph 的关系唯一键、重复三元组来源合并；
- 多关系路径的方向/关系类型过滤与真实关系边恢复；
- 图谱扩展、子图重排序、社区投影、参数扫描的 MultiDiGraph 兼容性；
- 文本/图谱两路原子证据的通道内排名、统一分数合并、Top-3 指标；
- Gold 事实去重、重复命中的单次统计；
- 八种检索方法的统一评估协议；
- 单独评估与批量评估同一方法的结果一致性。
"""
import importlib.util
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RETRIEVAL_DIR = ROOT / 'CDPKG' / 'retrieval_qa'
if str(RETRIEVAL_DIR) not in sys.path:
    sys.path.insert(0, str(RETRIEVAL_DIR))

from modules.kg import CropPestKG  # noqa: E402
from modules.retriever import Retriever  # noqa: E402
from graphrag.community import build_projection  # noqa: E402


def _ensure_stub(name, attrs):
    """在缺少重型依赖时，为 3_evaluate.py 的导入提供轻量占位模块。"""
    if name in sys.modules:
        return
    module = types.ModuleType(name)
    for attr in attrs:
        setattr(module, attr, object)
    sys.modules[name] = module


def _load_eval_module():
    if 'modules' not in sys.modules:
        pkg = types.ModuleType('modules')
        pkg.__path__ = []
        sys.modules['modules'] = pkg
    _ensure_stub('modules.prompts', [
        'FewShotSelector', 'BGEEmbedder', 'load_fewshot_pool',
        'DEFAULT_FEWSHOT_FILE', 'build_prompt_with_dynamic_fewshot',
    ])
    _ensure_stub('modules.llm', ['LLMClient'])
    _ensure_stub('modules.task_classifier', ['classify_task'])
    path = RETRIEVAL_DIR / '3_evaluate.py'
    spec = importlib.util.spec_from_file_location('cdpkg_eval3_tests', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_graph(nodes, relationships):
    """nodes: [(id,name,entity)], relationships: [(s,rel,e,props)]"""
    tmpdir = tempfile.mkdtemp(prefix='cropfusion_test_')
    nodes_path = os.path.join(tmpdir, 'nodes.csv')
    rels_path = os.path.join(tmpdir, 'relationships.csv')
    with open(nodes_path, 'w', encoding='utf-8-sig', newline='') as f:
        f.write('id:ID,name,entity,props\n')
        for nid, name, entity in nodes:
            f.write(f'{nid},{name},{entity},"{{}}"\n')
    with open(rels_path, 'w', encoding='utf-8-sig', newline='') as f:
        f.write(':START_ID,:TYPE,:END_ID,props\n')
        for src, rel, dst, props in relationships:
            f.write(f'{src},{rel},{dst},'
                    f'"{json.dumps(props, ensure_ascii=False).replace(chr(34), chr(34) * 2)}"\n')
    return tmpdir, nodes_path, rels_path


class FakeEmbedder:
    """固定二维向量，避免测试依赖 BGE/transformers。"""

    def __init__(self):
        self.vector = np.array([1.0, 0.0])

    def encode_query(self, query):
        return self.vector

    def encode(self, texts, batch_size=64):
        if isinstance(texts, str):
            texts = [texts]
        return np.stack([self.vector for _ in texts]) if texts else np.zeros((0, 2))


def _make_retriever(kg, scores_by_rel=None):
    retriever = Retriever.__new__(Retriever)
    retriever.kg = kg
    retriever.embedder = FakeEmbedder()
    retriever.expand_rels = {'为害', '影响', '症状有', '胁迫有', '防治方法有'}
    retriever.evidence_entities = {
        '作物', '病害', '虫害', '害虫', '病原体', '具体症状', '发病部位',
        '化学防治', '农业防治', '物理防治', '生物防治', '其他因素',
    }
    default_scores = {
        '为害': 1.0, '影响': 0.8, '症状有': 0.5,
        '胁迫有': 0.3, '防治方法有': 0.1,
    }
    scores = dict(default_scores)
    if scores_by_rel:
        scores.update(scores_by_rel)

    def _encode(edges):
        return np.array([[scores.get(e[2], 0.0), 0.0] for e in edges])

    retriever._encode_triples = _encode
    return retriever


class MultiDiGraphTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir, self.nodes, self.rels = _write_graph(
            [('A', '病害A', '病害'), ('B', '症状B', '具体症状'),
             ('C', '作物C', '作物')],
            [
                ('A', 'R1', 'B', {'来源ID': ['s1']}),
                ('A', 'R2', 'B', {'来源ID': ['s2']}),
                ('A', 'R1', 'B', {'来源ID': ['s3']}),
                ('B', 'R3', 'C', {'来源ID': ['s4']}),
                ('C', 'R4', 'B', {'来源ID': ['s5']}),
            ],
        )
        self.kg = CropPestKG(self.nodes, self.rels)

    def test_multiple_relations_and_duplicate_merge(self):
        g = self.kg.graph
        self.assertTrue(g.is_multigraph())
        # 4 个唯一 (source, rel, target)，重复的 A-R1-B 不新增边
        self.assertEqual(g.number_of_edges(), 4)
        self.assertEqual(sorted(self.kg.neighbors('A')),
                         [('B', 'R1'), ('B', 'R2')])
        self.assertEqual(sorted(self.kg.in_neighbors('B')),
                         [('A', 'R1'), ('A', 'R2'), ('C', 'R4')])
        props = g.get_edge_data('A', 'B', key='R1')['props']
        self.assertEqual(sorted(props['来源ID']), ['s1', 's3'])

    def test_ego_subgraph_keeps_all_relations(self):
        nodes, edges = self.kg.ego_subgraph(
            ['A'], hop=1, rel_types={'R1', 'R2'})
        self.assertEqual(nodes, {'A', 'B'})
        self.assertIn(('A', 'B', 'R1'), edges)
        self.assertIn(('A', 'B', 'R2'), edges)

    def test_paths_direction_and_relation_variants(self):
        paths = self.kg.paths_between(
            ['A', 'C'], max_len=2, rel_types={'R1', 'R2', 'R3'})
        self.assertEqual(len(paths), 2)
        edge_sets = {tuple((u, rel) for u, _v, rel in p[3]) for p in paths}
        self.assertEqual(edge_sets, {(('A', 'R1'), ('B', 'R3')),
                                     (('A', 'R2'), ('B', 'R3'))})
        # 关系类型过滤：R1 无法到达 C
        self.assertEqual(self.kg.paths_between(
            ['A', 'C'], max_len=2, rel_types={'R1'}), [])
        # 方向约束：不存在 C -> A 的有向路径
        self.assertEqual(self.kg.paths_between(
            ['C', 'A'], max_len=2,
            rel_types={'R1', 'R2', 'R3', 'R4'}), [])
        # 无向视图仅在显式请求时使用
        undirected = self.kg.paths_between(
            ['C', 'A'], max_len=2, directed=False)
        self.assertTrue(undirected)

    def test_community_projection_accumulates_relations(self):
        # A-B 上的 R1/R2 都按诊断关系权重 1.0，应累加为 2.0
        proj = build_projection(self.kg)
        self.assertAlmostEqual(proj['A']['B']['weight'], 2.0, places=6)


class RetrieverGraphTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir, nodes, rels = _write_graph(
            [('A', '病害A', '病害'), ('B', '症状B', '具体症状'),
             ('C', '作物C', '作物'), ('D', '部位D', '发病部位'),
             ('E', '药剂E', '化学防治')],
            [
                ('A', '为害', 'B', {}),
                ('A', '影响', 'B', {}),
                ('A', '胁迫有', 'C', {}),
                ('A', '症状有', 'D', {}),
                ('A', '防治方法有', 'E', {}),
            ],
        )
        self.kg = CropPestKG(nodes, rels)
        self.retriever = _make_retriever(self.kg)

    def test_directed_expand_keeps_multi_relations(self):
        _nodes, edges = self.retriever._directed_expand(['A'], hop=1)
        self.assertIn(('A', 'B', '为害'), edges)
        self.assertIn(('A', 'B', '影响'), edges)

    def test_graph_rerank_preserves_triple_order_and_metadata(self):
        _nodes, edges = self.retriever._directed_expand(['A'], hop=1)
        ranked = self.retriever.graph_rerank(
            'query', [edges], anchors=['A'], top_k=1)[0]
        # 子图内按三元组-查询相关性排序：为害 > 影响 > 症状有 > 胁迫有
        self.assertEqual(ranked['ordered_edges'][0], ('A', 'B', '为害'))
        self.assertEqual(ranked['ordered_edges'][1], ('A', 'B', '影响'))
        self.assertEqual(ranked['subgraph_rank'], 1)
        self.assertIn('subgraph_id', ranked)
        evidence = self.retriever._edges_to_evidence(
            ranked['ordered_edges'],
            triple_scores=ranked['triple_scores'],
            subgraph_rank=ranked['subgraph_rank'],
            subgraph_score=ranked['score'],
            subgraph_id=ranked['subgraph_id'],
            anchor='病害A', anchor_id='A', source='graph_rerank',
        )
        self.assertEqual(evidence[0]['triple_rank'], 1)
        self.assertEqual(evidence[0]['triple_score'], 1.0)
        self.assertEqual(evidence[0]['subgraph_rank'], 1)
        self.assertEqual(evidence[0]['subgraph_id'],
                         ranked['subgraph_id'])
        self.assertEqual(evidence[0]['channel'], 'graph')

    def test_get_evidence_ranks_and_path_ids_flow_to_evaluation(self):
        retriever = _make_retriever(self.kg)
        retriever.block_by_id = {
            'b1': {
                'id': 'b1', 'text': '文本事实', 'entity_name': '病害A',
                'entity_type': '病害', 'source': 'KB',
                'meta': {'segment_id': 'b1'},
            }
        }
        retriever.text_fusion = lambda *a, **k: [('b1', 0.9)]
        retriever.semantic_graph_search = lambda *a, **k: ([], [])
        path_edges = [('A', 'B', '为害')]
        retriever.entity_graph_search = lambda *a, **k: (
            ['病害A'], ['A'], [set(path_edges)],
            [('A', 'C', ['A', 'B', 'C'], path_edges, 'path::test')],
        )
        text_ev, graph_ev, _detail = retriever.get_evidence('query')
        self.assertEqual(text_ev[0]['rrf_rank'], 1)
        self.assertTrue(graph_ev)
        for item in graph_ev:
            self.assertIn('graph_rank', item)
            self.assertIn('subgraph_rank', item)
            self.assertIn('triple_rank', item)
            self.assertIn('triple_score', item)
            self.assertEqual(item['anchor_id'], 'A')
            self.assertIn('subgraph_id', item)
            self.assertEqual(item['path_ids'], ['path::test'])
        self.assertEqual(
            [x['graph_rank'] for x in graph_ev],
            list(range(1, len(graph_ev) + 1)),
        )
        # 图谱重排序排名可直接进入统一原子评估
        E = _load_eval_module()
        atoms = E._build_ranked_atoms(text_ev, graph_ev, 'Ours')
        self.assertEqual(atoms[0]['id'], 'text::b1')
        self.assertIn('triple::', atoms[1]['id'])
        self.assertEqual(atoms[1]['channel'], 'graph')
        self.assertEqual(atoms[1]['graph_rank'], graph_ev[0]['graph_rank'])


class AtomicEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.E = _load_eval_module()

    def test_unified_channel_rank_and_top3(self):
        text = [
            {'id': 'b1', 'text': '事实1', 'segment_id': 'seg1',
             'score': 0.9, 'rrf_rank': 1, 'source': 'KB'},
            {'id': 'b2', 'text': '事实2', 'segment_id': 'seg2',
             'score': 0.8, 'rrf_rank': 2, 'source': 'KB'},
            {'id': 'b3', 'text': '事实3', 'segment_id': 'seg3',
             'score': 0.7, 'rrf_rank': 3, 'source': 'KB'},
        ]
        graph = [
            {'head': 'A', 'rel': 'R1', 'tail': 'B', 'triple_id': 't1',
             'graph_rank': 1, 'subgraph_id': 'sg1', 'subgraph_rank': 1,
             'triple_rank': 1, 'triple_score': 0.9, 'source': 'graph_rerank'},
            {'head': 'A', 'rel': 'R2', 'tail': 'C', 'triple_id': 't2',
             'graph_rank': 2, 'subgraph_id': 'sg1', 'subgraph_rank': 1,
             'triple_rank': 2, 'triple_score': 0.8, 'source': 'graph_rerank'},
        ]
        atoms = self.E._build_ranked_atoms(text, graph, 'M1')
        ids = [a['id'] for a in atoms]
        self.assertEqual(ids[:4],
                         ['text::seg1', 'triple::t1',
                          'text::seg2', 'triple::t2'])
        self.assertAlmostEqual(atoms[0]['unified_score'], 1 / 61)
        self.assertAlmostEqual(atoms[1]['unified_score'], 1 / 61)
        # 同分按与检索方法名无关的原子 ID 打破平局
        atoms2 = self.E._build_ranked_atoms(text, graph, 'OtherMethod')
        self.assertEqual([a['id'] for a in atoms2], ids)
        self.assertEqual([a['rank'] for a in atoms2], list(range(1, len(ids) + 1)))

    def test_graph_evidence_keeps_path_and_subgraph_ids(self):
        graph = [{
            'path_id': 'path::1', 'path_rank': 1, 'graph_rank': 1,
            'triple_rank': 1, 'source': 'pathrag',
            'edges': [
                {'head': 'A', 'rel': 'R1', 'tail': 'B', 'triple_rank': 1},
                {'head': 'B', 'rel': 'R2', 'tail': 'C', 'triple_rank': 2},
            ],
        }]
        atoms = self.E._build_ranked_atoms([], graph, 'PathRAG')
        self.assertEqual([a['relation'] for a in atoms], ['R1', 'R2'])
        self.assertTrue(all(a['path_id'] == 'path::1' for a in atoms))
        self.assertEqual([a['triple_rank'] for a in atoms], [1, 2])

    def test_dedup_merges_sources(self):
        graph = [
            {'head': 'A', 'rel': 'R', 'tail': 'B', 'triple_id': 't1',
             'graph_rank': 1, 'subgraph_id': 'sg1', 'source': 'graph_rerank'},
            {'head': 'A', 'rel': 'R', 'tail': 'B', 'triple_id': 't1',
             'graph_rank': 5, 'subgraph_id': 'sg2', 'source': 'graph_expand'},
        ]
        atoms = self.E._build_ranked_atoms([], graph, 'M')
        self.assertEqual(len(atoms), 1)
        atom = atoms[0]
        self.assertIn('graph_expand', atom.get('duplicate_sources', []))
        self.assertIn('sg2', atom.get('parents', []))

    def test_gold_fact_dedup_and_duplicate_hit_counted_once(self):
        qa = {
            'gold_evidence_ids': ['segA', 'segB'],
            'gold_evidence_texts': ['同一事实', '同一事实'],
            'gold_pest': 'P',
            'gold_graph_evidence': [{'head': 'A', 'rel': 'R', 'tail': 'B'}],
        }
        blocks_by_entity = {'P': [
            {'id': 'b1', 'text': '同一事实', 'entity_name': 'P'},
            {'id': 'b2', 'text': '同一事实', 'entity_name': 'P'},
        ]}
        seg_index = {
            'segA': {'text': '同一事实', 'block_id': 'b1',
                     'exact': True, 'pest': 'P'},
            'segB': {'text': '同一事实', 'block_id': 'b2',
                     'exact': True, 'pest': 'P'},
        }
        gold_atoms, _blocks, _stats = self.E._gold_text_atoms(
            qa, blocks_by_entity, seg_index)
        self.assertEqual(len(gold_atoms), 1)
        self.assertEqual(gold_atoms[0]['segment_ids'], {'segA', 'segB'})

        text = [
            {'id': 'b1', 'text': '同一事实', 'segment_id': 'segA',
             'source': 'KB'},
            {'id': 'b2', 'text': '同一事实', 'segment_id': 'segB',
             'source': 'KB'},
        ]
        graph = [{'head': 'A', 'rel': 'R', 'tail': 'B', 'triple_id': 't1'}]
        ranked = self.E._build_ranked_atoms(text, graph, 'M')
        score = self.E._score_atom_ranking(
            ranked, gold_atoms, {('A', 'R', 'B')}, 3)
        self.assertAlmostEqual(score['p'], 2 / 3)
        self.assertAlmostEqual(score['r'], 1.0)
        self.assertAlmostEqual(score['f1'], 0.8)
        self.assertAlmostEqual(score['ndcg'], 1.0)

    def test_eight_methods_share_protocol_and_single_matches_batch(self):
        E = self.E
        tmpdir = tempfile.mkdtemp(prefix='cropfusion_eval_')
        extract_json = os.path.join(tmpdir, 'extract.json')
        with open(extract_json, 'w', encoding='utf-8') as f:
            json.dump([{
                'crop': '作物', 'pest': 'P', 'domain': '病害', 'row': 1,
                'segments': [{'id': 'seg1', 'text': '事实一的具体描述内容'}],
            }], f, ensure_ascii=False)
        blocks = [{
            'id': 'seg1', 'text': '事实一的具体描述内容',
            'entity_name': 'P', 'entity_type': '病害', 'source': 'KB',
            'meta': {'segment_id': 'seg1', 'source_id': 's1'},
        }]
        store = SimpleNamespace(blocks=blocks)
        triple_id = E._triple_id('A', 'R', 'B')
        qa = {
            'id': 'q1', 'question': '问题', 'gold_pest': 'P',
            'gold_evidence_ids': ['seg1'],
            'gold_evidence_texts': ['事实一的具体描述内容'],
            'gold_graph_evidence': [{
                'head': 'A', 'rel': 'R', 'tail': 'B',
                'triple_id': triple_id,
            }],
        }
        text_ev = [{
            'id': 'seg1', 'text': '事实一的具体描述内容',
            'segment_id': 'seg1', 'rrf_rank': 1, 'score': 0.9,
            'source': 'KB', 'meta': {'segment_id': 'seg1'},
        }]
        graph_ev = [{
            'head': 'A', 'rel': 'R', 'tail': 'B', 'triple_id': triple_id,
            'graph_rank': 1, 'subgraph_id': 'sg1', 'subgraph_rank': 1,
            'triple_rank': 1, 'triple_score': 0.8,
            'source': 'graph_rerank',
        }]

        original_engines = E._build_retrieval_engines
        original_run = E._run_retrieval_method
        original_build = E._build_ranked_atoms
        build_calls = []

        def _spy_build(*args, **kwargs):
            build_calls.append(args[2] if len(args) > 2 else kwargs.get('method_label'))
            return original_build(*args, **kwargs)

        try:
            E._build_retrieval_engines = lambda *a, **k: {}
            E._run_retrieval_method = lambda *a, **k: (text_ev, graph_ev, {})
            E._build_ranked_atoms = _spy_build
            methods = list(E.ALL_RETRIEVAL_METHODS)
            results, _diag = E.eval_retrieval_methods(
                None, None, store, [qa], methods, k=3,
                extract_json=extract_json, llm=object())
            self.assertEqual(len(results), 8)
            self.assertEqual(len(build_calls), 8)
            signatures = {
                (round(m['p'], 12), round(m['r'], 12),
                 round(m['f1'], 12), round(m['ndcg'], 12))
                for m in results.values()
            }
            self.assertEqual(len(signatures), 1)

            single, _n, _diag = E.eval_retrieval(
                None, None, store, [qa], k=3,
                extract_json=extract_json, llm=object())
            self.assertAlmostEqual(single['p'], next(iter(signatures))[0], places=12)
            self.assertAlmostEqual(single['r'], next(iter(signatures))[1], places=12)
            self.assertAlmostEqual(single['f1'], next(iter(signatures))[2], places=12)
            self.assertAlmostEqual(single['ndcg'], next(iter(signatures))[3], places=12)
        finally:
            E._build_retrieval_engines = original_engines
            E._run_retrieval_method = original_run
            E._build_ranked_atoms = original_build


class DownstreamCompatibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.E = _load_eval_module()

    def setUp(self):
        self.tmpdir, self.nodes, self.rels = _write_graph(
            [('A', '病害A', '病害'), ('B', '症状B', '具体症状'),
             ('C', '作物C', '作物')],
            [
                ('A', '症状有', 'B', {}),
                ('A', '为害', 'B', {}),
                ('A', '胁迫有', 'C', {}),
            ],
        )
        self.kg = CropPestKG(self.nodes, self.rels)

    def test_param_sweep_neighbors_compatible(self):
        import param_sweep_retrieval
        neighbors = param_sweep_retrieval.gold_graph_neighbors(self.kg, '病害A')
        self.assertEqual(neighbors, {'病害A', '症状B', '作物C'})

    def test_table3_neighbors_compatible(self):
        import table3_graph_metrics
        neighbors = table3_graph_metrics.gold_graph_neighbors(self.kg, '病害A')
        self.assertEqual(neighbors, {'病害A', '症状B', '作物C'})

    def test_summarize_profile_compatible(self):
        from graphrag.summarize import _build_community_profile
        profile = _build_community_profile(self.kg, {'A', 'B', 'C'})
        self.assertIn('症状B', profile['symptoms'])
        self.assertIn('作物C', profile['crops'])

    def test_generate_global_qa_compatible(self):
        from generate_global_qa import collect_crop_values
        values = collect_crop_values(
            self.kg, 'C', '症状有', {'具体症状'})
        self.assertIn('症状B', values)


class GraphRAGBaselineCompatibilityTests(unittest.TestCase):
    """各 GraphRAG 基线在 MultiDiGraph 上仍能正常检索。"""

    def setUp(self):
        self.tmpdir, nodes, rels = _write_graph(
            [('A', '病害A', '病害'), ('B', '症状B', '具体症状'),
             ('C', '作物C', '作物'), ('D', '药剂D', '化学防治')],
            [
                ('A', '症状有', 'B', {}),
                ('A', '为害', 'B', {}),
                ('A', '胁迫有', 'C', {}),
                ('A', '防治方法有', 'D', {}),
            ],
        )
        self.kg = CropPestKG(nodes, rels)
        self.retriever = _make_retriever(self.kg)
        self.retriever.blocks = SimpleNamespace(blocks=[{
            'id': 'b1', 'text': '病害A 的症状描述', 'entity_id': 'A',
            'entity_name': '病害A', 'entity_type': '病害', 'source': 'KB',
            'meta': {'segment_id': 'b1', 'source_id': 's1'},
        }])
        self.retriever.block_by_id = {b['id']: b for b in self.retriever.blocks.blocks}
        self.retriever.pest_vecs = None
        self.retriever.pest_vec_ids = []
        self.retriever.parser = None
        self.retriever._find_target_pest = lambda q: {'A'}
        self.retriever._keyword_target_pests = lambda q, top_k=2: ['A']
        _nodes, edges = self.retriever._directed_expand(['A'], hop=2)
        self.retriever.semantic_graph_search = lambda q, top_k=None: (
            ['A'], [edges])

    def _assert_graph_evidence(self, graph_ev):
        self.assertTrue(graph_ev)
        triples = {(x['head'], x['rel'], x['tail']) for x in graph_ev}
        self.assertIn(('病害A', '症状有', '症状B'), triples)
        self.assertIn(('病害A', '为害', '症状B'), triples)

    def test_graphrag_local(self):
        from graphrag.search import GraphRAGEngine
        text_ev, graph_ev, _detail = GraphRAGEngine(
            self.kg, self.retriever, llm=None).local_search('病害A 症状')
        self.assertTrue(text_ev)
        self._assert_graph_evidence(graph_ev)

    def test_g_retriever(self):
        from graphrag.gretriever import GRetrieverEngine
        _text, graph_ev, _detail = GRetrieverEngine(
            self.kg, self.retriever).search('病害A 症状')
        self._assert_graph_evidence(graph_ev)

    def test_pathrag(self):
        from graphrag.pathrag import PathRAGEngine
        _text, graph_ev, _detail = PathRAGEngine(
            self.kg, self.retriever).search('病害A 症状')
        self._assert_graph_evidence(graph_ev)
        self.assertTrue(any(x.get('path_id') for x in graph_ev))

    def test_sgrag(self):
        from graphrag.sgrag import SGRAGEngine
        _text, graph_ev, _detail = SGRAGEngine(
            self.kg, self.retriever).search('病害A 症状')
        self._assert_graph_evidence(graph_ev)

    def test_lightrag(self):
        from graphrag.lightrag import LightRAGEngine
        _text, graph_ev, _detail = LightRAGEngine(
            self.kg, self.retriever).search('病害A 防治')
        self._assert_graph_evidence(graph_ev)


if __name__ == '__main__':
    unittest.main(verbosity=2)
