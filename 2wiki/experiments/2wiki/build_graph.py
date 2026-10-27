# -*- coding: utf-8 -*-
"""
build_graph.py —— question-specific local graph 构建
=====================================================
每个 question 单独建图：
- 节点：candidate context 中的 article title
- 节点属性：{title, paragraph_text, sentence_ids, embedding}
- 边：title mention 匹配（优先主名匹配，无超链接元数据）

匹配规则（仅从 candidate context 建边，无泄漏）：
1. title 全字符串出现在另一 article 正文中 → A→B
2. title 主名（去掉括号消歧，如 "Theodred II (Bishop of Elmham)" → "Theodred II"）
   出现在另一 article 正文中 → A→B
3. 句子开头实体名匹配（很多 title 在句首以完整形式出现）

节点 embedding = BGE(title + paragraph_text)。
"""
import sys
import os
import json
import re

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DATA, SAMPLE_FILE
import numpy as np

GRAPH_FILE = os.path.join(DATA, 'graphs.json')


def main_title(t):
    """去掉括号消歧，得到主名。如 'Theodred II (Bishop of Elmham)' -> 'Theodred II'"""
    m = re.match(r'^(.+?)\s*\(', t)
    return m.group(1) if m else t


def build_sentence_lookup():
    """question_id -> {title -> [(sentence_id, text), ...]}"""
    lookup = {}
    for line in open(os.path.join(DATA, 'sentences.jsonl'), encoding='utf-8'):
        s = json.loads(line)
        lookup.setdefault(s['question_id'], {}).setdefault(s['title'], []).append(
            (s['sentence_id'], s['text']))
    return lookup


def main():
    import torch
    from transformers import AutoTokenizer, AutoModel
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
    from common import EMBED_MODEL_PATH
    tok = AutoTokenizer.from_pretrained(EMBED_MODEL_PATH)
    model = AutoModel.from_pretrained(EMBED_MODEL_PATH)
    model.eval()

    sent_lookup = build_sentence_lookup()
    rows = [json.loads(l) for l in open(SAMPLE_FILE, encoding='utf-8')]
    graphs = {}

    for r in rows:
        qid = r['id']
        ctx = r['context']
        titles = list(ctx['title'])
        # 每篇文章的完整段落文本 + 句子 id 列表
        para_text = {}
        sent_ids = {}
        for ti, t in enumerate(titles):
            sents = ctx['sentences'][ti] if ti < len(ctx['sentences']) else []
            para_text[t] = ' '.join(sents)
            sent_ids[t] = list(range(len(sents)))

        # 边：title mention（主名匹配）
        edges = []   # (src, dst)
        main_names = {t: main_title(t) for t in titles}
        for i, src in enumerate(titles):
            src_main = main_names[src]
            for j, dst in enumerate(titles):
                if i == j:
                    continue
                dst_main = main_names[dst]
                body = para_text[src]
                # 主名出现在正文（且不是出现在 src 自身开头的自身定义中，用词边界限制）
                if dst_main and dst_main.lower() in body.lower():
                    edges.append((src, dst))

        # 去重边
        seen = set()
        dedup_edges = []
        for e in edges:
            if e not in seen:
                seen.add(e)
                dedup_edges.append(e)

        # 节点 embedding（title + paragraph）
        node_texts = [f'{t} {para_text.get(t, "")}' for t in titles]
        node_embs = []
        with torch.no_grad():
            for i in range(0, len(node_texts), 32):
                batch = node_texts[i:i + 32]
                enc = tok(batch, padding=True, truncation=True, max_length=512,
                          return_tensors='pt')
                out = model(**enc)
                node_embs.append(out.pooler_output.numpy())
        node_embs = np.vstack(node_embs) if node_embs else np.zeros((0, 768))

        graphs[qid] = {
            'nodes': [
                {'title': t, 'paragraph_text': para_text.get(t, ''),
                 'sentence_ids': sent_ids.get(t, [])}
                for t in titles
            ],
            'edges': [{'source': s, 'target': d} for s, d in dedup_edges],
            'node_embeddings': node_embs.tolist(),
        }

    with open(GRAPH_FILE, 'w', encoding='utf-8') as f:
        json.dump(graphs, f, ensure_ascii=False)

    # 统计
    n_edges = [len(g['edges']) for g in graphs.values()]
    import statistics
    print(f'图数量: {len(graphs)}')
    print(f'平均边数: {statistics.mean(n_edges):.2f} (中位 {statistics.median(n_edges)})')
    print(f'0边图: {sum(1 for e in n_edges if e == 0)}')
    print(f'节点数范围: {min(len(g["nodes"]) for g in graphs.values())}-{max(len(g["nodes"]) for g in graphs.values())}')
    print(f'已保存 {GRAPH_FILE}')


if __name__ == '__main__':
    main()
