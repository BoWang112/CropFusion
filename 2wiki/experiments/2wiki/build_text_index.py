# -*- coding: utf-8 -*-
"""
build_text_index.py —— 句子语料 + Dense embedding + BM25 索引
=============================================================
- 从 sample 和 fewshot_pool 的 candidate context 拆分句子
- 每句存为 {question_id, title, sentence_id, text}
- evidence_id = title + "::" + sentence_id
- 对所有句子批量生成 BGE 英文 embedding（离线模型）
- 保存句子表、embedding、question embedding（test + fewshot）

知识范围：每个 question 只包含其自身 candidate context 的句子，
严格禁止跨 question 访问。BM25 索引在运行时按 question 过滤构建。
"""
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (DATA, SAMPLE_FILE, FEWSHOT_FILE, EMBED_MODEL_PATH,
                    EMBED_DIM)
import numpy as np

SENT_FILE = os.path.join(DATA, 'sentences.jsonl')
EMB_FILE = os.path.join(DATA, 'sentence_embeddings.npy')
Q_EMB_FILE = os.path.join(DATA, 'question_embeddings.npy')
LOOKUP_FILE = os.path.join(DATA, 'sentence_lookup.json')


def load_embedder():
    import torch
    from transformers import AutoTokenizer, AutoModel
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
    tok = AutoTokenizer.from_pretrained(EMBED_MODEL_PATH)
    model = AutoModel.from_pretrained(EMBED_MODEL_PATH)
    model.eval()
    return tok, model


def embed_texts(tok, model, texts, batch_size=32):
    import torch
    vecs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = tok(batch, padding=True, truncation=True, max_length=512,
                      return_tensors='pt')
            out = model(**enc)
            # BGE 建议对 query 加 'Represent this sentence...' 前缀，但统一公平起见
            # 全部用 pooler 输出（sentence 和 question 同处理）
            vecs.append(out.pooler_output.numpy())
    return np.vstack(vecs)


def main():
    tok, model = load_embedder()
    print(f'embedder 加载完成 (dim={EMBED_DIM})')

    # 收集所有 question
    all_q = []
    n_by_file = {}
    for path in (SAMPLE_FILE, FEWSHOT_FILE):
        n = 0
        for line in open(path, encoding='utf-8'):
            all_q.append(json.loads(line))
            n += 1
        n_by_file[path] = n

    # 拆句：每个 question 的 candidate context → sentence 列表
    sentences = []       # [{question_id, title, sentence_id, text}]
    q_texts = []         # question 文本（对应 all_q 顺序）
    q_sent_idx = {}      # question_id -> sentence 在 sentences 中的 index 范围
    for qi, q in enumerate(all_q):
        qid = q['id']
        q_texts.append(q['question'])
        ctx = q['context']
        idx_list = []
        for ti, title in enumerate(ctx['title']):
            sents = ctx['sentences'][ti] if ti < len(ctx['sentences']) else []
            for si, stext in enumerate(sents):
                idx_list.append(len(sentences))
                sentences.append({
                    'question_id': qid,
                    'title': title,
                    'sentence_id': si,
                    'text': stext,
                    'evidence_id': f'{title}::{si}',
                })
        q_sent_idx[qid] = idx_list

    print(f'句子总数: {len(sentences)}')
    print(f'问题总数: {len(all_q)} '
          f'(test {n_by_file[SAMPLE_FILE]} + fewshot {n_by_file[FEWSHOT_FILE]})')

    # 批量嵌入句子（去重，节省内存）
    uniq_texts = []
    uniq_map = {}
    for s in sentences:
        if s['text'] not in uniq_map:
            uniq_map[s['text']] = len(uniq_texts)
            uniq_texts.append(s['text'])
    print(f'去重后句子文本: {len(uniq_texts)}')
    uniq_emb = embed_texts(tok, model, uniq_texts)
    sent_emb = uniq_emb[[uniq_map[s['text']] for s in sentences]]
    print(f'sentence embedding shape: {sent_emb.shape}')

    # 问题 embedding
    q_emb = embed_texts(tok, model, q_texts)
    print(f'question embedding shape: {q_emb.shape}')

    # 保存
    with open(SENT_FILE, 'w', encoding='utf-8') as f:
        for s in sentences:
            f.write(json.dumps(s, ensure_ascii=False) + '\n')
    np.save(EMB_FILE, sent_emb)
    np.save(Q_EMB_FILE, q_emb)
    with open(LOOKUP_FILE, 'w', encoding='utf-8') as f:
        json.dump({
            'question_order': [q['id'] for q in all_q],
            'question_type': [q['type'] for q in all_q],
            'q_sent_idx': q_sent_idx,
        }, f, ensure_ascii=False, indent=1)
    print('已保存:')
    print(f'  {SENT_FILE}')
    print(f'  {EMB_FILE}')
    print(f'  {Q_EMB_FILE}')
    print(f'  {LOOKUP_FILE}')


if __name__ == '__main__':
    main()
