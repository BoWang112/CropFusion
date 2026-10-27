# -*- coding: utf-8 -*-
"""
2wiki 实验公共模块
====================
统一配置、数据加载、evidence 预算控制、answer 归一化、提示词构造。
所有 9 种方法共享此模块，保证公平性（相同 LLM / 相同生成参数 /
相同 evidence 预算 / 相同 normalize / 相同 prompt 约束）。

LLM: DeepSeek API (deepseek-chat) —— 与 CDPKG 主实验一致。
Embedding: BAAI/bge-base-en-v1.5（本地离线，768 维）。
"""
import os
import re
import json
import hashlib
import time
import unicodedata
from collections import Counter

# ---------- 路径 ----------
BASE = os.path.dirname(os.path.abspath(__file__))        # .../2wiki/experiments/2wiki
EXP = os.path.dirname(BASE)                              # .../2wiki/experiments
ROOT = os.path.dirname(EXP)                              # .../2wiki
DATA = os.path.join(ROOT, 'data')
OUT = os.path.join(ROOT, 'outputs')
PRED = os.path.join(OUT, 'predictions')
TABLES = os.path.join(OUT, 'tables')
for d in (PRED, TABLES):
    os.makedirs(d, exist_ok=True)

# ---------- 数据文件 ----------
SAMPLE_FILE = os.path.join(DATA, 'sample.jsonl')
IDS_FILE = os.path.join(DATA, 'sample_ids.json')
FEWSHOT_FILE = os.path.join(DATA, 'fewshot_pool.jsonl')
VAL_PARQUET = os.path.join(DATA, 'validation.parquet')
TRAIN_PARQUET = os.path.join(DATA, 'train_0.parquet')

# ---------- 固定实验参数 ----------
RANDOM_SEED = 42
TEST_TYPES = ['comparison', 'inference', 'compositional', 'bridge-comparison']
TEST_PER_TYPE = 100
FEWSHOT_PER_TYPE = 10

# 检索参数
DENSE_TOP_K = 20
BM25_TOP_K = 20
RRF_K = 60
SEMANTIC_ANCHOR_TOP_K = 5
GRAPH_HOPS = 2
FEWSHOT_TOP_K = 3
GRAPH_TOP_K = 5

# 证据预算
MAX_EVIDENCE_UNITS = 8
MAX_EVIDENCE_TOKENS = 1500

# 图重排序权重
SEMANTIC_WEIGHT = 0.5
STRUCTURAL_WEIGHT = 0.5

# 生成参数
TEMPERATURE = 0.2
TOP_P = 0.9
MAX_NEW_TOKENS = 64

# Embedding（CDPKG 项目下的离线模型）
EMBED_MODEL_PATH = os.path.join(
    os.path.dirname(ROOT),           # .../CDPKG-LLM
    'CDPKG', 'models', 'bge-base-en-v1.5')
EMBED_DIM = 768

# LLM（DeepSeek API）
LLM_MODEL = 'deepseek-chat'
LLM_TEMPERATURE = 0.2
LLM_TOP_P = 0.9
LLM_MAX_TOKENS = 64

# ---------- type 归一化 ----------
_TYPE_ALIAS = {
    'comparison': 'comparison',
    'inference': 'inference',
    'compositional': 'compositional',
    'bridge_comparison': 'bridge-comparison',
    'bridge-comparison': 'bridge-comparison',
    'bridgecomparison': 'bridge-comparison',
}


def normalize_type(t):
    return _TYPE_ALIAS.get(str(t).strip().lower(), str(t).strip())


# ---------- answer 归一化（标准英文 QA） ----------
_ARTICLES = {'a', 'an', 'the'}
_PUNCT = re.compile(r'[^\w\s]', re.UNICODE)


def normalize_answer(s):
    """lowercase → remove punct → remove articles → collapse whitespace"""
    s = unicodedata.normalize('NFD', s or '')
    s = s.lower()
    s = _PUNCT.sub(' ', s)
    words = s.split()
    words = [w for w in words if w not in _ARTICLES]
    return ' '.join(words)


def f1_score(pred, gold):
    """token 级 F1（HotpotQA 官方多重集口径）。
    用 Counter（多重集）而非 set，含重复词的答案 pred==gold 时 F1 必为 1.0。
    """
    p_tok = pred.split()
    g_tok = gold.split()
    if not p_tok and not g_tok:
        return 1.0
    if not p_tok or not g_tok:
        return 0.0
    common = Counter(p_tok) & Counter(g_tok)
    num_same = sum(common.values())
    if not num_same:
        return 0.0
    prec = num_same / len(p_tok)
    rec = num_same / len(g_tok)
    return 2 * prec * rec / (prec + rec)


# ---------- 生成提示词（统一格式） ----------
SYSTEM_PROMPT = (
    "You are a question answering assistant. Use the available information "
    "to answer the question. Return only the shortest final answer. "
    "Do not provide explanations. Do not provide reasoning steps. "
    "Do not repeat the question."
)

ANSWER_REQ = (
    "Use the available information to answer the question.\n"
    "Return only the shortest final answer.\n"
    "Do not provide explanations.\n"
    "Do not provide reasoning steps.\n"
    "Do not repeat the question."
)


def build_base_prompt(question):
    """零检索提示：仅问题 + 回答约束"""
    return (
        f"Question: {question}\n\n"
        f"{ANSWER_REQ}"
    )


def build_evidence_prompt(question, evidence_texts, fewshot_examples=None):
    """RAG 提示：可选 few-shot + 文本证据 + 图证据 + 问题。
    evidence_texts: 已经排序并截断的 evidence 字符串列表（≤8 units）"""
    parts = []
    if fewshot_examples:
        fs_parts = []
        for ex in fewshot_examples:
            fs_parts.append(
                f"Example Question: {ex['question']}\n"
                f"Example Evidence: {ex['evidence']}\n"
                f"Example Answer: {ex['answer']}\n")
        parts.append("Here are some examples:\n" + "\n".join(fs_parts))
    if evidence_texts:
        parts.append("Evidence:\n" + "\n".join(f"- {t}" for t in evidence_texts))
    parts.append(f"Question: {question}")
    parts.append(ANSWER_REQ)
    return "\n\n".join(parts)


# ---------- LLM 客户端（带缓存） ----------
_cache_dir = os.path.join(OUT, 'llm_cache')


def llm_cache_key(model, method, question_id, prompt, gen_cfg):
    h = hashlib.sha256()
    h.update(model.encode())
    h.update(method.encode())
    h.update(question_id.encode())
    h.update(prompt.encode())
    h.update(json.dumps(gen_cfg, sort_keys=True).encode())
    return h.hexdigest()


def llm_cache_get(key):
    p = os.path.join(_cache_dir, key + '.json')
    if os.path.exists(p):
        with open(p, encoding='utf-8') as f:
            return json.load(f)
    return None


def llm_cache_put(key, data):
    os.makedirs(_cache_dir, exist_ok=True)
    with open(os.path.join(_cache_dir, key + '.json'), 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)


_llm_client = None
_usage = {'calls': 0, 'input_tokens': 0, 'output_tokens': 0}


def _load_api_key():
    """读取 api_key：环境变量 > CDPKG .env"""
    k = os.environ.get('CDPKG_LLM_API_KEY', '')
    if k:
        return k
    envp = os.path.join(os.path.dirname(ROOT), 'CDPKG', 'retrieval_qa', '.env')
    if os.path.exists(envp):
        for line in open(envp, encoding='utf-8'):
            if line.startswith('CDPKG_LLM_API_KEY='):
                return line.split('=', 1)[1].strip().strip('"').strip("'")
    return ''


def get_llm():
    global _llm_client
    if _llm_client is None:
        import openai
        api_key = _load_api_key()
        _llm_client = openai.OpenAI(
            api_key=api_key,
            base_url=os.environ.get('CDPKG_LLM_BASE_URL', 'https://api.deepseek.com/v1'),
            timeout=120,
        )
    return _llm_client


def complete(model, method, question_id, prompt, gen_cfg=None, force=False):
    """带缓存的 LLM 调用。返回 (text, input_tokens, output_tokens)"""
    gen_cfg = gen_cfg or {
        'temperature': LLM_TEMPERATURE,
        'top_p': LLM_TOP_P,
        'max_tokens': LLM_MAX_TOKENS,
    }
    key = llm_cache_key(model, method, question_id, prompt, gen_cfg)
    if not force:
        cached = llm_cache_get(key)
        if cached:
            return cached['text'], cached['input_tokens'], cached['output_tokens']
    client = get_llm()
    t0 = time.time()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': prompt},
        ],
        temperature=gen_cfg['temperature'],
        top_p=gen_cfg['top_p'],
        max_tokens=gen_cfg['max_tokens'],
    )
    text = resp.choices[0].message.content
    itok = getattr(resp, 'usage', None).prompt_tokens if getattr(resp, 'usage', None) else 0
    otok = getattr(resp, 'usage', None).completion_tokens if getattr(resp, 'usage', None) else 0
    _usage['calls'] += 1
    _usage['input_tokens'] += itok
    _usage['output_tokens'] += otok
    llm_cache_put(key, {'text': text, 'input_tokens': itok, 'output_tokens': otok,
                        'latency': time.time() - t0})
    return text, itok, otok


def complete_consistency(model, method, question_id, prompt, evidence_texts=None,
                         gen_cfg=None, n=3, force=False):
    """三次独立生成 + 结论一致性投票 + 证据覆盖度选择。

    2Wiki 答案是短答案，核心结论直接用 normalize_answer 后的答案表示：
    - 相同归一化答案视为同一结论，票数 = 该结论出现次数；
    - 票数相同时，选择答案 token 在证据文本中覆盖率更高的候选；
    - 仍相同时选择最早生成的候选。
    返回 (answer, input_tokens, output_tokens, details)。
    """
    gen_cfg = dict(gen_cfg or {
        'temperature': LLM_TEMPERATURE,
        'top_p': LLM_TOP_P,
        'max_tokens': LLM_MAX_TOKENS,
    })
    candidates = []
    itok_total = 0
    otok_total = 0
    errors = []
    for i in range(n):
        text = ''
        for _attempt in range(2):
            try:
                text, itok, otok = complete(
                    model, f'{method}_sample{i}', question_id, prompt,
                    gen_cfg, force=force)
                itok_total += itok
                otok_total += otok
                if text and text.strip():
                    break
            except Exception as e:
                errors.append(str(e)[:160])
                text = ''
        candidates.append(text or '')

    valid = [t for t in candidates if t and t.strip()]
    if not valid:
        details = {
            'candidates': candidates, 'votes': [0] * len(candidates),
            'coverage': [0.0] * len(candidates), 'selected_index': None,
            'generation_method': 'failed',
            'generation_error': errors[-1] if errors else 'empty response',
            'temperature': gen_cfg.get('temperature'),
            'top_p': gen_cfg.get('top_p'),
            'max_tokens': gen_cfg.get('max_tokens'),
        }
        return (candidates[0] if candidates else '', itok_total, otok_total,
                details)

    norms = [normalize_answer(t) for t in candidates]
    vote_counts = Counter(n for n in norms if n)
    ev_tokens = set(normalize_answer(' '.join(evidence_texts or [])).split())
    coverage = []
    for t in candidates:
        toks = normalize_answer(t).split()
        if not toks or not ev_tokens:
            coverage.append(0.0)
        else:
            coverage.append(
                sum(1 for tok in toks if tok in ev_tokens) / len(toks))
    votes = [vote_counts.get(n, 0) for n in norms]

    best = max(range(len(candidates)),
               key=lambda i: (votes[i], coverage[i], -i))
    details = {
        'candidates': candidates,
        'votes': votes,
        'coverage': [round(x, 6) for x in coverage],
        'selected_index': best,
        'generation_method': 'consistency',
        'generation_error': (f'{len(errors)} 次调用失败' if errors else None),
        'temperature': gen_cfg.get('temperature'),
        'top_p': gen_cfg.get('top_p'),
        'max_tokens': gen_cfg.get('max_tokens'),
    }
    return candidates[best], itok_total, otok_total, details


def usage_reset():
    global _usage
    _usage = {'calls': 0, 'input_tokens': 0, 'output_tokens': 0}


def usage_snapshot():
    return dict(_usage)


# ---------- token 估算（evidence 预算用；实际统计用 API usage） ----------
def estimate_tokens(text):
    """粗略估算，用于 evidence 预算控制；最终 token 统计读 API usage 字段。"""
    import tiktoken
    try:
        enc = tiktoken.get_encoding('cl100k_base')
        return len(enc.encode(text or ''))
    except Exception:
        return len((text or '').split())
