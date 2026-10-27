# -*- coding: utf-8 -*-
"""
CDPKG-LLM 检索与问答系统 配置文件
=================================
集中管理数据路径、检索参数与 LLM API 配置。
"""
import os

# ---------- 路径 ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CDPKG_DIR = os.path.dirname(BASE_DIR)
REPO_ROOT = os.path.dirname(CDPKG_DIR)
RETRIEVAL_QA_DIR = BASE_DIR

NODES_CSV = os.path.join(CDPKG_DIR, 'results_merged', 'kg_27x20', 'nodes.csv')
RELS_CSV = os.path.join(CDPKG_DIR, 'results_merged', 'kg_27x20', 'relationships.csv')
EXTRACT_JSON = os.path.join(CDPKG_DIR, 'results_merged', '抽取结果.json')

# 模型缓存目录
MODELS_DIR = os.path.join(CDPKG_DIR, 'models')
BGE_MODEL = os.path.join(MODELS_DIR, 'bge-large-zh-v1.5')   # 嵌入模型
# BGE-large-zh-v1.5 查询与文档使用不同前缀，但该模型归一化后无前缀也能工作；此处保留接口
BGE_QUERY_PREFIX = ''
BGE_PASSAGE_PREFIX = ''

# 数据输出目录（全部基于项目绝对路径，避免不同工作目录读取失败）
DATA_DIR = os.path.join(BASE_DIR, 'data')
INDEX_DIR = os.path.join(DATA_DIR, 'index')
EVAL_FILE = os.path.join(DATA_DIR, 'qa_test.json')          # 评测 QA 集
FEWSHOT_FILE = os.path.join(DATA_DIR, 'fewshot_pool.json')  # 动态 few-shot 池
RESULT_DIR = os.path.join(DATA_DIR, 'results')

# ---------- 检索参数（与论文 4.2.1 一致） ----------
RETRIEVAL = {
    'vector_top_k': 20,          # 向量检索候选数
    'keyword_top_k': 20,         # BM25 候选数
    'semantic_top_k': 20,        # 语义驱动图检索 锚点候选数
    'entity_top_k': 20,          # 实体驱动图检索 锚点候选数
    'graph_hop': 2,              # 图谱邻域扩展跳数
    'rrf_k': 60,                 # 文本层 RRF 平滑常数
    'text_top_k': 10,            # 文本层融合后保留数
    'graph_top_k': 5,            # 图层重排序后保留子图数
    'path_max_len': 3,           # 锚点间路径最大长度
    'graph_rerank_lambda': 0.4,  # 图重排序：λ*结构紧凑性 + (1-λ)*语义相关性
    'fewshot_top_k': 3,          # 动态 few-shot 示例数（生成阶段）
}

# ---------- LLM API 配置 ----------
# api_key 读取顺序：环境变量 > 本地 .env 文件。
# 真实 Key 不写入代码文件（避免凭证落盘到源码工件），
# 请把 Key 放入 retrieval_qa/.env 或设置环境变量。
def _load_key(env_var):
    """从环境变量或本地 .env 读取指定凭证（代码中不落盘任何真实 Key）"""
    k = os.environ.get(env_var, '')
    if k:
        return k
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
    if os.path.exists(env_path):
        try:
            for line in open(env_path, encoding='utf-8'):
                line = line.strip()
                if line.startswith(env_var + '='):
                    return line.split('=', 1)[1].strip().strip('"').strip("'")
        except Exception:
            pass
    return ''


def _load_api_key():
    return _load_key('CDPKG_LLM_API_KEY')

LLM = {
    'provider': os.environ.get('CDPKG_LLM_PROVIDER', 'deepseek'),
    'base_url': os.environ.get('CDPKG_LLM_BASE_URL', 'https://api.deepseek.com/v1'),
    'api_key': _load_api_key(),
    'model': os.environ.get('CDPKG_LLM_MODEL', 'deepseek-chat'),
    'temperature': 0.2,
    'top_p': 0.9,
    'max_tokens': 2048,
    'timeout': 120,
    # 自一致性聚合时生成候选数
    'num_samples': 3,
}

# 神农大模型（中国农大）——通过 CDPKG_LLM_PROVIDER=shennong 环境变量切换。
# 与 DeepSeek 走同一 OpenAI 兼容客户端，便于做"不同基座 LLM 的泛化性"对比。
SHENNONG_LLM = {
    'provider': 'shennong',
    'base_url': os.environ.get('CDPKG_SHENNONG_BASE_URL', 'https://api.agent-tech.cc/api/v1'),
    'api_key': _load_key('CDPKG_SHENNONG_API_KEY'),
    'model': os.environ.get('CDPKG_SHENNONG_MODEL', 'sn'),
    'temperature': 0.2,
    'top_p': 0.9,
    'max_tokens': 2048,
    'timeout': 120,
    'num_samples': 3,
}


def _provider_config(provider, base_url_env, base_url_default,
                     api_key_env, model_env, default_model=''):
    """构造 OpenAI 兼容 provider 配置。

    实际 API 模型 ID 必须通过环境变量/外部配置传入；这里不硬编码
    未经验证的模型标识。default_model 仅用于仓库既有、已验证的默认模型。
    """
    return {
        'provider': provider,
        'base_url': os.environ.get(base_url_env, base_url_default),
        'api_key': _load_key(api_key_env),
        'model': os.environ.get(model_env, default_model),
        'temperature': 0.2,
        'top_p': 0.9,
        'max_tokens': 2048,
        'timeout': 120,
        'num_samples': 3,
    }


# 新版 provider：模型 ID 通过环境变量传入，避免猜测/硬编码不存在的标识。
DEEPSEEK_V4_FLASH = _provider_config(
    'deepseek-v4-flash',
    'CDPKG_DEEPSEEK_V4_BASE_URL', 'https://api.deepseek.com/v1',
    'CDPKG_DEEPSEEK_V4_API_KEY', 'CDPKG_DEEPSEEK_V4_MODEL', '')
SHENNONG_V4 = _provider_config(
    'shennong-v4',
    'CDPKG_SHENNONG_V4_BASE_URL', 'https://api.agent-tech.cc/api/v1',
    'CDPKG_SHENNONG_V4_API_KEY', 'CDPKG_SHENNONG_V4_MODEL', '')
QWEN3_7_PLUS = _provider_config(
    'qwen3.7-plus',
    'CDPKG_QWEN_BASE_URL', '',
    'CDPKG_QWEN_API_KEY', 'CDPKG_QWEN_MODEL', '')

LLM_PROVIDERS = {
    'deepseek': LLM,
    'deepseek-v4-flash': DEEPSEEK_V4_FLASH,
    'deepseek-v4-flash-0731': DEEPSEEK_V4_FLASH,
    'shennong': SHENNONG_LLM,
    'shennong-v4': SHENNONG_V4,
    'shennong-v4.0': SHENNONG_V4,
    'qwen3.7-plus': QWEN3_7_PLUS,
}


def get_llm_config(provider=None):
    """返回指定 provider 的 LLM 配置副本。

    provider 为空时读取 CDPKG_LLM_PROVIDER（默认 deepseek）。
    若配置缺少 model/base_url，LLMClient 会返回明确错误而不会猜测模型 ID。
    """
    name = provider or os.environ.get('CDPKG_LLM_PROVIDER', 'deepseek')
    if name not in LLM_PROVIDERS:
        raise ValueError(
            f'未知 LLM provider：{name!r}。'
            f'可选：{", ".join(sorted(LLM_PROVIDERS))}')
    return dict(LLM_PROVIDERS[name])

# ---------- GraphRAG 对比实验配置 ----------
GRAPHRAG_DATA = os.path.join(DATA_DIR, 'graphrag')
GLOBAL_QA_FILE = os.path.join(DATA_DIR, 'qa_global.json')
GRAPHRAG = {
    'projection_ctrl_ctrl_weight': 0.05,   # 防治→防治噪声链降权（69% 是噪声）
    'projection_pest_ctrl_weight': 0.5,    # 病虫害→防治 保留部分信号
    'community_min_size': 10,              # 最小社区规模
    'community_method': 'leiden',          # leiden / louvain
    'summary_top_k': 8,                    # Global Search 选中社区数
    'summary_max_lines': 400,
    'global_answer_max_len': 200,          # Global 回答最大长度
}

# ---------- 实验变体清单（统一 run_all_variants / make_final_table） ----------
EVALUATION_PIPELINE_VERSION = 'v2'
ABLATION_VARIANTS = [
    '-no-text-rrf', '-no-graph-rerank', '-no-semantic-graph', '-no-entity-graph',
    '-no-evidence-prompt', '-no-fewshot', '-no-diag-constraints',
]
QA_BASELINE_VARIANTS = [
    'naive-text', 'hybrid-text', 'graph-only',
    'grag-local', 'lightrag', 'g-retriever', 'pathrag', 'sg-rag',
]
FORMAL_VARIANTS = ['full'] + ABLATION_VARIANTS + QA_BASELINE_VARIANTS
EVAL_VARIANTS = ['full', 'ours-retrieval'] + ABLATION_VARIANTS + QA_BASELINE_VARIANTS
