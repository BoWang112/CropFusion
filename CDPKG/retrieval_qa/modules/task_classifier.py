# -*- coding: utf-8 -*-
"""
task_classifier.py —— LLM 自动任务分类（六类任务路由）
========================================================
评测时不得读取 qa['task']（测试集真实任务标签）。本模块只根据“问题文本”
判断任务类型：

- 真实 LLM：temperature=0，只输出任务名；失败重试一次后降级为规则。
- 模拟 LLM（无 API Key）：直接使用规则分类，不发出无效请求。
- 分类结果按“问题 + 模型 + 分类器版本”缓存，跨变体/脚本复用。

被 run_eval_resumable.py、3_evaluate.py、run_eval_zeroshot.py 共同使用，
保证三个评测入口的任务路由口径一致。
"""
import os
import sys
import json
import hashlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RESULT_DIR  # noqa: E402

from .prompts import VALID_TASKS, TASK_ALIASES  # noqa: E402


CLASSIFIER_VERSION = 'v2'
# 新版独立缓存文件，避免旧版规则分类结果被复用
DEFAULT_CACHE = os.path.join(RESULT_DIR, 'task_classification_cache_v2.json')

TASK_CLASSIFY_SYSTEM = (
    '你是作物病虫害问答系统的任务路由器。'
    '只根据用户问题判断任务类型，只输出任务名称，不要输出解释、标点或其它内容。'
)

TASK_CLASSIFY_TEMPLATE = """请判断下面这个问题属于哪一类任务，只能从以下六类中选择一个：

- 病虫害识别：根据症状、受害部位或环境条件判断最可能的病虫害
- 症状解析：解释某个已知病虫害的典型症状、受害部位或表现
- 条件判断：判断温度、湿度、降雨、栽培等条件是否有利于病虫害发生
- 防治措施推荐：给出针对某个病虫害的防治措施、用药或田间管理建议
- 相似病虫害鉴别：比较两个或多个相似病虫害的关键差异
- 多条件综合推理：需要综合寄主、症状、部位、环境等多项证据才能得出结论

只输出任务名称（例如：病虫害识别）。

【问题】
{question}
"""


def _parse_task_label(response):
    """从 LLM 输出中解析任务名称（容忍前后说明文字）。"""
    if not response:
        return None
    text = str(response).strip()
    if text.startswith('[LLM 调用失败'):
        return None
    # 1) 完全匹配
    for task in VALID_TASKS:
        if text == task:
            return task
    # 2) 别名 / 规范名匹配（按出现位置取最先出现的任务）
    candidates = []
    for alias, task in TASK_ALIASES.items():
        idx = text.find(alias)
        if idx >= 0:
            candidates.append((idx, task))
    for task in VALID_TASKS:
        idx = text.find(task)
        if idx >= 0:
            candidates.append((idx, task))
    if candidates:
        candidates.sort(key=lambda x: (x[0], -len(x[1])))
        return candidates[0][1]
    return None


def _rule_classify_task(question):
    """无 LLM 时的规则分类（仅依据问题文本）。"""
    q = question or ''
    if any(k in q for k in ('区别', '区分', '鉴别', '不同点', '差异')):
        return '相似病虫害鉴别'
    if any(k in q for k in ('怎么治', '如何防治', '防治方法', '防治措施', '用什么药',
                            '打什么药', '喷什么药', '药剂', '防控', '防除', '治疗')):
        return '防治措施推荐'
    # 多条件综合：同时出现症状/部位与气候/环境/栽培条件，且要求综合判断
    has_symptom = any(k in q for k in ('病斑', '症状', '叶片', '茎', '果实', '根',
                                       '萎蔫', '腐烂', '变色', '枯'))
    has_env = any(k in q for k in ('温度', '湿度', '降雨', '阴雨', '高温', '低温',
                                   '气候', '环境', '土壤', '栽培', '连作'))
    if has_symptom and has_env and any(k in q for k in ('最可能', '综合', '可能是什么',
                                                        '判断', '诊断')):
        return '多条件综合推理'
    if any(k in q for k in ('条件', '温度', '湿度', '降雨', '阴雨', '气候', '环境',
                            '什么情况下', '是否有利', '会不会发生', '适宜')):
        return '条件判断'
    if any(k in q for k in ('什么病', '什么虫', '哪种病', '哪种虫', '识别',
                            '判断是什么', '是什么病', '是什么虫')):
        return '病虫害识别'
    if any(k in q for k in ('症状', '表现', '怎么回事', '什么原因', '说明什么')):
        return '症状解析'
    return '病虫害识别'


def _load_task_cache(path):
    try:
        with open(path, encoding='utf-8') as f:
            cache = json.load(f)
        return cache if isinstance(cache, dict) else {}
    except Exception:
        return {}


def _save_task_cache(path, cache):
    try:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(cache, f, ensure_ascii=False, indent=0)
    except Exception:
        pass


def classify_task(question, llm, cache_path=None):
    """自动判断问题所属的六类任务之一。

    返回 (task, method)，method 为 'llm' / 'rule'。
    只依据问题文本，不读取任何测试集任务标签。
    """
    cache_path = cache_path or DEFAULT_CACHE
    model = 'rule' if getattr(llm, 'mock_mode', True) else str(llm.cfg.get('model', ''))
    raw = '||'.join([CLASSIFIER_VERSION, model, question or ''])
    key = hashlib.sha1(raw.encode('utf-8')).hexdigest()
    cache = _load_task_cache(cache_path)
    cached = cache.get(key)
    if isinstance(cached, dict) and cached.get('task') in VALID_TASKS:
        return cached['task'], cached.get('method', 'cache')

    task = None
    method = 'rule'
    if not getattr(llm, 'mock_mode', True):
        prompt = TASK_CLASSIFY_TEMPLATE.format(question=question)
        for _ in range(2):
            response = llm.complete(prompt, system=TASK_CLASSIFY_SYSTEM,
                                    temperature=0.0, max_tokens=32)
            task = _parse_task_label(response)
            if task:
                method = 'llm'
                break
    if not task:
        task = _rule_classify_task(question)

    cache[key] = {
        'task': task, 'method': method, 'model': model,
        'version': CLASSIFIER_VERSION,
    }
    _save_task_cache(cache_path, cache)
    return task, method
