# -*- coding: utf-8 -*-
"""
query_parser.py —— 大语言模型问题解析
======================================
对应论文 3.3.4 的查询语义归纳与实体抽取：

(1) 语义驱动图检索：LLM 从查询中归纳 作物/症状/受害部位/环境条件/意图 五要素，
    并输出判别性症状词（替代硬编码症状词表）
(2) 实体驱动图检索：LLM 按领域本体约束抽取显式实体提及（作物/病害/虫害/病原体/害虫），
    与字典匹配融合（LLM 结果优先，字典兜底）

设计：
- 真实模式：调用 DeepSeek API（结构化 JSON 输出）
- 模拟模式：未配置 Key 时降级为字典匹配 + 规则提取（保持链路可用）
"""
import os
import sys
import json
import re

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import LLM  # noqa: E402


# ---------- LLM 提示模板 ----------
QUERY_PARSE_SYSTEM = """你是作物病虫害领域的资深专家。请解析用户的问题，提取诊断所需的关键信息。
只依据用户问题中明确出现的内容进行提取，不得推测、不得补充问题中未提及的信息。
输出严格为 JSON，不要输出任何其他内容。"""

QUERY_PARSE_TEMPLATE = """请解析以下作物病虫害问题，提取结构化信息。

【问题】
{question}

【输出 JSON 格式】
{{
  "crop": "寄主作物（用标准名，如 马铃薯/玉米/水稻，不用 土豆/苞谷/稻子），未提及则为空字符串",
  "symptom": "症状或为害表现描述，未提及则为空字符串",
  "site": "受害部位（如叶片/茎秆/穗部/根部），未提及则为空字符串",
  "environment": "环境或发生条件（如高湿/阴雨/高温），未提及则为空字符串",
  "intent": "用户意图，取以下之一：识别/防治/条件/症状解析/鉴别/其他",
  "entities": [
    {{"name": "问题中明确出现的实体名（用领域标准名，如 稻瘟病/小麦条锈病/水稻，不用口语俗名）", "type": "实体类型，限 作物/病害/虫害/病原体/害虫/症状/发病部位/防治措施 之一"}}
  ],
  "symptom_terms": ["判别性症状词列表：病斑形态、颜色、质地等强判别特征词，最多5个；无则空数组"]
}}

【示例】
问题：水稻叶片出现梭形病斑，中央灰白，边缘褐色，这是什么病？
输出：{{"crop": "水稻", "symptom": "叶片出现梭形病斑，中央灰白，边缘褐色", "site": "叶片", "environment": "", "intent": "识别", "entities": [{{"name": "水稻", "type": "作物"}}, {{"name": "叶片", "type": "发病部位"}}], "symptom_terms": ["梭形病斑", "灰白", "边缘褐色"]}}

问题：小麦条锈病用什么药防治？
输出：{{"crop": "小麦", "symptom": "", "site": "", "environment": "", "intent": "防治", "entities": [{{"name": "小麦条锈病", "type": "病害"}}], "symptom_terms": []}}"""


def _parse_json_response(text):
    """从 LLM 输出中解析 JSON（容忍前后噪音/代码块标记）"""
    text = text.strip()
    # 去掉 markdown 代码块
    text = re.sub(r'^```(?:json)?\s*|\s*```$', '', text)
    # 提取第一个 { 到最后一个 }
    si, ei = text.find('{'), text.rfind('}')
    if si >= 0 and ei > si:
        text = text[si:ei + 1]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


class QueryParser:
    """LLM 问题解析器（真实 API / 模拟降级）"""

    def __init__(self, llm_client=None, kg=None):
        self.llm = llm_client
        self.kg = kg
        # 模拟模式：无 LLM 客户端或无 API Key 时用字典匹配
        self.mock_mode = self.llm is None or getattr(self.llm, 'mock_mode', True)

    # ---------- 主入口 ----------
    def parse(self, question):
        """完整解析：返回 {crop, symptom, site, environment, intent, entities, symptom_terms}"""
        if self.mock_mode:
            return self._mock_parse(question)
        parsed = self._llm_parse(question)
        if parsed is None:
            return self._mock_parse(question)  # LLM 解析失败时降级
        # 融合字典匹配：LLM 结果优先，字典兜底补充实体
        self._fuse_with_dict(parsed, question)
        return parsed

    # ---------- LLM 解析（真实模式） ----------
    def _llm_parse(self, question):
        prompt = QUERY_PARSE_TEMPLATE.format(question=question)
        try:
            resp = self.llm.complete(prompt, system=QUERY_PARSE_SYSTEM, temperature=0.0)
        except Exception:
            return None
        parsed = _parse_json_response(resp)
        if not parsed:
            return None
        # 规范化字段
        for field in ('crop', 'symptom', 'site', 'environment', 'intent'):
            if field not in parsed or not isinstance(parsed[field], str):
                parsed[field] = ''
        if 'entities' not in parsed or not isinstance(parsed['entities'], list):
            parsed['entities'] = []
        if 'symptom_terms' not in parsed or not isinstance(parsed['symptom_terms'], list):
            parsed['symptom_terms'] = []
        return parsed

    # ---------- 字典融合 ----------
    def _fuse_with_dict(self, parsed, question):
        """LLM 抽取的实体与字典匹配结果融合，补充漏掉的规范实体"""
        if self.kg is None:
            return
        # 容错：不同 LLM 输出格式不同，entities 元素可能是字符串（非 dict），
        # 例如神农大模型解析返回的实体数组元素不带 name/type 字段。
        # 过滤非法元素，避免 AttributeError 中断评测。
        valid = []
        for e in parsed['entities']:
            if isinstance(e, dict):
                valid.append(e)
        parsed['entities'] = valid
        dict_ents = self._dict_extract(question)
        llm_names = {e.get('name', '') for e in parsed['entities']}
        for e in dict_ents:
            if e['name'] not in llm_names:
                parsed['entities'].append(e)

    def _dict_extract(self, query):
        """字典精确匹配（模拟模式也用）"""
        from .retriever import normalize_text
        qn = normalize_text(query)
        found = []
        for name, nid in self.kg.norm_names:
            if name in qn:
                ent = self.kg.node_entity(nid)
                if ent in ('作物', '病害', '虫害', '病原体', '害虫'):
                    orig = self.kg.norm_to_name.get(name, name)
                    found.append({'name': orig, 'id': nid, 'entity': ent})
        return found

    # ---------- 模拟模式（字典 + 规则） ----------
    def _mock_parse(self, question):
        from .retriever import normalize_text
        # 字典实体抽取
        entities = self._dict_extract(question)
        # 简单规则：作物/症状/部位推断
        crop = ''
        site = ''
        for e in entities:
            if e.get('entity') == '作物':
                crop = e['name']
            elif e.get('entity') == '发病部位':
                site = e['name']
        # 意图规则
        intent = '识别'
        if any(k in question for k in ['怎么治', '如何防治', '用药', '防治方法', '打啥药', '怎么防']):
            intent = '防治'
        elif any(k in question for k in ['条件', '发生', '环境', '影响', '加重']):
            intent = '条件'
        elif any(k in question for k in ['区分', '区别', '鉴别']):
            intent = '鉴别'
        elif any(k in question for k in ['怎么回事', '说明什么', '什么原因', '代表啥']):
            intent = '症状解析'
        # 症状词（硬编码表，模拟模式兜底）
        SYMPTOM_WORDS = ['病斑', '坏死', '霉层', '褪绿', '变褐', '斑点', '枯死', '腐烂',
                         '水渍', '凹陷', '溃疡', '穿孔', '畸形', '卷曲', '萎蔫', '枯萎',
                         '圆形', '椭圆形', '梭形', '轮纹', '晕圈', '边缘', '中央',
                         '褐色', '黑色', '白色', '黄色', '灰色', '斑', '粉状', '霉', '小点']
        symptom_terms = [w for w in SYMPTOM_WORDS if w in question]
        symptom = ''
        return {
            'crop': crop, 'symptom': symptom, 'site': site,
            'environment': '', 'intent': intent,
            'entities': entities, 'symptom_terms': symptom_terms[:5],
        }
