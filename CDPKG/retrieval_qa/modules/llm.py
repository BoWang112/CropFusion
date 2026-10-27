# -*- coding: utf-8 -*-
"""
llm.py —— LLM 生成客户端
=========================
- 真实模式：调用 DeepSeek（OpenAI 兼容）API
- 模拟模式：未配置 API Key 时，用证据驱动 + 模板规则生成可用的占位回答，
  便于在无 Key 环境下跑通检索→提示→生成的整条链路。

对应论文 3.4 Evidence-driven optimized prompting 的生成环节。
"""
import os
import re
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import LLM  # noqa: E402


# ============================================================
# 自一致性聚合（论文 3.4（4））所需的文本工具
# ============================================================
_FAILURE_PREFIX = '[LLM 调用失败'

# 图谱证据行：[G1]（锚点：X）head -rel-> tail（来源：...；证据ID：...）
_GRAPH_ANCHOR_RE = re.compile(
    r'（锚点：([^）]+)）\s*(.+?)\s*-\s*([^-=>\n]+?)\s*->\s*(.+?)(?:（来源|$)',
    re.MULTILINE,
)
# 简化提示中的图谱证据行：- head -rel-> tail
_GRAPH_PLAIN_RE = re.compile(
    r'^\s*-\s*(.+?)\s*-\s*([^-=>\n]+?)\s*->\s*(.+?)\s*$',
    re.MULTILINE,
)
# 文本证据元信息：实体：X；症状/作物/病虫害模板前缀
_TEXT_ENTITY_RE = re.compile(r'实体[:：]\s*([^；;）)\n]+)')
_TEXT_PEST_RE = re.compile(r'病虫害[:：]\s*([^，,。；;\n]+)')
_TEXT_CROP_RE = re.compile(r'作物[:：]\s*([^，,。；;\n]+)')

# 不作为证据原子的占位词
_ATOM_STOPWORDS = {
    '无图谱证据', '无文本证据', '未知', '通用', 'nan', 'None', 'null',
    '图谱关系', '检索到的文本',
}


def _section(text, start, end):
    """截取 start 与 end 之间的提示词片段；任一端不存在时返回空串。

    返回空串（而不是截到末尾）便于调用方按多个可能的结束标记依次尝试，
    避免把后续的任务要求/输出规范误当成证据内容。
    """
    si = text.find(start)
    if si < 0:
        return ''
    si += len(start)
    ei = text.find(end, si)
    if ei < 0:
        return ''
    return text[si:ei].strip()


def _normalize_answer(text):
    """去除引用编号和空白，用于候选答案的相似度比较。"""
    if not text:
        return ''
    t = re.sub(r'\[[EG]\d+\]', '', str(text))
    return re.sub(r'\s+', '', t)


def _answer_similarity(a, b):
    """候选答案两两相似度（0~1）。

    中文生成结果常用词序和标点略有差异，因此先去掉空白与证据编号；
    完全一致/包含关系直接给高分，其余用字符级 SequenceMatcher。
    """
    from difflib import SequenceMatcher
    na, nb = _normalize_answer(a), _normalize_answer(b)
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0
    if na in nb or nb in na:
        return min(len(na), len(nb)) / max(len(na), len(nb))
    return SequenceMatcher(None, na, nb).ratio()


# 从回答中抽取“核心结论”的模式（按优先级）
_CONCLUSION_PATTERNS = (
    re.compile(r'最可能是\s*([^。！？!?\n，,；;]+)'),
    re.compile(r'该(?:病虫害|病害|虫害)\s*(?:为|是)\s*([^。！？!?\n，,；;]+)'),
    re.compile(r'结论[:：]\s*([^。！？!?\n]+)'),
    re.compile(r'判断为\s*([^。！？!?\n，,；;]+)'),
    re.compile(r'答案是\s*([^。！？!?\n，,；;]+)'),
)


def _normalize_conclusion(text):
    """规范化核心结论：去掉空白、标点与引用编号，便于一致性比较。"""
    return re.sub(r"[\s，,。；;：:、！？!?\[\]【】（）()\"“”'‘’]", '', str(text or ''))


def _extract_core_conclusion(answer):
    """抽取答案的核心结论（用于一致性投票）。

    优先读取【诊断结论】小节，其次匹配“最可能是 X / 该病虫害为 X /
    结论：X / 判断为 X / 答案是 X”，最后回退到第一句。
    """
    text = re.sub(r'\[[EG]\d+\]', '', str(answer or ''))
    section = re.search(r'【诊断结论】\s*([^\n【]+)', text)
    if section:
        text = section.group(1)
    for pattern in _CONCLUSION_PATTERNS:
        match = pattern.search(text)
        if match:
            core = _normalize_conclusion(match.group(1))
            if core:
                return core
    for sentence in re.split(r'[。！？!?\n]', text):
        core = _normalize_conclusion(sentence)
        if len(core) >= 4:
            return core
    return _normalize_conclusion(text)


def _conclusion_similarity(a, b):
    """核心结论相似度（0~1）。

    短结论通常是病虫害名/判断语（如“番茄早疫病”“会加重”），必须完全一致
    才算同一结论，避免“番茄早疫病”和“番茄晚疫病”这类仅一字之差的名字
    被误合并；较长结论才使用字符相似度。
    """
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    if max(len(a), len(b)) <= 12:
        return 0.0
    if a in b or b in a:
        return min(len(a), len(b)) / max(len(a), len(b))
    return _answer_similarity(a, b)


def _add_atoms(atoms, values):
    for v in values:
        if not v:
            continue
        v = str(v).strip().strip('（）()[]【】 ')
        if len(v) < 2 or v in _ATOM_STOPWORDS:
            continue
        atoms.append(v)


def _extract_evidence_atoms(prompt):
    """从提示词中抽取“证据原子”（图谱头/尾/锚点、文本实体/作物/病虫害）。

    这些原子用于衡量候选答案实际覆盖了多少检索证据，作为一致性投票
    平票时的选择依据。只读取提示词中已有的证据，不调用额外模型。
    """
    atoms = []
    if not prompt:
        return atoms

    # 标准提示：文本证据 + 图谱证据
    text_sec = _section(prompt, '【文本证据】', '【图谱证据】')
    graph_sec = _section(prompt, '【图谱证据】', '【同类少样本示例】')
    if not graph_sec:
        graph_sec = _section(prompt, '【图谱证据】', '【当前任务的组织要求】')
    if not graph_sec:
        graph_sec = _section(prompt, '【图谱证据】', '【输出规范】')

    # 简化提示（ours-retrieval / -no-evidence-prompt）
    if not text_sec:
        text_sec = _section(prompt, '检索到的文本：', '图谱关系：')
    if not text_sec:
        text_sec = _section(prompt, '检索到的文本：', '用户问题：')
    if not graph_sec:
        graph_sec = _section(prompt, '图谱关系：', '用户问题：')
    if not graph_sec:
        graph_sec = _section(prompt, '图谱关系：', '请依据')

    # ---- 文本证据原子 ----
    _add_atoms(atoms, _TEXT_ENTITY_RE.findall(text_sec))
    _add_atoms(atoms, _TEXT_PEST_RE.findall(text_sec))
    _add_atoms(atoms, _TEXT_CROP_RE.findall(text_sec))

    # ---- 图谱证据原子 ----
    for anchor, head, _rel, tail in _GRAPH_ANCHOR_RE.findall(graph_sec):
        _add_atoms(atoms, [anchor, head, tail])
    # 简化格式：- head -rel-> tail（标准格式行以 [G1] 开头，不会误匹配）
    for head, _rel, tail in _GRAPH_PLAIN_RE.findall(graph_sec):
        _add_atoms(atoms, [head, tail])

    # 去重并保持顺序
    return list(dict.fromkeys(atoms))


def _evidence_coverage(answer, atoms):
    """答案对证据原子的覆盖度：命中的证据原子数 / 证据原子总数。"""
    if not atoms:
        return 0.0
    na = _normalize_answer(answer)
    if not na:
        return 0.0
    hit = 0
    for a in atoms:
        ta = _normalize_answer(a)
        if ta and ta in na:
            hit += 1
    return hit / len(atoms)


class LLMClient:
    def __init__(self, cfg=None):
        self.cfg = cfg or LLM
        self._client = None
        self.mock_mode = not self.cfg.get('api_key')

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(
                api_key=self.cfg['api_key'],
                base_url=self.cfg['base_url'],
                timeout=self.cfg.get('timeout', 120),
            )
        return self._client

    def complete(self, prompt, system=None, temperature=None, top_p=None,
                 max_tokens=None):
        """单次生成；无 Key 时走模拟。

        统一采样参数：temperature 默认 0.2，top_p 默认 0.9，
        max_tokens 默认 2048（可由 config 覆盖）。
        """
        if self.mock_mode:
            return self._mock_complete(prompt, system)
        # 不猜测 API 模型 ID / base_url：未配置时返回明确错误
        if not self.cfg.get('model'):
            return ('[LLM 调用失败：未配置模型 ID，请通过环境变量或配置传入'
                    '实际 API 模型标识]')
        if not self.cfg.get('base_url'):
            return '[LLM 调用失败：未配置 base_url]'
        msgs = []
        if system:
            msgs.append({'role': 'system', 'content': system})
        msgs.append({'role': 'user', 'content': prompt})
        try:
            resp = self._get_client().chat.completions.create(
                model=self.cfg['model'],
                messages=msgs,
                temperature=(temperature if temperature is not None
                             else self.cfg.get('temperature', 0.2)),
                top_p=(top_p if top_p is not None
                       else self.cfg.get('top_p', 0.9)),
                max_tokens=max_tokens or self.cfg.get('max_tokens', 2048),
            )
            return resp.choices[0].message.content
        except Exception as e:
            return f"[LLM 调用失败：{e}]"

    def complete_with_consistency(self, prompt, system=None, n=None):
        """兼容接口：返回核心结论一致性投票 + 证据覆盖度选择后的答案。"""
        details = self.complete_with_consistency_details(prompt, system, n)
        return details.get('answer', '')

    def complete_with_consistency_details(self, prompt, system=None, n=None):
        """
        三次独立生成 → 核心结论一致性投票 → 证据覆盖度选择。

        流程：
        1) 以统一采样参数（temperature=0.2、top_p=0.9、max_tokens=2048）
           独立生成 n 个候选（默认 3）；单次调用失败时重试一次，
           仍失败则保留失败标记，不参与投票；
        2) 一致性投票：抽取每个候选的“核心结论”（诊断结论/首句），
           核心结论相似度达到阈值的候选归为同一结论簇，票数 = 簇大小；
        3) 证据覆盖度选择：先选票数最多的结论簇（同票比较簇内平均覆盖度），
           再在胜出簇内选择覆盖更多检索证据原子的候选；仍相同时用
           完整答案平均相似度决胜。

        返回固定字段的 dict，供评测脚本保存完整生成审计信息：
          answer / candidates / valid_indices / core_conclusions /
          votes / coverage / agreement / selected_index / method /
          temperature / top_p / max_tokens / error

        模拟模式（无 API Key）下仍生成 n 份确定性占位回答，保持字段完整。
        """
        try:
            n = int(n if n is not None else self.cfg.get('num_samples', 3))
        except (TypeError, ValueError):
            n = 3
        n = max(1, n)
        temperature = float(self.cfg.get('temperature', 0.2))
        top_p = float(self.cfg.get('top_p', 0.9))
        max_tokens = int(self.cfg.get('max_tokens', 2048))

        candidates = []
        for _ in range(n):
            candidate = ''
            for _attempt in range(2):  # 单次失败重试一次
                candidate = self.complete(
                    prompt, system, temperature=temperature,
                    top_p=top_p, max_tokens=max_tokens)
                if candidate and not str(candidate).startswith(_FAILURE_PREFIX):
                    break
            candidates.append(candidate or '')

        valid_indices = [
            i for i, answer in enumerate(candidates)
            if answer and not str(answer).startswith(_FAILURE_PREFIX)
        ]
        atoms = _extract_evidence_atoms(prompt)
        coverage = [
            _evidence_coverage(answer, atoms) if i in valid_indices else 0.0
            for i, answer in enumerate(candidates)
        ]
        core_conclusions = [
            _extract_core_conclusion(answer) if i in valid_indices else ''
            for i, answer in enumerate(candidates)
        ]
        votes = [0] * n
        agreement = [0.0] * n
        method = 'mock' if self.mock_mode else 'consistency'

        if not valid_indices:
            error = candidates[0] if candidates else 'LLM 未返回任何候选'
            return {
                'answer': candidates[0] if candidates else '',
                'candidates': candidates,
                'valid_indices': [],
                'core_conclusions': core_conclusions,
                'votes': votes,
                'coverage': coverage,
                'agreement': agreement,
                'selected_index': None,
                'method': 'failed',
                'provider': self.cfg.get('provider'),
                'model': self.cfg.get('model'),
                'temperature': temperature,
                'top_p': top_p,
                'max_tokens': max_tokens,
                'error': error,
            }

        # 1) 核心结论一致性聚类：相似度达到阈值的候选归为同一簇
        threshold = float(self.cfg.get('consistency_threshold', 0.6))
        clusters = []
        for i in valid_indices:
            for cluster in clusters:
                if _conclusion_similarity(
                        core_conclusions[i], core_conclusions[cluster[0]]) >= threshold:
                    cluster.append(i)
                    break
            else:
                clusters.append([i])
        for cluster in clusters:
            for i in cluster:
                votes[i] = len(cluster)

        # 2) 完整答案平均相似度（仅作为最终排序的次级依据）
        for i in valid_indices:
            sims = [_answer_similarity(candidates[i], candidates[j])
                    for j in valid_indices if j != i]
            agreement[i] = sum(sims) / len(sims) if sims else 0.0

        # 3) 胜出簇：票数最多；同票时比较簇内平均证据覆盖度
        best_cluster = max(
            clusters,
            key=lambda cluster: (
                len(cluster),
                sum(coverage[i] for i in cluster) / len(cluster),
            ),
        )
        # 4) 簇内选择：证据覆盖度最高；仍相同则平均相似度高者优先
        selected = max(
            best_cluster,
            key=lambda i: (coverage[i], agreement[i], -i),
        )

        errors = [answer for answer in candidates
                  if not answer or str(answer).startswith(_FAILURE_PREFIX)]
        return {
            'answer': candidates[selected],
            'candidates': candidates,
            'valid_indices': valid_indices,
            'core_conclusions': core_conclusions,
            'votes': votes,
            'coverage': coverage,
            'agreement': agreement,
            'selected_index': selected,
            'method': method,
            'provider': self.cfg.get('provider'),
            'model': self.cfg.get('model'),
            'temperature': temperature,
            'top_p': top_p,
            'max_tokens': max_tokens,
            'error': (f'{len(errors)}/{n} 次生成失败' if errors else None),
        }

    # ---------- 模拟模式 ----------
    def _mock_complete(self, prompt, system=None):
        """无 API Key 时基于提示词内容提取证据，生成结构化占位回答。
        用于链路验证；提示词中已含证据，规则化拼接即得可用回答。"""
        # 从提示词中提取证据行
        text_ev = self._extract_section(prompt, '【文本证据】', '【图谱证据】')
        graph_ev = self._extract_section(prompt, '【图谱证据】', '【诊断要求】')
        question = self._extract_question(prompt)
        return self._mock_answer(question, text_ev, graph_ev)

    @staticmethod
    def _extract_section(text, start, end):
        si = text.find(start)
        ei = text.find(end)
        if si < 0:
            return ''
        si += len(start)
        if ei < 0:
            return text[si:].strip()
        return text[si:ei].strip()

    @staticmethod
    def _extract_question(prompt):
        si = prompt.find('【用户问题】')
        if si < 0:
            return ''
        si += len('【用户问题】')
        ei = prompt.find('【学习目标】')
        return prompt[si:ei].strip() if ei > si else prompt[si:].strip()

    def _mock_answer(self, question, text_ev, graph_ev):
        lines = []
        lines.append('【诊断结论】')
        # 尝试从图谱证据中找病虫害实体
        pests = []
        for ln in graph_ev.split('\n'):
            if '病害' in ln or '虫害' in ln or '为害' in ln:
                head = ln.split(' -')[0].split('[')[-1] if ' -' in ln else ''
                if head:
                    pests.append(head)
        if pests:
            lines.append(f"根据证据分析，该病虫害可能与「{pests[0]}」相关。")
        else:
            lines.append("检索证据有限，建议结合田间实况进一步确认。")
        lines.append('')
        lines.append('【检索证据】')
        if text_ev:
            lines.append(text_ev[:400])
        else:
            lines.append('（无文本证据）')
        if graph_ev:
            lines.append('图谱关系：' + '；'.join(
                ln for ln in graph_ev.split('\n') if ln.strip())[:400])
        lines.append('')
        lines.append('【注意】当前为模拟回答（未配置 LLM API）。配置 config.LLM.api_key 后即可获得完整生成。')
        return '\n'.join(lines)


class LLMClientWithRetry(LLMClient):
    """带重试的客户端（网络波动容忍）"""

    def complete(self, prompt, system=None, temperature=None, top_p=None,
                 max_tokens=None, retries=3):
        last = ''
        for i in range(retries):
            try:
                last = super().complete(
                    prompt, system=system, temperature=temperature,
                    top_p=top_p, max_tokens=max_tokens)
            except Exception as e:
                last = f"[LLM 调用失败：{e}]"
            if last and not str(last).startswith(_FAILURE_PREFIX):
                return last
        return last or '[LLM 调用失败：未知错误]'
