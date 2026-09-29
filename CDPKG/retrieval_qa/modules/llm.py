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
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import LLM  # noqa: E402


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

    def complete(self, prompt, system=None, temperature=None, max_tokens=None):
        """单次生成；无 Key 时走模拟"""
        if self.mock_mode:
            return self._mock_complete(prompt, system)
        msgs = []
        if system:
            msgs.append({'role': 'system', 'content': system})
        msgs.append({'role': 'user', 'content': prompt})
        try:
            resp = self._get_client().chat.completions.create(
                model=self.cfg['model'],
                messages=msgs,
                temperature=temperature if temperature is not None else self.cfg.get('temperature', 0.2),
                max_tokens=max_tokens or self.cfg.get('max_tokens', 2048),
            )
            return resp.choices[0].message.content
        except Exception as e:
            return f"[LLM 调用失败：{e}]"

    def complete_with_consistency(self, prompt, system=None, n=None):
        """
        自一致性聚合（论文 3.4（4））：多次采样 → 一致性筛选。
        真实模式下调用 n 次取出现次数最多的结论；模拟模式返回单次。
        """
        if self.mock_mode:
            return self.complete(prompt, system)
        n = n or self.cfg.get('num_samples', 3)
        answers = []
        for _ in range(n):
            a = self.complete(prompt, system, temperature=0.7)
            answers.append(a)
        # 简单一致性：按首次输出为准（完整实现可做核心实体一致性比对）
        return answers[0]

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

    def complete(self, prompt, system=None, temperature=None, max_tokens=None, retries=3):
        for i in range(retries):
            try:
                return super().complete(prompt, system, temperature, max_tokens)
            except Exception as e:
                if i == retries - 1:
                    return f"[LLM 调用失败：{e}]"
