# -*- coding: utf-8 -*-
"""
graphrag/prompts.py —— GraphRAG 社区摘要 + Global Search 提示模板
==================================================================
社区摘要：对每个社区的结构化统计生成一段知识摘要（论文 3.x 全局索引）。
Global Search：map（每社区提炼要点）+ reduce（综合成最终回答）。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

COMMUNITY_SUMMARY_SYSTEM = (
    "你是作物病虫害知识图谱的分析专家。"
    "请基于给出的社区统计信息，用简洁中文生成该社区的知识摘要。"
)

COMMUNITY_SUMMARY_TEMPLATE = """该知识图谱社区包含以下内容：
【作物】{crops}
【病虫害】{pests}
【危害部位】{sites}
【发病时期】{periods}
【传播方式】{spread}
【防治要点】{controls}
【实体分布】{entity_dist}

请输出一段 200-400 字的中文摘要，覆盖：
1) 本社区核心主题（围绕哪些作物/病害虫类群）；
2) 主要病虫害及其典型危害部位/症状要点；
3) 共同的发生条件或传播方式（如有）；
4) 代表性防治措施（概括性，不列具体药名清单）。
摘要末尾单独一行用“关键词：”列出 3-6 个核心作物或病虫害名。"""

# Global Search - Map（每社区提炼相关事实）
GLOBAL_MAP_SYSTEM = "你是作物病虫害领域的知识聚合助手。只依据提供的社区摘要提取事实，不补充外部知识。"
GLOBAL_MAP_TEMPLATE = """【社区摘要】
{summary}

【问题】
{question}

请只从上述社区摘要中提取与本问题相关的事实，输出不超过 5 条要点（每条 ≤ 40 字），
用“- ”列表。若社区摘要与本问题无关，输出“（无相关事实）”。"""

# Global Search - Reduce（综合要点成最终回答）
GLOBAL_REDUCE_SYSTEM = "你是作物病虫害领域的知识综合助手。请综合提供的要点，用简洁中文回答。"
GLOBAL_REDUCE_TEMPLATE = """【各社区相关要点】
{points}

【问题】
{question}

请综合以上要点，用简洁中文作答（结论 + 关键要点，总长不超过 200 字）。
直接给出最终回答，不要解释过程。"""
