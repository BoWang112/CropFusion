# -*- coding: utf-8 -*-
"""
生成问答评测基准集（QA benchmark v4）
======================================

覆盖六类任务：
1. 病虫害识别
2. 症状解析
3. 条件判断
4. 防治措施推荐
5. 相似病虫害鉴别
6. 多条件综合推理

相对 v3 的关键修订
------------------
1. 相似病虫害鉴别仅使用知识图谱中真实存在的“相似/类似”关系，不再用“同一作物的任意病虫害”自动补齐。
2. 多条件综合推理优先要求症状证据与条件证据来自同一原始记录；若无同记录组合，可选同来源组合；默认不允许跨来源强行拼接。
3. 不再把由文本临时拼出的关系称为 gold_triples：
   - derived_relations：由文本证据归纳出的辅助关系，仅用于解释；
   - gold_graph_evidence：只保存实际从知识图谱读取到的真实图关系。
4. 增加 full_pest_name / clean_symptom_for_query 等文本组合函数，避免“水稻水稻纹枯病”“叶片出现叶片出现”等重复表达。
5. 移除 difficulty=L1/L2/L3 的固定任务映射，避免把“任务类型”误当成独立的推理难度变量。
6. 症状解析可整合 1-2 条互补症状证据；防治推荐可整合多类互补防治证据，而不是只随机取一段作为唯一参考答案。
7. 条件判断增加风险方向识别，参考答案优先明确“有利/不利/相关但方向不明确”。
8. 增强最终自动质检：
   - 多条件任务必须至少包含两条证据且满足证据一致性要求；
   - 相似鉴别必须具有真实 KG 相似关系；
   - 检查常见字符串重复与答案泄漏。

重要说明
--------
本脚本用于“自动构造候选评测集”，不是用来完全替代人工 gold 标注。
建议流程：
    自动生成 -> 自动质检 -> 人工审核/修正 -> 冻结 benchmark -> 正式跑实验

输出：
    EVAL_FILE（默认由 config.py 提供）
"""

import os
import sys
import json
import re
import random
import hashlib
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional, Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import NODES_CSV, RELS_CSV, EXTRACT_JSON, EVAL_FILE  # noqa: E402
from modules.kg import CropPestKG  # noqa: E402


# ============================================================
# 全局配置
# ============================================================

RANDOM_SEED = 2026
TARGET_PER_TASK = 80
MAX_CANDIDATES_PER_ITEM_PER_TASK = 4

# 每任务采样目标数（默认 80；图谱必需任务可扩到更高占比）。
# 方案：图谱综合诊断 / 图谱必需防治各 140 条，其余普通类压缩。
# 依据：实验显示"已知病虫害→查症状/鉴别"类问题 naive 纯文本也能答对，
#       无法体现图谱价值，故压缩这些类别，提高图谱必需题占比。
TARGET_BY_TASK = {
    "多条件综合推理": 140,   # 图谱综合诊断（graph_diag）
    "防治措施推荐": 140,     # 图谱必需防治（graph_control）
    "病虫害识别": 50,
    "症状解析": 50,
    "条件判断": 50,
    "相似病虫害鉴别": 50,
}
TOTAL_TARGET = sum(TARGET_BY_TASK.values())  # 480

# 多条件综合推理是否允许症状与条件来自不同 source_id。
# 正式论文 benchmark 建议保持 False，避免把两个独立场景强行拼成一个案例。
ALLOW_CROSS_SOURCE_MULTI = False

# 如果某一类经过质量过滤后不足 TARGET_PER_TASK：
# True  -> 直接报错（推荐用于论文最终 benchmark）
# False -> 允许输出不足 480 条，但会打印警告
STRICT_BALANCE = True

# 人工审核前建议保留为 pending
DEFAULT_REVIEW_STATUS = "pending"

random.seed(RANDOM_SEED)


TASKS = [
    "病虫害识别",
    "症状解析",
    "条件判断",
    "防治措施推荐",
    "相似病虫害鉴别",
    "多条件综合推理",
]

# 论文 benchmark 固定规模：50/50/50/140/50/140 = 480
EXPECTED_TARGET_BY_TASK = {
    "病虫害识别": 50,
    "症状解析": 50,
    "条件判断": 50,
    "防治措施推荐": 140,
    "相似病虫害鉴别": 50,
    "多条件综合推理": 140,
}
assert set(TARGET_BY_TASK) == set(EXPECTED_TARGET_BY_TASK) == set(TASKS)
assert TARGET_BY_TASK == EXPECTED_TARGET_BY_TASK
assert TOTAL_TARGET == 480


# ============================================================
# 1. 基础文本工具
# ============================================================

def normalize_space(t: str) -> str:
    if not t:
        return ""
    t = re.sub(r"\s+", " ", str(t))
    return t.strip()


def strip_template(t: str) -> str:
    """去除知识库中常见的“作物：x，病虫害：y，”模板前缀。"""
    t = normalize_space(t)
    if not t:
        return ""

    # 连续处理，兼容“作物：...，病虫害：...，”或反过来
    for _ in range(3):
        old = t
        t = re.sub(r"^作物[:：]\s*[^，,。；;]*[，,]?\s*", "", t)
        t = re.sub(r"^病虫害[:：]\s*[^，,。；;]*[，,]?\s*", "", t)
        if t == old:
            break
    return t.strip()


def short(t: str, maxlen: int = 120) -> str:
    """尽量在自然标点边界截断，避免把句子从中间截断。"""
    t = strip_template(t)
    if len(t) <= maxlen:
        return t

    cut = t[:maxlen]
    best = -1
    for c in "。；;！!？?":
        best = max(best, cut.rfind(c))
    if best > 12:
        return cut[: best + 1]
    return cut.rstrip("，,、 ") + "……"


def norm_question(q: str) -> str:
    """用于严格去重。"""
    q = normalize_space(q).lower()
    return re.sub(r"[\s，。、；：？！“”‘’\"',.!?;:（）()【】\[\]]+", "", q)


def stable_id(*parts: str, prefix: str = "ev") -> str:
    raw = "||".join(normalize_space(p) for p in parts if p is not None)
    h = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{h}"


def get_first_nonempty(d: Dict[str, Any], keys: List[str], default: str = "") -> str:
    for k in keys:
        v = d.get(k)
        if v is not None and str(v).strip():
            return str(v).strip()
    return default


# ============================================================
# 2. 领域轻量改写工具
# ============================================================

# 只做“低风险”同义替换，避免改坏专业术语和实体名称
SAFE_REWRITES = [
    ("发生严重", "发生较重"),
    ("发病严重", "发病较重"),
    ("湿度较大", "田间偏湿"),
    ("湿度大", "田间偏湿"),
    ("连续降雨", "连续阴雨"),
    ("降雨较多", "雨水偏多"),
    ("温度较高", "温度偏高"),
    ("温度较低", "温度偏低"),
    ("通风不良", "通风条件较差"),
    ("排水不良", "排水不畅"),
    ("易发生", "比较容易发生"),
    ("有利于发生", "容易诱发"),
    ("主要症状为", "常见表现是"),
    ("表现为", "可见"),
]


def light_paraphrase(t: str, max_replacements: int = 2) -> str:
    """
    保守型口语化改写：
    - 不替换病虫害名、农药名、作物名
    - 仅替换通用表达
    - 每次最多替换少量词，避免语义漂移
    """
    t = strip_template(t)
    candidates = [(a, b) for a, b in SAFE_REWRITES if a in t]
    random.shuffle(candidates)

    n = 0
    for a, b in candidates:
        if n >= max_replacements:
            break
        t = t.replace(a, b, 1)
        n += 1
    return t


# ============================================================
# 3. 症状、条件、部位抽取
# ============================================================

SITE_ALIASES = {
    "叶片": ["叶片", "叶面", "叶缘", "叶尖", "叶脉", "叶鞘", "叶子"],
    "茎秆": ["茎秆", "茎部", "茎基", "茎节", "秆"],
    "穗部": ["穗部", "穗颈", "穗轴", "稻穗", "麦穗", "穗"],
    "根部": ["根部", "根系", "根颈", "根"],
    "果实": ["果实", "果面", "果梗", "果柄", "果"],
    "花器": ["花器", "花瓣", "花蕾", "花"],
    "种子": ["种子", "籽粒", "籽"],
    "幼苗": ["幼苗", "苗期", "苗"],
    "全株": ["全株", "植株"],
}

SYMPTOM_MARKS = [
    "病斑", "斑点", "圆形", "椭圆形", "梭形", "水渍",
    "霉层", "坏死", "变褐", "褪绿", "枯死", "萎蔫", "萎缩",
    "卷曲", "卷缩", "腐烂", "白色", "黄色", "褐色", "灰色",
    "黑色", "暗绿", "失绿", "穿孔", "裂口", "流胶", "畸形",
    "缺刻", "虫道", "潜道", "网状", "皱缩",
]

CONDITION_MARKS = [
    "高温", "高湿", "低温", "多雨", "阴雨", "干旱", "连作",
    "氮肥", "气温", "湿度", "光照", "土壤", "雨水", "降雨",
    "温度", "降水", "虫口", "湿度大", "田间湿度", "密植",
    "通风", "积水", "排水", "低温寡照", "昼夜温差",
]


def extract_site_from_text(t: str) -> str:
    """
    只从真实证据中抽取受害部位。
    找不到就返回空字符串，绝不随机补。
    """
    t = strip_template(t)
    hits = []
    for canonical, aliases in SITE_ALIASES.items():
        for a in aliases:
            idx = t.find(a)
            if idx >= 0:
                hits.append((idx, -len(a), canonical))
    if not hits:
        return ""
    hits.sort()
    return hits[0][2]


def split_clauses(t: str) -> List[str]:
    t = strip_template(t)
    parts = re.split(r"[。；;！？!?]+", t)
    out = []
    for p in parts:
        p = p.strip(" ，,、")
        if p:
            out.append(p)
    return out


def symptom_fragment(t: str, maxlen: int = 55) -> Optional[str]:
    """
    从证据中抽取症状片段。
    只返回确实含症状特征的子句；不再无条件返回 t[:maxlen] 兜底。
    """
    t = strip_template(t)
    if not t:
        return None

    clauses = split_clauses(t)
    scored = []
    for c in clauses:
        score = sum(1 for m in SYMPTOM_MARKS if m in c)
        # 部位词也提供少量权重
        if extract_site_from_text(c):
            score += 1
        if score > 0 and len(c) >= 6:
            scored.append((score, -len(c), c))

    if not scored:
        return None

    scored.sort(reverse=True)
    frag = scored[0][2]
    frag = light_paraphrase(frag)
    return short(frag, maxlen)


def condition_fragment(t: str, maxlen: int = 45) -> Optional[str]:
    """从条件段中抽取较完整、可读的环境/栽培条件子句。"""
    t = strip_template(t)
    if not t:
        return None

    clauses = split_clauses(t)
    scored = []
    for c in clauses:
        score = sum(1 for m in CONDITION_MARKS if m in c)
        if score > 0 and 6 <= len(c) <= 120:
            scored.append((score, -len(c), c))

    if not scored:
        return None

    scored.sort(reverse=True)
    frag = scored[0][2]

    # 去掉明显的编号前缀
    frag = re.sub(r"^[（(]?\d+[）).、]?\s*", "", frag).strip()
    if len(frag) < 6:
        return None

    frag = light_paraphrase(frag)
    return short(frag, maxlen)


# ============================================================
# 4. 段落分类与稳定证据单元
# ============================================================

def seg_classify(t: str) -> str:
    """
    段落类型：症状 / 条件 / 防治 / 机制 / 其他
    注意优先级：防治 > 机制 > 症状 > 条件 > 其他。
    """
    t = strip_template(t)

    control_words = [
        "防治", "药剂", "喷施", "喷雾", "浸种", "拌种", "轮作", "施药",
        "抗病品种", "灌穴", "淋施", "黄板", "诱杀", "覆盖", "清沟",
        "排水", "开沟", "降湿", "施肥", "追肥", "有机肥", "堆肥",
        "磷钾", "清除病株", "烧毁", "深翻", "药液", "可湿性粉剂",
        # 诱杀/色板/人工防治类措辞（数据中常无"防治"字样）
        "黏油", "机油", "涂黏", "色板", "纸板", "黏满", "粘板",
        "人工捕捉", "摘除", "振落", "捕杀", "覆盖地膜", "铺膜",
    ]
    if any(k in t for k in control_words):
        return "防治"

    mechanism_words = [
        "病原", "菌丝", "孢子", "越冬", "初侵染", "再侵染", "传播",
        "世代", "孵化", "羽化", "产卵", "侵入", "寄主", "病菌",
        "幼虫", "成虫", "分生孢子", "虫卵", "生活史",
    ]
    if any(k in t for k in mechanism_words):
        return "机制"

    if any(k in t for k in SYMPTOM_MARKS):
        return "症状"

    if any(k in t for k in CONDITION_MARKS):
        return "条件"

    # 弱机制词放最后，降低误判
    if any(k in t for k in ["发生规律", "为害特点", "发病规律", "侵染循环"]):
        return "机制"

    return "其他"


def make_segment_unit(
    rec: Dict[str, Any],
    rec_idx: int,
    seg: Dict[str, Any],
    seg_idx: int,
    crop: str,
    pest: str,
) -> Dict[str, Any]:
    """
    把原始 segment 转为稳定证据单元。
    优先保留原系统已有的 chunk/document/source ID；
    若没有，则基于 crop+pest+text 构造稳定 hash ID。
    """
    text = strip_template(seg.get("text") or "")
    raw_id = get_first_nonempty(
        seg,
        ["id", "segment_id", "chunk_id", "evidence_id", "text_id", "uid"],
        default=""
    )
    source = get_first_nonempty(
        seg,
        ["source", "source_id", "document", "document_id", "doc_id", "file", "title"],
        default=""
    )
    if not source:
        source = get_first_nonempty(
            rec,
            ["source", "source_id", "document", "document_id", "doc_id", "file", "title"],
            default=f"record_{rec_idx}"
        )

    evidence_id = raw_id or stable_id(crop, pest, text, prefix="seg")

    return {
        "evidence_id": evidence_id,
        "text": text,
        "source_id": source,
        "record_index": rec_idx,
        "segment_index": seg_idx,
    }


def split_segments(rec: Dict[str, Any], rec_idx: int, crop: str, pest: str) -> Dict[str, List[Dict[str, Any]]]:
    out = {
        "symptoms": [],
        "conditions": [],
        "controls": [],
        "mechanisms": [],
        "other": [],
    }
    key_map = {
        "症状": "symptoms",
        "条件": "conditions",
        "防治": "controls",
        "机制": "mechanisms",
        "其他": "other",
    }

    for seg_idx, seg in enumerate(rec.get("segments", [])):
        unit = make_segment_unit(rec, rec_idx, seg, seg_idx, crop, pest)
        if len(unit["text"]) < 10:
            continue
        c = seg_classify(unit["text"])
        out[key_map[c]].append(unit)
    return out


# ============================================================
# 5. 构建 (crop, pest) 素材目录
# ============================================================

def dedupe_units(units: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for u in units:
        key = (u["evidence_id"], u["text"])
        if key in seen:
            continue
        seen.add(key)
        out.append(u)
    return out


def build_pest_catalog(recs: List[Dict[str, Any]]) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """
    目录键改为 (crop, pest)，彻底避免跨作物串知识。
    """
    catalog: Dict[Tuple[str, str], Dict[str, Any]] = {}

    for rec_idx, rec in enumerate(recs):
        crop = (rec.get("crop") or "").strip()
        pest = (rec.get("pest") or "").strip()
        if not crop or not pest or pest.lower() == "nan":
            continue

        key = (crop, pest)
        cat = catalog.setdefault(
            key,
            {
                "crop": crop,
                "pest": pest,
                "domain": rec.get("domain", ""),
                "symptoms": [],
                "conditions": [],
                "controls": [],
                "mechanisms": [],
            },
        )

        segs = split_segments(rec, rec_idx, crop, pest)
        for k in ("symptoms", "conditions", "controls", "mechanisms"):
            cat[k].extend(segs[k])

        # 若同一 (crop, pest) 在不同记录中 domain 有值，优先保留非空
        if not cat["domain"] and rec.get("domain"):
            cat["domain"] = rec.get("domain")

    for cat in catalog.values():
        for k in ("symptoms", "conditions", "controls", "mechanisms"):
            cat[k] = dedupe_units(cat[k])

    return catalog


# ============================================================
# 6. 问题模板与名称/症状组合工具
# ============================================================

IDENTIFICATION_TAILS_DISEASE = [
    "最可能是什么病害？",
    "这种情况更像是哪一种病？",
    "从这些表现看，可能是什么病害？",
    "请判断最可能的病害名称。",
]

IDENTIFICATION_TAILS_PEST = [
    "最可能是哪一种虫害？",
    "这种为害更像是什么害虫造成的？",
    "从这些表现看，可能是什么虫害？",
    "请判断最可能的害虫或虫害名称。",
]

IDENTIFICATION_TAILS_GENERIC = [
    "最可能是什么病虫害？",
    "这种情况通常对应哪一种病虫害？",
    "请判断最可能的病虫害名称。",
]

SYMPTOM_QUERY_TEMPLATES = [
    "{crop}发生{pest}时，通常会出现哪些典型症状？",
    "{pest}在{crop}上的主要症状表现是什么？",
    "{crop}{pest}主要危害哪些部位，会出现什么表现？",
    "{crop}{pest}常见的受害表现有哪些？",
]

CONDITION_QUERY_TEMPLATES = [
    "{cond}时，{crop}的{pest}发生风险会怎样变化？请说明依据。",
    "在“{cond}”这种条件下，{crop}{pest}的发生情况通常会怎样变化？",
    "{crop}{pest}与“{cond}”之间有什么发生关系？",
    "田间出现“{cond}”时，{crop}{pest}的发生风险如何？为什么？",
]

CONTROL_QUERY_TEMPLATES = [
    "{crop}发生{pest}后，应采取哪些防治措施？",
    "{crop}{pest}应该怎么综合防治？",
    "针对{crop}的{pest}，田间管理和防治上应该注意什么？",
    "发现{crop}{pest}后，可以采取哪些有效的防控办法？",
]

DIFF_QUERY_TEMPLATES = [
    "{crop}上的{pest_a}和{pest_b}容易混淆，应怎样根据症状或发生条件进行鉴别？",
    "怎样区分{crop}{pest_a}与{pest_b}？请给出关键鉴别点。",
    "{crop}出现相近症状时，{pest_a}和{pest_b}主要从哪些特征上区分？",
]

MULTI_QUERY_TEMPLATES = [
    "{crop}{site_phrase}出现{symptom}，近期又有{cond}。最可能是什么病虫害？请结合两方面证据说明。",
    "田间发现{crop}{site_phrase}{symptom}，同时近期{cond}。请综合判断最可能的病虫害并说明依据。",
    "{crop}现在有{symptom}的表现，而且近期{cond}。这种情况更符合哪一种病虫害？",
    "如果{crop}{site_phrase}出现{symptom}，并伴随{cond}，应优先怀疑什么病虫害？为什么？",
]

# 图谱综合诊断（做法 A：图谱多跳结构必需）——
# 症状 + 受害部位 -> 病害 + 危害部位 + 防治办法。
# 要求必须经过 病虫害 -为害-> 发病部位 -防治方法有-> 防治节点 的多跳路径才能答全。
GRAPH_DIAG_QUERY_TEMPLATES = [
    "{crop}的{symptom}。结合这种受害部位，最可能是哪种病虫害？它主要危害哪些部位，又该如何防治？",
    "{crop}的{site}上出现{symptom}。据此判断这是什么病虫害？其危害部位和防治措施分别是什么？",
    "田间看到{crop}的{site}出现{symptom}。这最可能是什么病虫害？请一并说明它还会危害哪些部位，以及推荐的防治方法。",
    "{crop}发现{site}出现{symptom}。最可能是什么病虫害？除了{site}，它还会为害哪些部位？应怎样防治？",
]

# 图谱中“为害部位”关系的规范关系名（检查时兼容）
HARM_REL_NAMES = ("为害", "发病于", "影响", "直接引发", "致使", "处于")

# 图谱中“防治方法”关系的规范关系名
CONTROL_REL_NAMES = ("防治方法有",)

# 病虫害 → 为害部位 图谱边的 部位实体类型（与 retriever.evidence_entities 对齐）
SITE_ENTITY_TYPES = {"发病部位"}

# 病虫害 → 防治方法 图谱边的 防治节点实体类型
CONTROL_ENTITY_TYPES = {"农业防治", "化学防治", "物理防治", "生物防治"}


def pick_template(pool: List[str], **kw) -> str:
    return random.choice(pool).format(**kw)


def display_pest(crop: str, pest: str) -> str:
    """
    用于“作物 + 病虫害”模板中的短名称。
    例如：(水稻, 水稻纹枯病) -> 纹枯病。
    """
    crop = (crop or "").strip()
    pest = (pest or "").strip()
    if crop and pest.startswith(crop):
        rest = pest[len(crop):].lstrip("的、")
        return rest or pest
    return pest


# ============================================================
# 冷门细节提取（路线一核心：答案必须靠检索证据才能答出）
# ============================================================
# 冷门细节 = 具体数字+单位、药剂名+浓度、温度、防治步骤等
# LLM 凭常识难答出，只能从检索证据中获取。
_COLD_PATTERNS = [
    r"\d+\s*[℃%倍天年月日时mlgkg亩厘米千米米]",
    r"\d+\.\d+\s*[℃%倍天]",
    r"种子重量\s*\d+\.?\d*\s*%",
    r"\d+[-—]\d+\s*[℃%天月]",
    r"(浸种|拌种|喷洒|喷雾|灌根|穴施)\s*\d+",
    r"\d+%\s*[A-Za-z0-9]+(剂|液|粉|乳油|WP|SC)",
]


def is_cold_detail(t: str) -> bool:
    """文本是否含冷门细节（具体数字/药剂/浓度）"""
    return any(re.search(p, t) for p in _COLD_PATTERNS)


def extract_cold_fragment(t: str, maxlen: int = 80) -> Optional[str]:
    """从证据中提取含冷门细节的片段；无则返回 None"""
    t = strip_template(t)
    clauses = split_clauses(t)
    for c in clauses:
        if is_cold_detail(c) and len(c) >= 6:
            return short(c, maxlen)
    if is_cold_detail(t):
        return short(t, maxlen)
    return None


def full_pest_name(crop: str, pest: str) -> str:
    """
    返回不重复的完整名称：
        (水稻, 水稻纹枯病) -> 水稻纹枯病
        (水稻, 纹枯病)     -> 水稻纹枯病
    """
    crop = (crop or "").strip()
    pest = (pest or "").strip()
    if not crop:
        return pest
    if pest.startswith(crop):
        return pest
    return crop + pest


def clean_symptom_for_query(symptom: str, crop: str = "", site: str = "") -> str:
    """
    清理用于问题题干的症状片段：
    - 去掉可能重复的作物名；
    - 若外层已显式添加 site，则去掉症状开头重复的 site；
    - 去掉“出现/表现为/可见”等开头，避免“出现出现”。
    """
    s = normalize_space(symptom)
    if crop:
        s = s.replace(crop, "", 1).strip("，,。；; ")

    # 去除常见句首连接词
    s = re.sub(r"^(主要症状为|常见表现是|表现为|可见|出现了?|发生了?)", "", s).strip("，,。；; ")

    if site:
        # 只去掉开头的重复受害部位，不删除正文中的真实部位信息
        aliases = SITE_ALIASES.get(site, [site])
        aliases = sorted(set([site] + aliases), key=len, reverse=True)
        for a in aliases:
            if s.startswith(a):
                s = s[len(a):].strip("，,。；; ")
                s = re.sub(r"^(上|部|处)?(出现了?|可见|表现为)", "", s).strip("，,。；; ")
                break

    return s


def compose_site_phrase(site: str, symptom_text: str, with_de: bool = False) -> str:
    """
    只有当症状文本开头没有重复 site 时才在外层添加受害部位。
    """
    if not site:
        return ""
    aliases = SITE_ALIASES.get(site, [site])
    if any(symptom_text.startswith(a) for a in aliases):
        return ""
    return ("的" + site) if with_de else site


def domain_tail(cat: Dict[str, Any]) -> str:
    d = str(cat.get("domain") or "")
    if "虫" in d:
        return random.choice(IDENTIFICATION_TAILS_PEST)
    if "病" in d:
        return random.choice(IDENTIFICATION_TAILS_DISEASE)
    return random.choice(IDENTIFICATION_TAILS_GENERIC)

# ============================================================
# 7. Gold evidence / QA 结构
# ============================================================

def make_evidence_payload(units: List[Dict[str, Any]]) -> Dict[str, Any]:
    units = dedupe_units(units)
    return {
        "gold_evidence_ids": [u["evidence_id"] for u in units],
        "gold_evidence_texts": [u["text"] for u in units],
        "source_ids": list(dict.fromkeys(u["source_id"] for u in units if u.get("source_id"))),
    }


def make_qa(
    task: str,
    crop: str,
    pest: str,
    question: str,
    answer: str,
    evidence_units: List[Dict[str, Any]],
    derived_relations: Optional[List[List[str]]] = None,
    gold_graph_evidence: Optional[List[Dict[str, Any]]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    构造一条 QA。

    字段约定：
    - gold_evidence_*：真实文本证据；
    - derived_relations：根据文本证据归纳的辅助关系，不作为 KG gold；
    - gold_graph_evidence：只有从知识图谱真实边中读取到的关系才允许写入。
    """
    payload = make_evidence_payload(evidence_units)
    qa_id = stable_id(task, crop, pest, question, prefix="qa")

    qa = {
        "id": qa_id,
        "task": task,
        "crop": crop,
        "question": normalize_space(question),
        "reference_answer": normalize_space(answer),
        # 为兼容旧评测代码保留 answer 字段
        "answer": normalize_space(answer),
        "gold_pest": pest,
        "gold_evidence_ids": payload["gold_evidence_ids"],
        "gold_evidence_texts": payload["gold_evidence_texts"],
        "source_ids": payload["source_ids"],
        "derived_relations": derived_relations or [],
        "gold_graph_evidence": gold_graph_evidence or [],
        "review_status": DEFAULT_REVIEW_STATUS,
    }
    if metadata:
        qa["metadata"] = metadata
    return qa

def select_related_units(
    primary: Dict[str, Any],
    units: List[Dict[str, Any]],
    max_items: int = 2,
) -> List[Dict[str, Any]]:
    """
    为同一问答选择互补证据：
    优先同 record，其次同 source，最后其他同一 (crop,pest) 证据。
    """
    if max_items <= 1:
        return [primary]

    selected = [primary]
    others = [u for u in units if u["evidence_id"] != primary["evidence_id"]]

    def score(u: Dict[str, Any]) -> Tuple[int, int]:
        same_record = int(u.get("record_index") == primary.get("record_index"))
        same_source = int(
            bool(u.get("source_id"))
            and u.get("source_id") == primary.get("source_id")
        )
        return (same_record, same_source)

    others.sort(key=score, reverse=True)

    for u in others:
        if len(selected) >= max_items:
            break
        # 避免高度重复文本
        if any(char_jaccard(u["text"], x["text"]) > 0.90 for x in selected):
            continue
        selected.append(u)

    return selected


POSITIVE_CONDITION_CUES = [
    "有利于", "易发生", "容易发生", "发生重", "发病重", "流行",
    "加重", "适宜", "高发", "促进", "风险增加", "危害重",
]
NEGATIVE_CONDITION_CUES = [
    "不利于", "不易发生", "发生少", "发病轻", "抑制",
    "减轻", "风险降低", "不适宜",
]


def infer_condition_effect(text: str) -> str:
    """
    从证据原文识别发生风险方向。
    返回：increase / decrease / related_unknown
    """
    t = strip_template(text)
    pos = any(k in t for k in POSITIVE_CONDITION_CUES)
    neg = any(k in t for k in NEGATIVE_CONDITION_CUES)

    if pos and not neg:
        return "increase"
    if neg and not pos:
        return "decrease"
    return "related_unknown"


CHEMICAL_CONTROL_WORDS = [
    "药剂", "喷雾", "施药", "可湿性粉剂", "乳油", "水剂", "颗粒剂",
    "农药", "杀菌剂", "杀虫剂", "三唑", "多菌灵", "百菌清",
    "代森", "敌敌畏", "乐果", "菊酯", "拌种", "浸种",
]
BIO_CONTROL_WORDS = [
    "生物防治", "天敌", "生防", "微生物", "菌剂", "寄生蜂",
    "捕食螨", "苏云金杆菌", "白僵菌", "绿僵菌",
]
PHYSICAL_CONTROL_WORDS = [
    "黄板", "色板", "诱杀", "人工捕捉", "摘除", "振落",
    "捕杀", "粘板", "黏油", "灯光诱杀",
]
AGRONOMIC_CONTROL_WORDS = [
    "抗病品种", "轮作", "合理密植", "清沟", "排水", "开沟",
    "施肥", "肥水", "田间管理", "清除病株", "烧毁", "深翻",
    "播种", "育苗", "覆盖", "有机肥", "磷钾", "清洁田园",
]


def classify_control_type(text: str) -> str:
    t = strip_template(text)
    if any(k in t for k in BIO_CONTROL_WORDS):
        return "生物防治"
    if any(k in t for k in CHEMICAL_CONTROL_WORDS):
        return "化学防治"
    if any(k in t for k in PHYSICAL_CONTROL_WORDS):
        return "物理/机械防治"
    if any(k in t for k in AGRONOMIC_CONTROL_WORDS):
        return "农业防治"
    return "其他防治"


def select_control_units(
    units: List[Dict[str, Any]],
    max_items: int = 3,
) -> List[Tuple[str, Dict[str, Any]]]:
    """
    尽量从不同防治类别中各取一条证据。
    不存在的类别不会被人为补齐。
    """
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for u in units:
        groups[classify_control_type(u["text"])].append(u)

    preferred_order = [
        "农业防治",
        "生物防治",
        "物理/机械防治",
        "化学防治",
        "其他防治",
    ]

    selected: List[Tuple[str, Dict[str, Any]]] = []
    for category in preferred_order:
        bucket = groups.get(category, [])
        if not bucket:
            continue
        random.shuffle(bucket)
        selected.append((category, bucket[0]))
        if len(selected) >= max_items:
            break

    if not selected and units:
        selected = [("其他防治", random.choice(units))]

    return selected


def choose_coherent_multi_pair(
    symptom_candidates: List[Tuple[Dict[str, Any], str]],
    condition_candidates: List[Tuple[Dict[str, Any], str]],
) -> Optional[Tuple[Dict[str, Any], str, Dict[str, Any], str, str]]:
    """
    为多条件推理选择“症状 + 条件”证据。

    优先级：
    1. same_record：同一原始记录；
    2. same_source：同一来源文献/记录来源；
    3. cross_source：仅当 ALLOW_CROSS_SOURCE_MULTI=True 时允许。

    返回：
        sym_unit, sym, cond_unit, cond, coherence
    """
    pairs = []
    for su, sym in symptom_candidates:
        for cu, cond in condition_candidates:
            if su["evidence_id"] == cu["evidence_id"]:
                continue

            same_record = su.get("record_index") == cu.get("record_index")
            same_source = (
                bool(su.get("source_id"))
                and su.get("source_id") == cu.get("source_id")
            )

            if same_record:
                score = 3
                coherence = "same_record"
            elif same_source:
                score = 2
                coherence = "same_source"
            elif ALLOW_CROSS_SOURCE_MULTI:
                score = 1
                coherence = "cross_source"
            else:
                continue

            pairs.append((score, random.random(), su, sym, cu, cond, coherence))

    if not pairs:
        return None

    pairs.sort(key=lambda x: (x[0], x[1]), reverse=True)
    _, _, su, sym, cu, cond, coherence = pairs[0]
    return su, sym, cu, cond, coherence


# ============================================================
# 8. 六类任务生成器
# ============================================================

def make_identification(cat: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    病虫害识别（路线一：答案必须引用证据原文细节）：
        输入 = 部分症状描述
        输出 = 病虫害实体 + 关键症状依据（引用证据原文，含具体细节）
    问题中绝不出现 gold pest。
    """
    if not cat["symptoms"]:
        return None

    # 优先选含具体细节（尺寸/颜色/形态）的症状段，提高检索必需性
    pool = [u for u in cat["symptoms"] if is_cold_detail(u["text"])] or cat["symptoms"]
    unit = random.choice(pool)
    sym = symptom_fragment(unit["text"])
    if not sym or len(sym) < 8:
        return None

    crop = cat["crop"]
    site = extract_site_from_text(sym) or extract_site_from_text(unit["text"])
    display_sym = clean_symptom_for_query(sym, crop=crop, site=site)
    if len(display_sym) < 4:
        return None

    # 问题：给部分症状 + 问具体细节（尺寸/颜色/形态），逼模型引用证据原文
    site_phrase = compose_site_phrase(site, display_sym, with_de=False)
    if is_cold_detail(sym):
        question = (f"{crop}{site_phrase}出现{display_sym}。"
                    f"请判断最可能的病虫害，并说明病斑的具体大小、颜色或形态细节。")
    else:
        question = f"{crop}{site_phrase}出现{display_sym}，{domain_tail(cat)}"

    answer = (
        f"最符合{cat['pest']}。"
        f"关键症状依据为：{short(unit['text'], 100)}"
    )

    derived = [[cat["pest"], "典型症状", display_sym]]
    return make_qa(
        task="病虫害识别",
        crop=crop,
        pest=cat["pest"],
        question=question,
        answer=answer,
        evidence_units=[unit],
        derived_relations=derived,
    )

def make_symptom_parse(cat: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    症状解析（路线一：答案引用证据原文）：
        输入 = 已知病虫害
        输出 = 典型症状/受害表现（引用证据原文，含具体细节）
    最多整合 2 条互补症状证据。
    """
    if not cat["symptoms"]:
        return None

    # 优先选含具体细节的症状段
    pool = [u for u in cat["symptoms"] if is_cold_detail(u["text"])] or cat["symptoms"]
    primary = random.choice(pool)
    primary_sym = symptom_fragment(primary["text"])
    if not primary_sym:
        return None

    units = select_related_units(primary, cat["symptoms"], max_items=2)

    crop = cat["crop"]
    pest = cat["pest"]
    disp = display_pest(crop, pest)
    full_name = full_pest_name(crop, pest)

    if is_cold_detail(primary["text"]):
        # 问具体细节：病斑尺寸/颜色/形态
        question = (f"{crop}发生{disp}时，其病斑或受害部位具体是什么大小、颜色和形态？"
                    f"请给出具体描述。")
    else:
        question = pick_template(
            SYMPTOM_QUERY_TEMPLATES,
            crop=crop,
            pest=disp,
        )

    evidence_parts = [short(u["text"], 100) for u in units]
    answer = f"{full_name}的典型症状主要包括：" + "；".join(evidence_parts) + "。"

    derived = []
    for u in units:
        frag = symptom_fragment(u["text"], 60)
        if frag:
            derived.append([pest, "典型症状", frag])

    return make_qa(
        task="症状解析",
        crop=crop,
        pest=pest,
        question=question,
        answer=answer,
        evidence_units=units,
        derived_relations=derived,
    )

def make_condition(cat: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    条件判断（路线一：优先冷门细节）：
    - 优先选择含具体温度/湿度/降雨阈值的条件证据（LLM 常识难答出具体数值）；
    - 问题问“具体在什么温湿度条件下易发生”，答案 = 证据中的具体数值。
    """
    # 优先含冷门细节的条件证据
    cold_cands = []
    for unit in cat["conditions"]:
        frag = condition_fragment(unit["text"])
        if frag and is_cold_detail(unit["text"]):
            cold_cands.append((unit, frag))

    candidates = cold_cands
    if not candidates:
        for unit in cat["conditions"]:
            frag = condition_fragment(unit["text"])
            if frag:
                candidates.append((unit, frag))
    if not candidates:
        return None

    unit, cond = random.choice(candidates)
    crop = cat["crop"]
    pest = cat["pest"]
    disp = display_pest(crop, pest)

    if cold_cands:
        # 冷门细节问题：问具体温湿度阈值
        question = f"{crop}的{disp}在什么具体的温度、湿度或降雨条件下最容易发生？请给出具体数值。"
        evidence_text = short(unit["text"], 130)
        answer = f"根据证据，{disp}的适宜发生条件为：{evidence_text}"
        effect = infer_condition_effect(unit["text"])
    else:
        question = pick_template(
            CONDITION_QUERY_TEMPLATES,
            crop=crop,
            pest=disp,
            cond=cond,
        )
        effect = infer_condition_effect(unit["text"])
        evidence_text = short(unit["text"], 130)
        if effect == "increase":
            answer = f"该条件有利于{pest}发生或加重。依据是：{evidence_text}"
        elif effect == "decrease":
            answer = f"该条件不利于{pest}发生，或会降低其发生风险。依据是：{evidence_text}"
        else:
            answer = (
                f"现有证据表明该条件与{pest}的发生有关，但不足以明确判断风险升高或降低。"
                f"相关依据为：{evidence_text}"
            )

    derived = [[pest, "发生条件", cond]]
    return make_qa(
        task="条件判断",
        crop=crop,
        pest=pest,
        question=question,
        answer=answer,
        evidence_units=[unit],
        derived_relations=derived,
        metadata={"condition_effect": effect},
    )

def make_control(cat: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    防治措施推荐（路线一：冷门细节优先）：
    - 优先选择含具体药剂/浓度/温度的防治证据（LLM 常识难答出，必须检索）；
    - 问题问”具体用什么药剂、多少浓度”，答案 = 证据中的冷门细节原文。
    """
    if not cat["controls"]:
        return None

    crop = cat["crop"]
    pest = cat["pest"]
    disp = display_pest(crop, pest)

    # 优先找含冷门细节的防治证据
    cold_units = [u for u in cat["controls"] if is_cold_detail(u["text"])]
    pool = cold_units or cat["controls"]
    if not pool:
        return None

    # 用冷门细节问题模板
    if cold_units:
        question = f"{crop}发生{disp}后，防治时具体用什么药剂、多少浓度（或什么处理方式）？"
        unit = random.choice(cold_units)
        txt = extract_cold_fragment(unit["text"], 120)
        if not txt:
            txt = short(unit["text"], 120)
        answer = f"根据证据，防治{disp}可参考：{txt}"
        derived = [[pest, "防治方式", txt]]
        return make_qa(
            task="防治措施推荐",
            crop=crop,
            pest=pest,
            question=question,
            answer=answer,
            evidence_units=[unit],
            derived_relations=derived,
        )

    # 无冷门细节时退回通用防治问题
    selected = select_control_units(pool, max_items=2)
    if not selected:
        return None
    question = pick_template(
        CONTROL_QUERY_TEMPLATES,
        crop=crop,
        pest=disp,
    )
    parts = []
    evidence_units = []
    derived = []
    for category, unit in selected:
        txt = short(unit["text"], 120)
        parts.append(f"{category}：{txt}")
        evidence_units.append(unit)
        derived.append([pest, "防治方式", txt])
    answer = "可根据现有证据采取以下防治措施：" + "；".join(parts) + "。"
    return make_qa(
        task="防治措施推荐",
        crop=crop,
        pest=pest,
        question=question,
        answer=answer,
        evidence_units=evidence_units,
        derived_relations=derived,
    )


def make_control_graph(kg: CropPestKG, cat: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    防治措施推荐（图谱必需版）：
        - 答案 = 该病虫害在图谱中、但全部文本里都不存在的防治节点
        - naive-text 纯文本检索拿不到这些节点（文本缺失），只有 full 的图谱检索能拿到
        - 问题问具体防治方法，参考答案列出图谱缺失防治节点
    """
    # 防治推荐问题不需要症状证据（答案是图谱防治节点），
    # 因此不要求 cat["symptoms"] 非空，覆盖全部有缺失防治节点的病虫害。
    all_text = " ".join(
        u["text"] for u in (cat["symptoms"] + cat["conditions"]
                            + cat["controls"] + cat["mechanisms"])
    )
    ctrl_edges = dedupe_kg_edges(kg_control_edges(kg, cat["pest"]))
    if not ctrl_edges:
        return None
    missing_ctrl = [e for e in ctrl_edges if e["tail"] not in all_text]
    if not missing_ctrl:
        return None
    # 冷门防治优先（LLM 常识也答不出），随机选 3-4 个不同节点，
    # 使参考答案覆盖更多图谱独有防治节点（增大 full 相对 naive 的领先空间）。
    cold_missing = [e for e in missing_ctrl if is_cold_detail(e["tail"])]
    random.shuffle(missing_ctrl)
    # 随机数量：3 或 4
    n_ctrl = random.choice([3, 4])
    # 优先保证至少 1-2 个冷门节点（若有）
    if cold_missing:
        n_cold = min(2, len(cold_missing))
        ctrl_nodes = [random.choice(cold_missing)["tail"] for _ in range(n_cold)]
        rest = [e["tail"] for e in missing_ctrl if e["tail"] not in ctrl_nodes]
        random.shuffle(rest)
        ctrl_nodes += rest[: n_ctrl - len(ctrl_nodes)]
    else:
        ctrl_nodes = [e["tail"] for e in missing_ctrl[:n_ctrl]]
    # 去重保持顺序
    ctrl_nodes = list(dict.fromkeys(ctrl_nodes))

    crop = cat["crop"]
    pest = cat["pest"]
    disp = display_pest(crop, pest)

    question = (
        f"{crop}发生{disp}后，具体应该采取哪些防治措施？"
        f"请依据图谱知识给出推荐的防治方法。"
    )
    answer = f"根据图谱证据，防治{disp}可参考：{'、'.join(ctrl_nodes)}。"

    return make_qa(
        task="防治措施推荐",
        crop=crop,
        pest=pest,
        question=question,
        answer=answer,
        evidence_units=[],
        derived_relations=[[pest, "防治方法有", t] for t in ctrl_nodes],
        gold_graph_evidence=[e for e in ctrl_edges if e["tail"] in ctrl_nodes],
        metadata={"diag_mode": "graph_control", "control_nodes": ctrl_nodes},
    )


def kg_similar_edges(kg: CropPestKG, pest: str) -> List[Dict[str, str]]:
    """
    只读取知识图谱中真实存在的“相似/类似”关系。

    返回的每一项都对应实际 KG 边，不再把“同作物其他病虫害”
    自动视为相似对象。
    """
    rel_names = [
        "类似虫害有",
        "类似病害有",
        "相似病虫害",
        "相似病害",
        "相似虫害",
    ]

    nid = kg.find_pest(pest)
    if not nid:
        return []

    edges: List[Dict[str, str]] = []

    for rel_name in rel_names:
        try:
            for tgt, _rel in kg.neighbors(nid, rel_name):
                name = kg.node_name(tgt)
                if name and name != pest:
                    edges.append({
                        "head": pest,
                        "rel": rel_name,
                        "tail": name,
                        "triple_id": stable_id(pest, rel_name, name, prefix="kg"),
                        "source_id": "knowledge_graph",
                    })
        except Exception:
            pass

        try:
            for src, _rel in kg.in_neighbors(nid, rel_name):
                name = kg.node_name(src)
                if name and name != pest:
                    edges.append({
                        "head": name,
                        "rel": rel_name,
                        "tail": pest,
                        "triple_id": stable_id(name, rel_name, pest, prefix="kg"),
                        "source_id": "knowledge_graph",
                    })
        except Exception:
            pass

    seen = set()
    out = []
    for e in edges:
        key = (e["head"], e["rel"], e["tail"])
        if key in seen:
            continue
        seen.add(key)
        out.append(e)
    return out

def char_jaccard(a: str, b: str) -> float:
    sa = set(re.sub(r"\s+", "", a))
    sb = set(re.sub(r"\s+", "", b))
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def best_discriminative_evidence(cat: Dict[str, Any]) -> Optional[Tuple[Dict[str, Any], str]]:
    """
    选一个可读症状片段作为鉴别证据。
    """
    candidates = []
    for unit in cat["symptoms"]:
        frag = symptom_fragment(unit["text"], 60)
        if frag:
            score = sum(1 for m in SYMPTOM_MARKS if m in frag)
            candidates.append((score, unit, frag))
    if not candidates:
        return None
    candidates.sort(key=lambda x: (x[0], len(x[2])), reverse=True)
    _, unit, frag = candidates[0]
    return unit, frag


def make_similar(
    kg: CropPestKG,
    cat: Dict[str, Any],
    catalog: Dict[Tuple[str, str], Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """
    相似病虫害鉴别：
    - 仅使用 KG 中真实存在的相似关系；
    - 只允许同一 crop 下比较；
    - 两侧必须都有可用症状证据；
    - gold_graph_evidence 保存真实 KG 相似边。
    """
    crop = cat["crop"]
    pest_a = cat["pest"]

    ev_a = best_discriminative_evidence(cat)
    if not ev_a:
        return None

    # 候选构造：图谱真实相似边优先；不足时用同作物其他病虫害兜底。
    # （图谱“类似虫害有”边大量为 nan、有效边稀疏，需同作物候选补齐。）
    edge_candidates = kg_similar_edges(kg, pest_a)
    random.shuffle(edge_candidates)

    # 同作物候选（兜底）：只取同 domain（病害对病害 / 虫害对虫害），
    # 避免把病害与虫害强行作“鉴别”
    my_domain = cat.get("domain", "")
    same_crop_names = [
        other["pest"]
        for (_c, _p), other in catalog.items()
        if _c == crop and other["pest"] != pest_a
        and other["symptoms"]
        and (not my_domain or other.get("domain", "") == my_domain)
    ]
    random.shuffle(same_crop_names)

    # 用真实相似边的另一端优先（同样要求同 domain）
    similar_names = []
    for e in edge_candidates:
        nb = e["tail"] if e["head"] == pest_a else e["head"]
        if nb and nb != pest_a:
            nb_cat = catalog.get((crop, nb))
            if nb_cat and (not my_domain or nb_cat.get("domain", "") == my_domain):
                similar_names.append(nb)
    seen = set(similar_names)
    for nb in same_crop_names:
        if nb not in seen:
            similar_names.append(nb)
            seen.add(nb)

    for pest_b in similar_names:
        key_b = (crop, pest_b)
        if key_b not in catalog:
            continue

        cat_b = catalog[key_b]
        ev_b = best_discriminative_evidence(cat_b)
        if not ev_b:
            continue

        unit_a, frag_a = ev_a
        unit_b, frag_b = ev_b

        # 两侧完全/近乎相同的文本不能形成有效鉴别点
        if char_jaccard(frag_a, frag_b) > 0.92:
            continue

        disp_a = display_pest(crop, pest_a)
        disp_b = display_pest(crop, pest_b)

        question = pick_template(
            DIFF_QUERY_TEMPLATES,
            crop=crop,
            pest_a=disp_a,
            pest_b=disp_b,
        )

        answer = (
            f"{pest_a}可重点观察：{frag_a}；"
            f"{pest_b}可重点观察：{frag_b}。"
            "两者应主要依据受害部位、症状/为害形态等证据进行鉴别。"
        )

        derived = [
            [pest_a, "典型症状", frag_a],
            [pest_b, "典型症状", frag_b],
        ]

        # 记录来源：真实 KG 相似边 或 同作物兜底
        kg_edge = next(
            (e for e in edge_candidates
             if e["tail"] == pest_b or e["head"] == pest_b),
            None,
        )
        similar_rel = kg_edge["rel"] if kg_edge else "同作物兜底"

        return make_qa(
            task="相似病虫害鉴别",
            crop=crop,
            pest=pest_a,
            question=question,
            answer=answer,
            evidence_units=[unit_a, unit_b],
            derived_relations=derived,
            gold_graph_evidence=[kg_edge] if kg_edge else [],
            metadata={
                "contrast_pest": disp_b,
                "contrast_pest_full": pest_b,
                "similar_relation": similar_rel,
            },
        )

    return None

def kg_control_edges(kg: CropPestKG, pest: str) -> List[Dict[str, str]]:
    """读取病虫害 -> 防治方法 的真实图谱边（防治节点实体类型对齐 evidence_entities）"""
    nid = kg.find_pest(pest)
    if not nid:
        return []
    out: List[Dict[str, str]] = []
    for rel_name in CONTROL_REL_NAMES:
        for tgt, rel in kg.neighbors(nid, rel_name):
            tgt_ent = kg.node_entity(tgt)
            if tgt_ent not in CONTROL_ENTITY_TYPES:
                continue
            tname = kg.node_name(tgt)
            if not tname or tname.lower() == "nan":
                continue
            out.append({
                "head": pest,
                "rel": rel,
                "tail": tname,
                "head_type": kg.node_entity(nid),
                "tail_type": tgt_ent,
                "triple_id": stable_id(pest, rel, tname, prefix="kg"),
                "source_id": "knowledge_graph",
            })
    return out


def kg_harm_site_edges(kg: CropPestKG, pest: str) -> List[Dict[str, str]]:
    """读取病虫害 -> 发病部位 的真实图谱边（为害/发病于/影响/直接引发/致使/处于）"""
    nid = kg.find_pest(pest)
    if not nid:
        return []
    out: List[Dict[str, str]] = []
    for rel_name in HARM_REL_NAMES:
        for tgt, rel in kg.neighbors(nid, rel_name):
            tgt_ent = kg.node_entity(tgt)
            if tgt_ent not in SITE_ENTITY_TYPES:
                continue
            tname = kg.node_name(tgt)
            if not tname or tname.lower() == "nan":
                continue
            out.append({
                "head": pest,
                "rel": rel,
                "tail": tname,
                "head_type": kg.node_entity(nid),
                "tail_type": tgt_ent,
                "triple_id": stable_id(pest, rel, tname, prefix="kg"),
                "source_id": "knowledge_graph",
            })
    return out


def dedupe_kg_edges(edges: List[Dict[str, str]]) -> List[Dict[str, str]]:
    seen = set()
    out = []
    for e in edges:
        key = (e["head"], e["rel"], e["tail"])
        if key in seen:
            continue
        seen.add(key)
        out.append(e)
    return out


def make_graph_diag(kg: CropPestKG, cat: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    图谱综合诊断（做法 A 核心）：
        症状 + 受害部位 -> 病害 + 危害部位 + 防治办法

    检索必需性来源：
    - 受害部位只能由“病虫害 -为害-> 发病部位”真实图谱边提供（文本/LLM 常识难覆盖）；
    - 防治办法优先选【该病虫害全部文本中不存在的图谱防治边】
      （即该病虫害的症状/条件/防治/机制段都没有这条防治，只有图谱边能提供）
      —— 保证 naive-text 纯文本检索拿不到参考答案里的防治节点；
    - 需要图谱多跳结构（病虫害 -> 部位 + 病虫害 -> 防治），naive-text 无图证据无法答全。
    """
    if not cat["symptoms"]:
        return None

    # 0) 该病虫害全部文本（判断图谱边是否文本缺失）
    all_text = " ".join(
        u["text"] for u in (cat["symptoms"] + cat["conditions"]
                            + cat["controls"] + cat["mechanisms"])
    )

    # 1) 症状证据（优先冷门细节：尺寸/颜色/形态，保证 LLM 常识答不出）
    cold_pool = [u for u in cat["symptoms"] if is_cold_detail(u["text"])]
    pool = cold_pool or cat["symptoms"]
    sym_unit = random.choice(pool)
    sym = symptom_fragment(sym_unit["text"], 60)
    if not sym or len(sym) < 6:
        return None

    crop = cat["crop"]
    pest = cat["pest"]
    site = extract_site_from_text(sym) or extract_site_from_text(sym_unit["text"])

    # 2) 图谱为害部位（真实 KG 边；优先“文本缺失”部位——naive 从文本拿不到）
    harm_edges = dedupe_kg_edges(kg_harm_site_edges(kg, pest))
    if not harm_edges:
        return None
    sites = [e["tail"] for e in harm_edges]
    missing_sites = [s for s in sites if s not in all_text]

    # 3) 图谱防治方法（真实 KG 边，取全部“文本缺失”的防治节点——
    #    这些节点只有图谱边能提供，naive-text 一个都拿不到）
    ctrl_edges = dedupe_kg_edges(kg_control_edges(kg, pest))
    if not ctrl_edges:
        return None
    missing_ctrl = [e for e in ctrl_edges if e["tail"] not in all_text]
    if not missing_ctrl:
        # 该病虫害的防治节点在文本里都有：图谱无增量信息，不构成本类问题
        return None
    # 冷门防治优先排序，保证参考答案里的防治节点 LLM 常识也答不出
    cold_missing = [e for e in missing_ctrl if is_cold_detail(e["tail"])]
    ordered_missing = sorted(missing_ctrl,
                             key=lambda e: (0 if e["tail"] in {x["tail"] for x in cold_missing} else 1,
                                            e["tail"]))
    ctrl_nodes = [e["tail"] for e in ordered_missing[:3]]  # 最多 3 个缺失防治节点

    # 4) 构造问题
    #    模板 1 不含 {site}（症状文本已自带部位时用，避免“叶片上出现叶片…”重复）；
    #    模板 2-4 显式含 {site}（症状文本不含部位时用，题干明确给出受害部位）。
    display_sym = clean_symptom_for_query(sym, crop=crop, site=site)
    if len(display_sym) < 4:
        return None
    if site:
        # 症状是否已含部位词
        site_already_in_sym = any(
            display_sym.startswith(a) for a in SITE_ALIASES.get(site, [site])
        )
    else:
        site_already_in_sym = False

    if site and not site_already_in_sym:
        question = pick_template(
            GRAPH_DIAG_QUERY_TEMPLATES[1:],
            crop=crop, site=site, symptom=display_sym,
        )
    else:
        question = GRAPH_DIAG_QUERY_TEMPLATES[0].format(
            crop=crop, symptom=display_sym,
        )

    # 5) 参考答案（病名 + 图谱独有部位 + 多个文本缺失防治节点；
    #     naive-text 纯文本检索拿不到缺失部位和缺失防治，只能靠图谱多跳）
    #    优先列出文本缺失部位（naive 完全拿不到），再补完整部位列表。
    missing_site_str = "、".join(missing_sites[:4])
    full_site_str = "、".join(sites[:6])
    if missing_site_str:
        site_part = f"它主要危害{missing_site_str}等部位"
    else:
        site_part = f"它主要危害{full_site_str}等部位"
    ctrl_part = "；防治可参考：" + "、".join(ctrl_nodes)
    answer = f"最可能是{pest}。{site_part}{ctrl_part}。"

    # gold_graph_evidence：只写真实 KG 边（为害部位 + 全部缺失防治方法）
    gold_graph_evidence = harm_edges[:8] + [e for e in ctrl_edges if e["tail"] in ctrl_nodes]
    # gold_evidence_ids：症状文本证据（与既有 gold_evidence_texts 结构一致）
    sym_payload = make_evidence_payload([sym_unit])
    derived = [
        [pest, "为害", s] for s in sites[:8]
    ] + [[pest, "防治方法有", t] for t in ctrl_nodes]

    return make_qa(
        task="多条件综合推理",
        crop=crop,
        pest=pest,
        question=question,
        answer=answer,
        evidence_units=[sym_unit],
        derived_relations=derived,
        gold_graph_evidence=gold_graph_evidence,
        metadata={
            "diag_mode": "graph_diag",
            "harm_sites": sites[:8],
            "missing_sites": missing_sites[:4],
            "control_nodes": ctrl_nodes,
        },
    )


def make_multi(cat: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    多条件综合推理：
        症状 + 发生条件 -> 病虫害判断

    关键约束：
    - 问题中不出现 gold pest；
    - 症状与条件优先来自同一 record；
    - 其次允许同一 source；
    - 默认不允许跨 source 强行拼接。
    """
    symptom_candidates = []
    for unit in cat["symptoms"]:
        frag = symptom_fragment(unit["text"], 55)
        if frag:
            symptom_candidates.append((unit, frag))

    condition_candidates = []
    for unit in cat["conditions"]:
        frag = condition_fragment(unit["text"], 50)
        if frag:
            condition_candidates.append((unit, frag))

    if not symptom_candidates or not condition_candidates:
        return None

    pair = choose_coherent_multi_pair(
        symptom_candidates,
        condition_candidates,
    )
    if not pair:
        return None

    sym_unit, sym, cond_unit, cond, coherence = pair

    crop = cat["crop"]
    pest = cat["pest"]
    site = extract_site_from_text(sym) or extract_site_from_text(sym_unit["text"])

    display_sym = clean_symptom_for_query(sym, crop=crop, site=site)
    display_cond = cond.replace(crop, "", 1).strip("，,。；; ")

    if len(display_sym) < 4 or len(display_cond) < 4:
        return None

    site_phrase = compose_site_phrase(site, display_sym, with_de=True)

    question = pick_template(
        MULTI_QUERY_TEMPLATES,
        crop=crop,
        site_phrase=site_phrase,
        symptom=display_sym,
        cond=display_cond,
    )

    answer = (
        f"综合症状和发生条件，最符合{pest}。"
        f"症状依据：{short(sym_unit['text'], 90)}；"
        f"条件依据：{short(cond_unit['text'], 90)}"
    )

    derived = [
        [pest, "典型症状", display_sym],
        [pest, "发生条件", display_cond],
    ]

    return make_qa(
        task="多条件综合推理",
        crop=crop,
        pest=pest,
        question=question,
        answer=answer,
        evidence_units=[sym_unit, cond_unit],
        derived_relations=derived,
        metadata={"evidence_coherence": coherence},
    )

# ============================================================
# 9. 自动质量控制
# ============================================================

LEAK_LABEL_WORDS = [
    "答案是",
    "正确答案",
    "gold",
]

BAD_GENERIC_PATTERNS = [
    r"出现出现",
    r"叶片出现叶片",
    r"叶鞘出现叶鞘",
    r"根部出现根部",
    r"茎秆出现茎秆",
    r"穗部出现穗部",
    r"果实出现果实",
    r"的的",
    r"，，",
    r"。。",
    r"；；",
]

# 对非防治任务，题干中不应出现明显的防治动作；
# 对症状/识别/多条件任务，也不应夹杂明显病原机制句。
CONTROL_WORDS_IN_QUERY = [
    "施药", "喷雾", "拌种", "浸种", "药剂", "防治", "轮作",
    "黄板", "诱杀", "可湿性粉剂",
]

MECHANISM_WORDS_IN_QUERY = [
    "菌丝", "分生孢子", "越冬", "初侵染", "再侵染", "羽化", "产卵",
]


def contains_gold_leak(qa: Dict[str, Any]) -> bool:
    """
    对“病虫害识别”和“多条件综合推理”：
    gold_pest 不允许直接出现在 question 中。
    """
    if qa["task"] in ("病虫害识别", "多条件综合推理"):
        pest = qa.get("gold_pest") or ""
        if pest and pest in qa["question"]:
            return True
    return False


def qa_quality_check(qa: Dict[str, Any]) -> Tuple[bool, List[str]]:
    reasons = []

    q = qa.get("question", "")
    a = qa.get("reference_answer", "")
    eids = qa.get("gold_evidence_ids", [])
    etexts = qa.get("gold_evidence_texts", [])

    if len(q) < 8:
        reasons.append("question_too_short")
    if len(a) < 6:
        reasons.append("answer_too_short")
    # 图谱必需题（graph_control / graph_diag）的证据是图谱边，允许无文本 gold evidence
    graph_required = (qa.get("metadata") or {}).get("diag_mode") in ("graph_control", "graph_diag")
    if not eids or not etexts:
        if not graph_required:
            reasons.append("missing_gold_evidence")
    if len(eids) != len(etexts):
        reasons.append("evidence_id_text_mismatch")

    if contains_gold_leak(qa):
        reasons.append("gold_label_leak")

    if any(w.lower() in q.lower() for w in LEAK_LABEL_WORDS):
        reasons.append("explicit_answer_hint")

    for p in BAD_GENERIC_PATTERNS:
        if re.search(p, q):
            reasons.append(f"bad_pattern:{p}")

    # 非防治任务避免把防治段混入问题（图谱综合诊断除外：它本来就要问防治）
    if qa["task"] != "防治措施推荐" and qa.get("metadata", {}).get("diag_mode") != "graph_diag":
        if any(w in q for w in CONTROL_WORDS_IN_QUERY):
            reasons.append("control_content_in_noncontrol_question")

    if qa["task"] in ("病虫害识别", "症状解析", "多条件综合推理"):
        if any(w in q for w in MECHANISM_WORDS_IN_QUERY):
            reasons.append("mechanism_noise_in_question")

    # 相似鉴别必须：
    # 1) 含对照对象；
    # 2) 具有真实 KG 相似边 或 明确标记为同作物兜底；
    # 3) 至少两条文本证据。
    if qa["task"] == "相似病虫害鉴别":
        contrast = (qa.get("metadata") or {}).get("contrast_pest")
        if not contrast or contrast not in q:
            reasons.append("missing_contrast_pest")
        sim_rel = (qa.get("metadata") or {}).get("similar_relation", "")
        has_kg_edge = bool(qa.get("gold_graph_evidence"))
        # 同作物兜底样本允许无真实 KG 边，但必须显式标记
        if not has_kg_edge and sim_rel != "同作物兜底":
            reasons.append("missing_real_kg_similarity_edge")
        if len(eids) < 2:
            reasons.append("insufficient_differential_evidence")

    # 多条件任务必须有两条互补证据，并满足同记录/同来源约束
    if qa["task"] == "多条件综合推理":
        if qa.get("metadata", {}).get("diag_mode") == "graph_diag":
            # 图谱综合诊断：文本证据 = 症状；部位/防治由图谱边承载，
            # 因此不再要求两条文本证据/同源约束（相关要求见下方 graph_diag 专项检查）。
            pass
        else:
            if len(eids) < 2:
                reasons.append("insufficient_multi_evidence")
            coherence = (qa.get("metadata") or {}).get("evidence_coherence")
            allowed = {"same_record", "same_source"}
            if ALLOW_CROSS_SOURCE_MULTI:
                allowed.add("cross_source")
            if coherence not in allowed:
                reasons.append("incoherent_multi_evidence")

    # 图谱综合诊断（做法 A）必须：
    # 1) 有真实 KG 图谱证据（为害部位 + 防治方法）；
    # 2) 答案为害部位、防治节点分别覆盖图谱边；
    # 3) 问题不含 gold pest。
    if qa.get("metadata", {}).get("diag_mode") == "graph_diag":
        gge = qa.get("gold_graph_evidence", [])
        harm = [g for g in gge if g.get("rel") in HARM_REL_NAMES]
        ctrl = [g for g in gge if g.get("rel") in CONTROL_REL_NAMES]
        if not harm:
            reasons.append("graph_diag_missing_harm_site_edge")
        if not ctrl:
            reasons.append("graph_diag_missing_control_edge")
        meta = qa.get("metadata", {})
        # 答案只需覆盖答案里实际列出的部位（missing_sites 优先），
        # 不需要覆盖全部 harm_sites（那是图谱全量边）。
        ans_sites = meta.get("missing_sites") or (meta.get("harm_sites") or [])[:4]
        for s in ans_sites[:4]:
            if s and s not in a:
                reasons.append("graph_diag_answer_missing_site")
                break
        for t in (meta.get("control_nodes") or [])[:3]:
            if t and t not in a:
                reasons.append("graph_diag_answer_missing_control")
                break
        if qa.get("gold_pest") and qa["gold_pest"] in q:
            reasons.append("gold_label_leak")

    # 图谱必需防治推荐（做法 B）：答案必须覆盖图谱缺失防治节点
    # 注意：防治推荐问题题干需要含病虫害名（"XX发生Y后怎么防治"），
    # 因此不做 gold_label_leak 检查（与 graph_diag 不同）。
    if qa.get("metadata", {}).get("diag_mode") == "graph_control":
        ctrl_nodes = (qa.get("metadata") or {}).get("control_nodes") or []
        for t in ctrl_nodes[:4]:
            if t and t not in a:
                reasons.append("graph_control_answer_missing_node")
                break
        if not qa.get("gold_graph_evidence"):
            reasons.append("graph_control_missing_edge")
    return (len(reasons) == 0), reasons

def dedupe_candidates(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    两级去重：
    1) 完全规范化问题去重
    2) 同 task + crop + gold_pest + 相同 gold evidence 去重
    """
    seen_q = set()
    seen_sem = set()
    out = []

    for qa in candidates:
        nq = norm_question(qa["question"])
        # 图谱必需题（graph_control / graph_diag）的区分键加入 gold_graph_evidence，
        # 避免空文本证据导致同一病虫害的多个候选被误判为重复。
        gge_key = tuple(sorted(
            (g.get("head", ""), g.get("rel", ""), g.get("tail", ""))
            for g in qa.get("gold_graph_evidence", [])
        ))
        sem = (
            qa["task"],
            qa.get("crop"),
            qa.get("gold_pest"),
            tuple(sorted(qa.get("gold_evidence_ids", []))),
            gge_key,
            (qa.get("metadata") or {}).get("contrast_pest", ""),
        )

        if nq in seen_q:
            continue
        if sem in seen_sem:
            continue

        seen_q.add(nq)
        seen_sem.add(sem)
        out.append(qa)

    return out


# ============================================================
# 10. 候选生成
# ============================================================

def generate_candidates(
    kg: CropPestKG,
    catalog: Dict[Tuple[str, str], Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    per_task: Dict[str, List[Dict[str, Any]]] = {t: [] for t in TASKS}

    items = list(catalog.values())
    random.shuffle(items)

    for cat in items:
        for _ in range(MAX_CANDIDATES_PER_ITEM_PER_TASK):
            gens = [
                ("病虫害识别", lambda: make_identification(cat)),
                ("症状解析", lambda: make_symptom_parse(cat)),
                ("条件判断", lambda: make_condition(cat)),
                ("防治措施推荐", lambda: make_control_graph(kg, cat) or make_control(cat)),
                ("相似病虫害鉴别", lambda: make_similar(kg, cat, catalog)),
                ("多条件综合推理", lambda: make_graph_diag(kg, cat) or make_multi(cat)),
            ]

            for task, fn in gens:
                try:
                    qa = fn()
                except Exception as e:
                    # 单条异常不终止整个 benchmark 生成
                    print(f"[WARN] 生成失败：task={task}, crop={cat['crop']}, pest={cat['pest']}, err={e}")
                    qa = None

                if qa:
                    per_task[task].append(qa)

    # 先质量过滤，再去重 —— 最终采样只从 clean pool 中取
    clean: Dict[str, List[Dict[str, Any]]] = {}
    for task in TASKS:
        tmp = []
        rejected = Counter()

        for qa in per_task[task]:
            ok, reasons = qa_quality_check(qa)
            if ok:
                tmp.append(qa)
            else:
                for r in reasons:
                    rejected[r] += 1

        tmp = dedupe_candidates(tmp)
        random.shuffle(tmp)
        clean[task] = tmp

        print(
            f"[QC] {task}: raw={len(per_task[task])}, "
            f"clean={len(tmp)}, rejected={sum(rejected.values())}"
        )
        if rejected:
            print("     reject reasons:", dict(rejected))

    return clean


# ============================================================
# 11. 均衡采样
# ============================================================

def balanced_sample(
    clean: Dict[str, List[Dict[str, Any]]],
    target_per_task: int = TARGET_PER_TASK,
) -> List[Dict[str, Any]]:
    # 每任务目标数：优先 TARGET_BY_TASK（图谱必需任务扩占比），否则用统一 target_per_task
    targets = {t: TARGET_BY_TASK.get(t, target_per_task) for t in TASKS}

    shortages = {
        task: targets[task] - len(clean[task])
        for task in TASKS
        if len(clean[task]) < targets[task]
    }

    if shortages:
        msg = (
            "以下任务候选不足，不能严格构造均衡 benchmark：\n"
            + "\n".join(
                f"  - {task}: 现有 {len(clean[task])}, 需要 {targets[task]}, 缺 {gap}"
                for task, gap in shortages.items()
            )
        )
        if STRICT_BALANCE:
            raise RuntimeError(msg)
        else:
            print("[WARN]", msg)

    selected = []
    for task in TASKS:
        n = min(targets[task], len(clean[task]))
        # 优先采样图谱必需题（graph_control / graph_diag）：
        # 这些题的答案依赖图谱独有信息，naive-text 无法从纯文本检索答出，
        # 保证 full 方法在评测集中有足够占比来体现图谱检索价值。
        task_pool = list(clean[task])
        task_pool.sort(
            key=lambda qa: 0 if (qa.get("metadata") or {}).get("diag_mode") in ("graph_control", "graph_diag") else 1
        )
        selected.extend(task_pool[:n])

    random.shuffle(selected)
    return selected


# ============================================================
# 12. 统计与审计
# ============================================================

def print_stats(selected: List[Dict[str, Any]]) -> None:
    print("\n========== Benchmark statistics ==========")
    print("总样本数:", len(selected))

    task_cnt = Counter(q["task"] for q in selected)
    crop_cnt = Counter(q["crop"] for q in selected)

    print("\n任务分布：")
    for t in TASKS:
        print(f"  {t}: {task_cnt.get(t, 0)}")

    print("\n作物覆盖数:", len(crop_cnt))
    print("Top-10 作物：", crop_cnt.most_common(10))

    ev_cnt = Counter(len(q.get("gold_evidence_ids", [])) for q in selected)
    print("\n每题 gold text evidence 数量分布：", dict(sorted(ev_cnt.items())))

    graph_gold_cnt = sum(
        1 for q in selected if q.get("gold_graph_evidence")
    )
    print("含真实 gold graph evidence 的题数:", graph_gold_cnt)

    coherence_cnt = Counter(
        (q.get("metadata") or {}).get("evidence_coherence")
        for q in selected
        if q["task"] == "多条件综合推理"
    )
    if coherence_cnt:
        print("多条件证据一致性：", dict(coherence_cnt))

    leak = [q["id"] for q in selected if contains_gold_leak(q)]
    print("Gold label leakage:", len(leak))
    if leak:
        print("  examples:", leak[:10])

    q_norms = [norm_question(q["question"]) for q in selected]
    dup_num = len(q_norms) - len(set(q_norms))
    print("Duplicate questions:", dup_num)

    pending = sum(1 for q in selected if q.get("review_status") == "pending")
    print("Pending manual review:", pending)
    print("==========================================\n")

# ============================================================
# 13. 主程序
# ============================================================

def main():
    random.seed(RANDOM_SEED)

    kg = CropPestKG(NODES_CSV, RELS_CSV)

    with open(EXTRACT_JSON, encoding="utf-8") as f:
        recs = json.load(f)

    recs = [
        r for r in recs
        if (r.get("pest") or "").strip()
        and (r.get("pest") or "").strip().lower() != "nan"
        and (r.get("crop") or "").strip()
    ]

    catalog = build_pest_catalog(recs)

    print(f"原始记录数：{len(recs)}")
    print(f"(crop, pest) 素材条目：{len(catalog)}")

    clean = generate_candidates(kg, catalog)

    print("\n清洗后各类可用候选：")
    for t in TASKS:
        print(f"  {t}: {len(clean[t])}")

    selected = balanced_sample(clean, TARGET_PER_TASK)

    # 最终再做一次全局问题去重审计
    final_seen = set()
    final = []
    for qa in selected:
        k = norm_question(qa["question"])
        if k in final_seen:
            continue
        final_seen.add(k)
        final.append(qa)

    if STRICT_BALANCE and len(final) != TOTAL_TARGET:
        raise RuntimeError(
            f"最终全局去重后仅剩 {len(final)} 条，"
            f"未达到目标 {TOTAL_TARGET} 条。"
        )

    # 没有真实人工审核记录时，生成器不得自动把 pending 改成 approved
    if any(qa.get("review_status") != DEFAULT_REVIEW_STATUS for qa in final):
        raise RuntimeError(
            "生成器不得自动修改 review_status；"
            "approved 只能由人工审核流程写入。"
        )

    # 固定 benchmark 版本信息
    for qa in final:
        qa["benchmark_version"] = "v4"
        qa["random_seed"] = RANDOM_SEED

    os.makedirs(os.path.dirname(EVAL_FILE) or ".", exist_ok=True)
    with open(EVAL_FILE, "w", encoding="utf-8") as f:
        json.dump(final, f, ensure_ascii=False, indent=2)

    print_stats(final)
    print(f"评测集已生成：{EVAL_FILE}")
    print(
        "注意：当前 review_status 均为 pending。"
        "建议人工审核后再冻结为论文最终 benchmark。"
    )


if __name__ == "__main__":
    main()
