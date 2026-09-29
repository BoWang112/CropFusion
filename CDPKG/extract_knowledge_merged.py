# -*- coding: utf-8 -*-
"""
基于统一模型的作物病虫害知识抽取流水线
============================================
两阶段端到端知识抽取（统一模型，病虫害合并训练）：
  阶段1（NER）：RoBERTa-BiLSTM-CRF 识别实体（作物/病虫害/病原/症状等，含类型与字符位置）
  阶段2（RE）： RoBERTa-CoPET-CasRel 抽取关系三元组 (头实体, 关系, 尾实体)

用法：
  1) 单条文本抽取
     python extract_knowledge.py --text "水稻稻瘟病主要为害叶片，由灰梨孢菌引起。"
  2) 抽取 Excel 知识库中指定列的长文本
     python extract_knowledge.py --excel 病害知识库.xlsx --columns 病害症状 发病规律 防治方法
  3) 结果默认输出到 results/ 目录

依赖环境：
  阶段1 需要 .venv-ner（transformers + torchcrf）
  阶段2 需要 .venv-re（bert4torch）
  由于两个模型依赖不同环境，本脚本需要在这两个 venv 中都能运行，
  实际执行时请使用各自的解释器，或按需调整模型路径。
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys

import torch

# ---------------------------------------------------------------------------
# 路径配置（按需修改）
# ---------------------------------------------------------------------------
NER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "命名实体识别")
RE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "关系抽取")

# NER checkpoint：统一病虫害模型
NER_CKPT_CANDIDATES = [
    os.path.join(NER_DIR, "checkpoint_merged811"),
]
# RE checkpoint：统一病虫害模型
RE_CKPT_CANDIDATES = [
    os.path.join(RE_DIR, "checkpoint_merged811", "best_model.pt"),
]

MAX_SEQ_LEN = 256  # NER 最大序列长度（与训练一致）
MAX_SEQ_LEN_RE = 512  # RE 最大序列长度


def _pick_first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None


# ---------------------------------------------------------------------------
# 阶段1：NER（RoBERTa-BiLSTM-CRF）
# ---------------------------------------------------------------------------
class NERExtractor:
    """封装 NER 模型推理，输出字符级实体 (text, type, start, end)"""

    def __init__(self, ckpt_dir: str | None = None, device: str | None = None):
        self.ckpt_dir = ckpt_dir or _pick_first_existing(NER_CKPT_CANDIDATES)
        if self.ckpt_dir is None:
            raise FileNotFoundError("未找到 NER checkpoint，请检查 NER_CKPT_CANDIDATES 路径")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # 动态导入（依赖 .venv-ner 环境）
        sys.path.insert(0, NER_DIR)
        from transformers import BertTokenizer
        from models import BERT_BiLSTM_CRF

        args = torch.load(os.path.join(self.ckpt_dir, "training_args.bin"),
                          map_location="cpu", weights_only=False)
        self.label2id = pickle.load(open(os.path.join(self.ckpt_dir, "label2id.pkl"), "rb"))
        self.id2label = {v: k for k, v in self.label2id.items()}
        self.tokenizer = BertTokenizer.from_pretrained(self.ckpt_dir,
                                                       do_lower_case=args.do_lower_case)
        self.model = BERT_BiLSTM_CRF.from_pretrained(
            self.ckpt_dir, need_birnn=args.need_birnn, rnn_dim=args.rnn_dim)
        self.model.to(self.device)
        self.model.eval()
        print(f"[NER] 已加载模型: {self.ckpt_dir} (need_birnn={args.need_birnn})")

    def _segment_text(self, text: str):
        """将长文本切分为 <= MAX_SEQ_LEN 的句子段（按标点切分）"""
        if len(text) <= MAX_SEQ_LEN - 2:
            return [text]
        # 按中文标点切分，尽量保持语义完整
        segments, cur = [], ""
        for ch in text:
            cur += ch
            if ch in "。；！？\n" or len(cur) >= MAX_SEQ_LEN - 2:
                segments.append(cur)
                cur = ""
        if cur:
            segments.append(cur)
        return segments

    def extract(self, text: str) -> list[dict]:
        """识别实体，返回 [{'text', 'type', 'start', 'end'}]（字符级，相对整段文本）"""
        chars = list(text)
        tokens = ["[CLS]"] + chars + ["[SEP]"]
        if len(tokens) > MAX_SEQ_LEN:
            tokens = tokens[: MAX_SEQ_LEN - 1] + ["[SEP]"]
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        while len(input_ids) < MAX_SEQ_LEN:
            input_ids.append(0)
            input_mask.append(0)
        input_ids = torch.tensor([input_ids], device=self.device)
        input_mask = torch.tensor([input_mask], device=self.device)
        with torch.no_grad():
            preds = self.model.predict(input_ids, input_mask=input_mask)
        tags = [self.id2label[idx] for idx in preds[0]]

        entities, cur = [], None
        real_len = min(len(chars), len(tags) - 2)  # 去掉 CLS/SEP
        for i in range(real_len):
            tag = tags[i + 1]  # 跳过 [CLS]
            char_idx = i
            if tag.startswith("B-"):
                if cur:
                    entities.append(cur)
                cur = {"text": chars[char_idx], "type": tag[2:],
                       "start": char_idx, "end": char_idx}
            elif tag.startswith("I-"):
                if cur and cur["type"] == tag[2:]:
                    cur["text"] += chars[char_idx]
                    cur["end"] = char_idx
                else:
                    if cur:
                        entities.append(cur)
                    cur = {"text": chars[char_idx], "type": tag[2:],
                           "start": char_idx, "end": char_idx}
            else:  # O
                if cur:
                    entities.append(cur)
                    cur = None
        if cur:
            entities.append(cur)
        return entities


# ---------------------------------------------------------------------------
# 阶段2：RE（RoBERTa-CoPET-CasRel）
# ---------------------------------------------------------------------------
class REExtractor:
    """封装 RE 模型推理，输入实体特征，输出三元组"""

    def __init__(self, ckpt_path: str | None = None, device: str | None = None):
        self.ckpt_path = ckpt_path or _pick_first_existing(RE_CKPT_CANDIDATES)
        if self.ckpt_path is None:
            raise FileNotFoundError("未找到 RE checkpoint，请检查 RE_CKPT_CANDIDATES 路径")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # 动态导入主脚本（依赖 .venv-re 环境）
        sys.path.insert(0, RE_DIR)
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "rc_main", os.path.join(RE_DIR, "roberta_copet_casrel_merged811.py"))
        self.rc = importlib.util.module_from_spec(spec)
        # 主脚本使用相对路径（chinese-roberta-wwm-ext/），导入和模型构建期间保持在 RE 目录
        self._saved_cwd = os.getcwd()
        os.chdir(RE_DIR)
        try:
            spec.loader.exec_module(self.rc)
            self.tokenizer = self.rc.tokenizer
            self.entity_type2id = self.rc.entity_type2id
            self.id2predicate = self.rc.id2predicate
            self.maxlen = self.rc.maxlen
            self.model = self.rc.Model().to(self.device)
            self.model.load_weights(self.ckpt_path)
            self.model.eval()
        finally:
            os.chdir(self._saved_cwd)
        print(f"[RE] 已加载模型: {self.ckpt_path}")

    def extract(self, text: str, entities: list[dict], forced_subject: str | None = None) -> list[tuple]:
        """用实体特征抽三元组，返回 [(头实体, 关系, 尾实体)]

        forced_subject: 若给定（如病虫害名），则只以该实体为主语做条件宾语预测，
        避免 CasRel 把所有名词短语都当主语产生的噪声。
        """
        token_ids, segment_ids = self.tokenizer.encode(text, maxlen=self.maxlen)
        entity_type2id = self.entity_type2id

        entity_types = torch.zeros(len(token_ids), dtype=torch.long)
        start_pos = torch.zeros(len(token_ids), dtype=torch.long)
        end_pos = torch.zeros(len(token_ids), dtype=torch.long)

        mapping = self.tokenizer.rematch(
            text, self.tokenizer.tokenize(text, maxlen=self.maxlen))
        for ent in entities:
            e_type = ent["type"]
            if e_type not in entity_type2id:
                continue
            tid = entity_type2id[e_type]
            c_start, c_end = ent["start"], ent["end"]
            tok_start = tok_end = None
            for ti, spans in enumerate(mapping):
                if spans and spans[0] <= c_start <= spans[-1]:
                    tok_start = ti
                    break
            for ti, spans in enumerate(mapping):
                if spans and spans[-1] >= c_end:
                    tok_end = ti
                    break
            if tok_start is None or tok_end is None:
                continue
            entity_types[tok_start:tok_end + 1] = tid
            start_pos[tok_start] = tid + 1
            end_pos[tok_end] = tid + 1

        device = self.device
        inputs = [torch.tensor([token_ids], device=device),
                  torch.tensor([segment_ids], device=device),
                  entity_types.unsqueeze(0).to(device),
                  start_pos.unsqueeze(0).to(device),
                  end_pos.unsqueeze(0).to(device)]
        seq_output, subject_preds = self.model.predict_subject(inputs)

        subjects = []
        if forced_subject:
            # 方案B：强制病虫害名为主语，在 token 序列中定位其位置
            dis_tok_ids = self.tokenizer.encode(forced_subject)[0][1:-1]
            for i in range(len(token_ids)):
                if token_ids[i:i + len(dis_tok_ids)] == dis_tok_ids:
                    subjects.append((i, i + len(dis_tok_ids) - 1))
                    break
        else:
            # 默认：CasRel 检测所有潜在主语
            subject_preds[:, [0, -1]] *= 0
            start = torch.where(subject_preds[0, :, 0] > 0.6)[0]
            end = torch.where(subject_preds[0, :, 1] > 0.5)[0]
            for i in start:
                j = end[end >= i]
                if len(j) > 0:
                    subjects.append((i.item(), j[0].item()))

        spoes = []
        if subjects:
            seq_output = seq_output.repeat([len(subjects)] + [1] * (len(seq_output.shape) - 1))
            subjects_t = torch.tensor(subjects, dtype=torch.long, device=device)
            object_preds = self.model.predict_object([seq_output, subjects_t])
            object_preds[:, [0, -1]] *= 0
            for subject, object_pred in zip(subjects_t, object_preds):
                start_o = torch.where(object_pred[:, :, 0] > 0.6)
                end_o = torch.where(object_pred[:, :, 1] > 0.5)
                for _start, pred1 in zip(*start_o):
                    for _end, pred2 in zip(*end_o):
                        if _start <= _end and pred1 == pred2:
                            s_char = (mapping[subject[0]][0], mapping[subject[1]][-1])
                            o_char = (mapping[_start][0], mapping[_end][-1])
                            if s_char[0] <= s_char[1] and o_char[0] <= o_char[1]:
                                spoes.append((
                                    text[s_char[0]:s_char[1] + 1],
                                    self.id2predicate[pred1.item()],
                                    text[o_char[0]:o_char[1] + 1],
                                ))
                            break
        return spoes


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def _clean_prefix(arg: str, crop: str = "", pest: str = "") -> str:
    """清洗被前缀污染的三元组实体（去掉 '作物：X，病虫害：Y' 部分）"""
    arg = str(arg or "")
    patterns = [
        f"作物：{crop}，病虫害：{pest}",
        f"作物：{crop}，病害：{pest}",
        f"作物：{crop}，虫害：{pest}",
        f"{crop}，病虫害：{pest}",
        f"{crop}，病害：{pest}",
        f"{crop}，虫害：{pest}",
        f"作物：{crop}，",
        f"病虫害：{pest}",
        f"病害：{pest}",
        f"虫害：{pest}",
        f"{crop}，{pest}",
        "病虫害：",
        "病害：",
        "虫害：",
        "作物：",
    ]
    for pat in patterns:
        if arg.startswith(pat):
            arg = arg[len(pat):]
            break
    return arg.strip("，,、")


def run_pipeline(ner: NERExtractor, re: REExtractor, text: str, crop: str = "", pest: str = "") -> dict:
    """对单段文本执行 NER + RE 全流程（方案B：病虫害主语 + 方案A：自由主语 合并）"""
    entities = ner.extract(text)
    spoes = []
    # 方案B：强制病虫害名为主语
    if pest and pest in text:
        spoes += re.extract(text, entities, forced_subject=pest)
    # 方案A：自由主语
    spoes += re.extract(text, entities, forced_subject=None)
    # 去重
    seen = set()
    dedup = []
    for s, p, o in spoes:
        key = (s, p, o)
        if key not in seen:
            seen.add(key)
            dedup.append(key)
    triples = []
    for s, p, o in dedup:
        s_clean = _clean_prefix(s, crop, pest)
        o_clean = _clean_prefix(o, crop, pest)
        if not s_clean or not o_clean:
            continue
        triples.append({"subject": s_clean, "predicate": p, "object": o_clean})
    return {
        "text": text,
        "entities": entities,
        "triples": triples,
    }


def main():
    parser = argparse.ArgumentParser(description="作物病虫害知识抽取流水线（NER + RE）")
    parser.add_argument("--text", type=str, default=None, help="单条文本")
    parser.add_argument("--excel", type=str, default=None, help="Excel 知识库路径")
    parser.add_argument("--columns", type=str, default="病害症状,发病规律,防治方法",
                        help="要抽取的 Excel 列名（逗号分隔）")
    parser.add_argument("--output", type=str, default="results", help="输出目录")
    parser.add_argument("--ner_ckpt", type=str, default=None, help="NER checkpoint 目录")
    parser.add_argument("--re_ckpt", type=str, default=None, help="RE checkpoint 文件")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    if args.text is None and args.excel is None:
        parser.error("请提供 --text 或 --excel")

    ner = NERExtractor(args.ner_ckpt)
    re = REExtractor(args.re_ckpt)

    if args.text:
        result = run_pipeline(ner, re, args.text)
        print("\n" + "=" * 60)
        print("文本:", result["text"])
        print("\n识别实体:")
        for e in result["entities"]:
            print(f"  {e['text']} [{e['type']}] ({e['start']}-{e['end']})")
        print("\n抽取三元组:")
        for t in result["triples"]:
            print(f"  ({t['subject']}, {t['predicate']}, {t['object']})")
        out = os.path.join(args.output, "result_single.json")
        with open(out, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\n结果已保存: {out}")

    if args.excel:
        try:
            import pandas as pd
        except ImportError:
            print("需要 pandas，请安装: .venv-ner/Scripts/pip install pandas openpyxl")
            return
        df = pd.read_excel(args.excel)
        # 从知识库文件名判断领域（病害/虫害），统一写入 抽取结果.json
        excel_name = os.path.basename(args.excel)
        domain = '虫害' if '虫害' in excel_name else '病害'
        cols = [c.strip() for c in args.columns.split(",") if c.strip()]
        cols = [c for c in cols if c in df.columns]
        if not cols:
            print(f"指定的列不存在。可用列: {list(df.columns)}")
            return
        out = os.path.join(args.output, "抽取结果.json")

        # 断点续跑：若已有结果文件，从已有行数继续
        all_results = []
        start_row = 0
        if os.path.exists(out):
            try:
                all_results = json.load(open(out, encoding="utf-8"))
                # 仅统计本领域已有记录数，避免两个知识库互相干扰
                start_row = len([r for r in all_results if r.get('domain') == domain])
                all_results = [r for r in all_results if r.get('domain') != domain]
                print(f"发现已有结果 {start_row} 行，从第 {start_row + 1} 行继续")
            except Exception:
                print("已有结果文件损坏，重新开始")

        total = len(df)
        for idx in range(start_row, total):
            row = df.iloc[idx]
            crop = str(row.get("作物名") or "").strip()
            pest = str(row.get("病虫害名称") or "").strip()
            record = {"row": idx + 1, "crop": crop, "pest": pest, "domain": domain}
            # 注入主语前缀（模拟 RE 训练数据格式，为关系抽取提供主语）
            prefix = f"作物：{crop}，病虫害：{pest}，"
            for col in cols:
                cell = row[col]
                if not isinstance(cell, str) or not cell.strip():
                    continue
                # 长文本按句切分后逐段抽取，每段前加前缀提供主语
                segments = ner._segment_text(cell)
                for seg in segments:
                    if len(seg) < 4:
                        continue
                    full = prefix + seg
                    res = run_pipeline(ner, re, full, crop=crop, pest=pest)
                    if res["entities"] or res["triples"]:
                        record.setdefault("segments", []).append(res)
            all_results.append(record)
            if (idx + 1) % 50 == 0:
                print(f"  进度: {idx + 1}/{total}，临时保存...")
                with open(out, "w", encoding="utf-8") as f:
                    json.dump(all_results, f, ensure_ascii=False, indent=2)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"\nExcel 抽取完成，共 {len(all_results)} 条记录，结果已保存: {out}")


if __name__ == "__main__":
    main()
