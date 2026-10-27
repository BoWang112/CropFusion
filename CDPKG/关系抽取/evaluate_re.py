# -*- coding: utf-8 -*-
"""
evaluate_re.py —— 关系抽取统一评价入口
========================================
对真实模型（或外部原始基线）输出的 SPO 预测做统一的精确匹配评价：

  Precision / Recall / F1（micro，按 SPO 集合）
  每个关系的 P/R/F1
  sentence-level exact match
  缺失预测条数（缺失按空预测计入，不静默跳过）

支持 gold/pred 两种输入格式：
  - JSON list / 单个 JSON 对象（例如包含 spo_list / spo_list_pred 的预测记录）
  - 连续 JSON 对象流（每行一个对象；仓库内
    checkpoint_merged811/bert_casrel_pred_0.7.json 即此格式）
  - JSONL（每行一个样本）

字段名可显式指定：
  --gold-field triple_list / spo_list / triples ...
  --pred-field spo_list_pred / triple_list_pred / spo_list ...

用法：
  # 检查各基线是否已关联真实实现/预测文件
  python evaluate_re.py --check

  # 评价仓库内真实模型预测
  python evaluate_re.py --baseline roberta_copet_casrel \
      --gold checkpoint_merged811/bert_casrel_pred_0.7.json \
      --gold-field spo_list --pred-field spo_list_pred

  # CopyRE/SpERT/TPLinker/OneRel/CasRel 需要外部原始预测文件：
  CDPKG_COPYRE_PRED=/path/copyre_pred.json \
  python evaluate_re.py --baseline copyre --gold gold.json
"""
import os
import sys
import json
import argparse
from collections import defaultdict

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)

from baselines_config import (  # noqa: E402
    BASELINES, describe_baseline, resolve_prediction,
)


GOLD_FIELDS = ('triple_list', 'spo_list', 'triples', 'gold_triples')
PRED_FIELDS = ('spo_list_pred', 'triple_list_pred', 'pred_triples',
               'spo_list', 'triples')


def _parse_json_stream(text):
    """解析连续 JSON 对象流（每行一个对象，或 pretty-print 后首尾相接）。

    checkpoint_merged811/bert_casrel_pred_0.7.json 由训练脚本以
    json.dumps(...) + '\\n' 逐条写入，扩展名虽是 .json，内容并非单个
    合法 JSON 数组/对象；这里用 raw_decode 逐对象解析，保证真实预测文件
    可以直接进入统一评价入口。
    """
    decoder = json.JSONDecoder()
    records = []
    idx = 0
    length = len(text)
    while idx < length:
        while idx < length and (text[idx].isspace() or text[idx] in ',;'):
            idx += 1
        if idx >= length:
            break
        obj, idx = decoder.raw_decode(text, idx)
        records.append(obj)
    return records


def load_records(path):
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f'文件不存在：{path}')
    with open(path, encoding='utf-8') as f:
        text = f.read()
    if not text.strip():
        return []

    # JSONL：先按行解析（纯单行对象）
    if path.lower().endswith('.jsonl'):
        records = []
        ok = True
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                ok = False
                break
        if ok:
            return records
        return _parse_json_stream(text)

    # .json：先按标准 JSON 解析；失败则按连续对象流解析
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return _parse_json_stream(text)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ('data', 'records', 'predictions'):
            if isinstance(data.get(key), list):
                return data[key]
        return [data]
    raise ValueError(f'无法识别的记录格式：{path}')


def _normalize_text(text):
    return ' '.join(str(text or '').strip().lower().split())


def _as_triple(item):
    """兼容 dict / list / tuple 的 (subject, predicate, object)。"""
    if isinstance(item, dict):
        subject = item.get('subject', item.get('head', item.get('s')))
        predicate = item.get('predicate', item.get('rel', item.get('relation', item.get('p'))))
        obj = item.get('object', item.get('tail', item.get('o')))
    elif isinstance(item, (list, tuple)) and len(item) >= 3:
        subject, predicate, obj = item[0], item[1], item[2]
    else:
        return None
    subject = _normalize_text(subject)
    predicate = _normalize_text(predicate)
    obj = _normalize_text(obj)
    if not (subject and predicate and obj):
        return None
    return subject, predicate, obj


def extract_triples(record, fields):
    """按字段优先级读取 SPO 集合；返回 set[(s,p,o)]。"""
    for field in fields:
        if field not in record:
            continue
        value = record.get(field)
        if value is None:
            continue
        triples = set()
        for item in value:
            triple = _as_triple(item)
            if triple:
                triples.add(triple)
        return triples
    return set()


def _record_key(record, keys=('id', 'text')):
    for key in keys:
        value = record.get(key)
        if value is not None and str(value).strip():
            return key, str(value).strip()
    return None, None


def evaluate(gold_records, pred_records, gold_fields, pred_fields):
    """统一微平均评价；缺失预测按空集计入。"""
    pred_by_id = {}
    pred_by_text = {}
    for rec in pred_records:
        rid = rec.get('id')
        if rid is not None and str(rid).strip():
            pred_by_id[str(rid).strip()] = rec
        text = rec.get('text')
        if text is not None and str(text).strip():
            pred_by_text[_normalize_text(text)] = rec

    overall_tp = overall_fp = overall_fn = 0
    gold_total = pred_total = 0
    per_rel = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
    exact = 0
    missing = 0
    matched = 0

    for gold in gold_records:
        gold_id = gold.get('id')
        gold_text = gold.get('text')
        pred = None
        if gold_id is not None and str(gold_id).strip():
            pred = pred_by_id.get(str(gold_id).strip())
        if pred is None and gold_text is not None and str(gold_text).strip():
            pred = pred_by_text.get(_normalize_text(gold_text))
        if pred is None:
            missing += 1
        else:
            matched += 1
        gold_set = extract_triples(gold, gold_fields)
        pred_set = extract_triples(pred, pred_fields) if pred else set()
        gold_total += len(gold_set)
        pred_total += len(pred_set)
        if gold_set == pred_set:
            exact += 1
        tp_set = gold_set & pred_set
        fp_set = pred_set - gold_set
        fn_set = gold_set - pred_set
        overall_tp += len(tp_set)
        overall_fp += len(fp_set)
        overall_fn += len(fn_set)
        for _s, rel, _o in tp_set:
            per_rel[rel]['tp'] += 1
        for _s, rel, _o in fp_set:
            per_rel[rel]['fp'] += 1
        for _s, rel, _o in fn_set:
            per_rel[rel]['fn'] += 1

    def _prf(tp, fp, fn):
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return precision, recall, f1

    precision, recall, f1 = _prf(overall_tp, overall_fp, overall_fn)
    rel_metrics = {}
    for rel, c in sorted(per_rel.items()):
        p, r, f = _prf(c['tp'], c['fp'], c['fn'])
        rel_metrics[rel] = {
            'precision': p, 'recall': r, 'f1': f,
            'tp': c['tp'], 'fp': c['fp'], 'fn': c['fn'],
        }
    n = len(gold_records)
    return {
        'n_gold_records': n,
        'n_pred_records': len(pred_records),
        'n_matched_records': matched,
        'n_missing_predictions': missing,
        'n_gold_triples': gold_total,
        'n_pred_triples': pred_total,
        'sentence_exact_match': exact / n if n else 0.0,
        'micro_precision': precision,
        'micro_recall': recall,
        'micro_f1': f1,
        'tp': overall_tp, 'fp': overall_fp, 'fn': overall_fn,
        'per_relation': rel_metrics,
    }


def main():
    ap = argparse.ArgumentParser(description='关系抽取统一评价入口')
    ap.add_argument('--baseline', choices=sorted(BASELINES),
                    help='基线名称（用于解析默认/外部预测文件）')
    ap.add_argument('--gold', default='',
                    help='gold 文件（JSON/JSONL）；--check/--list-baselines 时可省略')
    ap.add_argument('--pred', default='', help='预测文件；缺省用 baseline 配置')
    ap.add_argument('--gold-field', default='', help='gold SPO 字段名')
    ap.add_argument('--pred-field', default='', help='预测 SPO 字段名')
    ap.add_argument('--out', default='', help='可选：结果 JSON 输出路径')
    ap.add_argument('--list-baselines', action='store_true',
                    help='列出基线实现/预测文件状态后退出')
    ap.add_argument('--check', action='store_true',
                    help='检查基线实现/预测文件是否存在后退出')
    args = ap.parse_args()

    if args.list_baselines or args.check:
        for name in sorted(BASELINES):
            info = describe_baseline(name)
            print(f'{info["display"]:<24} status={info["status"]:<17} '
                  f'impl={info["implementation_exists"]} '
                  f'pred={info["prediction_exists"]} '
                  f'missing={",".join(info["missing"]) or "-"}')
            print(f'  implementation: {info["implementation"] or "-"}')
            print(f'  prediction    : {info["prediction"] or "-"}')
            if info['trained_checkpoint']:
                print(f'  checkpoint    : {info["trained_checkpoint"]}')
            if info['notes']:
                print(f'  notes         : {info["notes"]}')
        return 0 if all(not describe_baseline(n)['missing']
                        for n in BASELINES
                        if BASELINES[n]['status'] == 'available') else 1

    if not args.gold:
        raise ValueError('必须提供 --gold（gold JSON/JSONL 文件）')

    pred_path = args.pred
    if not pred_path:
        if not args.baseline:
            raise ValueError('必须提供 --pred，或用 --baseline 指定预测文件来源')
        pred_path = resolve_prediction(args.baseline)
        if not pred_path or not os.path.exists(pred_path):
            spec = BASELINES[args.baseline]
            env_name = spec.get('pred_env', '')
            raise FileNotFoundError(
                f'{spec["display"]} 预测文件未找到。'
                + (f' 请设置 {env_name} 指向原始预测文件。' if env_name else ''))

    gold_fields = (args.gold_field,) if args.gold_field else GOLD_FIELDS
    pred_fields = (args.pred_field,) if args.pred_field else PRED_FIELDS
    gold_records = load_records(args.gold)
    pred_records = load_records(pred_path)
    result = evaluate(gold_records, pred_records, gold_fields, pred_fields)
    result['baseline'] = args.baseline or ''
    result['gold_file'] = args.gold
    result['pred_file'] = pred_path

    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.out:
        os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
        with open(args.out, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
