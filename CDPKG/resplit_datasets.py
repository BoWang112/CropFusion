# -*- coding: utf-8 -*-
"""
标注数据合并与 8:1:1 重划分脚本
================================
把所有作物病虫害标注数据（实体标注 + 关系标注）合并后，
按 8:1:1 重新随机划分为训练集、验证集、测试集。

划分结果输出到 合并数据811/：
  NER 数据：bio_train.txt / bio_val.txt / bio_test.txt
  RE 数据： train.json / val.json / test.json

原始标注数据默认从以下目录读取（可在文件开头修改路径）：
  - 标准数据/费/实体bio/crop_dis_{train,val,test}.txt
  - 病害标注数据/Data/{train,val}_output.json
  - 虫害数据/{pest_ner_*,*_output.json}
"""
import sys, os, json, random
sys.stdout.reconfigure(encoding='utf-8')

_HERE = os.path.dirname(os.path.abspath(__file__))   # <repo>/CDPKG
_ROOT = os.path.dirname(_HERE)                       # <repo>
# 标注数据根目录：默认取本仓库的上级目录，可用环境变量 DATA_ROOT 覆盖
DATA_ROOT = os.environ.get('DATA_ROOT', os.path.dirname(_ROOT))

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '合并数据811')
os.makedirs(OUT_DIR, exist_ok=True)

# 允许用 --out 指定输出目录（用于验证/对比，不覆盖现有合并数据）
import argparse
_argp = argparse.ArgumentParser()
_argp.add_argument('--out', default=OUT_DIR)
_argp.add_argument('--seed', type=int, default=2019)
_args = _argp.parse_args()
OUT_DIR = _args.out
random.seed(_args.seed)  # 固定种子，可复现

NER_SOURCES = [
    os.path.join(DATA_ROOT, '标准数据', '费', '实体bio', 'crop_dis_train.txt'),
    os.path.join(DATA_ROOT, '标准数据', '费', '实体bio', 'crop_dis_val.txt'),
    os.path.join(DATA_ROOT, '标准数据', '费', '实体bio', 'crop_dis_test.txt'),
    '虫害数据/pest_ner_train.txt',
    '虫害数据/pest_ner_val.txt',
    '虫害数据/pest_ner_test.txt',
]

RE_SOURCES = [
    os.path.join(DATA_ROOT, '病害标注数据', 'Data', 'train_output.json'),
    os.path.join(DATA_ROOT, '病害标注数据', 'Data', 'val_output.json'),
    '虫害数据/train_output.json',
    '虫害数据/val_output.json',
    '虫害数据/test_output.json',
]

# ---------------- 标签映射（统一病虫害标签） ----------------
ENTITY_MAP = {
    'CROP': 'CROP', 'DISEASE': 'PEST', 'PATHOGEN': 'PATHOGEN',
    'SITE': 'SITE', 'SYMPTOM': 'SYMPTOM', 'STAGE': 'STAGE',
    'ENVIRON': 'ENVIRON', 'OTHERFACT': 'OTHERFACT',
    'AGRCONTROL': 'AGRCONTROL', 'CHEMCTRL': 'CHEMCTRL',
    'PHYSICTRL': 'PHYSICTRL', 'CULTIVF': 'CULTIVF',
    'TRANSMODE': 'TRANSMODE', 'INFSOURCE': 'INFSOURCE',
    'PEST': 'PEST', 'PAT': 'SITE', 'OTC': 'SYMPTOM', 'GEN': 'STAGE',
    'OVW': 'OVERWINTER', 'ENV': 'ENVIRON', 'OTH': 'OTHERFACT',
    'AGR': 'AGRCONTROL', 'MED': 'CHEMCTRL', 'BIO': 'BIOCONTROL',
}
REL_MAP = {
    'SUFFER_IS': 'SUFFER_IS', 'SYM_HAS': 'SYM_HAS', 'STAND_IN': 'STAGE_IN',
    'EFFECT_TO': 'EFFECT_TO', 'DIR_CAUSE': 'DIR_CAUSE', 'TRS_MTH': 'TRS_MTH',
    'HAS_SYMPTOM': 'SYM_HAS', 'LIFE_GEN': 'STAGE_IN', 'OVERWINTER': 'STAGE_IN',
    'ENV_COND': 'EFFECT_TO', 'CONTROL_AGR': 'CONTROL_AGR',
    'CONTROL_BIO': 'CONTROL_BIO', 'CONTROL_MED': 'CONTROL_MED',
}
PREVENT_TO_CONTROL = {
    'AGRCONTROL': 'CONTROL_AGR', 'PHYSICTRL': 'CONTROL_AGR',
    'CHEMCTRL': 'CONTROL_MED', 'BIOCONTROL': 'CONTROL_BIO',
}


def map_entity_label(label):
    """映射 BIO 标签"""
    if label == 'O':
        return 'O'
    prefix, ent = label.split('-', 1)
    ent_mapped = ENTITY_MAP.get(ent)
    if ent_mapped is None:
        return None
    return f'{prefix}-{ent_mapped}'


def map_relation(pred, obj_type):
    if pred == 'PREVENT_IS':
        return PREVENT_TO_CONTROL.get(obj_type)
    return REL_MAP.get(pred)


def load_ner_sents(paths):
    """加载所有 NER 句子，并映射为统一标签"""
    sents = []
    dropped = 0
    for path in paths:
        if not os.path.exists(path):
            print(f'  [跳过] 未找到 NER 数据: {path}')
            continue
        with open(path, encoding='utf-8') as f:
            for block in f.read().split('\n\n'):
                block = block.strip()
                if not block:
                    continue
                lines = block.split('\n')
                mapped_lines = []
                ok = True
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) != 2:
                        continue
                    ch, label = parts
                    mapped = map_entity_label(label)
                    if mapped is None:
                        dropped += 1
                        ok = False
                        break
                    mapped_lines.append(f'{ch} {mapped}')
                if ok and mapped_lines:
                    sents.append(mapped_lines)
    return sents, dropped


def load_re_records(sources):
    """加载所有 RE 记录，并映射为统一标签"""
    records = []
    dropped = 0
    for path in sources:
        if not os.path.exists(path):
            print(f'  [跳过] 未找到 RE 数据: {path}')
            continue
        data = json.load(open(path, encoding='utf-8'))
        for rec in data:
            entities = []
            ent_map = {}
            for e in rec['entities']:
                new_type = ENTITY_MAP.get(e['type'])
                if new_type is None:
                    dropped += 1
                    continue
                entities.append({'text': e['text'], 'type': new_type,
                                 'start': e['start'], 'end': e['end']})
                ent_map[e['text']] = new_type
            triples = []
            for t in rec['triple_list']:
                obj_type = ent_map.get(t['object'], '')
                new_rel = map_relation(t['predicate'], obj_type)
                if new_rel is None:
                    dropped += 1
                    continue
                triples.append({'subject': t['subject'], 'predicate': new_rel,
                                'object': t['object']})
            if entities and triples:
                records.append({'text': rec['text'], 'entities': entities,
                                'triple_list': triples})
    return records, dropped


def split_811(items):
    """按 8:1:1 划分"""
    random.shuffle(items)
    n = len(items)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    return items[:n_train], items[n_train:n_train + n_val], items[n_train + n_val:]


def write_ner(path, sents):
    with open(path, 'w', encoding='utf-8') as f:
        for s in sents:
            f.write('\n'.join(s) + '\n\n')


def write_re(path, records):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def main():
    # ---------------- NER ----------------
    print('NER 数据源:')
    ner_sents, ner_dropped = load_ner_sents(NER_SOURCES)
    print(f'NER 合并后总句子数: {len(ner_sents)} (丢弃 {ner_dropped})')
    train, val, test = split_811(ner_sents)
    print(f'NER 划分: train={len(train)}, val={len(val)}, test={len(test)}')
    write_ner(os.path.join(OUT_DIR, 'bio_train.txt'), train)
    write_ner(os.path.join(OUT_DIR, 'bio_val.txt'), val)
    write_ner(os.path.join(OUT_DIR, 'bio_test.txt'), test)

    # ---------------- RE ----------------
    print('\nRE 数据源:')
    re_records, re_dropped = load_re_records(RE_SOURCES)
    print(f'\nRE 合并后总记录数: {len(re_records)} (丢弃 {re_dropped})')
    train2, val2, test2 = split_811(re_records)
    print(f'RE 划分: train={len(train2)}, val={len(val2)}, test={len(test2)}')
    write_re(os.path.join(OUT_DIR, 'train.json'), train2)
    write_re(os.path.join(OUT_DIR, 'val.json'), val2)
    write_re(os.path.join(OUT_DIR, 'test.json'), test2)

    print(f'\n已保存到: {OUT_DIR}')


if __name__ == '__main__':
    main()
