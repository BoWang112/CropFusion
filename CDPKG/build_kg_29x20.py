# -*- coding: utf-8 -*-
"""
按 27 类实体 + 20 类关系构建知识图谱
======================================
将「统一模型抽取结果」（results_merged/抽取结果.json）映射到
「实体+关系.docx」定义的 27 类实体 + 20 类关系体系，
并用知识库结构化列补充，输出可导入 Neo4j 的 CSV。

实体映射（抽取 → 本体 27 类）：
  CROP→作物, PEST→病害/虫害(按领域), PATHOGEN→病原体/害虫,
  SITE→发病部位, SYMPTOM→具体症状, STAGE→发病时期,
  OVERWINTER→越冬情况, ENVIRON→气候因素/环境因素, OTHERFACT→其他因素,
  CULTIVF→栽培因素, TRANSMODE→传播方式, INFSOURCE→初侵染源,
  AGRCONTROL→农业防治, CHEMCTRL→化学防治, PHYSICTRL→物理防治,
  BIOCONTROL→生物防治

关系映射（抽取 → 本体 20 类）：
  SUFFER_IS→为害, SYM_HAS→症状有, CONTROL_*→防治方法有,
  STAGE_IN→发病于作物/世代情况, EFFECT_TO→影响, DIR_CAUSE→直接引发, TRS_MTH→传播

注：PEST 类实体与 STAGE_IN/STAGE 相关映射按每条记录的领域（病害/虫害）区分，
领域信息来自统一抽取结果中各记录的 domain 字段。
"""
import sys, json, csv, os
import pandas as pd
from collections import defaultdict

sys.stdout.reconfigure(encoding='utf-8')

# 知识库与统一抽取结果（每条记录含 domain 字段：病害/虫害）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KBS = [
    (os.path.join(BASE_DIR, '病害知识库.xlsx'), '病害'),
    (os.path.join(BASE_DIR, '虫害知识库.xlsx'), '虫害'),
]
EXTRACT_RESULT = 'results_merged/抽取结果.json'
OUT_DIR = 'results_merged/kg_27x20'

# ---------- 实体映射：抽取实体类型 -> 本体实体（27 类） ----------
ENTITY_MAP = {
    'CROP': '作物',
    'PEST': None,   # 按领域动态映射病害/虫害
    'PATHOGEN': '病原体',
    'SITE': '发病部位',
    'SYMPTOM': '具体症状',
    'STAGE': '发病时期',
    'OVERWINTER': '越冬情况',
    'ENVIRON': '环境因素',
    'OTHERFACT': '其他因素',
    'CULTIVF': '栽培因素',
    'TRANSMODE': '传播方式',
    'INFSOURCE': '初侵染源',
    'AGRCONTROL': '农业防治',
    'CHEMCTRL': '化学防治',
    'PHYSICTRL': '物理防治',
    'BIOCONTROL': '生物防治',
}

# ---------- 关系映射：抽取关系 -> 本体关系（20 类） ----------
# STAGE_IN 按领域区分：病害→发病于作物, 虫害→世代情况
REL_MAP = {
    'SUFFER_IS': '为害',
    'SYM_HAS': '症状有',
    'CONTROL_AGR': '防治方法有',
    'CONTROL_MED': '防治方法有',
    'CONTROL_BIO': '防治方法有',
    'STAGE_IN': None,   # 按领域动态：病害→发病于作物, 虫害→世代情况
    'EFFECT_TO': '影响',
    'DIR_CAUSE': '直接引发',
    'TRS_MTH': '传播',
}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    nodes = []
    node_key_to_id = {}   # (name, entity, context) -> id
    relationships = []
    nid = 0

    # 属性/防治/来源类实体按上下文消歧；作物/病虫害/病原/害虫等命名实体
    # 按 (name, entity_type) 合并，避免同一真实实体被记录切碎。
    GENERIC_ENTITY_TYPES = {
        '具体症状', '发病部位', '发病时期', '气候因素', '环境因素',
        '栽培因素', '其他因素', '世代情况', '越冬情况', '传播方式',
        '初侵染源', '易感时间', '易感地区', '农业防治', '化学防治',
        '物理防治', '生物防治', '类似虫害', '分类地位', '拉丁文名称',
        '专业书籍', '权威网站', '作物类型',
    }

    def _copy_props(props):
        out = dict(props or {})
        for key in ('source_ids', 'sources'):
            if isinstance(out.get(key), (list, set, tuple)):
                out[key] = list(out[key])
        return out

    def _merge_props(target, props):
        """合并节点属性；source_ids/sources 合并列表，保留全部来源。"""
        if not props:
            return
        for key, value in props.items():
            if key in ('source_ids', 'sources'):
                old = target.get(key, [])
                if not isinstance(old, list):
                    old = list(old) if old else []
                values = value if isinstance(value, (list, set, tuple)) else [value]
                for item in values:
                    if item and item not in old:
                        old.append(item)
                target[key] = old
            elif key not in target or target[key] in (None, ''):
                if isinstance(value, (list, set, tuple)):
                    target[key] = list(value)
                else:
                    target[key] = value

    def get_node(name, ontology_entity, props=None, force=False,
                 context_key=None):
        """获取/创建节点，按实体类型和上下文消歧。

        - 同名但实体类型不同 → 不同节点；
        - 属性/防治/来源类同名实体 → 以 context_key（record/crop/pest）
          区分不同上下文；
        - 命名实体（作物/病害/虫害/病原体/害虫）按 name+type 合并；
        - 节点 props 的 source_ids 合并去重，保留全部可追溯来源。
        """
        nonlocal nid
        if not name:
            return None
        name = str(name).strip()
        if context_key is not None and ontology_entity in GENERIC_ENTITY_TYPES:
            key = (name, ontology_entity, str(context_key))
        else:
            key = (name, ontology_entity)
        if key in node_key_to_id:
            for n in nodes:
                if n['id'] == node_key_to_id[key]:
                    if force and ontology_entity:
                        n['entity'] = ontology_entity
                    _merge_props(n['props'], props)
                    break
            return node_key_to_id[key]
        nid += 1
        node_key_to_id[key] = nid
        node_props = _copy_props(props)
        # 属性/防治类节点按上下文消歧：把上下文键写入属性，
        # 供 CropPestKG.find_node(context=...) 做可追溯的实体定位。
        if context_key is not None and ontology_entity in GENERIC_ENTITY_TYPES:
            node_props.setdefault('上下文', str(context_key))
        node = {'id': nid, 'name': name, 'entity': ontology_entity,
                'props': node_props}
        nodes.append(node)
        return nid

    def add_rel(src, rel, dst, source, source_id, **extra):
        """添加关系并保留来源/来源ID（重复关系在最后统一合并）。"""
        if not src or not dst:
            return
        props = {'来源': source, '来源ID': source_id}
        props.update(extra)
        relationships.append((src, rel, dst, props))

    def _find_ent(entities, text, domain):
        """查找实体类型，按领域区分：
        病害 STAGE→发病时期, 虫害 STAGE→世代情况
        ENVIRON 含气候关键词→气候因素, 否则→环境因素
        """
        import re as _re
        text = text.strip()
        for e in entities:
            if e['text'] == text:
                et = e['type']
                if et == 'STAGE':
                    return '发病时期' if domain == '病害' else '世代情况'
                if et == 'PEST':
                    return '病害' if domain == '病害' else '虫害'
                if et == 'ENVIRON':
                    # 气候关键词 → 气候因素
                    if _re.search(r'温|湿|雨|气温|℃|降雨|气候|露|雾|光照|日照|干旱|水', text):
                        return '气候因素'
                    return '环境因素'
                return ENTITY_MAP.get(et, '其他因素')
        return '其他因素'

    # 统一抽取结果（含 domain 字段）
    results_by_row = {}
    extract_results = json.load(open(EXTRACT_RESULT, encoding='utf-8'))
    for rec in extract_results:
        results_by_row[rec['row']] = rec
        # 同一 row 可能同时存在于病害/虫害两份知识库（row 编号各自从 1 开始）
        domain = rec.get('domain', '病害')
        results_by_row[(rec['row'], domain)] = rec

    for excel_path, domain in KBS:
        df = pd.read_excel(excel_path)
        for idx, row in df.iterrows():
            row_num = idx + 1
            crop = str(row.get('作物名') or '').strip()
            crop_type = str(row.get('作物类型') or '').strip()
            pest = str(row.get('病虫害名称') or '').strip()
            pathogen = str(row.get('病原体') or row.get('病原') or '').strip()
            similar = str(row.get('类似病虫害') or '').strip()
            latin = str(row.get('拉丁文名称') or '').strip()
            kingdom = str(row.get('界') or '').strip()
            taxon = str(row.get('分类地位') or '').strip()
            area = str(row.get('易感地区') or '').strip()
            time = str(row.get('易感时间') or '').strip()
            source = str(row.get('信息来源') or row.get('数据来源') or '').strip()

            # 领域：病害/虫害（PEST 实体按领域映射）
            pest_ent = '病害' if domain == '病害' else '虫害'
            record_key = f'{domain}:row{row_num}'
            record_source_id = f'structured:{domain}:row{row_num}'
            node_props = {'source_ids': [record_source_id]}

            crop_id = get_node(
                crop, '作物', {**node_props, '作物类型': crop_type},
                context_key=record_key)
            pest_id = get_node(
                pest, pest_ent,
                {**node_props, '领域': domain, '易感地区': area,
                 '易感时间': time, '来源': source},
                force=True, context_key=record_key)
            # 病原体（病害库）vs 害虫（虫害库：三化螟等昆虫）
            if pathogen:
                path_type = '病原体' if domain == '病害' else '害虫'
                path_id = get_node(
                    pathogen, path_type, node_props, force=True,
                    context_key=record_key)
            else:
                path_id = None

            # 结构化关系（保留结构化来源与行号）
            add_rel(crop_id, '胁迫有', pest_id, '结构化', record_source_id)
            add_rel(path_id, '致使', pest_id, '结构化', record_source_id)
            if pathogen and taxon:
                tax_id = get_node(
                    taxon, '分类地位', node_props, context_key=record_key)
                add_rel(path_id, '属', tax_id, '结构化', record_source_id)
            if pathogen and kingdom:
                king_id = get_node(
                    kingdom, '分类地位', node_props, context_key=record_key)
                add_rel(path_id, '属', king_id, '结构化', record_source_id)
            if pathogen and latin:
                latin_id = get_node(
                    latin, '拉丁文名称', node_props, context_key=record_key)
                add_rel(path_id, '拉丁名称', latin_id, '结构化', record_source_id)
            if crop and crop_type:
                type_id = get_node(
                    crop_type, '作物类型', node_props, context_key=record_key)
                add_rel(crop_id, '归类于', type_id, '结构化', record_source_id)
            if pest and area:
                area_id = get_node(
                    area, '易感地区', node_props, context_key=record_key)
                add_rel(pest_id, '易感区域', area_id, '结构化', record_source_id)
            if pest and time:
                time_id = get_node(
                    time, '易感时间', node_props, context_key=record_key)
                add_rel(pest_id, '易感时间', time_id, '结构化', record_source_id)
            # 类似病虫害（虫害库「类似病虫害」列）
            if domain == '虫害' and similar:
                sim_id = get_node(
                    similar, '类似虫害', node_props, context_key=record_key)
                add_rel(pest_id, '类似虫害有', sim_id, '结构化', record_source_id)
            # 信息来源 → 专业书籍/权威网站
            if pest and source:
                src_type = '专业书籍' if ('《' in source or '书' in source) else '权威网站'
                src_id = get_node(
                    source, src_type, node_props, context_key=record_key)
                add_rel(pest_id, '知识来源于', src_id, '结构化', record_source_id)

            # 抽取三元组映射
            res = results_by_row.get((row_num, domain)) or results_by_row.get(row_num)
            if not res:
                continue
            # 收集本记录抽到的 发病时期/发病部位/世代/越冬 实体及其来源
            # 用 (node_id, source_id) 列表保留每个抽取片段来源，最后统一合并。
            crop_stage_pairs = []
            crop_site_pairs = []
            gen_pairs = []       # 世代情况（虫害）
            overwin_pairs = []   # 越冬情况（虫害）

            for seg_idx, seg in enumerate(res.get('segments', [])):
                for tri_idx, t in enumerate(seg.get('triples', [])):
                    sub, rel, obj = t['subject'], t['predicate'], t['object']
                    # STAGE_IN 按领域区分
                    if rel == 'STAGE_IN':
                        new_rel = '发病于作物' if domain == '病害' else '世代情况'
                    else:
                        new_rel = REL_MAP.get(rel)
                    if new_rel is None:
                        continue
                    extract_source_id = (
                        f'extract:{domain}:row{row_num}:seg{seg_idx}:tri{tri_idx}')
                    extract_props = {'source_ids': [extract_source_id]}
                    # 按领域映射实体类型（病害 STAGE→发病时期, 虫害 STAGE→世代情况）
                    sub_type = _find_ent(seg['entities'], sub, domain)
                    obj_type = _find_ent(seg['entities'], obj, domain)
                    s_id = get_node(
                        sub, sub_type, extract_props, context_key=record_key)
                    o_id = get_node(
                        obj, obj_type, extract_props, context_key=record_key)
                    if s_id and o_id:
                        add_rel(s_id, new_rel, o_id, '知识抽取',
                                extract_source_id)
                        if obj_type == '发病时期':
                            crop_stage_pairs.append((o_id, extract_source_id))
                        if obj_type == '发病部位':
                            crop_site_pairs.append((o_id, extract_source_id))
                        if obj_type == '世代情况':
                            gen_pairs.append((o_id, extract_source_id))
                        if obj_type == '越冬情况':
                            overwin_pairs.append((o_id, extract_source_id))
                        if sub_type == '发病时期':
                            crop_stage_pairs.append((s_id, extract_source_id))
                        if sub_type == '发病部位':
                            crop_site_pairs.append((s_id, extract_source_id))

            # 补充：虫害文本中的 OVERWINTER 实体 → 越冬情况关系
            if domain == '虫害':
                for seg_idx, seg in enumerate(res.get('segments', [])):
                    for ent_idx, e in enumerate(seg.get('entities', [])):
                        if e['type'] == 'OVERWINTER':
                            source_id = (
                                f'extract:{domain}:row{row_num}:seg{seg_idx}:ent{ent_idx}')
                            ow_id = get_node(
                                e['text'], '越冬情况',
                                {'source_ids': [source_id]},
                                context_key=record_key)
                            if ow_id:
                                overwin_pairs.append((ow_id, source_id))

            # 「处于」关系：作物 → 发病时期
            if crop_id:
                for stage_id, source_id in crop_stage_pairs:
                    add_rel(crop_id, '处于', stage_id, '知识抽取', source_id)
            # 「发病于」关系：作物 → 发病部位
            if crop_id:
                for site_id, source_id in crop_site_pairs:
                    add_rel(crop_id, '发病于', site_id, '知识抽取', source_id)
            # 「世代情况」关系：虫害 → 世代情况
            if pest_id:
                for g_id, source_id in gen_pairs:
                    add_rel(pest_id, '世代情况', g_id, '知识抽取', source_id)
            # 「越冬情况」关系：虫害 → 越冬情况
            if pest_id:
                for ow_id, source_id in overwin_pairs:
                    add_rel(pest_id, '越冬情况', ow_id, '知识抽取', source_id)

    # 去重关系：同一 (src, rel, dst) 合并来源，保留全部真实来源信息
    merged_rels = {}
    for src, rel, dst, props in relationships:
        key = (src, rel, dst)
        entry = merged_rels.setdefault(
            key, {'来源': [], '来源ID': [], 'other': {}})
        source_label = props.get('来源')
        if source_label and source_label not in entry['来源']:
            entry['来源'].append(source_label)
        source_ids = props.get('来源ID')
        if isinstance(source_ids, (list, tuple, set)):
            source_values = list(source_ids)
        else:
            source_values = [source_ids] if source_ids else []
        for sid in source_values:
            if sid and sid not in entry['来源ID']:
                entry['来源ID'].append(sid)
        for k, v in props.items():
            if k not in ('来源', '来源ID'):
                entry['other'].setdefault(k, v)

    relationships = []
    for (src, rel, dst), entry in merged_rels.items():
        props = {
            '来源': '、'.join(entry['来源']),
            '来源ID': entry['来源ID'],
        }
        props.update(entry['other'])
        relationships.append((src, rel, dst, props))

    # 写出 CSV
    with open(os.path.join(OUT_DIR, 'nodes.csv'), 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id:ID', 'name', 'entity', 'props'])
        for n in nodes:
            writer.writerow([n['id'], n['name'], n['entity'],
                             json.dumps(n['props'], ensure_ascii=False)])

    with open(os.path.join(OUT_DIR, 'relationships.csv'), 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([':START_ID', ':TYPE', ':END_ID', 'props'])
        for src, rel, dst, props in relationships:
            writer.writerow([src, rel, dst, json.dumps(props, ensure_ascii=False)])

    print(f"节点总数: {len(nodes)}")
    print(f"关系总数: {len(relationships)}")
    ent_cnt = defaultdict(int)
    for n in nodes:
        ent_cnt[n['entity']] += 1
    print("\n实体类型分布:")
    for t, c in sorted(ent_cnt.items(), key=lambda x: -x[1]):
        print(f"  {t}: {c}")
    rel_cnt = defaultdict(int)
    for _, r, _, _ in relationships:
        rel_cnt[r] += 1
    print("\n关系分布:")
    for r, c in sorted(rel_cnt.items(), key=lambda x: -x[1]):
        print(f"  {r}: {c}")
    print(f"\n已保存到 {OUT_DIR}/")


if __name__ == '__main__':
    main()
