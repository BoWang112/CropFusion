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
    node_id = {}   # name -> id
    node_ent = {}  # name -> 本体实体
    relationships = []
    nid = 0

    def get_node(name, ontology_entity, props=None, force=False):
        nonlocal nid
        if not name:
            return None
        name = str(name).strip()
        if name in node_id:
            if force or node_ent.get(name) in (None, '其他因素'):
                node_ent[name] = ontology_entity
                for n in nodes:
                    if n['id'] == node_id[name]:
                        n['entity'] = ontology_entity
                        if props:
                            n['props'].update(props)
                        break
            return node_id[name]
        nid += 1
        node_id[name] = nid
        node_ent[name] = ontology_entity
        nodes.append({'id': nid, 'name': name, 'entity': ontology_entity, 'props': props or {}})
        return nid

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
            crop_id = get_node(crop, '作物', {'作物类型': crop_type})
            pest_id = get_node(pest, pest_ent, {'领域': domain, '易感地区': area,
                                                '易感时间': time, '来源': source}, force=True)
            # 病原体（病害库）vs 害虫（虫害库：三化螟等昆虫）
            if pathogen:
                path_type = '病原体' if domain == '病害' else '害虫'
                path_id = get_node(pathogen, path_type, force=True)
            else:
                path_id = None

            # 结构化关系
            if crop_id and pest_id:
                relationships.append((crop_id, '胁迫有', pest_id, {'来源': '结构化'}))
            if path_id and pest_id:
                relationships.append((path_id, '致使', pest_id, {'来源': '结构化'}))
            if pathogen and taxon:
                tax_id = get_node(taxon, '分类地位')
                relationships.append((path_id, '属', tax_id, {'来源': '结构化'}))
            if pathogen and kingdom:
                king_id = get_node(kingdom, '分类地位')
                relationships.append((path_id, '属', king_id, {'来源': '结构化'}))
            if pathogen and latin:
                latin_id = get_node(latin, '拉丁文名称')
                relationships.append((path_id, '拉丁名称', latin_id, {'来源': '结构化'}))
            if crop and crop_type:
                type_id = get_node(crop_type, '作物类型')
                relationships.append((crop_id, '归类于', type_id, {'来源': '结构化'}))
            if pest and area:
                area_id = get_node(area, '易感地区')
                relationships.append((pest_id, '易感区域', area_id, {'来源': '结构化'}))
            if pest and time:
                time_id = get_node(time, '易感时间')
                relationships.append((pest_id, '易感时间', time_id, {'来源': '结构化'}))
            # 类似病虫害（虫害库「类似病虫害」列）
            if domain == '虫害' and similar:
                sim_id = get_node(similar, '类似虫害')
                relationships.append((pest_id, '类似虫害有', sim_id, {'来源': '结构化'}))
            # 信息来源 → 专业书籍/权威网站
            if pest and source:
                src_type = '专业书籍' if ('《' in source or '书' in source) else '权威网站'
                src_id = get_node(source, src_type)
                relationships.append((pest_id, '知识来源于', src_id, {'来源': '结构化'}))

            # 抽取三元组映射
            res = results_by_row.get((row_num, domain)) or results_by_row.get(row_num)
            if not res:
                continue
            # 收集本记录抽到的 发病时期/发病部位/世代/越冬 实体
            crop_stage_nodes = set()
            crop_site_nodes = set()
            gen_nodes = set()      # 世代情况（虫害）
            overwin_nodes = set()  # 越冬情况（虫害）
            for seg in res.get('segments', []):
                for t in seg.get('triples', []):
                    sub, rel, obj = t['subject'], t['predicate'], t['object']
                    # STAGE_IN 按领域区分
                    if rel == 'STAGE_IN':
                        new_rel = '发病于作物' if domain == '病害' else '世代情况'
                    else:
                        new_rel = REL_MAP.get(rel)
                    if new_rel is None:
                        continue
                    # 按领域映射实体类型（病害 STAGE→发病时期, 虫害 STAGE→世代情况）
                    sub_type = _find_ent(seg['entities'], sub, domain)
                    obj_type = _find_ent(seg['entities'], obj, domain)
                    s_id = get_node(sub, sub_type)
                    o_id = get_node(obj, obj_type)
                    if s_id and o_id:
                        relationships.append((s_id, new_rel, o_id, {'来源': '知识抽取'}))
                        if obj_type == '发病时期':
                            crop_stage_nodes.add(o_id)
                        if obj_type == '发病部位':
                            crop_site_nodes.add(o_id)
                        if obj_type == '世代情况':
                            gen_nodes.add(o_id)
                        if obj_type == '越冬情况':
                            overwin_nodes.add(o_id)
                        if sub_type == '发病时期':
                            crop_stage_nodes.add(s_id)
                        if sub_type == '发病部位':
                            crop_site_nodes.add(s_id)

            # 补充：虫害文本中的 OVERWINTER 实体 → 越冬情况关系
            if domain == '虫害':
                for seg in res.get('segments', []):
                    for e in seg['entities']:
                        if e['type'] == 'OVERWINTER':
                            ow_id = get_node(e['text'], '越冬情况')
                            overwin_nodes.add(ow_id)

            # 「处于」关系：作物 → 发病时期
            if crop_id and crop_stage_nodes:
                for stage_id in crop_stage_nodes:
                    relationships.append((crop_id, '处于', stage_id, {'来源': '知识抽取'}))
            # 「发病于」关系：作物 → 发病部位
            if crop_id and crop_site_nodes:
                for site_id in crop_site_nodes:
                    relationships.append((crop_id, '发病于', site_id, {'来源': '知识抽取'}))
            # 「世代情况」关系：虫害 → 世代情况
            if pest_id and gen_nodes:
                for g_id in gen_nodes:
                    relationships.append((pest_id, '世代情况', g_id, {'来源': '知识抽取'}))
            # 「越冬情况」关系：虫害 → 越冬情况
            if pest_id and overwin_nodes:
                for ow_id in overwin_nodes:
                    relationships.append((pest_id, '越冬情况', ow_id, {'来源': '知识抽取'}))

    # 去重关系
    seen = set()
    dedup_rel = []
    for src, rel, dst, props in relationships:
        key = (src, rel, dst)
        if key not in seen:
            seen.add(key)
            dedup_rel.append((src, rel, dst, props))
    relationships = dedup_rel

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
