# 作物病虫害知识抽取与知识图谱构建

本目录实现农业主系统的前半部分：基于 RoBERTa 的统一知识抽取、
27 类实体 / 20 类关系本体映射，以及可被检索问答系统加载的多重关系知识图谱。

> 代码、脚本与本体定义随仓库提供；标注数据、知识库 Excel、预训练权重和
> 微调检查点不随仓库提供。下面的流程只会在输入文件和权重存在时运行。
>
> 以下代码块均以**仓库根目录**为起点，可独立执行；每个代码块内部的 `cd`
> 会切换到该步骤所需的工作目录。

## 目录与模型

| 任务 | 模型 | 代码 | 输入 | 输出 |
|---|---|---|---|---|
| 命名实体识别 | RoBERTa-BiLSTM-CRF | `命名实体识别/ner.py`、`models.py` | `合并数据811/bio_{train,val,test}.txt` | `命名实体识别/checkpoint_merged811/` |
| 关系抽取 | RoBERTa-CoPET-CasRel | `关系抽取/roberta_copet_casrel_merged811.py` | `合并数据811/{train,val}.json` | `关系抽取/checkpoint_merged811/best_model.pt` |
| 知识抽取 | 上述 NER + RE | `extract_knowledge_merged.py` | 病害/虫害知识库 Excel | `results_merged/抽取结果.json` |
| 图谱构建 | 本体映射 + 多重图 | `build_kg_29x20.py` | 知识库 Excel + 抽取结果 | `results_merged/kg_27x20/{nodes,relationships}.csv` |

当前仓库中的合并数据规模（若未提供原始标注，则不会重新生成）：

- NER：train 4,296 句 / val 537 句 / test 537 句（BIO 格式）；
- RE：train 1,400 条 / val 175 条 / test 175 条（JSON）。

预训练主干使用 `chinese-roberta-wwm-ext`，分别放在：

```
命名实体识别/chinese-roberta-wwm-ext/
关系抽取/chinese-roberta-wwm-ext/
```

NER 微调检查点目录为 `命名实体识别/checkpoint_merged811/`；RE 微调权重为
`关系抽取/checkpoint_merged811/best_model.pt`。这些权重目录均被 `.gitignore`
排除，不随仓库提供。

---

## 1. 生成 8:1:1 合并数据

`resplit_datasets.py` 从外部标注目录读取 NER BIO 与 RE JSON，按固定种子
重新划分并输出到 `合并数据811/`：

```bash
cd CDPKG
python resplit_datasets.py --out 合并数据811 --seed 2019
```

默认读取路径：

- 病害/标准数据：`DATA_ROOT/标准数据/费/实体bio/crop_dis_{train,val,test}.txt`
- 病害关系数据：`DATA_ROOT/病害标注数据/Data/{train,val}_output.json`
- 虫害数据：`虫害数据/pest_ner_{train,val,test}.txt`、
  `虫害数据/{train,val,test}_output.json`

`DATA_ROOT` 默认是仓库根目录的上级目录，可用环境变量覆盖：

```bash
DATA_ROOT=/path/to/annotation_root python resplit_datasets.py --out 合并数据811
```

注意：`虫害数据/` 使用相对当前工作目录的路径，因此应在 `CDPKG/` 下执行。
原始标注数据不随仓库提供，脚本不会凭空重建这些文件。

输出：

```
合并数据811/
├── bio_train.txt / bio_val.txt / bio_test.txt
└── train.json / val.json / test.json
```

---

## 2. 训练 NER（RoBERTa-BiLSTM-CRF）

在 `CDPKG/命名实体识别/` 下执行：

```bash
cd CDPKG/命名实体识别

python ner.py \
  --model_variant roberta_bilstm_crf \
  --model_name_or_path chinese-roberta-wwm-ext \
  --output_dir checkpoint_merged811 \
  --train_file ../合并数据811/bio_train.txt \
  --eval_file ../合并数据811/bio_val.txt \
  --test_file ../合并数据811/bio_test.txt \
  --max_seq_length 256 \
  --train_batch_size 16 --eval_batch_size 16 \
  --learning_rate 3e-5 --num_train_epochs 100 --seed 2019 \
  --do_train True --do_eval True --do_test True
```

`--model_variant` 还支持 `roberta_crf`、`bert_bilstm_crf`、`bert_crf` 等基线；
论文主模型为 `roberta_bilstm_crf`。脚本会在 `output_dir` 中保存模型、
标签映射和 tokenizer 文件，并执行验证/测试集评估。

---

## 3. 训练 RE（RoBERTa-CoPET-CasRel）

在 `CDPKG/关系抽取/` 下直接运行脚本：

```bash
cd CDPKG/关系抽取
python roberta_copet_casrel_merged811.py
```

脚本中的主要配置（写在脚本内，不是命令行参数）：

- 基础模型：`chinese-roberta-wwm-ext/`
- 训练/验证数据：`../合并数据811/train.json`、`../合并数据811/val.json`
- `maxlen=512`，`batch_size=8`，AdamW 学习率 `1e-5`，训练 `50` 个 epoch
- 模型结构：CoPET 双向跨注意力 + Conditional LayerNorm + CasRel 主客体解码
- 权重输出：`checkpoint_merged811/best_model.pt`
- 评估预测文件：`checkpoint_merged811/bert_casrel_pred_0.7.json`

关系抽取脚本同时包含训练与评估逻辑；评估阶段会写出真实预测文件。若只想对
已有预测做统一 SPO 评价，可使用：

```bash
cd CDPKG/关系抽取
python evaluate_re.py --check
python evaluate_re.py \
  --baseline roberta_copet_casrel \
  --gold checkpoint_merged811/bert_casrel_pred_0.7.json \
  --gold-field spo_list --pred-field spo_list_pred
```

`baselines_config.py` 登记了 CopyRE、SpERT、TPLinker、OneRel、纯 CasRel 等
外部基线；它们的原始实现/预测文件未随仓库提供，需要通过
`CDPKG_*_IMPL` / `CDPKG_*_PRED` 环境变量指定，不能使用模拟模型代替。

---

## 4. 抽取整个知识库

在 `CDPKG/` 下执行两次：第一次处理病害知识库，第二次处理虫害知识库。
两次结果会合并写入同一个 `results_merged/抽取结果.json`，每条记录带
`domain` 字段（`病害` 或 `虫害`）。

```bash
cd CDPKG

python extract_knowledge_merged.py \
  --excel 病害知识库.xlsx \
  --columns 病害症状,发病规律,防治方法 \
  --output results_merged

python extract_knowledge_merged.py \
  --excel 虫害知识库.xlsx \
  --columns 病害症状,发病规律,防治方法 \
  --output results_merged
```

抽取器会：

1. 读取 Excel 的 `作物名`、`病虫害名称` 等字段；
2. 按句切分指定列，并在每段前加 `作物：…，病虫害：…，` 前缀；
3. 用 NER 识别实体、RE 抽取三元组；
4. 将每段结果保存为 `segments`，每条记录含 `row`、`crop`、`pest`、`domain`。

默认 checkpoint 路径：

- NER：`命名实体识别/checkpoint_merged811/`
- RE：`关系抽取/checkpoint_merged811/best_model.pt`

可用 `--ner_ckpt` / `--re_ckpt` 覆盖。也支持 `--text "..."` 做单条文本抽取，
输出 `results_merged/result_single.json`。

---

## 5. 构建知识图谱

在 `CDPKG/` 下运行：

```bash
cd CDPKG
python build_kg_29x20.py
```

输入：

- `results_merged/抽取结果.json`
- `病害知识库.xlsx`
- `虫害知识库.xlsx`

输出：

```
results_merged/kg_27x20/
├── nodes.csv          # id:ID, name, entity, props
└── relationships.csv  # :START_ID, :TYPE, :END_ID, props
```

脚本名中的 `29x20` 是历史命名；实际使用的是下述 27 类实体 / 20 类关系本体，
输出目录为 `kg_27x20/`。

### 5.1 本体：27 类实体

| # | 实体 | # | 实体 | # | 实体 |
|---|---|---|---|---|---|
| 1 | 作物 | 10 | 专业书籍 | 19 | 其他因素 |
| 2 | 作物类型 | 11 | 权威网站 | 20 | 栽培因素 |
| 3 | 病原体 | 12 | 具体症状 | 21 | 气候因素 |
| 4 | 害虫 | 13 | 发病时期 | 22 | 世代情况 |
| 5 | 拉丁文名称 | 14 | 发病部位 | 23 | 越冬情况 |
| 6 | 分类地位 | 15 | 易感地区 | 24 | 农业防治 |
| 7 | 病害 | 16 | 易感时间 | 25 | 物理防治 |
| 8 | 虫害 | 17 | 初侵染源 | 26 | 生物防治 |
| 9 | 类似虫害 | 18 | 传播方式 | 27 | 化学防治 |

> 实现中还有一个非本体兜底标签 **`环境因素`**：当 ENVIRON 实体不含气候关键词
> 时由 `_find_ent()` 映射得到。它用于保留抽取结果，不改变 27 类本体清单。

### 5.2 本体：20 类关系

```
致使, 防治方法有, 影响, 胁迫有, 类似虫害有, 属, 知识来源于,
拉丁名称, 发病于作物, 症状有, 为害, 归类于, 世代情况, 越冬情况,
处于, 易感区域, 易感时间, 发病于, 直接引发, 传播
```

### 5.3 抽取标签到本体的映射

| 抽取标签 | 本体实体 |
|---|---|
| CROP | 作物 |
| PEST | 病害（病害库）/ 虫害（虫害库） |
| PATHOGEN | 病原体（病害库）/ 害虫（虫害库） |
| SITE | 发病部位 |
| SYMPTOM | 具体症状 |
| STAGE | 发病时期（病害）/ 世代情况（虫害） |
| OVERWINTER | 越冬情况 |
| ENVIRON | 气候因素；无气候关键词时为 环境因素 |
| OTHERFACT / CULTIVF | 其他因素 / 栽培因素 |
| TRANSMODE / INFSOURCE | 传播方式 / 初侵染源 |
| AGRCONTROL / PHYSICTRL | 农业防治 / 物理防治 |
| CHEMCTRL / BIOCONTROL | 化学防治 / 生物防治 |

关系映射：

| 抽取关系 | 本体关系 |
|---|---|
| SUFFER_IS | 为害 |
| SYM_HAS | 症状有 |
| CONTROL_AGR / CONTROL_MED / CONTROL_BIO | 防治方法有 |
| STAGE_IN | 发病于作物（病害）/ 世代情况（虫害） |
| EFFECT_TO | 影响 |
| DIR_CAUSE | 直接引发 |
| TRS_MTH | 传播 |

结构化字段还会生成：`作物-胁迫有->病虫害`、`病原体/害虫-致使->病虫害`、
`病原体-属->分类地位`、`病原体-拉丁名称->拉丁文名称`、
`作物-归类于->作物类型`、`病虫害-易感区域->易感地区`、
`病虫害-易感时间->易感时间`、`病虫害-类似虫害有->类似虫害`、
`病虫害-知识来源于->专业书籍/权威网站` 等边。

### 5.4 实体消歧、关系去重与来源保留

- **命名实体**（作物、病害、虫害、病原体、害虫等）按
  `(name, entity_type)` 合并；同名不同类型会得到不同节点。
- **属性/防治/来源类实体**按 `(name, entity_type, context_key)` 区分；
  `context_key` 形如 `病害:row12`，并写入节点 `props["上下文"]`，供
  `CropPestKG.find_node(name, entity=..., context=...)` 精确定位。
- **关系去重**以 `(起点节点, 关系类型, 终点节点)` 为唯一标识；这里的
  “起点节点”是知识图谱实体 ID，不是来源记录 ID。重复三元组合并
  `来源` 与 `来源ID`（provenance），不丢失任何来源。
- 输出 CSV 中的 `props` 保存 `source_ids`、领域、上下文、易感信息等元数据。
- 检索侧 `CropPestKG` 使用 `networkx.MultiDiGraph`，以关系类型作为边 key：
  同一实体对上的 `为害` 与 `影响` 等不同关系会分别保留；相同三元组重复出现
  时合并来源属性。

---

## 6. 推荐执行顺序

| 步骤 | 工作目录 | 输入 | 输出 |
|---|---|---|---|
| 1 | `CDPKG/` | 外部标注数据 | `合并数据811/` |
| 2 | `CDPKG/命名实体识别/` | `bio_*.txt`、RoBERTa 主干 | `checkpoint_merged811/` |
| 3 | `CDPKG/关系抽取/` | `train.json`、`val.json`、RoBERTa 主干 | `best_model.pt`、预测 JSON |
| 4 | `CDPKG/` | `病害知识库.xlsx`、`虫害知识库.xlsx`、两个 checkpoint | `results_merged/抽取结果.json` |
| 5 | `CDPKG/` | `抽取结果.json`、两个知识库 | `results_merged/kg_27x20/*.csv` |
| 6 | `CDPKG/retrieval_qa/` | KG CSV、抽取结果、BGE-large-zh | `data/index/` |

第 6 步及检索问答流程见 [`retrieval_qa/README.md`](retrieval_qa/README.md)。

## 7. 环境

```bash
cd CDPKG
pip install -r ../requirements.txt
```

- Python 3.12、PyTorch 2.5.1、Transformers 4.43.x（见
  [`../requirements.txt`](../requirements.txt)）；
- NER 额外使用 `torchcrf`；RE 使用 `bert4torch`；
- 预训练权重、微调检查点、知识库 Excel 和标注数据均需自行准备。

## 8. 已知限制

- 原始标注、知识库 Excel、合并数据和 checkpoint 均被 `.gitignore` 排除；
  本文档不声称它们可以由仓库自动重建。
- `resplit_datasets.py` 的外部标注目录需按上方结构提供；缺失时会直接失败。
- `build_kg_29x20.py` 的路径和输入文件名是固定的，运行前需保证
  `results_merged/抽取结果.json` 与两个知识库文件存在。
- `evaluate_re.py` 的外部基线（CopyRE/SpERT/TPLinker/OneRel/纯 CasRel）
  需要真实实现或真实预测文件，仓库不提供模拟版本。
