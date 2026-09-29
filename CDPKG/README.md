# 作物病虫害知识抽取项目

基于 RoBERTa 的作物病虫害统一知识抽取，使用**病虫害合并训练的统一模型**，从作物病虫害知识库中抽取实体与关系，构建知识图谱。

## 模型

- **NER**：RoBERTa-BiLSTM-CRF（统一模型，16 类实体标签映射到 27 类本体实体）
  - 训练数据：`合并数据811/bio_train.txt`（4296 句）
  - 模型文件：`命名实体识别/checkpoint_merged811/`
- **RE**：RoBERTa-CoPET-CasRel（统一模型，9 种关系映射到 20 类本体关系）
  - 训练数据：`合并数据811/train.json`（1400 条）
  - 模型文件：`关系抽取/checkpoint_merged811/best_model.pt`

## 复现步骤

### 1. 生成合并数据（8:1:1 划分）

```bash
python resplit_datasets.py
```

从 `标准数据/费/实体bio/`、`病害标注数据/Data/`、`虫害数据/` 合并所有标注数据，按 8:1:1 重划分为 `合并数据811/`（固定随机种子）。

> 标注数据默认从本仓库**上级目录**下的 `标准数据/`、`病害标注数据/` 读取，
> 可用环境变量 `DATA_ROOT` 覆盖。

### 2. 训练统一 NER 模型

```bash
cd 命名实体识别
python ner.py \
  --model_name_or_path chinese-roberta-wwm-ext \
  --output_dir checkpoint_merged811 \
  --train_file ../合并数据811/bio_train.txt \
  --eval_file ../合并数据811/bio_val.txt \
  --test_file ../合并数据811/bio_test.txt \
  --need_birnn True --rnn_dim 128 \
  --num_train_epochs 100 --learning_rate 3e-5 \
  --train_batch_size 16 --eval_batch_size 16 \
  --do_train True --do_eval True --do_test True
```

### 3. 训练统一 RE 模型

```bash
cd 关系抽取
python roberta_copet_casrel_merged811.py
```

（脚本内直接引用 `../合并数据811/train.json`、`../合并数据811/val.json`）

### 4. 抽取整个知识库

```bash
python extract_knowledge_merged.py \
  --excel 病害知识库.xlsx --columns 病害症状,发病规律,防治方法 \
  --output results_merged
# 重复执行虫害知识库：两条记录合并写入 results_merged/抽取结果.json

python extract_knowledge_merged.py \
  --excel 虫害知识库.xlsx --columns 病害症状,发病规律,防治方法 \
  --output results_merged
# 输出统一 results_merged/抽取结果.json（每条记录含 domain 字段：病害/虫害）
```

### 5. 构建知识图谱数据（27 类实体 + 20 类关系）

```bash
python build_kg_29x20.py
# 输入 results_merged/抽取结果.json，输出 results_merged/kg_27x20/nodes.csv + relationships.csv
```

## 本体定义

- 实体+关系定义见 `实体+关系.docx`：**27 类实体 + 20 类关系**

## 环境

```bash
pip install -r ../requirements.txt
```

- 预训练模型 `chinese-roberta-wwm-ext/` 需自行下载，放在 `命名实体识别/` 与 `关系抽取/` 各一份
