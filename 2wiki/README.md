# 2WikiMultiHopQA 外部验证实验

本目录是在 **2WikiMultiHopQA** 英文多跳数据集上单独实现的适配实验，用于
在统一证据预算和生成设置下比较 9 种方法。它**不是**农业主系统
（`CDPKG/`）的同一套实现，也不使用农业系统的 MultiDiGraph 本体、中文
BGE 模型、480 条 QA 集或 2048-token 生成参数。

| 项目 | 本目录 | 农业主系统 |
|---|---|---|
| 数据 | 2WikiMultiHopQA，400 条测试 + 40 条 few-shot | 480 条自动候选 QA + 180 条 few-shot |
| 文本单元 | 英文句子 | 中文 segment 片段 |
| 嵌入 | BGE-base-en-v1.5（768 维） | BGE-large-zh |
| 图 | 每个问题独立的 article mention 图，无关系标签 | 27 实体 / 20 关系 MultiDiGraph |
| 生成 | `deepseek-chat`，max_tokens=64 | 可切换 provider，max_tokens=2048 |
| 指标 | Answer EM / F1，分任务，token/延迟，bootstrap | P@3/R@3/F1@3/NDCG@3 与 BLEU/ROUGE/BERTScore |

> 原始 parquet、处理后的语料、嵌入、预测、缓存和结果表均不随仓库提供。

---

## 1. 数据准备

`common.py` 中的固定配置：

| 配置 | 值 |
|---|---:|
| 测试类型 | `comparison`, `inference`, `compositional`, `bridge-comparison` |
| `TEST_PER_TYPE` | 100（共 400 条） |
| `FEWSHOT_PER_TYPE` | 10（共 40 条） |
| 随机种子 | 42 |
| 测试来源 | development split |
| Few-shot 来源 | train split |

### 1.1 下载原始数据

```bash
cd 2wiki/experiments/2wiki
python download_data.py
```

脚本从 `hf-mirror.com` 的 `datadream/wiki_hop` parquet 镜像下载：

```
2wiki/data/validation.parquet
2wiki/data/train_0.parquet
```

已存在的文件会跳过。若镜像不可达，需要自行从 2WikiMultiHopQA 的公开来源
获取同等字段的 parquet 文件。

### 1.2 分层抽样

```bash
python prepare_data.py
```

- 从 development split 按 4 个 type 各抽取 100 条，共 400 条；
- 从 train split 按 4 个 type 各抽取 10 条，共 40 条 few-shot；
- 固定 seed=42，输出 `sample.jsonl`、`sample_ids.json`、
  `fewshot_pool.jsonl`；
- 校验测试集与 few-shot 的 id/问题文本不重叠。

### 1.3 文本索引

```bash
python build_text_index.py
```

从 `sample.jsonl` 和 `fewshot_pool.jsonl` 的 candidate context 拆句：

- `sentences.jsonl`：`{question_id, title, sentence_id, text, evidence_id}`；
- `sentence_embeddings.npy`：BGE-base-en 句子向量；
- `question_embeddings.npy`：测试/ few-shot 问题向量；
- `sentence_lookup.json`：question 顺序、type、question→句子索引。

每个问题只能访问自己的 candidate context；BM25 在运行时按 question 过滤
构建，避免跨问题检索。

### 1.4 构图

```bash
python build_graph.py
```

每个问题单独构建一个 article mention 图：

- 节点：candidate context 中的 article title；
- 节点属性：`paragraph_text`、`sentence_ids`、BGE(title+paragraph) 向量；
- 边：title 全名或去括号主名出现在另一 article 正文中；
- 无边/重复边会去重；输出 `graphs.json`。

该图**没有关系类型标签**；图重排序的结构分数使用 mention 有向边，若严格
有向路径不存在则回退无向视图。

### 1.5 协议检查

```bash
python validate_protocol.py
```

`validate_protocol.py` 静态检查数据规模、类型均衡、id/问题隔离、证据预算、
生成参数和泄漏防护等 14 项。任一检查失败时不应继续正式实验。

---

## 2. 九种方法

`run_method.py` 中的方法如下。所有方法共享同一 sentence corpus、question
embedding、BM25、LLM、证据预算和输出格式。

| # | 方法 | 本目录中的实际实现 |
|---:|---|---|
| 1 | LLM-No Retrieval | 仅问题 + 回答约束，无检索 |
| 2 | NaiveRAG | Dense（BGE 余弦）Top-20 |
| 3 | Hybrid RAG | Dense + BM25 → RRF → 证据单元 |
| 4 | GraphRAG | 语义锚点 Top-5 → 2-hop BFS → 节点段落/邻接线性化 |
| 5 | LightRAG | 显式实体锚点 + 语义锚点 → 2-hop BFS |
| 6 | G-Retriever | 问题感知节点 embedding 打分 → Top-5 锚点 → 2-hop BFS |
| 7 | PathRAG | 实体+语义锚点 → 2-hop 路径枚举/打分/去重 → Top-5 路径 |
| 8 | SG-RAG | 实体+语义候选 → 查询条件词匹配 → Top-5 子图 |
| 9 | Ours-Full | Dense+BM25 RRF + 语义/实体图检索 + 图重排序 + 动态 Few-shot + 三次生成一致性 |

`run_ours_ablation.py` 另含 15 个独立消融/变体（如 `anchor_retain`、
`pure_graph8_fs_type` 等），不属于上述 9 个主方法。

### 2.1 共享协议

| 参数 | 值 |
|---|---:|
| Dense Top-K | 20 |
| BM25 Top-K | 20 |
| RRF k | 60 |
| 语义锚点数 | 5 |
| 图扩展跳数 | 2 |
| Few-shot Top-K | 3（仅 Ours-Full 使用） |
| 图证据 Top-K | 5 |
| 证据预算 | ≤ 8 units，≤ 1500 tokens |
| LLM | `deepseek-chat` |
| Temperature / top_p / max_tokens | 0.2 / 0.9 / 64 |
| 文本/结构图重排序权重（Ours） | 0.5 / 0.5 |

### 2.2 Ours-Full 细节

1. Dense + BM25 用 RRF 融合；
2. 语义锚点（问题向量 vs 节点向量）与显式实体锚点合并；
3. 2-hop BFS 得到候选节点；
4. 对候选节点计算：
   - 语义分数：问题向量与节点 embedding 的余弦；
   - 结构分数：候选子图内以该节点为根的有效有向路径长度倒数均值；
   - `graph_score = 0.5 × semantic + 0.5 × structural`；
5. 取 Top-5 图节点/路径证据，与 RRF 文本证据一起按 8 units / 1500 tokens
   预算截断；
6. 从 40 条 few-shot 池中按问题 embedding 选 Top-3；
7. Ours-Full 使用三次独立生成 + 答案一致性投票 + 证据覆盖度选择；
   其余 8 个方法为单次生成。

---

## 3. 运行

以下命令在 `2wiki/experiments/2wiki/` 下执行：

```bash
cd 2wiki/experiments/2wiki

# 0. 下载/抽样/索引/构图
python download_data.py
python prepare_data.py
python build_text_index.py
python build_graph.py

# 1. 协议检查
python validate_protocol.py

# 2. 单方法冒烟测试（--limit 2）
python run_method.py --method hybrid_rag --limit 2

# 3. 九种方法全量运行（可断点续跑）
python run_all.py

# 4. 只跑某个方法
python run_method.py --method ours_full
python run_all.py --method pathrag

# 5. 消融/变体（--variant 必填）
python run_ours_ablation.py --variant pure_graph8_fs_type

# 6. 评价与置信区间
python evaluate.py
python bootstrap.py
python summary.py
```

参数：

- `run_method.py`：`--method`（必填）、`--limit`、`--start`、`--force`；
- `run_all.py`：`--limit`、`--force`、`--method`；
- `run_ours_ablation.py`：`--variant`（必填）、`--limit`、`--start`、
  `--force`。

`run_method.py` 读取已有 `outputs/predictions/<method>.jsonl` 并跳过已完成
题目；`--force` 强制重跑。LLM 请求缓存位于 `outputs/llm_cache/`，key 为
`sha256(model + method + question_id + prompt + gen_cfg)`。

---

## 4. 输出与评价

### 4.1 预测文件

```
2wiki/outputs/predictions/
├── llm_no_retrieval.jsonl
├── naive_rag.jsonl
├── hybrid_rag.jsonl
├── graphrag.jsonl
├── lightrag.jsonl
├── g_retriever.jsonl
├── pathrag.jsonl
├── sg_rag.jsonl
├── ours_full.jsonl
└── <ablation_variant>.jsonl
```

每条记录包含 `id/method/question/gold_answer/gold_type/prediction/
retrieved_evidence/input_tokens/output_tokens/latency_sec`；Ours-Full 还保存
三次生成、投票、覆盖度和选中下标等 `generation` 审计字段。

### 4.2 评价

```bash
python evaluate.py
```

默认写出：

```
2wiki/outputs/tables/table_overall.csv     # Answer EM/F1、平均 token、平均延迟
2wiki/outputs/tables/table_taskwise.csv    # 4 个 type × Answer EM/F1
2wiki/outputs/tables/token_usage.csv       # 调用次数与 token 统计
```

评价使用 `normalize_answer`（小写、去标点、去冠词、压缩空白）和 token 级
F1。`evaluate.py` 还包含 support/joint 指标辅助函数，但默认 CSV 输出是
Answer EM/F1、分任务 Answer EM/F1 和 token/延迟，不是 Support/Joint 表。

### 4.3 Bootstrap 与摘要

```bash
python bootstrap.py
python summary.py
```

- `bootstrap.py`：从 8 个基线中选 Answer F1 最高者，计算 Ours-Full 与它的
  paired bootstrap（1,000 次重采样，seed=42）95% CI；
- `summary.py`：打印各方法 EM/F1、最佳基线、Delta F1、95% CI 以及证据预算
  检查；需要全部 9 个预测文件存在。

### 4.4 可选绘图

`plot_results.py` 需要一张包含 `Overall` 和 `Task-wise` 两个 sheet 的
外部 xlsx；仓库未提供该工作簿。若已有结果表：

```bash
python plot_results.py --xlsx /path/to/results.xlsx --out /path/to/charts
```

---

## 5. 已知限制

- 2Wiki 的原始数据、处理后的 sample/graph/embedding、预测、缓存和结果表
  均未随仓库提供；`download_data.py` 只负责从公开镜像下载原始 parquet，
  后续文件需要按顺序运行脚本生成。
- `run_method.py`/`run_all.py` 的生成需要有效 API Key；没有农业主系统那样
  的模拟 LLM 降级。
- `bootstrap.py` 和 `summary.py` 直接读取全部 9 个预测文件，缺少文件会失败。
- 2Wiki 图是 article mention 图，没有农业 KG 的关系类型；结构分数不能与
  主系统的 MultiDiGraph 路径紧凑性逐项等同。
- `run_ours_ablation.py` 是探索性变体集合，不应与主系统 README 中的七项
  消融混为一谈。
- 本目录不声明完整复现了论文结果，也不在文档中写入任何未经验证的指标。
