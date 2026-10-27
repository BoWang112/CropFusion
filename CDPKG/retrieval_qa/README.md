# 多视图检索与证据驱动问答

本目录是农业主系统的检索问答部分。它在
`../results_merged/kg_27x20/` 的病虫害知识图谱之上，构建文本索引与
多重图检索，融合文本/图谱证据，并用 LLM 生成有证据支撑的回答。

> 本目录的脚本默认从 `CDPKG/retrieval_qa/` 运行，路径由 `config.py`
> 统一定位。若已在该目录，不要再加 `retrieval_qa/` 前缀。

## 1. 总体架构

```
用户问题
  │
  ├─ 自动任务分类（六类，只读问题文本）
  │
  ├─ 文本通道
  │    BGE-large-zh + FAISS（向量） ─┐
  │    jieba + BM25（关键词）       ─┴─ RRF → Top-10 文本证据
  │
  ├─ 图谱通道
  │    语义驱动：问题向量 vs 病虫害锚点 → 有向邻域
  │    实体驱动：实体/别名/类型链接 → 有向邻域 + 多锚点路径
  │    → 结构紧凑性 + 三元组语义相关性重排序
  │    → Top-5 候选子图 → 展开为三元组证据
  │
  ├─ 证据驱动提示：文本证据 + 图谱证据 + 动态 Few-shot + 任务自适应要求
  │
  └─ 三次独立生成 → 核心结论一致性投票 → 证据覆盖度选择
```

检索结果由 `Retriever.get_evidence()` 返回
`(text_evidence, graph_evidence, detail)`。文本证据和图谱证据分别保留
自己的通道内排名，由评测脚本统一合并；生成阶段则按证据类型分别渲染。

---

## 2. 索引构建

### 2.1 输入

`1_build_index.py` 从 `config.py` 读取：

| 配置 | 默认路径 |
|---|---|
| 知识图谱节点 | `CDPKG/results_merged/kg_27x20/nodes.csv` |
| 知识图谱关系 | `CDPKG/results_merged/kg_27x20/relationships.csv` |
| 抽取结果 | `CDPKG/results_merged/抽取结果.json` |
| 输出目录 | `CDPKG/retrieval_qa/data/index/` |

抽取结果中的每条记录包含 `crop`、`pest`、`domain` 和 `segments`。每个
segment 是一个独立文本片段；过短（去模板后少于 10 字）的片段不进入索引。
片段 ID 优先使用原始 `segment_id`/`id`，否则使用与 QA 生成器一致的
`seg_<sha1(crop||pest||text)>` 稳定 ID。片段保留：

- `entity_id/entity_name/entity_type/crop/source`；
- `record_index/segment_index/segment_id/source_id`；
- 原始文本与去模板文本。

同一稳定 ID 的重复片段会合并 `source_ids`，不会产生重复文本块。

### 2.2 输出

```bash
cd CDPKG/retrieval_qa
python 1_build_index.py
```

产物位于 `data/index/`：

| 文件 | 内容 |
|---|---|
| `text_blocks.json` | segment 级文本片段、实体/来源/ID 元数据 |
| `vector.faiss` | BGE-large-zh 归一化向量索引（FAISS Inner Product） |
| `vector_doc_ids.json` | FAISS 向量位置到文本片段 ID 的映射 |
| `bm25.pkl` | jieba 分词后的 BM25 倒排索引 |
| `pest_vecs.npy` | 每个病虫害实体一个代表片段的查询向量 |
| `pest_vec_ids.json` | `pest_vecs.npy` 与片段 ID 的对齐顺序 |
| `tri_cache.pkl` | 三元组线性化文本的向量缓存（图重排序加速） |

`data/` 被 `.gitignore` 排除；索引需要在本机重新生成。

### 2.3 检索参数

`config.RETRIEVAL` 中的默认值：

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `vector_top_k` | 20 | 向量检索候选数 |
| `keyword_top_k` | 20 | BM25 候选数 |
| `semantic_top_k` | 20 | 语义图检索的病虫害锚点候选数 |
| `graph_hop` | 2 | 有向邻域扩展跳数 |
| `rrf_k` | 60 | 文本 RRF 平滑常数 |
| `text_top_k` | 10 | 文本层最终保留数 |
| `graph_top_k` | 5 | 图重排序后保留的候选子图数 |
| `path_max_len` | 3 | 多锚点简单路径最大长度 |
| `graph_rerank_lambda` | 0.4 | 图重排序结构权重 λ |
| `fewshot_top_k` | 3 | 动态 Few-shot 示例数 |

`entity_top_k` 虽在配置和参数扫描脚本中列出，但当前
`entity_graph_search()` 并未消费该键：实体图检索使用问题中链接到的全部
显式实体。论文实验以实际代码路径为准。

---

## 3. 四路检索

### 3.1 向量检索

`vector_search()` 用 BGE-large-zh 编码问题，在 FAISS 归一化向量索引上做
余弦相似度检索，默认取 `vector_top_k=20`。索引只包含 segment 级片段，
每个片段对应一条知识库语义单元，不复用实体级拼接文本块。

### 3.2 关键词检索

`keyword_search()` 使用 jieba 分词和 `rank_bm25` 的 BM25Okapi，默认取
`keyword_top_k=20`。BM25 索引与文本片段一一对应。

### 3.3 语义驱动图检索

`semantic_graph_search()`：

1. 用问题向量与每个病虫害代表片段向量做余弦相似度；
2. 取 `semantic_top_k=20` 个病虫害锚点；
3. 每个锚点独立调用 `_directed_expand()`，得到候选子图。

### 3.4 实体驱动图检索

`entity_graph_search()`：

1. 优先使用 LLM `QueryParser` 抽取实体；无 LLM 时回退字典匹配；
2. `CropPestKG.find_node(name, entity=..., context=...)` 支持实体类型、
   上下文和别名消歧；
3. 病虫害/害虫锚点做有向邻域扩展；作物锚点只沿 `症状有` 取症状边；
4. 多个锚点之间调用 `paths_between()`，恢复真实关系边并生成稳定
   `path_id`；路径边加入对应候选子图，路径上的全部有效关系均保留。

### 3.5 有向扩展与多重关系

`CropPestKG` 使用 `networkx.MultiDiGraph`，边 key 为关系类型。因此：

- `A-为害->B` 与 `A-影响->B` 是两条独立边；
- `neighbors()/in_neighbors()` 返回同一实体对上的全部关系；
- `ego_subgraph()` 保留子图内全部有效关系三元组；
- `paths_between()` 先按方向/关系类型筛选节点路径，再恢复该路径对应的
  全部真实关系边；
- 重复的相同三元组在 KG 加载时合并来源，不重复建边。

`_directed_expand()` 只沿白名单关系扩展，属性、症状、部位、条件、防治等
节点作为路径终点，不再继续向外扩散，避免经共享枢纽节点拉入无关实体。

---

## 4. 两级融合与图重排序

### 4.1 文本层 RRF

`text_fusion()` 对向量和 BM25 排名做标准 Reciprocal Rank Fusion：

```
RRF(d) = Σ_i 1 / (rrf_k + rank_i(d)),  rrf_k = 60
```

取 `text_top_k=10` 条文本证据。`get_evidence()` 默认
`use_extra_heuristics=False`，因此症状词加权、关键词目标锚点推断、目标锚点
提升/强制保留等启发式都不参与正式评测路径；它们只在显式启用额外启发式
时生效。

### 4.2 图谱重排序

`graph_rerank()` 对每个候选子图计算：

```
Score = λ × Structural + (1−λ) × Semantic,  λ = 0.4
```

- **Structural**：候选子图内以锚点为根、沿有效有向关系枚举长度不超过
  `path_max_len=3` 的简单关系路径，取路径长度倒数的平均值
  `mean(1/len(path))`。属性/防治/症状等终止节点不继续扩展。
- **Semantic**：候选子图内三元组线性化文本与问题向量的平均余弦相似度。
- 子图内三元组按各自查询相关性排序，并在证据中保留
  `triple_score`、`triple_rank`、`subgraph_score`、`subgraph_rank`、
  `graph_rank`、`anchor_id`、`subgraph_id`、`source`。
- 保留 `graph_top_k=5` 个候选子图；每个子图默认最多保留
  `per_graph_edges=25` 条边。因此“Top-5 子图”不是“5 条三元组”，
  最终展开的三元组数量可以更多。

### 4.3 路径证据

路径边在 `entity_graph_search()` 中加入候选子图，并由
`_path_id()` 生成稳定路径 ID；`_edges_to_evidence()` 将边按子图/三元组
排名输出。评测侧再把边展开为 triple 原子，同时保留路径/子图来源。

---

## 5. 证据驱动生成

### 5.1 自动任务分类

`modules/task_classifier.py` 只根据问题文本分为六类：

1. 病虫害识别
2. 症状解析
3. 条件判断
4. 防治措施推荐
5. 相似病虫害鉴别
6. 多条件综合推理

真实 LLM 以 `temperature=0` 输出任务名，失败时降级规则分类；分类结果按
“问题 + 模型 + 分类器版本”缓存。评测不读取 `qa['task']`。

### 5.2 动态 Few-shot

`modules/prompts.FewShotSelector` 从 180 条示例池（六类各 30 条）中按问题
BGE 相似度选取 `fewshot_top_k=3` 条。正式提示由
`build_prompt_with_dynamic_fewshot()` 构造，包含角色、任务说明、文本证据、
图谱证据、示例和输出规范。

### 5.3 三次生成与一致性选择

`LLMClient.complete_with_consistency_details()`：

1. 以 `temperature=0.2`、`top_p=0.9`、`max_tokens=2048` 独立生成 3 个候选；
   单次失败重试一次；
2. 抽取核心结论并按相似度聚类（阈值 0.6），票数等于簇大小；
3. 先选票数最多的簇（同票比较平均证据覆盖度），再在簇内选证据覆盖度
   最高者；仍相同则用完整答案平均相似度；
4. 返回 `answer/candidates/votes/coverage/core_conclusions/selected_index`
   等审计字段。

无 API Key 时 `LLMClient` 进入确定性模拟模式，仅用于链路验证，不能当作
真实实验结果。

---

## 6. 统一原子证据评估

`3_evaluate.py` 是所有检索方法的统一入口。

### 6.1 原子证据

- 文本片段 → `text` 原子，ID 来自 `segment_id`/文本块 ID；
- 三元组 → `triple` 原子，ID 为 `kg_<sha1(head||rel||tail)>`；
- 关系路径/候选子图 → 展开为 triple 原子，保留
  `path_id/subgraph_id/relation/triple_rank/graph_rank`；
- 原子保留 `source/evidence_source/refs/parents/path_ids`。

### 6.2 两通道统一排序

文本证据使用 RRF 排名，图谱证据使用图重排序排名。两路分别映射：

```
text_score  = 1 / (60 + text_rank)
graph_score = 1 / (60 + graph_rank)
```

合并后按 `text_score/graph_score` 降序排列；同分使用与方法名无关的稳定
原子 ID 打破平局；随后按原子 ID 去重并合并全部来源信息，取 Top-3 计算
Precision@3、Recall@3、F1@3、NDCG@3。

> 这里的 `1/(60+rank)` 是**评测排序规则**。文本检索内部的 RRF 也使用
> `rrf_k=60`，但评测不会直接使用原始 RRF 分数。

### 6.3 Gold 匹配

- Gold 文本：按 `segment_id` 或 Gold 事实原文匹配；文本相同但 segment ID
  不同的重复 Gold 事实合并为一个原子。
- Gold 图谱：按完整 `(head, relation, tail)` 精确匹配。
- 同一 Gold 事实被多条原子重复命中时只统计一次；Recall 使用去重后的
  Gold 事实集合。
- 检索失败计入 `failures` 并按零分参与汇总，所有方法使用同一
  evaluable 问题集合，不静默缩小样本量。
- `Ours` 单独评估（`--methods full`）与八方法批量评估共用同一
  `eval_retrieval_methods()`/原子排序函数，口径一致。

### 6.4 八种检索方法

```bash
cd CDPKG/retrieval_qa
python 3_evaluate.py --metrics retrieval --methods full
python 3_evaluate.py --metrics retrieval --methods all
```

`--methods all` 包括：

```
NaiveRAG, Hybrid RAG, GraphRAG (Local Search), LightRAG,
G-Retriever, PathRAG, SG-RAG, Ours-Retrieval
```

`full` 与 `ours-retrieval` 都可单独选择；前者走完整证据提示路径，后者用于
隔离检索贡献。`graph-only` 也可单独指定。

### 6.5 问答指标

```bash
python 3_evaluate.py --metrics qa --subset 20
```

`3_evaluate.py` 计算 BLEU-4、ROUGE-L、BERTScore-F1，写入
`data/results/qa_results_v2.json`。正式提示固定为 concise 模式，
与 `run_eval_resumable.py` 保持一致。

`run_eval_resumable.py` 的汇总还包含 BLEU-1/2/4、ROUGE-1/2/L、
BERTScore 和 Gscore；进度/汇总文件采用 v2 命名：

```
data/results/qa_progress_v2_<variant>[_global][_provider].json
data/results/qa_summary_v2_<variant>[_global][_provider].json
```

---

## 7. 消融、基线与参数分析

### 7.1 七项消融

`config.ABLATION_VARIANTS`：

| 变体 | 关闭的部分 |
|---|---|
| `-no-text-rrf` | 文本层 RRF |
| `-no-graph-rerank` | 图谱结构—语义重排序 |
| `-no-semantic-graph` | 语义驱动图检索 |
| `-no-entity-graph` | 实体驱动图检索 |
| `-no-evidence-prompt` | 证据驱动提示模板 |
| `-no-fewshot` | 动态 Few-shot |
| `-no-diag-constraints` | 结构化诊断约束 |

### 7.2 完整对比运行

```bash
python run_eval_resumable.py --variant full --provider deepseek
python run_all_variants.py --provider deepseek
python make_final_table.py
```

`run_all_variants.py` 串行运行完整模型、7 个消融和 8 个问答基线，共 16 个
变体；`make_final_table.py` 读取 `qa_summary_v2_*.json` 并打印对比表/消融表，
不写出新的结果文件。

`run_all_variants.py` 当前通过
`CDPKG/.venv-re/Scripts/python.exe` 启动子进程（Windows 路径）。若该解释器
不存在，请改用当前 Python 逐个运行：

```bash
python run_eval_resumable.py --variant <variant> --provider deepseek
```

### 7.3 参数敏感性

```bash
python param_sweep_retrieval.py --param graph_hop
python param_sweep_retrieval.py --param all
python param_sweep_gen.py --param text_top_k --subset 50
```

`param_sweep_retrieval.py` 的扫描项包括 `graph_rerank_lambda`、`graph_hop`、
`semantic_top_k`、`entity_top_k`、`rrf_k`、`text_top_k`、`graph_top_k`，
以及 `hop_x_anchor` 二维组合；指标复用 `3_evaluate.py` 的原子证据逻辑。
`param_sweep_gen.py` 额外支持 `fewshot_top_k`（含 0-shot），并使用生成侧
BLEU/ROUGE/BERTScore 指标。

### 7.4 GraphRAG 预处理

```bash
python build_communities.py --method louvain
python build_summaries.py --subset 3
python generate_global_qa.py
python table3_graph_metrics.py
```

- `build_communities.py` 在无向投影上做 Leiden（安装 `cdlib` 时）或 Louvain
  社区检测，写 `data/graphrag/communities.json`。
- `build_summaries.py` 为社区生成摘要，写
  `data/graphrag/community_summaries.json`，需要真实 LLM 配置；无 API Key
  时客户端进入模拟模式，生成的占位摘要不能用于正式实验。
- `generate_global_qa.py` 从图谱统计事实生成全局聚合题，目标约 90 条，
  写 `data/qa_global.json`。
- `table3_graph_metrics.py` 计算 Graph-Hit@3、Graph-Recall@5、Anchor-MRR，
  写 `data/results/table3_graph_metrics.json`；它需要上述社区/摘要文件。

GraphRAG-Local 在缺少摘要/社区文件时可以退化为仅使用锚点实体和关系；
GraphRAG-Global 要得到有意义的结果则需要社区摘要。

---

## 8. 评测数据与 Few-shot 数据

### 8.1 480 条 QA 候选

```bash
python generate_qa_test.py
```

- 输出：`data/qa_test.json`（本机文件，不随仓库提供）。
- 规模：480 条，固定随机种子 2026。
- 分布：多条件综合推理 140、防治措施推荐 140、病虫害识别 50、
  症状解析 50、条件判断 50、相似病虫害鉴别 50。
- 每条包含问题、参考答案、`gold_pest`、`gold_evidence_ids`、
  `gold_evidence_texts`、`gold_graph_evidence`、`derived_relations`、
  `review_status` 等字段。
- **这些是自动生成的候选数据，当前 `review_status` 均为 `pending`。**
  脚本不会自动把状态改成 `approved`；人工审核需另行完成。仓库没有附带
  已批准的最终 benchmark。

### 8.2 180 条 Few-shot 候选

```bash
python generate_fewshot_pool.py --per_task 30
```

- 输出：`data/fewshot_pool.json`、`data/fewshot_pool_audit.json`、
  `data/fewshot_pool.review.tsv`。
- 规模：六类任务各 30 条，共 180 条；`review_status` 同样为 `pending`。
- 默认生成会在 `qa_test.json` 存在时与其做问题/证据层面的去重；可用
  `--require_eval` 强制要求测试集存在。
- `--semantic_audit` 需要外部 `evidence_prompting_v3.py`，该文件不在仓库中；
  默认命令不启用它。若只需要生成候选池，请使用上面的默认命令。

### 8.3 全局聚合题

`generate_global_qa.py` 生成约 90 条图谱统计型问题，供
`run_eval_resumable.py --dataset global` 和 GraphRAG-Global 使用。它不是
480 条主测试集的一部分。

---

## 9. 已知限制

- 索引、QA/few-shot 文件、社区摘要、预测和结果均未随仓库提供；需在本地
  准备输入数据与模型后重新生成。
- 自动生成的 QA/few-shot 数据是 `pending` 候选，不是人工审核完成的 gold
  benchmark；本目录不声称已完成该审核。
- `generate_fewshot_pool.py --semantic_audit` 依赖仓库未提供的
  `evidence_prompting_v3.py`。
- `run_all_variants.py` 的子进程解释器路径是 Windows `.venv-re` 路径。
- `entity_top_k` 当前未被实体图检索消费（见第 2.3 节）。
- `table3_graph_metrics.py` 需要社区/摘要文件，否则会在加载阶段失败。
- 旧索引若没有 segment 级 `meta.segment_id`，`3_evaluate.py` 会提示先重建
  索引；新旧索引的文本原子粒度不同，指标不可直接混用。
- 本目录不运行完整实验，也不在文档中声明任何指标数值。

## 10. 单元测试

在仓库根目录运行：

```bash
python -m unittest discover -s tests -p "test_multigraph_atomic_protocol.py" -v
```

测试覆盖多重图关系存储、重复三元组来源合并、路径方向/关系变体、图重排序
元数据、统一原子证据排序、Gold 匹配、P/R/F1/NDCG、八方法口径一致性以及
下游 MultiDiGraph 兼容性。测试不需要真实 LLM 或 BGE 权重。
