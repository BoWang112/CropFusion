# CropFusion 检索与问答系统

在既有作物病虫害知识图谱（`../results_merged/kg_27x20/`）基础上，
实现多视图检索 + 两级融合 + 证据驱动提示问答。

## 架构

```
用户问题
   │
   ├─[多视图表示]─────────────────────────────┐
   │  图结构视图(Graph)  文本视图(Text)  词汇视图(Lexical)
   │  networkx           BGE-large-zh   jieba+BM25
   │
   ├─[四路并行检索]───────────────────────────┘
   │  ① 向量检索(FAISS)      ② 关键词检索(BM25)
   │  ③ 语义驱动图检索       ④ 实体驱动图检索
   │
   ├─[两级融合]
   │  文本层: RRF(倒数排序融合)  → Top-10 文本证据
   │  图层:   语义+结构一致性重排序 → Top-5 图谱证据
   │
   └─[证据驱动提示词] → LLM 生成 → 结构化回答
       模型角色 + 文本证据 + 图谱证据 + 诊断要求 + 输出规范
```

## 目录结构

```
retrieval_qa/
├── config.py              # 路径 / 检索参数 / LLM API 配置
├── 1_build_index.py       # 构建文本块+向量索引+BM25+图谱检索加速缓存
├── 2_demo_cli.py          # 交互式问答 CLI
├── 3_evaluate.py          # 论文实验：检索指标 + 问答指标
├── generate_qa_test.py    # 自动生成问答评测集
├── modules/
│   ├── kg.py              # 图结构(Graph View) + 邻域/路径检索
│   ├── retriever.py       # 四路检索 + RRF 融合 + 图重排序
│   ├── prompts.py         # 证据驱动提示词 + 动态少样本示例
│   └── llm.py             # LLM API 客户端（DeepSeek + 模拟模式）
└── data/                  # 生成的索引与评测数据（缓存）
```

## 使用方法

### 0. 环境准备

```bash
pip install -r ../../requirements.txt
```

BGE-large-zh 模型需自行下载至 `../models/bge-large-zh-v1.5/`。

### 1. 构建索引

```bash
python retrieval_qa/1_build_index.py
```

产物：`data/index/`（text_blocks.json、vector.faiss、bm25.pkl、pest_vecs.npy、tri_cache.pkl）

### 2. 交互式问答

```bash
# 交互模式
python retrieval_qa/2_demo_cli.py

# 单问题 + 展示检索证据
python retrieval_qa/2_demo_cli.py -e "水稻叶片有梭形病斑是什么病？"
```

### 3. 论文实验评测

```bash
# 生成评测集（已生成，data/qa_test.json）
python retrieval_qa/generate_qa_test.py

# 检索性能评测（Precision@5 / Recall@5 / F1@5 / NDCG@5）
python retrieval_qa/3_evaluate.py --metrics retrieval

# 问答性能评测（BLEU-4 / ROUGE-L / BERTScore-F1；--subset N 控制规模）
python retrieval_qa/3_evaluate.py --metrics qa --subset 30
```

### 4. 接入真实 LLM（DeepSeek）

API Key **不写入代码文件**。请设置环境变量 `CDPKG_LLM_API_KEY`，或将其写入
`retrieval_qa/.env`（该文件已被 `.gitignore` 排除，不会入库）。

接入真实 LLM 后自动启用：
- **LLM 问题解析**（`modules/query_parser.py`）：从查询提取 作物/症状/部位/环境/意图
  五要素 + 实体抽取（对应论文 3.3.4），口语化查询（如"稻子""土豆"）自动归一化为
  标准名（水稻/马铃薯）并链接到图谱节点
- **证据驱动问答生成**：真实模型按提示词模板生成结构化回答

未配置 Key 时自动使用模拟 LLM（规则拼接证据生成占位回答），便于链路验证。

## 已实现特性（对应论文）

| 论文方法 | 实现位置 |
|---------|---------|
| 图结构视图表示 | `modules/kg.py` |
| 文本视图（实体级文本块） | `1_build_index.py: build_text_blocks` |
| 词汇视图（BM25 倒排索引） | `modules/retriever.py: BM25Index` |
| 向量语义检索 | `modules/retriever.py: vector_search` |
| 关键词检索 | `modules/retriever.py: keyword_search` |
| LLM 问题解析（五要素+实体抽取） | `modules/query_parser.py` |
| 症状判别词加权（排序改进） | `modules/retriever.py: _extract_symptom_terms / _apply_symptom_boost` |
| 语义驱动图检索 | `modules/retriever.py: semantic_graph_search` |
| 实体驱动图检索 | `modules/retriever.py: entity_graph_search` |
| 实体模糊链接（口语→规范名） | `modules/retriever.py: _fuzzy_link` |
| 文本层 RRF 融合 | `modules/retriever.py: rrf_fusion` |
| 图层结构+语义重排序（诊断关系优先） | `modules/retriever.py: graph_rerank` |
| 证据驱动提示词 | `modules/prompts.py` |
| 动态少样本示例 | `modules/prompts.py: select_fewshot` |
| 自一致性回答聚合 | `modules/llm.py: complete_with_consistency` |

## 评测命令

```bash
# 检索评测
python retrieval_qa/3_evaluate.py --metrics retrieval
# 端到端问答评测（精简回答模式，与参考答案可比）
python retrieval_qa/3_evaluate.py --metrics qa --concise --subset 20
# 全量评测（可续跑，约30分钟）
python retrieval_qa/run_eval_resumable.py
```

## 评测集（qa_test.json）

由 `generate_qa_test.py` 自动生成，覆盖六类任务；问题经口语化改写以模拟真实农业咨询
（不直接复制知识库原文）。每种 QA 含 `gold_pest`（标准病虫害）与 `task`（任务类型），
供证据级评测。

## 已知问题与改进方向

1. **图谱数据噪声**：69% 的"防治方法有"边是防治节点互相串联（抽取噪声），
   系统已通过关系白名单 + 一跳防治限制缓解。
2. **症状边结构**：症状关系挂在作物节点而非病虫害节点（`水稻-症状有->症状`），
   使病虫害子图不含症状边。可通过知识图谱构建时补充 `病虫害-症状有->症状` 边改进。
3. **评测集**：六类任务已覆盖；"相似病虫害区分"一类受图谱"类似虫害有"关系
   数量限制，如需扩充可补充更多类似关系。
4. **图谱构建缺陷**：`build_kg_29x20.py` 中 `发病于作物` 关系映射有误——
   该关系实际连接的是发病时期实体（如"抽穗期"）而非作物，导致病虫害节点
   无法沿"作物"桥接取症状。可考虑修正映射或补充 `病虫害-胁迫有-作物` 边。
