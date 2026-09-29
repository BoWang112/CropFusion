# 2WikiMultiHopQA Multi-hop QA Comparison

在 2WikiMultiHopQA 上，于同一协议下并列实现并对比 9 种检索增强方法。

## 方法

| # | 方法 | 检索通道 |
|---|---|---|
| 1 | LLM-No Retrieval | 无 |
| 2 | NaiveRAG | Dense (BGE) |
| 3 | Hybrid RAG | Dense + BM25 + RRF |
| 4 | GraphRAG | 语义锚点 → 邻域扩展 |
| 5 | LightRAG | 低层实体 + 高层结构 |
| 6 | G-Retriever | 问题感知节点打分 → 子图 |
| 7 | PathRAG | 关系路径枚举 + 剪枝 |
| 8 | SG-RAG | 多条件子图匹配 |
| 9 | Ours-Full | Dense + BM25 + RRF + 语义图 + 实体图 + 图 rerank + 动态 few-shot |

`run_ours_ablation.py` 另含本文方法的消融变体。

## 公平性约束（代码中强制）

- 同一 LLM（API model id `deepseek-chat`）、同一生成参数（temp=0.2, top_p=0.9, max=64）
- 同一 sentence corpus / embedding（BAAI/bge-base-en-v1.5, 768 维）/ BM25（k1=1.5, b=0.75）
- 统一 evidence 预算（≤8 units, ≤1500 tokens）
- LLM 请求缓存（hash(model+method+qid+prompt+config)），全部可断点续跑
- 禁止泄漏：gold answer/supporting_facts/evidences/type 只用于评测，不进入检索与生成
  （由 `validate_protocol.py` 做静态审计）

## 使用

```bash
cd 2wiki

# 0. 下载 2WikiMultiHopQA 原始数据（经 hf-mirror 镜像）
python experiments/2wiki/download_data.py

# 1. 分层抽样（规模见 common.py 的 TEST_PER_TYPE / FEWSHOT_PER_TYPE）
python experiments/2wiki/prepare_data.py

# 2. 文本索引（sentence corpus + dense + BM25）
python experiments/2wiki/build_text_index.py

# 3. 构图（每个 question 一个 local graph）
python experiments/2wiki/build_graph.py

# 4. 协议检查（先跑，全过再继续）
python experiments/2wiki/validate_protocol.py

# 5. 单方法（smoke: --limit 2）
python experiments/2wiki/run_method.py --method hybrid_rag --limit 2

# 6. 全量运行（9 方法，断点续跑）
python experiments/2wiki/run_all.py

# 7. 消融变体
python experiments/2wiki/run_ours_ablation.py --variant pure_graph8_fs_type

# 8. 评价（总体 + 分任务 + token）与置信区间
python experiments/2wiki/evaluate.py
python experiments/2wiki/bootstrap.py
python experiments/2wiki/summary.py
```

## 输出

以下文件由上述脚本在本地生成，不随仓库提供：

```
outputs/predictions/*.jsonl        每方法逐题预测（含检索证据、token、延迟）
outputs/tables/table_overall.csv   总体指标（Answer EM / Answer F1 / token / 延迟）
outputs/tables/table_taskwise.csv  分任务指标（4 type × Answer EM/F1）
outputs/tables/token_usage.csv     token 统计
outputs/llm_cache/                 LLM 请求缓存
```
