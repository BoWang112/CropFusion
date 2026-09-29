# CropFusion

**Multi-View Evidence Fusion for Knowledge Graph-Enhanced Crop Disease and Pest Question Answering.**

Code accompanying our paper. The repository has two parts:

- **`CDPKG/`** — crop disease and pest knowledge-graph construction (NER + RE) and the multi-view evidence-fusion retrieval-QA pipeline.
- **`2wiki/`** — the same retrieval pipeline applied to 2WikiMultiHopQA, with nine retrieval-augmented methods implemented side by side under one protocol.

> 📌 **Only code is provided.** Data, model weights and outputs are not included — see [Installation](#installation) and [Getting the data](#getting-the-data).

---

## What the code does

### Retrieval pipeline

Domain knowledge is organised into **graph-structural**, **semantic textual**, and **lexical** views, and retrieved through four parallel channels:

| Channel | Implementation |
|---|---|
| ① Vector | FAISS + BGE dense retrieval over text blocks |
| ② Keyword | BM25 (jieba tokenisation) |
| ③ Semantic-driven graph | Question–node embedding similarity → anchors → neighborhood expansion |
| ④ Entity-driven graph | Explicit entity linking to KG nodes → anchors → neighborhood expansion |

Textual candidates and candidate subgraphs are fused separately — **text-level Reciprocal Rank Fusion** and **graph-level re-ranking** — followed by evidence-grounded prompting with dynamic few-shot selection.

### Knowledge extraction

| Task | Model | Code |
|---|---|---|
| Entity recognition | RoBERTa-BiLSTM-CRF | `CDPKG/命名实体识别/` |
| Relation extraction | RoBERTa-CoPET-CasRel | `CDPKG/关系抽取/` |

### Compared methods (`2wiki/`)

`run_method.py` implements nine retrieval-augmented methods over a shared corpus, embeddings and evidence budget:

| Method | Retrieval channel |
|---|---|
| LLM-No Retrieval | none |
| NaiveRAG | dense |
| Hybrid RAG | dense + BM25 + RRF |
| GraphRAG | semantic anchors → neighborhood expansion |
| LightRAG | entity anchors + semantic anchors |
| G-Retriever | question-aware node scoring → subgraph |
| PathRAG | relational path enumeration + pruning |
| SG-RAG | multi-condition subgraph matching |
| Ours-Full | four channels + two-level fusion + dynamic few-shot |

`run_ours_ablation.py` contains the ablation variants of the proposed method.

---

## Repository structure

```
CropFusion/
├── CDPKG/                          # Main (agricultural) system
│   ├── build_kg_29x20.py           # Knowledge-graph construction
│   ├── extract_knowledge_merged.py # Unified NER+RE inference
│   ├── resplit_datasets.py         # 8:1:1 re-split of annotation data
│   ├── 命名实体识别/                # NER: RoBERTa-BiLSTM-CRF
│   ├── 关系抽取/                    # RE: RoBERTa-CoPET-CasRel
│   └── retrieval_qa/               # Retrieval-QA pipeline
│       ├── modules/                # kg / llm / prompts / query_parser / retriever
│       ├── graphrag/               # GraphRAG, LightRAG, G-Retriever, PathRAG, SG-RAG
│       ├── 1_build_index.py        # Build text index (FAISS + BM25)
│       ├── 2_demo_cli.py           # Interactive demo
│       ├── 3_evaluate.py           # Retrieval + QA evaluation
│       ├── run_eval_resumable.py   # Resumable LLM evaluation
│       ├── param_sweep_*.py        # Hyper-parameter sensitivity
│       └── table3_graph_metrics.py # Graph-channel metrics
│
└── 2wiki/                       # 2WikiMultiHopQA comparison
    └── experiments/2wiki/
        ├── download_data.py        # Data acquisition
        ├── prepare_data.py         # Stratified sampling
        ├── build_text_index.py     # Sentence corpus + BGE embeddings + BM25
        ├── build_graph.py          # Question-specific local graphs
        ├── run_method.py           # The nine methods
        ├── run_ours_ablation.py    # Ablation variants
        ├── validate_protocol.py    # Leakage / consistency checks
        ├── evaluate.py             # EM / F1
        ├── bootstrap.py            # Paired bootstrap CI
        └── summary.py              # Run summary
```

---

## Installation

```bash
git clone https://github.com/BoWang112/CropFusion.git
cd CropFusion
pip install -r requirements.txt
```

**Model weights** are not included. Download them to the paths the code expects:

| Purpose | Model | Expected path |
|---|---|---|
| Chinese text embedding (main system) | `BAAI/bge-large-zh-v1.5` | `CDPKG/models/bge-large-zh-v1.5/` |
| English text embedding (2Wiki) | `BAAI/bge-base-en-v1.5` | `CDPKG/models/bge-base-en-v1.5/` |
| NER backbone | `hfl/chinese-roberta-wwm-ext` | `CDPKG/命名实体识别/chinese-roberta-wwm-ext/` |
| RE backbone | `hfl/chinese-roberta-wwm-ext` | `CDPKG/关系抽取/chinese-roberta-wwm-ext/` |

```bash
# Example: download the BGE models
huggingface-cli download BAAI/bge-large-zh-v1.5 --local-dir CDPKG/models/bge-large-zh-v1.5
huggingface-cli download BAAI/bge-base-en-v1.5  --local-dir CDPKG/models/bge-base-en-v1.5
```

> If `huggingface.co` is unreachable, set `HF_ENDPOINT=https://hf-mirror.com`.

### API credentials

Keys are never hardcoded. Provide them via environment variables or a `.env` file:

```bash
# CDPKG/retrieval_qa/.env   (git-ignored)
CDPKG_LLM_API_KEY=your_key
CDPKG_SHENNONG_API_KEY=your_key   # optional, for base-LLM generalisation
```

Environment variables take precedence over `.env`. The client is OpenAI-compatible:

| Variable | Default |
|---|---|
| `CDPKG_LLM_BASE_URL` | `https://api.deepseek.com/v1` |
| `CDPKG_LLM_MODEL` | `deepseek-chat` |
| `CDPKG_LLM_PROVIDER` | `deepseek` |

---

## Getting the data

Data is not redistributed. Regenerate it with the scripts below.

### 2WikiMultiHopQA

```bash
cd 2wiki
python experiments/2wiki/download_data.py      # download the raw dataset
python experiments/2wiki/prepare_data.py       # stratified sampling
python experiments/2wiki/build_text_index.py   # sentence corpus + embeddings + BM25
python experiments/2wiki/build_graph.py        # question-specific local graphs
```

Sampling size is controlled by `TEST_PER_TYPE` / `FEWSHOT_PER_TYPE` in
[`common.py`](2wiki/experiments/2wiki/common.py).

### Agricultural system

The knowledge base and annotation data are not redistributed. Supply your own in the same
schema, then:

```bash
cd CDPKG
python build_kg_29x20.py          # build the knowledge graph
cd retrieval_qa
python 1_build_index.py           # build the FAISS + BM25 index
```

---

## Running the code

### 2WikiMultiHopQA comparison

```bash
cd 2wiki/experiments/2wiki

python validate_protocol.py                       # protocol checks — run first

python run_method.py --method ours_full           # one method (resumable, cached)
python run_all.py                                 # all nine methods

python run_ours_ablation.py --variant pure_graph8_fs_type   # ablation

python evaluate.py                                # metrics
python bootstrap.py                               # paired bootstrap CI
python summary.py                                 # summary

python plot_results.py                   # figures
```

### Agricultural system

```bash
cd CDPKG/retrieval_qa

python 2_demo_cli.py                 # interactive QA demo
python 3_evaluate.py                 # retrieval + QA metrics
python run_eval_resumable.py         # full evaluation (resumable)
python param_sweep_retrieval.py      # graph_hop / rrf_k sensitivity
python table3_graph_metrics.py       # graph-channel metrics
```

### Implementation notes

- Baselines in `2wiki/` are re-implemented over a shared corpus, embeddings, generation
  model and evidence budget (≤ 8 units, ≤ 1500 tokens), so all methods are directly comparable.
- LLM responses are cached (`hash(model + method + question_id + prompt + config)`) and every
  run is resumable.
- `validate_protocol.py` statically audits the retrieval and generation code to confirm that
  `answer` / `supporting_facts` / `evidences` / `type` are used only for evaluation and never
  reach retrieval or prompting.
- Graph evidence is paragraph-level (one evidence unit per graph node); `evaluate.py` therefore
  reports Answer EM / F1 rather than sentence-level Support / Joint metrics.

---

## License

Released under the MIT License. See [LICENSE](LICENSE).

## Contact

For questions about the code, please open an issue or contact the corresponding author.

---

## 中文说明

本项目为论文配套代码，含两部分：

- **`CDPKG/`** — 农业病虫害知识图谱构建（NER + RE）与多视图证据融合检索问答系统；
- **`2wiki/`** — 同一套检索管线在 2WikiMultiHopQA 上的对比实现，统一协议下并列 9 种方法。

**仅提供代码**：数据、模型权重与输出结果均未随仓库提供，请按上文
[Getting the data](#getting-the-data) 自行获取或重建。API Key 请通过环境变量或
`CDPKG/retrieval_qa/.env` 提供（代码中不含任何密钥）。
