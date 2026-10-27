# CropFusion

**Multi-View Evidence Fusion for Knowledge Graph-Enhanced Crop Disease and Pest Question Answering**

Research code accompanying our paper. The repository contains two related but
**separately implemented** experimental systems:

| System | Directory | Purpose | Evaluation data |
|---|---|---|---|
| Agricultural main system | `CDPKG/` | Crop disease/pest knowledge extraction, knowledge-graph construction, multi-view retrieval and evidence-grounded QA | 480 auto-generated QA candidates (`pending` review) |
| 2WikiMultiHopQA external validation | `2wiki/` | A separate adaptation of retrieval/QA methods to an English multi-hop benchmark | 400 stratified test questions (100 per type) |

> **Scope of this repository.** Only source code is included. Knowledge bases,
> annotation data, fine-tuned checkpoints, embedding models, indexes, generated
> QA/few-shot files, predictions, caches and result tables are **not tracked**.
> The scripts can process these files when supplied, but they do not recreate
> missing source data or trained weights by themselves. No experimental result
> is claimed by this README.

---

## Overview

The agricultural system organises domain knowledge into three complementary
views and retrieves evidence through four parallel channels:

1. **Graph view** — a `networkx.MultiDiGraph` knowledge graph whose edge key is
   the relation type, so multiple relations between the same entity pair are
   preserved.
2. **Semantic textual view** — segment-level text fragments encoded by
   BGE-large-zh and indexed with FAISS.
3. **Lexical view** — jieba tokenisation and a BM25 inverted index.

The four retrieval channels are (i) dense vector retrieval, (ii) BM25 keyword
retrieval, (iii) semantic-driven graph retrieval and (iv) entity-driven graph
retrieval. Text candidates are fused with Reciprocal Rank Fusion (RRF); graph
candidates are re-ranked by structural compactness and triple–query semantic
relevance. The selected evidence is then used in an evidence-grounded prompt
with automatic task classification, dynamic few-shot selection and
three-generation consistency selection.

The `2wiki/` system is a **different, adapted experiment**. It uses
question-local sentence corpora and mention graphs from the English
2WikiMultiHopQA dataset, its own BGE-base-en embeddings, its own evidence
budget and its own generation settings. It should not be read as the same
implementation or the same parameterisation as the agricultural system.

---

## Key implementation features

- **Knowledge extraction.** RoBERTa-BiLSTM-CRF for named-entity recognition and
  RoBERTa-CoPET-CasRel for relation extraction over a unified crop-pest
  annotation schema (`CDPKG/命名实体识别/`, `CDPKG/关系抽取/`).
- **Ontology-driven KG construction.** A 27-class entity / 20-relation ontology
  (plus an implementation-level `环境因素` fallback for non-climate ENVIRON
  mentions) is mapped to `nodes.csv` and `relationships.csv`, with entity-type
  and context-aware disambiguation and provenance preservation.
- **Multi-relation graph storage.** `CropPestKG` uses `MultiDiGraph`; parallel
  relations such as `A-为害->B` and `A-影响->B` are stored separately, while
  repeated identical triples merge their source IDs.
- **Four-path retrieval.** Dense, BM25, semantic-anchor graph retrieval and
  explicit-entity graph retrieval run in parallel.
- **Two-level fusion.** Text-level RRF produces Top-10 text evidence; graph
  candidates are re-ranked by
  `0.4 × structural compactness + 0.6 × triple–query semantic relevance` and
  the Top-5 candidate subgraphs are expanded into relation evidence.
- **Evidence-grounded generation.** Task classification from the question text,
  dynamic few-shot examples, task-adaptive answer organisation, and three
  independent generations with core-conclusion voting and evidence-coverage
  selection (`CDPKG/retrieval_qa/modules/`).
- **Unified retrieval evaluation.** Eight retrieval methods are evaluated on
  the same atomic evidence representation with P@3/R@3/F1@3/NDCG@3; seven
  ablations, end-to-end QA metrics and parameter-sensitivity scripts are also
  provided.
- **Separate 2Wiki adaptation.** Nine comparison methods, 400 stratified test
  samples, answer EM/F1, bootstrap confidence intervals and a static protocol
  audit are implemented in `2wiki/`.

---

## Technical framework

### 1. Three knowledge views

| View | Representation | Main implementation |
|---|---|---|
| Graph | Directed multi-relation KG (`MultiDiGraph`) | `CDPKG/retrieval_qa/modules/kg.py` |
| Semantic text | Segment-level text fragments + BGE + FAISS | `CDPKG/retrieval_qa/1_build_index.py`, `modules/retriever.py` |
| Lexical | jieba tokens + BM25 inverted index | `CDPKG/retrieval_qa/modules/retriever.py` |

### 2. Four retrieval channels

| Channel | Query signal | Output |
|---|---|---|
| Dense vector | BGE query embedding vs segment embeddings | Ranked text fragments |
| Keyword | BM25 over jieba tokens | Ranked text fragments |
| Semantic-driven graph | Query vs pest/disease anchor text embeddings | Anchor entities and directed neighbourhood subgraphs |
| Entity-driven graph | LLM/dictionary entity extraction, type-aware linking, aliases | Anchor entities, directed subgraphs and multi-anchor relation paths |

Graph expansion follows allowed relation directions, a relation whitelist and
terminal property/control nodes. Property, symptom, site, condition and control
nodes can appear as path endpoints but are not expanded through.

### 3. Two-level fusion

- **Text level:** RRF over dense and BM25 rankings, retaining Top-10 text
  evidence (`rrf_k=60`, `text_top_k=10`).
- **Graph level:** candidate subgraphs are scored by
  `λ · structural + (1−λ) · semantic`, with `λ=0.4`.
  - Structural score: mean reciprocal length of effective directed relation
    paths within the candidate subgraph (simple paths, bounded by
    `path_max_len=3`).
  - Semantic score: mean cosine similarity between query and linearised
    triples in the subgraph.
- **Selection:** `graph_top_k=5` candidate subgraphs are retained. Each
  subgraph may expand into more than one triple (`per_graph_edges=25` by
  default), so “Top-5 subgraphs” does not mean “five triples”.
- Graph evidence records the subgraph score/rank, triple score/rank, anchor ID,
  subgraph ID, relation type and source metadata.

### 4. MultiDiGraph knowledge graph

- Edge key is the relation type, so multiple relations between the same
  directed entity pair are preserved.
- Repeated identical `(source, relation, target)` triples merge provenance
  fields such as `来源ID`/`source_ids`.
- Entity linking supports entity type, context and aliases; domain entities
  (crop, disease, pest, pathogen) and generic property/control/source entities
  are disambiguated by different keys.
- Path search first checks direction and relation-type feasibility, then
  restores all valid relation edges for the selected node path. Multi-anchor
  paths retain every valid relation variant.

### 5. Evidence-grounded generation

- The task classifier uses only the question text and routes it to one of six
  tasks: diagnosis, symptom analysis, condition judgement, control
  recommendation, similar-pest discrimination and multi-condition reasoning.
- Dynamic few-shot examples are selected by question embedding similarity from
  the 180-example pool (`fewshot_top_k=3`).
- Prompts contain separate text and graph evidence blocks, task-specific
  instructions and output rules.
- Generation uses three independent samples (`temperature=0.2`, `top_p=0.9`,
  `max_tokens=2048`), clusters candidates by core conclusion, votes across
  clusters, and selects the candidate with the best evidence coverage.

### 6. Unified retrieval evaluation

All retrieval methods are converted to atomic evidence:

- a text fragment becomes a `text` atom with a stable segment/block ID;
- a triple becomes a `triple` atom;
- a path or subgraph is expanded into triple atoms while retaining its
  `path_id`/`subgraph_id`, relation type and original rank.

Text atoms use the RRF rank; graph atoms use the graph-reranking rank. Each
channel is mapped to `1/(60 + channel_rank)`, the two lists are merged, ties are
broken by a method-independent stable atom ID, duplicate atom IDs are merged
with all source information retained, and P@3/R@3/F1@3/NDCG@3 are computed on
the merged Top-3. The `1/(60+rank)` mapping is an evaluation rule; it is not the
raw RRF score used inside text retrieval.

Gold text facts are matched by segment ID or fact text; Gold graph facts are
matched by the complete `(head, relation, tail)` triple. Repeated Gold facts are
counted once. Retrieval failures are recorded and counted as zero rather than
silently reducing the sample size. `Ours` evaluated alone and inside the
eight-method batch uses the same function and ranking protocol.

---

## Repository layout

```
CropFusion/
├── README.md
├── requirements.txt
├── LICENSE
├── tests/
│   └── test_multigraph_atomic_protocol.py
├── CDPKG/
│   ├── README.md
│   ├── resplit_datasets.py
│   ├── extract_knowledge_merged.py
│   ├── build_kg_29x20.py
│   ├── 命名实体识别/                 # RoBERTa-BiLSTM-CRF training
│   ├── 关系抽取/                     # RoBERTa-CoPET-CasRel training/evaluation
│   └── retrieval_qa/
│       ├── README.md
│       ├── config.py
│       ├── 1_build_index.py
│       ├── 2_demo_cli.py
│       ├── 3_evaluate.py
│       ├── run_eval_resumable.py
│       ├── run_all_variants.py
│       ├── make_final_table.py
│       ├── param_sweep_retrieval.py
│       ├── param_sweep_gen.py
│       ├── table3_graph_metrics.py
│       ├── generate_qa_test.py
│       ├── generate_fewshot_pool.py
│       ├── generate_global_qa.py
│       ├── build_communities.py
│       ├── build_summaries.py
│       ├── modules/
│       │   ├── kg.py
│       │   ├── retriever.py
│       │   ├── prompts.py
│       │   ├── query_parser.py
│       │   ├── task_classifier.py
│       │   └── llm.py
│       └── graphrag/
│           ├── search.py
│           ├── gretriever.py
│           ├── pathrag.py
│           ├── sgrag.py
│           ├── lightrag.py
│           ├── community.py
│           └── summarize.py
└── 2wiki/
    ├── README.md
    └── experiments/2wiki/
        ├── common.py
        ├── download_data.py
        ├── prepare_data.py
        ├── build_text_index.py
        ├── build_graph.py
        ├── validate_protocol.py
        ├── run_method.py
        ├── run_all.py
        ├── run_ours_ablation.py
        ├── evaluate.py
        ├── bootstrap.py
        ├── summary.py
        └── plot_results.py
```

Data, model weights, indexes and outputs are intentionally absent.

---

## Installation

The code targets the dependency versions in [`requirements.txt`](requirements.txt)
(Python 3.12, PyTorch 2.5.1, Transformers 4.43.x).

```bash
git clone https://github.com/BoWang112/CropFusion.git
cd CropFusion
pip install -r requirements.txt
```

`requirements.txt` includes retrieval, graph, NER/RE, LLM-client and evaluation
dependencies. A GPU is optional for the retrieval/QA scripts but is normally
needed for NER/RE training.

---

## Configuration

### Agricultural system

LLM credentials are read from environment variables, with a fallback to
`CDPKG/retrieval_qa/.env` (git-ignored). Environment variables take precedence.
The client is OpenAI-compatible. The provider is selected with
`CDPKG_LLM_PROVIDER`; evaluation scripts that expose `--provider` accept the
same keys.

| Provider key | API key | Base URL | Model |
|---|---|---|---|
| `deepseek` | `CDPKG_LLM_API_KEY` | `CDPKG_LLM_BASE_URL` | `CDPKG_LLM_MODEL` (code default `deepseek-chat`) |
| `shennong` | `CDPKG_SHENNONG_API_KEY` | `CDPKG_SHENNONG_BASE_URL` | `CDPKG_SHENNONG_MODEL` (code default `sn`) |
| `deepseek-v4-flash`, `deepseek-v4-flash-0731` | `CDPKG_DEEPSEEK_V4_API_KEY` | `CDPKG_DEEPSEEK_V4_BASE_URL` | `CDPKG_DEEPSEEK_V4_MODEL` (must be set) |
| `shennong-v4`, `shennong-v4.0` | `CDPKG_SHENNONG_V4_API_KEY` | `CDPKG_SHENNONG_V4_BASE_URL` | `CDPKG_SHENNONG_V4_MODEL` (must be set) |
| `qwen3.7-plus` | `CDPKG_QWEN_API_KEY` | `CDPKG_QWEN_BASE_URL` | `CDPKG_QWEN_MODEL` (must be set) |

The code defaults are fallbacks, not claims about the exact API snapshot used
in a particular experiment. Set the model ID to the actual provider model you
use. No key is hardcoded. If no key is configured, the agricultural QA client
falls back to a deterministic mock mode for pipeline checks; retrieval
evaluation does not require generation.

### 2Wiki system

The 2Wiki scripts use a separate configuration in
[`common.py`](2wiki/experiments/2wiki/common.py):

- API model: `deepseek-chat` (fixed in `common.py`, not `CDPKG_LLM_MODEL`);
- API key: `CDPKG_LLM_API_KEY` from the environment or the agricultural
  `CDPKG/retrieval_qa/.env` fallback;
- base URL: `CDPKG_LLM_BASE_URL` (default DeepSeek);
- generation: `temperature=0.2`, `top_p=0.9`, `max_tokens=64`;
- embedding: `CDPKG/models/bge-base-en-v1.5` (768 dimensions).

Do not apply the agricultural system's provider switching or 2048-token
generation setting to the 2Wiki experiments.

---

## Data preparation

### Agricultural system

The knowledge bases, annotation data, extraction results and KG CSVs are not
included. Supply them in the expected schema, then run the following from
`CDPKG/`.

```bash
cd CDPKG

# 1. Re-split annotation data 8:1:1 (requires the external annotation roots).
python resplit_datasets.py --out 合并数据811 --seed 2019

# 2. Train NER (run from CDPKG/命名实体识别).
cd 命名实体识别
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

# 3. Train RE (run from CDPKG/关系抽取; paths are fixed in the script).
cd ../关系抽取
python roberta_copet_casrel_merged811.py

# 4. Extract entities/relations from the two knowledge bases.
cd ..
python extract_knowledge_merged.py \
  --excel 病害知识库.xlsx \
  --columns 病害症状,发病规律,防治方法 \
  --output results_merged
python extract_knowledge_merged.py \
  --excel 虫害知识库.xlsx \
  --columns 病害症状,发病规律,防治方法 \
  --output results_merged

# 5. Build the 27-entity / 20-relation KG CSVs.
python build_kg_29x20.py
```

Set `DATA_ROOT` if the raw annotation directories are not at the default
location expected by `resplit_datasets.py`. The `虫害数据/` sources are resolved
relative to the working directory, so run the script from `CDPKG/`. See
[`CDPKG/README.md`](CDPKG/README.md) for the ontology, file schemas and exact
execution order.

### 2WikiMultiHopQA

The 2Wiki raw parquet files and processed data are not included. They can be
downloaded and rebuilt with the scripts below.

```bash
cd 2wiki/experiments/2wiki

python download_data.py       # validation.parquet + train_0.parquet via hf-mirror
python prepare_data.py        # 400 test samples + 40 few-shot samples
python build_text_index.py    # sentences, BGE embeddings, lookup
python build_graph.py         # question-local mention graphs
```

The downloader uses the public `datadream/wiki_hop` parquet mirror. The
scripts write to `2wiki/data/`, which is git-ignored.

---

## Running the agricultural system

All commands below are run from `CDPKG/retrieval_qa/`. Do **not** add a
`retrieval_qa/` prefix when already in that directory. Detailed architecture
and evaluation notes are in
[`CDPKG/retrieval_qa/README.md`](CDPKG/retrieval_qa/README.md).

```bash
cd CDPKG/retrieval_qa

# Build segment-level text blocks, FAISS, BM25 and graph caches.
python 1_build_index.py

# Interactive or single-question demo.
python 2_demo_cli.py -e "水稻叶片有梭形病斑是什么病？"

# Generate the 480 auto-candidate QA set and the 180 auto-candidate few-shot pool.
python generate_qa_test.py
python generate_fewshot_pool.py --per_task 30

# Retrieval evaluation: one detailed method or all eight methods.
python 3_evaluate.py --metrics retrieval --methods full
python 3_evaluate.py --metrics retrieval --methods all

# End-to-end QA evaluation (first 20 questions for a smoke test).
python 3_evaluate.py --metrics qa --subset 20

# Resumable end-to-end evaluation for one formal variant.
python run_eval_resumable.py --variant full --provider deepseek

# Run full model + seven ablations + eight QA baselines.
python run_all_variants.py --provider deepseek

# Print comparison/ablation tables from the v2 summaries.
python make_final_table.py
```

`run_all_variants.py` currently launches child processes through
`CDPKG/.venv-re/Scripts/python.exe` (a Windows virtual-environment path). If
that interpreter is unavailable, run `run_eval_resumable.py --variant ...`
directly with your own Python for each variant.

Optional GraphRAG preprocessing (needed for community-based GraphRAG-Global and
for `table3_graph_metrics.py`):

```bash
python build_communities.py --method louvain   # or leiden when cdlib is installed
python build_summaries.py --subset 3           # requires LLM configuration
python generate_global_qa.py                   # target ~90 global questions
python table3_graph_metrics.py                 # requires communities + summaries
```

Optional parameter scans:

```bash
python param_sweep_retrieval.py --param graph_hop
python param_sweep_retrieval.py --param all
python param_sweep_gen.py --param text_top_k --subset 50
```

`param_sweep_retrieval.py` uses the same atomic retrieval metrics as
`3_evaluate.py`; `param_sweep_gen.py` runs generation-side scans and reports
BLEU/ROUGE/BERTScore/Gscore generation metrics.

---

## Running the 2Wiki experiment

Run from `2wiki/experiments/2wiki/` (the scripts use absolute project paths):

```bash
cd 2wiki/experiments/2wiki

python validate_protocol.py
python run_method.py --method ours_full --limit 2
python run_all.py
python run_ours_ablation.py --variant pure_graph8_fs_type
python evaluate.py
python bootstrap.py
python summary.py
```

`run_method.py` supports `--method`, `--limit`, `--start` and `--force`.
`run_all.py` supports `--limit`, `--force` and `--method`.
`run_ours_ablation.py` requires `--variant` and supports `--limit`, `--start`
and `--force`.

The default `evaluate.py` tables report **Answer EM / Answer F1**, per-type
Answer EM/F1, and token/latency statistics. The file also contains support and
joint metric helper functions, but they are not part of the default CSV
outputs. `bootstrap.py` computes a paired bootstrap confidence interval
(1,000 resamples, seed 42).

`plot_results.py` is optional and requires an external workbook containing
`Overall` and `Task-wise` sheets; no such workbook is included in the
repository. Specify it explicitly:

```bash
python plot_results.py --xlsx /path/to/results.xlsx --out /path/to/charts
```

See [`2wiki/README.md`](2wiki/README.md) for the nine methods, sampling
configuration, evidence budget and output files.

---

## Unit tests

Run from the repository root:

```bash
python -m unittest discover -s tests -p "test_multigraph_atomic_protocol.py" -v
```

The tests cover MultiDiGraph relation storage, duplicate-triple provenance,
path direction/relation variants, graph reranking metadata, unified atomic
evidence merging, Gold matching, P/R/F1/NDCG, eight-method protocol consistency
and downstream MultiDiGraph compatibility. They do not replace a full
end-to-end experiment and do not require a real LLM or BGE model.

---

## Evaluation protocols at a glance

| Aspect | Agricultural main system | 2Wiki external validation |
|---|---|---|
| Retrieval methods | 8 methods: NaiveRAG, Hybrid RAG, GraphRAG (Local Search), LightRAG, G-Retriever, PathRAG, SG-RAG, Ours-Retrieval | 9 comparison methods (1 no-retrieval + 8 retrieval-augmented) |
| Retrieval metrics | P@3, R@3, F1@3, NDCG@3 over unified atomic evidence | Answer EM/F1; support/joint helpers exist but default tables are Answer EM/F1 |
| QA metrics | BLEU-4, ROUGE-L, BERTScore; resumable runner also reports BLEU-1/2, ROUGE-1/2 and Gscore | Answer EM/F1, per-type EM/F1, token/latency, bootstrap CI |
| Ablations | 7: text RRF, graph rerank, semantic graph, entity graph, evidence prompt, few-shot, diagnostic constraints | Separate ablation variants in `run_ours_ablation.py`, not the same seven |
| Data scale | 480 QA candidates, 180 few-shot candidates | 400 test samples, 40 few-shot samples |
| Graph | MultiDiGraph crop-pest KG | Question-local article mention graphs without relation labels |

---

## Implementation notes and known limitations

- The agricultural QA/few-shot files in `data/` are **auto-generated
  candidates** with `review_status=pending`; no manually approved benchmark is
  included. Use them for pipeline validation, not as a claim of completed human
  annotation.
- `generate_fewshot_pool.py --semantic_audit` requires an external
  `evidence_prompting_v3.py` that is not included; the default command omits
  that flag.
- `run_all_variants.py` uses the hard-coded Windows `.venv-re` child
  interpreter described above.
- `table3_graph_metrics.py` needs the GraphRAG community and summary files;
  build them first.
- The 2Wiki graph is a mention graph with no relation labels. Its structural
  path score uses directed mention edges and falls back to an undirected view
  when no directed path exists.
- The retrieval index must be rebuilt after changing the extraction data or
  index construction code. `3_evaluate.py` warns when it detects an index
  without segment-level `meta.segment_id`.
- No full experiment is run by the README commands, and no metric values are
  asserted here.

---

## License

Released under the MIT License. See [LICENSE](LICENSE).

---

## 中文简介

本项目为论文配套代码，包含两个**分别实现**的实验系统：

- **`CDPKG/`**：农业病虫害知识抽取、27 类实体 / 20 类关系知识图谱构建、
  三视图表示、四路检索、文本 RRF 与图谱结构—语义联合重排序，以及证据驱动
  问答和统一原子证据评估。
- **`2wiki/`**：在 2WikiMultiHopQA 英文多跳数据集上的适配实验，采用独立的
  句级语料、mention 图、BGE-base-en 嵌入、证据预算和生成参数，不能与农业
  主系统的实现或参数混为一谈。

仓库只提供代码。知识库、标注数据、微调权重、嵌入模型、索引、生成的
QA/few-shot 文件、预测和结果表均未随仓库提供；请按各子目录 README 准备
所需输入。自动生成的 480 条 QA 与 180 条 few-shot 当前均为
`review_status=pending`，不代表已经完成人工审核。API Key 请通过环境变量或
`CDPKG/retrieval_qa/.env` 提供，代码中不包含任何密钥。
