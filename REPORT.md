# Air Bank Public Information RAG Assistant

## 1. Purpose

This project is a small proof of concept for a Retrieval-Augmented Generation (RAG) assistant over publicly available Air Bank information.

The goal is not to build a production banking assistant, but to demonstrate the main workflow behind a practical LLM application:

- collecting and processing public data;
- chunking and embedding documents;
- semantic retrieval;
- grounded answer generation;
- prompt and guardrail design;
- evaluation of different RAG configurations.

The assistant can answer questions about topics such as accounts, cards, payments, savings, loans, mortgages, fees, and digital banking.

It is an unofficial portfolio project and uses only publicly available Air Bank data.

---

## 2. System design

A language model can answer many general questions from its internal knowledge, but this is unreliable for company-specific or frequently changing information.

RAG addresses this by retrieving relevant information from an external knowledge base before generating the answer.

The application follows this flow:

```text
Air Bank public pages and PDFs
        ↓
collection and cleaning
        ↓
chunking
        ↓
embeddings
        ↓
local vector index
        ↓
semantic retrieval
        ↓
PydanticAI agent
        ↓
grounded answer + sources
```

The data is collected from Air Bank's public website and selected official downloadable documents. HTML and PDF sources are converted into a common document format containing the text and metadata such as title, URL, source type, and collection time.

The final demo uses:

- **LLM:** OpenAI GPT-5.6 Luna
- **Embedding model:** `text-embedding-3-large`
- **Chunk size:** 800 characters
- **Chunk overlap:** 100 characters
- **Retrieval:** top 5 chunks, with at most 2 chunks from one document

The vector index is intentionally simple. Embeddings are stored locally and cosine similarity is calculated using NumPy. For a small proof of concept, introducing a dedicated vector database would add unnecessary complexity.

---

## 3. Evaluation approach

The evaluation is intentionally lightweight. It is **not sufficient for estimating production-level performance**.

Its main purpose is to demonstrate a complete RAG evaluation workflow and compare several retrieval configurations under the same conditions.

The benchmark is small and somewhat noisy, but because every configuration is evaluated using the same dataset and procedure, it can still provide, to some extent, useful information about their relative performance.

For generation tasks, including dataset generation, chatbot responses, and LLM-based judging, the project uses **OpenAI GPT-5.6 Luna** through the OpenAI API.

### Evaluation dataset

The evaluation dataset contains 30 automatically generated question-answer pairs.

For each item:

1. one source document is selected;
2. an excerpt is sampled from the document;
3. GPT-5.6 Luna generates a factual Czech question and a concise reference answer based only on the excerpt;
4. the source document is stored as the known relevant source.

This provides a simple ground truth for retrieval evaluation.

The generated questions were not manually validated and are mostly straightforward factual questions. The benchmark is therefore mainly useful for comparing configurations rather than estimating real-world chatbot accuracy.

### Retrieval experiment

Nine configurations were tested in a **3 × 3 grid**.

Three OpenAI embedding models were compared:

- **`text-embedding-3-small`** — a newer, smaller and more efficient embedding model;
- **`text-embedding-3-large`** — a newer, higher-capacity embedding model with 3072-dimensional vectors;
- **`text-embedding-ada-002`** — an older embedding model used as a legacy baseline.

These models were selected to compare:

- an older baseline;
- a modern efficient model;
- a modern higher-capability model.

All three are available through the OpenAI Embeddings API.

Three chunking strategies were tested:

- **small:** 800 characters / 100 overlap
- **medium:** 1600 / 250
- **large:** 2400 / 400

Retrieval quality was measured at the source-document level using:

- **Recall/Hit@1** — how often the correct source was ranked first;
- **Recall/Hit@3** — how often it appeared in the top 3;
- **Recall/Hit@5** — how often it appeared in the top 5;
- **MRR@5** — rewards placing the correct source higher in the ranking;
- **nDCG@5** — another rank-sensitive retrieval metric.

Because every evaluation question has exactly one known relevant source document, Hit@K and Recall@K are equivalent in this experiment.

### End-to-end answer evaluation

The same nine configurations were also evaluated through the complete chatbot pipeline.

For each question:

1. the RAG system retrieved relevant context;
2. the chatbot generated an answer;
3. GPT-5.6 Luna compared the generated answer with the reference answer.

The LLM judge assigned a score from 1 to 5:

- **5** — fully correct and complete
- **4** — essentially correct
- **3** — partially correct
- **2** — mostly incorrect
- **1** — incorrect or irrelevant

The percentage of answers scoring at least 4 was also recorded.

The judge score should only be treated as an approximate comparative signal, because the evaluator is itself an LLM and was not calibrated.

---

## 4. Results

| Embedding model | Chunking | R@1 | R@3 | R@5 | MRR@5 | nDCG@5 | Judge | Score ≥4 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| text-embedding-3-small | 800/100 | 70.0% | 93.3% | 96.7% | 0.814 | 0.853 | 4.83 | 96.7% |
| text-embedding-3-small | 1600/250 | 70.0% | 90.0% | 93.3% | 0.796 | 0.830 | 4.73 | 93.3% |
| text-embedding-3-small | 2400/400 | 80.0% | 93.3% | 96.7% | 0.862 | 0.888 | **4.97** | **100.0%** |
| **text-embedding-3-large** | **800/100** | **83.3%** | **100.0%** | **100.0%** | **0.917** | **0.938** | 4.87 | 96.7% |
| text-embedding-3-large | 1600/250 | **83.3%** | 93.3% | **100.0%** | 0.898 | 0.924 | 4.80 | 96.7% |
| text-embedding-3-large | 2400/400 | 80.0% | 93.3% | **100.0%** | 0.872 | 0.904 | 4.90 | **100.0%** |
| text-embedding-ada-002 | 800/100 | 80.0% | 96.7% | 96.7% | 0.878 | 0.901 | 4.80 | 96.7% |
| text-embedding-ada-002 | 1600/250 | 80.0% | 93.3% | 93.3% | 0.867 | 0.884 | 4.80 | 96.7% |
| text-embedding-ada-002 | 2400/400 | 80.0% | 93.3% | 93.3% | 0.856 | 0.875 | 4.87 | **100.0%** |

The strongest retrieval configuration was:

**`text-embedding-3-large` + 800/100 chunking**

It achieved:

- R@1: **83.3%**
- R@3: **100%**
- R@5: **100%**
- MRR@5: **0.917**
- nDCG@5: **0.938**

The highest LLM-judge score was achieved by:

**`text-embedding-3-small` + 2400/400 chunking**

with:

- mean judge score: **4.97/5**
- score ≥4: **100%**

The fact that these two configurations differ is useful, but the difference may also partly reflect noise in the small evaluation set and in the LLM-based judge. Retrieval metrics measure whether the known relevant source is found and how highly it is ranked, while final-answer quality also depends on the exact retrieved passages and how the language model uses them.

For the final demo, I selected **`text-embedding-3-large` with 800/100 chunking** because it provided the strongest retrieval performance overall while still achieving a very high end-to-end score of 4.87/5.

Given the small evaluation dataset, the difference between judge scores should not be considered statistically significant.

---

## 5. Limitations

This is deliberately a small proof of concept. The points below are only examples of the main limitations; a production system would require a much broader analysis and additional safeguards.

- **The evaluation dataset is small and not manually validated.** The questions are LLM-generated from source excerpts but were not individually checked for realism, correctness, difficulty, or whether the expected answer is the best possible one.

- **The evaluation questions are too easy and narrow.** They mostly test straightforward factual lookup. They do not cover unsupported questions, out-of-scope requests, multi-hop reasoning, inappropriate requests, ambiguous wording, conversational follow-ups, or realistic human phrasing with incomplete context, typos, shorthand, and imprecise terminology.

- **Retrieval is based on basic dense semantic search.** There is no hybrid lexical search, reranking, query reformulation, answer-aware retrieval, keyword matching, or other second-stage relevance processing.

- **Retrieval quality is evaluated at the whole-document level.** A document can count as correctly retrieved even when the exact answer-bearing passage was not returned.

- **Displayed sources are not independently filtered for usefulness.** The application displays retrieved sources in retrieval order, even when only one or two directly support the answer.

- **The knowledge base covers only part of Air Bank's public information.** The demo contains roughly 200 public pages and documents rather than the complete public website.

- **Guardrails are minimal.** The application mainly relies on prompt-level instructions and simple application logic.

- **The system relies on general-purpose hosted models.** Generation and embeddings use external OpenAI APIs rather than domain-specific or fine-tuned models, and hosted models may change over time.

- **Operational performance was not evaluated.** Latency, token usage, API cost, throughput, rate limits, infrastructure cost, and failure rates were not measured.

- **Conversation handling is simplified.** Sessions are stored in memory and are appropriate for a demo, not for a distributed production system.

These limitations also explain why the measured evaluation scores are significantly better than the assistant's expected performance on real customer traffic.

---

## 6. What I would improve for a production system

The next steps should be driven by observed failure cases rather than adding complexity by default.

The most important improvements would be:

1. **A realistic evaluation dataset** based on manually validated questions or anonymized real user queries.
2. **Passage-level relevance labels** instead of only document-level labels.
3. **Hybrid retrieval**, combining semantic embeddings with lexical/BM25 search.
4. **Reranking** of a larger candidate set before providing context to the LLM.
5. **Query reformulation** for unclear, incomplete, or conversational user questions.
6. **Structure-aware chunking** using headings, sections, tables, and document structure.
7. **Source and freshness prioritization**, especially for rates, contractual documents, and superseded information.
8. **Stronger grounding checks** to ensure important claims are directly supported by retrieved passages.
9. **Stronger guardrails** for unsupported, sensitive, or inappropriate requests.
10. **Observability and operational evaluation**, including latency, token usage, cost, failure rates, and user feedback.

---

## 7. Reproducibility

The experiment is designed to be reproducible from frozen inputs.

The repository stores:

- the processed Air Bank corpus;
- the generated evaluation dataset;
- the evaluation results;
- corpus and dataset fingerprints;
- the evaluation configuration;
- the dependency lock file (`uv.lock`);
- a `reproducibility_manifest.json`.

The evaluation code also uses a fixed source-selection seed (`42`).

For the closest reproduction of the reported results, the frozen corpus and evaluation dataset should be reused.

There are two important limitations to reproducibility:

1. **Air Bank's website can change.** Re-running data collection creates a new dataset snapshot and may produce different results.
2. **Hosted models can change.** LLM generation, embeddings, and LLM judging are not guaranteed to be bit-for-bit deterministic over time.

The project therefore distinguishes between:

- **reproducing the experiment** — using the frozen corpus and evaluation dataset;
- **replicating the methodology** — collecting current data and rerunning the same evaluation workflow.

The latter may produce different numbers while still following the same procedure.

---

## 8. Conclusion

This project demonstrates a complete RAG workflow rather than only a chatbot interface.

Public Air Bank data is collected and processed, divided into chunks, embedded, retrieved through semantic search, and provided to a PydanticAI agent to generate grounded answers.

A simple evaluation pipeline compares nine combinations of embedding models and chunking strategies.

On the final benchmark, **`text-embedding-3-large` with 800/100 chunking achieved the strongest retrieval performance**, including 100% R@3 and R@5.

A different configuration achieved a slightly higher LLM-judge score, demonstrating that retrieval quality and final-answer quality measure different parts of the RAG pipeline.

The final demo therefore uses `text-embedding-3-large` with 800/100 chunking.
