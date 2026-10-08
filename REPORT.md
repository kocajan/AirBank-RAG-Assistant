# Air Bank Public Information RAG Assistant

## 1. Purpose

This project is a small proof of concept for a retrieval-augmented generation (RAG) assistant over publicly available Air Bank information. The goal is not to reproduce a production banking assistant. The goal is to demonstrate the complete workflow behind a practical LLM application: collecting public data, cleaning and indexing it, retrieving relevant context, generating grounded answers, and evaluating different RAG configurations.

The assistant answers questions about topics such as accounts, cards, payments, savings, loans, mortgages, fees, and digital banking. It uses a PydanticAI agent backed by an OpenAI model. For factual Air Bank questions, the agent must retrieve information from the local knowledge base before answering. The application also includes simple guardrails: it does not access customer accounts, perform banking actions, request authentication secrets, or provide personalized financial or legal decisions.

This is an unofficial portfolio demonstration and uses only public Air Bank information.

## 2. Main idea

A basic chatbot can answer many questions from model memory, but that is not reliable enough for information that is specific, frequently updated, or legally important. RAG addresses this by separating the language model from the factual source of truth.

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
grounded answer + source links
```

The corpus is collected from Air Bank's public website and official downloadable documents. HTML and PDF content is normalized into the same document format and stored together with metadata such as title, URL, category, source type, and collection time.

For the final demo, the selected configuration is:

- embedding model: `text-embedding-3-large`
- chunk size: 800 characters
- overlap: 100 characters
- retrieval: top 5 chunks, with at most 2 chunks from one document

The vector store is deliberately simple: normalized embeddings are stored locally and cosine similarity is calculated with NumPy. For this proof of concept, a separate vector database would add operational complexity without providing meaningful value.

## 3. Evaluation approach

The evaluation was designed to compare RAG configurations rather than to claim production-level accuracy.

A synthetic evaluation dataset of 30 question-answer pairs was generated automatically. For each item:

1. one source document was randomly selected;
2. a contiguous excerpt was sampled from that document;
3. an LLM generated one factual Czech question and a concise ground-truth answer using only that excerpt;
4. the source document ID was stored as the known relevant source.

This makes it possible to evaluate retrieval automatically because the correct source document is known in advance.

### Retrieval experiment

Nine configurations were evaluated in a 3 × 3 grid.

Embedding models:

- `text-embedding-3-small`
- `text-embedding-3-large`
- `text-embedding-ada-002`

Chunking strategies:

- small: 800 characters / 100 overlap
- medium: 1600 / 250
- large: 2400 / 400

Retrieval was measured at the source-document level using:

- Recall/Hit@1
- Recall/Hit@3
- Recall/Hit@5
- MRR@5
- nDCG@5

Because every question has exactly one known relevant source document, Hit@K and Recall@K are equivalent in this experiment.

### End-to-end answer evaluation

The same configurations were also tested through the complete chatbot pipeline. The generated answer was compared with the synthetic ground-truth answer by an LLM judge using a 1–5 scale:

- 5: fully correct and complete
- 4: essentially correct with only a minor omission
- 3: partially correct
- 2: mostly incorrect
- 1: incorrect, irrelevant, or unjustified refusal

The report also records the proportion of answers scoring at least 4.

## 4. Final evaluation results

The following table contains the results from the final evaluation run.

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

The best retrieval configuration was **`text-embedding-3-large` with 800/100 chunking**. It achieved R@1 of 83.3%, perfect R@3 and R@5, MRR@5 of 0.917, and nDCG@5 of 0.938.

The best end-to-end mean judge score was achieved by **`text-embedding-3-small` with 2400/400 chunking**, with 4.97/5 and 100% of answers scoring at least 4.

The fact that these two winners differ is useful. Retrieval metrics measure whether the known relevant source is found and how highly it is ranked. Final-answer quality additionally depends on the exact passages returned, how much surrounding context is included, and how the language model interprets that context.

For the final demo, I selected **`text-embedding-3-large` with 800/100 chunking**. It provides the strongest retrieval results overall while maintaining very high end-to-end answer quality at 4.87/5. The 0.10 difference between the two highest judge averages is small on a 30-question benchmark and is not strong enough evidence to prefer a configuration with weaker retrieval.

## 5. Why the evaluation looks better than real usage

The measured results are intentionally useful for relative comparison, but they should not be interpreted as a production accuracy estimate.

The main reason is that the benchmark is easier than real user traffic. Questions are generated directly from the same source corpus that the retriever searches. The generator sees a source excerpt and creates a factual question from it, so the resulting vocabulary and concepts are naturally aligned with the source. Real users may use different terminology, incomplete descriptions, spelling mistakes, or vague references.

The experiment is also narrow by design. It excludes ambiguous questions, multi-turn behavior, out-of-scope requests, adversarial prompts, and questions that require combining several sources. Those are common sources of failure in a real assistant.

Another important limitation is that retrieval is evaluated at the **document level**, not at the exact answer-bearing chunk level. If the correct PDF appears in the top results but the retrieved chunk is from the wrong section of that PDF, the retrieval metric still counts the source as correct. This can make the retrieval scores optimistic for large documents.

The answer judge also introduces uncertainty. It is an LLM-based evaluator rather than a human reviewer. It is useful for scalable comparison, but it may accept semantically similar answers that a domain expert would consider incomplete or insufficiently precise.

Finally, the dataset contains only 30 questions. One question changes a percentage metric by roughly 3.3 percentage points. Small differences between configurations therefore should not be treated as statistically conclusive.

For these reasons, the correct interpretation of the benchmark is:

> The evaluation shows that the RAG pipeline works and provides a controlled way to compare retrieval and chunking configurations. It does not show that the assistant is 95–100% reliable on real customer questions.

## 6. What real-world evaluation would look like

For a production system, I would replace or complement the synthetic set with a larger human-written benchmark representative of actual user behavior. If privacy and access policies allowed it, anonymized real support/search queries would be especially valuable.

The evaluation set should include:

- natural paraphrases that do not reuse the source wording;
- short and underspecified questions;
- spelling mistakes and colloquial language;
- questions requiring information from multiple sources;
- time-sensitive questions where effective dates matter;
- questions for which the knowledge base does not contain an answer;
- multi-turn questions that depend on previous context;
- difficult PDF/table content;
- conflicting or superseded documents.

Retrieval labels should ideally be made at the **chunk or passage level**, not only at the source-document level. Human review of a subset of final answers would also be used to calibrate the LLM judge.

## 7. Improvements for a production version

The current system intentionally stays simple. If this were developed further, I would focus on measured failure modes rather than adding complexity by default.

The most likely improvements would be:

1. **Better retrieval evaluation.** Build a human-authored benchmark and label the exact relevant passages.
2. **Hybrid retrieval.** Combine dense semantic retrieval with lexical/BM25 search. Exact product names, fees, numbers, and legal terminology often benefit from lexical matching.
3. **Reranking.** Retrieve a wider candidate set and use a reranker to select the most relevant passages before sending context to the LLM.
4. **Structure-aware chunking.** Split documents using headings, sections, tables, and semantic boundaries instead of only character limits.
5. **Freshness and version metadata.** Track effective dates, prefer current documents, and automatically re-crawl/re-index changed sources.
6. **Source prioritization.** Prefer authoritative product pages and current contractual documents when several sources contain similar information.
7. **Grounding verification.** Validate that important claims and numbers in the final answer are supported by retrieved passages, and abstain when support is weak.
8. **Observability.** Store anonymized retrieval traces, response quality feedback, latency, model usage, and failure categories for continuous evaluation.
9. **Production session/storage design.** Replace the in-memory session store with a persistent or distributed store and introduce appropriate authentication, rate limits, monitoring, and privacy controls.

## 8. Reproducibility

The experiment is designed to be reproducible from frozen inputs.

The evaluation code fixes the complete 3 × 3 embedding/chunking grid, uses a fixed source-selection seed (`42`) when creating the synthetic evaluation dataset, records fingerprints for the processed corpus and dataset, and writes a `reproducibility_manifest.json` for the evaluation run. The environment is also locked with `uv.lock`.

For the closest reproduction of the results reported above, the repository should preserve:

- `data/processed/documents.jsonl`
- `data/processed/collection_summary.json`
- `evaluation/dataset.jsonl`
- `evaluation/dataset.manifest.json`
- `evaluation/results/reproducibility_manifest.json`
- the final evaluation result files
- `uv.lock`

With these frozen inputs, another user can install the same dependency set, rebuild all nine vector indexes, and rerun the same evaluation procedure.

There are two important limits to this reproducibility claim. First, collecting the Air Bank website again creates a **new corpus snapshot**. Public pages, interest rates, downloadable documents, and site structure can change. A fresh crawl therefore reproduces the methodology, but not necessarily the exact experiment.

Second, the QA generator, chatbot, embedding API, and judge use hosted models. LLM generation and judging are not guaranteed to be bit-for-bit deterministic, and hosted implementations may evolve over time.

The project therefore distinguishes between:

- **Reproducing the reported experiment:** reuse the frozen corpus and QA dataset from this run, install dependencies from `uv.lock`, rebuild the indexes, and rerun evaluation.
- **Replicating the methodology:** recollect the current Air Bank website, generate a new QA benchmark using the same seed and settings, and rerun the same evaluation grid.

The final results in this report came from a fresh end-to-end run in which the corpus and QA benchmark were regenerated before evaluation. They differ somewhat from an earlier development run. That is expected and demonstrates why the exact corpus and benchmark must be frozen when reporting reproducible numbers.

After this final run, its corpus, QA dataset, lock file, result files, and `reproducibility_manifest.json` should be archived or committed together. Those files define the frozen experiment behind the reported numbers.

## 9. Conclusion

The project demonstrates an end-to-end RAG workflow rather than only a chatbot interface. Public Air Bank data is collected and normalized, indexed with embeddings, retrieved through a PydanticAI tool, and used to ground LLM answers. A controlled evaluation compares nine embedding/chunking configurations and measures both retrieval and final-answer quality.

On the final evaluation run, **`text-embedding-3-large` with 800/100 chunking produced the strongest retrieval performance**, with 100% R@3 and R@5 and the best MRR@5 and nDCG@5 values. A different configuration, `text-embedding-3-small` with 2400/400 chunking, achieved the highest LLM-judge score. This illustrates that retrieval metrics and final-generation metrics measure related but different parts of the RAG pipeline.

The final demo therefore uses `text-embedding-3-large` with 800/100 chunking because it gives the strongest and most consistent retrieval behavior while maintaining very high end-to-end answer quality.

The most important limitation is also the most useful lesson from the experiment: strong synthetic RAG metrics do not automatically imply strong real-world behavior. A production system would require a harder, representative evaluation set, passage-level relevance labels, freshness controls, and iterative improvements based on observed failure cases.
