# Final reported evaluation results

This file records the final experiment values used in `README.md` and `REPORT.md`.
The machine-generated artifacts from the same run (`latest_summary.json`,
`latest_details.json`, and `reproducibility_manifest.json`) should be preserved
alongside this file in the published repository.

| Embedding model | Chunking | R@1 | R@3 | R@5 | MRR@5 | nDCG@5 | Judge | >=4 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| text-embedding-3-small | 800/100 | 70.0% | 93.3% | 96.7% | 0.814 | 0.853 | 4.83 | 96.7% |
| text-embedding-3-small | 1600/250 | 70.0% | 90.0% | 93.3% | 0.796 | 0.830 | 4.73 | 93.3% |
| text-embedding-3-small | 2400/400 | 80.0% | 93.3% | 96.7% | 0.862 | 0.888 | **4.97** | **100.0%** |
| **text-embedding-3-large** | **800/100** | **83.3%** | **100.0%** | **100.0%** | **0.917** | **0.938** | 4.87 | 96.7% |
| text-embedding-3-large | 1600/250 | 83.3% | 93.3% | 100.0% | 0.898 | 0.924 | 4.80 | 96.7% |
| text-embedding-3-large | 2400/400 | 80.0% | 93.3% | 100.0% | 0.872 | 0.904 | 4.90 | 100.0% |
| text-embedding-ada-002 | 800/100 | 80.0% | 96.7% | 96.7% | 0.878 | 0.901 | 4.80 | 96.7% |
| text-embedding-ada-002 | 1600/250 | 80.0% | 93.3% | 93.3% | 0.867 | 0.884 | 4.80 | 96.7% |
| text-embedding-ada-002 | 2400/400 | 80.0% | 93.3% | 93.3% | 0.856 | 0.875 | 4.87 | 100.0% |

Best retrieval by MRR@5: `text-embedding-3-large + 800/100`.

Best mean LLM-judge score: `text-embedding-3-small + 2400/400`.

Selected demo configuration: `text-embedding-3-large + 800/100`.
