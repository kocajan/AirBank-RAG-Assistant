# Air Bank RAG evaluation

Retrieval relevance is evaluated at the source-document level. Because every generated question has exactly one known relevant source document, Hit@K is equivalent to Recall@K in this experiment.

```text
Embedding model             Chunking size/ovl       R@1     R@3     R@5    MRR@5   nDCG@5   Judge     >=4
---------------------------------------------------------------------------------------------------------
text-embedding-3-small      small     800/100    70.0%   93.3%   96.7%    0.814    0.853    4.83   96.7%
text-embedding-3-small      medium   1600/250    70.0%   90.0%   93.3%    0.796    0.830    4.73   93.3%
text-embedding-3-small      large    2400/400    80.0%   93.3%   96.7%    0.862    0.888    4.97  100.0%
text-embedding-3-large      small     800/100    83.3%  100.0%  100.0%    0.917    0.938    4.87   96.7%
text-embedding-3-large      medium   1600/250    83.3%   93.3%  100.0%    0.898    0.924    4.80   96.7%
text-embedding-3-large      large    2400/400    80.0%   93.3%  100.0%    0.872    0.904    4.90  100.0%
text-embedding-ada-002      small     800/100    80.0%   96.7%   96.7%    0.878    0.901    4.80   96.7%
text-embedding-ada-002      medium   1600/250    80.0%   93.3%   93.3%    0.867    0.884    4.80   96.7%
text-embedding-ada-002      large    2400/400    80.0%   93.3%   93.3%    0.856    0.875    4.87  100.0%
```

Best retrieval by MRR@5: **text-embedding-3-large__small** (MRR@5=0.917, R@5=100.0%).
Best end-to-end mean judge score: **text-embedding-3-small__large** (4.97/5).

Metric notes:
- R@1/R@3/R@5: fraction of questions whose known source document appears in the top K retrieved documents.
- MRR@5: rewards putting the correct source as high as possible in the first five results.
- nDCG@5: rank-sensitive retrieval score with one relevant source per question.
- Judge: mean LLM score from 1–5; >=4 is the fraction of answers judged essentially or fully correct.
