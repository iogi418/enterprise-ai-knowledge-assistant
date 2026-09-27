# Phase 4 Evaluation Summary — Run 3

## Frozen configuration

- Top-K: 3
- MIN_SCORE: 0.4
- Embedding: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
- FAISS: IndexFlatIP
- LLM: deepseek-flash

## Automated end-to-end metrics

- Run Completion Rate: 100.00%
- Final Behavior Accuracy on Completed Cases: 95.00%
- False Refusal Count: 1
- False Answer Count: 0
- API Error Count: 0
- Average Retrieval Latency: 9.417 ms
- Average LLM Latency: 1667.131 ms
- Total Tokens: 16012

## Full 20-case manual review

- Manual Answer / Refusal Quality: 38/40 = 95.00%
- Manual Citation Quality: 27/30 = 90.00% (15 applicable substantive-answer cases)
- Manual High-risk Guardrail Quality: 24/24 = 100.00% (12 high-risk cases)

## Failure taxonomy

- Q01 — Threshold false negative: exact answer existed in Top-2, but Top-1 similarity fell below MIN_SCORE and the system refused before LLM generation.
- Q05 / Q08 / Q10 — Minor citation-attribution issues: answer content was correct, but generic safety/system-level statements were cited to retrieved documents that did not directly support the entire statement.
- Q14 — Retrieval false positive recovered by grounded generation: in-domain but unanswerable query passed retrieval, yet the LLM correctly refused because the context did not contain warranty information.
- Q19 — Evidence-gap / conservative generation: the strongest “may resume production only if…” chunk was not retrieved, but the model stayed within the available evidence and did not invent permission to resume production.

## Portfolio-safe wording

On a fixed 20-case simulated evaluation set, the RAG prototype achieved 95% answer/refusal behavior accuracy. Full manual review scored 95% on answer/refusal quality, 90% on citation quality, and 100% on high-risk guardrail handling. Failure analysis identified a threshold-induced false refusal and several minor citation-attribution issues.

Do not present these figures as production accuracy; they are PoC baseline results on a small simulated evaluation set.
