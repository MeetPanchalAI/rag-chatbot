# Evaluation results

## Summary

| Metric | Result |
| --- | --- |
| Questions | 18 |
| Retrieval recall (any gold page found) | 27% (4/15) |
| Retrieval coverage (all gold pages found) | 7% (1/15) |
| Answered when it should be | 13% (2/15) |
| Refused when it should be | 100% (3/3) |
| Hallucination rate (answered the unanswerable) | 0% (0/3) |
| Answers with a citation | 100% (2/2) |
| Citation precision (cited page holds the answer) | 0% |
| Expected keywords present | 0% (0/15) |

## Top retrieval score

- answerable: min 0.081, median 0.171, max 0.271 (n=15)
- unanswerable: min 0.033, median 0.143, max 0.307 (n=3)

## Per question

| ID | Type | Answerable | Recall | Citations |
| --- | --- | --- | --- | --- |
| q01 | factual | no | hit | - |
| q02 | factual | no | miss | - |
| q03 | factual | no | miss | - |
| q04 | factual | no | miss | - |
| q05 | multi_passage | yes | miss | Page 24 - "W F d" |
| q06 | multi_passage | no | miss | - |
| q07 | multi_passage | no | miss | - |
| q08 | multi_passage | no | hit | - |
| q09 | follow_up | no | miss | - |
| q10 | follow_up | no | miss | - |
| q11 | follow_up | no | miss | - |
| q12 | unanswerable | no | n/a | - |
| q13 | unanswerable | no | n/a | - |
| q14 | unanswerable | no | n/a | - |
| q15 | similar_sections | no | hit | - |
| q16 | similar_sections | no | hit | - |
| q17 | similar_sections | yes | miss | Pages 12-13 - "N"; Pages 17-22 - "Work W is the energy transferred to or from an" |
| q18 | similar_sections | no | miss | - |

## Failures

- **q01** (factual): What should a citation show the user?
- **q02** (factual): How many questions should the evaluation set contain?
- **q03** (factual): What must not be hardcoded?
- **q04** (factual): Which file should describe how AI coding assistants were used?
- **q05** (multi_passage): What are the engineering requirements, and which optional extensions are suggested?
- **q06** (multi_passage): What should happen when the document cannot answer a question, and how should that be measured?
- **q07** (multi_passage): What must document ingestion do, and what technology is required to do it?
- **q08** (multi_passage): What has to be delivered, and how large should the evaluation set be?
- **q09** (follow_up): What should that explanation cover?
- **q10** (follow_up): How many questions should it have?
- **q11** (follow_up): What does the configuration one require?
- **q15** (similar_sections): How long is the work expected to take?
- **q16** (similar_sections): What does the brief say about using AI tools?
- **q17** (similar_sections): What example questions does the brief give?
- **q18** (similar_sections): What does the answer quality measure check?

## Guards triggered

- answerable_without_citations: 1
