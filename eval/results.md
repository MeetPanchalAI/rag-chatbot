# Evaluation results

## Summary

| Metric | Result |
| --- | --- |
| Questions | 20 |
| Retrieval | 84% |
| Correctness | 85% |
| Groundedness | 88% |
| Citation support | 88% |
| Abstention | 75% |
| Unsupported-answer rate | 25% (1/4) |
| Answers judged | 20/20 |

## By category

| Category | n | Retrieval | Correctness | Groundedness | Citations | Abstention |
| --- | --- | --- | --- | --- | --- | --- |
| factual | 4 | 100% | 100% | 100% | 100% | n/a |
| multi passage | 4 | 75% | 62% | 62% | 50% | n/a |
| follow up | 4 | 62% | 100% | 100% | 100% | n/a |
| unanswerable | 4 | n/a | 75% | 75% | n/a | 75% |
| similar sections | 4 | 100% | 88% | 100% | 100% | n/a |

## Top retrieval score

- answerable: min 0.468, median 0.677, max 0.738 (n=16)
- unanswerable: min 0.211, median 0.397, max 0.516 (n=4)

## Per question

| ID | Doc | Type | Expected | Result | Recall | Corr | Grnd | Cite |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| q01 | free221 | factual | answer | answered | 100% | 2 | 2 | 2 |
| q02 | free221 | factual | answer | answered | 100% | 2 | 2 | 2 |
| q03 | free221 | factual | answer | answered | 100% | 2 | 2 | 2 |
| q04 | Lecture10 | factual | answer | answered | 100% | 2 | 2 | 2 |
| q05 | free221 | multi passage | answer | refused | 50% | 1 | 1 | 0 |
| q06 | free221 | multi passage | answer | answered | 50% | 0 | 0 | 0 |
| q07 | free221 | multi passage | answer | answered | 100% | 2 | 2 | 2 |
| q08 | Lecture10 | multi passage | answer | answered | 100% | 2 | 2 | 2 |
| q09 | free221 | follow up | answer | answered | 50% | 2 | 2 | 2 |
| q10 | free221 | follow up | answer | answered | 100% | 2 | 2 | 2 |
| q11 | free221 | follow up | answer | answered | 0% | 2 | 2 | 2 |
| q12 | Lecture10 | follow up | answer | answered | 100% | 2 | 2 | 2 |
| q13 | free221 | unanswerable | refuse | refused | - | 2 | 2 | 0 |
| q14 | free221 | unanswerable | refuse | answered | - | 0 | 0 | 0 |
| q15 | Lecture10 | unanswerable | refuse | refused | - | 2 | 2 | 0 |
| q16 | Lecture10 | unanswerable | refuse | refused | - | 2 | 2 | 0 |
| q17 | free221 | similar sections | answer | answered | 100% | 2 | 2 | 2 |
| q18 | free221 | similar sections | answer | answered | 100% | 1 | 2 | 2 |
| q19 | Lecture10 | similar sections | answer | answered | 100% | 2 | 2 | 2 |
| q20 | Lecture10 | similar sections | answer | answered | 100% | 2 | 2 | 2 |

## Worth a look

- **q05** (multi passage): The notes say differentiability implies continuity, but the converse is false. Explain both statements using the example they give. - The reply correctly explains differentiability implies continuity, but it omits the requested |x| counterexample and provides no citation.
- **q06** (multi passage): For f(x)=x^3-x, where are the local minimum and local maximum, and why are neither global extrema? - The response identifies the extrema and their non-global nature but gives cube roots instead of ±1/√3 and reverses the cited limit behavior, which is not supported by the cited page.
- **q09** (follow up): How would I apply the chain rule to sin(2x)? - The reply correctly identifies the composition and applies the chain rule to obtain 2cos(2x), fully supported by the cited Page 57 evidence.
- **q11** (follow up): Then how do the notes use an antiderivative to compute a definite integral? - The reply accurately states that one first finds an antiderivative F of f and then evaluates the definite integral as F(b)−F(a), fully supported by pages 99–100.
- **q13** (unanswerable): A 5 N force moves an object 3 m at 30 degrees to the direction of motion. What work does the force do? - The reply appropriately abstains because the retrieved document lacks the angled-force work formula, but it provides no citation.
- **q14** (unanswerable): What is the period of a 2 m radius circular orbit if the object moves at 5 m/s? - The reply gives an outside-knowledge calculation instead of abstaining, and the cited page only discusses the unit circle's length without supporting the stated period for a 2 m orbit at 5 m/s.
- **q15** (unanswerable): Give the formal epsilon-delta definition of the limit of f(x) as x approaches a. - The reply correctly abstains and states that the retrieved Lecture10 evidence does not provide the epsilon-delta definition, but it includes no citation.
- **q16** (unanswerable): What is the derivative of ln(x)? - The reply appropriately abstains and does not provide an unsupported calculus answer, but it refers only to the provided evidence rather than explicitly stating that Lecture10 does not contain the derivative, and it gives no citation.
- **q17** (similar sections): What is the relationship between the left-hand and right-hand limits at a point and the existence of the two-sided limit? - The reply correctly states the necessary and sufficient equality condition and its citations support it, but it omits that the left-hand limit approaches through values less than a and the right-hand limit through values greater than a.
- **q18** (similar sections): What is the difference between a stationary point, a local extremum, and a global extremum in the notes? - The reply correctly defines stationary, local, and global extrema and their relationship, but omits the important point that a stationary point need not be an extremum.
- **q20** (similar sections): How do positive work, negative work, and change in kinetic energy differ according to the lecture? - The reply correctly distinguishes positive and negative work and explains the effect of net work on kinetic energy, but it does not explicitly state the key equation ΔK = W_net.
