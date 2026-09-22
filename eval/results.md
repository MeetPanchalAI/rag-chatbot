# Evaluation results

## Summary

| Metric | Result |
| --- | --- |
| Questions | 20 |
| Retrieval | 84% |
| Correctness | 90% |
| Groundedness | 92% |
| Citation support | 84% |
| Abstention | 100% |
| Unsupported-answer rate | 0% (0/4) |
| Answers judged | 20/20 |

## By category

| Category | n | Retrieval | Correctness | Groundedness | Citations | Abstention |
| --- | --- | --- | --- | --- | --- | --- |
| factual | 4 | 100% | 75% | 100% | 75% | n/a |
| multi passage | 4 | 75% | 75% | 88% | 62% | n/a |
| follow up | 4 | 62% | 100% | 100% | 100% | n/a |
| unanswerable | 4 | n/a | 100% | 75% | n/a | 100% |
| similar sections | 4 | 100% | 100% | 100% | 100% | n/a |

## Top retrieval score

- answerable: min 0.028, median 0.033, max 0.033 (n=16)
- unanswerable: min 0.030, median 0.031, max 0.033 (n=4)

## Per question

| ID | Doc | Type | Expected | Result | Recall | Corr | Grnd | Cite |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| q01 | free221 | factual | answer | refused | 100% | 0 | 2 | 0 |
| q02 | free221 | factual | answer | answered | 100% | 2 | 2 | 2 |
| q03 | free221 | factual | answer | answered | 100% | 2 | 2 | 2 |
| q04 | Lecture10 | factual | answer | answered | 100% | 2 | 2 | 2 |
| q05 | free221 | multi passage | answer | answered | 50% | 1 | 2 | 1 |
| q06 | free221 | multi passage | answer | answered | 100% | 2 | 2 | 2 |
| q07 | free221 | multi passage | answer | refused | 50% | 1 | 1 | 0 |
| q08 | Lecture10 | multi passage | answer | answered | 100% | 2 | 2 | 2 |
| q09 | free221 | follow up | answer | answered | 50% | 2 | 2 | 2 |
| q10 | free221 | follow up | answer | answered | 100% | 2 | 2 | 2 |
| q11 | free221 | follow up | answer | answered | 0% | 2 | 2 | 2 |
| q12 | Lecture10 | follow up | answer | answered | 100% | 2 | 2 | 2 |
| q13 | free221 | unanswerable | refuse | refused | - | 2 | 2 | 0 |
| q14 | free221 | unanswerable | refuse | refused | - | 2 | 2 | 0 |
| q15 | Lecture10 | unanswerable | refuse | refused | - | 2 | 1 | 0 |
| q16 | Lecture10 | unanswerable | refuse | refused | - | 2 | 1 | 0 |
| q17 | free221 | similar sections | answer | answered | 100% | 2 | 2 | 2 |
| q18 | free221 | similar sections | answer | answered | 100% | 2 | 2 | 2 |
| q19 | Lecture10 | similar sections | answer | answered | 100% | 2 | 2 | 2 |
| q20 | Lecture10 | similar sections | answer | answered | 100% | 2 | 2 | 2 |

## Worth a look

- **q01** (factual): What is the formal epsilon-delta definition in the notes for the limit of f(x) as x approaches a? - The response accurately notes that the retrieved excerpts omit the full two-sided epsilon-delta clauses, but it fails to provide the definition required by the reference and gives no citation.
- **q05** (multi passage): The notes say differentiability implies continuity, but the converse is false. Explain both statements using the example they give. - The response correctly explains both implications using a valid continuous-but-nondifferentiable example, but it does not use the reference example |x| and the cited pages only partially state the specific claims about ℓ.
- **q07** (multi passage): When f(x) changes sign on [a,b], how does the interpretation of the definite integral change, and how does the Fundamental Theorem give a computation for it? - The reply correctly states the Fundamental Theorem computation, but it omits—and incorrectly claims the material does not explain—the signed-area interpretation, and it provides no citations.
- **q09** (follow up): How would I apply the chain rule to sin(2x)? - The reply correctly identifies the composition and derives 2cos(2x), fully supported by the cited Page 57 evidence.
- **q11** (follow up): Then how do the notes use an antiderivative to compute a definite integral? - The reply correctly states that F is an antiderivative of f and that the definite integral from a to b equals F(b)−F(a), both directly supported by pages 99–100.
- **q12** (follow up): Then why can an astronaut in orbit appear weightless? - The reply correctly identifies constant free-fall and gravity as the orbital force, supported by the cited pages, but omits that the astronaut and spacecraft accelerate together.
- **q13** (unanswerable): A 5 N force moves an object 3 m at 30 degrees to the direction of motion. What work does the force do? - The reply appropriately abstains because the retrieved text lacks the angled-force work formula, though it provides no citation.
- **q14** (unanswerable): What is the period of a 2 m radius circular orbit if the object moves at 5 m/s? - The reply appropriately abstains because the retrieved evidence does not provide the circular-motion period relationship, but it includes no citation.
- **q15** (unanswerable): Give the formal epsilon-delta definition of the limit of f(x) as x approaches a. - The reply correctly abstains and indicates the retrieved Lecture10 evidence lacks the epsilon-delta definition, but it provides no citation and its document-wide absence claim is only partially supported by the excerpts.
- **q16** (unanswerable): What is the derivative of ln(x)? - The reply appropriately abstains, but it refers only to the retrieved evidence rather than explicitly stating that Lecture10 does not contain the derivative, and it provides no citation.
- **q19** (similar sections): What is the distinction between centripetal acceleration and centripetal force in uniform circular motion? - The response correctly distinguishes inward acceleration from the net inward force and gives both formulas, but it omits that gravity, tension, or friction may provide the centripetal force.
