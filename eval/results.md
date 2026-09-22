# Evaluation results

## Summary

| Metric | Result |
| --- | --- |
| Questions | 20 |
| Retrieval | 84% |
| Correctness | 85% |
| Groundedness | 88% |
| Citation support | 81% |
| Abstention | 75% |
| Unsupported-answer rate | 25% (1/4) |
| Answers judged | 20/20 |

## By category

| Category | n | Retrieval | Correctness | Groundedness | Citations | Abstention |
| --- | --- | --- | --- | --- | --- | --- |
| factual | 4 | 100% | 100% | 100% | 100% | n/a |
| multi passage | 4 | 75% | 62% | 62% | 25% | n/a |
| follow up | 4 | 62% | 100% | 100% | 100% | n/a |
| unanswerable | 4 | n/a | 75% | 75% | n/a | 75% |
| similar sections | 4 | 100% | 88% | 100% | 100% | n/a |

## Top retrieval score

- answerable: min 0.509, median 0.673, max 0.738 (n=16)
- unanswerable: min 0.211, median 0.397, max 0.516 (n=4)

## Per question

| ID | Doc | Type | Expected | Result | Recall | Corr | Grnd | Cite |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| q01 | free221 | factual | answer | answered | 100% | 2 | 2 | 2 |
| q02 | free221 | factual | answer | answered | 100% | 2 | 2 | 2 |
| q03 | free221 | factual | answer | answered | 100% | 2 | 2 | 2 |
| q04 | Lecture10 | factual | answer | answered | 100% | 2 | 2 | 2 |
| q05 | free221 | multi passage | answer | refused | 50% | 1 | 1 | 0 |
| q06 | free221 | multi passage | answer | answered | 50% | 1 | 0 | 0 |
| q07 | free221 | multi passage | answer | refused | 100% | 1 | 2 | 0 |
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

- **q05** (multi passage): The notes say differentiability implies continuity, but the converse is false. Explain both statements using the example they give. - It correctly states differentiability implies continuity, but omits the required |x| example and the explanation of its continuity and failure to be differentiable, and provides no citations.
- **q06** (multi passage): For f(x)=x^3-x, where are the local minimum and local maximum, and why are neither global extrema? - The locations and conclusion that neither point is global are correct, but the stated end behavior contradicts the cited evidence/reference and the citations do not support those limit claims.
- **q07** (multi passage): When f(x) changes sign on [a,b], how does the interpretation of the definite integral change, and how does the Fundamental Theorem give a computation for it? - The reply correctly gives the Fundamental Theorem computation but omits the required signed-area interpretation (area above the axis minus area below it), and it provides no citations.
- **q09** (follow up): How would I apply the chain rule to sin(2x)? - The reply correctly identifies the composition, applies the chain rule, and derives 2cos(2x), all directly supported by the cited Page 57 evidence.
- **q11** (follow up): Then how do the notes use an antiderivative to compute a definite integral? - The reply correctly states that F is an antiderivative of f and that the definite integral over [a,b] equals F(b)-F(a), fully supported by the cited pages 99–100.
- **q12** (follow up): Then why can an astronaut in orbit appear weightless? - The reply correctly explains constant free-fall and the absence of a supporting force, but it does not explicitly state that gravity supplies the centripetal force and accelerates the astronaut and spacecraft together.
- **q13** (unanswerable): A 5 N force moves an object 3 m at 30 degrees to the direction of motion. What work does the force do? - The reply appropriately abstains because the retrieved document lacks the angled-force work formula, though it provides no citation.
- **q14** (unanswerable): What is the period of a 2 m radius circular orbit if the object moves at 5 m/s? - The reply incorrectly uses an unstated circular-motion period relationship and outside knowledge instead of abstaining, and the cited pages do not explicitly provide the claimed period formula or calculation.
- **q15** (unanswerable): Give the formal epsilon-delta definition of the limit of f(x) as x approaches a. - The reply correctly abstains and states that the evidence does not provide the epsilon-delta definition, but it includes no citation despite making a document-content claim.
- **q16** (unanswerable): What is the derivative of ln(x)? - The reply correctly abstains and states that the retrieved Lecture10 evidence does not provide the derivative, but it gives no citation despite making a document-based claim.
- **q17** (similar sections): What is the relationship between the left-hand and right-hand limits at a point and the existence of the two-sided limit? - The response correctly states the iff relationship and the unequal-limit consequence, fully supported by page 26, but it does not explicitly explain that the left-hand limit approaches through values less than a and the right-hand limit through values greater than a.
- **q18** (similar sections): What is the difference between a stationary point, a local extremum, and a global extremum in the notes? - The reply correctly defines stationary, local, and global extrema and their relationships, but it omits the important point that a stationary point need not be an extremum.
- **q19** (similar sections): What is the distinction between centripetal acceleration and centripetal force in uniform circular motion? - The reply correctly distinguishes inward acceleration from the net inward force and gives both formulas, but it omits that gravity, tension, or friction can provide the centripetal force.
