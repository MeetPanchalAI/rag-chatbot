# Prompts

Every instruction this system sends to a model. Edit a file and the next
question uses it — no restart.

| File | Sent by | Job |
| --- | --- | --- |
| `answer.txt` | `app/generation.py` | Answer from the numbered evidence, or refuse |
| `answer_retry.txt` | `app/generation.py` | Asks again after unparseable output |
| `rewrite_query.txt` | `app/retrieval.py` | Turns a follow-up into a standalone query |
| `rerank.txt` | `app/rerank.py` | Orders candidate passages by usefulness |
| `judge.txt` | `app/judge.py` | Scores an answer during evaluation |

Each file is the **system message**. The user message is assembled in code, because
it is runtime data — the evidence, the conversation, the candidate passages —
under fixed labels like `Evidence:` and `Question:`.

## Rules an edit must not break

**`answer.txt` must keep two things.** The model cites *evidence numbers*, never
page numbers: we resolve those numbers against the chunks we retrieved, which is
what stops an invented page reaching the user. And it must be able to set
`answerable` to false, which is how refusal works at all.

**Four prompts must contain the word "json".** `answer`, `answer_retry`, `rerank`
and `judge` are sent with JSON mode on, and the API requires the word to appear
in the messages. The reply shape must also match what the code parses:

| Prompt | Shape the code expects |
| --- | --- |
| `answer.txt` | `{"answer": string, "answerable": boolean, "citations": [int]}` |
| `rerank.txt` | `{"order": [int, ...]}` |
| `judge.txt` | `{"correctness": int, "groundedness": int, "citation_support": int, "completeness": int, "reason": string}` |

**`rewrite_query.txt` must return a bare query.** Anything longer than roughly six
times the question is thrown away and the original question is used instead.

**`judge.txt` scores 0, 1 or 2.** A number outside that range voids the whole
verdict and leaves the question unscored.

Tests in `tests/test_prompts.py` check each of these.

## Substitution

`answer_retry.txt` takes `{error}`. Substitution is a literal replace, not
`str.format`, because these prompts contain JSON examples with braces.

## After editing

Run the evaluation and compare against the run before it. Every run records the
settings it used, so a change in the numbers can be attributed:

```bash
python -m eval.run_eval
```
