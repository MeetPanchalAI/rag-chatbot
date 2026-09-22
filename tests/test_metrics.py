"""The five metrics behind the evaluation tiles."""

from app.evaluation import breakdown, headline, metrics, score_ranges, summarise


def row(**kwargs) -> dict:
    base = {
        "id": "q", "doc": "free221", "type": "factual", "question": "?",
        "answer": "a", "expected_answer": "a",
        "should_be_answerable": True, "answerable": True, "citations": ["Page 1"],
        "top_score": 0.8, "guards": [], "recall": 1.0, "cited_gold": 1.0,
        "unsupported": False, "judge_reason": "",
        "correctness": 2, "groundedness": 2, "citation_support": 2, "completeness": 2,
    }
    base.update(kwargs)
    return base


def refusal(**kwargs) -> dict:
    """An unanswerable question the system correctly refused."""
    defaults = {
        "type": "unanswerable", "should_be_answerable": False, "answerable": False,
        "recall": None, "cited_gold": None, "citations": [], "unsupported": False,
        "citation_support": None,
    }
    defaults.update(kwargs)
    return row(**defaults)


def tiles(rows: list[dict]) -> dict:
    return {t["label"]: t for t in headline(rows)}


def test_the_five_tiles_are_the_five_things_the_brief_asks_about():
    labels = [t["label"] for t in headline([row()])]

    assert labels == ["Retrieval", "Correctness", "Groundedness",
                      "Citation support", "Abstention"]


def test_every_metric_reads_higher_is_better():
    # The meters all fill the same way, so no metric may be one you want small.
    perfect = metrics([row(), refusal()])

    assert all(value == 1.0 for value in perfect.values())


def test_retrieval_is_the_share_of_gold_pages_found_not_a_hit_or_miss():
    # A two-passage question that found one passage is half right, not a success.
    half = metrics([row(recall=0.5)])

    assert half["retrieval"] == 0.5


def test_judge_scores_are_halved_onto_the_same_scale():
    partial = metrics([row(correctness=1, groundedness=0, citation_support=1)])

    assert partial["correctness"] == 0.5
    assert partial["groundedness"] == 0.0
    assert partial["citation_support"] == 0.5


def test_answering_an_unanswerable_question_lowers_abstention():
    leaked = refusal(answerable=True, unsupported=True, correctness=0, groundedness=0)

    result = metrics([row(), leaked])

    assert result["abstention"] == 0.0
    assert "1 of 1 unanswerable questions were answered anyway" in tiles([row(), leaked])["Abstention"]["detail"]


def test_an_unjudged_run_reports_retrieval_and_abstention_only():
    # Running without the judge must leave the answer metrics blank rather than
    # reporting a confident zero.
    unjudged = row(correctness=None, groundedness=None,
                   citation_support=None, completeness=None)

    result = metrics([unjudged, refusal(correctness=None, groundedness=None)])

    assert result["retrieval"] == 1.0
    assert result["abstention"] == 1.0
    assert result["correctness"] is None
    assert result["groundedness"] is None


def test_groundedness_can_be_low_while_correctness_is_high():
    # Right answer, but from the model's own knowledge rather than the evidence.
    result = metrics([row(correctness=2, groundedness=0)])

    assert result["correctness"] == 1.0
    assert result["groundedness"] == 0.0


def test_the_breakdown_reports_the_same_metrics_per_category():
    rows = [row(type="factual"), row(type="multi_passage", recall=0.5), refusal()]

    groups = {g["type"]: g for g in breakdown(rows)}

    assert [g["type"] for g in breakdown(rows)] == ["factual", "multi_passage", "unanswerable"]
    assert groups["factual"]["retrieval"] == 1.0
    assert groups["multi_passage"]["retrieval"] == 0.5
    assert groups["unanswerable"]["abstention"] == 1.0
    assert groups["factual"]["count"] == 1


def test_the_summary_states_the_unsupported_answer_rate_outright():
    rows = [row(), refusal(), refusal(answerable=True, unsupported=True)]

    summary = summarise(rows)

    assert summary["Unsupported-answer rate"] == "50% (1/2)"
    assert summary["Questions"] == "3"


def test_score_ranges_split_answerable_from_unanswerable():
    rows = [row(top_score=0.9), row(top_score=0.5), refusal(top_score=0.2)]

    ranges = {r["group"]: r for r in score_ranges(rows)}

    assert ranges["answerable"]["min"] == 0.5
    assert ranges["answerable"]["max"] == 0.9
    assert ranges["unanswerable"]["count"] == 1
