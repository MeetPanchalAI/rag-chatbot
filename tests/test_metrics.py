"""The numbers behind the evaluation tiles."""

from app.evaluation import headline, score_ranges


def row(**kwargs) -> dict:
    base = {
        "id": "q", "type": "factual", "question": "?", "answer": "a",
        "should_be_answerable": True, "answerable": True, "citations": ["Page 1"],
        "top_score": 0.8, "guards": [], "recall": True, "coverage": True,
        "citation_precision": 1.0, "keywords_found": True,
    }
    base.update(kwargs)
    return base


def tiles(rows: list[dict]) -> dict:
    return {t["label"]: t for t in headline(rows)}


def test_the_four_tiles_are_the_four_things_the_brief_measures():
    labels = [t["label"] for t in headline([row()])]

    assert labels == ["Retrieval", "Answer", "Citation", "Refusal"]


def test_every_tile_reads_higher_is_better():
    # The meters all fill the same way, so no tile may be a "lower is better"
    # number wearing a bar that grows as things get worse.
    perfect = tiles([row(), row(should_be_answerable=False, answerable=False,
                             recall=None, coverage=None, citation_precision=None,
                             keywords_found=None, citations=[])])

    assert all(t["value"] == 1.0 for t in perfect.values())


def test_a_hallucination_lowers_refusal_and_is_counted_in_the_detail():
    hallucinated = row(should_be_answerable=False, answerable=True, recall=None,
                       coverage=None, citation_precision=None, keywords_found=None)

    refusal = tiles([row(), hallucinated])["Refusal"]

    assert refusal["value"] == 0.0
    assert "1 answered that should not have been" in refusal["detail"]


def test_retrieval_separates_finding_one_page_from_finding_all_of_them():
    partial = tiles([row(recall=True, coverage=False)])["Retrieval"]

    assert partial["value"] == 1.0, "a gold page was found"
    assert "every gold page found for 0%" in partial["detail"]


def test_a_tile_with_nothing_to_measure_is_empty_not_zero():
    # An empty set must not render as a confident 0%.
    only_unanswerable = [row(should_be_answerable=False, answerable=False, recall=None,
                             coverage=None, citation_precision=None, keywords_found=None,
                             citations=[])]

    assert tiles(only_unanswerable)["Retrieval"]["value"] is None


def test_score_ranges_summarise_each_group():
    rows = [
        row(top_score=0.9), row(top_score=0.5), row(top_score=0.7),
        row(should_be_answerable=False, answerable=False, top_score=0.2,
            recall=None, coverage=None, citation_precision=None, keywords_found=None),
    ]

    ranges = {r["group"]: r for r in score_ranges(rows)}

    assert ranges["answerable"] == {
        "group": "answerable", "min": 0.5, "median": 0.7, "max": 0.9, "count": 3
    }
    assert ranges["unanswerable"]["count"] == 1


def test_a_group_with_no_scores_is_left_out_rather_than_plotted_empty():
    assert [r["group"] for r in score_ranges([row()])] == ["answerable"]
