"""Run the evaluation set from the command line.

    python -m eval.run_eval [--questions eval/questions.jsonl] [--doc-id ID]

The scoring lives in `app/evaluation.py`, so the command line and the UI run
exactly the same thing. Needs a populated index and a working API key, and
costs one or two model calls per question.
"""

import argparse
import json
from pathlib import Path

from app.config import get_settings
from app.evaluation import load_questions, run_evaluation, write_report
from app.main import Providers
from app.vector_store import VectorStore

ROOT = Path(__file__).resolve().parent


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the RAG evaluation set.")
    parser.add_argument("--questions", type=Path, default=ROOT / "questions.jsonl")
    parser.add_argument("--doc-id", default=None, help="restrict every query to one document")
    parser.add_argument("--out", type=Path, default=ROOT / "results.md")
    args = parser.parse_args(argv)

    settings = get_settings()
    store = VectorStore(settings.data_dir)
    store.load()
    if store.chunk_count == 0:
        print("The index is empty. Ingest a PDF first.")
        return 1

    providers = Providers(settings)
    questions = load_questions(args.questions)

    def progress(done: int, total: int) -> None:
        print("  {}/{}".format(done, total), end="\r")

    rows = run_evaluation(
        questions,
        store,
        providers.embedder,
        providers.llm,
        providers.rewrite_llm,
        settings,
        doc_id=args.doc_id,
        on_progress=progress,
    )

    write_report(rows, args.out)
    args.out.with_suffix(".jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )
    print("\nWrote {} and {}".format(args.out, args.out.with_suffix(".jsonl")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
