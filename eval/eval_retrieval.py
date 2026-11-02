"""Code grader for retrieval: is the expected source page in the top-k results?

Run from the project root:  python eval/eval_retrieval.py --name baseline
No LLM involved - deterministic, free, and fast to re-run after every change.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, "src")
from retrieve import bm25_search, hybrid_search, vector_search  # noqa: E402

QUESTION_FILES = [Path("eval/test_set.json"), Path("eval/manual_questions.json")]
RUNS_DIR = Path("eval/runs")
K = 5
METHODS = {"vector": vector_search, "bm25": bm25_search, "hybrid": hybrid_search}


def load_questions():
    """Merge both files; fail loudly on duplicate ids instead of silently skipping one."""
    questions = []
    for path in QUESTION_FILES:
        questions += json.loads(path.read_text(encoding="utf-8"))
    ids = [q["id"] for q in questions]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise ValueError(f"Duplicate question ids: {sorted(dupes)}")
    return questions


def run_path(name, kind):
    """Return eval/runs/<name>_<kind>.json.

    Every run gets its own file, so we can compare runs (baseline vs. a change)
    instead of overwriting the previous results. Refuses to overwrite an existing run.
    """
    path = RUNS_DIR / f"{name}_{kind}.json"
    if path.exists():
        sys.exit(f"{path} already exists - pick another --name (or delete it).")
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    return path


def find_rank(results, q):
    """1-based position of the first result from the expected PDF + page, or None.

    We match on source + page (not chunk_id): a page can have several chunks,
    and any of them lets us cite the right page.
    """
    for rank, r in enumerate(results, start=1):
        if r["source"] == q["source"] and r["page"] == q["page"]:
            return rank
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="run name, e.g. baseline")
    out_file = run_path(parser.parse_args().name, "retrieval")
    questions = [q for q in load_questions() if q["source"] is not None]
    print(f"Evaluating {len(questions)} questions with a known source (top {K})\n")

    rows = []
    for q in questions:
        row = {"id": q["id"], "question": q["question"], "expected": f"{q['source']}, str. {q['page']}"}
        for name, search in METHODS.items():
            results = search(q["question"], K)
            row[name] = {
                "rank": find_rank(results, q),
                "found": [f"{r['source']}, str. {r['page']}" for r in results],
            }
        rows.append(row)

    # --- Summary table ---
    summary = {}
    for name in METHODS:
        ranks = [row[name]["rank"] for row in rows]
        summary[name] = {
            "hit@1": sum(r == 1 for r in ranks) / len(ranks),
            "hit@5": sum(r is not None for r in ranks) / len(ranks),
        }
    print(f"{'metoda':<8} | {'hit@1':>6} | {'hit@5':>6}")
    print("-" * 26)
    for name, s in summary.items():
        print(f"{name:<8} | {s['hit@1']:>6.0%} | {s['hit@5']:>6.0%}")

    # --- Misses: expected page not in top 5 ---
    print(f"\nPromašaji (tačna strana nije u top {K}):")
    any_miss = False
    for row in rows:
        for name in METHODS:
            if row[name]["rank"] is None:
                any_miss = True
                found = "; ".join(row[name]["found"][:3]) or "(ništa)"
                print(f"- [{row['id']}] {name}: {row['question']}\n"
                      f"    očekivano: {row['expected']}\n    nađeno:    {found}")
    if not any_miss:
        print("  (nema)")

    # --- Paraphrases vs their originals ---
    by_id = {row["id"]: row for row in rows}
    pairs = [(q["paraphrase_of"], q["id"]) for q in questions if q.get("paraphrase_of")]
    if pairs:
        print("\nParafraze vs originali (rang tačne strane, '-' = nije u top 5):")
        print(f"{'pitanje':<8} | " + " | ".join(f"{m:>6}" for m in METHODS))
        for orig, para in pairs:
            for qid in (orig, para):
                ranks = [by_id[qid][m]["rank"] for m in METHODS]
                print(f"{qid:<8} | " + " | ".join(f"{r if r else '-':>6}" for r in ranks))
            print()

    out_file.write_text(
        json.dumps({"k": K, "n_questions": len(rows), "summary": summary, "questions": rows},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Saved -> {out_file}")


if __name__ == "__main__":
    main()
