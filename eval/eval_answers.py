"""Model grader for answers: a stronger model (Sonnet) judges what the app (Haiku) answered.

Run from the project root:  python eval/eval_answers.py --name baseline [--search vector]
Costs money (one Haiku + one Sonnet call per question) - check the estimate first.
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import anthropic
from dotenv import load_dotenv

sys.path.insert(0, "src")
from answer import answer, parse_response, plain_text  # noqa: E402
from eval_retrieval import load_questions, run_path  # noqa: E402  (same folder)

# Judge != app model: a model grading its own answers tends to be too lenient.
JUDGE_MODEL = "claude-sonnet-5-5"
REPEAT_NGRAM = 8  # 8 identical words in a row twice = the same claim written twice

JUDGE_ANSWERABLE = """Ti si strogi ocjenjivač odgovora chatbota koji odgovara iz fakultetskih dokumenata.
Dobićeš pitanje, očekivani (tačan) odgovor i odgovor chatbota. Uporedi SADRŽAJ, ne stil.

- TAČNO: odgovor sadrži ključne činjenice iz očekivanog odgovora i ništa mu ne protivrječi.
  Dodatni detalji su u redu ako ne protivrječe očekivanom odgovoru.
- DJELIMIČNO: dio ključnih činjenica je tačan, ali neka nedostaje ili je nejasna.
- NETAČNO: ključne činjenice su pogrešne, protivrječe očekivanom odgovoru, ili chatbot kaže da ne zna.

U polju "reason" u jednoj rečenici na bosanskom objasni ocjenu."""

JUDGE_UNANSWERABLE = """Ti si strogi ocjenjivač odgovora chatbota koji smije odgovarati samo iz fakultetskih dokumenata.
Na ovo pitanje odgovor NE POSTOJI u dokumentima. Dobićeš pitanje i odgovor chatbota.

- TAČNO: chatbot jasno kaže da ne zna / da informacija nije u dokumentima. Smije spomenuti srodne
  informacije ako jasno kaže da one ne odgovaraju na pitanje.
- HALUCINACIJA: chatbot daje konkretan odgovor (broj, datum, iznos, pravilo) kao da je odgovor na pitanje.

U polju "reason" u jednoj rečenici na bosanskom objasni ocjenu."""


def verdict_schema(options):
    # Structured outputs: the judge can only return one of these exact verdicts.
    return {
        "type": "object",
        "properties": {"verdict": {"type": "string", "enum": options}, "reason": {"type": "string"}},
        "required": ["verdict", "reason"],
        "additionalProperties": False,
    }


def judge(client, q, app_answer):
    if q["source"] is None:
        system, options = JUDGE_UNANSWERABLE, ["TAČNO", "HALUCINACIJA"]
        content = f"Pitanje: {q['question']}\n\nOdgovor chatbota:\n{app_answer}"
    else:
        system, options = JUDGE_ANSWERABLE, ["TAČNO", "DJELIMIČNO", "NETAČNO"]
        content = (f"Pitanje: {q['question']}\n\nOčekivani odgovor: {q['expected_answer']}"
                   f"\n\nOdgovor chatbota:\n{app_answer}")

    response = client.messages.create(
        model=JUDGE_MODEL,
        max_tokens=4000,
        system=system,
        messages=[{"role": "user", "content": content}],
        output_config={
            "format": {"type": "json_schema", "schema": verdict_schema(options)},
            "effort": "medium",  # enough to compare facts carefully; "high" mostly adds cost
        },
    )
    # If the judge is blocked we record it instead of silently switching to another judge model,
    # which would make the results inconsistent.
    if response.stop_reason == "refusal":
        return {"verdict": "GREŠKA", "reason": "Sudija je odbio ocijeniti."}, response.usage
    text = next(b.text for b in response.content if b.type == "text")
    return json.loads(text), response.usage


def repeated_phrase(text, n=REPEAT_NGRAM):
    """Code check: return the first n-word sequence that appears twice, else None.

    Catches "…jedanput poništiti ocjenu. Student ima pravo … jedanput poništiti ocjenu."
    even when there's no space after the period or the two sentences differ slightly.
    """
    words = re.findall(r"\w+", text.lower())
    seen = set()
    for i in range(len(words) - n + 1):
        gram = tuple(words[i:i + n])
        if gram in seen:
            return " ".join(gram)
        seen.add(gram)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="run name, e.g. baseline")
    parser.add_argument("--search", choices=["hybrid", "vector"], default="hybrid")
    args = parser.parse_args()
    out_file = run_path(args.name, "answers")  # check --name BEFORE spending money on API calls

    load_dotenv()
    client = anthropic.Anthropic()
    questions = load_questions()
    print(f"Evaluating {len(questions)} answers (search={args.search}, judge {JUDGE_MODEL})\n")

    rows, tokens = [], Counter()
    for i, q in enumerate(questions, start=1):
        response, chunks, _ = answer(q["question"], search=args.search)
        tokens["app_in"] += response.usage.input_tokens + (response.usage.cache_read_input_tokens or 0)
        tokens["app_out"] += response.usage.output_tokens

        if response.stop_reason == "refusal":
            app_answer, citations = "(app refused)", []
        else:
            app_answer = plain_text(response)
            _, citations = parse_response(response, chunks)

        result, usage = judge(client, q, app_answer)
        tokens["judge_in"] += usage.input_tokens
        tokens["judge_out"] += usage.output_tokens

        row = {
            "id": q["id"],
            "type": q["type"],
            "answerable": q["source"] is not None,
            "question": q["question"],
            "expected_answer": q["expected_answer"],
            "app_answer": app_answer,
            "cited": sorted({f"{c['source']}, str. {c['page']}" for c in citations}),
            "verdict": result["verdict"],
            "reason": result["reason"],
            "repeated_phrase": repeated_phrase(app_answer),
        }
        rows.append(row)
        print(f"  [{i:2d}/{len(questions)}] {q['id']:7} {row['verdict']:<12}"
              f"{'  (ponavljanje)' if row['repeated_phrase'] else ''}")

    # --- Summary ---
    answerable = [r for r in rows if r["answerable"]]
    unanswerable = [r for r in rows if not r["answerable"]]
    print(f"\nPitanja s odgovorom ({len(answerable)}):")
    for v in ["TAČNO", "DJELIMIČNO", "NETAČNO", "GREŠKA"]:
        n = sum(r["verdict"] == v for r in answerable)
        if n or v != "GREŠKA":
            print(f"  {v:<12} {n:3d}  ({n / len(answerable):.0%})")
    print(f"\nPitanja bez odgovora ({len(unanswerable)}):")
    for v in ["TAČNO", "HALUCINACIJA", "GREŠKA"]:
        n = sum(r["verdict"] == v for r in unanswerable)
        if n or v != "GREŠKA":
            print(f"  {v:<12} {n:3d}")
    n_rep = sum(bool(r["repeated_phrase"]) for r in rows)
    print(f"\nOdgovori s ponovljenom frazom ({REPEAT_NGRAM}+ riječi): {n_rep}/{len(rows)}")

    print("\nNije TAČNO:")
    for r in rows:
        if r["verdict"] != "TAČNO":
            print(f"- [{r['id']}] {r['verdict']}: {r['question']}\n    razlog: {r['reason']}")
    if n_rep:
        print("\nPonavljanja:")
        for r in rows:
            if r["repeated_phrase"]:
                print(f"- [{r['id']}] \"{r['repeated_phrase']}\"")

    print(f"\nTokeni: {dict(tokens)}")
    out_file.write_text(json.dumps({"judge_model": JUDGE_MODEL, "search": args.search,
                                    "tokens": dict(tokens), "answers": rows},
                                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved -> {out_file}")


if __name__ == "__main__":
    main()
