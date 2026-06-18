"""Convert HotpotQA distractor JSON to lab QAExample format."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def hotpot_to_qa(item: dict, idx: int) -> dict:
    context = [
        {"title": title, "text": " ".join(sentences)}
        for title, sentences in item["context"]
    ]
    hop_count = len(item.get("supporting_facts", []))
    if hop_count <= 2:
        difficulty = "easy"
    elif hop_count <= 4:
        difficulty = "medium"
    else:
        difficulty = "hard"
    return {
        "qid": str(item.get("_id", f"hotpot_{idx}")),
        "difficulty": difficulty,
        "question": item["question"],
        "gold_answer": item["answer"],
        "context": context,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/hotpot_dev_distractor_v1.json")
    parser.add_argument("--output", default="data/hotpot_dev_50.json")
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()

    raw = json.loads(Path(args.input).read_text(encoding="utf-8"))
    converted = [hotpot_to_qa(item, i) for i, item in enumerate(raw[: args.limit])]
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(converted, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(converted)} examples to {out}")


if __name__ == "__main__":
    main()
