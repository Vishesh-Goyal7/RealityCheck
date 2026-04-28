import json
import re
from pathlib import Path

import pandas as pd


def normalize_whitespace(text: str) -> str:
    return " ".join(str(text).split())


def parse_answer_array(text):
    """
    Parses TruthfulQA stringified arrays like:
    ['ans1' 'ans2' 'ans3']
    or regular quoted list variants.
    """
    if pd.isna(text):
        return []

    s = str(text).strip()
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]

    matches = re.findall(r"'([^']*)'|\"([^\"]*)\"", s, flags=re.DOTALL)
    values = []
    for single_q, double_q in matches:
        value = single_q if single_q else double_q
        value = normalize_whitespace(value)
        if value:
            values.append(value)
    return values


def parse_mc_targets(text):
    """
    Parses TruthfulQA multiple-choice target strings like:
    {'choices': array([...], dtype=object), 'labels': array([1,0,0,0], dtype=int32)}
    """
    if pd.isna(text):
        return {"choices": [], "labels": []}

    s = str(text)

    choices_match = re.search(
        r"'choices':\s*array\(\[(.*?)\],\s*dtype=object\)",
        s,
        flags=re.DOTALL,
    )
    labels_match = re.search(
        r"'labels':\s*array\(\[(.*?)\],\s*dtype=.*?\)",
        s,
        flags=re.DOTALL,
    )

    choices = []
    labels = []

    if choices_match:
        raw_choices = re.findall(
            r"'([^']*)'|\"([^\"]*)\"", choices_match.group(1), flags=re.DOTALL
        )
        choices = [normalize_whitespace(a if a else b) for a, b in raw_choices]

    if labels_match:
        labels = [int(x.strip()) for x in labels_match.group(1).split(",") if x.strip()]

    return {"choices": choices, "labels": labels}


def build_generation_records(gen_df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for idx, row in gen_df.iterrows():
        records.append(
            {
                "id": f"tqa_{idx + 1:04d}",
                "source_dataset": "TruthfulQA",
                "split": "validation",
                "task_type": "generation",
                "question_type": row.get("type"),
                "category": row.get("category"),
                "question": normalize_whitespace(row.get("question", "")),
                "best_answer": normalize_whitespace(row.get("best_answer", "")),
                "correct_answers": parse_answer_array(row.get("correct_answers")),
                "incorrect_answers": parse_answer_array(row.get("incorrect_answers")),
                "source": row.get("source"),
            }
        )
    return pd.DataFrame(records)


def merge_multiple_choice(gen_records: pd.DataFrame, mc_df: pd.DataFrame) -> pd.DataFrame:
    mc_df = mc_df.copy()
    mc_df["question"] = mc_df["question"].astype(str).map(normalize_whitespace)

    mc_records = []
    for _, row in mc_df.iterrows():
        mc1 = parse_mc_targets(row.get("mc1_targets"))
        mc2 = parse_mc_targets(row.get("mc2_targets"))
        mc_records.append(
            {
                "question": normalize_whitespace(row.get("question", "")),
                "mc1_choices": mc1["choices"],
                "mc1_labels": mc1["labels"],
                "mc2_choices": mc2["choices"],
                "mc2_labels": mc2["labels"],
            }
        )

    mc_clean = pd.DataFrame(mc_records)
    merged = gen_records.merge(mc_clean, on="question", how="left")

    for col in ["mc1_choices", "mc1_labels", "mc2_choices", "mc2_labels"]:
        merged[col] = merged[col].apply(lambda x: x if isinstance(x, list) else [])

    return merged


def write_outputs(df: pd.DataFrame, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "truthfulqa_processed.json"
    jsonl_path = output_dir / "truthfulqa_processed.jsonl"
    csv_path = output_dir / "truthfulqa_processed.csv"

    records = df.to_dict(orient="records")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    df.to_csv(csv_path, index=False)

    print(f"Saved JSON  : {json_path}")
    print(f"Saved JSONL : {jsonl_path}")
    print(f"Saved CSV   : {csv_path}")
    print(f"Total rows  : {len(df)}")


def main():
    base_dir = Path("Database")
    generation_csv = base_dir / "generation_validation.csv"
    multiple_choice_csv = base_dir / "multiple_choice_validation.csv"
    output_dir = base_dir / "truthfulqa_processed"

    gen_df = pd.read_csv(generation_csv)
    mc_df = pd.read_csv(multiple_choice_csv)

    gen_records = build_generation_records(gen_df)
    final_df = merge_multiple_choice(gen_records, mc_df)

    write_outputs(final_df, output_dir)


if __name__ == "__main__":
    main()
