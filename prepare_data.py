import csv
import random
from pathlib import Path


FEATURES = (
    "word_count",
    "exclamations",
    "money_words",
    "all_caps",
    "link_count",
    "question_marks",
)

EMAILS = [
    {"word_count": 120, "exclamations": 5, "money_words": 4, "all_caps": 3, "link_count": 2, "question_marks": 0, "label": "spam"},
    {"word_count": 95, "exclamations": 4, "money_words": 3, "all_caps": 2, "link_count": 1, "question_marks": 1, "label": "spam"},
    {"word_count": 130, "exclamations": 6, "money_words": 5, "all_caps": 4, "link_count": 2, "question_marks": 0, "label": "spam"},
    {"word_count": 110, "exclamations": 4, "money_words": 4, "all_caps": 3, "link_count": 3, "question_marks": 1, "label": "spam"},
    {"word_count": 100, "exclamations": 5, "money_words": 3, "all_caps": 2, "link_count": 2, "question_marks": 0, "label": "spam"},
    {"word_count": 125, "exclamations": 7, "money_words": 6, "all_caps": 4, "link_count": 3, "question_marks": 1, "label": "spam"},
    {"word_count": 90, "exclamations": 3, "money_words": 2, "all_caps": 2, "link_count": 1, "question_marks": 0, "label": "spam"},
    {"word_count": 115, "exclamations": 5, "money_words": 4, "all_caps": 3, "link_count": 2, "question_marks": 1, "label": "spam"},
    {"word_count": 75, "exclamations": 0, "money_words": 0, "all_caps": 0, "link_count": 0, "question_marks": 0, "label": "ham"},
    {"word_count": 85, "exclamations": 1, "money_words": 0, "all_caps": 0, "link_count": 0, "question_marks": 0, "label": "ham"},
    {"word_count": 70, "exclamations": 0, "money_words": 0, "all_caps": 0, "link_count": 1, "question_marks": 0, "label": "ham"},
    {"word_count": 95, "exclamations": 1, "money_words": 0, "all_caps": 1, "link_count": 0, "question_marks": 1, "label": "ham"},
    {"word_count": 80, "exclamations": 0, "money_words": 0, "all_caps": 0, "link_count": 0, "question_marks": 0, "label": "ham"},
    {"word_count": 105, "exclamations": 1, "money_words": 1, "all_caps": 0, "link_count": 1, "question_marks": 0, "label": "ham"},
    {"word_count": 65, "exclamations": 0, "money_words": 0, "all_caps": 0, "link_count": 0, "question_marks": 0, "label": "ham"},
    {"word_count": 90, "exclamations": 1, "money_words": 0, "all_caps": 0, "link_count": 1, "question_marks": 0, "label": "ham"},
]

OUTPUT_PATH = Path(__file__).resolve().parent / "data" / "email_features.csv"
VARIATIONS_PER_EMAIL = 20
RANDOM_SEED = 42


def build_dataset() -> list[dict[str, int | str]]:
    rng = random.Random(RANDOM_SEED)
    rows: list[dict[str, int | str]] = []

    for source_id, email in enumerate(EMAILS):
        rows.append({"source_id": source_id, **email})
        for _ in range(VARIATIONS_PER_EMAIL):
            variation: dict[str, int | str] = {"source_id": source_id}
            for feature in FEATURES:
                change = rng.randint(-10, 10) if feature == "word_count" else rng.randint(-1, 1)
                variation[feature] = max(0, email[feature] + change)
            variation["label"] = email["label"]
            rows.append(variation)

    return rows


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    rows = build_dataset()
    fieldnames = ["source_id", *FEATURES, "label"]

    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(
        f"Created {len(rows)} feature rows from {len(EMAILS)} original emails "
        f"at {OUTPUT_PATH}."
    )
    print(
        f"Each original has {VARIATIONS_PER_EMAIL} small random variations; "
        "source_id keeps related rows together during evaluation."
    )


if __name__ == "__main__":
    main()
