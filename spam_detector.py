import re
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


DATA_PATH = Path(__file__).resolve().parent / "data" / "email_features.csv"
FEATURES = [
    "word_count",
    "exclamations",
    "money_words",
    "all_caps",
    "link_count",
    "question_marks",
]
LABELS = ["ham", "spam"]
MONEY_WORDS = {
    "bonus",
    "cash",
    "claim",
    "credit",
    "discount",
    "free",
    "money",
    "offer",
    "prize",
    "refund",
    "urgent",
    "win",
    "winner",
}

CHALLENGE_EMAILS = [
    (
        "Spam written to look ordinary",
        "spam",
        "Hi Jordan, your account summary is ready. Review the details at "
        "https://account-review.example when you have a moment.",
    ),
    (
        "Legitimate school fundraiser with attention-grabbing wording",
        "ham",
        "IMPORTANT!!! The school fundraiser is offering a FREE prize drawing "
        "to support our class. Please review the event details at "
        "https://school.example and https://calendar.example. Questions?",
    ),
    (
        "Spam written as a routine delivery update",
        "spam",
        "Hello, the package you asked about is ready. Check the delivery "
        "details at https://parcel-status.example to see the next steps.",
    ),
]


def extract_features(message: str) -> dict[str, int]:
    words = re.findall(r"\b[\w'-]+\b", message)
    lower_words = re.findall(r"\b[a-z]+\b", message.lower())

    return {
        "word_count": len(words),
        "exclamations": message.count("!"),
        "money_words": sum(word in MONEY_WORDS for word in lower_words),
        "all_caps": len(re.findall(r"\b[A-Z]{2,}\b", message)),
        "link_count": len(
            re.findall(r"(?:https?://|www\.)[^\s]+", message, flags=re.IGNORECASE)
        ),
        "question_marks": message.count("?"),
    }


def main() -> None:
    if not DATA_PATH.is_file():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. Run prepare_data.py first."
        )

    df = pd.read_csv(DATA_PATH)
    required_columns = {"source_id", *FEATURES, "label"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing columns: {', '.join(sorted(missing_columns))}")

    original_emails = df[["source_id", "label"]].drop_duplicates()
    train_ids, test_ids = train_test_split(
        original_emails["source_id"],
        test_size=0.25,
        random_state=42,
        stratify=original_emails["label"],
    )
    train_df = df[df["source_id"].isin(train_ids)]
    test_df = df[df["source_id"].isin(test_ids)]

    model = DecisionTreeClassifier(random_state=42)
    model.fit(train_df[FEATURES], train_df["label"])
    predictions = model.predict(test_df[FEATURES])

    print(f"Original emails: {len(original_emails)}")
    print(f"Generated feature rows: {len(df)}")
    print(f"Train rows / source emails: {len(train_df)} / {len(train_ids)}")
    print(f"Test rows / source emails: {len(test_df)} / {len(test_ids)}")
    print(f"Accuracy: {accuracy_score(test_df['label'], predictions):.3f}")
    print("\nClassification report:")
    print(classification_report(test_df["label"], predictions, labels=LABELS))
    print("Confusion matrix (rows: actual ham/spam; columns: predicted ham/spam):")
    print(confusion_matrix(test_df["label"], predictions, labels=LABELS))

    print("\nDecision-tree feature importances:")
    for feature, importance in sorted(
        zip(FEATURES, model.feature_importances_),
        key=lambda item: item[1],
        reverse=True,
    ):
        print(f"  {feature}: {importance:.3f}")

    print("\nThree challenge emails (features are counted from the text):")
    for name, intended_label, message in CHALLENGE_EMAILS:
        feature_values = extract_features(message)
        prediction = model.predict(pd.DataFrame([feature_values], columns=FEATURES))[0]
        result = "FOOLED" if prediction != intended_label else "not fooled"
        print(f"\n{name} — intended {intended_label}, predicted {prediction}: {result}")
        print(f"  {message}")
        print(f"  Features: {feature_values}")

    print("\nTry your own email text; type 'quit' or 'exit' to stop.")
    while True:
        try:
            message = input("\nEmail: ").strip()
        except EOFError:
            print()
            break

        if message.lower() in {"quit", "exit"}:
            break
        if not message:
            print("Please enter an email, or type 'quit' to stop.")
            continue

        feature_values = extract_features(message)
        prediction = model.predict(pd.DataFrame([feature_values], columns=FEATURES))[0]
        print(f"Prediction: {'SPAM' if prediction == 'spam' else 'NOT SPAM'}")
        print(f"Counted features: {feature_values}")


if __name__ == "__main__":
    main()
