# Email Spam Detector Analysis

## Features

The model uses six numeric features from each email:

- `word_count`: message length.
- `exclamations`: number of `!` characters, which can indicate urgency.
- `money_words`: count of terms such as “free,” “cash,” “prize,” and “claim.”
- `all_caps`: number of all-uppercase words, which can be used to attract attention.
- `link_count`: number of URLs, which can direct readers to a website.
- `question_marks`: number of `?` characters, which can be used to prompt a reply.

Links and question marks were included because they can occur in spam designed
to attract clicks or responses. They also appear in ordinary messages, so none
of these features proves that an email is spam.

## Evaluation method and results

`prepare_data.py` starts with 16 manually created feature rows (8 spam and 8
ham) and creates 20 small random variations per original. The resulting 336
rows are not real emails. The script splits the original email IDs first and
keeps every variation of an original in the same split. This prevents a
variation of a training example from leaking into the test set.

The fixed, stratified split used 12 original emails (252 rows) for training and
4 originals (84 rows) for testing. The measured accuracy was **94.0%**.

Confusion matrix (rows are actual ham/spam; columns are predicted ham/spam):

|                 | Predicted ham | Predicted spam |
|-----------------|--------------:|---------------:|
| Actual ham      | 42            | 0              |
| Actual spam     | 5             | 37             |

- Spam correctly caught: **37**
- Legitimate messages correctly passed: **42**
- Legitimate messages incorrectly marked spam: **0**
- Spam messages missed: **5**

| Class | Precision | Recall |
|-------|----------:|-------:|
| Ham   | 0.89      | 1.00   |
| Spam  | 1.00      | 0.88   |

Spam precision is 37 / (37 + 0): every test message predicted as spam was spam.
Spam recall is 37 / 42: the model caught 37 of the 42 test spam rows.
Ham precision is 42 / (42 + 5), and ham recall is 42 / 42.

## Which error matters more?

For this project, I would prioritize catching more spam, because a missed scam
or malicious message can reach someone's inbox. The tradeoff is that a more
aggressive filter may incorrectly send a legitimate email to spam, where the
recipient might not notice it. One possible next experiment is to give spam a
higher class weight or tune the spam decision threshold, then compare spam
recall against ham precision on validation data. I have not applied or tested
that change here.

## What the model learned

The decision tree assigned importance **1.000** to `exclamations` and **0.000**
to each of the other five features in this run. In other words, it made all its
splits using exclamation count. That is a useful result to question: the model
did not rely on link count or question marks, even though those features were
part of the original reasoning. This likely reflects how the tiny synthetic
examples and their variations were constructed, not a general rule about spam.

## Attempts to fool the classifier

The program extracts the six feature counts from each sample message and then
prints the model's actual prediction. On this run:

| Intended message | Actual prediction | Result |
|------------------|-------------------|--------|
| Spam written as an ordinary account-summary notice | Ham | Fooled |
| Legitimate school fundraiser with attention-grabbing wording | Spam | Fooled |
| Spam written as a routine delivery update | Ham | Fooled |

These are demonstrations using simple word and URL counting, not a semantic
understanding of the messages. The second sample shows how a legitimate message
can be flagged when it uses attention-grabbing language.

## Data limitations and confidence

The split keeps all generated variations grouped with their source, and the
test set has no source IDs in common with training. However, the 336 rows come
from only 16 manually chosen feature summaries, and the test set contains just
four original emails. The variations are correlated synthetic data, not a
large independent collection of labeled email. Therefore, the **94.0% accuracy
is a result on this small exercise dataset, not a reliable estimate of
real-world email-filter performance**. More varied, independently labeled
emails and a larger held-out test set would be needed for that.
