# Model Card: Mood Machine

This model card covers **both** versions of the Mood Machine mood classifier:

1. A **rule-based model** (`mood_analyzer.py`)
2. A **machine-learning model** (`ml_experiments.py`, scikit-learn logistic regression)

---

## 1. Model Overview

**Model type:**  
Both models were built and compared.

**Intended purpose:**  
Classify short social-media-style text posts into one of four mood labels:
`positive`, `negative`, `neutral`, or `mixed`.

**How it works (brief):**

*Rule-based model* — Text is tokenized and each token is checked against a
hand-crafted list of positive words, negative words, and emoji signals.
Scores accumulate (+1 for positive, −1 for negative signals); a negation
word directly before a sentiment word flips its sign. Posts with both
positive and negative hits are labelled `mixed`; otherwise the sign of
the total score determines the label.

*ML model* — A bag-of-words representation (scikit-learn `CountVectorizer`)
is fit on all training posts and fed into a `LogisticRegression` classifier.
The model learns which word frequencies correlate with each label from the
labeled examples in `dataset.py`.

---

## 2. Data

**Dataset description:**  
The dataset contains **16 labeled posts** (6 starter examples + 10 new ones
added in Parts 1–3).

**Labeling process:**  
Each post was labeled manually by considering what a human reader would most
naturally interpret as the intended sentiment.  Posts with both positive and
negative cues (e.g. "Exhausted but so proud…") were labeled `mixed`.
Posts with no clear sentiment signal were labeled `neutral`.

Hard-to-label posts and edge cases:
- `"I absolutely love being stuck in traffic 🙄"` — sarcasm.  Everyone
  recognizes this as negative, but the literal words include "love."
- `"lol this is a complete disaster 💀"` — ironic negativity.  The skull
  emoji is Gen-Z slang for "dying of laughter," yet the overall tone is
  still negative.
- `"Could be worse I guess"` — understated; different people might call
  this `neutral` or `negative`.

**Important characteristics of the dataset:**
- Contains modern slang ("fire," "sick," "no cap," "lowkey," "done")
- Includes Unicode emoji (🔥 🙄 💀 😤)
- One sarcasm example that deliberately confuses the rule-based model
- Several mixed-emotion posts
- Several negation examples ("not happy," "not bad")

**Possible issues:**
- Very small (16 examples) — insufficient for reliable ML generalization
- All posts are in American English; slang may not transfer to other dialects
- Labels reflect one person's interpretation; reasonable people could disagree
  on several of them

---

## 3. How the Rule-Based Model Works

**Scoring rules:**

| Signal type | Effect |
|---|---|
| Token in `POSITIVE_WORDS` | score += 1 |
| Token in `NEGATIVE_WORDS` | score −= 1 |
| Token is a positive emoji (🔥 😊 😄 ❤️ …) | score += 1 |
| Token is a negative emoji (💀 🙄 😡 😤 …) | score −= 1 |
| Negation word immediately precedes a sentiment token | flip ±1 |

**Label mapping:**
- Both positive and negative hits present → `"mixed"`
- score > 0 → `"positive"`
- score < 0 → `"negative"`
- score == 0, no hits → `"neutral"`

**Key enhancements added (Part 2):**
1. *Negation handling* — words like "not," "never," "don't," "can't" flip the
   polarity of the immediately following sentiment word.  This correctly handles
   "I am **not** happy" (negative) and "**Not** bad, pretty good" (positive).
2. *Emoji signals* — `preprocess()` extracts emoji characters as standalone
   tokens so they contribute to the score.  🔥 adds +1; 🙄 adds −1.
3. *Extended vocabulary* — `POSITIVE_WORDS` and `NEGATIVE_WORDS` were expanded
   with common slang and emotion words (`proud`, `hopeful`, `fire`, `sick`,
   `exhausted`, `disaster`, `done`, `rough`, …).

**Strengths:**
- Fully transparent: every decision traces back to a specific token and rule
- Handles negation ("not bad" → positive)
- Responds to emoji sentiment
- No training data needed; deterministic and fast

**Weaknesses:**
- Cannot detect sarcasm ("I love being stuck in traffic" reads as positive/mixed)
- Slang polysemy: "sick" traditionally means unwell (negative) but in this
  vocabulary is treated as positive slang; context determines which meaning
  applies in reality
- Misses sentiment in unknown words (e.g., "wicked," "goated," brand names)
- Only looks one token back for negation; "I am really not that happy" could
  still fool it
- No understanding of sentence structure, emphasis, or irony

---

## 4. How the ML Model Works

**Features used:**  
Bag-of-words via `CountVectorizer` (raw token counts, no stop-word removal,
no TF-IDF weighting).

**Training data:**  
Trained on all 16 posts in `SAMPLE_POSTS` with labels from `TRUE_LABELS`.

**Training behavior:**  
With only 16 examples, the logistic regression model achieves **100% training
accuracy** — but this is training accuracy on data the model was fit on.  It is
almost certainly memorizing the examples rather than learning general patterns.
Adding even a handful of new or paraphrased posts would reveal overfitting.

**Strengths:**
- Automatically learns word–label associations from data
- Correctly classified the sarcasm post (`"I absolutely love being stuck in
  traffic 🙄"` → `negative`) because it memorized that exact sentence
- Does not require hand-crafted rules

**Weaknesses:**
- 100% training accuracy on 16 examples is a strong sign of overfitting
- The model has effectively memorized the training set; it would likely fail on
  any genuinely new post that uses different phrasing
- Sensitive to label choices: changing one label in the tiny dataset visibly
  shifts the model's behavior
- Cannot generalize to unseen emoji or slang that never appeared in training

---

## 5. Evaluation

**How the model was evaluated:**  
Both models were evaluated on the same 16 labeled posts in `dataset.py` using
prediction vs. true label comparisons in `main.py` and `ml_experiments.py`.

| Model | Accuracy (16 posts) |
|---|---|
| Rule-based | 0.94 (15/16) |
| ML (LogisticRegression) | 1.00 (16/16, training set) |

**Examples of correct predictions (rule-based):**

| Post | Predicted | True | Why it worked |
|---|---|---|---|
| `"I love this class so much"` | positive | positive | "love" is in POSITIVE_WORDS |
| `"I am not happy about this"` | negative | negative | "not" negates "happy" → −1 |
| `"Exhausted but so proud of what I accomplished today"` | mixed | mixed | "exhausted" (−1) + "proud" (+1) = 0 with both hit lists non-empty |

**Examples of incorrect predictions (rule-based):**

| Post | Predicted | True | Why it failed |
|---|---|---|---|
| `"I absolutely love being stuck in traffic 🙄"` | mixed | negative | Sarcasm: "love" (+1) and 🙄 (−1) cancel out; model cannot infer ironic intent |

The ML model got the sarcasm post correct, but only because it memorized the
exact text during training — not because it understands sarcasm.

---

## 6. Limitations

1. **Small dataset** — 16 examples cannot adequately cover the diversity of
   real human language.  Both models' reported accuracies are unreliable
   indicators of real-world performance.

2. **No sarcasm detection** — Sarcastic text contains literal positive words
   but an ironic negative intent.  The rule-based model is structurally
   incapable of detecting this without context or tone signals.

3. **Slang polysemy** — "Sick," "wicked," and "fire" can mean very different
   things depending on dialect and context.  Hard-coding them as positive slang
   will misclassify uses where they carry their traditional negative meaning
   (e.g., "I feel sick").

4. **Negation scope** — The model only looks at the immediately preceding
   token.  Multi-word negation ("I really don't feel very happy") may be
   partially missed.

5. **Emoji coverage** — Only a curated subset of emoji are recognized.
   Many common emoji (🤷, 🥴, 😬) are treated as neutral tokens.

6. **Training accuracy ≠ generalization** — The ML model's 100% score is
   inflated because it was evaluated on its own training data.

---

## 7. Ethical Considerations

- **Misclassifying distress** — A model that reads "I'm fine 🙂" as neutral
  (when it is actually a masked cry for help) could cause real harm in an
  application that monitors mental health or flags support needs.

- **Dialect and cultural bias** — The dataset contains only American-English
  Gen-Z slang.  Words like "sick" (positive slang) or "fire" (positive slang)
  would be misread in other dialects or age groups.  The model optimized for
  this specific voice and will systematically misinterpret users outside that
  demographic.

- **Privacy** — Mood classification of personal messages is sensitive.  Even
  an imperfect model deployed on private communications could expose or
  stigmatize users based on inferred emotional states.

- **Label subjectivity** — All labels were assigned by a single person.
  Different annotators would disagree on mixed or sarcastic posts.  Models
  trained on subjective labels inherit those biases.

---

## 8. Ideas for Improvement

- **More labeled data** — Even 100 diverse, human-labeled posts would
  significantly improve the ML model and reveal gaps in the rule-based lists.
- **Real train/test split** — Evaluate on held-out data, not the training set.
- **TF-IDF weighting** — Replace `CountVectorizer` raw counts with TF-IDF to
  downweight very common words.
- **Negation scope expansion** — Look ahead 2–3 tokens for the negated word,
  not just 1.
- **Sarcasm cues** — Add a small list of sarcasm markers (e.g., "oh great,"
  "absolutely love") whose presence overrides the literal word scores.
- **Emoji-to-word mapping** — Translate emoji to textual descriptions before
  tokenizing so the ML model can learn from them.
- **Small transformer** — Replace logistic regression with a fine-tuned
  DistilBERT or similar model that captures word order and context.
