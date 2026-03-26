# mood_analyzer.py
"""
Rule based mood analyzer for short text snippets.

This class starts with very simple logic:
  - Preprocess the text
  - Look for positive and negative words
  - Compute a numeric score
  - Convert that score into a mood label
"""

import re
from typing import List, Tuple, Optional

from dataset import POSITIVE_WORDS, NEGATIVE_WORDS

# ---------------------------------------------------------------------------
# Emoji sentiment sets (module level so they are constructed once)
# ---------------------------------------------------------------------------

# Explicit emoji → sentiment mapping.
# Positive emojis add +1; negative emojis add -1.
_POSITIVE_EMOJIS = {
    "🔥", "😊", "😄", "😃", "😁", "❤️", "😍", "🥰",
    "😂", "🤣", "👍", "✨", "🎉", "🙌", "💪",
}
_NEGATIVE_EMOJIS = {
    "😢", "😭", "😡", "🤬", "💀", "🙄", "😞", "😔",
    "😒", "👎", "😤", "😩", "😫", "😣", "🥲",
}

# Union used in preprocess to decide which characters to pull out as tokens.
_SENTIMENT_EMOJIS = _POSITIVE_EMOJIS | _NEGATIVE_EMOJIS

# Words that negate the sentiment of the following token.
_NEGATION_WORDS = {
    "not", "never", "no", "don't", "dont", "didn't", "didnt",
    "can't", "cant", "isn't", "isnt", "wasn't", "wasnt",
    "won't", "wont", "hardly", "barely",
}


class MoodAnalyzer:
    """
    A very simple, rule based mood classifier.
    """

    def __init__(
        self,
        positive_words: Optional[List[str]] = None,
        negative_words: Optional[List[str]] = None,
    ) -> None:
        # Use the default lists from dataset.py if none are provided.
        positive_words = positive_words if positive_words is not None else POSITIVE_WORDS
        negative_words = negative_words if negative_words is not None else NEGATIVE_WORDS

        # Store as sets for faster lookup.
        self.positive_words = set(w.lower() for w in positive_words)
        self.negative_words = set(w.lower() for w in negative_words)

    # ---------------------------------------------------------------------
    # Preprocessing
    # ---------------------------------------------------------------------

    def preprocess(self, text: str) -> List[str]:
        """
        Convert raw text into a list of tokens the model can work with.

        Steps:
          1. Strip leading/trailing whitespace.
          2. Split on spaces.
          3. For each chunk, extract any Unicode emoji characters as their
             own tokens (so "fire🔥" becomes ["🔥", "fire"]).
          4. Remove emojis and punctuation from the remaining text, then
             lowercase and add as a word token if non-empty.
          5. Drop empty strings.
        """
        tokens: List[str] = []
        for chunk in text.strip().split():
            # Extract recognized sentiment emoji as individual tokens.
            for char in chunk:
                if char in _SENTIMENT_EMOJIS:
                    tokens.append(char)

            # Strip all non-word characters (including emoji) and lowercase.
            word = re.sub(r"[^\w]", "", chunk, flags=re.UNICODE).lower()
            if word:
                tokens.append(word)

        return tokens

    # ---------------------------------------------------------------------
    # Internal analysis helper
    # ---------------------------------------------------------------------

    def _analyze(self, text: str) -> Tuple[int, List[str], List[str]]:
        """
        Core analysis routine shared by score_text, predict_label, and explain.

        Returns (score, positive_hits, negative_hits).

        Improvements over the naive approach:
          - Negation handling: a negation word (e.g. "not") immediately before
            a sentiment word flips its contribution (+1 ↔ -1).
          - Emoji signals: common positive/negative emoji are recognized as
            sentiment tokens and contribute ±1 to the score.
        """
        tokens = self.preprocess(text)
        positive_hits: List[str] = []
        negative_hits: List[str] = []
        score = 0

        for i, token in enumerate(tokens):
            # Is the previous token a negation word?
            negated = i > 0 and tokens[i - 1] in _NEGATION_WORDS

            if token in _POSITIVE_EMOJIS:
                if negated:
                    negative_hits.append(token)
                    score -= 1
                else:
                    positive_hits.append(token)
                    score += 1
            elif token in _NEGATIVE_EMOJIS:
                if negated:
                    positive_hits.append(token)
                    score += 1
                else:
                    negative_hits.append(token)
                    score -= 1
            elif token in self.positive_words:
                if negated:
                    negative_hits.append(token)
                    score -= 1
                else:
                    positive_hits.append(token)
                    score += 1
            elif token in self.negative_words:
                if negated:
                    positive_hits.append(token)
                    score += 1
                else:
                    negative_hits.append(token)
                    score -= 1

        return score, positive_hits, negative_hits

    # ---------------------------------------------------------------------
    # Scoring logic
    # ---------------------------------------------------------------------

    def score_text(self, text: str) -> int:
        """
        Compute a numeric "mood score" for the given text.

        Positive words (and positive emojis) increase the score.
        Negative words (and negative emojis) decrease the score.
        Negation words immediately before a sentiment word flip its sign.

        See _analyze() for full implementation details.
        """
        score, _, _ = self._analyze(text)
        return score

    # ---------------------------------------------------------------------
    # Label prediction
    # ---------------------------------------------------------------------

    def predict_label(self, text: str) -> str:
        """
        Turn the numeric score for a piece of text into a mood label.

        Mapping:
          - Both positive and negative signals present → "mixed"
          - score > 0  → "positive"
          - score < 0  → "negative"
          - score == 0 → "neutral"

        The "mixed" check runs before the numeric thresholds so that a post
        like "Exhausted but proud" (score = 0 with hits on both sides) is
        labelled "mixed" rather than "neutral".
        """
        score, positive_hits, negative_hits = self._analyze(text)

        has_positive = len(positive_hits) > 0
        has_negative = len(negative_hits) > 0

        if has_positive and has_negative:
            return "mixed"
        elif score > 0:
            return "positive"
        elif score < 0:
            return "negative"
        else:
            return "neutral"

    # ---------------------------------------------------------------------
    # Explanations (optional but recommended)
    # ---------------------------------------------------------------------

    def explain(self, text: str) -> str:
        """
        Return a short string explaining WHY the model chose its label.

        Example:
          'Score = 2 (positive words: ["love", "great"]; negative words: [])'
        """
        score, positive_hits, negative_hits = self._analyze(text)
        label = self.predict_label(text)
        return (
            f"Score = {score}, label = {label!r} "
            f"(positive: {positive_hits}, "
            f"negative: {negative_hits})"
        )
