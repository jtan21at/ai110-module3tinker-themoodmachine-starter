"""
Shared data for the Mood Machine lab.

This file defines:
  - POSITIVE_WORDS: starter list of positive words
  - NEGATIVE_WORDS: starter list of negative words
  - SAMPLE_POSTS: short example posts for evaluation and training
  - TRUE_LABELS: human labels for each post in SAMPLE_POSTS
"""

# ---------------------------------------------------------------------
# Starter word lists
# ---------------------------------------------------------------------

POSITIVE_WORDS = [
    "happy",
    "great",
    "good",
    "love",
    "excited",
    "awesome",
    "fun",
    "chill",
    "relaxed",
    "amazing",
    # extended vocabulary
    "proud",
    "hopeful",
    "joy",
    "wonderful",
    "fantastic",
    "grateful",
    "lucky",
    "perfect",
    "yay",
    "fire",   # slang: "this is fire" = excellent
    "sick",   # slang: "that was sick" = awesome
    "lit",    # slang: "the party was lit" = great
    "accomplished",
    "nice",
    "peaceful",
]

NEGATIVE_WORDS = [
    "sad",
    "bad",
    "terrible",
    "awful",
    "angry",
    "upset",
    "tired",
    "stressed",
    "hate",
    "boring",
    # extended vocabulary
    "exhausted",
    "disaster",
    "miserable",
    "worried",
    "horrible",
    "rough",
    "done",      # slang: "I'm so done" = fed up
    "frustrated",
    "dread",
    "worthless",
    "stuck",
]

# ---------------------------------------------------------------------
# Starter labeled dataset
# ---------------------------------------------------------------------

# Short example posts written as if they were social media updates or messages.
SAMPLE_POSTS = [
    "I love this class so much",
    "Today was a terrible day",
    "Feeling tired but kind of hopeful",
    "This is fine",
    "So excited for the weekend",
    "I am not happy about this",
]

# Human labels for each post above.
# Allowed labels in the starter:
#   - "positive"
#   - "negative"
#   - "neutral"
#   - "mixed"
TRUE_LABELS = [
    "positive",  # "I love this class so much"
    "negative",  # "Today was a terrible day"
    "mixed",     # "Feeling tired but kind of hopeful"
    "neutral",   # "This is fine"
    "positive",  # "So excited for the weekend"
    "negative",  # "I am not happy about this"
]

# New posts added for Parts 1–3.
# These cover slang, emojis, sarcasm, negation, and mixed feelings
# to stress-test the rule-based model and reveal its limitations.

SAMPLE_POSTS.extend([
    "This is lowkey fire 🔥",                                    # slang + emoji
    "I absolutely love being stuck in traffic 🙄",               # sarcasm  (BREAKER)
    "Exhausted but so proud of what I accomplished today",       # mixed emotions
    "That movie was sick, no cap",                               # modern slang
    "lol this is a complete disaster 💀",                        # ironic negativity
    "Not bad at all, actually pretty good",                      # negation
    "Honestly idk how I feel right now",                         # ambiguous / neutral
    "Today started rough but ended on a happy note",             # mixed arc
    "I'm so done with everything 😤",                           # slang + emoji
    "Could be worse I guess",                                    # understated neutral
])

# Human labels for the new posts above.
# Note: post 2 ("I absolutely love being stuck in traffic 🙄") is sarcasm.
# A human reads it as negative, but the rule-based model sees "love" as
# positive and 🙄 as negative, predicting "mixed" — an intentional failure
# documented in the model card (Part 3 breaker sentence).
TRUE_LABELS.extend([
    "positive",   # "This is lowkey fire 🔥"
    "negative",   # "I absolutely love being stuck in traffic 🙄" (sarcasm)
    "mixed",      # "Exhausted but so proud of what I accomplished today"
    "positive",   # "That movie was sick, no cap"
    "negative",   # "lol this is a complete disaster 💀"
    "positive",   # "Not bad at all, actually pretty good"
    "neutral",    # "Honestly idk how I feel right now"
    "mixed",      # "Today started rough but ended on a happy note"
    "negative",   # "I'm so done with everything 😤"
    "neutral",    # "Could be worse I guess"
])

# Quick sanity check — will raise an error early if lists fall out of sync.
assert len(SAMPLE_POSTS) == len(TRUE_LABELS), (
    f"SAMPLE_POSTS ({len(SAMPLE_POSTS)}) and TRUE_LABELS "
    f"({len(TRUE_LABELS)}) must be the same length."
)
