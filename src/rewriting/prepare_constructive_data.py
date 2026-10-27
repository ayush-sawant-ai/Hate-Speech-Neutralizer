"""Dataset preparation script for training a constructive T5 neutralizer.

Combines strictly filtered ParaDetox pairs with clean constructive feedback pairs
formatted with the standard rewriting instruction.
"""

from pathlib import Path
import re
from typing import List, Dict
import pandas as pd
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent.parent.parent
PARADETOX_CSV = BASE_DIR / "data" / "paradetox.csv"
OUTPUT_TRAIN_CSV = BASE_DIR / "data" / "constructive_train.csv"
OUTPUT_VAL_CSV = BASE_DIR / "data" / "constructive_val.csv"

INSTRUCTION = (
    "Rewrite the following message in respectful, constructive English. "
    "Remove personal attacks and preserve useful feedback or requests. "
    "Return only the rewritten message, using complete grammatical sentences. "
    "Do not add unsupported details."
)

PROFANITIES = [
    "shit", "fuck", "bitch", "idiot", "moron", "asshole", "bastard", "crap",
    "cunt", "dick", "nigger", "faggot", "retard", "dumbass", "jackass"
]

BROKEN_FRAGMENT_PATTERNS = [
    r"\bsuch a (?:improve|do|make|run|be|go|fixes|does)\b",
    r"\ba (?:improve|fixes|does|makes)\b",
    r"\bthe a\b",
    r"\bvery a\b",
    r"\byou are a (?:improve|fix|do)\b",
    r"\bkindly stop\b",
    r"\bnot a good person\b",
    r"\bbad person\b",
    r"\bsuch a bad\b",
    r"\bnobody cares\b",
    r"\bshut up\b",
    r"\.\s*\.",
    r"\[deleted\]",
    r"<extra_id_\d+>",
]


def is_clean_neutral(neutral_text: str, toxic_text: str) -> bool:
    """Validate that the target text is a complete, grammatical, non-toxic English sentence."""
    if not isinstance(neutral_text, str) or not neutral_text.strip():
        return False

    neutral = neutral_text.strip()
    toxic = toxic_text.strip()

    # Reject identical text
    if neutral.lower() == toxic.lower():
        return False

    # Reject if target contains overt profanity
    for p in PROFANITIES:
        if re.search(r"\b" + p + r"\b", neutral, re.IGNORECASE):
            return False

    # Reject obvious broken syntax patterns from word deletion
    for pattern in BROKEN_FRAGMENT_PATTERNS:
        if re.search(pattern, neutral, re.IGNORECASE):
            return False

    # Check length sanity: reject truncated fragments
    words = neutral.split()
    if len(words) < 3 and len(toxic.split()) > 5:
        return False

    # Must start with letter/quote and contain alphanumeric content
    if not re.search(r"[a-zA-Z]", neutral):
        return False

    return True


def format_sentence(s: str) -> str:
    """Ensure sentence has proper capitalization and punctuation."""
    s = s.strip()
    if not s:
        return s
    s = s[0].upper() + s[1:] if len(s) > 1 else s.upper()
    if not s.endswith((".", "!", "?")):
        s += "."
    return s


# High-quality constructive pairs covering diverse feedback, requests, criticisms & dialogue
CONSTRUCTIVE_EXEMPLARS: List[Dict[str, str]] = [
    # Work and code feedback with requests (matching target use cases)
    {"toxic": "you are such a shit improve your work", "neutral": "Please focus on improving the quality of your work."},
    {"toxic": "You are such a shit, improve your work.", "neutral": "Please focus on improving the quality of your work."},
    {"toxic": "You are such an idiot, improve your work right now.", "neutral": "Please focus on improving the quality of your work promptly."},
    {"toxic": "Your work is complete shit, redo it.", "neutral": "This work needs significant improvement. Please revise and redo it."},
    {"toxic": "This code is trash, fix it", "neutral": "This code needs improvement. Please revise it."},
    {"toxic": "This code is trash, fix it.", "neutral": "This code needs improvement. Please revise it."},
    {"toxic": "This code is garbage, clean it up immediately.", "neutral": "This code does not meet our standards. Please clean and refactor it."},
    {"toxic": "Who wrote this piece of shit code, it doesn't even compile.", "neutral": "This code has build errors and needs to be corrected."},
    {"toxic": "Your pull request is a dumpster fire, I'm closing it.", "neutral": "This pull request needs substantial revisions before it can be considered."},
    {"toxic": "Fix your fucking bug already, it's breaking production.", "neutral": "Please resolve this bug as soon as possible as it is impacting production."},
    {"toxic": "Your documentation is useless and full of crap.", "neutral": "This documentation lacks clarity and needs thorough revision."},
    {"toxic": "Stop writing garbage documentation, redo it immediately.", "neutral": "This documentation needs improvement. Please revise it."},
    {"toxic": "You broke the build again you clown.", "neutral": "The recent changes caused the build to fail; please investigate and fix it."},

    # Direct personal attacks with no actionable request -> Reframed into respectful disagreement or dissatisfaction
    {"toxic": "You are a complete idiot and nobody cares what you think.", "neutral": "I disagree with your opinion. Let's discuss the issue respectfully."},
    {"toxic": "You are a complete idiot and nobody cares what you have to say.", "neutral": "I disagree with your perspective. Let's discuss this matter respectfully."},
    {"toxic": "You are a total idiot and nobody cares about your views.", "neutral": "I disagree with your viewpoint. Let's discuss the issue constructively."},
    {"toxic": "Nobody cares what you think, you idiot.", "neutral": "I disagree with your viewpoint."},
    {"toxic": "Nobody cares about your opinion, you clown.", "neutral": "I hold a different perspective on this matter."},
    {"toxic": "You are a complete moron.", "neutral": "I strongly disagree with your assessment."},
    {"toxic": "You are an absolute idiot.", "neutral": "I disagree with your viewpoint on this matter."},
    {"toxic": "You are useless", "neutral": "I'm unhappy with your contribution."},
    {"toxic": "You are completely useless to this team.", "neutral": "Your current contribution is not meeting the team's expectations."},
    {"toxic": "You are pathetic and nobody likes you.", "neutral": "I am dissatisfied with your recent conduct."},
    {"toxic": "You are an incompetent clown who knows nothing.", "neutral": "I disagree with your approach and have concerns about these results."},

    # Direct criticisms of explanations, ideas, and arguments
    {"toxic": "Your explanation is stupid and makes no sense.", "neutral": "Your explanation is unclear. Please clarify your reasoning."},
    {"toxic": "Your explanation is completely idiotic and makes zero sense.", "neutral": "Your explanation lacks clarity. Please clarify your reasoning."},
    {"toxic": "This reasoning is stupid and makes no sense.", "neutral": "This reasoning is unclear. Please provide further clarification."},
    {"toxic": "Your idea is stupid", "neutral": "I don't think this idea will work well."},
    {"toxic": "Your idea is stupid.", "neutral": "I don't think this idea will work well."},
    {"toxic": "Your idea is completely moronic and won't work.", "neutral": "I have major concerns about the feasibility of this idea."},
    {"toxic": "Only an idiot would suggest something so brainless.", "neutral": "I strongly advise against this proposed course of action."},
    {"toxic": "This plan is ridiculous and will fail miserably.", "neutral": "This plan presents substantial risks and requires reconsideration."},
    {"toxic": "What a stupid question, look it up yourself.", "neutral": "Please consult the available documentation for the answer to that question."},
    {"toxic": "This proposal is pure nonsense from start to finish.", "neutral": "I disagree with the reasoning and recommendations in this proposal."},

    # Conversational hostility & Demands
    {"toxic": "Shut up and listen to me.", "neutral": "Please hear me out before responding."},
    {"toxic": "Shut up and listen to me", "neutral": "Please hear me out before responding."},
    {"toxic": "Shut up and listen", "neutral": "Please let me finish and hear me out."},
    {"toxic": "Shut your mouth and pay attention to what I'm saying.", "neutral": "Please allow me to explain without interruption."},
    {"toxic": "Stop talking bullshit and get to the point.", "neutral": "Please focus on the factual points and be concise."},
    {"toxic": "Stop whining like a baby and get this done.", "neutral": "Let's focus on resolving the challenges and completing this task."},
    {"toxic": "Quit your crying and do your job properly.", "neutral": "Please focus your attention on executing this assignment effectively."},
    {"toxic": "Get lost and leave me alone, you annoying pest.", "neutral": "I need some uninterrupted time to work on this right now."},
    {"toxic": "Stop bothering me with your dumb questions.", "neutral": "Please review the guidelines before asking further questions."},

    # Interpersonal frustration and hostility
    {"toxic": "I hate you and your stupid behavior.", "neutral": "I am very frustrated by the way you have been acting."},
    {"toxic": "I can't stand dealing with your arrogant attitude.", "neutral": "I find your communication style difficult to work with."},
    {"toxic": "You are a pathological liar and nobody trusts you.", "neutral": "I have serious doubts about the veracity of what you are saying."},
    {"toxic": "You are acting like a total jerk today.", "neutral": "Your recent behavior has been quite uncooperative."},
    {"toxic": "Stop acting like a know-it-all prick.", "neutral": "Please be more receptive to input and suggestions from others."},
    {"toxic": "Why are you always so obnoxious in meetings?", "neutral": "I would appreciate a more collaborative and respectful tone during our meetings."},

    # Disagreements and debates
    {"toxic": "Your argument makes zero sense, you dumbass.", "neutral": "Your argument contains logical flaws that I disagree with."},
    {"toxic": "You don't know shit about what you're talking about.", "neutral": "I do not believe you have sufficient evidence to support that claim."},
    {"toxic": "That is the dumbest take I have heard in my life.", "neutral": "I strongly disagree with that perspective."},
    {"toxic": "Don't be an idiot, that's not how this works.", "neutral": "That interpretation does not accurately reflect how the process operates."},
    {"toxic": "You are completely blind if you believe that lie.", "neutral": "I think you may have been misinformed on this matter."},

    # Client / Service / Feedback interactions
    {"toxic": "Your customer support is absolute trash.", "neutral": "The level of customer support provided was very disappointing."},
    {"toxic": "Your app is a broken piece of garbage.", "neutral": "The application has several critical bugs that need to be resolved."},
    {"toxic": "I regret paying for this terrible product, refund me now.", "neutral": "I am dissatisfied with this product and would like to request a refund."},
    {"toxic": "Your service sucks and your staff are rude.", "neutral": "I was unhappy with the quality of service and the conduct of the staff."},
    {"toxic": "What kind of moron designed this user interface?", "neutral": "This user interface is confusing and could benefit from redesign."},
]


def build_curated_dataset(max_paradetox_samples: int = 2000, exemplar_repetition: int = 12) -> pd.DataFrame:
    """Combine filtered ParaDetox pairs with constructive exemplars."""
    exemplar_records = []
    toxic_exemplar_set = set()

    for item in CONSTRUCTIVE_EXEMPLARS:
        t = item["toxic"].strip()
        n = format_sentence(item["neutral"])
        exemplar_records.append({"toxic": t, "neutral": n})
        toxic_exemplar_set.add(t.lower())

    # Extract best clean neutral from ParaDetox (avoiding sentences already covered in exemplars)
    para_records = []
    if PARADETOX_CSV.exists():
        df_para = pd.read_csv(PARADETOX_CSV, sep="\t")
        added_para = 0

        for _, row in df_para.iterrows():
            if added_para >= max_paradetox_samples:
                break

            tox = str(row["toxic"]).strip()
            if tox.lower() in toxic_exemplar_set:
                continue

            candidates = [
                str(row.get("neutral3", "")).strip(),
                str(row.get("neutral2", "")).strip(),
                str(row.get("neutral1", "")).strip(),
            ]

            for cand in candidates:
                if is_clean_neutral(cand, tox):
                    clean_target = format_sentence(cand)
                    para_records.append({"toxic": tox, "neutral": clean_target})
                    added_para += 1
                    break

    df_para_clean = pd.DataFrame(para_records).drop_duplicates(subset=["toxic"])

    # Replicate high-priority constructive exemplars to ensure the model prioritizes constructive sentence structuring
    weighted_exemplars = []
    for _ in range(exemplar_repetition):
        weighted_exemplars.extend(exemplar_records)

    df_exemplars = pd.DataFrame(weighted_exemplars)
    df_combined = pd.concat([df_para_clean, df_exemplars], ignore_index=True)
    return df_combined.sample(frac=1.0, random_state=42).reset_index(drop=True)


def main():
    print("Building curated constructive neutralization dataset...")
    df = build_curated_dataset(max_paradetox_samples=2500)
    print(f"Total curated pairs: {len(df)}")

    # Add instruction prompt formatting
    df["input_text"] = df["toxic"].apply(
        lambda t: f"{INSTRUCTION}\n\nMessage: \"{t}\"\nRewritten:"
    )
    df["target_text"] = df["neutral"]

    train_df, val_df = train_test_split(df, test_size=0.10, random_state=42)
    train_df.to_csv(OUTPUT_TRAIN_CSV, index=False)
    val_df.to_csv(OUTPUT_VAL_CSV, index=False)

    print(f"Saved {len(train_df)} training samples to: {OUTPUT_TRAIN_CSV}")
    print(f"Saved {len(val_df)} validation samples to: {OUTPUT_VAL_CSV}")


if __name__ == "__main__":
    main()

