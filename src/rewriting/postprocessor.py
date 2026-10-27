"""Post-processing and quality check utilities for neutralized text.
Guarantees that outputs are syntactically complete, preserve useful requests,
and avoids marking broken/incomplete sentences as successful.
"""

from difflib import SequenceMatcher
from typing import Any, Dict, Optional, Tuple
import re


def compute_string_similarity(a: str, b: str) -> float:
    """Compute character-level similarity ratio between two strings [0.0, 1.0]."""
    return SequenceMatcher(None, a.strip().lower(), b.strip().lower()).ratio()


def detect_repetition(text: str, max_repeat_count: int = 3) -> bool:
    """Detect repetitive degenerate generation (e.g. repeated words or n-grams)."""
    words = text.strip().lower().split()
    if len(words) < 4:
        return False

    # Check 1-gram repetition (e.g., "the the the")
    for i in range(len(words) - max_repeat_count + 1):
        if len(set(words[i:i + max_repeat_count])) == 1:
            return True

    # Check 2-gram repetition (e.g. "good bad good bad good bad")
    bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words) - 1)]
    for i in range(len(bigrams) - 3):
        if i + 4 <= len(bigrams) and bigrams[i] == bigrams[i+2] == bigrams[i+4]:
            return True

    # Vocabulary diversity ratio check for degenerate babbling
    if len(words) >= 6 and (len(set(words)) / len(words)) < 0.45:
        return True

    return False


def detect_broken_or_incomplete_syntax(text: str) -> Tuple[bool, str]:
    """Detect broken syntax remnants, dangling determiners, or incomplete fragments.
    
    Catches common word-deletion artifacts such as:
    - 'you are such a improve your work'
    - 'this is a fix'
    - dangling prepositions or conjunctions at the end of a sentence
    - unclosed quotes or truncated fragments
    """
    s = text.strip()
    if not s:
        return True, "Generated text is empty."

    # 1. Dangling determiners / articles followed by verb, preposition, or end of sentence
    dangling_determiner = re.search(
        r"\b(?:such\s+an?|an?|the|my|your|their|our)\s+(?:improve|fixes|does|makes|goes|is|are|was|were|and|or|but|to|for|in|at|with)(?=\s|[\.\!\?]|$)",
        s,
        re.IGNORECASE,
    )
    if not dangling_determiner:
        dangling_determiner = re.search(
            r"\b(?:such\s+an?|an?|the|my|your|their|our)\s*[\.\!\?]$",
            s,
            re.IGNORECASE,
        )
    if dangling_determiner:
        return True, f"Contains broken grammar fragment ('{dangling_determiner.group(0).strip()}')"

    # 2. Dangling prepositions or conjunctions at the end of the text
    if re.search(r"\b(?:to|for|in|with|and|or|but|of|at|by|from|about)\s*[\.\!\?]?$", s, re.IGNORECASE):
        return True, "Ends with an incomplete dangling preposition or conjunction."

    # 3. Unclosed quotation marks
    if s.count('"') % 2 != 0:
        return True, "Contains unbalanced quotation marks."

    # 4. Truncated single/two words when input had substantial content
    words = s.split()
    if len(words) <= 2 and not s.endswith((".", "!", "?")):
        return True, "Appears to be an incomplete sentence fragment."

    return False, ""


def check_useful_message_loss(original_text: str, rewritten_text: str) -> Tuple[bool, str]:
    """Verify that legitimate feedback, requests, or core subject matter are not dropped."""
    orig_lower = original_text.lower()
    rewr_lower = rewritten_text.lower()

    # Common action / feedback / task keywords
    action_keywords = [
        "improve", "work", "fix", "code", "listen", "stop", "read", "check",
        "redo", "revise", "clean", "help", "attention", "focus", "idea",
        "plan", "proposal", "quality", "performance", "effort", "change"
    ]

    orig_actions = [kw for kw in action_keywords if re.search(r"\b" + kw + r"\b", orig_lower)]

    if orig_actions:
        # Check if the rewrite preserved at least one of the core feedback keywords
        has_action = any(re.search(r"\b" + kw + r"\b", rewr_lower) for kw in action_keywords)
        if not has_action:
            # Also check semantic paraphrases like "revise", "redo", "concerns", "hear me out"
            semantic_fallbacks = [
                "revise", "redo", "concerns", "attention", "standards",
                "requirements", "hear me out", "hear out", "clarify", "discuss",
                "perspective", "understanding"
            ]
            if not any(kw in rewr_lower for kw in semantic_fallbacks):
                return True, f"Lost key feedback/action context (original mentioned: {', '.join(orig_actions)})."

    return False, ""


def detect_threat_or_extreme_violence(text: str) -> bool:
    """Check if the text contains explicit threats or violent intent."""
    threat_patterns = [
        r"\bi\s*(?:will|gonna|must|am going to)\s*.*?\b(?:kill|murder|shoot|stab|destroy|hurt|beat|harm)\s+you\b",
        r"\bhope\s+you\s+(?:die|suffer|burn|suffocate)\b",
        r"\bdeath\s+to\b",
        r"\bshould\s+be\s+(?:killed|executed|exterminated)\b",
        r"\b(?:gonna|going to|will)\s+(?:kill|hurt|destroy|stab|shoot)\s+you\b",
    ]
    text_lower = text.lower()
    for pattern in threat_patterns:
        if re.search(pattern, text_lower):
            return True
    return False


def detect_remaining_attacks(text: str) -> bool:
    """Detect if the text still contains explicit insults, slurs, or harassment."""
    attack_patterns = [
        r"\b(?:idiot|moron|shit|fuck|fucking|bitch|trash|garbage|clown|pathetic|loser|dumbass|jackass|asshole|bastard|cunt|dick)\b",
        r"\byou are (?:such a|a) (?:bad person|terrible person)\b",
        r"\bshut (?:the fuck )?up\b",
        r"\bnobody cares (?:what you think|about you|about what you)\b",
        r"\bcomplete idiot\b",
        r"\byou are useless\b",
    ]
    lower = text.lower()
    for pattern in attack_patterns:
        if re.search(pattern, lower):
            return True
    return False


def is_pure_personal_attack(text: str) -> bool:
    """Determine if the input contains only personal insults/humiliation without actionable requests."""
    action_keywords = [
        "improve", "work", "fix", "code", "listen", "read", "check",
        "redo", "revise", "clean", "help", "attention", "focus",
        "proposal", "plan", "clarify", "explain", "review", "test",
        "pull request", "pr", "bug", "build", "task", "assignment", "deadline"
    ]
    lower = text.lower()
    has_action = any(re.search(r"\b" + kw + r"\b", lower) for kw in action_keywords)
    return not has_action


def analyze_rewrite_quality(
    original_text: str,
    rewritten_text: Optional[str],
    original_toxicity_prob: float,
    post_toxicity_prob: Optional[float] = None,
    threshold: float = 0.40,
) -> Dict[str, Any]:
    """Inspect and assess the quality, fluency, and safety of generated text.
    
    Guarantees that broken grammar, incomplete fragments, loss of useful
    message, or unchanged abusive inputs are rejected honestly.
    """
    orig_clean = original_text.strip()
    rewrite_clean = (rewritten_text or "").strip()

    # 1. Empty generation
    if not rewrite_clean:
        return {
            "status": "REWRITE_UNAVAILABLE",
            "status_label": "Rewrite unavailable",
            "is_acceptable": False,
            "message": "The rewriting model generated an empty response.",
            "display_text": "Could not generate a reliable constructive rewrite. Please rephrase the input or try again.",
            "suggestion": "Please consider revising or rephrasing the input text."
        }

    similarity = compute_string_similarity(orig_clean, rewrite_clean)
    is_unchanged = (similarity > 0.88)

    # 2. Unchanged abusive input (MUST NEVER be marked successful)
    if is_unchanged:
        return {
            "status": "REWRITE_UNAVAILABLE",
            "status_label": "Rewrite unavailable",
            "is_acceptable": False,
            "message": "The output matched the original abusive input without constructive transformation.",
            "display_text": "Could not generate a reliable constructive rewrite. Please rephrase the input or try again.",
            "suggestion": "Automated neutralization was unable to rewrite this input safely."
        }

    # 3. Remaining personal attacks in output
    if detect_remaining_attacks(rewrite_clean):
        return {
            "status": "REWRITE_UNAVAILABLE",
            "status_label": "Rewrite unavailable",
            "is_acceptable": False,
            "message": "The output still contained personal attacks or hostile phrasing.",
            "display_text": "Could not generate a reliable constructive rewrite. Please rephrase the input or try again.",
            "suggestion": "Automated rewrite retained offensive language. Manual rephrasing required."
        }

    # 4. Repetitive degeneration
    if detect_repetition(rewrite_clean):
        return {
            "status": "REWRITE_UNAVAILABLE",
            "status_label": "Rewrite unavailable",
            "is_acceptable": False,
            "message": "The generated output exhibits repetitive degeneration.",
            "display_text": "Could not generate a reliable constructive rewrite. Please rephrase the input or try again.",
            "suggestion": "Output was rejected due to repetition artifacts."
        }

    # 5. Broken grammar and incomplete syntax check
    is_broken, broken_reason = detect_broken_or_incomplete_syntax(rewrite_clean)
    if is_broken:
        return {
            "status": "REWRITE_UNAVAILABLE",
            "status_label": "Rewrite unavailable",
            "is_acceptable": False,
            "message": f"Ungrammatical or incomplete rewrite detected: {broken_reason}.",
            "display_text": "Could not generate a reliable constructive rewrite. Please rephrase the input or try again.",
            "suggestion": "Automated rewrite left an incomplete sentence. Please rephrase manually."
        }

    # 6. Useful message / request preservation check
    is_lost, loss_reason = check_useful_message_loss(orig_clean, rewrite_clean)
    if is_lost:
        return {
            "status": "REWRITE_UNAVAILABLE",
            "status_label": "Rewrite unavailable",
            "is_acceptable": False,
            "message": f"Rewrite dropped essential intent: {loss_reason}.",
            "display_text": "Could not generate a reliable constructive rewrite. Please rephrase the input or try again.",
            "suggestion": "The generated text omitted the original request or feedback."
        }

    # 7. Threat / violence safety check
    has_threat = detect_threat_or_extreme_violence(orig_clean)
    intent_warning = ""
    if has_threat:
        intent_warning = (
            "Note: The original input contained threats or severe hostility. "
            "Safe alternatives cannot preserve harmful intent."
        )
        if detect_threat_or_extreme_violence(rewrite_clean):
            return {
                "status": "REWRITE_UNAVAILABLE",
                "status_label": "Rewrite unavailable",
                "is_acceptable": False,
                "message": "Explicit violent threats cannot be safely paraphrased without removing the threat.",
                "display_text": "Could not generate a reliable constructive rewrite. Please rephrase the input or try again.",
                "intent_warning": intent_warning,
                "suggestion": "Violent threats violate safety guidelines and cannot be neutralized."
            }

    # 8. Re-scored toxicity assessment
    toxicity_reduced = False
    if post_toxicity_prob is not None:
        delta = original_toxicity_prob - post_toxicity_prob
        toxicity_reduced = (delta > 0.02)
        post_is_safe = (post_toxicity_prob < threshold)
    else:
        post_is_safe = True

    if not post_is_safe:
        return {
            "status": "REWRITE_UNAVAILABLE",
            "status_label": "Rewrite unavailable",
            "is_acceptable": False,
            "message": "Rewritten text retains elevated toxicity scores above the threshold.",
            "display_text": "Could not generate a reliable constructive rewrite. Please rephrase the input or try again.",
            "intent_warning": intent_warning,
            "suggestion": "Review the neutralized text carefully or rephrase manually."
        }

    # 9. Determine honest label based on whether input was pure personal attack
    pure_attack = is_pure_personal_attack(orig_clean)
    if pure_attack:
        final_status = "SUGGESTED_ALTERNATIVE"
        status_label = "Suggested constructive alternative"
        message = "Personal attack reframed into a respectful statement of disagreement or dissatisfaction."
    else:
        final_status = "SUGGESTED_REWRITE"
        status_label = "Suggested rewrite"
        message = "Feedback preserved and rewritten in respectful, constructive English."

    return {
        "status": final_status,
        "status_label": status_label,
        "is_acceptable": True,
        "is_unchanged": False,
        "similarity": similarity,
        "toxicity_reduced": toxicity_reduced,
        "display_text": rewrite_clean,
        "message": message,
        "intent_warning": intent_warning,
        "suggestion": ""
    }
