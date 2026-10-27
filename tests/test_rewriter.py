"""Unit tests for rewriting and postprocessor modules."""

import pytest
from src.rewriting.postprocessor import (
    compute_string_similarity,
    detect_repetition,
    detect_threat_or_extreme_violence,
    analyze_rewrite_quality,
)
from src.rewriting.neutralizer import TextNeutralizer


def test_compute_string_similarity():
    sim = compute_string_similarity("Hello world", "Hello world")
    assert sim == 1.0
    sim_diff = compute_string_similarity("Apple", "Orange")
    assert sim_diff < 0.4


def test_detect_repetition():
    assert detect_repetition("the the the the something") is True
    assert detect_repetition("good bad good bad good bad good bad") is True
    assert detect_repetition("This is a completely normal English sentence.") is False


def test_detect_threat():
    assert detect_threat_or_extreme_violence("I will kill you if you come back.") is True
    assert detect_threat_or_extreme_violence("I hope you die soon.") is True
    assert detect_threat_or_extreme_violence("I disagree with your argument.") is False


def test_analyze_rewrite_quality():
    # Empty generation test
    q_empty = analyze_rewrite_quality("bad text", "", original_toxicity_prob=0.8)
    assert q_empty["status"] == "REWRITE_UNAVAILABLE"
    assert q_empty["status_label"] == "Rewrite unavailable"
    assert not q_empty["is_acceptable"]
    assert q_empty["display_text"] == "Could not generate a reliable constructive rewrite. Please rephrase the input or try again."

    # Repetitive degeneration test
    q_rep = analyze_rewrite_quality(
        "bad text",
        "sorry sorry sorry sorry sorry",
        original_toxicity_prob=0.8
    )
    assert q_rep["status"] == "REWRITE_UNAVAILABLE"
    assert not q_rep["is_acceptable"]

    # Unchanged abusive input test - MUST NEVER be marked successful
    q_unchanged = analyze_rewrite_quality(
        "You are a complete idiot and nobody cares what you think.",
        "You are a complete idiot and nobody cares what you think.",
        original_toxicity_prob=0.95,
        post_toxicity_prob=0.10,
        threshold=0.40
    )
    assert q_unchanged["status"] == "REWRITE_UNAVAILABLE"
    assert not q_unchanged["is_acceptable"]
    assert q_unchanged["display_text"] == "Could not generate a reliable constructive rewrite. Please rephrase the input or try again."

    # Successful rewrite of feedback test
    q_feedback = analyze_rewrite_quality(
        "This code is trash, fix it.",
        "This code needs improvement. Please revise it.",
        original_toxicity_prob=0.85,
        post_toxicity_prob=0.08,
        threshold=0.40
    )
    assert q_feedback["status"] == "SUGGESTED_REWRITE"
    assert q_feedback["status_label"] == "Suggested rewrite"
    assert q_feedback["is_acceptable"]

    # Successful reframing of pure personal attack test
    q_attack = analyze_rewrite_quality(
        "You are a complete idiot and nobody cares what you think.",
        "I disagree with your opinion. Let's discuss the issue respectfully.",
        original_toxicity_prob=0.95,
        post_toxicity_prob=0.05,
        threshold=0.40
    )
    assert q_attack["status"] == "SUGGESTED_ALTERNATIVE"
    assert q_attack["status_label"] == "Suggested constructive alternative"
    assert q_attack["is_acceptable"]

