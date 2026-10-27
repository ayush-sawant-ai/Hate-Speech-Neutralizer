"""Unit tests for preprocessing and text cleaning functions."""

import pytest
from src.preprocessing.cleaner import (
    clean_for_classifier,
    clean_for_rewriter,
    normalize_unicode,
    validate_input_text,
)


def test_normalize_unicode():
    raw = "Caf\u00e9"
    norm = normalize_unicode(raw)
    assert norm == "Café"
    assert normalize_unicode(None) == ""


def test_clean_for_classifier_removes_urls_and_mentions():
    text = "Check out https://evil.com and @john_doe you idiot!"
    cleaned = clean_for_classifier(text)
    assert "https" not in cleaned
    assert "evil" not in cleaned
    assert "john" not in cleaned
    assert "you idiot" in cleaned


def test_clean_for_classifier_handles_empty_and_special_chars():
    assert clean_for_classifier("") == ""
    assert clean_for_classifier("   ") == ""
    assert clean_for_classifier("!@#$%^&*()") == ""
    assert clean_for_classifier(None) == ""


def test_clean_for_rewriter_preserves_casing_and_punctuation():
    text = "This is a TEST! Do not break my punctuation, okay?"
    cleaned = clean_for_rewriter(text)
    assert "TEST!" in cleaned
    assert "okay?" in cleaned
    assert "," in cleaned


def test_validate_input_text():
    is_valid, msg = validate_input_text("")
    assert not is_valid
    assert "empty" in msg.lower()

    is_valid, msg = validate_input_text("Hello world")
    assert is_valid
    assert msg == ""

    is_valid, msg = validate_input_text("!@#$$%^^")
    assert not is_valid

    is_valid, msg = validate_input_text("a" * 1500, max_chars=1000)
    assert not is_valid
    assert "exceeds" in msg.lower()

