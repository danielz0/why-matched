from whymatched.text import remove_span, replace_span, split_words


def test_split_words_basic():
    words = split_words("Remote work is not allowed.")
    assert [w.text for w in words] == ["Remote", "work", "is", "not", "allowed", "."]


def test_remove_span_collapses_whitespace():
    words = split_words("Remote work is not allowed.")
    not_span = next(w for w in words if w.text == "not")
    result = remove_span("Remote work is not allowed.", not_span)
    assert result == "Remote work is allowed."


def test_replace_span():
    words = split_words("The value is high.")
    high_span = next(w for w in words if w.text == "high")
    result = replace_span("The value is high.", high_span, "low")
    assert result == "The value is low."
