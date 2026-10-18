from ticket_classifier.preprocessing import clean_text


def test_lowercases_and_collapses_whitespace():
    assert clean_text("  Hello   WORLD \n\t again ") == "hello world again"


def test_replaces_urls_and_emails_with_tokens():
    cleaned = clean_text("See https://example.com/help or email jane.doe@acme.io")
    assert "urltoken" in cleaned
    assert "emailtoken" in cleaned
    assert "example.com" not in cleaned
    assert "@" not in cleaned


def test_replaces_long_numbers_and_keeps_short_ones():
    cleaned = clean_text("Order #48213 has 2 items")
    assert "numbertoken" in cleaned
    assert "48213" not in cleaned
    assert "2 items" in cleaned


def test_handles_none_and_non_string_input():
    assert clean_text(None) == ""
    assert clean_text(42) == "42"
