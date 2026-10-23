"""Speech requests must keep every word, direction and active speaker."""

import re

import pytest

from drama_low_vram.text_chunks import split_text


def recover(chunks: list[str], original: str) -> str:
    """Strip only automatically inserted prefixes, never original speaker markers."""
    offset = 0
    parts = []
    for chunk in chunks:
        if not original.startswith(chunk, offset):
            chunk = re.sub(r"^<\|speaker:[0-9]+\|>", "", chunk, count=1)
        assert original.startswith(chunk, offset)
        parts.append(chunk)
        offset += len(chunk)
    return "".join(parts)


def test_short_text_is_unchanged():
    text = "  [calm]<|speaker:3|> Hello, world! \n"
    assert split_text(text, 200) == [text]


def test_chinese_sentences_pack_without_losing_punctuation_or_spaces():
    text = "\u4f60\u597d\u3002 \u4eca\u5929\u5929\u6c14\u5f88\u597d\uff01\n\u6211\u4eec\u51fa\u53bb\u8d70\u8d70\u5427\uff1f"
    chunks = split_text(text, 10)
    assert len(chunks) > 1
    assert recover(chunks, text) == text
    assert chunks[1].startswith("<|speaker:0|>")


def test_speaker_changes_propagate_to_later_chunks():
    text = (
        "<|speaker:3|>One sentence. A second sentence. "
        "<|speaker:12|>[excited]A third sentence! A fourth sentence?"
    )
    chunks = split_text(text, 20)
    assert recover(chunks, text) == text
    assert chunks[1].startswith("<|speaker:3|>")
    assert chunks[-1].startswith("<|speaker:12|>")
    assert text.count("<|speaker:12|>") == 1


def test_directions_and_angle_tags_are_indivisible():
    text = "[very slowly, with a warm smile]This is one sentence.<voice style='soft'> Next one."
    chunks = split_text(text, 13)
    assert recover(chunks, text) == text
    assert any("[very slowly, with a warm smile]" in chunk for chunk in chunks)
    assert any("<voice style='soft'>" in chunk for chunk in chunks)
    assert all(chunk.count("[") == chunk.count("]") for chunk in chunks)
    assert all(chunk.count("<") == chunk.count(">") for chunk in chunks)


def test_nested_brackets_do_not_split():
    text = "[direction [with nested detail]]Hello there. Another sentence."
    chunks = split_text(text, 14)
    assert recover(chunks, text) == text
    assert "[direction [with nested detail]]" in chunks[0]


@pytest.mark.parametrize("prefix", ["Mr. Smith", "Dr. Jones", "Mrs. Brown", "U.S. news", "A. Smith"])
def test_common_abbreviations_do_not_become_sentence_boundaries(prefix):
    text = prefix + " is here today. The next sentence is here."
    chunks = split_text(text, len(prefix) + 8)
    assert prefix in chunks[0]
    assert recover(chunks, text) == text


def test_decimal_is_preserved():
    text = "The result is 3.14 meters, and this is correct. Another sentence."
    chunks = split_text(text, 18)
    assert any("3.14" in chunk for chunk in chunks)
    assert recover(chunks, text) == text


def test_single_decimal_is_indivisible_even_above_soft_limit():
    text = "123.456789 then another part."
    chunks = split_text(text, 3)
    assert chunks[0] == "123.456789 "
    assert recover(chunks, text) == text


def test_single_long_chinese_sentence_has_bounded_lossless_fallback():
    text = "\u6d4b\u8bd5\u8bed\u97f3\u751f\u6210" * 20
    chunks = split_text(text, 17)
    assert recover(chunks, text) == text
    assert len(chunks) == 8
    assert len(chunks[0]) == 17
    assert all(len(re.sub(r"^<\|speaker:0\|>", "", chunk)) <= 17 for chunk in chunks)


def test_long_sentence_prefers_comma_then_word_space():
    text = "One two three, four five six seven eight nine ten eleven."
    chunks = split_text(text, 20)
    assert chunks[0] == "One two three, "
    assert recover(chunks, text) == text
    assert all(not chunk.endswith("sev") for chunk in chunks)


def test_single_long_word_is_kept_whole_past_soft_limit():
    text = "pneumonoultramicroscopicsilicovolcanoconiosis another word."
    chunks = split_text(text, 10)
    assert chunks[0].strip() == "pneumonoultramicroscopicsilicovolcanoconiosis"
    assert recover(chunks, text) == text


def test_explicit_leading_speaker_does_not_get_an_extra_prefix():
    text = "<|speaker:1|>First sentence. <|speaker:9|>Second sentence."
    chunks = split_text(text, 16)
    assert recover(chunks, text) == text
    assert sum(chunk.count("<|speaker:9|>") for chunk in chunks) == 1


@pytest.mark.parametrize("text", [" \n", "[calm]", "[unfinished direction", "<unfinished tag"])
def test_unusual_input_is_preserved_for_official_validation(text):
    assert "".join(split_text(text, 4)) == text


@pytest.mark.parametrize("max_chars", [0, -1, 2.5, True])
def test_invalid_limit_is_rejected(max_chars):
    with pytest.raises(ValueError):
        split_text("hello", max_chars)


def test_empty_text_returns_no_requests():
    assert split_text("") == []


def test_closing_quotes_and_ellipsis_stay_with_the_sentence():
    text = 'She said, "Hello!"  Next sentence...  The final sentence.'
    chunks = split_text(text, 25)
    assert chunks[0] == 'She said, "Hello!"  '
    assert recover(chunks, text) == text


def test_tiny_opening_sentence_does_not_force_a_tiny_first_audio_chunk():
    text = "Hi. " + "one two three four five six seven eight nine ten eleven twelve."
    chunks = split_text(text, 30)
    assert len(chunks[0]) >= 20
    assert recover(chunks, text) == text


def test_direction_at_chunk_limit_stays_with_the_following_words():
    text = "[calm]hello[angry]world"
    chunks = split_text(text, 5)
    assert chunks == ["[calm]hello", "<|speaker:0|>[angry]world"]
    assert recover(chunks, text) == text


def test_speaker_at_chunk_limit_starts_the_following_request():
    text = "<|speaker:3|>hello<|speaker:8|>world"
    chunks = split_text(text, 5)
    assert chunks == ["<|speaker:3|>hello", "<|speaker:8|>world"]
    assert recover(chunks, text) == text
