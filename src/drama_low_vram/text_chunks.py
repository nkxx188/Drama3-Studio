"""Lossless request splitting at sentence, clause and word boundaries.

``max_chars`` is a soft limit for characters outside control tags, not an audio
duration estimate. Tags and an indivisible word may make a chunk longer. The
caller should retain official prompt history for voice continuity.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


SPEAKER_TAG = re.compile(r"<\|speaker:([0-9]+)\|>")
_ABBREVIATIONS = {
    "mr.",
    "mrs.",
    "ms.",
    "dr.",
    "prof.",
    "sr.",
    "jr.",
    "st.",
    "vs.",
    "etc.",
    "e.g.",
    "i.e.",
    "a.m.",
    "p.m.",
    "no.",
    "fig.",
}
_TERMINATORS = ".!?\u3002\uff01\uff1f\u2026"
_CLOSERS = "\"')]}>\u201d\u2019\u300d\u300f\u3011\uff09"
_CLAUSES = ",;:\uff0c\uff1b\uff1a\u3001"


@dataclass(frozen=True)
class _Unit:
    start: int
    end: int
    value: str
    protected: bool


def _units(text: str) -> list[_Unit]:
    result = []
    offset = 0
    while offset < len(text):
        start = offset
        opening = text[offset]
        if opening in "[<":
            closing = "]" if opening == "[" else ">"
            depth = 1
            offset += 1
            while offset < len(text) and depth:
                character = text[offset]
                if opening == "[" and character == opening:
                    depth += 1
                elif character == closing:
                    depth -= 1
                offset += 1
            result.append(_Unit(start, offset, text[start:offset], True))
        else:
            offset += 1
            result.append(_Unit(start, offset, text[start:offset], False))
    return result


def _period_ends_sentence(text: str, offset: int) -> bool:
    previous = text[offset - 1] if offset else ""
    following = text[offset + 1] if offset + 1 < len(text) else ""
    if previous.isdigit() and following.isdigit():
        return False
    if following == ".":
        return False
    if previous.isalnum() and following.isalnum():
        return False
    start = offset
    while start and (text[start - 1].isalpha() or text[start - 1] == "."):
        start -= 1
    word = text[start : offset + 1]
    if word.lower() in _ABBREVIATIONS:
        return False
    if re.fullmatch(r"(?:[A-Za-z]\.){2,}", word):
        return False
    if re.fullmatch(r"[A-Z]\.", word):
        return False
    return True


def _is_word_character(character: str) -> bool:
    # Do not split a Latin word merely because its length crosses the soft limit.
    # CJK characters remain legal fallback boundaries when there is no punctuation.
    return (
        character.isascii() and (character.isalnum() or character in "'_-")
    ) or (
        character.isalpha()
        and ("LATIN" in unicodedata.name(character, ""))
    )


def _inside_word(units: list[_Unit], boundary: int) -> bool:
    left, right = units[boundary - 1], units[boundary]
    if left.protected or right.protected:
        return False
    if _is_word_character(left.value) and _is_word_character(right.value):
        return True
    # Decimal dots and acronym dots belong to their adjoining word.
    if right.value == "." and boundary + 1 < len(units):
        next_unit = units[boundary + 1]
        return (
            not next_unit.protected
            and _is_word_character(left.value)
            and _is_word_character(next_unit.value)
        )
    if left.value == "." and boundary > 1:
        previous = units[boundary - 2]
        return (
            not previous.protected
            and _is_word_character(previous.value)
            and _is_word_character(right.value)
        )
    return False


def _boundaries(text: str, units: list[_Unit]) -> tuple[set[int], set[int], set[int]]:
    sentences, clauses, spaces = set(), set(), set()
    for index, unit in enumerate(units):
        if unit.protected:
            continue
        character = unit.value
        if character.isspace():
            spaces.add(index + 1)
        if character in _CLAUSES:
            clauses.add(index + 1)
        if character not in _TERMINATORS:
            continue
        if character == "." and not _period_ends_sentence(text, unit.start):
            continue
        end = index + 1
        while end < len(units) and not units[end].protected:
            if units[end].value in _TERMINATORS + _CLOSERS or units[end].value.isspace():
                end += 1
            else:
                break
        sentences.add(end)
    return sentences, clauses, spaces


def _speaker_before(units: list[_Unit], boundary: int) -> str:
    speaker = "<|speaker:0|>"
    for unit in units[:boundary]:
        match = SPEAKER_TAG.fullmatch(unit.value) if unit.protected else None
        if match:
            speaker = unit.value
    return speaker


def _has_leading_speaker(units: list[_Unit], start: int, end: int) -> bool:
    for unit in units[start:end]:
        if unit.value.isspace():
            continue
        if SPEAKER_TAG.fullmatch(unit.value):
            return True
        return False
    return False


def split_text(text: str, max_chars: int = 200) -> list[str]:
    """Preserve text and tags; inherit the active speaker in every later request.

    No whitespace or punctuation is stripped. Joining the chunks after removing
    inserted leading speaker tags recovers the original text exactly. Text
    validity (including allowed speaker IDs and structural tokens) remains the
    official prompt builder's responsibility.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if isinstance(max_chars, bool) or not isinstance(max_chars, int) or max_chars < 1:
        raise ValueError("max_chars must be a positive integer")
    if not text:
        return []
    units = _units(text)
    sentences, clauses, spaces = _boundaries(text, units)
    cumulative = [0]
    for unit in units:
        cumulative.append(cumulative[-1] + (0 if unit.protected else len(unit.value)))
    chunks = []
    start = 0
    while start < len(units):
        if cumulative[-1] - cumulative[start] <= max_chars:
            end = len(units)
        else:
            limit = start
            while (
                limit < len(units)
                and cumulative[limit + 1] - cumulative[start] <= max_chars
            ):
                limit += 1
            # Control tags have zero counted width; a fallback still makes progress.
            limit = max(start + 1, limit)
            end = None
            for preferred in (sentences, clauses, spaces):
                options = [
                    boundary
                    for boundary in preferred
                    if start < boundary <= limit
                    and cumulative[boundary] - cumulative[start] >= max_chars // 3
                ]
                if options:
                    end = max(options)
                    break
            if end is None:
                end = limit
                while (
                    end < len(units)
                    and _inside_word(units, end)
                ):
                    end += 1
            # Attach trailing whitespace to its text, avoiding a whitespace-only tail.
            while end < len(units) and units[end].value.isspace():
                end += 1
            # A direction or speaker at the limit belongs to the following words.
            # Zero-width tags must not drift onto the previous generation request.
            if end < len(units):
                probe, tag_start = end, None
                while probe > start:
                    unit = units[probe - 1]
                    if unit.value.isspace():
                        probe -= 1
                    elif unit.protected and not unit.value.startswith("</"):
                        probe -= 1
                        tag_start = probe
                    else:
                        break
                if (
                    tag_start is not None
                    and cumulative[tag_start] > cumulative[start]
                ):
                    end = tag_start
        chunk = text[units[start].start : units[end - 1].end]
        if chunks and not _has_leading_speaker(units, start, end):
            chunk = _speaker_before(units, start) + chunk
        chunks.append(chunk)
        start = end
    return chunks
