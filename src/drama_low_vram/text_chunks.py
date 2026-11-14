"""Request splitting at natural boundaries without losing spoken text.

``max_chars`` is a soft target for characters outside control tags, not an audio
duration estimate. A sentence may extend past the target so the model can finish
it; the splitter falls back to clauses, words, and finally characters when a
sentence is longer than the soft overflow window. Paired directions stay whole
within that window; longer scopes are closed and reopened at chunk boundaries.
"""

from __future__ import annotations

import re
import unicodedata
from math import ceil
from dataclasses import dataclass


SPEAKER_TAG = re.compile(r"<\|speaker:([0-9]+)\|>")
PHONEME_START = "<|phoneme_start|>"
PHONEME_END = "<|phoneme_end|>"
_SCOPE_TAG = re.compile(r"<(\/?)(whisper|emphasis|soft|fast|slow|stress|forceful)>")
_ANGLE_TAG = re.compile(r"<[^<>\r\n]*>")
_PHONEME_PAIR = re.compile(r"<\|phoneme_start\|>([^<>]*?)<\|phoneme_end\|>")
_PINYIN_SYLLABLE = re.compile(r"[a-z\u00fcv\u00ea]+[1-5]", re.IGNORECASE)
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
_TERMINATORS = ".!?\u3002\uff01\uff1f\u2026;\n"
_CLOSERS = "\"')]}>\u201d\u2019\u300d\u300f\u3011\uff09"
_CLAUSES = ",;:\uff0c\uff1b\uff1a\u3001"


@dataclass(frozen=True)
class _Unit:
    start: int
    end: int
    value: str
    protected: bool
    width: int


def _phoneme_width(value: str) -> int:
    syllables = value.split()
    if syllables and all(_PINYIN_SYLLABLE.fullmatch(part) for part in syllables):
        return len(syllables)
    return len(value)


def _scope_pairs(units: list[_Unit]) -> dict[int, int]:
    """Match nested scopes, including repeated names, without repairing bad input."""
    pairs, stack = {}, []
    for index, unit in enumerate(units):
        match = _SCOPE_TAG.fullmatch(unit.value) if unit.protected else None
        if match is None:
            continue
        closing, name = match.groups()
        if not closing:
            stack.append((name, index))
        elif stack and stack[-1][0] == name:
            _, start = stack.pop()
            pairs[start] = index + 1
        else:
            stack.clear()
    return pairs


def _units(text: str, *, scope_budget: int | None = None) -> list[_Unit]:
    bracket_ends, stack = {}, []
    for index, character in enumerate(text):
        if character == "[":
            stack.append(index)
        elif character == "]" and stack:
            bracket_ends[stack.pop()] = index + 1
    result = []
    offset = 0
    while offset < len(text):
        start = offset
        phoneme = _PHONEME_PAIR.match(text, offset) if text[offset] == "<" else None
        angle = _ANGLE_TAG.match(text, offset) if text[offset] == "<" else None
        if phoneme is not None:
            offset = phoneme.end()
            result.append(_Unit(start, offset, text[start:offset], True,
                                _phoneme_width(phoneme[1])))
        elif offset in bracket_ends:
            offset = bracket_ends[offset]
            result.append(_Unit(start, offset, text[start:offset], True, 0))
        elif angle is not None:
            offset = angle.end()
            result.append(_Unit(start, offset, text[start:offset], True, 0))
        else:
            offset += 1
            result.append(_Unit(start, offset, text[start:offset], False, 1))
    for index, unit in enumerate(result[:-1]):
        if SPEAKER_TAG.fullmatch(unit.value) and result[index + 1].value == " ":
            following = result[index + 1]
            result[index + 1] = _Unit(
                following.start, following.end, following.value, following.protected, 0)
    if scope_budget is None:
        return result
    pairs, cumulative = _scope_pairs(result), [0]
    for unit in result:
        cumulative.append(cumulative[-1] + unit.width)
    grouped, index = [], 0
    while index < len(result):
        end = pairs.get(index)
        if end is not None and cumulative[end] - cumulative[index] <= scope_budget:
            start, stop = result[index].start, result[end - 1].end
            grouped.append(_Unit(start, stop, text[start:stop], True,
                                 cumulative[end] - cumulative[index]))
            index = end
        else:
            grouped.append(result[index])
            index += 1
    return grouped


def _balanced_slices(text: str, units: list[_Unit], ranges: list[tuple[int, int]]) -> list[str]:
    pairs = _scope_pairs(units)
    closings = {end - 1 for end in pairs.values()}
    active, chunks = [], []
    for start, end in ranges:
        prefix = "".join(f"<{name}>" for name in active)
        for index in range(start, end):
            if index in pairs:
                active.append(units[index].value[1:-1])
            elif index in closings:
                active.pop()
        suffix = "".join(f"</{name}>" for name in reversed(active))
        chunks.append(prefix + text[units[start].start:units[end - 1].end] + suffix)
    return chunks


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


def _unit_width(unit: _Unit) -> int:
    return unit.width


def _content_width(text: str) -> int:
    return sum(_unit_width(unit) for unit in _units(text))


def _choose_boundary(
    preferred: set[int],
    *,
    start: int,
    target_end: int,
    soft_end: int,
    cumulative: list[int],
    minimum_width: int,
) -> int | None:
    """Choose a natural boundary, allowing a sentence to finish after target."""
    within_target = [
        boundary for boundary in preferred
        if start < boundary <= target_end
        and cumulative[boundary] - cumulative[start] >= minimum_width
    ]
    if within_target:
        return max(within_target)
    after_target = [
        boundary for boundary in preferred
        if target_end < boundary <= soft_end
    ]
    return min(after_target) if after_target else None


def _split_plain_text(text: str, max_chars: int) -> list[str]:
    """Split text without adding speaker prefixes.

    No whitespace or punctuation is stripped. Long scopes receive balanced
    closing/reopening tags on each side of a split. Text validity (including
    speaker IDs and structural tokens) remains the prompt builder's responsibility.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if isinstance(max_chars, bool) or not isinstance(max_chars, int) or max_chars < 1:
        raise ValueError("max_chars must be a positive integer")
    if not text:
        return []
    soft_budget = max(max_chars + 1, ceil(max_chars * 1.5))
    units = _units(text, scope_budget=soft_budget)
    sentences, clauses, spaces = _boundaries(text, units)
    cumulative = [0]
    for unit in units:
        cumulative.append(cumulative[-1] + _unit_width(unit))
    ranges = []
    start = 0
    while start < len(units):
        if cumulative[-1] - cumulative[start] <= max_chars:
            end = len(units)
        else:
            target_end = start
            while (
                target_end < len(units)
                and cumulative[target_end + 1] - cumulative[start] <= max_chars
            ):
                target_end += 1
            # A leading cue must travel with an oversized protected speech unit.
            has_speech = any(
                unit.width and (unit.protected or not unit.value.isspace())
                for unit in units[start:target_end]
            )
            while target_end < len(units) and not has_speech:
                unit = units[target_end]
                has_speech = unit.width and (unit.protected or not unit.value.isspace())
                target_end += 1
            target_end = max(start + 1, target_end)
            soft_end = target_end
            while (
                soft_end < len(units)
                and cumulative[soft_end + 1] - cumulative[start] <= soft_budget
            ):
                soft_end += 1
            soft_end = max(target_end, soft_end)
            end = None
            minimum_width = max(1, max_chars // 3)
            for preferred in (sentences, clauses, spaces):
                end = _choose_boundary(
                    preferred, start=start, target_end=target_end,
                    soft_end=soft_end, cumulative=cumulative,
                    minimum_width=minimum_width)
                if end is not None:
                    break
            if end is None:
                end = target_end
                while (
                    end < len(units)
                    and _inside_word(units, end)
                ):
                    end += 1
            # Keep closing tags and trailing whitespace with their preceding text.
            while end < len(units) and (
                units[end].value.isspace()
                or units[end].width == 0 and units[end].value.startswith("</")
            ):
                end += 1
            # A direction or speaker at the limit belongs to the following words.
            # Zero-width tags must not drift onto the previous generation request.
            if end < len(units):
                probe, tag_start = end, None
                while probe > start:
                    unit = units[probe - 1]
                    if unit.value.isspace():
                        probe -= 1
                    elif unit.protected and unit.width == 0 and not unit.value.startswith("</"):
                        probe -= 1
                        tag_start = probe
                    else:
                        break
                if (
                    tag_start is not None
                    and cumulative[tag_start] > cumulative[start]
                ):
                    end = tag_start
        ranges.append((start, end))
        start = end
    return _balanced_slices(text, units, ranges)


def _speaker_segments(text: str) -> list[tuple[str, str]]:
    """Return balanced speaker turns, keeping untagged speech as speaker 0."""
    units = _units(text)
    matches = [(index, match[1]) for index, unit in enumerate(units)
               if (match := SPEAKER_TAG.fullmatch(unit.value)) is not None]
    if not matches:
        return []
    speakers, ranges = [], []
    prefix_end = matches[0][0]
    has_prefix_text = any(unit.width and not unit.value.isspace() for unit in units[:prefix_end])
    if has_prefix_text:
        speakers.append("0")
        ranges.append((0, prefix_end))
    for index, (start, speaker) in enumerate(matches):
        end = matches[index + 1][0] if index + 1 < len(matches) else len(units)
        if index == 0 and not has_prefix_text:
            start = 0
        speakers.append(speaker)
        ranges.append((start, end))
    return list(zip(speakers, _balanced_slices(text, units, ranges)))


def _split_speaker_text(text: str, max_chars: int, max_speakers: int = 3) -> list[str]:
    """Split tagged dialogue while keeping each chunk to at most three speakers."""
    if (_content_width(text) <= max_chars
            and len(set(SPEAKER_TAG.findall(text))) <= max_speakers):
        return [text]
    pieces = []
    for speaker, body in _speaker_segments(text):
        for piece in _split_plain_text(body, max_chars):
            if piece:
                tagged = piece if SPEAKER_TAG.search(piece) else f"<|speaker:{speaker}|>{piece}"
                pieces.append((speaker, tagged))
    if not pieces:
        return _split_plain_text(text, max_chars)
    chunks, batch, speakers, width = [], [], set(), 0
    for speaker, piece in pieces:
        piece_width = _content_width(piece)
        new_speaker = speaker not in speakers
        too_many_speakers = new_speaker and len(speakers) >= max_speakers
        too_long = bool(batch) and width + piece_width > max_chars
        if too_many_speakers or too_long:
            chunks.append("".join(batch))
            batch, speakers, width = [], set(), 0
            new_speaker = True
        batch.append(piece)
        speakers.add(speaker)
        width += piece_width
    if batch:
        chunks.append("".join(batch))
    return chunks


def split_text(text: str, max_chars: int = 200, *, max_speakers: int = 3) -> list[str]:
    """Split text and tagged dialogue without breaking natural control units.

    Later chunks receive an inherited speaker tag when needed. Those inserted
    prefixes and scope-balancing tags are only for individual inference requests;
    callers retain the original full transcript in saved takes.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if isinstance(max_chars, bool) or not isinstance(max_chars, int) or max_chars < 1:
        raise ValueError("max_chars must be a positive integer")
    if (isinstance(max_speakers, bool) or not isinstance(max_speakers, int)
            or max_speakers < 1):
        raise ValueError("max_speakers must be a positive integer")
    if not text:
        return []
    if SPEAKER_TAG.search(text):
        return _split_speaker_text(text, max_chars, max_speakers)
    chunks = _split_plain_text(text, max_chars)
    if len(chunks) > 1:
        chunks = [
            chunk if index == 0 else "<|speaker:0|>" + chunk
            for index, chunk in enumerate(chunks)
        ]
    return chunks
