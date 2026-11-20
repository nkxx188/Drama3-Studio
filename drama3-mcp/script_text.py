"""Translate Studio dialogue and match spoken excerpts without rewriting the script."""

import re
import unicodedata

SPEAKER = re.compile(r"[Ss]([0-9]{1,2})")
TURN = re.compile(r"^[ \t]*([Ss][0-9]+)[ \t]*[:\uFF1A][ \t]*(.*)$")
NATIVE_SPEAKER = re.compile(r"<\|speaker:[0-9]+\|>")
MARKUP = re.compile(r"<\|phoneme_start\|>[^<>]*<\|phoneme_end\|>|<[^<>\r\n]*>|\[[^\]\r\n]*\]")
SCOPE = re.compile(r"<(/?)(whisper|emphasis|soft|fast|slow|stress|forceful)>")
SENTENCE_PUNCTUATION = ".!?;\u2026\u3002\uff01\uff1f\uff1b"


def speaker_label(value: str) -> str:
    match = SPEAKER.fullmatch(value)
    if not match or not 1 <= int(match[1]) <= 16:
        raise ValueError("Use speaker labels S1 through S16.")
    return f"S{int(match[1])}"


def dialogue_lines(script: str) -> list[dict[str, str]]:
    if NATIVE_SPEAKER.search(script):
        raise ValueError("Use S1/S2 prefixes instead of internal speaker tags.")
    turns = []
    for number, line in enumerate(script.splitlines(), 1):
        match = TURN.match(line)
        if match:
            turns.append((speaker_label(match[1]), [match[2]]))
        elif turns:
            turns[-1][1].append(line)
        elif line.strip():
            raise ValueError(f"Dialogue line {number} must start with a speaker label such as S1:.")
    if not turns:
        raise ValueError("Provide a dialogue script with S1/S2 speaker labels.")
    lines = []
    for label, parts in turns:
        text = "\n".join(parts).strip()
        if not text:
            raise ValueError(f"A turn for {label} has no spoken text.")
        lines.append({"character": label, "text": text})
    return lines


def _spoken(text: str) -> tuple[str, list[int]]:
    characters, positions = [], []
    start = 0
    # Record source offsets so matching can ignore punctuation and tags without losing them.
    for tag in [*MARKUP.finditer(text), None]:
        end = tag.start() if tag else len(text)
        for offset in range(start, end):
            for character in unicodedata.normalize("NFKC", text[offset]).casefold():
                if not character.isspace() and not unicodedata.category(character).startswith("P"):
                    characters.append(character)
                    positions.append(offset)
        if tag:
            start = tag.end()
    return "".join(characters), positions


def _word_character(character: str) -> bool:
    return character.isdecimal() or "LATIN" in unicodedata.name(character, "")


def _inside_word(text: str, offset: int) -> bool:
    return 0 < offset < len(text) and _word_character(text[offset - 1]) and _word_character(text[offset])


def excerpt_spans(text: str, excerpt: str) -> list[tuple[int, int]]:
    spoken, offsets = _spoken(text)
    wanted, _ = _spoken(excerpt)
    if not wanted:
        return []
    spans = []
    start = spoken.find(wanted)
    while start >= 0:
        end = start + len(wanted)
        left, right = offsets[start], offsets[end - 1] + 1
        # Do not split an English word, number, or normalized ligature.
        split_normalized = (start > 0 and offsets[start - 1] == left
                            or end < len(spoken) and offsets[end] == right - 1)
        if not (split_normalized or _inside_word(text, left) or _inside_word(text, right)):
            spans.append((left, right))
        start = spoken.find(wanted, start + 1)
    return spans


def contextual_span(text: str, excerpt: str, context: str) -> tuple[int, int] | None:
    matches = excerpt_spans(text, context)
    if len(matches) != 1 or len(excerpt_spans(context, excerpt)) != 1:
        return None
    start, end = matches[0]
    inner = excerpt_spans(text[start:end], excerpt)
    return (start + inner[0][0], start + inner[0][1]) if len(inner) == 1 else None


def replace_span(text: str, span: tuple[int, int], replacement: str) -> str:
    def balanced(value):
        stack = []
        for match in SCOPE.finditer(value):
            if not match[1]:
                stack.append(match[2])
            elif not stack or stack.pop() != match[2]:
                return False
        return not stack

    start, end = span
    if NATIVE_SPEAKER.search(replacement) or not balanced(text[start:end]) or not balanced(replacement):
        raise ValueError("The replacement must preserve speaker labels and paired tags. Check the selected words.")
    if MARKUP.sub("", replacement).rstrip().endswith(tuple(SENTENCE_PUNCTUATION)):
        while end < len(text) and text[end] in SENTENCE_PUNCTUATION:
            end += 1
    return text[:start] + replacement + text[end:]
