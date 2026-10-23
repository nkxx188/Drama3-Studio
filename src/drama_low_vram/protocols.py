"""Drama3 task prompts from the official report, Appendix B/C.

Audio spans are actual codec rows, not transcript placeholders. The ordinary
TTS builder remains upstream's byte-for-byte template. FIM markers and scene
separators are constructed here and cannot be injected through user content.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence
import math
import re

import torch
from torch import Tensor

from fish_speech.models.text2semantic.fish_omni.generate import (
    DRAMA3_GENERATION_START, DRAMA3_MESSAGE_END, DRAMA3_MAX_PROMPT_ROWS,
    DRAMA3_REFERENCE_INSTRUCTION,
    SPEAKER_TAG_PATTERN, UNICODE_WHITE_SPACE,
    build_drama3_prompt, validate_drama3_text, with_speaker_tag,
)


FIM_SYSTEM = (
    "fill in the missing part of the speech; the suffix follows <|fim_suffix|> "
    "and the prefix follows <|fim_prefix|>"
)
STREAM_SYSTEM = "streaming speech continuation with one-chunk lookahead split by <|endofturn|>"
LINE_SEPARATOR = "<|audio_separator|>"
END_OF_TURN = "<|endofturn|>"
FIM_SUFFIX = "<|fim_suffix|>"
FIM_PREFIX = "<|fim_prefix|>"
RESERVED_CONTENT = (LINE_SEPARATOR, END_OF_TURN, FIM_SUFFIX, FIM_PREFIX)


def validate_content(text: str, location: str, *, empty: bool = False) -> None:
    if not isinstance(text, str):
        raise TypeError(f"{location} must be a string")
    if empty and not text.strip(UNICODE_WHITE_SPACE):
        return
    validate_drama3_text(text, location)
    if any(token in text for token in RESERVED_CONTENT):
        raise ValueError(f"{location}: structural task markers belong to the prompt builder")
    for token in re.findall(r"<\|[^|\n]*\|>", text):
        if SPEAKER_TAG_PATTERN.fullmatch(token) is None:
            raise ValueError(f"{location}: special control tokens belong to the prompt builder")


def validate_codes(codes: Tensor, config, location="audio", *, empty=False) -> Tensor:
    if (not isinstance(codes, Tensor) or codes.ndim != 2
            or codes.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)
            or codes.shape[0] != config.audio_decoder.num_codebooks
            or (codes.shape[1] < 1 and not empty)):
        raise ValueError(f"{location}: expected integer [codebooks, frames] codec codes")
    if codes.numel() and (int(codes.min()) < 0 or int(codes.max()) >= config.audio_decoder.vocab_size):
        raise ValueError(f"{location}: codec code outside vocabulary")
    return codes.detach().to(device="cpu", dtype=torch.int64).contiguous()


@dataclass(frozen=True)
class VoiceReference:
    audio: str
    text: str
    speaker: int = 0

    def transcript(self):
        validate_content(self.text, "reference transcript")
        if isinstance(self.speaker, bool) or not isinstance(self.speaker, int) or not 0 <= self.speaker <= 15:
            raise ValueError("Reference speaker must be an integer from 0 to 15")
        stripped = self.text.lstrip(UNICODE_WHITE_SPACE)
        leading = SPEAKER_TAG_PATTERN.match(stripped)
        if leading is not None:
            if int(leading.group(1)) != self.speaker:
                raise ValueError("Reference tag conflicts with its assigned speaker")
            return self.text
        return f"<|speaker:{self.speaker}|> {self.text}"


@dataclass(frozen=True)
class PreviousTake:
    text: str
    codes: Tensor
    lookahead: str | None = None


@dataclass(frozen=True)
class Character:
    name: str
    description: str
    speaker: int = 0


@dataclass(frozen=True)
class SceneLine:
    character: str
    text: str


@dataclass(frozen=True)
class InfillSpan:
    """Codec frame boundaries [start,end), with an optional new-span hint."""
    source: Tensor
    start: int
    end: int
    duration_frames: int | None = None

    def parts(self, config):
        source = validate_codes(self.source, config, "infill source")
        if (isinstance(self.start, bool) or not isinstance(self.start, int)
                or isinstance(self.end, bool) or not isinstance(self.end, int)
                or not 0 <= self.start <= self.end <= source.shape[1]):
            raise ValueError("Infill boundaries must lie within the source codec frames")
        if self.duration_frames is not None and (
            isinstance(self.duration_frames, bool) or not isinstance(self.duration_frames, int)
            or self.duration_frames < 1
        ):
            raise ValueError("Infill duration hint must be a positive frame count")
        return source[:, :self.start], source[:, self.end:]

    @classmethod
    def from_seconds(cls, source, start, end, *, sample_rate=44100, frame_length=2048,
                     duration_seconds=None):
        if (isinstance(start, bool) or isinstance(end, bool)
                or not math.isfinite(start) or not math.isfinite(end)
                or start < 0 or end < start):
            raise ValueError("Infill time boundaries must be finite and ordered")
        limit = source.shape[1]
        begin = min(limit, math.floor(start * sample_rate / frame_length))
        stop = begin if start == end else min(limit, math.ceil(end * sample_rate / frame_length))
        if start * sample_rate > limit * frame_length + 1 or end * sample_rate > limit * frame_length + 1:
            raise ValueError("Infill time exceeds the source duration")
        duration = None
        if duration_seconds is not None:
            if (isinstance(duration_seconds, bool) or not math.isfinite(duration_seconds)
                    or duration_seconds <= 0):
                raise ValueError("Infill duration hint must be finite and positive")
            duration = max(1, round(duration_seconds * sample_rate / frame_length))
        return cls(source, begin, stop, duration)

    def join(self, replacement: Tensor, config) -> Tensor:
        prefix, suffix = self.parts(config)
        middle = validate_codes(replacement, config, "infill replacement", empty=True)
        return torch.cat((prefix, middle, suffix), dim=1)


class _Rows:
    def __init__(self, tokenizer, config):
        self.tokenizer, self.config, self.parts = tokenizer, config, []

    def text(self, text):
        ids = torch.tensor(self.tokenizer.encode(text).ids, dtype=torch.long)
        rows = torch.zeros(len(ids), 1 + self.config.audio_decoder.num_codebooks, dtype=torch.long)
        rows[:, 0] = ids
        self.parts.append(rows)

    def audio(self, codes, *, empty=False):
        codes = validate_codes(codes, self.config, empty=empty)
        self.parts.append(torch.cat(
            (codes[:1] + self.config.semantic_start_token_id, codes), dim=0).T)

    def token(self, token):
        token_id = self.tokenizer.token_to_id(token)
        if token_id is None:
            raise ValueError(f"The tokenizer is missing required task token {token}")
        row = torch.zeros(1, 1 + self.config.audio_decoder.num_codebooks, dtype=torch.long)
        row[0, 0] = token_id
        self.parts.append(row)

    def result(self):
        result = torch.cat(self.parts)
        if len(result) > DRAMA3_MAX_PROMPT_ROWS:
            raise ValueError(f"Task prompt exceeds {DRAMA3_MAX_PROMPT_ROWS} rows")
        return result


def _history(rows, history):
    for index, take in enumerate(history):
        validate_content(take.text, f"previous take {index + 1}")
        rows.text("<|im_start|>user\n")
        if take.lookahead is None:
            rows.text(with_speaker_tag(take.text))
        else:
            validate_content(take.lookahead, "previous lookahead", empty=True)
            rows.text(take.text)
            rows.token(END_OF_TURN)
            rows.text(take.lookahead)
        rows.text(DRAMA3_MESSAGE_END)
        rows.text(DRAMA3_GENERATION_START)
        rows.audio(take.codes)
        rows.text(DRAMA3_MESSAGE_END)


def directed_text(text: str, direction: str = "", *, language=None, duration_frames=None):
    """Natural-language controls stay ordinary text; speaker tokens stay leading."""
    validate_content(text, "text")
    validate_content(direction, "direction", empty=True)
    tagged = with_speaker_tag(text)
    stripped = tagged.lstrip(UNICODE_WHITE_SPACE)
    tag = SPEAKER_TAG_PATTERN.match(stripped)
    body = stripped[tag.end():].lstrip(UNICODE_WHITE_SPACE)
    if direction:
        body = f"Direction: {direction}\nLine: {body}"
    if language is not None:
        if not isinstance(language, str) or not language.isascii() or not language.replace("-", "").isalpha():
            raise ValueError("Language hint must be a language code")
        body += f" [language: {language}]"
    if duration_frames is not None:
        if (isinstance(duration_frames, bool) or not isinstance(duration_frames, int)
                or duration_frames < 1):
            raise ValueError("Duration hint must be a positive frame count")
        body += f" [audio tokens: {duration_frames}]"
    return tag.group(0) + " " + body


def build_tts_prompt(text, tokenizer, config, *, transcripts=(), references=(),
                     history: Sequence[PreviousTake] = (), direction="", language=None,
                     duration_frames=None):
    """No-option ordinary TTS delegates unchanged to the pinned upstream builder."""
    validate_content(text, "text")
    history_rows = [(take.text, validate_codes(take.codes, config, "previous take")) for take in history]
    for take in history:
        validate_content(take.text, "previous take")
    target = (
        directed_text(text, direction, language=language, duration_frames=duration_frames)
        if direction or language is not None or duration_frames is not None else text
    )
    return build_drama3_prompt(
        target, tokenizer, config, prompt_texts=transcripts, prompt_tokens=references,
        history=history_rows,
    )


def build_infill_prompt(text, tokenizer, config, span: InfillSpan):
    validate_content(text, "target transcript")
    prefix, suffix = span.parts(config)
    rows = _Rows(tokenizer, config)
    rows.text("<|im_start|>system\n")
    rows.text(FIM_SYSTEM)
    rows.text(DRAMA3_MESSAGE_END)
    rows.text("<|im_start|>user\n")
    target = directed_text(text, duration_frames=span.duration_frames)
    rows.text(target)
    rows.text("\n")
    rows.token(FIM_SUFFIX)
    rows.audio(suffix, empty=True)
    rows.token(FIM_PREFIX)
    rows.audio(prefix, empty=True)
    rows.text(DRAMA3_MESSAGE_END)
    rows.text(DRAMA3_GENERATION_START)
    return rows.result()


def validate_scene_inputs(scene, characters, lines):
    validate_content(scene, "scene")
    if not characters or not lines or len(characters) > 16:
        raise ValueError("A scene needs 1..16 characters and at least one line")
    names, speakers = set(), set()
    for character in characters:
        validate_content(character.name, "character name")
        validate_content(character.description, "character description")
        if "\n" in character.name or ":" in character.name or character.name in names:
            raise ValueError("Character names must be unique, single-line names")
        if (isinstance(character.speaker, bool) or not isinstance(character.speaker, int)
                or not 0 <= character.speaker <= 15 or character.speaker in speakers):
            raise ValueError("Scene speaker IDs must be unique integers from 0 to 15")
        names.add(character.name)
        speakers.add(character.speaker)
    for line in lines:
        if line.character not in names:
            raise ValueError("Scene line refers to an unknown character")
        validate_content(line.text, "scene line")


def build_scene_prompt(scene, characters, lines, tokenizer, config, *, history=()):
    validate_scene_inputs(scene, characters, lines)
    rows = _Rows(tokenizer, config)
    rows.text("<|im_start|>system\n")
    rows.text("Scene: " + scene + "\nCharacters:\n")
    for character in characters:
        rows.text(f"- {character.name}: {character.description}\n")
    rows.text("Voice every line of the script in order, with <|audio_separator|> between consecutive lines.")
    rows.text(DRAMA3_MESSAGE_END)
    _history(rows, history)
    rows.text("<|im_start|>user\n")
    for index, line in enumerate(lines):
        if index:
            rows.text("\n")
            rows.token(LINE_SEPARATOR)
            rows.text("\n")
        rows.text(f"{line.character}: {line.text}")
    rows.text(DRAMA3_MESSAGE_END)
    rows.text(DRAMA3_GENERATION_START)
    return rows.result()


def design_character(name, description, scene=""):
    validate_content(name, "character name")
    validate_content(description, "voice description")
    validate_content(scene, "scene", empty=True)
    if any(character in name for character in "\r\n:") or len(name) > 80:
        raise ValueError("Character name must be a short single-line name without a colon")
    return {"name": name, "description": description, "scene": scene}


def build_design_prompt(text, tokenizer, config, *, name="Voice", description, scene=""):
    """Official Space's single-character Figure 13 prompt, without output separators."""
    character = design_character(name, description, scene)
    validate_content(text, "text")
    rows = _Rows(tokenizer, config)
    rows.text("<|im_start|>system\n")
    lines = [f"Scene: {scene}"] if scene else []
    lines += ["Characters:", f"- {name}: {description}",
              "Voice every line of the script in order, with <|audio_separator|> "
              "between consecutive lines."]
    rows.text("\n".join(lines))
    rows.text(DRAMA3_MESSAGE_END)
    rows.text("<|im_start|>user\n")
    rows.text(f"{character['name']}: {text}")
    rows.text(DRAMA3_MESSAGE_END)
    rows.text(DRAMA3_GENERATION_START)
    return rows.result()


def build_stream_prompt(text, lookahead, tokenizer, config, *, history=(),
                        transcripts=(), references=()):
    validate_content(text, "stream chunk")
    validate_content(lookahead, "lookahead", empty=True)
    rows = _Rows(tokenizer, config)
    rows.text("<|im_start|>system\n")
    rows.text(STREAM_SYSTEM)
    if len(transcripts) != len(references):
        raise ValueError("Every streaming reference needs both a transcript and codes")
    if references:
        # Combine the report's stream instruction with its ordinary reference
        # blocks. The no-reference Figure 10 layout stays byte-for-byte intact.
        rows.text("\n" + DRAMA3_REFERENCE_INSTRUCTION)
        for index, (transcript, codes) in enumerate(zip(transcripts, references), 1):
            validate_content(transcript, f"stream reference {index}")
            tagged = with_speaker_tag(transcript)
            body = tagged.lstrip(UNICODE_WHITE_SPACE)
            tag = SPEAKER_TAG_PATTERN.match(body)
            leading = tagged[:len(tagged) - len(body)]
            canonical = tag.group(0) + leading + body[tag.end():]
            rows.text(f"\n\n[Reference {index}]\nText: {canonical}\nSpeech: ")
            rows.audio(codes)
    rows.text(DRAMA3_MESSAGE_END)
    _history(rows, history)
    rows.text("<|im_start|>user\n")
    rows.text(text)
    rows.token(END_OF_TURN)
    rows.text(lookahead)
    rows.text(DRAMA3_MESSAGE_END)
    rows.text(DRAMA3_GENERATION_START)
    return rows.result()
