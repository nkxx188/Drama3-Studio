"""Official task serialization, real audio rows, and unchanged ordinary TTS."""

from types import SimpleNamespace
import re

import pytest
import torch

from drama_low_vram.protocols import (
    Character, InfillSpan, PreviousTake, SceneLine, VoiceReference,
    FIM_PREFIX, FIM_SUFFIX, LINE_SEPARATOR, END_OF_TURN,
    build_infill_prompt, build_scene_prompt, build_stream_prompt,
    build_tts_prompt, directed_text, validate_codes, validate_content,
)
from fish_speech.models.text2semantic.fish_omni.generate import build_drama3_prompt


class Tokenizer:
    def __init__(self):
        self.special = {"<|im_end|>": 3, FIM_PREFIX: 90, FIM_SUFFIX: 91,
                        LINE_SEPARATOR: 92, END_OF_TURN: 93}
        self.special.update({f"<|semantic:{index}|>": 100 + index for index in range(8)})

    def token_to_id(self, text):
        return self.special.get(text)

    def encode(self, text):
        ids = []
        while text:
            special = next((token for token in self.special if text.startswith(token)), None)
            if special is not None:
                ids.append(self.special[special])
                text = text[len(special):]
            else:
                ids.append(1000 + ord(text[0]))
                text = text[1:]
        return SimpleNamespace(ids=ids)


@pytest.fixture
def layout():
    return Tokenizer(), SimpleNamespace(
        audio_decoder=SimpleNamespace(num_codebooks=3, vocab_size=8),
        semantic_start_token_id=100, semantic_end_token_id=107, eos_token_id=3,
    )


def codes(count=6):
    return (torch.arange(count * 3).reshape(3, count) % 8).long()


def audible_rows(prompt, config):
    return prompt[(prompt[:, 0] >= config.semantic_start_token_id)
                  & (prompt[:, 0] <= config.semantic_end_token_id)]


def test_plain_and_multireference_history_prompts_match_upstream_bytes(layout):
    tokenizer, config = layout
    reference = [codes(), codes(4)]
    transcripts = ["<|speaker:0|> 第一份参考。", "<|speaker:1|> 第二份参考。"]
    history = [PreviousTake("<|speaker:0|> Previous.", codes(3))]
    actual = build_tts_prompt(
        "Hello.", tokenizer, config, transcripts=transcripts,
        references=reference, history=history)
    expected = build_drama3_prompt(
        "Hello.", tokenizer, config, prompt_texts=transcripts, prompt_tokens=reference,
        history=[(history[0].text, history[0].codes)])
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("start,end", [(0, 0), (0, 2), (2, 4), (6, 6), (4, 6)])
def test_infill_suffix_precedes_prefix_and_keeps_original_code_bytes(layout, start, end):
    tokenizer, config = layout
    source = codes()
    original = source.clone()
    span = InfillSpan(source, start, end, duration_frames=2)
    prompt = build_infill_prompt("The demo is Thursday.", tokenizer, config, span)
    # System instructions contain marker mentions too; find the final actual
    # input markers immediately before the two audio blocks.
    suffix_at = int(torch.where(prompt[:, 0] == tokenizer.token_to_id(FIM_SUFFIX))[0][-1])
    prefix_at = int(torch.where(prompt[:, 0] == tokenizer.token_to_id(FIM_PREFIX))[0][-1])
    assert suffix_at < prefix_at
    suffix, prefix = source[:, end:], source[:, :start]
    assert torch.equal(prompt[suffix_at + 1:suffix_at + 1 + suffix.shape[1], 1:], suffix.T)
    assert torch.equal(prompt[prefix_at + 1:prefix_at + 1 + prefix.shape[1], 1:], prefix.T)
    replacement = codes(3)
    joined = span.join(replacement, config)
    assert torch.equal(joined, torch.cat((prefix, replacement, suffix), dim=1))
    assert torch.equal(source, original)


def test_seconds_to_frames_covers_the_whole_selected_window(layout):
    _, config = layout
    source = torch.zeros(3, 100, dtype=torch.long)
    span = InfillSpan.from_seconds(source, 0.1, 0.5, duration_seconds=0.2)
    assert span.start == 2 and span.end == 11 and span.duration_frames == 4
    span.parts(config)
    for start, end in ((-1, 0), (1, 0), (0, 9), (float("nan"), 1)):
        with pytest.raises(ValueError):
            InfillSpan.from_seconds(source, start, end)


def test_insertion_at_a_fractional_frame_keeps_a_zero_width_window(layout):
    _, config = layout
    source = codes(20)
    span = InfillSpan.from_seconds(source, 0.101, 0.101)
    assert span.start == span.end == 2
    assert torch.equal(span.join(torch.empty(3, 0, dtype=torch.long), config), source)


def test_an_empty_infill_middle_can_delete_a_span_without_changing_either_side(layout):
    _, config = layout
    source = codes(8)
    joined = InfillSpan(source, 2, 5).join(torch.empty(3, 0, dtype=torch.long), config)
    assert torch.equal(joined, torch.cat((source[:, :2], source[:, 5:]), dim=1))


def test_scene_and_stream_use_structural_tokens_without_fake_audio(layout):
    tokenizer, config = layout
    prompt = build_scene_prompt(
        "A quiet room.", [Character("Ada", "steady female voice", 0),
                         Character("Bo", "warm male voice", 1)],
        [SceneLine("Ada", "Hello."), SceneLine("Bo", "Good morning.")],
        tokenizer, config)
    assert (prompt[:, 0] == tokenizer.token_to_id(LINE_SEPARATOR)).sum() >= 1
    assert len(audible_rows(prompt, config)) == 0
    history = [PreviousTake("It was late.", codes(2))]
    streamed = build_stream_prompt(
        "and the hall was empty.", "", tokenizer, config, history=history)
    assert (streamed[:, 0] == tokenizer.token_to_id(END_OF_TURN)).sum() >= 1
    assert torch.equal(audible_rows(streamed, config)[:, 1:], history[0].codes.T)


def test_stream_history_keeps_the_original_user_lookahead_and_audio(layout):
    tokenizer, config = layout
    history = [PreviousTake("It was late", codes(2), "and the hall was empty.")]
    prompt = build_stream_prompt(
        "and the hall was empty.", "", tokenizer, config, history=history)
    markers = torch.where(prompt[:, 0] == tokenizer.token_to_id(END_OF_TURN))[0]
    # The system mentions the marker; each actual user turn has one too.
    assert len(markers) == 3
    marker = int(markers[1])
    before = tokenizer.encode("<|im_start|>user\nIt was late").ids
    after = tokenizer.encode("and the hall was empty.").ids
    assert prompt[marker - len(before):marker, 0].tolist() == before
    assert prompt[marker + 1:marker + 1 + len(after), 0].tolist() == after
    assert torch.equal(audible_rows(prompt, config)[:, 1:], history[0].codes.T)


def test_reference_speakers_and_free_directions_are_preserved():
    assert VoiceReference("voice.wav", "Hello", 3).transcript() == "<|speaker:3|> Hello"
    assert VoiceReference("voice.wav", "<|speaker:3|> Hello", 3).transcript() == "<|speaker:3|> Hello"
    with pytest.raises(ValueError, match="conflicts"):
        VoiceReference("voice.wav", "<|speaker:2|> Hello", 3).transcript()
    target = directed_text("<|speaker:2|> Hello.", "softer than before", language="en", duration_frames=21)
    assert target.startswith("<|speaker:2|> Direction: softer than before\nLine: Hello.")
    assert "[language: en]" in target and "[audio tokens: 21]" in target


def test_stream_reference_blocks_keep_all_codes_and_global_speaker_tags(layout):
    tokenizer, config = layout
    first, second = codes(3), codes(4)
    prompt = build_stream_prompt(
        "<|speaker:2|> One.", "Two.", tokenizer, config,
        transcripts=["<|speaker:2|> Reference one.", "<|speaker:2|> Reference two."],
        references=[first, second])
    assert torch.equal(audible_rows(prompt, config)[:, 1:], torch.cat((first, second), dim=1).T)
    text = "".join(chr(int(value) - 1000) for value in prompt[:, 0] if int(value) >= 1000)
    assert "[Reference 1]\nText: <|speaker:2|>" in text
    assert "[Reference 2]\nText: <|speaker:2|>" in text
    with pytest.raises(ValueError, match="both a transcript and codes"):
        build_stream_prompt("One.", "Two.", tokenizer, config, references=[first])


@pytest.mark.parametrize("token", [FIM_PREFIX, FIM_SUFFIX, LINE_SEPARATOR, END_OF_TURN,
                                 "<|im_start|>", "<|semantic:0|>"])
def test_user_content_cannot_inject_task_structure(token):
    with pytest.raises((ValueError, RuntimeError)):
        validate_content(f"Hello {token} world", "text")


def test_invalid_scene_and_infill_fail_before_model_work(layout):
    tokenizer, config = layout
    with pytest.raises(ValueError, match="unknown character"):
        build_scene_prompt("A room", [Character("Ada", "voice")],
                           [SceneLine("Bo", "Hi")], tokenizer, config)
    for span in (InfillSpan(codes(), -1, 3), InfillSpan(codes(), 3, 2),
                 InfillSpan(codes(), 0, 8), InfillSpan(codes(), 1, 3, 0)):
        with pytest.raises(ValueError):
            span.parts(config)
    with pytest.raises(ValueError):
        validate_codes(codes().float(), config)
