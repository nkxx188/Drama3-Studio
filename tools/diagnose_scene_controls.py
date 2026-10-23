"""Measure actual separator logits for the report's scene protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from benchmark_engine_decode import DramaEngine, EngineOptions, generate, seed_sampling, torch
from drama_low_vram.protocols import Character, SceneLine, build_scene_prompt, LINE_SEPARATOR
from drama_low_vram.codec import write_wav
from drama_low_vram.workflows import audio_segments


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 123, 2026])
    parser.add_argument("--output", type=Path, default=ROOT / "research/scene-control-diagnosis.json")
    parser.add_argument("--probe-specials", action="store_true")
    parser.add_argument("--paper-example", action="store_true")
    parser.add_argument("--joined-text", action="store_true",
                        help="Tokenize each complete system/user literal without builder seams.")
    parser.add_argument("--max-frames", type=int, default=384)
    parser.add_argument("--prefix", default="scene-diagnostic")
    args = parser.parse_args()
    torch.set_num_threads(1)
    report = {"scope": "Actual model separator probability/rank; no forced separator or fabricated segmentation.",
              "runs": []}
    with DramaEngine(EngineOptions(
        checkpoint=ROOT / "checkpoints/drama-3", expert_storage="mmap",
        cache_experts=1, staged_experts=8, pair_rows=64, prefill_chunk=32,
        backbone_graph=True, combined_decode_bank=True, cpu_experts=2, host_gather="auto",
    )) as engine:
        engine.load()
        prompt = build_scene_prompt(
            "A quiet station platform before sunrise. Two travelers speak in short, restrained lines.",
            [Character("Morgan", "curious, mid-range voice, slightly breathy", 0),
             Character("Riley", "steady, dry, lower-mid voice", 1)],
            [SceneLine("Morgan", "I thought the train had already left."),
             SceneLine("Riley", "It hasn't. Listen for it.")],
            engine.model.tokenizer, engine.model.config)
        if args.paper_example:
            prompt = build_scene_prompt(
                "a small office on a Monday morning; the weekly meeting is about to start.",
                [Character("Mr. Carter", "steady male voice, about 40, formal."),
                 Character("Emma", "young female voice, sleepy, half talking to herself.", 1)],
                [SceneLine("Mr. Carter", "[clears throat][serious] Good morning, everyone. Let’s get started."),
                 SceneLine("Mr. Carter", "[reading from a sheet] First item: the launch date."),
                 SceneLine("Emma", "[softly][to herself] Here we go again...")],
                engine.model.tokenizer, engine.model.config)
        if args.joined_text:
            tokenizer = engine.model.tokenizer
            # There are no embedded VQ rows in a text-only scene prompt.
            tokens = tokenizer.decode(prompt[:, 0].tolist(), skip_special_tokens=False)
            ids = tokenizer.encode(tokens).ids
            prompt = torch.zeros(
                len(ids), 1 + engine.model.config.audio_decoder.num_codebooks, dtype=torch.int64)
            prompt[:, 0] = torch.tensor(ids)
        report.update(
            paper_example=args.paper_example, joined_text=args.joined_text,
            max_frames=args.max_frames, prompt_rows=len(prompt),
            serialized_prompt=engine.model.tokenizer.decode(
                prompt[:, 0].tolist(), skip_special_tokens=False))
        plan, _ = engine._prepare(len(prompt) + args.max_frames, output_controls=(LINE_SEPARATOR,))
        head = engine.model.backbone.protocol_head
        ids = head.candidates(engine.model.backbone.lm_head)
        separator = engine.model.tokenizer.token_to_id(LINE_SEPARATOR)
        sep_index = int(torch.where(ids == separator)[0][0])
        probe = None
        if args.probe_specials:
            from drama_low_vram.control_head import ControlHead

            probe = ControlHead.load(
                engine.checkpoint, engine.model.tokenizer, engine.model.backbone.lm_head,
                ("<|endofturn|>", "<|audio_separator|>", "<|speaker_change|>",
                 *[f"<|speaker:{index}|>" for index in range(16)]),
            )
            names = {
                int(engine.model.tokenizer.token_to_id(token)): token
                for token in ("<|endofturn|>", "<|audio_separator|>", "<|speaker_change|>",
                              *[f"<|speaker:{index}|>" for index in range(16)])
            }
        for seed in args.seeds:
            plan, _ = engine._prepare(len(prompt) + args.max_frames, output_controls=(LINE_SEPARATOR,))
            head = engine.model.backbone.protocol_head
            ids = head.candidates(engine.model.backbone.lm_head)
            samples = []
            last = time.perf_counter()

            def logits(hidden):
                values = head.logits(hidden, engine.model.backbone.lm_head)
                p = values.float().softmax(-1)
                samples.append({
                    "step": len(samples), "separator_rank": int((values[0] > values[0, sep_index]).sum()) + 1,
                    "separator_probability": float(p[0, sep_index]),
                    "top_token": int(ids[values.argmax(-1)[0]]),
                    "separator_logit": float(values[0, sep_index]),
                })
                if probe is not None:
                    raw = probe.logits(hidden, engine.model.backbone.lm_head)[0].float()
                    probe_ids = probe.candidates(engine.model.backbone.lm_head)
                    extra_mask = torch.isin(probe_ids, probe.ids)
                    extra_values, extra_ids = raw[extra_mask], probe_ids[extra_mask]
                    top = int(extra_values.argmax())
                    samples[-1]["top_extra_token"] = names[int(extra_ids[top])]
                    samples[-1]["top_extra_rank"] = int((raw > extra_values[top]).sum()) + 1
                    samples[-1]["top_extra_probability"] = float(raw.softmax(-1)[extra_mask][top])
                return values

            def progress(count, _limit):
                nonlocal last
                now = time.perf_counter()
                if now - last > 15:
                    print(json.dumps({"seed": seed, "frames": count}), flush=True)
                    last = now

            seed_sampling(seed)
            result = generate(
                engine.model, prompt, max_new_tokens=args.max_frames,
                prefill_chunk_size=plan.prefill_chunk, backbone_graph=engine.backbone_graph,
                tower_graph=engine.graph, logit_head=logits, candidate_ids=ids,
                control_token_ids=(separator,), progress=progress)
            segments, boundaries = audio_segments(result.frames, engine.model.config, separator)
            audio = engine._decode_audio_segments([torch.cat(segments, dim=1)])
            output = ROOT / "outputs" / f"{args.prefix}-{seed}.wav"
            write_wav(audio, output)
            row = {
                "seed": seed, "finish_reason": result.finish_reason, "frames": len(result.frames),
                "boundaries": boundaries, "segments": len(segments), "output": str(output),
                "min_separator_rank": min(item["separator_rank"] for item in samples),
                "peak_separator_probability": max(item["separator_probability"] for item in samples),
                "samples": samples,
            }
            report["runs"].append(row)
            args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            print(json.dumps({key: value for key, value in row.items() if key != "samples"}), flush=True)
            if len(segments) == 2:
                break


if __name__ == "__main__":
    main()
