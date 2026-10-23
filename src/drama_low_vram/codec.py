"""Memory-bounded loading and use of the Fish Audio DAC codec.

The upstream codec implementation allocates a fixed ``32768 x 32768`` boolean
causal mask while constructing every Transformer.  That allocation is much
larger than any practical reference clip or generated sample and can exhaust
an 8 GiB GPU before the codec weights are loaded.  This module keeps the
upstream model and state-dict layout intact, but temporarily uses a bounded
Transformer constructor while the codec is instantiated.

The patch is scoped to codec construction, protected by a lock, and restored
before this function returns.  No vendor source or process-wide ``torch.ones``
behavior is changed.

The loading and encode/decode flow is adapted from
``vendor/fish-speech/fish_speech/models/text2semantic/inference.py`` and
remains subject to the Fish Audio Community License in
``vendor/fish-speech/LICENSE``.
"""

from __future__ import annotations

import gc
import os
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import torch
from torch import nn


_CODEC_BUILD_LOCK = threading.RLock()
_DEFAULT_MAX_FRAMES = 2048


def codec_operation_roots(codec: nn.Module, operation: str = "all") -> tuple[nn.Module, ...]:
    """Stage only branches used by the unchanged official encode/decode calls.

    Unknown implementations use whole-codec placement. Exact types matter:
    subclasses may override forward/from_indices and need additional modules.
    """
    if operation not in ("all", "encode", "decode", "decode_latents", "decode_waveform"):
        raise ValueError("Unknown codec operation")
    if (operation == "all"
            or type(codec).__module__ != "fish_speech.models.dac.modded_dac"
            or type(codec).__name__ != "DAC"):
        return (codec,)
    from fish_speech.models.dac.modded_dac import DAC
    from fish_speech.models.dac.rvq import DownsampleResidualVectorQuantize

    if type(codec) is not DAC or type(codec.quantizer) is not DownsampleResidualVectorQuantize:
        return (codec,)
    if operation == "encode":
        # Official quantizer.forward also reconstructs latents; keep that path
        # intact even though encode() returns only indices.
        return codec.encoder, codec.quantizer
    quantizer = codec.quantizer
    latents = (
        quantizer.semantic_quantizer, quantizer.quantizer,
        quantizer.post_module, quantizer.upsample,
    )
    if operation == "decode_latents":
        return latents
    if operation == "decode_waveform":
        return (codec.decoder,)
    return (*latents, codec.decoder)


def codec_supports_split_decode(codec: nn.Module) -> bool:
    """Only the exact official implementation has an audited branch contract."""
    return codec_operation_roots(codec, "decode_latents") != (codec,)


def codec_workspace_bytes(max_frames: int, operation: str = "all") -> int:
    """Admission estimate; actual CUDA OOM still retries with parked backbone."""
    _require_max_frames(max_frames)
    if operation == "decode_latents":
        return 256 * 1024**2 + max_frames * 256 * 1024
    return 256 * 1024**2 + max_frames * 2 * 1024**2


def codec_operation_modules(codec: nn.Module, operation: str = "all") -> tuple[nn.Module, ...]:
    """Unique descendants, for direct tensor accounting and CPU restoration."""
    modules = {}
    for root in codec_operation_roots(codec, operation):
        for module in root.modules():
            modules.setdefault(id(module), module)
    return tuple(modules.values())


def _require_max_frames(max_frames: int | None) -> int:
    if max_frames is None:
        return _DEFAULT_MAX_FRAMES
    if isinstance(max_frames, bool) or not isinstance(max_frames, int):
        raise TypeError("max_frames must be an integer or None")
    if max_frames < 1:
        raise ValueError("max_frames must be positive")
    return max_frames


def _bounded_transformer_init(
    module: Any,
    original_init: Any,
    max_frames: int,
) -> Any:
    """Build the upstream Transformer with a context-sized cache.

    ``WindowLimitedTransformer`` calls ``Transformer.__init__`` through normal
    Python method lookup.  Replacing that one method for the duration of
    ``hydra.instantiate`` therefore covers the encoder and quantizer
    transformers without changing their classes or state-dict keys.
    """

    def init(self: nn.Module, config: Any) -> None:
        # This is the upstream constructor with only the two hard-coded
        # sequence lengths bounded.  Keep the module hierarchy and buffers
        # identical so the official checkpoint remains loadable.
        nn.Module.__init__(self)
        self.config = config
        self.layers = nn.ModuleList(
            module.TransformerBlock(config) for _ in range(config.n_layer)
        )
        self.norm = module.RMSNorm(config.dim, eps=config.norm_eps)

        # The encoder sees approximately four times as many positions as the
        # final codec frames. The upstream block_size is a training setting,
        # not an inference limit; capping to it breaks longer references.
        context = max_frames * 4

        if config.pos_embed_type == "rope":
            freqs_cis = module.precompute_freqs_cis(
                context, config.head_dim, config.rope_base
            )
            self.register_buffer("freqs_cis", freqs_cis, persistent=False)
        else:
            self.register_buffer("freqs_cis", None)

        causal_mask = torch.tril(torch.ones(context, context, dtype=torch.bool))
        self.register_buffer("causal_mask", causal_mask, persistent=False)

        self.max_batch_size = -1
        self.max_seq_length = -1
        self.use_kv_cache = False

    # ``original_init`` is kept in the closure to make the restoration intent
    # explicit and to fail closed if this helper is ever reused incorrectly.
    if original_init is None:
        raise RuntimeError("upstream codec Transformer constructor is missing")
    return init


@contextmanager
def _bounded_codec_transformers(max_frames: int) -> Iterator[None]:
    """Temporarily bound upstream codec Transformer allocations."""

    # Imports stay lazy: importing drama_low_vram must not pull in Hydra,
    # audiotools, or the full Fish Speech server stack.
    from fish_speech.models.dac import modded_dac

    with _CODEC_BUILD_LOCK:
        original_init = modded_dac.Transformer.__init__
        modded_dac.Transformer.__init__ = _bounded_transformer_init(
            modded_dac, original_init, max_frames
        )
        try:
            yield
        finally:
            modded_dac.Transformer.__init__ = original_init


def _codec_config_path() -> Path:
    import fish_speech

    package_file = getattr(fish_speech, "__file__", None)
    if package_file:
        package_root = Path(package_file).resolve().parent
    else:
        package_root = Path(next(iter(fish_speech.__path__))).resolve()
    return package_root / "configs" / "modded_dac_vq.yaml"


def _normalise_codec_state(state_dict: dict[str, torch.Tensor], codec: nn.Module) -> dict[str, torch.Tensor]:
    """Apply the same checkpoint normalization as Fish's reference loader."""

    if "state_dict" in state_dict and isinstance(state_dict["state_dict"], dict):
        state_dict = state_dict["state_dict"]
    if any("generator" in key for key in state_dict):
        state_dict = {
            key.replace("generator.", ""): value
            for key, value in state_dict.items()
            if "generator." in key
        }
    rebuilt_buffers = {
        f"{name}.{buffer}" if name else buffer
        for name, module in codec.named_modules()
        for buffer in module._non_persistent_buffers_set
    }
    return {key: value for key, value in state_dict.items() if key not in rebuilt_buffers}


@torch.inference_mode(False)
@torch.no_grad()
def load_codec(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device = "cuda:0",
    dtype: torch.dtype = torch.bfloat16,
    max_frames: int | None = _DEFAULT_MAX_FRAMES,
) -> nn.Module:
    """Load the official DAC codec with bounded construction-time memory.

    ``max_frames`` is the maximum number of codec frames supported by this
    instance.  At 44.1 kHz, one frame is 2048 samples (about 21.5 frames/sec).
    It controls only the non-persistent positional/mask buffers; model weights
    and the official state-dict names are unchanged.
    """

    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"codec checkpoint does not exist: {checkpoint_path}")
    if dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("codec dtype must be float16, bfloat16, or float32")
    max_frames = _require_max_frames(max_frames)

    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from torch.nn.utils.parametrize import is_parametrized, remove_parametrizations

    with _bounded_codec_transformers(max_frames):
        codec = instantiate(OmegaConf.load(str(_codec_config_path())))

    try:
        state_dict = torch.load(
            str(checkpoint_path),
            map_location="cpu",
            mmap=True,
            weights_only=True,
        )
        if not isinstance(state_dict, dict):
            raise TypeError("codec checkpoint must contain a state-dict mapping")
        state_dict = _normalise_codec_state(state_dict, codec)
        codec.load_state_dict(state_dict, strict=True)
        codec.eval()

        # Weight norm is folded in fp32 before casting, matching upstream.
        for submodule in list(codec.modules()):
            if is_parametrized(submodule, "weight"):
                remove_parametrizations(submodule, "weight", leave_parametrized=True)
            elif hasattr(submodule, "weight_g"):
                torch.nn.utils.remove_weight_norm(submodule)

        codec.to(device=torch.device(device), dtype=dtype)
        from .codec_decoder import flatten_decoder_calls

        flatten_decoder_calls(codec)
        codec._drama_low_vram_max_frames = max_frames
        return codec
    except Exception:
        codec.cpu()
        del codec
        gc.collect()
        raise


def load_codec_model(
    codec_checkpoint_path: str | Path,
    device: str | torch.device,
    precision: torch.dtype = torch.bfloat16,
    *,
    max_frames: int | None = _DEFAULT_MAX_FRAMES,
) -> nn.Module:
    """Compatibility wrapper matching Fish Speech's ``load_codec_model``."""

    return load_codec(
        codec_checkpoint_path,
        device=device,
        dtype=precision,
        max_frames=max_frames,
    )


def _codec_device(codec: nn.Module, device: str | torch.device | None) -> torch.device:
    if device is not None:
        return torch.device(device)
    try:
        return next(codec.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _check_frame_budget(codec: nn.Module, frame_count: int) -> None:
    limit = getattr(codec, "_drama_low_vram_max_frames", None)
    if limit is not None and frame_count > limit:
        raise ValueError(
            f"audio needs {frame_count} codec frames, but this codec was loaded "
            f"with max_frames={limit}; reload with a larger max_frames"
        )


@torch.inference_mode()
def encode_audio(
    audio_path: str | Path,
    codec: nn.Module,
    device: str | torch.device | None = None,
    *,
    return_cpu: bool = True,
) -> torch.Tensor:
    """Encode an audio file to ``(num_codebooks, frames)`` VQ indices."""

    import torchaudio

    target_device = _codec_device(codec, device)
    wav, sample_rate = torchaudio.load(str(audio_path))
    if wav.ndim != 2:
        raise ValueError(f"expected audio with shape (channels, samples), got {wav.shape}")
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    wav = torchaudio.functional.resample(wav, sample_rate, int(codec.sample_rate))
    sample_count = int(wav.shape[-1])
    frame_length = int(getattr(codec, "frame_length", 2048))
    frame_count = (sample_count + frame_length - 1) // frame_length
    _check_frame_budget(codec, frame_count)

    model_dtype = next(codec.parameters()).dtype
    audio = wav.to(device=target_device, dtype=model_dtype)[None]
    audio_lengths = torch.tensor([sample_count], device=target_device, dtype=torch.long)
    indices, feature_lengths = codec.encode(audio, audio_lengths)
    indices = indices[0, :, : int(feature_lengths[0].item())]
    return indices.cpu() if return_cpu else indices


@torch.inference_mode()
def decode_to_audio(
    codes: torch.Tensor,
    codec: nn.Module,
    device: str | torch.device | None = None,
    *,
    max_frames: int | None = None,
    return_cpu: bool = True,
) -> torch.Tensor:
    """Decode ``(num_codebooks, frames)`` VQ indices to a mono waveform."""

    if codes.ndim == 2:
        codes = codes.unsqueeze(0)
    if codes.ndim != 3:
        raise ValueError(f"codes must have shape (codebooks, frames) or (1, codebooks, frames), got {codes.shape}")
    if codes.shape[0] != 1:
        raise ValueError("only batch size 1 is supported by the low-VRAM codec API")
    frame_count = int(codes.shape[-1])
    limit = _require_max_frames(max_frames) if max_frames is not None else None
    if limit is not None and frame_count > limit:
        raise ValueError(f"codes contain {frame_count} frames, exceeding max_frames={limit}")
    _check_frame_budget(codec, frame_count)

    target_device = _codec_device(codec, device)
    decoded = codec.from_indices(codes.to(device=target_device, dtype=torch.long))
    audio = decoded[0, 0]
    return audio.cpu() if return_cpu else audio


def write_wav(
    audio: torch.Tensor,
    output_path: str | Path,
    *,
    sample_rate: int = 44100,
) -> Path:
    """Write a mono or channel-first waveform using soundfile."""

    import soundfile as sf

    output_path = Path(output_path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    array = audio.detach().cpu().float().numpy()
    if array.ndim == 3:
        array = array[0]
    if array.ndim == 2 and array.shape[0] <= 8:
        array = array.T
    sf.write(str(output_path), array, int(sample_rate))
    return output_path


def concatenate_wavs(paths, output_path, *, sample_rate=44100):
    """Bounded PCM16 export; preserve delivered samples without whole-file RAM."""
    import soundfile as sf

    paths = tuple(Path(path).resolve() for path in paths)
    target = Path(output_path).resolve()
    if not paths or target in paths:
        raise ValueError("Export needs separate nonempty source WAV files")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    try:
        with sf.SoundFile(
            temporary, mode="w", samplerate=sample_rate, channels=1,
            format="WAV", subtype="PCM_16",
        ) as destination:
            for path in paths:
                with sf.SoundFile(path) as source:
                    if (source.samplerate != sample_rate or source.channels != 1
                            or source.subtype != "PCM_16"):
                        raise ValueError("Export sources must be mono PCM16 at the same sample rate")
                    for block in source.blocks(blocksize=65536, dtype="int16"):
                        destination.write(block)
        os.replace(temporary, target)
        return target
    finally:
        if temporary.exists():
            temporary.unlink()


def release_codec(codec: nn.Module | None) -> None:
    """Move codec buffers off the accelerator and release cached allocations."""

    if codec is None:
        return
    codec.cpu()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


__all__ = [
    "codec_operation_modules",
    "codec_operation_roots",
    "codec_supports_split_decode",
    "codec_workspace_bytes",
    "decode_to_audio",
    "encode_audio",
    "load_codec",
    "load_codec_model",
    "release_codec",
    "write_wav",
]
