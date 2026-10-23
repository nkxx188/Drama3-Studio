"""Low-VRAM Drama 3 runtime components."""

from .offload import (
    ExpertOffloadManager,
    install_expert_offload,
    materialize_offloaded_backbone,
)
from .codec import (
    decode_to_audio,
    encode_audio,
    load_codec,
    release_codec,
    write_wav,
)
from .semantic_head import SemanticHead, install_semantic_head
from .checkpoint_reader import iter_readonly_checkpoint_tensors
from .autotune import (
    AutotuneConfig,
    AutotuneResult,
    HardwareInfo,
    HardwareSnapshot,
    PostLoadPlan,
    RuntimeConfig,
    autotune,
    choose_profile,
    inspect_and_autotune,
    inspect_hardware,
    plan_post_load,
    select_profile,
)

__all__ = [
    "ExpertOffloadManager",
    "install_expert_offload",
    "materialize_offloaded_backbone",
    "decode_to_audio",
    "encode_audio",
    "load_codec",
    "release_codec",
    "write_wav",
    "DramaEngine",
    "EngineOptions",
    "SemanticHead",
    "install_semantic_head",
    "iter_readonly_checkpoint_tensors",
    "AutotuneResult",
    "AutotuneConfig",
    "HardwareInfo",
    "HardwareSnapshot",
    "PostLoadPlan",
    "RuntimeConfig",
    "autotune",
    "choose_profile",
    "inspect_and_autotune",
    "inspect_hardware",
    "plan_post_load",
    "select_profile",
    "load_offloaded_drama3",
    "configure_model_runtime",
    "release_drama3",
]


def __getattr__(name: str):
    if name in {"DramaEngine", "EngineOptions"}:
        from .engine import DramaEngine, EngineOptions

        return {"DramaEngine": DramaEngine, "EngineOptions": EngineOptions}[name]
    if name in {"configure_model_runtime", "load_offloaded_drama3", "release_drama3"}:
        from .model import configure_model_runtime, load_offloaded_drama3, release_drama3

        return {
            "configure_model_runtime": configure_model_runtime,
            "load_offloaded_drama3": load_offloaded_drama3,
            "release_drama3": release_drama3,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
