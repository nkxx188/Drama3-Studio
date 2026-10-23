"""Deterministic policy tests for the hardware autotuner."""

import json
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

from drama_low_vram.autotune import (
    GIB,
    MIB,
    HostAllocationState,
    HostMemoryRequirements,
    HardwareInfo,
    PROFILE_NAMES,
    _read_memory_status,
    autotune,
    inspect_and_autotune,
    inspect_hardware,
    plan_post_load,
    select_profile,
)


def snapshot(
    *,
    vram: float,
    vram_free: float | None = None,
    ram: float = 64,
    ram_available: float | None = None,
    pagefile: float = 32,
    pagefile_available: float | None = None,
    commit_limit: float | None = None,
    commit_available: float | None = None,
    cuda: bool = True,
    triton: bool = True,
    cores: int = 16,
) -> HardwareInfo:
    return HardwareInfo.from_gib(
        platform="Windows",
        cuda_available=cuda,
        cuda_device_index=0 if cuda else None,
        device_name="Test GPU" if cuda else None,
        compute_capability=(8, 9) if cuda else None,
        vram=vram,
        vram_free=vram if vram_free is None else vram_free,
        ram=ram,
        ram_available=ram if ram_available is None else ram_available,
        pagefile=pagefile,
        pagefile_available=pagefile
        if pagefile_available is None
        else pagefile_available,
        commit_limit=commit_limit,
        commit_available=commit_available,
        triton_available=triton,
        cpu_logical_cores=cores,
        cpu_physical_cores=max(1, cores // 2),
    )


class AutotuneTests(unittest.TestCase):
    def test_checkpoint_gpu_estimate_and_expert_count_replace_fp8_assumptions(self):
        requirements = replace(
            HostMemoryRequirements(), fixed_gpu_bytes=GIB,
            expert_bytes=GIB, num_experts_per_layer=4)
        small = select_profile(
            snapshot(vram=8, vram_free=3), host_requirements=requirements)
        self.assertTrue(small.can_run)
        self.assertEqual(small.fixed_gpu_bytes, GIB)
        large = select_profile(
            snapshot(vram=24), host_requirements=requirements)
        self.assertLessEqual(large.max_staged_experts, 4)
        self.assertLessEqual(large.cache_experts, 4)

    def test_ample_host_ram_restores_measured_prefill_and_reserves_codec_headroom(self):
        requirements = replace(
            HostMemoryRequirements(), cpu_weight_bytes=0, parking_bytes=32776,
            file_backed_cpu_weight_bytes=GIB, file_backed_parking_bytes=5 * GIB)
        # Startup estimates on either side of the former 1.5GiB gate request
        # the same bounded trial. The post-load plan uses actual allocations.
        for free in (7, 6.902, 6.5):
            roomy = select_profile(
                snapshot(vram=8, vram_free=free, ram=48, ram_available=39),
                expert_storage="mmap", host_requirements=requirements)
            self.assertEqual(roomy.prefill_chunk, 32)
            self.assertEqual(roomy.max_auto_cache_experts, 1)
            self.assertEqual(roomy.max_staged_experts, 8)
            actual = post_load_plan(
                vram_free_bytes=775 * MIB, prefill_chunk=roomy.prefill_chunk,
                max_staged_experts=roomy.max_staged_experts,
                max_pair_rows=roomy.max_pair_rows,
                max_auto_cache_experts=roomy.max_auto_cache_experts)
            self.assertTrue(actual.can_run)
            self.assertLess(actual.prefill_chunk, roomy.prefill_chunk)
        for hardware in (
            snapshot(vram=8, vram_free=7, ram=32, ram_available=23),
            snapshot(vram=8, vram_free=7, ram=48, ram_available=20),
        ):
            constrained = select_profile(
                hardware, expert_storage="mmap", host_requirements=requirements)
            self.assertEqual(constrained.prefill_chunk, 8)
            self.assertIsNone(constrained.max_auto_cache_experts)

    def test_file_backed_fp8_keeps_a_bounded_transfer_pool_on_32gb_ram(self):
        requirements = replace(
            HostMemoryRequirements(), cpu_weight_bytes=0, parking_bytes=32776,
            file_backed_cpu_weight_bytes=GIB, file_backed_parking_bytes=5 * GIB)
        config = select_profile(
            snapshot(vram=8, vram_free=7, ram=32, ram_available=23),
            expert_storage="mmap", host_requirements=requirements)
        self.assertTrue(config.can_run)
        self.assertEqual(config.max_staged_experts, 8)
        self.assertEqual(config.max_pair_rows, 64)
        self.assertEqual(config.prefill_chunk, 8)
        self.assertTrue(config.pin_memory)
        constrained = select_profile(
            snapshot(vram=8, vram_free=6, ram=32, ram_available=10),
            expert_storage="mmap", host_requirements=requirements)
        self.assertLessEqual(constrained.max_staged_experts, 2)
        self.assertFalse(constrained.pin_memory)

    def test_profile_names_are_stable(self):
        self.assertEqual(PROFILE_NAMES, ("8gb", "12gb", "16gb", "24gb", "32gb+"))

    def test_eight_gib_is_runnable_and_conservative(self):
        config = select_profile(snapshot(vram=8, ram=48, ram_available=32))
        self.assertEqual(config.profile, "8gb")
        self.assertTrue(config.can_run)
        self.assertEqual(config.status, "ready")
        self.assertEqual(config.device, "cuda:0")
        self.assertEqual(config.lm_head_device, "cpu")
        self.assertEqual(config.cache_experts, 0)
        self.assertLessEqual(config.max_staged_experts, 4)
        self.assertLessEqual(config.prefill_chunk, 8)
        self.assertTrue(config.pin_memory)
        self.assertEqual(config.backend, "triton")
        self.assertTrue(
            any("8 GB mode remains runnable" in item for item in config.explanations)
        )

    def test_busy_eight_gib_gpu_reports_resources_without_rejecting_hardware(self):
        config = select_profile(
            snapshot(vram=8, vram_free=0.55, ram=48, ram_available=32)
        )
        self.assertFalse(config.can_run)
        self.assertEqual(config.profile, "8gb")
        self.assertEqual(config.status, "resource_busy")
        self.assertEqual(config.lm_head_device, "cpu")
        self.assertEqual(config.cache_experts, 0)
        self.assertLessEqual(config.max_staged_experts, 2)
        self.assertLessEqual(config.max_pair_rows, 16)
        self.assertLessEqual(config.prefill_chunk, 4)
        self.assertLessEqual(config.codec_max_frames, 1024)
        self.assertTrue(config.warnings)

    def test_zero_free_vram_is_not_replaced_with_total_capacity(self):
        config = select_profile(snapshot(vram=16, vram_free=0))
        self.assertFalse(config.can_run)
        self.assertEqual(config.status, "resource_busy")
        self.assertEqual(config.safe_vram_budget_bytes, 0)
        self.assertEqual(config.gpu_headroom_bytes, 0)

    def test_twelve_gib_enables_gpu_head_when_headroom_exists(self):
        config = select_profile(snapshot(vram=12, vram_free=12, ram=64, ram_available=48))
        self.assertEqual(config.profile, "12gb")
        self.assertEqual(config.lm_head_device, "cuda")
        self.assertGreaterEqual(config.max_staged_experts, 4)
        self.assertGreater(config.prefill_chunk, 8)
        self.assertGreater(config.codec_max_frames, 2048)

    def test_busy_twelve_gib_gpu_falls_back_to_cpu_head(self):
        config = select_profile(snapshot(vram=12, vram_free=6, ram=64, ram_available=48))
        self.assertEqual(config.profile, "12gb")
        self.assertEqual(config.lm_head_device, "cpu")
        self.assertEqual(config.cache_experts, 0)
        self.assertLessEqual(config.max_staged_experts, 4)

    def test_ram_pressure_disables_pinning_and_insufficient_commit_blocks_loading(self):
        config = select_profile(
            snapshot(vram=16, ram=24, ram_available=8, pagefile=8, pagefile_available=1)
        )
        self.assertFalse(config.can_run)
        self.assertEqual(config.status, "resource_busy")
        self.assertFalse(config.pin_memory)
        self.assertEqual(config.expert_storage, "ram")
        self.assertEqual(config.cache_experts, 0)
        self.assertLessEqual(config.max_staged_experts, 2)
        self.assertTrue(any("RAM headroom is tight" in item for item in config.warnings))

    def test_moderate_ram_limits_cache_but_keeps_cpu_experts(self):
        config = select_profile(
            snapshot(vram=24, ram=40, ram_available=16, pagefile=32, pagefile_available=32)
        )
        self.assertTrue(config.can_run)
        self.assertEqual(config.expert_storage, "ram")
        self.assertLessEqual(config.cache_experts, 2)
        self.assertTrue(config.pin_memory)

    def test_sufficient_real_commit_allows_paging_with_honest_ram_storage(self):
        config = select_profile(
            snapshot(
                vram=8,
                ram=48,
                ram_available=35,
                pagefile=32,
                pagefile_available=0,
                commit_limit=96,
                commit_available=69,
            )
        )
        self.assertTrue(config.can_run)
        self.assertEqual(config.status, "ready")
        self.assertEqual(config.expert_storage, "ram")
        self.assertEqual(config.available_commit_bytes, 69 * GIB)
        self.assertFalse(config.commit_budget_estimated)
        self.assertTrue(any("paging" in item for item in config.warnings))
        self.assertTrue(any("process RAM" in item for item in config.explanations))

    def test_actual_commit_controls_budget_even_if_legacy_pagefile_estimate_is_large(self):
        config = select_profile(
            snapshot(
                vram=8,
                ram=48,
                ram_available=35,
                pagefile_available=64,
                commit_available=8,
            )
        )
        self.assertFalse(config.can_run)
        self.assertEqual(config.status, "resource_busy")
        self.assertEqual(config.available_commit_bytes, 8 * GIB)
        self.assertFalse(config.commit_budget_estimated)

    def test_zero_actual_commit_is_not_replaced_with_estimated_capacity(self):
        config = select_profile(
            snapshot(vram=8, ram=64, pagefile_available=64, commit_available=0)
        )
        self.assertFalse(config.can_run)
        self.assertEqual(config.status, "resource_busy")
        self.assertEqual(config.available_commit_bytes, 0)
        self.assertFalse(config.commit_budget_estimated)

    def test_small_ram_and_pagefile_are_resource_busy_without_mmap_promise(self):
        config = select_profile(
            snapshot(vram=8, ram=32, ram_available=8, pagefile=4, pagefile_available=4)
        )
        self.assertFalse(config.can_run)
        self.assertEqual(config.status, "resource_busy")
        self.assertEqual(config.expert_storage, "ram")
        self.assertEqual(config.available_commit_bytes, 12 * GIB)
        self.assertTrue(config.commit_budget_estimated)
        self.assertGreater(config.required_host_commit_bytes, 30 * GIB)
        self.assertTrue(any("very slow" in item for item in config.warnings))

    def test_triton_fallback_is_torch_and_explained(self):
        config = select_profile(snapshot(vram=8, triton=False))
        self.assertEqual(config.backend, "torch")
        self.assertTrue(any("Triton FP8 backend" in item for item in config.warnings))

    def test_no_cuda_is_inspection_only_not_a_false_ready_config(self):
        config = select_profile(snapshot(vram=0, cuda=False, triton=False, ram=32))
        self.assertFalse(config.can_run)
        self.assertEqual(config.status, "blocked")
        self.assertEqual(config.device, "cpu")
        self.assertEqual(config.backend, "torch")
        self.assertTrue(any("CUDA GPU is required" in item for item in config.warnings))

    def test_cpu_count_reduces_batching(self):
        config = select_profile(snapshot(vram=32, cores=2))
        self.assertLessEqual(config.prefill_chunk, 8)
        self.assertLessEqual(config.max_pair_rows, 32)

    def test_json_is_serializable_and_contains_hardware(self):
        result = inspect_and_autotune(snapshot(vram=8, ram=48, ram_available=32))
        encoded = result.to_json()
        payload = json.loads(encoded)
        self.assertEqual(payload["config"]["profile"], "8gb")
        self.assertEqual(payload["hardware"]["vram_total_gib"], 8.0)
        self.assertIn("explanations", payload["config"])
        self.assertIn("can_run", payload["config"])
        self.assertIn("commit_available_bytes", payload["hardware"])
        self.assertIn("required_host_commit_bytes", payload["config"])
        self.assertEqual(
            payload["config"]["host_plan"]["required_commit_bytes"],
            payload["config"]["required_host_commit_bytes"],
        )

    def test_auto_uses_shared_host_plan_for_low_ram_mmap(self):
        config = select_profile(
            snapshot(vram=8, ram=16, ram_available=10, commit_available=12),
            expert_storage="auto",
        )
        self.assertEqual(config.expert_storage, "mmap")
        self.assertTrue(config.can_run)
        self.assertTrue(config.host_plan.can_run)
        self.assertLess(config.required_host_commit_bytes, 12 * GIB)
        self.assertFalse(any("remain in process RAM" in item for item in config.warnings))
        self.assertEqual(
            config.safe_ram_budget_bytes,
            config.hardware.ram_available_bytes - config.host_plan.buffer_bytes,
        )

    def test_supplied_requirements_and_existing_codec_reach_all_public_wrappers(self):
        requirements = HostMemoryRequirements(
            expert_bytes=30 * GIB,
            cpu_weight_bytes=GIB,
            parking_bytes=GIB,
            codec_resident_bytes=GIB,
            codec_build_bytes=2 * GIB,
        )
        allocated = HostAllocationState(codec_bytes=GIB)
        hardware = snapshot(vram=8, ram=48, ram_available=34, commit_available=34)
        for factory in (select_profile, autotune):
            with self.subTest(factory=factory.__name__):
                result = factory(
                    hardware, expert_storage="auto",
                    host_requirements=requirements, host_allocated=allocated,
                )
                self.assertEqual(result.expert_storage, "ram")
                self.assertEqual(result.host_plan.requirements, requirements)
                self.assertEqual(result.host_plan.already_allocated_bytes, GIB)
        wrapped = inspect_and_autotune(
            hardware, expert_storage="auto", host_requirements=requirements,
            host_allocated=allocated,
        )
        self.assertEqual(wrapped.config.expert_storage, "ram")
        self.assertEqual(wrapped.config.host_plan.already_allocated_bytes, GIB)

    def test_alias_autotune_accepts_snapshot(self):
        config = autotune(snapshot(vram=16, ram=64, ram_available=48))
        self.assertEqual(config.profile, "16gb")
        self.assertEqual(config.hardware.vram_total_gib, 16.0)

    def test_inspection_preserves_zero_free_vram_and_requested_device(self):
        properties = SimpleNamespace(total_memory=8 * GIB, name="GPU 1", major=8, minor=9)
        cuda = SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: 2,
            get_device_properties=lambda index: properties if index == 1 else None,
            mem_get_info=lambda index: (0, 8 * GIB) if index == 1 else None,
        )
        fake_torch = SimpleNamespace(
            cuda=cuda, __version__="test", version=SimpleNamespace(cuda="test")
        )
        with (
            patch("drama_low_vram.autotune.torch", fake_torch),
            patch(
                "drama_low_vram.autotune._read_memory_status",
                return_value=(48 * GIB, 35 * GIB, 48 * GIB, 34 * GIB, (), 96 * GIB, 69 * GIB),
            ),
            patch("drama_low_vram.autotune._cpu_core_counts", return_value=(16, 8, ())),
            patch("drama_low_vram.autotune._triton_is_installed", return_value=True),
        ):
            hardware = inspect_hardware("cuda:1")
            self.assertEqual(hardware.cuda_device_index, 1)
            self.assertEqual(hardware.vram_free_bytes, 0)
            self.assertEqual(hardware.vram_total_bytes, 8 * GIB)
            self.assertEqual(hardware.commit_available_bytes, 69 * GIB)
            self.assertEqual(hardware.commit_limit_bytes, 96 * GIB)
            with self.assertRaisesRegex(ValueError, "cuda:2 is unavailable"):
                inspect_hardware("cuda:2")

    def test_windows_commit_reader_preserves_raw_os_values(self):
        def fill_memory_status(pointer):
            status = pointer._obj
            status.ullTotalPhys = 48 * GIB
            status.ullAvailPhys = 35 * GIB
            status.ullTotalPageFile = 96 * GIB
            status.ullAvailPageFile = 69 * GIB
            return 1

        win_api = SimpleNamespace(
            kernel32=SimpleNamespace(GlobalMemoryStatusEx=fill_memory_status)
        )
        with (
            patch("drama_low_vram.autotune.os.name", "nt"),
            patch("drama_low_vram.autotune.ctypes.windll", win_api, create=True),
        ):
            physical, available, _, _, notes, commit_limit, commit_available = _read_memory_status()
        self.assertEqual(physical, 48 * GIB)
        self.assertEqual(available, 35 * GIB)
        self.assertEqual(commit_limit, 96 * GIB)
        self.assertEqual(commit_available, 69 * GIB)
        self.assertEqual(notes, ())


def post_load_plan(**overrides):
    values = {
        "fixed_gpu_bytes": 5 * GIB,
        "vram_free_bytes": int(1.5 * GIB),
        "vram_total_bytes": 8 * GIB,
        "expert_total_bytes": 32_220_119_040,
        "num_layers": 40,
        "num_experts_per_layer": 256,
        "max_staged_experts": 4,
        "max_pair_rows": 32,
        "prefill_chunk": 8,
    }
    values.update(overrides)
    return plan_post_load(**values)


class PostLoadPlanTests(unittest.TestCase):
    def test_automatic_cache_cap_keeps_budget_fallback_and_manual_overrides(self):
        automatic = post_load_plan(max_auto_cache_experts=1)
        self.assertEqual(automatic.cache_experts, 1)
        self.assertIsNone(automatic.requested_cache_experts)
        self.assertEqual(
            post_load_plan(max_auto_cache_experts=1, cache_experts=3).cache_experts, 3)
        self.assertEqual(
            post_load_plan(max_auto_cache_experts=1, vram_free_bytes=800 * MIB).cache_experts, 0)
        for bad in (-1, True, 257):
            with self.subTest(cap=bad), self.assertRaises(ValueError):
                post_load_plan(max_auto_cache_experts=bad)

    def test_real_fp8_budget_enables_cache_on_eight_gib_gpu(self):
        plan = post_load_plan()
        self.assertTrue(plan.can_run)
        self.assertEqual(plan.status, "ready")
        self.assertEqual(plan.staged_expert_bytes, 3_146_496)
        self.assertEqual(plan.cache_slot_bytes, 125_859_840)
        self.assertEqual(plan.cache_experts, 6)
        self.assertEqual(plan.cache_bytes, 6 * plan.cache_slot_bytes)
        self.assertGreaterEqual(plan.gpu_headroom_bytes, plan.reserve_bytes)
        self.assertLessEqual(
            plan.planned_extra_gpu_bytes + plan.reserve_bytes,
            plan.available_gpu_bytes,
        )

    def test_staging_is_one_shared_pool_not_forty_layer_pools(self):
        plan = post_load_plan(cache_experts=0, max_staged_experts=8)
        self.assertEqual(plan.staging_bytes, plan.staged_expert_bytes * 8)
        self.assertEqual(plan.cache_bytes, 0)
        self.assertEqual(plan.cache_experts, 0)
        self.assertTrue(plan.can_run)

    def test_post_load_respects_current_free_memory_and_reserved_pool(self):
        plan = post_load_plan(
            vram_free_bytes=GIB,
            vram_allocated_bytes=5 * GIB,
            vram_reserved_bytes=5 * GIB + 512 * MIB,
        )
        self.assertEqual(plan.reusable_reserved_bytes, 512 * MIB)
        self.assertEqual(plan.available_gpu_bytes, int(1.5 * GIB))
        self.assertEqual(plan.cache_experts, 6)
        self.assertTrue(plan.can_run)

    def test_reserved_accounting_cannot_create_more_than_total_vram(self):
        plan = post_load_plan(
            fixed_gpu_bytes=7 * GIB,
            vram_free_bytes=2 * GIB,
            vram_allocated_bytes=7 * GIB,
            vram_reserved_bytes=20 * GIB,
        )
        self.assertEqual(plan.available_gpu_bytes, GIB)
        self.assertLessEqual(
            plan.planned_extra_gpu_bytes + plan.reserve_bytes, GIB
        )

    def test_no_post_load_headroom_is_resource_busy(self):
        plan = post_load_plan(vram_free_bytes=0)
        self.assertFalse(plan.can_run)
        self.assertEqual(plan.status, "resource_busy")
        self.assertEqual(plan.available_gpu_bytes, 0)
        self.assertEqual(plan.cache_experts, 0)
        self.assertTrue(any("short by" in item for item in plan.warnings))

    def test_temporary_limits_shrink_when_that_makes_generation_fit(self):
        plan = post_load_plan(
            vram_free_bytes=775 * MIB,
            max_staged_experts=8,
            max_pair_rows=64,
            prefill_chunk=8,
        )
        self.assertTrue(plan.can_run)
        self.assertEqual(plan.max_staged_experts, 2)
        self.assertEqual(plan.max_pair_rows, 16)
        self.assertEqual(plan.prefill_chunk, 2)
        self.assertEqual(plan.cache_experts, 0)

    def test_explicit_batch_limits_remain_intact_when_requested(self):
        plan = post_load_plan(
            vram_free_bytes=775 * MIB,
            max_staged_experts=8,
            allow_staging_reduction=False,
        )
        self.assertFalse(plan.can_run)
        self.assertEqual(plan.status, "resource_busy")
        self.assertEqual(plan.max_staged_experts, 8)
        self.assertEqual(plan.max_pair_rows, 32)
        self.assertEqual(plan.prefill_chunk, 8)

    def test_manual_cache_capacity_is_preserved_when_it_does_not_fit(self):
        plan = post_load_plan(cache_experts=16)
        self.assertEqual(plan.requested_cache_experts, 16)
        self.assertEqual(plan.cache_experts, 16)
        self.assertEqual(plan.cache_bytes, 16 * plan.cache_slot_bytes)
        self.assertFalse(plan.can_run)
        self.assertEqual(plan.status, "resource_busy")

    def test_manual_cache_with_room_remains_exact(self):
        plan = post_load_plan(cache_experts=4)
        self.assertEqual(plan.cache_experts, 4)
        self.assertEqual(plan.requested_cache_experts, 4)
        self.assertTrue(plan.can_run)

    def test_expert_scales_are_included_and_rounding_never_underestimates(self):
        plan = post_load_plan(expert_total_bytes=32_220_119_041)
        self.assertEqual(plan.staged_expert_bytes, 3_146_497)
        self.assertEqual(plan.cache_slot_bytes, 125_859_880)
        self.assertGreaterEqual(
            plan.cache_slot_bytes * plan.num_experts_per_layer,
            plan.expert_total_bytes,
        )

    def test_plan_serialization_exposes_actual_budget_and_status(self):
        plan = post_load_plan()
        payload = json.loads(plan.to_json())
        self.assertEqual(payload["cache_experts"], 6)
        self.assertTrue(payload["can_run"])
        self.assertEqual(payload["fixed_gpu_bytes"], 5 * GIB)
        self.assertIsInstance(payload["explanations"], list)

    def test_invalid_checkpoint_or_manual_capacity_fails_explicitly(self):
        for options in (
            {"expert_total_bytes": 0},
            {"num_layers": 0},
            {"num_experts_per_layer": 0},
            {"cache_experts": -1},
            {"cache_experts": 257},
            {"reserve_bytes": -1},
            {"runtime_workspace_bytes": -1},
        ):
            with self.subTest(options=options):
                with self.assertRaises(ValueError):
                    post_load_plan(**options)


if __name__ == "__main__":
    unittest.main()
