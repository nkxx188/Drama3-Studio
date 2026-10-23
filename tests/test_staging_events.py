"""CPU mocks check bounded CUDA fences without initializing a CUDA device."""

import unittest
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import torch

from drama_low_vram.offload import (
    EXPERT_BUFFER_NAMES,
    FP8_DTYPE,
    ExpertStore,
    OffloadStatistics,
    _StagingPool,
)


class FakeStream:
    def __init__(self, handle=1, device="cuda:0"):
        self.cuda_stream = handle
        self.device = torch.device(device)
        self.waits = []

    def wait_event(self, event):
        self.waits.append((event, len(event.records)))


class FakeEvent:
    def __init__(self, log, *, enable_timing=False):
        self.enable_timing = enable_timing
        self.records = []
        self.synchronizations = 0
        self.elapsed_reads = 0
        self.log = log

    def record(self, stream):
        self.records.append(stream)
        self.log.append(("record", self))

    def synchronize(self):
        self.synchronizations += 1
        self.log.append(("synchronize", self))

    def elapsed_time(self, end_event):
        if not self.enable_timing or not end_event.enable_timing:
            raise AssertionError("Transfer timing needs two timing events")
        self.elapsed_reads += 1
        return 2.5


def expert_store():
    values = torch.arange(4, dtype=torch.float32)[:, None, None] + 1
    return ExpertStore(
        w13_weight=values.expand(4, 256, 128).to(FP8_DTYPE).contiguous(),
        w13_scale_inv=values.expand(4, 2, 1).contiguous(),
        w2_weight=values.expand(4, 128, 128).to(FP8_DTYPE).contiguous(),
        w2_scale_inv=values.expand(4, 1, 1).contiguous(),
    )


@contextmanager
def fake_cuda(pool):
    state = SimpleNamespace(
        stream=FakeStream(), events=[], recorded_buffers=[], log=[]
    )
    allocate = pool._allocate
    index_select = torch.index_select

    def allocate_cpu_buffers(store, device):
        if pool._target is None:
            allocate(store, torch.device("cpu"))
            pool._target = device

    def new_event(**kwargs):
        event = FakeEvent(state.log, **kwargs)
        state.events.append(event)
        return event

    def record_stream(tensor, stream):
        state.recorded_buffers.append((tensor, stream))

    def gather(*args, **kwargs):
        state.log.append(("gather", None))
        return index_select(*args, **kwargs)

    with (
        patch.object(pool, "_allocate", side_effect=allocate_cpu_buffers),
        patch("torch.cuda.current_stream", side_effect=lambda _: state.stream),
        patch("torch.cuda.device", side_effect=lambda _: nullcontext()),
        patch("torch.cuda.Event", side_effect=new_event),
        patch.object(torch.Tensor, "record_stream", record_stream),
        patch.object(torch.Tensor, "is_pinned", return_value=True),
        patch("torch.index_select", side_effect=gather),
    ):
        yield state


class StagingEventTests(unittest.TestCase):
    def pool(self, *, packed=False, profile=False):
        return _StagingPool(
            2,
            pin_memory=True,
            profile_transfers=profile,
            stats=OffloadStatistics(),
            packed_staging=packed,
        )

    def test_same_stream_sessions_reuse_compute_event_without_waiting(self):
        pool = self.pool()
        with fake_cuda(pool) as state:
            for _ in range(100):
                with pool.session(expert_store(), torch.device("cuda:0")):
                    pass
            self.assertEqual(len(state.events), 1)
            self.assertEqual(len(state.events[0].records), 100)
            self.assertEqual(state.stream.waits, [])
            self.assertEqual(len(state.recorded_buffers), 400)

    def test_distinct_wrappers_of_same_stream_do_not_wait(self):
        pool = self.pool()
        with fake_cuda(pool) as state:
            with pool.session(expert_store(), torch.device("cuda:0")):
                pass
            previous = state.stream
            state.stream = FakeStream(previous.cuda_stream, str(previous.device))
            with pool.session(expert_store(), torch.device("cuda:0")):
                pass
            self.assertEqual(state.stream.waits, [])
            self.assertEqual(len(state.events), 1)

    def test_alternating_streams_wait_on_each_previous_record(self):
        pool = self.pool()
        streams = [FakeStream(11), FakeStream(22)]
        with fake_cuda(pool) as state:
            for index in range(4):
                state.stream = streams[index % 2]
                with pool.session(expert_store(), torch.device("cuda:0")):
                    pass
            event = pool._compute_done
            self.assertEqual(len(state.events), 1)
            self.assertEqual(streams[1].waits, [(event, 1), (event, 3)])
            self.assertEqual(streams[0].waits, [(event, 2)])

    def test_failed_session_still_records_completion(self):
        pool = self.pool()
        with fake_cuda(pool) as state:
            with self.assertRaisesRegex(RuntimeError, "backend failure"):
                with pool.session(expert_store(), torch.device("cuda:0")):
                    raise RuntimeError("backend failure")
            self.assertEqual(len(state.events), 1)
            self.assertEqual(state.events[0].records, [state.stream])
            pool.synchronize()
            self.assertEqual(state.events[0].synchronizations, 1)

    def test_packed_arena_records_storage_once_per_session(self):
        pool = self.pool(packed=True)
        with fake_cuda(pool) as state:
            with pool.session(expert_store(), torch.device("cuda:0")):
                arena = pool._device_arena
            self.assertEqual(len(state.recorded_buffers), 1)
            self.assertIs(state.recorded_buffers[0][0], arena)
            self.assertIs(state.recorded_buffers[0][1], state.stream)

    def test_copy_events_reuse_after_wait_and_gather_stays_after_dma(self):
        store = expert_store()
        for packed in (False, True):
            for profile in (False, True):
                with self.subTest(packed=packed, profile=profile):
                    pool = self.pool(packed=packed, profile=profile)
                    with fake_cuda(pool) as state:
                        with pool.session(store, torch.device("cuda:0")):
                            for offset in range(3):
                                ids = torch.tensor([offset, offset + 1])
                                staged = pool.stage(store, ids)
                                for name in EXPERT_BUFFER_NAMES:
                                    expected = getattr(store, name).view(torch.uint8)
                                    actual = getattr(staged, name).view(torch.uint8)
                                    self.assertTrue(
                                        torch.equal(
                                            actual, expected.index_select(0, ids)
                                        )
                                    )
                            done = pool._copy_done_event
                            start = pool._copy_start_event
                            self.assertIs(pool._copy_done, done)
                            self.assertEqual(done.synchronizations, 2)
                            self.assertEqual(len(done.records), 3)
                            if profile:
                                self.assertIs(pool._copy_start, start)
                                self.assertEqual(start.elapsed_reads, 2)
                            else:
                                self.assertIsNone(start)
                            pool._finish_copy()
                            self.assertIsNone(pool._copy_done)
                            self.assertIsNone(pool._copy_start)
                            self.assertIs(pool._copy_done_event, done)
                            self.assertEqual(done.synchronizations, 3)
                        self.assertEqual(len(state.events), 3 if profile else 2)
                        if profile:
                            self.assertEqual(start.elapsed_reads, 3)
                            self.assertAlmostEqual(pool.stats.h2d_seconds, 0.0075)
                        # Each later host gather follows the prior DMA fence,
                        # including the pinned same-stream path.
                        gathers = [
                            index
                            for index, (kind, _) in enumerate(state.log)
                            if kind == "gather"
                        ]
                        for batch in (1, 2):
                            previous = gathers[batch * 4 - 1]
                            current = gathers[batch * 4]
                            self.assertIn(
                                ("synchronize", done), state.log[previous:current]
                            )

    def test_close_synchronizes_and_releases_device_bound_events(self):
        pool = self.pool(profile=True, packed=True)
        store = expert_store()
        with fake_cuda(pool) as state:
            with pool.session(store, torch.device("cuda:0")):
                pool.stage(store, torch.tensor([0, 1]))
            previous = list(state.events)
            pool.close()
            self.assertEqual(previous[1].synchronizations, 1)
            self.assertEqual(previous[2].synchronizations, 1)
            self.assertIsNone(pool._copy_start_event)
            self.assertIsNone(pool._copy_done_event)
            self.assertIsNone(pool._compute_done)
            self.assertIsNone(pool._compute_stream_key)
            self.assertIsNone(pool._copy_start)
            self.assertIsNone(pool._copy_done)
            state.stream = FakeStream(5, "cuda:1")
            with pool.session(store, torch.device("cuda:1")):
                pool.stage(store, torch.tensor([0, 1]))
            self.assertEqual(len(state.events), 6)
            self.assertIsNot(pool._compute_done, previous[2])


if __name__ == "__main__":
    unittest.main()
