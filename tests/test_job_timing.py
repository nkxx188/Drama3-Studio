from drama_low_vram.job_timing import JobClock


class Time:
    def __init__(self):
        self.now = 0

    def __call__(self):
        return self.now

    def add(self, seconds):
        self.now += seconds


def test_cold_loading_initialization_and_queue_do_not_count_as_generation():
    time = Time()
    clock = JobClock(queued_seconds=12, clock=time)
    time.add(2)
    clock.switch("generation")
    clock.begin("model_load")
    time.add(18)
    assert clock.snapshot()["generation_seconds"] == 0
    clock.end("model_load")
    time.add(60)
    clock.switch("initialization")
    time.add(5)
    clock.switch("generation")
    time.add(8)
    clock.finish()
    result = clock.snapshot()
    assert result["generation_seconds"] == 68
    assert result["model_load_seconds"] == 18
    assert result["initialization_seconds"] == 7
    assert result["queue_seconds"] == 12
    assert result["total_seconds"] == 105


def test_warm_request_does_not_inherit_a_previous_model_load():
    time = Time()
    first = JobClock(clock=time)
    first.begin("model_load")
    time.add(20)
    first.end("model_load")
    first.finish()
    second = JobClock(clock=time)
    second.switch("generation")
    time.add(7)
    assert second.snapshot()["generation_seconds"] == 7
    assert second.snapshot()["model_load_seconds"] == 0


def test_reloading_during_a_retry_pauses_and_then_resumes_generation():
    time = Time()
    clock = JobClock(clock=time)
    clock.switch("generation")
    time.add(4)
    clock.begin("model_load")
    time.add(6)
    clock.end("model_load")
    time.add(9)
    result = clock.snapshot()
    assert result["generation_seconds"] == 13
    assert result["model_load_seconds"] == 6


def test_live_text_wait_is_excluded_and_finished_clock_stays_frozen():
    time = Time()
    clock = JobClock(clock=time)
    clock.switch("generation")
    time.add(3)
    clock.switch("waiting")
    time.add(100)
    clock.switch("generation")
    time.add(4)
    clock.finish()
    time.add(1000)
    clock.switch("generation")
    result = clock.snapshot()
    assert result["generation_seconds"] == 7
    assert result["waiting_seconds"] == 100
    assert result["total_seconds"] == 107
    assert result["phase"] == "finished"


def test_nested_loading_and_polling_do_not_double_count():
    time = Time()
    clock = JobClock(clock=time)
    clock.switch("generation")
    clock.begin("initialization")
    time.add(2)
    clock.begin("model_load")
    time.add(6)
    assert clock.snapshot()["model_load_seconds"] == 6
    assert clock.snapshot()["model_load_seconds"] == 6
    clock.end("model_load")
    time.add(1)
    clock.end("initialization")
    time.add(4)
    result = clock.snapshot()
    assert result["generation_seconds"] == 4
    assert result["model_load_seconds"] == 6
    assert result["initialization_seconds"] == 3
