from drama_low_vram.gguf_runtime import allocation_failure


def test_msvc_bad_allocation_is_recognized():
    assert allocation_failure("Native worker exited: std::bad allocation")


def test_ggml_assert_is_recognized():
    assert allocation_failure(
        "ggml.c:1685: GGML_ASSERT(ctx->mem_buffer != NULL) failed"
    )


def test_unrelated_load_error_is_not_recognized():
    assert not allocation_failure("Unsupported worker handshake")
