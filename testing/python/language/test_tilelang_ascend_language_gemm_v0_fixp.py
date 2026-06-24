import pytest
import tilelang
import tilelang.language as T
import torch

"""
gemm_v0_fixp: per-N-tile fixpipe GEMM correctness suite (target = ascendc).

Feature under test
------------------
``gemm_v0_fixp`` (src/tl_templates/ascend/common.h) is a GEMM that, unlike
``gemm_v0`` -- which keeps the whole ``[M, N]`` result resident in the L0C
accumulator until the caller copies it out -- tiles N and fixpipes each
``[M, nTile]`` tile straight to the GM destination as soon as that tile's K
accumulation finishes. The L0C accumulator is therefore only a single
``[M, nTile]`` slot, reused per tile.

Why it matters
--------------
For a large output the full ``[M, N]`` result does not fit L0C: e.g. a
``[64, 512]`` float32 result is ``64 * 512 * 4 = 128KB`` -- the entire L0C. With
``gemm_v0`` a kernel that needs several such results live at once overflows
L0C; ``gemm_v0_fixp`` caps the L0C footprint at one ``[M, nTile]`` tile, so the
same large-output gemm runs. ``test_gemm_fixp_large_n`` exercises exactly this
N = 512 case.

Runtime k_actual
----------------
``gemm_v0_fixp`` takes a runtime ``k_actual`` (<= K): only the first
``k_actual`` rows of the K dim are loaded and contracted, so a caller can
contract over a valid length shorter than the allocated K tile.
``test_gemm_fixp_k_actual`` checks that a ``k_actual < K`` contracts exactly the
first ``k_actual`` rows (result == A[:, :k_actual] @ B[:k_actual, :]).
"""

TARGET = "ascendc"

CUBE_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


@pytest.fixture(scope="session", autouse=True)
def clear_cache():
    """Clear tilelang cache before the session."""
    tilelang.cache.clear_cache()
    yield


def _torch_dtype(dtype):
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[dtype]


def _n_tile(N, dtype):
    # nTile = the widest N-tile whose B sub-block (kL0Size x nTile) fits the 32KB
    # L0B ping-pong slot: 32KB / (kL0Size=128 * sizeof(dtype)). This mirrors the
    # constexpr nTile the gemm_v0_fixp template computes; the caller allocates the
    # single L0C slot [M, nTile] accordingly.
    elem = 2 if dtype in ("float16", "bfloat16") else 4
    return min(N, (32 * 1024) // (128 * elem))


def gemm_fixp_plain(M, N, K, dtype, accum_dtype, k_actual=None):
    """Single-K-tile (K <= 128) GEMM via gemm_v0_fixp, fixpiped straight to GM."""
    n_tile = _n_tile(N, dtype)

    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),  # type: ignore
        B: T.Tensor((K, N), dtype),  # type: ignore
        C: T.Tensor((M, N), accum_dtype),  # type: ignore
    ):
        with T.Kernel(1, is_npu=True) as (cid, _):
            A_L1 = T.alloc_L1((M, K), dtype)
            B_L1 = T.alloc_L1((K, N), dtype)
            C_L0 = T.alloc_L0C((M, n_tile), accum_dtype)  # single [M, nTile] slot
            with T.Scope("C"):
                T.copy(A[0, 0], A_L1)
                T.copy(B[0, 0], B_L1)
                T.gemm_v0_fixp(A_L1, B_L1, C_L0, C, k_actual=k_actual, init=True)

    return main


def run_gemm_fixp(M, N, K, dtype, accum_dtype, k_actual=None):
    torch.manual_seed(0)
    func = gemm_fixp_plain(M, N, K, dtype, accum_dtype, k_actual)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=CUBE_CONFIGS, target=TARGET)
    td = _torch_dtype(dtype)
    a = torch.randn(M, K, dtype=td).npu()
    b = torch.randn(K, N, dtype=td).npu()
    torch.npu.synchronize()
    c = func(a, b)
    kk = K if k_actual is None else k_actual
    ref = (a[:, :kk].float() @ b[:kk, :].float()).to(torch.float32)
    torch.testing.assert_close(c, ref, rtol=1e-2, atol=1e-2)


# Large N: the full [64, 512] float32 output is 128KB = the entire L0C, so
# gemm_v0 would occupy all of L0C; gemm_v0_fixp keeps it to one [64, nTile] tile
# and fixpipes each N-tile out, so the gemm still runs.
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_gemm_fixp_large_n(dtype):
    run_gemm_fixp(64, 512, 128, dtype, "float", k_actual=None)


# k_actual < K: only the first k_actual rows of K are loaded and contracted, so
# the result contracts exactly the first k_actual rows (not the full K tile).
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_gemm_fixp_k_actual(dtype):
    run_gemm_fixp(64, 512, 128, dtype, "float", k_actual=64)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
