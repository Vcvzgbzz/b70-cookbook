# swa_fine_test.py: unit test of SlidingWindowManager._find_fine_grained_hit with a fake block pool
import os
os.environ["B70_SWA_FINE_GRAINED"] = "1"
from types import SimpleNamespace
from vllm.v1.core.single_type_kv_cache_manager import SlidingWindowManager
from vllm.v1.kv_cache_interface import SlidingWindowSpec
import torch

A, BS, W = 64, 896, 2048           # hash block, cache block, window
NULL = "NULL"


class Pool:
    def __init__(self, cached):
        self.cached = cached          # hash -> block name
        self.null_block = NULL
        self.hash_block_size = A

    def get_cached_block(self, h, group_ids):
        b = self.cached.get(h)
        return [b] * len(group_ids) if b is not None else None


spec = SlidingWindowSpec(block_size=BS, num_kv_heads=8, head_size=128, dtype=torch.bfloat16, sliding_window=W)
scale = BS // A


def run(n_tokens, full_cached, tail_at=None, drop=False, max_len=None):
    fine = [f"h{i}" for i in range(n_tokens // A)]            # chained prefix hashes at 64-token boundaries
    cached = {fine[(j + 1) * scale - 1]: f"B{j}" for j in full_cached}
    if tail_at is not None:
        cached[fine[tail_at // A - 1]] = f"T{tail_at}"
    blocks, hit = SlidingWindowManager._find_fine_grained_hit(
        fine, max_len or n_tokens, [0], Pool(cached), spec, drop, A)
    return blocks[0], hit


# 1) repeat of a 10_000-token prompt: full blocks 0..10 cached (11*896=9856), partial tail at 9984
b, hit = run(10_000, range(11), tail_at=9984)
assert hit == 9984, hit
first = (9984 - W + 1) // BS
assert b[:first] == [NULL] * first and b[-1] == "T9984" and len(b) == 12, b
# 2) same with EAGLE drop: one hash unit less, window rechecked
b, hit = run(10_000, range(11), tail_at=9984, drop=True)
assert hit == 9984 - A and b[-1] == "T9984", (hit, b)
# 3) hit inside a fully cached block (no tail registered): end at 9856+128 must use block 11 if cached
b, hit = run(12 * BS + 64, range(12), max_len=11 * BS + 128)
assert hit == 11 * BS + 128 and b[-1] == "B11", (hit, b)
# 4) a gap inside the window (block 9 missing) must not hit at the tail; falls back to an earlier end
b, hit = run(10_000, [0, 1, 2, 3, 4, 5, 6, 7, 8, 10], tail_at=9984)
assert hit <= 9 * BS, hit           # nothing at or past block 9 can satisfy the window
# 5) a gap far before the window is fine (null blocks there)
b, hit = run(10_000, [5, 6, 7, 8, 9, 10], tail_at=9984)
assert hit == 9984 and b[0] == NULL, (hit, b[:3])
# 6) nothing cached -> 0
b, hit = run(10_000, [])
assert hit == 0 and b == [], (hit, b)
# 7) max_length below the tail caps the hit
b, hit = run(10_000, range(11), tail_at=9984, max_len=5000)
assert hit <= 5000 and hit % A == 0, hit
# 8) tail registered LATER in the same block than the asked end (mamba checkpoint below the registered tail)
b, hit = run(10_000, range(11), tail_at=9984, max_len=9920)
assert hit == 9920 and b[-1] == "T9984", (hit, b[-2:])
# 9) a later boundary in the NEXT block must not be used for an end in this block
b, hit = run(12 * BS + 64, range(11), tail_at=11 * BS + 64 + 896, max_len=11 * BS + 64)
assert hit <= 11 * BS, (hit, b[-2:])
print("all SWA fine-grained finder checks passed")
