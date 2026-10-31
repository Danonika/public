# Soup T4 integration handoff — original Triton participation (2026-10-31)

Status: **public GPU development**, not jury admission. Keep all code/source baselines immutable.

## Hard finding

The 58.784 s / 9,325 MiB / eval 2.04966 full decoded-NF4-cache candidate is **not an eligible final submission unchanged**. Its separate exact-source original-`@triton.jit` timing audit (job `n_6b1700aedcaa42e3b2706a0f631dc9d3`) finished 64-step training but failed the 5% gate: **702.596 ms / 35.718 s = 1.967%** on steps 3+, including 390.524 ms RMSNorm backward, 303.477 ms RMSNorm forward, 8.595 ms frozen embedding and 0.000 ms NF4 decoding. The 3-GiB weight cache bypasses NF4 decoding once warmed up.

The Qwen2.5-1.5B-Instruct configuration has 28 layers / H=1536 / intermediate=8960 / Q heads12 / KV heads2. All seven transformer projection weights decode to about **2.4404 GiB FP16**, so a 3-GiB decoded-weight cache can hold the whole transformer. Estimated uncovered *weight capacity*, not measured kernel share: 1-GiB cache leaves ~59%, 512-MiB leaves ~80%. Restore **genuine training-consumed** Triton decoding: no dummy kernels, no skipped gradients, no changed batch/RNG.

Sources:
- [Official pinned jury checker](https://github.com/MakazhanAlpamys/soup-kernel-challenge/blob/c7c2fcd15972ce8b898d8d3ff732ee192c817ceb/harness/check_rules.py)
- [Official Qwen2.5-1.5B-Instruct configuration](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/blob/main/config.json)
- [Original NF4 source used by all candidates](https://github.com/Danonika/public/blob/master/soup-kernel-challenge/queue_20261031_native_dab4c473.ipynb)

## Measured public Pareto variants

| Candidate ID / row | 64-step command | Peak VRAM | Eval loss | Eligibility evidence |
|---|---:|---:|---:|---|
| Saved matching baseline | 71.85 s | 14,345 MiB | 2.04968 | Baseline, not solution |
| `chatgpt-fast5950-qproj-oproj-20261031-v1` / 206 | **58.784 s** | 9,325 MiB | 2.04966 | Same-source repeat useful-original share **FAIL 1.967%** |
| `chatgpt-mlp-fused-residual-20261031-v3` / 22 | **64.03 s** | **5,405 MiB** | **2.04960** | Not yet checked for own share + real 79 on this exact source |
| `chatgpt-mlp-padding-fused-scatter-20261031-v2` / 21 | 64.83 s | 5,345 MiB | 2.04965 | Uncached, own share + real 79 pending |
| `chatgpt-lowmem-triton-scatter-chunk128-20261031-v1` / 239 | 66.33 s | **5,005 MiB** | 2.04967 | Uncached, own share + real 79 pending |

A separate full-cache + valid-row-norm + Triton residual source passed **79/79 real pinned Soup regressions** using a test-only offline Pytest bundle after its 64-step training: `chatgpt-79-regressions-offline-pytest-20261031-v1` / row 240 (61.38s / 8,111MiB / loss2.04965). That does NOT transfer the 79/79 result to every derived candidate or prove the jury 5% gate.

## Submitted exact-source eligibility checks

Google [To Run queue](https://docs.google.com/spreadsheets/d/16tkklNfLrhT7lG__8n_awxFULmBd5GEk2vF6JIZs9Nw/edit#gid=163034553). Each row uses a new unique Request ID, Ready TRUE last, URL only, and preserved original harness result-export cells. The postrun diagnostics use pinned `check_rules.py` source attribution, exclude example SwiGLU and torch.compile output, calculate original JIT CUDA event timing / steps3+ timed loop, then run actual 79 pinned Soup tests using a self-contained test-only framework. Neither diagnostic changes the scored training command or `solution/`.

| Row | Request ID | Primary distinction |
|---|---|---|
| 420 | `chatgpt-fast5878-79-ownshare-20261031-v1` | Unchanged full cache, independent share replay |
| 460 | `chatgpt-lowmem5345-79-ownshare-20261031-v1` | Uncached original Triton MLP scatter |
| 500 | `chatgpt-fast5878-cache1g-79-ownshare-20261031-v1` | 1-GiB frozen decoded-weight cache |
| 540 | `chatgpt-fast5878-cache512m-79-ownshare-20261031-v1` | 512-MiB frozen decoded-weight cache |
| 580 | `chatgpt-fusedresid6403-79-ownshare-20261031-v1` | Uncached 64.03-s original Triton fused residual |
| 620 | `chatgpt-lowmem5005-79-ownshare-20261031-v1` | Uncached chunk128 low-memory original scatter |

**At latest read, all six remained `Ready — awaiting intake`; no receipts or new GPU results.** VM Queue `sheet_status` last successful poll remained **2026-10-31T10:27:00Z**. This is a Sheets/VM intake blocker, not proof of GPU test failure. Do not overwrite accepted requests, reuse IDs, or stop the allocated GPU VM. Authorized VM operator may inspect `soup-sheets-queue` systemd unit / journal. The existing hourly queue watch tracks all six through today's deadline.

## Correct validation order

1. Restore queue intake **without** stopping GPU allocation.
2. Measure useful original `@triton.jit` time against **actual step3+** denominator and require >=5%, with all required kernels actually producing consumed outputs.
3. Require **79/79** real unchanged Soup tests on the exact same source; reject any actual failures (do not mask assertions).
4. Require full baseline-paired first/mean losses, output/entropy/accuracy/rewind/adapter/seed/settings parity and hidden evaluation. Public loss + 79 + 5% are necessary but insufficient for jury admission.
5. Promote an exact source hash to the team repository only after those gates; prefer the nondominated lower-memory/speed variants to a faster but disqualified cache.

Don't merge unrelated H200/Ada/MoE or learned Engram/n-gram changes into the fixed T4 Qwen LoRA training task.
