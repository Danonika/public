# external-contributor-as-parent Soup/T4 fusion research (31 October 2026)

**This is an attributed research benchmark, NOT a competition submission claiming original kernels.**

The reference is the independently published Apache-2.0 solution from [external-contributor/hackathon-decentrathon-6](https://github.com/external-contributor/hackathon-decentrathon-6), pinned to commit [`ce62d67432ad8e530abff12bb36d3ab15668502a`](https://github.com/external-contributor/hackathon-decentrathon-6/commit/ce62d67432ad8e530abff12bb36d3ab15668502a). The source bytes are archived as `part1.json`, `part2.json`, `part3.json` (22 runtime modules, Git blob SHA validation at installation). [Upstream license](LICENSE-external-contributor.txt) and [upstream notice](NOTICE-external-contributor.txt) are retained here.

## Actual same-VM public development comparison

| Queue row | Variant | Complete command | Measured T4 peak | Eval loss | Status |
| --- | --- | ---: | ---: | ---: | --- |
| 46 | [Unmodified pinned control](benchmark_pinned_control.ipynb) | **32.94 s** | 3,461 MiB | 2.04989 | Completed; public comparison checks passed |
| 47 | [Original adaptive replay/budget policy](benchmark_hybrid_selective20_nf4split.ipynb) | **35.54 s** | **3,061 MiB** | 2.04982 | Completed; public comparison checks passed |
| 48 | [Our original NF4 word-flat Triton kernel](benchmark_original_nf4_flatword.ipynb) | Pending | Pending | Pending | Queued |
| 49 | [Our input-only grouped-NF4 MLP autograd](benchmark_own_inputonly_mlp_grouped.ipynb) | Pending | Pending | Pending | Queued |

Compare against the **same VM control**, not the upstream Kaggle pair (the upstream README reports 37.7 s / 3,461 MiB against an ~85 s baseline). These queue runs use a saved 71.85 s / 14,345 MiB reference, rather than a fully paired jury benchmark. They do not provide a hidden-dataset admission verdict.

## Our additions (independently written and separately inspectable)

- [Selective replay + near-peak buffer split](our_selective_layer_budget.py): use preexisting exact grouped-weight MLP gate replay for 20/28 layers, with forward/backward boundary split 2/3 instead of 1/2. This improves VRAM from 3,461 to 3,061 MiB at a 2.6 s time cost. Independent ablations needed to isolate each change.
- [Flat wordpack NF4 Triton](our_flatword_nf4.py): original independent 1-D packed-byte scheduling of the same nested-NF4 lookup/scale math. Per-launch output is FP16 packed pairs in 32-bit stores. Unsupported geometry delegates to the known-good call; GPU correctness/performance requires queue results.
- [Input-only frozen MLP autograd](our_inputonly_mlp_recompute.py): save only narrow MLP input for backward, rematerialize gate and up from the frozen *already grouped* NF4 projection weights, keep forward/backward starter SwiGLU numerics. Avoids full model gradient checkpointing and preserves the original task. GPU performance/LoRA parity needs measured results.

## Important originality limitation

The notebooks currently install the upstream external contributor kernel implementations; they are research artifacts, **not eligible to submit as our own original Triton kernel entry**. An independently eligible final package must replace copied GPU kernel code with original implementations and pass the organizer's originality, numeric, 79-regression, and paired-timing checks. Do not rebrand copied files or report source-derived kernel time as ours.

### Benchmark workflow
Use the [Google Sheets testing queue](https://docs.google.com/spreadsheets/d/16tkklNfLrhT7lG__8n_awxFULmBd5GEk2vF6JIZs9Nw/edit#gid=163034553) and the unchanged notebook training/export cells. Never overwrite result columns. Exact source commit, candidate, and request ID are embedded in each row's notes. GPU queue numbers alone are not a contest score.
