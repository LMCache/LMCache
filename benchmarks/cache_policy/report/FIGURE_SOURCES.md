# Figure sources

Every figure under `figures/` is rendered by
[`generate_figures.py`](generate_figures.py) from the checked-in JSON
result files listed below -- no figure contains hardcoded numbers copied
from an external calculation. Regenerate all of them with
[`../reproduce_figures.sh`](../reproduce_figures.sh).

| Figure | Source(s) |
|---|---|
| `fig1_hit_rate_vs_cache_size_multiseed.png` | `results/admission_control/multiseed_sweep_ci.json` |
| `fig2_latency_p95_vs_cache_size_multiseed.png` | `results/admission_control/multiseed_sweep_ci.json` |
| `fig3_evictions_vs_rejections.png` | `results/admission_control/sweep_results.json` |
| `fig4_real_data_paired_diff_200mib.png` | `results/real_data/real_dataset_paired_diff.json` |
| `fig5_zipf_robustness.png` | `results/admission_control/robustness_zipf_skew.json` |
| `fig6_ablation.png` | `results/admission_control/admission_control_ablation.json`, `results/admission_control/windowed_admission_control_ablation.json` |
| `fig7_freeze_illustration.png` | Generated live from a small, deterministic `novel_long` replay (no JSON input -- see `fig_freeze_illustration` in `generate_figures.py`) |
| `fig8_latency_distribution.png` | `results/real_data/real_dataset_raw.json` |
| `fig9_multi_round_chat_case_study.png` | `results/admission_control/multi_round_chat_case_study.json` |
| `fig10_admission_vs_lfu.png` | `results/admission_control/admission_vs_lfu_paired.json` (produced by `../compare_admission_vs_lfu.py` from `results/admission_control/multiseed_sweep_raw.json`) |

All paths above are relative to `benchmarks/cache_policy/`.
