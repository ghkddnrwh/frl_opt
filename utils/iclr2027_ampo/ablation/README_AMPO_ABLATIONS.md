# AMPO ablation-analysis scripts for `logs/wandb_logs_final_160`

## Expected input layout

Run every script from the project root. By default the scripts expect:

```text
logs/wandb_logs_final_160/
  ppo_avg/
  ampo_uniform/
  ampo_adaptive_dual_lr_1e-4/
  ampo_adaptive_dual_lr_3e-4/
```

with

```text
<variant>/<env>/<friction|gravity>/seed_<1..5>__*/history.jsonl
```

and the accompanying `config.json` files.

All outputs are written below:

```text
logs/iclr2027_ampo/ablation/
```

All generated LaTeX row blocks now print and save a copy-ready column header followed by `\\midrule`, e.g.

```latex
Environment & Perturbation & Gain(1e-4) & Gain\%(1e-4) & Gain(3e-4) & Gain\%(3e-4) \\
\midrule
...
```

so the terminal output or `latex_*_rows.txt` contents can be pasted directly inside an existing `tabular` environment.

Each ablation gets its own subdirectory. Figures are saved as PNG only (300 dpi); PDF files are not generated. Plotting follows the supplied publication style: serif/Times New Roman preference, STIX math text, paper-size font settings, no legend frame, light grid, tight bounding box.

## Recommended execution

Build the reduced cache once first because each raw W&B `history.jsonl` is large:

```bash
python ablation_00_build_cache.py
```

Then run an individual analysis, for example:

```bash
python ablation_02_dual_tracking_alignment.py
python ablation_03_lambda_dynamics.py
```

or run everything:

```bash
python run_all_ablation.py
```

Useful options shared by the scripts:

```bash
--log-root logs/wandb_logs_final_160
--out-root logs/iclr2027_ampo/ablation
--late-evals 10
--force-cache
```

The cache is stored under `logs/iclr2027_ampo/ablation/_cache` and is automatically reused.

## Why each result is a table or a plot

### 01. Adaptive vs Uniform — **main-paper table**

File: `ablation_01_adaptive_vs_uniform.py`

Question: Does adaptive dual reweighting itself improve robust performance relative to the same AMPO backbone with uniform lambda?

Primary result: LaTeX rows for late-training worst-case return, separated into friction and gravity. A table is preferred because the claim is a precise across-task performance comparison and the exact mean ± seed standard deviation matters. The best mean in each environment row is bolded automatically. Additional rows for average, nominal, bottom-2, and average-minus-worst gap are generated for appendix use. Adaptive-minus-uniform gains are paired by seed.

### 02. Dual tracking alignment — **main-paper heatmaps + exact appendix rows**

File: `ablation_02_dual_tracking_alignment.py`

Question: Does the dual actually track the performance-limiting environments?

Primary result: heatmaps. The mechanism is multi-dimensional and heterogeneous across eight task/perturbation settings, so a heatmap exposes the pattern better than a wide table. It separates:

- signal fidelity: `rho(J_dual, J_eval)`
- dual responsiveness: `rho(lambda_after, -J_dual)`
- actor/eval alignment: `rho(lambda_actor, -J_eval)`
- post-dual/eval alignment
- Top-1 match
- lambda mass on the eval-worst client
- lambda mass on the eval bottom-2 clients

Each seed is first averaged over the final evaluation points; the final mean/std is across seeds.

### 03. Per-client lambda dynamics — **main-paper temporal plot**

File: `ablation_03_lambda_dynamics.py`

Question: As client performance changes, how do the five lambda weights move?

Primary result: two-panel temporal plots for every environment, perturbation, and adaptive dual LR. The upper panel is per-client evaluation return and the lower panel is the corresponding actor lambda. Both use mean ± seed std. The client color/order is kept consistent between panels and the legend includes the fixed perturbation value.

Recommended main-paper examples are Ant-Friction and Ant-Gravity at `3e-4`; the other figures are appropriate for the appendix.

### 04. Policy-dependent worst-environment switching — **table**

File: `ablation_04_worst_environment_switching.py`

Question: Is the worst environment genuinely policy-dependent and changing over training, and is lambda slower/smoother than the instantaneous worst identity?

Primary result: LaTeX rows with full/late worst-ID switch rates, a 3-evaluation smoothed worst switch rate, top-lambda switch rate, dominant-ID shares, and Top-1 matching. A table is preferred because these are compact rates rather than trajectories.

### 05. Dual LR timescale — **main/appendix table**

File: `ablation_05_dual_lr_timescale.py`

Question: What changes when the dual LR increases from `1e-4` to `3e-4`?

Primary result: table containing lambda max, effective number of environments `K_eff`, normalized lambda entropy, lambda mass on the worst/bottom-2 environments, dual-update norm, and cap-hit rate. With only two dual-LR levels, exact values are more informative than a line plot.

### 06. Robustness–scalability trade-off — **main-paper scatter**

File: `ablation_06_robustness_scalability_tradeoff.py`

Question: Is stronger concentration (lower `K_eff`) empirically associated with larger worst-return gain over uniform weighting?

Primary result: scatter/error-bar plot of late `K_eff` versus seed-paired worst-return gain over AMPO-Uniform. Each point is one environment/perturbation setting; error bars are seed standard deviations. The script also reports seed-level and setting-level Spearman correlations.

Interpret this as an empirical association, not a causal proof of the theoretical speedup result.

### 07. Focus-response / bottleneck resolution — **appendix table**

File: `ablation_07_focus_response.py`

Question: During an interval between evaluations, does the client receiving the largest average lambda improve more by the next evaluation?

The script averages `lambda_actor` over all server rounds in each evaluation interval and computes:

```text
Delta J_focused - mean(Delta J_other clients)
```

plus the rank correlation between interval-average lambda and next-evaluation improvement. This is explicitly a temporal association rather than a causal estimate. The current data can reveal that immediate one-interval improvement is not necessary for long-run robust gain, so this is best kept as a mechanistic appendix result unless it becomes particularly strong.

### 08. Gradient diagnostics — **appendix heatmap**

File: `ablation_08_gradient_diagnostics.py`

Question: Are pairwise client-gradient conflicts actually transferred into conflict between the AMPO aggregate and high-priority clients?

The heatmaps compare pairwise conflict rate, high-lambda conflict rate, top-lambda/aggregate cosine, worst-return/aggregate cosine, lambda-vs-gradient-influence gap, and whether the max-influence client is the worst-return client. This directly supports or rejects a gradient-conflict explanation but is secondary to the main dual-tracking claim.

### 09. Lambda-cap activity — **sanity table, not a causal cap ablation**

File: `ablation_09_lambda_cap_activity.py`

Question: In which current runs does `dual_lambda_cap=0.5` actually bind?

This reports the late cap-hit rate, lambda max, support size and `K_eff`. The current archive has capped adaptive runs but no matched uncapped adaptive control, so these outputs must not be presented as evidence that the cap improves performance. A true cap ablation requires matched `cap=None` runs under identical seeds/settings.

## Statistical convention

For paper summaries, the scripts avoid treating evaluation points as independent samples. They first compute the requested late-training statistic within each seed, then report **mean ± standard deviation across the five seeds**. Adaptive-vs-uniform performance gains are paired by seed.

The default late window is the final 10 evaluation points (`--late-evals 10`), matching the current analysis convention. No interpolation is used when lambda and evaluation data are aligned: only exact communication-round matches are used.

## Suggested paper order

Main ablation/mechanistic section:

1. `01_adaptive_vs_uniform`: effect of adaptive reweighting on worst return.
2. `02_dual_tracking_alignment`: whether lambda tracks the available bottleneck signal and the eval-defined worst client.
3. `03_lambda_dynamics`: temporal example showing per-client return and lambda movement.
4. `05_dual_lr_timescale`: dual LR controls focus strength and `K_eff`.
5. `06_robustness_scalability_tradeoff`: empirical counterpart of the `K_eff` trade-off.

Appendix/supporting analysis:

- `04_worst_environment_switching`
- `07_focus_response`
- `08_gradient_diagnostics`
- `09_lambda_cap_activity`
