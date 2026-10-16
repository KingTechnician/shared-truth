# shared-truth

Code for *Shared Truth: Emergent Truth Properties in Large Language Models via Heterogeneous
Injection-based Transfer* (Freeman & Ngangmeni, NeurIPS 2026).

A learned adapter maps a source model's residual-stream activations into a target model's truth layer.
The fused activation `(1−α)·h_T + α·map(h_S)` is swept over α, and the target's own truth probe scores
each α. The paper's main quantity is **strict overshoot**: the best optimal-threshold accuracy over
α ∈ (0, 1] minus the α = 0 value.

- Results: [KingTechnician/shared-truth-results](https://huggingface.co/datasets/KingTechnician/shared-truth-results)
  (use the `camera-ready` tag; its dataset card says which file backs which table)
- Eval data: [KingTechnician/tiu-splits](https://huggingface.co/datasets/KingTechnician/tiu-splits)
- Adapters and truth probes: the `KingTechnician/*` model repos named in each run's `sweep_results.json`

## Reproducing the results

Four Colab notebooks re-check the published numbers against the saved results. Each one clones this
repo, imports `shared_truth`, and only reads from the results repo. Nothing is uploaded, and re-runs
write to a scratch directory.

| notebook | GPU? | reproduces |
|---|---|---|
| `shared_truth_bootstrap_ci.ipynb` | no | Table 7 strict-overshoot point estimates and bootstrap CIs (17/21 unique trajectories exclude zero) |
| `shared_truth_procrustes.ipynb` | optional | the Procrustes baseline (matches or beats the trained adapter on 18/21), its medians and Spearman correlations, and the extraction-parity check |
| `shared_truth_adapter_experiment.ipynb` | parts 1–2 | α sweeps for the trained adapters, statement-type Tables A–D, baseline Table E, the residual-geometry Table F, and the seeded 1000-draw random-weight null |
| `shared_truth_rebuttal_figures.ipynb` | no | figures R1–R8 and Tables E–F from saved results |

Every notebook opens with a table of its parts and what each part needs. The CPU parts run on a
free Colab runtime. The GPU parts need enough memory for the pair they sweep (an A100 covers every
pair).

## The `shared_truth` package

| module | what it holds |
|---|---|
| `naming` | model registries, adapter repo-name parsing, and canonical `pair_id`s (`<src>_l<N>_to_<tgt>_l<M>__<variant>`) |
| `activations` | model and adapter loading (`revision=` pins an adapter commit), the `ModelWrapper` hook patch, last-token extraction, and `check_extraction_parity` |
| `probes` | truth-probe loading, projection onto (t_G, t_P), and scoring |
| `env` | `pin_sklearn()`: installs the scikit-learn version the `.skops` probes were saved under (1.8) |
| `metrics` | `optimal_accuracy` (tie-guarded), strict overshoot, the paired stratified bootstrap, and `sweep_diff` |
| `sweep` | the α-sweep core for trained adapters and closed-form maps (`fit_procrustes`), plus `run_all` and `replot_all` |
| `residuals` | the rebuttal tables: A–D (`statement_type_tables`), E (`table_e`), and F (`table_f`), each with a comparison against the published CSVs |
| `storage` | provenance stamping, the results repo (`fetch_results`, `ensure_local`), and adapter revision checks (`resolve_revision`, `verify_adapter`) |
| `plots` | figure style and the three standard per-pair figures |

The package imports lazily and needs no GPU to import. Model loading uses `ModelWrapper` from the
representation-transfer codebase, so the GPU paths need that repo on the path:

```bash
git clone https://github.com/KingTechnician/Closing-Backdoors-Via-Representation-Transfer.git
export PYTHONPATH=$PWD/Closing-Backdoors-Via-Representation-Transfer:$PYTHONPATH
```

The notebooks do this themselves. Main dependencies: `torch`, `transformers`, `huggingface_hub`,
`datasets`, `numpy`, `pandas`, `scipy`, `scikit-learn==1.8` with `skops` (for the probes), and
`matplotlib`.

## Pipeline notebooks

These notebooks built the inputs the experiments consume: the eval splits, the layer choices, the
per-layer activations, the probes, and the adapters. They are kept as they ran, and they build on
open-source code they clone: a fork of [Truth is Universal](https://github.com/sciai-lab/Truth_is_Universal)
([KingTechnician/Truth_is_Universal](https://github.com/KingTechnician/Truth_is_Universal), which adds
the layer calibration and the Hub activation export) and
[Closing Backdoors via Representation Transfer](https://github.com/withmartian/Closing-Backdoors-Via-Representation-Transfer).
Some set a local path to the fork (`tiu_path`) that you will need to change.

| notebook | step |
|---|---|
| `tiu_dataset_collection.ipynb` | assembles the Truth-is-Universal statements into the `tiu-splits` dataset (topic, polarity, statement type) |
| `tiu_calibrate.ipynb` | extracts activations at every layer and scores candidate layers with the Fisher separation ratio to choose each model's truth layer |
| `tiu_activations_by_layer_index_hf.ipynb` | extracts last-token activations at the chosen layer and uploads them as `tiu-acts-*` datasets |
| `tiu_probe_training.ipynb` | trains the TTPD truth probes and uploads each probe (`model.skops`) with its truth directions |
| `shared_truth_calibration_test.ipynb` | checks the uploaded probes against the test split |
| `representation_similarity.ipynb` | cross-model RSA over the candidate layer pairs (the `transfer_rsa` layer choice) |
| `closing_backdoors_training.ipynb` | trains the source→target adapters |

## Tests

```bash
export PYTHONPATH=$PWD            # no install step; the package is imported from the repo root
pytest tests/test_adapter_helpers.py         # offline, seconds
HF_TOKEN=... pytest tests/ -v                # adds the published-number regressions
```

| test file | checks |
|---|---|
| `test_adapter_helpers.py` | adapter parsing, `sweep_diff`, and adapter revision verification (offline) |
| `test_bootstrap_regression.py` | recomputes all 22 bootstrap trajectories and matches `strict_overshoot_bootstrap.json` (a few minutes on CPU) |
| `test_statement_types_regression.py` | Tables A–D: an offline check against the original notebook code, then the published CSVs |
| `test_residuals_regression.py` | Tables E–F: the same two layers of checks (pulls ~280 MB of activation dumps) |

To reuse a results snapshot that is already downloaded, set `SHARED_TRUTH_RESULTS_DIR` to it. The
network tests skip if the Hub is unreachable.

## Reproducibility notes

- **Duplicated trajectory.** One trajectory appears in the saved results under two variant names, so
  there are 22 folders but 21 unique trajectories. Counts and medians use the 21.
- **Two evals.** The published and `truth_*` runs use simple statements (n = 1599). The `stmt_*` runs
  use the full test set (n = 2684).
- **Probe version.** Call `env.pin_sklearn()` before loading a probe. Under other scikit-learn versions
  the probes load but fail or mis-score.
- **Adapter revisions.** Older runs did not record the adapter commit. `storage.verify_adapter(payload)`
  checks a saved run's recorded adapter config against what the repo serves now and raises
  `AdapterMismatchError` if the weights differ. New runs record the commit.
- **Re-probe AUROC** in Table F varies at the third decimal across environments. The comparisons
  allow 0.01; quote it to 2 decimals.

The results dataset card lists these in full, with the floor-pair seed checkpoints and the
random-weight baseline.

## Citation

```bibtex
@inproceedings{freeman2026sharedtruth,
  title     = {Shared Truth: Emergent Truth Properties in Large Language Models via Heterogeneous Injection-based Transfer},
  author    = {Freeman, Isaiah and Ngangmeni, Joed},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```