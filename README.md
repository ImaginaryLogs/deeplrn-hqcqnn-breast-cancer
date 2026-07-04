```
qcqcnn-cbis/
├── pyproject.toml
├── uv.lock
├── README.md
│
├── configs/
│   ├── base.yaml              # shared hparams: lr, batch_size, ansatz_reps, etc.
│   └── cells.yaml             # the 7-12 (preprocessing, middle_layer) combos to run
│
├── src/
│   └──│       ├── __init__.py
│       ├── preprocessing.py       # naive_resize, clahe_resize, haar_dwt  [have]
│       ├── dataset.py             # CBISDDSMDataset, stratified split      [have]
│       ├── datamodule.py          # CBISDDSMDataModule (pl.LightningDataModule)
│       ├── middle_layers.py       # MiddleLayerBase + 4 variants           [have]
│       ├── quantum_filter.py      # wraps the original repo's quantum conv filter
│       ├── vqc_head.py            # TorchConnector-wrapped EstimatorQNN
│       ├── lightning_module.py    # QCQCNNLightningModule (assembles the 3 above)
│       └── metrics.py             # any custom torchmetrics beyond the built-ins
│
├── scripts/
│   ├── env_check.py            # isolated Aer-GPU import test — run this FIRST, alone
│   ├── run_cell.py             # single factorial cell: python run_cell.py --preprocessing clahe --middle_layer residual
│   ├── run_all_cells.py        # loops configs/cells.yaml, calls run_cell as subprocess or function
│   └── tune_baseline.py        # Optuna sweep on baseline×naive only, writes best hparams to configs/base.yaml
│
├── notebooks/
│   ├── 00_env_sanity.ipynb     # your Saturday MNIST-demo notebook, kept as historical record
│   ├── 01_shape_trace.ipynb    # trace quantum filter output shape once, by hand
│   └── 02_results_analysis.ipynb  # after runs finish: tables, plots for the paper
│
├── data/
│   ├── raw/                    # NBIA-downloaded DICOMs (gitignored, large)
│   ├── metadata/                # mass_case_description_*.csv from TCIA
│   └── cache/                  # preprocessed .npy cache, keyed by condition — matches
│                                 # the cache_dir param already in CBISDDSMDataset
│
├── outputs/
│   ├── checkpoints/             # per-cell model checkpoints (ModelCheckpoint callback)
│   ├── logs/                   # Lightning/tensorboard logs
│   └── results.csv             # one row per cell: condition, middle_layer, acc/auc/f1/train_time
│
├── tests/
│   ├── test_preprocessing.py   # shape/range contract checks (the __main__ blocks you saw, formalized)
│   ├── test_middle_layers.py   # param-parity check as an actual assertion, not eyeballed output
│   └── test_dataset.py         # split leakage check — assert no patient_id in >1 split
│
└── paper/
    └── m2_draft.tex             # ACM SIGCONF, lives in the repo so figures/tables can be scripted in

```