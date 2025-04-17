# Project Outline 

## 📂 Phase 0 – Project Setup & Coordination
> **Serial** (prerequisite for everything else)
- Establish code repository structure (e.g. GitHub/GitLab)
- Agree on coding style, version control & PR process
- Set up communication channels (Slack/Discord/email)
- Define meeting cadence (weekly stand‑up, biweekly deep‑dives)
- Create a shared project plan & timeline document
- Decide on task‑assignment process

---

## 1. Data Acquisition & Preprocessing
> **Serial**: can start only after Phase 0
- **Parallel**: Download raw LOB snapshots (100 ms BTC data; 9 days)
- **Parallel**: Download aggregated LOB CSVs (1 s / 1 min / 5 min for BTC/ETH/ADA)
- **Parallel**: Time alignment & cleaning per dataset
- **Parallel**: Normalization & scaling per dataset
- **Parallel**: Exploratory Data Analysis (summary stats, plots)

---

## 2. Feature Engineering
> **Serial**: must wait for all cleaning & EDA in Phase 1
- **Parallel**: Raw bid/ask tensor pipeline
- **Parallel**: Derived-feature pipeline (spread, imbalance, mid-price)
- **Parallel**: Sliding-window batching code
- Persist processed data (HDF5 / TFRecord / PyTorch format)

---

## 3. Baseline Modeling
> **Serial**: depends on processed features from Phase 2
- **Parallel**: Train linear models (classification & regression)
- **Parallel**: Train CNN baseline model
- Evaluate using classification metrics (accuracy, F1) and regression metrics (MAE/RMSE)

---

## 4. Deep-Learning Architectures
> **Serial**: only after baselines are in place (for fair comparison)
- **Parallel**: Develop & train CNN architectures
- **Parallel**: Develop & train Transformer (e.g. TLOB-style) architectures
- **Parallel**: Develop & train CNN→Transformer hybrid architectures

---

## 5. Hyperparameter Optimization
> **Serial**: requires working model code from Phase 4
- **Parallel**: Hyperparam tuning for CNN models
- **Parallel**: Hyperparam tuning for Transformer models
- **Parallel**: Hyperparam tuning for Hybrid models

---

## 6. Back-Testing & Strategy Simulation
> **Serial**: needs tuned models from Phase 5
- **Parallel**: Backtest linear-model strategy
- **Parallel**: Backtest CNN-model strategy
- **Parallel**: Backtest Transformer strategy
- **Parallel**: Backtest Hybrid strategy
- Compute P&L, Sharpe ratio, drawdown, etc.

---

## 7. Generalization & Robustness
> **Serial**: build on back-test results from Phase 6
- **Parallel**: Test best models on ETH dataset
- **Parallel**: Test best models on ADA dataset
- **Parallel**: Ablation study: order-book depth (10 vs 50 levels)
- **Parallel**: Ablation study: sampling frequency (100 ms vs 1 s vs 5 min)

---

## 8. Documentation & Reporting
> **Mixed**
- **Parallel (early drafting)**:
  - Methodology (data pipeline, feature engineering)
  - Model architectures & hyperparam-tuning write-up
  - Dataset descriptions & preprocessing steps
  - Metrics & evaluation methodology
- **Serial (final assembly)**:
  - Integrate all experiment results (tables, figures)
  - Write discussion, limitations, next-steps
  - Polish slide deck & LaTeX report **after** all experiments complete

---

## 9. Finalization & Submission
> **Serial** (end-to-end gate)
- Code freeze & final end-to-end tests
- Slide-deck rehearsal & dry run
- Package deliverables (code, report, slides)
- Submit before deadline

