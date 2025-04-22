project_root/
├── configs/                 # Configuration files (YAML/JSON)
│   ├── data_params.yaml
│   └── model_params.yaml
|   
├── data_handling/           # Data loading & preprocessing
│   ├── __init__.py
│   ├── loader.py            # Loads raw data (e.g., from Parquet )
│   ├── preprocessing.py     # Normalization, cleaning 
│   ├── feature_engineering.py # Derived features (spread, imbalance)
│   └── batching.py          # Creates sliding windows/batches
├── models/                  # Model definitions
│   ├── __init__.py
│   ├── base_model.py        # Abstract BaseModel class
│   ├── linear_model.py
│   ├── cnn_model.py         # Inspired by crypto_lob.pdf, basic_cnn_model.ipynb 
│   ├── transformer_model.py # TLOB-style, inspired by TLOB.pdf 
│   └── hybrid_model.py      # CNN-Transformer 
├── training/                # Training logic
│   ├── __init__.py
│   └── trainer.py           # Training loops, optimizers, loss functions
├── evaluation/              # Evaluation & metrics
│   ├── __init__.py
│   └── evaluator.py         # Calculates F1, accuracy, MAE, etc. 
├── tuning/                  # Hyperparameter optimization
│   ├── __init__.py
│   └── tuner.py             # Interfaces with Optuna/Ray Tune, etc.
├── backtesting/             # Strategy simulation 
│   ├── __init__.py
│   └── backtester.py        # Calculates P&L, Sharpe ratio
├── scripts/                 # Runnable scripts
│   ├── train_model.py
│   └── evaluate_model.py
├── notebooks/               # Exploration, visualization (like basic_cnn_model.ipynb)
├── data/                    # Raw and processed data (or symlinks)
└── README.md