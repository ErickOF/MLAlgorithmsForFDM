# ML Algorithms for FDM 3D Printing Quality

Research project for predicting and optimizing dimensional accuracy and surface roughness in **MEX/FDM 3D printing** using machine learning. Based on a replication of the Shirmohammadi (2021) experimental dataset.

## Objectives

- Predict **surface roughness** (Ra at 0.8 µm and 2.5 µm cutoff lengths)
- Predict **dimensional deviation** (length, width, and height)
- Compare multiple ML algorithms suitable for small tabular datasets
- Optimize FDM process parameters using surrogate-based methods

## Process Parameters (Inputs)

| Parameter | Unit |
|---|---|
| Nozzle temperature | °C |
| Layer height | mm |
| Print speed | mm/s |
| Infill percentage | % |

## Repository Structure

```
├── app/                        # Flask web application
│   ├── app.py                  # Main app entry point
│   ├── requirements.txt        # Python dependencies
│   ├── src/
│   │   ├── run_quality_analysis.py   # ML pipeline: training, evaluation, HTML report
│   │   ├── optimize_quality.py       # Surrogate-based process optimization
│   │   ├── run_optimization.py       # CLI wrapper for optimization
│   │   └── run_from_config.py        # Config-driven pipeline runner
│   └── docs/
│       └── plan.md             # Project plan and decision matrix
│
└── exploratory/                # Exploratory notebooks
    ├── data/                   # Raw dataset (Excel)
    └── algorithms/
        ├── ann/                # Artificial Neural Network experiments
        ├── bnn/                # Bayesian Neural Network experiments
        ├── kans/               # Kolmogorov-Arnold Network experiments
        ├── tabpfn/             # TabPFN (foundation model) experiments
        ├── test_models/        # Multi-algorithm benchmarking
        └── thesis_analysis/    # Final analysis and visualizations
```

## ML Models Compared

| Type | Models |
|---|---|
| Regression | Elastic Net, SVR (RBF), Gaussian Process (GPR), MLP, k-NN |
| Classification | SVC (RBF), k-NN Classifier |
| Exploratory | BNN, KAN, TabPFN |

## Optimization Algorithms

Each trained model can be used as a surrogate for process parameter optimization:

| Optimizer | Best suited for |
|---|---|
| PSO (Particle Swarm) | MLP |
| Bayesian Optimization (GP) | SVR, GPR, k-NN |
| DE (Differential Evolution) | MLP, SVR, GPR |
| DOE grid search | Elastic Net |
| Gradient descent | MLP, Elastic Net |

## Web Application

The Flask app provides a browser UI to run analyses and optimizations without CLI interaction.

### Setup

```bash
cd app
pip install -r requirements.txt
```

### Run

```bash
python app.py --data <path/to/data.xlsx> --port 5000
```

Then open `http://localhost:5000` in your browser.

## Evaluation Metrics

- **Regression:** MAE, RMSE, R², residual analysis
- **Classification:** F1 macro, balanced accuracy, confusion matrix
- **Stability:** repeated K-fold cross-validation, seed sensitivity

## Dataset Notes

- Source: experimental design after printing 43 samples.
- Experiments 28, 38, and 40 are excluded from roughness targets (identified bias)
- Small dataset — models selected for robustness with few samples

## Requirements

Python 3.10+ and the packages in [app/requirements.txt](app/requirements.txt):

```
matplotlib, numpy, openpyxl, pandas, plotly, pymoo, pyswarms,
scikit-learn, scikit-optimize, seaborn, scipy
```
