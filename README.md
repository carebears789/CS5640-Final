# German Credit Analysis with Multi-Objective Optimization

This project implements a comprehensive analysis of the German Credit dataset using multi-objective optimization (MOO) techniques from the LibMOON framework. It compares multiple gradient-based and learning-based solvers to find optimal neural network architectures that balance classification accuracy with model complexity.

## Overview

The analysis performs the following tasks:

1. **Data Preparation**: Loads the German Credit ARFF dataset and preprocesses it using standardization and one-hot encoding
2. **Problem Definition**: Defines a balanced multi-objective problem with two objectives:
   - Classification Error (minimize false positives/negatives)
   - Model Complexity (minimize L2 norm of weights)
3. **Solver Comparison**: Tests multiple solvers:
   - **Gradient-based solvers**: EPO, MGDA-UB, Random, PMGDA, HVGrad, MOOSVGD, PMTL
   - **Aggregation methods**: LS, AASF, PNorm, mTche, Tche, PBI, COSMOS, invagg, STche, SmTche
   - **Pareto Set Learning (PSL)**: Neural network hypernetwork approach
4. **Performance Evaluation**: Evaluates solutions using F1-score and other Pareto front metrics
5. **Visualization**: Generates Pareto front plots and comparative analysis

## Project Structure

```
Project3/
├── README.md
├── requirements.txt
├── GermanCreditAnalysis4.ipynb  # Main analysis notebook
├── libmoon/                      # LibMOON framework repository
│   ├── libmoon/                 # Core package
│   └── german_credit.arff       # Dataset file
└── pareto_plots/                # Output directory for visualizations
    ├── metrics/
    ├── solver_outputs/
    ├── figures/
    └── reproducibility/
```

## Installation

### Prerequisites
- Python 3.8+
- pip or conda

### Setup

1. Clone or extract the project:
```bash
cd Project3
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Analysis

Open and run the notebook in Jupyter:

```bash
jupyter notebook GermanCreditAnalysis4.ipynb
```

Or use JupyterLab:

```bash
jupyter lab GermanCreditAnalysis4.ipynb
```

### Key Cells

1. **Cell 1 - Environment Setup**: Sets up dynamic paths and creates output directories
2. **Cell 2 - Data Loading**: Loads and explores the German Credit dataset
3. **Cell 3 - Data Preprocessing**: Handles categorical/numeric features and train-test split
4. **Cell 4 - Problem Definition**: Defines the balanced MOO problem
5. **Cell 5 - Solver Import**: Loads all available gradient-based solvers
6. **Cell 6 - Solver Testing**: Runs all solvers and evaluates results
7. **Cell 7+ - Analysis & Visualization**: Generates metrics, plots, and comparative analysis

## Configuration

Key parameters you can adjust:

- **Complexity Weight** (Line in Problem Definition): Controls the balance between objectives
  - Higher values penalize complex models more
  - Lower values allow more complex models
  
- **Hidden Dimension** (Problem Definition): Size of hidden layer in the neural network
  - Default: 8
  
- **Number of Preferences** (Solver Testing): Number of preference vectors for MOO
  - Default: 5-100 (varies by solver)

- **Learning Rate** (PSL Solver): Learning rate for Pareto Set Learning
  - Default: 1e-4

- **Epochs/Iterations**: Number of iterations for each solver
  - Default: 1000-3000

## Output

The notebook generates:

- **Pareto Plots** (`pareto_plots/*.png`): Individual Pareto front visualizations for each solver
- **Metrics** (`pareto_plots/metrics/`): Performance metrics in CSV format
- **Solver Outputs** (`pareto_plots/solver_outputs/`): Detailed solver outputs
- **Summary Report**: Comparative table showing best F1-scores across solvers

## Key Functions

### `GermanCreditProblemBalanced`
Defines the multi-objective optimization problem with:
- Binary cross-entropy loss as objective 1
- L2-normalized weight penalty as objective 2

### `predict_with_flat_weights()`
Converts flat weight vectors to neural network predictions

### `process_and_visualize()`
Generates Pareto front plots for analysis

### `collect_metrics()`
Computes MOO metrics (HV, IGD, etc.)

## Results Interpretation

- **Best Error**: Lowest classification error achieved
- **Mean Error**: Average error across the Pareto front
- **Best Complexity**: Lowest model complexity on the front
- **Mean Complexity**: Average complexity across the front
- **F1 Score**: Weighted F1-score on test set

Solvers are ranked by their minimum classification error, allowing you to identify which approach best balances the two objectives for your use case.

## Troubleshooting

### Memory Issues
- Reduce number of preferences (n_prob)
- Reduce population size or number of epochs

### Import Errors
- Ensure LibMOON is properly installed
- Check that all paths in the setup cell match your system

### ARFF File Not Found
- Verify that `german_credit.arff` exists in the libmoon directory
- Update the path in the data loading cell if needed

## Dependencies

See `requirements.txt` for full list. Main packages:
- **torch**: Deep learning framework
- **scikit-learn**: Preprocessing and metrics
- **pandas**: Data manipulation
- **numpy**: Numerical computing
- **matplotlib/seaborn**: Visualization
- **pymoo**: Multi-objective optimization utilities
- **liac-arff**: ARFF file parsing

## References

- LibMOON: A Benchmark for Multi-Objective Optimization
- German Credit Dataset: UCI Machine Learning Repository

## License

This project uses the LibMOON framework. See libmoon/LICENSE for details.

## Contact

For questions or issues, refer to the libmoon documentation or create an issue in the project repository.
