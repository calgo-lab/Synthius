# Synthetic Data Generation Toolkit

This repository provides a comprehensive toolkit for generating synthetic data using seven different models. The toolkit evaluates the generated data for utility, similarity/fidelity, and privacy, specifically tailored for tabular datasets with binary classification problems (e.g., True/False, Yes/No).

## Models Included
The project implements the following models for synthetic data generation:
1. **CopulaGAN**
2. **CTGAN**
3. **Gaussian Copula**
4. **TVAE**
5. **Gaussian Multivariate**
6. **WGAN**
7. **ARF**

## Quick Start

### Step 1: Install the Package
Install the package using pip:
```bash
pip install synthius
```

### Step 2: Usage Example
To understand how to use this package, explore the three example Jupyter notebooks included in the repository:

1. **[Generator](pexample/1_generator.ipynb)**
   - Demonstrates how to generate synthetic data using seven different models.
   - Update paths and configurations (e.g., file paths, target column) to fit your dataset.
   - Run the cells to generate synthetic datasets.

2. **[AutoGloun](example/2_autogloun.ipynb)**
   - Evaluates the utility.
   - Update the paths as needed to analyze your data.

3. **[Evaluation](pexample/3_evaluation.ipynb)**
   - Provides examples of computing metrics for evaluating synthetic data, including:
     - Utility
     - Fidelity/Similarity
     - Privacy
   - Update paths and dataset-specific configurations and run the cells to compute the results.

These notebooks are provided as examples to help users understand and use the toolkit effectively.


## Acknowledgments
Special thanks to all contributors and the libraries used in this project.