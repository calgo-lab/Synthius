# Synthetic Data Generation Toolkit

This repository provides a comprehensive toolkit for generating synthetic data using seven different models. The toolkit evaluates the generated data for utility, similarity/fidelity, and privacy, specifically tailored for tabular datasets with binary classification problems (e.g., True/False, Yes/No).

## Quick Start

### Step 1: Install the Package
Install the package using pip:
```bash
pip install synthius
```

### Step 2: Usage Example
To understand how to use this package, explore the three example Jupyter notebooks included in the repository:

Go to `examples/1-getting-started.ipynb` to begin.

### Additional Setup for Mac Users
Mac users may encounter errors during installation. To resolve these issues, install the required dependencies and set up the environment:

1. Install dependencies using Homebrew:
   ```bash
   brew install libomp llvm
   ```

2. Set up the environment:
   ```bash
   export PATH="/opt/homebrew/opt/llvm/bin:$PATH"
   export CC=$(brew --prefix llvm)/bin/clang
   export CXX=$(brew --prefix llvm)/bin/clang++
   export CXXFLAGS="-I$(brew --prefix llvm)/include -I$(brew --prefix libomp)/include"
   export LDFLAGS="-L$(brew --prefix llvm)/lib -L$(brew --prefix libomp)/lib -lomp"
   ```


## Acknowledgments
Special thanks to all contributors and the libraries used in this project.

This package includes TabDiff (Copyright 2024 Minkai Xu) under the MIT License.
For more details, see synthius/TabDiff/LICENSE.