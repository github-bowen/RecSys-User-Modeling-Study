# User Modeling Strategies for Recommender Systems

This repository contains the code for a systematic experimental comparison of different user modeling strategies for recommender systems. The experiment focuses on three strategies (demographic-based, behavioral-based, and hybrid) and evaluates their performance across users with different interaction densities.

## Research Question

**How does the value of demographic information in user modeling vary with the density of user interaction data?**

We investigate this question by comparing three user modeling strategies:

1. **Demographic-based**: Models users solely based on demographic attributes (age, gender, occupation, etc.)
2. **Behavioral-based**: Models users solely based on their past interactions with items
3. **Hybrid**: Combines both demographic and behavioral information into a unified user representation

To understand the impact of interaction density, we stratify users into sparse, medium, and dense groups based on their number of ratings, and evaluate each modeling strategy on each stratum.

## Dataset

The experiments use the **MovieLens 1M** dataset, which contains:

- 1 million ratings from 6,000 users on 4,000 movies
- Demographic information about users (gender, age, occupation, zip code)
- Timestamp information for each rating

This dataset is ideal for our experiment as it provides both rich interaction data and demographic attributes.

## Setup

### Prerequisites

- Python 3.9+
- Required packages listed in `requirements.txt` or `environment.yml`

### Installation

#### Option 1: Using venv

```bash
# Clone the repository
git clone https://github.com/github-bowen/RecSys-User-Modeling-Study.git
cd RecSys-User-Modeling-Study

# Install dependencies
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

#### Option 2: Using Conda

```bash
# Clone the repository
git clone https://github.com/github-bowen/RecSys-User-Modeling-Study.git
cd RecSys-User-Modeling-Study

# Create and activate conda environment
conda env create -f environment.yml
conda activate rec-user-modeling
```

### Download the Dataset

```bash
# Make the script executable
chmod +x download_data.sh

# Download and extract the MovieLens 1M dataset
./download_data.sh
```

> [!Important]
> Known issue in the dataset:
> There is a known encoding issue in the original `movies.dat` file. After downloading the data, you must manually correct an invalid character to prevent `UnicodeDecodeError` during data processing.
>
> - **File**: `data/raw/ml-1m/movies.dat`
> - **Line**: 73
>
> Change the line from:
>
> ```plaintext
> 73::Mis�rables, Les (1995)::Drama|Musical
> ```
>
> to:
>
> ```plaintext
> 73::Miserables, Les (1995)::Drama|Musical
> ```

## Running the Experiments

### Single Experiment

To run a single experiment with a specific user modeling strategy:

```bash
python main.py --user_model demographic --algorithm knn
python main.py --user_model behavioral --algorithm knn
python main.py --user_model hybrid --algorithm knn
```

### All Experiments

To run all combinations of user models and algorithms:

```bash
# Make the script executable
chmod +x run_experiments.sh

# Run all experiments
./run_experiments.sh
```

## Project Structure

```bash
recommender_user_modeling_study/
│
├── data/                               # Data files
│   └── raw/                            # Raw MovieLens 1M dataset
│
├── notebooks/                          # Jupyter notebooks
│   ├── 1-data-exploration.ipynb        # Dataset analysis
│   └── 2-results-visualization.ipynb   # Visualizing experimental results
│
├── src/                                # Source code
│   ├── algorithms/                     # Recommendation algorithms
│   │   ├── base_recommender.py         # Abstract base class
│   │   └── knn.py                      # K-Nearest Neighbors implementation
│   │
│   ├── user_modeling/                  # User modeling strategies
│   │   ├── base_model.py               # Abstract base class
│   │   ├── demographic.py              # Demographic-based modeling
│   │   ├── behavioral.py               # Behavioral-based modeling
│   │   └── hybrid.py                   # Hybrid modeling
│   │
│   └── utils/                          # Utility functions
│       ├── data_loader.py              # Data loading and preprocessing
│       └── evaluation.py               # Evaluation metrics calculation
│
├── results/                            # Experiment results
│
├── main.py                             # Main experiment script
├── config.yaml                         # Configuration parameters
├── requirements.txt                    # Python dependencies (pip)
├── environment.yml                     # Python dependencies (conda)
├── download_data.sh                    # Dataset download script
├── run_experiments.sh                  # Script to run all experiments
└── README.md                           # This file
```

## Evaluation Protocol

The evaluation follows these steps:

1. **Data Stratification**: Users are divided into sparse, medium, and dense strata based on their number of interactions.

2. **Data Splitting**: For each user, the leave-one-out method is used, where the most recent rating is held out for testing, and all previous ratings are used for training.

3. **Testing**: For each test item, 100 random items that the user has not interacted with are sampled as negative examples. The recommender's task is to rank these 101 items.

4. **Metrics**: Performance is evaluated using:
   - **Success Rate@10 (SR@10)**: The proportion of cases where the correct item appears in the top 10 recommendations.
   - **Mean Reciprocal Rank (MRR)**: The average of the reciprocal ranks of the correct items.

## Interpreting Results

Results will be saved in the `results/metrics.csv` file, which can be analyzed using the `notebooks/2-results-visualization.ipynb` notebook.

The visualization notebook will:

1. Compare the overall performance of the three user modeling strategies
2. Break down performance by user stratum
3. Calculate relative improvement of hybrid and behavioral models over the demographic baseline
4. Provide insights about the value of demographic information under different user interaction densities

## License

This project is licensed under the MIT License - see the LICENSE file for details.
