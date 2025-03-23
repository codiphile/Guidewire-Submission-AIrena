# Archive Directory

This directory contains archived data files that are no longer actively used in the project but are kept for reference.

## Contents

- `large_dataset.csv` - Original large dataset before optimization
- `large_dataset_subset_*.csv` - Various subsets of the large dataset used for testing
- `sample_metrics.csv` - Sample metrics data used during development
- `kubernetes_node_metrics.csv` - Initial Kubernetes node metrics
- `simplified_metrics.csv` - Simplified test data
- `larger_metrics.csv` - Larger test dataset
- `training_metrics.csv` - Training data from previous iterations
- `improved_metrics.csv` - Dataset with improved metrics distribution
- `realistic_metrics.csv` - Dataset with realistic metrics (copied to generated_metrics.csv)

## Usage

If you need to reference these files or revert to a previous dataset, you can copy the relevant file from this directory back to the main data directory.

Example:

```bash
cp src/data/archive/realistic_metrics.csv src/data/generated_metrics.csv
```
