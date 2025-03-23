# Inside a Python script or notebook
from src.utils.synthetic_data import augment_dataset
import pandas as pd
from src.utils.multicode_dataScript import *

# Generate the datase
data_generator = KubernetesDataGenerator(num_days=366, sample_interval_minutes=5, num_nodes=10)
synthetic_data = data_generator.generate_data()

# Save to CSV
data_generator.save_to_csv('src/data/kubernetes_node_metrics.csv')

# Print statistics
data_generator.print_statistics()

# Load the original dataset
original_data = pd.read_csv('src/data/kubernetes_node_metrics.csv')

# Generate synthetic data to augment the dataset
# The synthetic_ratio parameter controls the target ratio of failures in the dataset
augmented_data = augment_dataset(original_data, synthetic_ratio=0.15)

# Save the augmented dataset if needed
augmented_data.to_csv('src/data/augmented_kubernetes_metrics.csv', index=False)