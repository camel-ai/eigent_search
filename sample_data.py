from datasets import load_dataset
import json
import os

def main():
    # Load the dataset
    dataset = list(load_dataset("basicv8vc/SimpleQA")["test"])
    
    # Calculate sample size (10%)
    total_size = len(dataset)
    partition_size = int(total_size * 0.1)
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Create 10 partitions
    for i in range(10):
        start_idx = i * partition_size
        end_idx = start_idx + partition_size if i < 9 else total_size
        partition_data = dataset[start_idx:end_idx]
        
        # Save each partition
        output_file = f'data/simpleqa/simpleqa_partition_{i}.json'
        with open(output_file, 'w') as f:
            json.dump(partition_data, f, indent=2)
        print(f"Saved partition {i} ({len(partition_data)} samples) to {output_file}")

if __name__ == "__main__":
    main()
