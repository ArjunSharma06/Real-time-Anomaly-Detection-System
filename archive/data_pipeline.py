import numpy as np
import pandas as pd
import os
from pathlib import Path

class DataPipeline:
    def __init__(self, window_size=50, train_split=0.3, synthetic_count=3000, random_seed=42):
        self.window_size = window_size
        self.train_split = train_split
        self.synthetic_count = synthetic_count
        np.random.seed(random_seed)
        
    def load_nab_data(self, file_path):
        df = pd.read_csv(file_path)
        self.real_data = df['value'].values
        return self.real_data
    
    def split_data(self):
        split_index = int(len(self.real_data) * self.train_split)
        self.real_train = self.real_data[:split_index]
        self.real_test = self.real_data[split_index:]
        return self.real_train, self.real_test
    
    def generate_synthetic_data(self):
        train_mean = np.mean(self.real_train)
        train_std = np.std(self.real_train)
        train_amplitude = (self.real_train.max() - self.real_train.min()) / 2
        
        synthetic_sequences = []
        sequence_length = self.window_size * 2
        
        for i in range(self.synthetic_count):
            freq1 = np.random.uniform(0.05, 0.15)
            freq2 = np.random.uniform(0.03, 0.10)
            t = np.linspace(0, 10, sequence_length)
            wave1 = np.sin(2 * np.pi * freq1 * t)
            wave2 = np.cos(2 * np.pi * freq2 * t)
            combined = (wave1 + wave2) / 2
            scaled = combined * train_amplitude + train_mean
            noise = np.random.normal(0, train_std * 0.3, sequence_length)
            synthetic_sequences.append(scaled + noise)
        
        self.synthetic_data = np.concatenate(synthetic_sequences)
        return self.synthetic_data
    
    def normalize_data(self):
        self.data_min = self.real_train.min()
        self.data_max = self.real_train.max()
        
        self.real_train_norm = (self.real_train - self.data_min) / (self.data_max - self.data_min)
        self.real_test_norm = (self.real_test - self.data_min) / (self.data_max - self.data_min)
        self.synthetic_norm = (self.synthetic_data - self.data_min) / (self.data_max - self.data_min)
        
        return self.real_train_norm, self.real_test_norm, self.synthetic_norm
    
    def create_sequences(self, data):
        sequences = [data[i:i + self.window_size] for i in range(len(data) - self.window_size + 1)]
        return np.array(sequences).reshape(-1, self.window_size, 1)
    
    def format_sequences(self):
        self.X_train_real = self.create_sequences(self.real_train_norm)
        self.X_train_synthetic = self.create_sequences(self.synthetic_norm)
        self.X_test_real = self.create_sequences(self.real_test_norm)
        return self.X_train_real, self.X_train_synthetic, self.X_test_real
    
    def merge_training_data(self):
        self.X_train_combined = np.concatenate([self.X_train_real, self.X_train_synthetic], axis=0)
        np.random.shuffle(self.X_train_combined)
        return self.X_train_combined
    
    def save_processed_data(self, output_dir="."):
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        np.save(output_path / "X_train_combined.npy", self.X_train_combined)
        np.save(output_path / "X_test_real.npy", self.X_test_real)
        np.savez(output_path / "normalization_params.npz", 
                 data_min=self.data_min, 
                 data_max=self.data_max,
                 window_size=self.window_size)
    
    def run_pipeline(self, nab_file_path, output_dir="."):
        self.load_nab_data(nab_file_path)
        self.split_data()
        self.generate_synthetic_data()
        self.normalize_data()
        self.format_sequences()
        self.merge_training_data()
        self.save_processed_data(output_dir)
        print(f"Processed: {self.X_train_combined.shape[0]} train, {self.X_test_real.shape[0]} test sequences")
        return self.X_train_combined, self.X_test_real


if __name__ == "__main__":
    pipeline = DataPipeline(window_size=50, train_split=0.3, synthetic_count=3000, random_seed=42)
    pipeline.run_pipeline(
        nab_file_path=r"normal data\realKnownCause\realKnownCause\machine_temperature_system_failure.csv",
        output_dir="processed_data"
    )
