# LSTM Autoencoder - Anomaly Detection

Real-time anomaly detection system using LSTM Autoencoder trained on time-series data.

## Quick Start

### 1. Activate Environment
```bash
andet\Scripts\Activate.ps1
```

### 2. Test on Any Dataset
```bash
python test_any_dataset.py
```

### 3. Configure Dataset
Edit `test_any_dataset.py`:
```python
FILE_PATH = 'anamoly/normal data/realTraffic/realTraffic/TravelTime_387.csv'
THRESHOLD_PERCENTILE = 95  # Adjust sensitivity (95 = more sensitive, 99 = less)
```

## Project Structure

```
lstm/
├── test_any_dataset.py          # Main script - test on any CSV
├── requirements.txt             # Python dependencies
├── README.md                    # This file
│
├── models/                      # Trained model & results
│   └── lstm_autoencoder.keras
│
├── processed_data/              # Training data
│   ├── X_train_combined.npy
│   ├── X_test_real.npy
│   └── normalization_params.npz
│
├── anamoly/                     # Test datasets
│   └── normal data/
│
└── archive/                     # Backend files
    ├── data_pipeline.py         # Data processing
    └── setup1.py                # Model training
```

## Usage

The model detects anomalies by learning normal patterns and flagging deviations.

**Input:** CSV file with 'value' column  
**Output:** Visualization showing detected anomalies

## Model Details

- **Architecture:** LSTM Autoencoder (128→64→64→128)
- **Training:** 30,629 sequences (real + synthetic)
- **Window Size:** 50 timesteps
- **Threshold:** 95th-99th percentile of reconstruction error

## Results

Anomalies are shown as:
- Red shaded regions in data plot
- Red spikes in error plot
- Red dots in scatter plot

Saved to: `models/anomaly_detection_[dataset_name].png`
