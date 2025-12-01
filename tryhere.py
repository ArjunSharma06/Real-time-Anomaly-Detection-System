import numpy as np
import pandas as pd
import keras
from matplotlib import pyplot as plt
import os
# just insert relative path to the csv file here | (files are in anamoly folder)
#    make sure to use forward slashes            V
FILE_PATH = 'anamoly/normal data/realAWSCloudwatch/realAWSCloudwatch/rds_cpu_utilization_e47b3b.csv'
THRESHOLD_PERCENTILE = 95



#dont change anything below thissss




model = keras.models.load_model('models/lstm_autoencoder.keras')
params = np.load('processed_data/normalization_params.npz')
window_size = int(params['window_size'])
dataset_name = os.path.basename(FILE_PATH).replace('.csv', '')

df = pd.read_csv(FILE_PATH)
data = df['value'].values

data_normalized = (data - data.min()) / (data.max() - data.min())

def create_sequences(data, window_size):
    return np.array([data[i:i + window_size] for i in range(len(data) - window_size + 1)]).reshape(-1, window_size, 1)

X_test = create_sequences(data_normalized, window_size)
X_pred = model.predict(X_test, verbose=0)
errors = np.mean(np.abs(X_test - X_pred), axis=(1, 2))

threshold = np.percentile(errors, THRESHOLD_PERCENTILE)
anomalies = errors > threshold

fig, axes = plt.subplots(3, 1, figsize=(15, 10))
fig.suptitle(f'{dataset_name} - Anomaly Detection', fontsize=16, fontweight='bold')

ax = axes[0]
ax.plot(data, alpha=0.7)
for idx in np.where(anomalies)[0]:
    ax.axvspan(idx, idx + window_size, alpha=0.3, color='red')
ax.set_ylabel('Value')
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(errors, alpha=0.7)
ax.axhline(threshold, color='red', linestyle='--', linewidth=2)
ax.fill_between(range(len(errors)), 0, threshold, alpha=0.2, color='green')
ax.fill_between(range(len(errors)), threshold, errors.max(), where=(errors > threshold), alpha=0.3, color='red')
ax.set_ylabel('Error')
ax.grid(True, alpha=0.3)

ax = axes[2]
normal = np.where(~anomalies)[0]
anom = np.where(anomalies)[0]
ax.scatter(normal, errors[normal], c='green', s=10, alpha=0.5, label='Normal')
ax.scatter(anom, errors[anom], c='red', s=20, alpha=0.8, label='Anomaly')
ax.axhline(threshold, color='red', linestyle='--', linewidth=2)
ax.set_xlabel('Sequence Index')
ax.set_ylabel('Error')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'models/visualizations/anomaly_detection_{dataset_name}.png', dpi=150)
plt.show()

print(f"{dataset_name}: {np.sum(anomalies)} anomalies ({np.sum(anomalies)/len(errors)*100:.2f}%)")
