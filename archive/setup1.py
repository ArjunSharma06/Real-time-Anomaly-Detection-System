import numpy as np
import keras
from keras import layers
from matplotlib import pyplot as plt
import os
import time

x_train = np.load('processed_data/X_train_combined.npy')
x_test = np.load('processed_data/X_test_real.npy')
params = np.load('processed_data/normalization_params.npz')

data_min = params['data_min']
data_max = params['data_max']
TIME_STEP = int(params['window_size'])

input_shape = (x_train.shape[1], x_train.shape[2])

model = keras.Sequential([
    layers.Input(shape=input_shape),
    layers.LSTM(128, activation='relu', return_sequences=True),
    layers.Dropout(0.2),
    layers.LSTM(64, activation='relu', return_sequences=False),
    layers.Dropout(0.2),
    layers.RepeatVector(input_shape[0]),
    layers.LSTM(64, activation='relu', return_sequences=True),
    layers.Dropout(0.2),
    layers.LSTM(128, activation='relu', return_sequences=True),
    layers.Dropout(0.2),
    layers.TimeDistributed(layers.Dense(input_shape[1]))
])

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse")

MODEL_PATH = "models/lstm_autoencoder.keras"

if os.path.exists(MODEL_PATH):
    model = keras.models.load_model(MODEL_PATH)
else:
    user_input = input("Train model? (yes/no): ")
    
    if user_input.lower().strip() == 'yes':
        start_time = time.time()
        
        history = model.fit(
            x_train, x_train,
            epochs=50,
            batch_size=32,
            validation_split=0.1,
            callbacks=[
                keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, mode="min", restore_best_weights=True),
                keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=0.00001)
            ],
            verbose=1
        )
        
        training_duration = (time.time() - start_time) / 60
        
        os.makedirs("models", exist_ok=True)
        model.save(MODEL_PATH)
        
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training')
        plt.plot(history.history['val_loss'], label='Validation')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training')
        plt.plot(history.history['val_loss'], label='Validation')

threshold = np.percentile(train_mae_loss, 99)

x_test_pred = model.predict(x_test, verbose=0)
test_mae_loss = np.mean(np.abs(x_test_pred - x_test), axis=(1, 2)).reshape((-1))

anomalies = test_mae_loss > threshold
num_anomalies = np.sum(anomalies)

fig, axes = plt.subplots(3, 1, figsize=(15, 10))
fig.suptitle('Anomaly Detection Results', fontsize=16, fontweight='bold')

ax = axes[0]
ax.hist(train_mae_loss, bins=50, alpha=0.6, label='Training', color='green')
ax.hist(test_mae_loss, bins=50, alpha=0.6, label='Test', color='blue')
ax.axvline(threshold, color='red', linestyle='--', linewidth=2, label='Threshold')
ax.set_xlabel('Reconstruction Error')
ax.set_ylabel('Frequency')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(test_mae_loss, alpha=0.7)
ax.axhline(threshold, color='red', linestyle='--', linewidth=2)
ax.fill_between(range(len(test_mae_loss)), 0, threshold, alpha=0.2, color='green')
ax.fill_between(range(len(test_mae_loss)), threshold, test_mae_loss.max(), 
                where=(test_mae_loss > threshold), alpha=0.3, color='red')
ax.set_xlabel('Sequence Index')
ax.set_ylabel('Reconstruction Error')
ax.grid(True, alpha=0.3)

ax = axes[2]
normal_idx = np.where(~anomalies)[0]
anomaly_idx = np.where(anomalies)[0]
ax.scatter(normal_idx, test_mae_loss[normal_idx], c='green', s=10, alpha=0.5, label='Normal')
ax.scatter(anomaly_idx, test_mae_loss[anomaly_idx], c='red', s=20, alpha=0.8, label='Anomaly')
import numpy as np
import keras
from keras import layers
from matplotlib import pyplot as plt
import os
import time

x_train = np.load('processed_data/X_train_combined.npy')
x_test = np.load('processed_data/X_test_real.npy')
params = np.load('processed_data/normalization_params.npz')

data_min = params['data_min']
data_max = params['data_max']
TIME_STEP = int(params['window_size'])

input_shape = (x_train.shape[1], x_train.shape[2])

model = keras.Sequential([
    layers.Input(shape=input_shape),
    layers.LSTM(128, activation='relu', return_sequences=True),
    layers.Dropout(0.2),
    layers.LSTM(64, activation='relu', return_sequences=False),
    layers.Dropout(0.2),
    layers.RepeatVector(input_shape[0]),
    layers.LSTM(64, activation='relu', return_sequences=True),
    layers.Dropout(0.2),
    layers.LSTM(128, activation='relu', return_sequences=True),
    layers.Dropout(0.2),
    layers.TimeDistributed(layers.Dense(input_shape[1]))
])

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse")

MODEL_PATH = "models/lstm_autoencoder.keras"

if os.path.exists(MODEL_PATH):
    model = keras.models.load_model(MODEL_PATH)
else:
    user_input = input("Train model? (yes/no): ")
    
    if user_input.lower().strip() == 'yes':
        start_time = time.time()
        
        history = model.fit(
            x_train, x_train,
            epochs=50,
            batch_size=32,
            validation_split=0.1,
            callbacks=[
                keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, mode="min", restore_best_weights=True),
                keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=0.00001)
            ],
            verbose=1
        )
        
        training_duration = (time.time() - start_time) / 60
        
        os.makedirs("models", exist_ok=True)
        model.save(MODEL_PATH)
        
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training')
        plt.plot(history.history['val_loss'], label='Validation')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training')
        plt.plot(history.history['val_loss'], label='Validation')

threshold = np.percentile(train_mae_loss, 99)

x_test_pred = model.predict(x_test, verbose=0)
test_mae_loss = np.mean(np.abs(x_test_pred - x_test), axis=(1, 2)).reshape((-1))

anomalies = test_mae_loss > threshold
num_anomalies = np.sum(anomalies)

fig, axes = plt.subplots(3, 1, figsize=(15, 10))
fig.suptitle('Anomaly Detection Results', fontsize=16, fontweight='bold')

ax = axes[0]
ax.hist(train_mae_loss, bins=50, alpha=0.6, label='Training', color='green')
ax.hist(test_mae_loss, bins=50, alpha=0.6, label='Test', color='blue')
ax.axvline(threshold, color='red', linestyle='--', linewidth=2, label='Threshold')
ax.set_xlabel('Reconstruction Error')
ax.set_ylabel('Frequency')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(test_mae_loss, alpha=0.7)
ax.axhline(threshold, color='red', linestyle='--', linewidth=2)
ax.fill_between(range(len(test_mae_loss)), 0, threshold, alpha=0.2, color='green')
ax.fill_between(range(len(test_mae_loss)), threshold, test_mae_loss.max(), 
                where=(test_mae_loss > threshold), alpha=0.3, color='red')
ax.set_xlabel('Sequence Index')
ax.set_ylabel('Reconstruction Error')
ax.grid(True, alpha=0.3)

ax = axes[2]
normal_idx = np.where(~anomalies)[0]
anomaly_idx = np.where(anomalies)[0]
ax.scatter(normal_idx, test_mae_loss[normal_idx], c='green', s=10, alpha=0.5, label='Normal')
ax.scatter(anomaly_idx, test_mae_loss[anomaly_idx], c='red', s=20, alpha=0.8, label='Anomaly')
ax.axhline(threshold, color='red', linestyle='--', linewidth=2)
ax.set_xlabel('Sequence Index')
ax.set_ylabel('Reconstruction Error')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('models/visualizations/anomaly_detection_results.png', dpi=150)
plt.show()

print(f"Anomalies: {num_anomalies} ({num_anomalies/len(test_mae_loss)*100:.2f}%)")