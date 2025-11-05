import pandas as pd
import numpy as np
import keras
from keras import layers
from matplotlib import pyplot as plt
import os # to avoid model training on every run
#imports done

df_smallnoise = pd.read_csv('anamoly/normal data/artificialNoAnomaly/artificialNoAnomaly/art_daily_small_noise.csv', parse_dates=['timestamp'], index_col='timestamp')
df_jumpsup= pd.read_csv('anamoly/normal data/artificialWithAnomaly/artificialWithAnomaly/art_daily_jumpsup.csv', parse_dates=['timestamp'], index_col='timestamp')
df_check = pd.read_csv('anamoly/normal data/artificialWithAnomaly/artificialWithAnomaly/art_load_balancer_spikes.csv', parse_dates=['timestamp'], index_col='timestamp')
#training
train_mean= df_smallnoise.mean()
train_std= df_smallnoise.std()
df_trainingvalue= (df_smallnoise-train_mean)/train_std
# print(len(df_trainingvalue))

TIME_STEP= 288 # THERE are 288 values in a day
def create_sequence(values, time_steps=TIME_STEP):
    output= []
    for i in range(len(values)- time_steps+ 1):
        output.append(values[i:(i+ time_steps)])
    return np.stack(output)

x_train= create_sequence(df_trainingvalue.values)
model = keras.Sequential(
    [
        layers.Input(shape=(x_train.shape[1], x_train.shape[2])),
        layers.Conv1D(
            filters=32,
            kernel_size=7,
            padding="same",
            strides=2,
            activation="relu",
        ),
        layers.Dropout(rate=0.2),
        layers.Conv1D(
            filters=16,
            kernel_size=7,
            padding="same",
            strides=2,
            activation="relu",
        ),
        layers.Conv1DTranspose(
            filters=16,
            kernel_size=7,
            padding="same",
            strides=2,
            activation="relu",
        ),
        layers.Dropout(rate=0.2),
        layers.Conv1DTranspose(
            filters=32,
            kernel_size=7,
            padding="same",
            strides=2,
            activation="relu",
        ),
        layers.Conv1DTranspose(filters=1, kernel_size=7, padding="same"),
    ]
)
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse")
# model.summary()
MODEL_PATH = "models/conv1d_autoencoder.keras"

if os.path.exists(MODEL_PATH):
    model = keras.models.load_model(MODEL_PATH)
else:
    # training za model
    history = model.fit( x_train, x_train, epochs=50, batch_size=128, validation_split=0.1, callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, mode="min")])
    model.save(MODEL_PATH)
 

#mean absolute error loss
x_train_pred = model.predict(x_train)
train_mae_loss = np.mean(np.abs(x_train_pred - x_train), axis= 1)



#reconstruction loss theshold
threshold = np.percentile(train_mae_loss, 99)
print("Reconstruction error threshold: ", threshold)

df_testvalue= (df_check-train_mean)/train_std
fig, axes= plt.subplots()
df_testvalue.plot(ax=axes, legend=False)


x_test= create_sequence(df_testvalue.values)
print("Test input shape: ", x_test.shape)

x_test_pred= model.predict(x_test)
test_mae_loss= np.mean(np.abs(x_test_pred- x_test), axis= 1)
test_mae_loss= test_mae_loss.reshape((-1))

anomalies = test_mae_loss > threshold
anomalous_data_indices = []
for data_idx in range(TIME_STEP - 1, len(df_testvalue) - TIME_STEP + 1):
    if np.all(anomalies[data_idx - TIME_STEP + 1 : data_idx]):
        anomalous_data_indices.append(data_idx)


df_subset = df_check.iloc[anomalous_data_indices]
fig, ax = plt.subplots()
df_check.plot(legend=False, ax=ax)
df_subset.plot(legend=False, ax=ax, color="g")
plt.show()