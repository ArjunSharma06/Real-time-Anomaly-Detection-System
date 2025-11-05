import pandas as pd
import numpy as np
import keras
from keras import layers
from matplotlib import pyplot as plt
#imports done

df_smallnoise = pd.read_csv('anamoly/normal data/artificialNoAnomaly/artificialNoAnomaly/art_daily_small_noise.csv', parse_dates=['timestamp'], index_col='timestamp')
df_jumpsup= pd.read_csv('anamoly/normal data/artificialWithAnomaly/artificialWithAnomaly/art_daily_jumpsup.csv', parse_dates=['timestamp'], index_col='timestamp')
# print(df_smallnoise.head())
# print(df_jumpsup.head())

#trying to initially plot
figure, axes = plt.subplots()
df_smallnoise.plot(ax=axes, legend=False)
# plt.show()

#training
train_mean= df_smallnoise.mean()
train_std= df_smallnoise.std()
df_trainingvalue= (df_smallnoise-train_mean)/train_std
print(len(df_trainingvalue))

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

# training za model
history = model.fit( x_train, x_train, epochs=50, batch_size=128, validation_split=0.1, callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, mode="min")])
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.legend()
plt.show()