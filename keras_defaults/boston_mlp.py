import keras.datasets.boston_housing as dataset
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout
from keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.feature_selection import SelectKBest, f_regression


(x_train, y_train), (x_test, y_test) = dataset.load_data(path="boston_housing.npz", test_split=0.1, seed=123)

feature_selector = SelectKBest(score_func=f_regression, k=8)
x_train = feature_selector.fit_transform(x_train, y_train)
x_test = feature_selector.transform(x_test)

y_train, y_test = y_train.reshape(-1, 1), y_test.reshape(-1, 1)

scalers = {'x': StandardScaler(), 'y': StandardScaler()}
x_train, y_train = scalers['x'].fit_transform(x_train), scalers['y'].fit_transform(y_train)
x_test = scalers['x'].transform(x_test)

model = Sequential()
model.add(Input(shape=(x_train.shape[1],)))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(1, activation='linear'))

opt = Adam(lr=0.01)
model.compile(loss="mse", optimizer=opt)

model.fit(x_train, y_train, batch_size=32, shuffle=True, epochs=100)
prediction = model.predict(x_test)
prediction = scalers['y'].inverse_transform(prediction)

offset, width = 0, 0.3
x = np.arange(len(prediction))
fig, ax = plt.subplots(layout='constrained')
ax.bar(x, y_test.reshape(-1), width, label="test")
ax.bar(x + width, prediction.reshape(-1), width, label="prediction")
ax.legend()
plt.show()

diff = [abs((y_t[0] - y_p[0]) / y_t[0]) for y_t, y_p in zip(y_test, prediction)]
mae = mean_absolute_error(y_true=y_test, y_pred=prediction)
r2 = r2_score(y_true=y_test, y_pred=prediction)
print(f"Average error: {int(mae * 1000)}$, {sum(diff) / len(diff) * 100:.2f}%")
print(f"R2 score: {r2:.5f}")
