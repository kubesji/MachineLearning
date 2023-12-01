import keras.datasets.boston_housing as dataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import LinearSVR
from sklearn.neighbors import KNeighborsRegressor


def fit_and_test(model, x, y, test, scaler):
    model.fit(x, y)
    prediction = model.predict(test)
    return scaler.inverse_transform(prediction.reshape(-1, 1))


(x_train, y_train), (x_test, y_test) = dataset.load_data(path="boston_housing.npz", test_split=0.1, seed=123)

models = {}
models["rfg"] = RandomForestRegressor(n_estimators=200, criterion="squared_error")
models["gbr"] = GradientBoostingRegressor(n_estimators=200, criterion="squared_error")
models["svm"] = LinearSVR()
models["knn"] = KNeighborsRegressor(n_neighbors=3, weights="distance")
model_names = list(models.keys())

#feature_selector = SelectKBest(score_func=f_regression, k=8)
#x_train = feature_selector.fit_transform(x_train, y_train)
#x_test = feature_selector.transform(x_test)

y_train, y_test = y_train.reshape(-1, 1), y_test.reshape(-1, 1)

scalers = {'x': StandardScaler(), 'y': StandardScaler()}
x_train, y_train = scalers['x'].fit_transform(x_train), scalers['y'].fit_transform(y_train)
x_test = scalers['x'].transform(x_test)

offset, width = 0.15, 0.15

fig, ax = plt.subplots(layout='constrained')
x = np.arange(len(y_test))
ax.bar(x, y_test.reshape(-1), width, label="true")

for name in model_names:
    prediction = fit_and_test(models[name], x_train, y_train, x_test, scalers["y"])

    ax.bar(x + offset, prediction.reshape(-1), width, label=name)
    offset += width

    diff = [abs((y_t[0] - y_p[0]) / y_t[0]) for y_t, y_p in zip(y_test, prediction)]
    mae = mean_absolute_error(y_true=y_test, y_pred=prediction)
    r2 = r2_score(y_true=y_test, y_pred=prediction)
    print(f"------- {name} -------")
    print(f"Average error: {int(mae * 1000)}$, {sum(diff) / len(diff) * 100:.2f}%")
    print(f"R2 score: {r2:.5f}")

ax.legend()
plt.show()
