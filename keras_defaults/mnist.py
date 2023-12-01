import keras.datasets.mnist as dataset
from keras.models import Sequential
from keras.layers import Input, Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from keras.optimizers import Adam
from keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
from utils import print_history

CLASSES, EPOCHS, LR = 10, 20, 0.005

(x_train, y_train), (x_test, y_test) = dataset.load_data()
assert x_train.shape == (60000, 28, 28)
assert x_test.shape == (10000, 28, 28)
assert y_train.shape == (60000,)
assert y_test.shape == (10000,)
classes = list(set(y_test))

x_train = np.reshape(x_train, (60000, 28, 28, 1)) / 255
x_test = np.reshape(x_test, (10000, 28, 28, 1)) / 255
y_train = to_categorical(y_train)
#y_test = to_categorical(y_test)

model = Sequential()
model.add(Input(shape=(28, 28, 1)))
model.add(Conv2D(32, (5, 5), data_format="channels_last"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(1024))
model.add(Dropout(0.1))
model.add(Dense(128))
model.add(Dropout(0.1))
model.add(Dense(CLASSES, activation="softmax"))

adam = Adam(learning_rate=LR)
model.compile(optimizer=adam, loss="categorical_crossentropy", metrics=['accuracy'])
model.summary()

history = model.fit(x_train, y_train, epochs=EPOCHS, batch_size=128, shuffle=True, validation_split=0.05)
print_history(history)

y_prob = model.predict(x_test)
y_classes = y_prob.argmax(axis=-1)

print(f"Accuracy: {accuracy_score(y_test, y_classes)}")
cm = confusion_matrix(y_test, y_classes)
print(cm)
