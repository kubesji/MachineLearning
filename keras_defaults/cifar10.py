import keras.datasets.cifar10 as dataset
from keras.models import Sequential
from keras.layers import Input, Dense, Conv2D, Dropout, Flatten, MaxPooling2D, BatchNormalization
from keras.optimizers import Adam
from keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score

CLASSES, EPOCHS, LR = 10, 50, 0.005


def print_history(h):
    x_axis = range(1, EPOCHS + 1)
    fig, ax1 = plt.subplots()

    colour = 'tab:red'
    ax1.set_xlabel('epochs')
    ax1.set_ylabel('loss', color=colour)
    ax1.plot(x_axis, h.history['loss'], color=colour, label="train loss")
    ax1.plot(x_axis, h.history['val_loss'], color='tab:orange', label="validation loss")
    ax1.tick_params(axis='y', labelcolor=colour)
    ax1.legend()

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    colour = 'tab:green'
    ax2.set_ylabel('accuracy', color=colour)  # we already handled the x-label with ax1
    ax2.plot(x_axis, h.history['accuracy'], color=colour, label="train accuracy")
    ax2.plot(x_axis, h.history['val_accuracy'], color='tab:blue', label="validation accuracy")
    ax2.tick_params(axis='y', labelcolor=colour)
    ax2.legend()

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()


(x_train, y_train), (x_test, y_test) = dataset.load_data()
assert x_train.shape == (50000, 32, 32, 3)
assert x_test.shape == (10000, 32, 32, 3)
assert y_train.shape == (50000, 1)
assert y_test.shape == (10000, 1)

x_train = x_train / 255
x_test = x_test / 255
y_train = to_categorical(y_train)

model = Sequential()
model.add(Input(shape=(32, 32, 3)))
model.add(Conv2D(64, (3, 3), data_format="channels_last", activation="relu"))
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3), data_format="channels_last", activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.35))
model.add(Conv2D(64, (3, 3), data_format="channels_last", activation="relu"))
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3), data_format="channels_last", activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.35))
model.add(Flatten())
model.add(Dense(1024, activation="relu"))
model.add(BatchNormalization())
model.add(Dropout(0.35))
model.add(Dense(512, activation="relu"))
model.add(BatchNormalization())
model.add(Dropout(0.35))
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
