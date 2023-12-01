import keras.datasets.cifar100 as dataset
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, Dropout, Flatten, MaxPooling2D, BatchNormalization, Activation, Add
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix

CLASSES, EPOCHS, LR = 100, 50, 0.01


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

def add_identity_block(tensor, filters):
    tensor_a, tensor_b = tensor, tensor

    tensor_a = Conv2D(filters[0], (1, 1), data_format="channels_last", padding='valid')(tensor_a)
    tensor_a = BatchNormalization(axis=3)(tensor_a)
    tensor_a = Activation('relu')(tensor_a)

    tensor_a = Conv2D(filters[1], (3, 3), data_format="channels_last", padding='same')(tensor_a)
    tensor_a = BatchNormalization(axis=3)(tensor_a)
    tensor_a = Activation('relu')(tensor_a)

    tensor_a = Conv2D(filters[2], (1, 1), data_format="channels_last", padding='valid')(tensor_a)
    tensor_a = BatchNormalization(axis=3)(tensor_a)

    tensor = Add()([tensor_a, tensor_b])
    tensor = Activation('relu')(tensor)

    return tensor

def add_convolution_block(tensor, filters):
    tensor_a, tensor_b = tensor, tensor

    tensor_a = Conv2D(filters[0], (1, 1), data_format="channels_last", padding='valid')(tensor_a)
    tensor_a = BatchNormalization(axis=3)(tensor_a)
    tensor_a = Activation('relu')(tensor_a)

    tensor_a = Conv2D(filters[1], (3, 3), data_format="channels_last", padding='same')(tensor_a)
    tensor_a = BatchNormalization(axis=3)(tensor_a)
    tensor_a = Activation('relu')(tensor_a)

    tensor_a = Conv2D(filters[2], (1, 1), data_format="channels_last", padding='valid')(tensor_a)
    tensor_a = BatchNormalization(axis=3)(tensor_a)

    tensor_b = Conv2D(filters[2], (1, 1), data_format="channels_last", padding='valid')(tensor_b)
    tensor_b = BatchNormalization(axis=3)(tensor_b)

    tensor = Add()([tensor_a, tensor_b])
    tensor = Activation('relu')(tensor)

    return tensor


(x_train, y_train), (x_test, y_test) = dataset.load_data(label_mode="fine")
assert x_train.shape == (50000, 32, 32, 3)
assert x_test.shape == (10000, 32, 32, 3)
assert y_train.shape == (50000, 1)
assert y_test.shape == (10000, 1)

#idg = ImageDataGenerator(rotation_range=90, width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.2)
x_train = x_train / 255
x_test = x_test / 255
y_train = to_categorical(y_train)
#train_data = idg.flow(x_train, y_train)

model = Model()
x_in = Input([32, 32, 3])
x = x_in
x = add_convolution_block(x, [32, 32, 128])
x = add_identity_block(x, [32, 32, 128])
x = add_identity_block(x, [32, 32, 128])
x = MaxPooling2D()(x)
x = Dropout(0.25)(x)
x = add_convolution_block(x, [32, 32, 128])
x = add_identity_block(x, [32, 32, 128])
x = add_identity_block(x, [32, 32, 128])
x = add_identity_block(x, [32, 32, 128])
x = MaxPooling2D()(x)
x = Dropout(0.25)(x)
x = add_convolution_block(x, [64, 64, 256])
x = add_identity_block(x, [64, 64, 256])
x = add_identity_block(x, [64, 64, 256])
x = add_identity_block(x, [64, 64, 256])
x = MaxPooling2D()(x)
x = Dropout(0.25)(x)
x = add_convolution_block(x, [128, 128, 512])
x = add_identity_block(x, [128, 128, 512])
x = add_identity_block(x, [128, 128, 512])
x = add_identity_block(x, [128, 128, 512])
x = MaxPooling2D()(x)
x = Flatten()(x)
x = Dense(CLASSES, activation='softmax')(x)

model = Model(inputs=x_in, outputs=x, name='resnet')

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
