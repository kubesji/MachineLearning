import keras.datasets.reuters as dataset
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Embedding
from keras.optimizers import Adam
from keras.preprocessing import sequence
from keras.utils import to_categorical
from utils import print_history

WORDS, MAX_LENGTH, CLASSES = 5000, 200, -1
EPOCHS, LR = 20, 0.01

(x_train, y_train), (x_test, y_test) = dataset.load_data(num_words=WORDS)
CLASSES = max(y_train) + 1
x_train = sequence.pad_sequences(x_train, maxlen=MAX_LENGTH)
y_train = to_categorical(y_train)
x_test = sequence.pad_sequences(x_test, maxlen=MAX_LENGTH)
y_test = to_categorical(y_test)

model = Sequential()
model.add(Embedding(WORDS + 1, 128, mask_zero=True))
#model.add(LSTM(16, activation='tanh', return_sequences=True))
#model.add(Dropout(0.35))
model.add(LSTM(16, activation='tanh'))
model.add(Dropout(0.35))
model.add(Dense(CLASSES, activation='softmax'))

adam = Adam(learning_rate=LR, clipnorm=1.0)
model.compile(optimizer=adam, loss="categorical_crossentropy", metrics=['accuracy'])
model.summary()

history = model.fit(x_train, y_train, epochs=EPOCHS, batch_size=128, shuffle=True, validation_split=0.05)
print_history(history)

_, acc = model.evaluate(x_test, y_test, batch_size=128)

print(f"Accuracy: {acc*100:.2f} %")
