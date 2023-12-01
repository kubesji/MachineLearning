import keras.datasets.imdb as dataset
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Embedding
from keras.optimizers import Adam
from keras.preprocessing import sequence
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from utils import print_history

WORDS, MAX_LENGTH, EPOCHS, LR = 15000, 200, 10, 0.01

(x_train, y_train), (x_test, y_test) = dataset.load_data(num_words=WORDS)
x_train = sequence.pad_sequences(x_train, maxlen=MAX_LENGTH)
x_test = sequence.pad_sequences(x_test, maxlen=MAX_LENGTH)

model = Sequential()
model.add(Embedding(WORDS, 16))
model.add(LSTM(8, activation='tanh', recurrent_regularizer='l2', kernel_regularizer='l2', bias_regularizer='l2',
               activity_regularizer='l2'))
model.add(Dropout(0.35))
model.add(Dense(1, activation='sigmoid'))

adam = Adam(learning_rate=LR, clipnorm=1.0)
model.compile(optimizer=adam, loss="binary_crossentropy", metrics=['accuracy'])
model.summary()

history = model.fit(x_train, y_train, epochs=EPOCHS, batch_size=128, shuffle=True, validation_split=0.05)
print_history(history)

_, acc = model.evaluate(x_test, y_test, batch_size=128)

print(f"Accuracy: {acc}")
