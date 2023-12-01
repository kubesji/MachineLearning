import keras.datasets.reuters as dataset
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Embedding, Input, Flatten, GlobalAveragePooling1D
from keras_nlp.layers import TokenAndPositionEmbedding, TransformerEncoder, TransformerDecoder
from keras.optimizers import Adam
from keras.losses import SparseCategoricalCrossentropy
from keras.preprocessing import sequence
from keras.utils import to_categorical
import tensorflow as tf
from utils import print_history


WORDS, MAX_LENGTH = 5000, 400
EPOCHS, LR, BATCH_SIZE = 4, 0.01, 128
EMBED_DIM, MULTI_HEAD_ATTENTION_DIM, NUM_HEADS = 16, 4, 2

(x_train, y_train), (x_test, y_test) = dataset.load_data(num_words=WORDS)
CLASSES = max(y_train) + 1
x_train = sequence.pad_sequences(x_train, maxlen=MAX_LENGTH)
y_train = to_categorical(y_train)
x_test = sequence.pad_sequences(x_test, maxlen=MAX_LENGTH)
y_test = to_categorical(y_test)

model = Sequential()
model.add(Input(shape=(MAX_LENGTH,), dtype=tf.int32))
model.add(TokenAndPositionEmbedding(vocabulary_size=WORDS, sequence_length=MAX_LENGTH, embedding_dim=EMBED_DIM,
                                    mask_zero=True))
model.add(TransformerEncoder(num_heads=NUM_HEADS, intermediate_dim=MULTI_HEAD_ATTENTION_DIM))
model.add(GlobalAveragePooling1D())
model.add(Dropout(0.35))
model.add(Dense(CLASSES, activation='softmax'))

adam = Adam(learning_rate=LR)
model.compile(optimizer=adam, loss="categorical_crossentropy", metrics=['accuracy'])
model.summary()

history = model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, shuffle=True, validation_split=0.05)
print_history(history)

_, acc = model.evaluate(x_test, y_test, batch_size=128)

print(f"Accuracy: {acc*100:.2f} %")
