# -*- coding: utf-8 -*-
import tensorflow as tf

tf.enable_eager_execution()

import numpy as np
import os
import time

text = open('texto_entrenamiento2.txt').read()
print ('Longitud del texto: {} caracteres'.format(len(text)))
# Ver a los primeros 250 caracteres del texto.
print(text[:250])
print("============")
# Los caracteres únicos en el archivo.
vocab = sorted(set(text))
print ('{} caracteres únicos'.format(len(vocab)))
print("2===========")
# Creating a mapping from unique characters to indices
char2idx = {u: i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)
text_as_int = np.array([char2idx[c] for c in text])
print('{')
for char, _ in zip(char2idx, range(20)):
    print('  {:4s}: {:3d},'.format(repr(char), char2idx[char]))
print('  ...\n}')
print("3===========")
# Show how the first 13 characters from the text are mapped to integers
print ('{} ---- caracteres asignados a int ---- > {}'.format(repr(text[:13]), text_as_int[:13]))

print("4===========")
# La frase de longitud máxima que queremos para una entrada única en caracteres
seq_length = 100
examples_per_epoch = len(text) // seq_length

# Crear ejemplos / objetivos de entrenamiento
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

for i in char_dataset.take(5):
    print(idx2char[i.numpy()])

print("5===========")
sequences = char_dataset.batch(seq_length + 1, drop_remainder=True)
for item in sequences.take(5):
    print(repr(''.join(idx2char[item.numpy()])))

print("6===========")


def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text


dataset = sequences.map(split_input_target)

for input_example, target_example in dataset.take(1):
    print ('Datos entrada: ', repr(''.join(idx2char[input_example.numpy()])))
    print ('Datos destino:', repr(''.join(idx2char[target_example.numpy()])))

print("7===========")
for i, (input_idx, target_idx) in enumerate(zip(input_example[:5], target_example[:5])):
    print("Paso {:4d}".format(i))
    print("  entrada: {} ({:s})".format(input_idx, repr(idx2char[input_idx])))
    print("  expectativa salida: {} ({:s})".format(target_idx, repr(idx2char[target_idx])))

print("8=========== Empaquetar en lotes")
BATCH_SIZE = 64
steps_per_epoch = examples_per_epoch // BATCH_SIZE

# Tamaño del búfer para barajar el conjunto de datos
# (Los datos de TF están diseñados para funcionar con secuencias posiblemente infinitas,
# para que no intente barajar toda la secuencia en la memoria. En lugar,
# mantiene un búfer en el que baraja elementos).
BUFFER_SIZE = 10000

dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

dataset
# Longitud del vocabulario en caracteres.

vocab_size = len(vocab)

# The embedding dimension
embedding_dim = 256

# Number of RNN units
rnn_units = 1024

if tf.test.is_gpu_available():
    rnn = tf.keras.layers.CuDNNGRU
else:
    import functools

    rnn = functools.partial(
        tf.keras.layers.GRU, recurrent_activation='sigmoid')


def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                  batch_input_shape=[batch_size, None]),
        rnn(rnn_units,
            return_sequences=True,
            recurrent_initializer='glorot_uniform',
            stateful=True),
        tf.keras.layers.Dense(vocab_size)
    ])
    return model


model = build_model(
    vocab_size=len(vocab),
    embedding_dim=embedding_dim,
    rnn_units=rnn_units,
    batch_size=BATCH_SIZE)

print("9=========== Tamanio del size")

for input_example_batch, target_example_batch in dataset.take(1):
    example_batch_predictions = model(input_example_batch)
    print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")
