# The project is based on Tensorflow's Text Generation with RNN tutorial
# Copyright Petros Demetrakopoulos 2020
import tensorflow as tf
import numpy as np
import os
import time

stopChars = [',','(',')','.','-','[',']','"']
# preprocessing the corpus by converting all letters to lowercase, 
# replacing blank lines with blank string and removing special characters
def preprocessText(text):
  text = text.replace('\n', ' ').replace('\t','')
  processedText = text.lower()
  for char in stopChars:
    processedText = processedText.replace(char,'')
  return processedText

def corpusToList(corpus):
  corpusList = [w for w in corpus.split(' ')] 
  corpusList = [i for i in corpusList if i] #removing empty strings from list
  return corpusList

corpus_path = './beatles_lyrics.txt'
text = open(corpus_path, 'rb').read().decode(encoding='utf-8')
text = preprocessText(text)
corpus_words = corpusToList(text) 
map(str.strip, corpus_words) #trim words

vocab = sorted(set(corpus_words))
print('Corpus length (in words):', len(corpus_words))
print('Unique words in corpus: {}'.format(len(vocab)))
word2idx = {u: i for i, u in enumerate(vocab)}
idx2words = np.array(vocab)
word_as_int = np.array([word2idx[c] for c in corpus_words])
# The maximum length sentence we want for a single input in words
seqLength = 10
examples_per_epoch = len(corpus_words)//(seqLength + 1)

# Create training examples / targets
wordDataset = tf.data.Dataset.from_tensor_slices(word_as_int)

sequencesOfWords = wordDataset.batch(seqLength + 1, drop_remainder=True) # generating batches of 10 words each

def split_input_target(chunk):
  input_text = chunk[:-1]
  target_text = chunk[1:]
  return input_text, target_text

dataset = sequencesOfWords.map(split_input_target)

BATCH_SIZE = 64 # each batch contains 64 sequences. Each sequence contains 10 words (seqLength)
BUFFER_SIZE = 100 # Number of batches that will be processed concurrently

dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

# Length of the vocabulary in words
vocab_size = len(vocab)
# The embedding dimension
embedding_dim = 256
# Number of GRU units
rnn_units = 1024

def createModel(vocab_size, embedding_dim, rnn_units, batch_size):
  model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim,
                              batch_input_shape=[batch_size, None]),
    tf.keras.layers.GRU(rnn_units,
                        return_sequences=True,
                        stateful=True,
                        recurrent_initializer='glorot_uniform'),
    tf.keras.layers.Dense(vocab_size)
  ])
  return model

model = createModel(vocab_size = len(vocab), embedding_dim=embedding_dim, rnn_units=rnn_units, batch_size=BATCH_SIZE)

model.summary()

def loss(labels, logits):
  return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

model.compile(optimizer='adam', loss=loss)
# We save training checkpoints to facilitate training procedure
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)

EPOCHS = 20
history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])

tf.train.latest_checkpoint(checkpoint_dir)
model = createModel(len(vocab), embedding_dim, rnn_units, batch_size=1)

model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

model.build(tf.TensorShape([1, None]))
model.summary()

def generateLyrics(model, startString, temp):
  print("---- Generating lyrics starting with '" + startString + "' ----")
  # Number of words to generate
  num_generate = 30

  # Converting our start string to numbers (vectorizing)
  start_string_list =  [w for w in startString.split(' ')]
  input_eval = [word2idx[s] for s in start_string_list]
  input_eval = tf.expand_dims(input_eval, 0)

  text_generated = []

  model.reset_states()
  for i in range(num_generate):
      predictions = model(input_eval)
      # remove the batch dimension
      predictions = tf.squeeze(predictions, 0)

      # temp represent how 'conservative' the predictions are. 
      # Lower temp leads to more predictable (or correct) text
      predictions = predictions / temp 
      predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

      # We pass the predicted word as the next input to the model
      # along with the previous hidden state
      input_eval = tf.expand_dims([predicted_id], 0)
      text_generated.append(' ' + idx2words[predicted_id])

  return (startString + ''.join(text_generated))

#save trained model for future use (so we do not have to train it every time we want to generate text)
model.save('saved_model.h5') 
print("Example:")
print(generateLyrics(model, startString=u"love", temp=0.6))
while (True):
  print('Enter start string:')
  input_str = input().lower()
  print('Enter temp:')
  temp = float(input())
  print(generateLyrics(model, startString=input_str, temp=temp))
