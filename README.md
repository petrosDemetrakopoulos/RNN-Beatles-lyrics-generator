# RNN Beatles lyrics generator

The project is implemented in Python using Tensorflow.
It is a word-based Recurrent neural network (RNN) inspired by this [Tensorflow tutorial](https://www.tensorflow.org/tutorials/text/text_generation).
The model is trained in a corpus containing the song titles and the lyrics of many famous songs of Beatles.
So, given a sequence of words from Beatles lyrics, it can predict the next words. 
We could say that it is a model generating Beatle-like lyrics.

# The corpus (aka training dataset)
As we said the corpus (beatles_lyrics.txt) contains the song titles and the lyrics of many Beatles songs in the following format:

```
Song title 1
-----------------
Lyric line 1
Lyric line 2
Lyric line 3
Lyric line 4
etc...

Song title 2
-----------------
Lyric line 1
Lyric line 2
Lyric line 3
Lyric line 4
etc...

etc...
```
I have manually created the corpus with the lyrics I found in [this amazing site](http://toti.eu.com/beatles/)

# Data preprocessing
As almost every machine learning model, training data need a bit of preprocessing before they are ready to be used as a training input for our RNN. 
We preprocess the initial corpus by doing the following tasks:

* Convert all letters to lowercase
* Remove blank lines
* Remove special characters (such as ',' , '(' , ')' , '[' , ']' etc)

The following function performs the preprocessing:
```python
stopChars = [',','(',')','.','-','[',']']
# preprocessing the corpus by converting all letters to lowercase, 
# replacing blank lines with blank string and removing special characters
def preprocessText(text):
  text = text.replace('\n', ' ').replace('\t','')
  processedText = text.lower()
  for char in stopChars:
    processedText = processedText.replace(char,'')
  return processedText
 ```

 After the text preprocessing step, we need to convert it to a list of words.
 This procedure is also known as "Tokenization".

 The following function performs the tokenization:

```python
 def corpusToList(corpus):
  corpusList = [w for w in corpus.split(' ')] 
  corpusList = [i for i in corpusList if i] #removing empty strings from list
  return corpusList
```

Then, we trim each word for leading or trailing spaces / tabs.
```python
map(str.strip, corpus_words) # trim words
```

Now, it is time to find the unique words (aka vocabulary) from which our dataset is composed from.
```python
vocab = sorted(set(corpus_words))
```

In order to train our model, we need to represent words with numbers. So we map a specific number to each unique word of our corpus and vice versa by creating the following lookup tables. Then we represent the whole corpus as a list of numbers.

```python
print('Corpus length (in words):', len(corpus_words))
print('Unique words in corpus: {}'.format(len(vocab)))
word2idx = {u: i for i, u in enumerate(vocab)}
idx2words = np.array(vocab)
word_as_int = np.array([word2idx[c] for c in corpus_words])
```
