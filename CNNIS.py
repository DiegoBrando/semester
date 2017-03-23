

from __future__ import print_function
import os
from os.path import join, exists, split
from gensim.models import word2vec
import numpy as np
import data_helpers
from keras.models import Sequential, Model
from keras.layers import Activation, Dense, Dropout, Embedding, Flatten, Input, Merge, Convolution1D, MaxPooling1D
from keras.optimizers import SGD


def padsent(vocab,sequence_length,test_sent):# returned format is np.asarray
    sent=data_helpers.clean_str(test_sent).split(' ')
    numb_pad=sequence_length-len(sent)
    finalsent=sent+['<PAD/>']*numb_pad
    voclist=np.hstack([vocab[words] for words in finalsent])

    return voclist.reshape(1,-1)


def train_word2vec(sentence_matrix, dicts,
                   vec_di=300, min_word_count=1, context=15):# Dimensions for word vector, context size, dicts is dictory of words str:int
    model_dir = 'model'
    model_name = "{:d}features_{:d}minwords_{:d}context".format(vec_di, min_word_count, context)
    model_name = join(model_dir, model_name)
    if exists(model_name):
        embedding_model = word2vec.Word2Vec.load(model_name)
        
    else:
        
        parathr = 2  # number of threads to run in parallel
        downsampling = 1e-4  # downsampling for fequently used words

        print('Training Word2Vec model...')
        sentences = [[dicts[w] for w in s] for s in sentence_matrix]
        embedding_model = word2vec.Word2Vec(sentences, workers=parathr, \
                                            size=vec_di, min_count=min_word_count, \
                                            window=context, sample=downsampling)
        embedding_model.init_sims(replace=True) 
        
        if not exists(model_dir):
            os.mkdir(model_dir)
        print('Saving Word2Vec model \'%s\'' % split(model_name)[-1])
        embedding_model.save(model_name)

    embedding_weights = [np.array([embedding_model[z] if z in embedding_model \
                                       else np.random.uniform(-0.25, 0.25, embedding_model.vector_size) \
                                   for z in dicts])]
    return embedding_weights



np.random.seed(4)


model_variation = 'CNN-non-static'

#model_variation = 'CNN-static'

#model_variation = 'CNN-rand'


sequence_length = 56
embedding_dim = 20          
filter_sizes = (3, 4)
num_filters = 3
dropout_prob = (0.25, 0.5)
hidden_dims = 100
batch_size = 32
num_epochs = 200
val_split = 0.1
min_word_count = 1                         
contwi =15# context window size
Act="prediction"
weights="model/weight_file"

x, y, vocab, vocab_inv = data_helpers.load_data()

if model_variation=='CNN-non-static':
    embedding_weights = train_word2vec(x, vocab_inv, embedding_dim, min_word_count, contwi)
if model_variation=='CNN-static':
    embedding_weights = train_word2vec(x, vocab_inv, embedding_dim, min_word_count, contwi)
    x = embedding_weights[0][x]
elif model_variation=='CNN-rand':
    embedding_weights = None
else:
    print('No choice selected')
print (model_variation)



data_in = Input(shape=(sequence_length, embedding_dim))
convs = []# convolutional layers in parallel with one input and one output
for filts in filter_sizes:
    conv = Convolution1D(nb_filter=num_filters,
                         filter_length=filts,
                         border_mode='valid',
                         activation='relu',
                         subsample_length=1)(data_in)
    pool = MaxPooling1D(pool_length=2)(conv)
    flatten = Flatten()(pool)
    convs.append(flatten)
    
if len(filter_sizes)>1:
    out = Merge(mode='concat')(convs)
else:
    out = convs[0]

graph = Model(input=data_in, output=out)


model = Sequential()
if model_variation=='CNN-non-static' or model_variation=="CNN-rand":
    model.add(Embedding(len(vocab), embedding_dim, input_length=sequence_length,
                        weights=embedding_weights))
model.add(Dropout(dropout_prob[0], input_shape=(sequence_length, embedding_dim)))
model.add(graph)
model.add(Dense(hidden_dims))
model.add(Dropout(dropout_prob[1]))
model.add(Activation('relu'))
model.add(Dense(1))
model.add(Activation('sigmoid'))
opt = SGD(lr=0.01, momentum=0.80, decay=1e-6, nesterov=True)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

if Act=="prediction":
    if os.path.exists(weights):
        model.load_weights(weights)
else:
    shuffled_data = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffled_data]
    y_shuffled = y[shuffled_data].argmax(axis=1)
    model.fit(x_shuffled, y_shuffled, batch_size=batch_size, nb_epoch=num_epochs, validation_split=val_split, verbose=2)
    model.save_weights(weights, overwrite=True)

testsent="the movie is really sad"
if Act=="predict":
    voclist=padsent(vocab, sequence_length, testsent)
    probab=model.predict(voclist)

    if probab<0.5:
        classif="Neg"
    else:
        classif="Post"
    print(testsent+" predicted to be "+classif)


