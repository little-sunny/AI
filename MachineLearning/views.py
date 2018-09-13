from django.shortcuts import render, render_to_response
from django.http import HttpResponse
from django.template.context import RequestContext
import numpy as np
np.random.seed(0)
import keras
from keras.models import load_model
np.random.seed(1)
import numpy as np
import pandas as pd
from . import training_function as tfun
import gc
import os
from . import algorithm

word_to_index, index_to_word, word_to_vec_map = tfun.read_glove_vecs('MachineLearning/glove.6B.50d.txt')


def train(request):
    if request.method == 'GET':
        return render(request, 'train.html')
    radio = request.POST.getlist("algorithm")
    print(radio)
    if radio:
        if radio[0] == "LSTM":
            return algorithm.LSTMs(request)
        elif radio[0] == 'Algorithm2':
            return algorithm.Algorithm2(request)
        elif radio[0] == "Algorithm3":
            return algorithm.Algorithm3(request)
        elif radio[0] == "Algorithm4":
            return algorithm.Algorithm4(request)
    else:
        return algorithm.NoSelection(request)


def prediction(request):
    x_test = []
    x_test.append(request.GET['test'])
    print(x_test)
    x_test = np.array(x_test)

    filename = os.listdir('data')[0]
    file = 'data/' + filename
    sample = pd.read_csv(file, names=['x', 'y'])
    X_sample = sample['x']
    sample_num = int(len(X_sample) * 0.75)
    X_train = X_train = X_sample[0:sample_num-1]
    maxLen = len(max(X_train, key=len).split())
    print(maxLen)

    X_test_indices = tfun.sentences_to_indices(x_test, word_to_index, maxLen)

    keras.backend.clear_session()
    print('load model...')
    model = load_model('model/LSTMs.h5', compile=False)
    print('load done.')

    result = x_test[0] + ' ' + tfun.label_to_emoji(np.argmax(model.predict(X_test_indices)))
    print(result)

    del model
    gc.collect()
    return render_to_response('prediction_result.html', RequestContext(request, {'result': result}))


class Predict():
    def __init__(self):
        #keras.backend.clear_session()
        print('load model...')
        model = load_model('model/LSTMs.h5')
        print('load done.')

        word_to_index, index_to_word, word_to_vec_map = tfun.read_glove_vecs('MachineLearning/glove.6B.50d.txt')
        sample_train = pd.read_csv('data/train_emoji_2.csv', names=['x', 'y'])
        X_train = sample_train['x']
        maxLen = len(max(X_train, key=len).split())
        print(maxLen)

    def pre(self, request):
        x_test = []
        x_test.append(request.GET['test'])
        print(x_test)
        x_test = np.array(x_test)
        result = x_test[0] + ' ' + tfun.label_to_emoji(np.argmax(self.model.predict(self.X_test_indices)))
        print(result)
        # keras.backend.clear_session()
        return render_to_response('prediction_result.html', RequestContext(self.request, {'result': result}))





