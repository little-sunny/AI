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

word_to_index, index_to_word, word_to_vec_map = tfun.read_glove_vecs('MachineLearning/glove.6B.50d.txt')
print("word_to_index is OK")

def  LSTMs(request):
    filename = os.listdir('data')[0]
    file = 'data/' + filename
    sample = pd.read_csv(file, names=['x', 'y'])
    X_sample = sample['x']
    Y_sample = sample['y']

    sample_num = int(len(X_sample)*0.75)
    print(sample_num)
    X_train = X_sample[0:sample_num-1]
    Y_train = Y_sample[0:sample_num-1]
    X_test = pd.core.series.Series(list(X_sample[sample_num:len(X_sample)]))
    Y_test = pd.core.series.Series(list(Y_sample[sample_num:len(Y_sample)]))
    #print(X_train)
    #print(list(X_test))
    #print(pd.core.series.Series(list(X_test)))
    #print(Y_test)

    print(type(X_train))

    maxLen= len(max(X_train, key=len).split())
    print(maxLen)

    index = 3
    print(X_train[index], tfun.label_to_emoji(Y_train[index]))

    #word_to_index, index_to_word, word_to_vec_map = tfun.read_glove_vecs('MachineLearning/glove.6B.50d.txt') # load Word Embeding
    print(len(word_to_index))
    print(len(word_to_vec_map['machine']))

    keras.backend.clear_session()

    model = tfun.Emojify_V2((maxLen,), word_to_vec_map, word_to_index)
    print(model.summary())

    # 模型概要定义模
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    X_train_indices = tfun.sentences_to_indices(X_train, word_to_index, maxLen)
    Y_train_oh = tfun.convert_to_one_hot(Y_train, C=5)

    #训练模型
    model.fit(X_train_indices, Y_train_oh, epochs=100, batch_size=32, shuffle=True)
    model.save('model/ai_model.h5')
    print('save model')


    #用test数据评价模型
    X_test_indices = tfun.sentences_to_indices(X_test, word_to_index, max_len=maxLen)
    print(X_test_indices)
    Y_test_oh = tfun.convert_to_one_hot(Y_test, C=5)
    print(Y_test_oh)
    loss, acc = model.evaluate(X_test_indices, Y_test_oh)
    print()
    print("Test accuracy = ", acc)
    del model
    gc.collect()
    return render_to_response('prediction.html', RequestContext(request, {'accuracy': acc}))


def Algorithm2(request):
    return HttpResponse('No Algorithm2')


def Algorithm3(request):
    return HttpResponse('No Algorithm3')


def Algorithm4(request):
    return HttpResponse('No Algorithm4')


def NoSelection(request):
    return HttpResponse('please select a algorithm!')
