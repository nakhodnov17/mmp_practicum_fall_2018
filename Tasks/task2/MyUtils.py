import re

import nltk
import numpy as np
import pandas as pd
import progressbar
from pymystem3 import Mystem


def get_data(path, way=False):
    data = pd.read_json(path)
    if way:
        data.loc[data['sentiment'] == 'negative', 'sentiment'] = 0
        data.loc[data['sentiment'] == 'neutral', 'sentiment'] = 1
        data.loc[data['sentiment'] == 'positive', 'sentiment'] = 2

        x_data, y_data = data['text'], data['sentiment']
    else:
        data.loc[data['sentiment'] == 'negative', 'sentiment'] = -1
        data.loc[data['sentiment'] == 'positive', 'sentiment'] = 1

        x_data, y_data = (
            data['text'][data['sentiment'] != 'neutral'],
            data['sentiment'][data['sentiment'] != 'neutral']
        )

    return x_data.values, y_data.values.astype(np.int)


def uniform_strings(data):
    for idx in range(data.shape[0]):
        data[idx] = re.sub('[^\w]+', ' ', data[idx].strip()).lower()
    return data


def lemmatize(data):
    lemmatizer = Mystem()
    for idx in progressbar.progressbar(range(data.shape[0])):
        data[idx] = ''.join(
            [
                word for word in lemmatizer.lemmatize(data[idx])
                if word not in nltk.corpus.stopwords.words('russian')
            ]
        )
    return data
