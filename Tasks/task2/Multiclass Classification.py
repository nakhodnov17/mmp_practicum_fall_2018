import time
import pickle
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from optimization import SGDClassifier
from multiclass import MulticlassStrategy


# load data from json
# x_train, y_train = get_data('./data/news_train.json', True)
# x_test, y_test = get_data('./data/news_test.json', True)

# load lemmatized data
with open('./data/train_full_lemmatized.pkz', 'rb') as file:
    x_train, y_train = pickle.load(file)
with open('./data/test_full_lemmatized.pkz', 'rb') as file:
    x_test, y_test = pickle.load(file)

# prepare original data (to downcase, drop non alphanumerical symbols)
# x_train, x_test = uniform_strings(x_train), uniform_strings(x_test)

# lemmatize data
# x_train, x_test = lemmatize(x_train), lemmatize(x_test)

# save processed data if needed
# with open('./data/train.pkz', 'wb') as file:
#     pickle.dump((x_train, y_train), file)
# with open('./data/test.pkz', 'wb') as file:
#     pickle.dump((x_test, y_test), file)

# create featurizer class
vectorizer = TfidfVectorizer(min_df=2)
# vectorizer = CountVectorizer(min_df=2)
# extract features
vectorizer.fit(np.concatenate([x_train, x_test]))
x_train = vectorizer.transform(x_train)
x_test = vectorizer.transform(x_test)

st = time.time()
# create model
# model = MulticlassStrategy(GDClassifier, 'one_vs_all', loss_function='binary_logistic', max_iter=100, tolerance=1e-8, step_alpha=0.1, step_beta=0.1)
# model = MulticlassStrategy(GDClassifier, 'all_vs_all', loss_function='binary_logistic', max_iter=100, tolerance=1e-8)
model = SGDClassifier(loss_function='multiclass_logistic', max_iter=5, step_alpha=0.1, step_beta=0.1, batch_size=1)

# fit model and get scores
model.fit(x_train, y_train, x_test=x_test, y_test=y_test)
predicts_test = model.predict(x_test)
predicts_train = model.predict(x_train)
print(np.mean(predicts_test == y_test))
print(np.mean(predicts_train == y_train))

print(time.time() - st)
print(model.__class__.__name__)
