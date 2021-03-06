import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from optimization import GDClassifier, SGDClassifier
from MyUtils import get_data, uniform_strings, lemmatize


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


model_1 = GDClassifier(loss_function='binary_logistic', max_iter=100, tolerance=1e-100, step_alpha=0.1, step_beta=0.1)
model_2 = SGDClassifier(loss_function='binary_logistic', max_iter=100, tolerance=1e-100, step_alpha=0.1, step_beta=0.1)

history_1 = model_1.fit(x_train, y_train, None, trace=True, x_test=x_test, y_test=y_test)
history_2 = model_2.fit(x_train, y_train, None, trace=True, x_test=x_test, y_test=y_test)

losses_train = []
losses_test = []
accs_train = []
accs_test = []

losses_train.append(history_1['func_train'])
losses_test.append(history_1['func_test'])
accs_train.append(history_1['accuracy_train'])
accs_test.append(history_1['accuracy_test'])
losses_train.append(history_2['func_train'])
losses_test.append(history_2['func_test'])
accs_train.append(history_2['accuracy_train'])
accs_test.append(history_2['accuracy_test'])

fig, axes = plt.subplots(2, 2, figsize=(16, 6))

bias = 0
axes[0][0].plot(losses_train[0][bias:], label='GD')
axes[0][1].plot(losses_test[0][bias:], label='GD')
axes[1][0].plot(accs_train[0][bias:], label='GD')
axes[1][1].plot(accs_test[0][bias:], label='GD')

axes[0][0].plot(losses_train[1][bias:], label='SGD')
axes[0][1].plot(losses_test[1][bias:], label='SGD')
axes[1][0].plot(accs_train[1][bias:], label='SGD')
axes[1][1].plot(accs_test[1][bias:], label='SGD')

axes[0][0].set_xlabel('Iteration'), axes[0][0].set_ylabel('Loss'), axes[0][0].set_title('Train Loss vs Iteration')
axes[0][1].set_xlabel('Iteration'), axes[0][1].set_ylabel('Loss'), axes[0][1].set_title('Test Loss vs Iteration')
axes[1][0].set_xlabel('Iteration'), axes[1][0].set_ylabel('Accuracy'), axes[1][0].set_title('Train Accuracy vs Iteration')
axes[1][1].set_xlabel('Iteration'), axes[1][1].set_ylabel('Accuracy'), axes[1][1].set_title('Test Accuracy vs Iteration')

axes[0][0].legend()
axes[0][1].legend()
axes[1][0].legend()
axes[1][1].legend()

axes[0][0].grid()
axes[0][1].grid()
axes[1][0].grid()
axes[1][1].grid()

left_bias = 3
axes[0][0].set_xlim(left=left_bias)
axes[0][1].set_xlim(left=left_bias)
axes[1][0].set_xlim(left=left_bias)
axes[1][1].set_xlim(left=left_bias)

fig.tight_layout()
fig.subplots_adjust(top=1.)
fig.savefig('./Report/Plots/GD_vs_SGD.png')
plt.show()
