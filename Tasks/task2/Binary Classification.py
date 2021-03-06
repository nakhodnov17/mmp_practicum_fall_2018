import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


from multiclass import MulticlassStrategy
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


losses_train = []
losses_test = []
accs_train = []
accs_test = []
times = []

# params = [0.5, 0.3, 0.1, 0.06, 0.02, 0.008]
# param_name = 'alpha'
# param_name = 'beta'

params = [1024, 512, 128, 32, 4, 1]
param_name = 'batch_size'
for param in params:
    model = SGDClassifier(loss_function='binary_logistic', max_iter=100, tolerance=1e-100, batch_size=param, step_alpha=0.1, step_beta=0.1)

    # model = SGDClassifier(loss_function='binary_logistic', max_iter=100, tolerance=1e-100, step_alpha=param)
    # model = SGDClassifier(loss_function='binary_logistic', max_iter=100, tolerance=1e-100, step_beta=param)
    # model = GDClassifier(loss_function='binary_logistic', max_iter=100, tolerance=1e-100, step_alpha=param)
    # model = GDClassifier(loss_function='binary_logistic', max_iter=100, tolerance=1e-100, step_beta=param)
    history = model.fit(x_train, y_train, None, trace=True, x_test=x_test, y_test=y_test)
    losses_train.append(history['func_train'])
    losses_test.append(history['func_test'])
    accs_train.append(history['accuracy_train'])
    accs_test.append(history['accuracy_test'])
    times.append(history['time'])


bias = 0
fig, axes = plt.subplots(4, 2, figsize=(16, 14))

for idx in range(len(params)):
    axes[0][0].plot(losses_train[idx][bias:], label=param_name + '={0}'.format(str(params[idx])))
    axes[0][1].plot(losses_test[idx][bias:], label=param_name + '={0}'.format(str(params[idx])))
    axes[1][0].plot(accs_train[idx][bias:], label=param_name + '={0}'.format(str(params[idx])))
    axes[1][1].plot(accs_test[idx][bias:], label=param_name + '={0}'.format(str(params[idx])))

    axes[2][0].plot(np.cumsum(times[idx])[bias:], losses_train[idx][bias:], label=param_name + '={0}'.format(str(params[idx])))
    axes[2][1].plot(np.cumsum(times[idx])[bias:], losses_test[idx][bias:], label=param_name + '={0}'.format(str(params[idx])))
    axes[3][0].plot(np.cumsum(times[idx])[bias:], accs_train[idx][bias:], label=param_name + '={0}'.format(str(params[idx])))
    axes[3][1].plot(np.cumsum(times[idx])[bias:], accs_test[idx][bias:], label=param_name + '={0}'.format(str(params[idx])))

axes[0][0].set_xlabel('Iteration'), axes[0][0].set_ylabel('Loss'), axes[0][0].set_title('Train Loss vs Iteration')
axes[0][1].set_xlabel('Iteration'), axes[0][1].set_ylabel('Loss'), axes[0][1].set_title('Test Loss vs Iteration')
axes[1][0].set_xlabel('Iteration'), axes[1][0].set_ylabel('Accuracy'), axes[1][0].set_title('Train Accuracy vs Iteration')
axes[1][1].set_xlabel('Iteration'), axes[1][1].set_ylabel('Accuracy'), axes[1][1].set_title('Test Accuracy vs Iteration')

axes[2][0].set_xlabel('Real Time'), axes[0][0].set_ylabel('Loss'), axes[2][0].set_title('Train Loss vs Iteration')
axes[2][1].set_xlabel('Real Time'), axes[0][1].set_ylabel('Loss'), axes[2][1].set_title('Test Loss vs Iteration')
axes[3][0].set_xlabel('Real Time'), axes[1][0].set_ylabel('Accuracy'), axes[3][0].set_title('Train Accuracy vs Iteration')
axes[3][1].set_xlabel('Real Time'), axes[1][1].set_ylabel('Accuracy'), axes[3][1].set_title('Test Accuracy vs Iteration')

axes[0][0].legend()
axes[0][1].legend()
axes[1][0].legend()
axes[1][1].legend()
axes[2][0].legend()
axes[2][1].legend()
axes[3][0].legend()
axes[3][1].legend()

axes[0][0].grid()
axes[0][1].grid()
axes[1][0].grid()
axes[1][1].grid()
axes[2][0].grid()
axes[2][1].grid()
axes[3][0].grid()
axes[3][1].grid()

left_bias = 3
axes[0][0].set_xlim(left=left_bias)
axes[0][1].set_xlim(left=left_bias)
axes[1][0].set_xlim(left=left_bias)
axes[1][1].set_xlim(left=left_bias)

fig.tight_layout()
fig.subplots_adjust(top=1.)
fig.savefig('./Report/Plots/' + model.__class__.__name__ + '_' + param_name + '.png')
plt.show()
