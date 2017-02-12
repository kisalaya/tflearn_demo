# # Sentiment analysis with TFLearn

import pandas as pd
import numpy as np
import tensorflow as tf
import tflearn
from tflearn.data_utils import to_categorical
from collections import Counter

# Preparing the data

reviews = pd.read_csv('reviews.txt', header=None)
labels = pd.read_csv('labels.txt', header=None)

stop_words = ['a', 'about', 'above', 'according', 'across', 'after', 'afterwards', 'again', 'against', 'albeit',
              'all', 'almost', 'alone', 'along', 'already', 'also', 'although', 'always', 'am', 'among',
              'amongst', 'an', 'and', 'another', 'any', 'anybody', 'anyhow', 'anyone', 'anything', 'anyway',
              'anywhere', 'apart', 'are', 'around', 'as', 'at', 'av', 'be', 'became', 'because', 'become',
              'becomes', 'becoming', 'been', 'before', 'beforehand', 'behind', 'being', 'below', 'beside',
              'besides', 'between', 'beyond', 'both', 'but', 'by', 'can', 'cannot', 'canst', 'certain', 'cf',
              'choose', 'contrariwise', 'cos', 'could', 'cu', 'day', 'do', 'does', "doesn't", 'doing', 'dost',
              'doth', 'double', 'down', 'dual', 'during', 'each', 'either', 'else', 'elsewhere', 'enough', 'et',
              'etc', 'even', 'ever', 'every', 'everybody', 'everyone', 'everything', 'everywhere', 'except',
              'excepted', 'excepting', 'exception', 'exclude', 'excluding', 'exclusive', 'far', 'farther',
              'farthest', 'few', 'ff', 'first', 'for', 'formerly', 'forth', 'forward', 'from', 'front',
              'further', 'furthermore', 'furthest', 'get', 'go', 'had', 'halves', 'hardly', 'has', 'hast',
              'hath', 'have', 'he', 'hence', 'henceforth', 'her', 'here', 'hereabouts', 'hereafter', 'hereby',
              'herein', 'hereto', 'hereupon', 'hers', 'herself', 'him', 'himself', 'hindmost', 'his', 'hither',
              'hitherto', 'how', 'however', 'howsoever', 'i', 'ie', 'if', 'in', 'inasmuch', 'inc', 'include',
              'included', 'including', 'indeed', 'indoors', 'inside', 'insomuch', 'instead', 'into', 'inward',
              'inwards', 'is', 'it', 'its', 'itself', 'just', 'kind', 'kg', 'km', 'last', 'latter', 'latterly',
              'less', 'lest', 'let', 'like', 'little', 'ltd', 'many', 'may', 'maybe', 'me', 'meantime',
              'meanwhile', 'might', 'moreover', 'most', 'mostly', 'more', 'mr', 'mrs', 'ms', 'much', 'must',
              'my', 'myself', 'namely', 'need', 'neither', 'never', 'nevertheless', 'next', 'no', 'nobody',
              'none', 'nonetheless', 'noone', 'nope', 'nor', 'not', 'nothing', 'notwithstanding', 'now',
              'nowadays', 'nowhere', 'of', 'off', 'often', 'ok', 'on', 'once', 'one', 'only', 'onto', 'or',
              'other', 'others', 'otherwise', 'ought', 'our', 'ours', 'ourselves', 'out', 'outside', 'over',
              'own', 'per', 'perhaps', 'plenty', 'provide', 'quite', 'rather', 'really', 'round', 'said', 'sake',
              'same', 'sang', 'save', 'saw', 'see', 'seeing', 'seem', 'seemed', 'seeming', 'seems', 'seen',
              'seldom', 'selves', 'sent', 'several', 'shalt', 'she', 'should', 'shown', 'sideways', 'since',
              'slept', 'slew', 'slung', 'slunk', 'smote', 'so', 'some', 'somebody', 'somehow', 'someone',
              'something', 'sometime', 'sometimes', 'somewhat', 'somewhere', 'spake', 'spat', 'spoke', 'spoken',
              'sprang', 'sprung', 'stave', 'staves', 'still', 'such', 'supposing', 'than', 'that', 'the', 'thee',
              'their', 'them', 'themselves', 'then', 'thence', 'thenceforth', 'there', 'thereabout',
              'thereabouts', 'thereafter', 'thereby', 'therefore', 'therein', 'thereof', 'thereon', 'thereto',
              'thereupon', 'these', 'they', 'this', 'those', 'thou', 'though', 'thrice', 'through', 'throughout',
              'thru', 'thus', 'thy', 'thyself', 'till', 'to', 'together', 'too', 'toward', 'towards', 'ugh',
              'unable', 'under', 'underneath', 'unless', 'unlike', 'until', 'up', 'upon', 'upward', 'upwards',
              'us', 'use', 'used', 'using', 'very', 'via', 'vs', 'want', 'was', 'we', 'week', 'well', 'were',
              'what', 'whatever', 'whatsoever', 'when', 'whence', 'whenever', 'whensoever', 'where',
              'whereabouts', 'whereafter', 'whereas', 'whereat', 'whereby', 'wherefore', 'wherefrom', 'wherein',
              'whereinto', 'whereof', 'whereon', 'wheresoever', 'whereto', 'whereunto', 'whereupon', 'wherever',
              'wherewith', 'whether', 'whew', 'which', 'whichever', 'whichsoever', 'while', 'whilst', 'whither',
              'who', 'whoa', 'whoever', 'whole', 'whom', 'whomever', 'whomsoever', 'whose', 'whosoever', 'why',
              'will', 'wilt', 'with', 'within', 'without', 'worse', 'worst', 'would', 'wow', 'ye', 'yet', 'year',
              'yippee', 'you', 'your', 'yours', 'yourself', 'yourselves', '']

# Counting word frequency

total_counts = Counter()
for review in reviews[0]:
	words = review.split(" ")
	for word in words:
		total_counts[word] += 1

print("Total words in data set: ", len(total_counts))

for word in stop_words:
	if word in total_counts:
		del total_counts[word]

print("Total words in data set after removing stop words: ", len(total_counts))

vocab = sorted(total_counts, key=total_counts.get, reverse=True)[:10000]

word2idx = {}

for i in range(len(vocab)):
	word2idx[vocab[i]] = i


# Text to vector function

def text_to_vector(text):
	w_vec = np.zeros(len(vocab))
	words = text.lower().split(' ')

	for word in words:
		if word in word2idx:
			w_to_i = word2idx[word]
			w_vec[w_to_i] += 1
	return w_vec


word_vectors = np.zeros((len(reviews), len(vocab)), dtype=np.int_)
for ii, (_, text) in enumerate(reviews.iterrows()):
	word_vectors[ii] = text_to_vector(text[0])

# Train, Validation, Test sets

Y = (labels == 'positive').astype(np.int_)
records = len(labels)

shuffle = np.arange(records)
np.random.shuffle(shuffle)
test_fraction = 0.9

train_split, test_split = shuffle[:int(records * test_fraction)], shuffle[int(records * test_fraction):]
trainX, trainY = word_vectors[train_split, :], to_categorical(Y.values[train_split], 2)
testX, testY = word_vectors[test_split, :], to_categorical(Y.values[test_split], 2)


# Network building
def build_model():
	net = tflearn.input_data([None, len(vocab)])
	net = tflearn.fully_connected(net, 100, activation='ReLU')
	net = tflearn.fully_connected(net, 25, activation='ReLU')
	net = tflearn.fully_connected(net, 2, activation='softmax')

	net = tflearn.regression(net, optimizer='sgd', learning_rate=0.1, loss='categorical_crossentropy')

	model = tflearn.DNN(net)
	return model


# Intializing the model

model = build_model()

# Training
model.fit(trainX, trainY, validation_set=0.1, show_metric=True, batch_size=128, n_epoch=20)

# Testing
predictions = (np.array(model.predict(testX))[:, 0] >= 0.5).astype(np.int_)
test_accuracy = np.mean(predictions == testY[:, 0], axis=0)
print("Test accuracy: ", test_accuracy)

# Try out your own text!
text = "This movie is so bad. It was awful and the worst"
positive_prob = model.predict([text_to_vector(text.lower())])[0][1]
print('P(positive) = {:.3f} :'.format(positive_prob), 'Positive' if positive_prob > 0.5 else 'Negative')
