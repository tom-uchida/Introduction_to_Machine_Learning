from sklearn.datasets import fetch_20newsgroups
from pprint import pprint

# Read datasets
train_set = fetch_20newsgroups(subset='train', random_state=42)
test_set = fetch_20newsgroups(subset='test', random_state=42)

# Prepare train data and test data
X_train = train_set.data
y_train = train_set.target
X_test  = test_set.data
y_test  = test_set.target

# print('カテゴリ一覧')
# pprint(train_set.target_names)
# print('\n')
# print('記事その1')
# print(f'News Script:\n{X_train[0]}')
# print('記事その1のカテゴリ')
# print(f'Text Category label: {y_train[0]}')

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(stop_words='english')
vectorizer.fit(X_train)
X_train_bow = vectorizer.transform(X_train)
X_test_bow  = vectorizer.transform(X_test)

# print('(テキスト番号, 単語番号) 出現回数')
# print(X_train_bow[0])
# print('\nBoW表現ベクトル')
# print(X_train_bow[0].toarray())

# Apply Naive Bayes
from sklearn.naive_bayes import MultinomialNB

mnb_small = MultinomialNB(alpha=0.001)
mnb_small.fit(X_train_bow, y_train)

mnb_large = MultinomialNB(alpha=100)
mnb_large.fit(X_train_bow, y_train)

# Print accuracy
print(f'Train Accuracy: {mnb_small.score(X_train_bow, y_train):.3f}')
print(f'Test Accuracy: {mnb_small.score(X_test_bow, y_test):.3f}')
print(f'Train Accuracy: {mnb_large.score(X_train_bow, y_train):.3f}')
print(f'Test Accuracy: {mnb_large.score(X_test_bow, y_test):.3f}')