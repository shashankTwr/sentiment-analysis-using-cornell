import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import load_files
import nltk
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
moviedir=r'C:\Users\Tiwari\Desktop\sentiment analysis\txt_sentoken'
movie_train=load_files(moviedir,shuffle=True)
print(len(movie_train.data)) 
print(movie_train.target_names)
movie_vec = CountVectorizer(min_df=2, tokenizer=nltk.word_tokenize)
movie_counts = movie_vec.fit_transform(movie_train.data)
print(movie_vec.vocabulary_)
print(movie_vec.vocabulary_.get('screen'))
print(movie_vec.vocabulary_.get('Tom'))
print(movie_counts.shape)
tfidf_transformer = TfidfTransformer()
movie_tfidf = tfidf_transformer.fit_transform(movie_counts)
print(movie_tfidf.shape)
docs_train, docs_test, y_train, y_test = train_test_split(movie_tfidf, movie_train.target, test_size = 0.20, random_state = 12)
clf = MultinomialNB().fit(docs_train, y_train)
y_pred = clf.predict(docs_test)
sklearn.metrics.accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
print(cm)
reviews_new = ['This movie was excellent', 'Absolute joy ride', 
            'Cringiest Movie with no action', '‘Mission: Impossible – Fallout’ has a great mix of plot, pacing and performances that undeniably makes it the best entry in the franchise', 
              'This was certainly a movie', 'Two thumbs up', 'I fell asleep halfway through', 
              "We can't wait for the sequel!!", '!', '?', 'I cannot recommend this highly enough', 
              'instant classic.', 'dwayne johnson was amazing', 'This movie is certainly Oscar-worthy.']
reviews_new_counts = movie_vec.transform(reviews_new)
reviews_new_tfidf = tfidf_transformer.transform(reviews_new_counts)
pred = clf.predict(reviews_new_tfidf)
for review, category in zip(reviews_new, pred):
    print('%r => %s' % (review, movie_train.target_names[category]))