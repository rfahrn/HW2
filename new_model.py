#%% 
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.sparse as sp
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
import re 
from nltk.corpus import stopwords
import nltk
nltk.download('punkt')  # For tokenization
nltk.download('averaged_perceptron_tagger') 
nltk.download('wordnet')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
custom_stopwords = ['says', 'said','one','new' ,'news']
stop_words.update(custom_stopwords)

from collections import Counter


class CTFIDFVectorizer(TfidfTransformer):
    """Convert a collection of raw documents to a matrix of c-TF-IDF features (class based tf-idf) - it is not a transformer model, it is a vectorizer model (it does not learn anything) - it is a modification of the TfidfTransformer class (it inherits from it) """
    def __init__(self, *args, **kwargs):
        super(CTFIDFVectorizer, self).__init__(*args, **kwargs)

    def fit(self, X: sp.csr_matrix, n_samples: int):
        """learn idf vector (global term weights) """
        _, n_features = X.shape
        df = np.squeeze(np.asarray(X.sum(axis=0)))
        idf = np.log(n_samples / df)
        self._idf_diag = sp.diags(idf, offsets=0,
                                  shape=(n_features, n_features),
                                  format='csr',
                                  dtype=np.float64)
        return self

    def transform(self, X: sp.csr_matrix) -> sp.csr_matrix:
        """transform a count-based matrix to c-TF-IDF / class based tf-idf """
        X = X * self._idf_diag
        X = normalize(X, axis=1, norm='l1', copy=False)
        return X
    
def get_top_tfidf_words(column, top_n=10):
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(train[column].dropna())
    indices = np.argsort(vectorizer.idf_)[::-1]
    features = vectorizer.get_feature_names_out()
    top_features = [features[i] for i in indices[:top_n]]
    return top_features

def plot_top_words(top_words, title):
    sns.barplot(x=np.arange(len(top_words)), y=top_words)
    plt.title(title)
    plt.ylabel('Word')
    plt.xlabel('Importance')
    plt.show()

def clean_text(text):
    """ Clean text data by removing special characters and stopwords"""
    if text.startswith("b'") or text.startswith('b"'):
        text = text[2:-1]
    
    # Remove special characters
    text = bytes(text, 'utf-8').decode('unicode_escape', 'ignore')
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I|re.A)
    text = text.lower()

    # Tokenization
    tokens = word_tokenize(text)
    pos_tags = pos_tag(tokens)
    # Keep only nouns and adjectives
    words = [word for word, tag in pos_tags if tag in ['NN', 'JJ']]
    #words = re.findall(r'\w+', text) 

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]

    # Remove stopwords
    words = [word for word in words if word not in stop_words]

    return ' '.join(words)

def add_date_feature(train):
    """ Add date features to the training data"""
    train['Date'] = pd.to_datetime(train['Date'])
    train['Year'] = train['Date'].dt.year
    train['Month'] = train['Date'].dt.month
    train['Day'] = train['Date'].dt.day

def plot_term_frequencies(counter, topic, top_n=20):
    most_common_terms = counter.most_common(top_n)
    terms, counts = zip(*most_common_terms)

    plt.figure(figsize=(10, 6))
    plt.bar(terms, counts)
    plt.xticks(rotation=45)
    plt.xlabel('Terms')
    plt.ylabel('Frequency')
    plt.title(f'Term Frequencies for {topic}')
    plt.show()

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


term_frequencies = {}
for i in range(1, 26):
    topic = f'Top{i}'
    processed_text = train[topic].dropna().apply(clean_text)
    all_words = ' '.join(processed_text).split()
    term_frequencies[topic] = Counter(all_words)


#%%¨
# add date features
add_date_feature(train)
add_date_feature(test)

# clean all text 
for i in range(1, 26):
    topic = f'Top{i}'
    train[topic] = train[topic].fillna('').apply(clean_text)
    test[topic] = test[topic].fillna('').apply(clean_text)

#%%
# naive approach: 
train['combined_text'] = train[[f'Top{i}' for i in range(1, 26)]].apply(lambda x: ' '.join(x.dropna().astype(str)), axis=1)
test['combined_text'] = test[[f'Top{i}' for i in range(1, 26)]].apply(lambda x: ' '.join(x.dropna().astype(str)), axis=1)
X_test = test['combined_text'].apply(clean_text)

X = train['combined_text']
y = train['Label']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(max_features=5000, stop_words=list(stop_words))
X_train_vec = vectorizer.fit_transform(X_train)
X_val_vec = vectorizer.transform(X_val)
X_test_vec = vectorizer.transform(X_test)

#%%

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)


y_val_pred = model.predict(X_val_vec)

preds_test = model.predict(X_test_vec)

print(classification_report(y_val, y_val_pred))
print(confusion_matrix(y_val, y_val_pred))

# %%
preds = model.predict(X_test_vec)
X_val_vec = vectorizer.transform(X_val)
val_accuracy = (model.predict(X_val_vec) == y_val).mean() * 100.0
print(f"Validation Accuracy: {val_accuracy}%")
# %%

preds
# %%
# samll changes 
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features=5000, stop_words=list(stop_words), ngram_range=(1, 2))
X_text = vectorizer.fit_transform(train['combined_text'])

# Convert date features to a format suitable for the model
date_features = train[['Year', 'Month', 'Day']]
date_features_sparse = csr_matrix(date_features)
X_combined = hstack([X_text, date_features_sparse])

X_train, X_val, y_train, y_val = train_test_split(X_combined, y, test_size=0.2, random_state=42)

# Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Logistic Regression
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)

#%%
from sklearn.ensemble import VotingClassifier

voting_clf = VotingClassifier(estimators=[
    ('rf', rf_model), 
    ('lr', lr_model)], 
    voting='hard')

voting_clf.fit(X_train, y_train)

#%%
from sklearn.metrics import accuracy_score

# Evaluate Random Forest
rf_preds = rf_model.predict(X_val)
print("Random Forest Accuracy:", accuracy_score(y_val, rf_preds))

# Evaluate Logistic Regression
lr_preds = lr_model.predict(X_val)
print("Logistic Regression Accuracy:", accuracy_score(y_val, lr_preds))

# Evaluate Voting Classifier
voting_preds = voting_clf.predict(X_val)
print("Voting Classifier Accuracy:", accuracy_score(y_val, voting_preds))

# %%
# classifier per column/topic 

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
# Standardize the date features

date_features = train[['Year', 'Month', 'Day']]
date_features_scaled = scaler.fit_transform(date_features)

classifiers = {}

for i in range(1, 26):
    topic = f'Top{i}'
    vectorizer = TfidfVectorizer(max_features=5000, stop_words=list(stop_words), ngram_range=(1, 2))
    
    # Vectorize the text for this topic
    X_topic = vectorizer.fit_transform(train[topic].dropna())

    # Combine with date features
    X_combined = hstack([X_topic, csr_matrix(date_features_scaled)])
    
    # Split into training and validation sets
    X_combined_train, X_combined_val, y_combined_train, y_combined_val = train_test_split(X_combined, y, test_size=0.2, random_state=42)
    
    # Train a classifier for this combined feature set
    classifier = MultinomialNB()
    classifier.fit(X_combined_train, y_combined_train)
    
    # Store the classifier and vectorizer
    classifiers[topic] = (classifier, vectorizer)


#%%
from scipy.stats import mode
from scipy.sparse import hstack, csr_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


predictions = {}

for topic, (classifier, vectorizer) in classifiers.items():
    X_topic_val = vectorizer.transform(train[topic].dropna())
    X_combined_val = hstack([X_topic_val, csr_matrix(date_features_scaled)])
    predictions[topic] = classifier.predict(X_combined_val)

# Combine predictions from all topics
combined_predictions = np.array([predictions[topic] for topic in predictions])

# Apply majority voting
majority_vote_predictions = mode(combined_predictions, axis=0)[0][0]

# Calculate accuracy
majority_vote_accuracy = (majority_vote_predictions == y_val).mean() * 100.0
print(f"Majority Voting Classifier Accuracy: {majority_vote_accuracy}%")

# %%

# Additional imports
from sklearn.svm import SVC

# Train individual classifiers including SVM
X_train, X_val, y_train, y_val = train_test_split(X_combined, y, test_size=0.2, random_state=42)

# SVM Classifier
svm_model = SVC(probability=True)
svm_model.fit(X_train, y_train)

# Evaluate each classifier
# Already defined: rf_model, lr_model
classifiers = {'RandomForest': rf_model, 'SVM': svm_model}

for name, clf in classifiers.items():
    y_pred = clf.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    print(f"{name} Accuracy: {acc}")

# Majority Voting Classifier
voting_clf = VotingClassifier(estimators=[
    ('rf', rf_model), 
    ('svm', svm_model)], 
    voting='hard')

voting_clf.fit(X_train, y_train)

# Evaluate Majority Voting Classifier
y_pred_majority_vote = voting_clf.predict(X_val)
majority_vote_acc = accuracy_score(y_val, y_pred_majority_vote)
print(f"Majority Voting Classifier Accuracy: {majority_vote_acc}")

# Predictions on Test Data
X_test_vec = vectorizer.transform(test['combined_text'])
test_preds = voting_clf.predict(X_test_vec)

# Output Test Predictions (if needed)
print(test_preds)

# %%
from sklearn.naive_bayes import MultinomialNB

topic_classifiers = {}
topic_predictions = {}

for i in range(1, 26):
    topic = f'Top{i}'
    vectorizer = TfidfVectorizer(max_features=5000, stop_words=list(stop_words), ngram_range=(1, 2))
    X_topic = vectorizer.fit_transform(train[topic].fillna(''))

    # Split into training and validation sets
    X_topic_train, X_topic_val, _, y_topic_val = train_test_split(X_train, y, test_size=0.2, random_state=42)

    # Train a classifier for this topic
    classifier = MultinomialNB()
    classifier.fit(X_topic_train, y_topic_train)

    # Store classifier and make predictions
    topic_classifiers[topic] = classifier
    topic_vectorizers[topic] = vectorizer
    topic_predictions[topic] = classifier.predict(X_topic_val)

predictions_df = pd.DataFrame(topic_predictions)
predictions_df.head()


#%%
# Add date features to the predictions DataFrame
for feature in ['Year', 'Month', 'Day']:
    predictions_df[feature] = train[feature]

# Split into training and validation sets for the final model
X_final_train, X_final_val, y_final_train, y_final_val = train_test_split(predictions_df, y, test_size=0.2, random_state=42)

# Train a classifier for the final model

# Using RandomForestClassifier as an example for the final model
final_model = RandomForestClassifier(n_estimators=100, random_state=42)
final_model.fit(X_final_train, y_final_train)

# Evaluate the final classifier
final_predictions = final_model.predict(X_final_val)
final_accuracy = accuracy_score(y_final_val, final_predictions)
print(f"Final Classifier Accuracy: {final_accuracy}")

# %%
# Generate predictions for each topic in the test data
test_topic_predictions = {topic: clf.predict(vectorizer.transform(test[topic].fillna(''))) for topic, clf in topic_classifiers.items()}
test_predictions_df = pd.DataFrame(test_topic_predictions)

# Add date features for the test data
for feature in ['Year', 'Month', 'Day']:
    test_predictions_df[feature] = test[feature]

# Final predictions on test data
test_final_predictions = final_model.predict(test_predictions_df)
