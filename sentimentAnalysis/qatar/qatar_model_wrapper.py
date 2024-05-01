import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

import logging
class QatarModelWrapper:
    def __init__(self):
        current_directory = os.getcwd()
        self.data = pd.read_csv(os.path.join(current_directory, "sentimentAnalysis", "qatar_airways_reviews.csv"))
        self.data['sentiment'] = self.data['Rating'].apply(lambda x: 'Positive' if x >= 4 else 'Negative')
        self.logistic_regression_model = None
        self.tfidf_vectorizer = None
        logging.debug("data loaded...")
        self.__train_model()

    def __train_model(self):
        x = self.data['Review Body']
        y = self.data['sentiment']
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000)

        x_train_tfidf = self.tfidf_vectorizer.fit_transform(x_train)
        x_test_tfidf = self.tfidf_vectorizer.transform(x_test)

        self.logistic_regression_model = LogisticRegression(solver='liblinear', C=10, penalty='l2')
        self.logistic_regression_model.fit(x_train_tfidf, y_train)

        y_pred = self.logistic_regression_model.predict(x_test_tfidf)
        accuracy = accuracy_score(y_test, y_pred)
        logging.debug(f"Accuracy:{accuracy}")

        logging.debug("Classification Report:")
        logging.debug(classification_report(y_test, y_pred))

    def __preprocess_review(self, review):
        preprocessed_review = review.lower()
        return preprocessed_review

    def __vectorize_review(self, preprocessed_review):
        vectorized_review = self.tfidf_vectorizer.transform([preprocessed_review])
        return vectorized_review

    def __predict_sentiment(self, vectorized_review, model):
        predicted_sentiment = model.predict(vectorized_review)
        return predicted_sentiment[0]

    def __process_new_review(self, new_review, model):
        preprocessed_review = self.__preprocess_review(new_review)
        vectorized_review = self.__vectorize_review(preprocessed_review)
        predicted_sentiment = self.__predict_sentiment(vectorized_review, model)
        return predicted_sentiment

    def get_review(self, review):
        return self.__process_new_review(review, self.logistic_regression_model)

