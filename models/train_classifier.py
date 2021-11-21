import nltk, sys
nltk.download(['punkt', 'wordnet', 'stopwords'])

import warnings
warnings.filterwarnings("ignore")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")

import pickle, joblib
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sqlalchemy import create_engine

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier


def load_data(database_filepath):

				engine = create_engine('sqlite:///' + database_filepath)
				df = pd.read_sql_table('EtlTable', engine)
				
				#X = df['message']
				X = df.message.values
				Y = np.asarray(df.drop(labels = ['id','message'], axis=1))
				category_names = df[df.columns[2:]].columns.tolist()
				
				return X, Y, category_names


def tokenize(text):
    
    # tokenize text (Split text into tokens)
    tokens = word_tokenize(text) 
    
    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    # iterate through each token
    clean_tokens = []
    
    for tok in tokens:
              
        # lemmatize, normalize case, and remove leading/trailing white space
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():

				parameters = {'n_estimators':[10, 50], 'min_samples_split':[2, 3]}
				
				cv = GridSearchCV(RandomForestClassifier(random_state=0, n_jobs=-1), param_grid=parameters, n_jobs=-1)

    # build pipeline
				pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1,3), stop_words='english', tokenizer=tokenize, sublinear_tf=True)),
    ('clf', MultiOutputClassifier(cv))
				])
				
				return pipeline


def evaluate_model(y_test, y_pred, category_names):
    
    for i in range (len(category_names)):
        
        report = classification_report(y_test[:,i], y_pred[:,i])
        accuracyScore = accuracy_score(y_test[:,i], y_pred[:,i])
        print("Category : ", category_names[i], '\n')
        print("Exact Match Ratio (Subset accuracy) = ", accuracyScore, '\n')
        print("Classification report : ", '\n\n', report, '\n')
        

def save_model(model, model_filepath):
       
    # save the model to disk
    joblib.dump(model, model_filepath)


def main():

				if len(sys.argv) == 3:
					
								database_filepath, model_filepath = sys.argv[1:]
								print('Loading data...\n    DATABASE: {}'.format(database_filepath))
								X, Y, category_names = load_data(database_filepath)
        
        # splitting the data to training and testing data set
								X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42, shuffle=True)
								
								print('Building model...')
								model = build_model()
								
								print('Training model...')
								model.fit(X_train, y_train)
								
								print('Predicting model...')
								y_pred = model.predict(X_test)
								
								print('Evaluating model...')
								evaluate_model(y_test, y_pred, category_names)
								
								print('Saving model...\n    MODEL: {}'.format(model_filepath))
								save_model(model, model_filepath)
								
								print('Trained model saved!')
								
				else:
					
							print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.sav')


if __name__ == '__main__':
    main()