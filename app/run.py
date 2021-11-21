import json, joblib, plotly
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sqlalchemy import create_engine
from returnFigures import return_figures

app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('EtlTable', engine)

# load model
model = joblib.load("../models/classifier.sav")

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

				# create visuals
				figures = return_figures()
    
  		# plot ids for the html id tag, encode plotly graphs in JSON
				ids = ['figure-{}'.format(i) for i, _ in enumerate(figures)]

    # Convert the plotly figures to JSON for javascript in html template
				figuresJSON = json.dumps(figures, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
				return render_template('master.html', ids=ids, figuresJSON=figuresJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():

				# save user input in query
				query = request.args.get('query', '')			
				
				# use model to predict classification for query
				classification_labels = model.predict([query])[0]
				classification_results = dict(zip(df.columns[2:], classification_labels))

    # This will render the go.html Please see that file.
				return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )
				
def main():
				app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()