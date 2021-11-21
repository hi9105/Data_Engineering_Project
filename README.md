# Project : Disaster Response

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [File Descriptions](#files)
4. [Results](#results)
5. [Licensing, Authors, and Acknowledgements](#licensing)

## Overview <a name="overview"></a>

In this project, we have a data set containing real messages that were sent during disaster events. We have created a machine learning pipeline to categorize these events so that we can send the messages to an appropriate disaster relief agency. We have included a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data. 

## Installation <a name="installation"></a>

There should be no necessary libraries to run the code here beyond the Anaconda distribution of Python.  The code should run with no issues using Python versions 3.*.

## File Descriptions <a name="files"></a>

1. Data (Project Workspace - ETL) : this folder contains following files
    
    1. disaster_categories.csv : data to process.
    
    2. disaster_messages.csv : data to process.
    
    3. process_data.py : read the dataset, clean the data, and then store it in a SQLite database.
    
    4. DisasterResponse.db : database for saving clean data.
    
 
2. Models (Project Workspace - Machine Learning Pipeline) : this folder contains following files
    
    1. train_classifier.py : load the database, build the model, evaluate the model and then save the model. Split the data into a training set and a test set.
       Create a machine learning pipeline that uses NLTK, scikit-learn's Pipeline and GridSearchCV to output a final model that uses the message column to predict
       classifications for 36 categories (multi-output classification). Finally, export model to a sav file.
       
    2. classifier.sav : saved model. File to save the model.
 
    

3. App : this folder contains following files
    
    1. returnFigures.py : create three simple data visualizations in web app based on data (extract from the SQLite database).
     
    2. run.py : Flask file that runs app. Display results in a Flask web app. Upload database file and sav file with model. 
    
    3. Templates : contains  html, css, and javascript files.
    
        - master.html : main page of web app.
        - go.html : classification result page of web app.
    
    5. DisasterResponse.db : database for saving clean data
    
 
