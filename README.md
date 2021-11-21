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

1. data : this folder contains following files
    
    1. disaster_categories.csv : data to process.
    2. disaster_messages.csv : data to process.
    3. process_data.py : read the dataset, clean the data, and then store it in a SQLite database.
    4. DisasterResponse.db : database for saving clean data
