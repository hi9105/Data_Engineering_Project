import pandas as pd
import plotly.graph_objs as go
from sqlalchemy import create_engine

def return_figures():

				"""Creates three plotly visualizations

    Args:
        None

    Returns:
        list (dict): list containing the three plotly visualizations

    """
				# load data
				engine = create_engine('sqlite:///../data/DisasterResponse.db')
				df = pd.read_sql_table('EtlTable', engine)
				messages = pd.read_csv('../data/disaster_messages.csv')

				# first chart plot as a bar chart
				graph_one = []

				genre_counts = messages.groupby('genre').count()['message']
				genre_names = list(genre_counts.index)
				
				graph_one.append(
						go.Bar(
						x = genre_names,
      y = genre_counts))

				layout_one = dict(title = 'Distribution of Message Genres',
																						xaxis = dict(title = 'Genre', autotick=False, tick0="direct"),
                						yaxis = dict(title = 'Count')
                	)

				# second chart plot as a bar chart    
				graph_two = []

				related_counts = df.groupby('related').count()['message']
				related_names = list(related_counts.index)

				graph_two.append(
      go.Bar(
      x = related_names,
      y = related_counts))
				
				layout_two = dict(title = 'Distribution of category Related',
                xaxis = dict(title = 'Category Related', autotick=False, tick0=0),
                yaxis = dict(title = 'Count'))

				# third chart plot as a bar chart
				graph_three = []
    
				request_counts = df.groupby('request').count()['message']
				request_names = list(request_counts.index)

				graph_three.append(
          go.Bar(
          x = request_names,
          y = request_counts))

				layout_three = dict(title = 'Distribution of category Request',
                xaxis = dict(title = 'Category Request', autotick=False, tick0=0),
                yaxis = dict(title = 'Count'))
      
    # append all charts to the figures list
				figures = []
				figures.append(dict(data=graph_one, layout=layout_one))
				figures.append(dict(data=graph_two, layout=layout_two))
				figures.append(dict(data=graph_three, layout=layout_three))
				
				return figures