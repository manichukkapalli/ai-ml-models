# importing dependencies

from time import time
from flask import Flask,request,render_template
import os
import yaml
from utils.data_collection import *
from utils.data_preprocessing import *
from utils.data_sentiment_analysis import *
import pyodbc
import datetime
import time
import warnings
warnings.filterwarnings('ignore')

#Load config file
with open('config/config.yaml') as file:
  config= yaml.safe_load(file)

#Get current working dir
cwd_path=os.getcwd()

# get the db config
db_config=config['db_config']

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('TwitterSentimentAnalysis.html')


@app.route('/twitter_sentiment_analysis', methods=['POST'])
def twitter_sentiment_analysis():
    
    #getting the data from the form
    search_query  = str(request.form.get('twitterTopic'))
    from_date = request.form.get('from')
    to_date =  request.form.get('to')
    max_no_tweets = int(request.form.get('NumofTweets'))

    # increasing the end date by one day 
    to_date=datetime.datetime.strptime(to_date, "%Y-%m-%d").date()+datetime. timedelta(days=1)

    Topic = search_query[1:]

    #connecting the database 
    conn = pyodbc.connect(**db_config)
    cursor=conn.cursor()

    # fetching  data for tbl_twitter_sentiment_analysis
    query = ''' SELECT * FROM tbl_twitter_sentiment_analysis WHERE 
        Date BETWEEN '{from_date}' AND '{to_date}' AND Search_topic = '{Topic}' '''.format(from_date=from_date, to_date=to_date, Topic=Topic)

    df = pd.read_sql_query(query,conn) 

    # this is checking in db data
    if len(df) == 0:  

        data = data_sentiment_analysis(search_query,from_date,to_date,max_no_tweets)

        if len(data)==0:
            error_msg = " There is no tweets for particular duration for your query"
            return render_template('error.html',error_msg=error_msg)

        else:

            for row in data.itertuples():
                cursor.execute(" INSERT INTO tbl_twitter_sentiment_analysis (Date,Time,Location,Likes,Retweet,Tweet_text,Sentiments,Search_topic) VALUES(?,?,?,?,?,?,?,?)",
                    row.date,
                    row.time,
                    row.location,
                    row.favorite_count,
                    row.retweet_count,
                    row.text,
                    row.AnalysisVad,
                    row.Topic
                )
                conn.commit()

           
            return render_template('dashboard.html') 

    else:
        
        if len(df) < max_no_tweets:

            # fetching  data for tbl_twitter_sentiment_analysis
            query = ''' SELECT  TOP 1 Date FROM tbl_twitter_sentiment_analysis WHERE Search_topic = '{Topic}'
            ORDER BY Date,Time ASC'''.format(Topic=Topic)

            df1 = pd.read_sql_query(query,conn)

            #to get last record from fetch date
            latest_date =  df1['Date'][0]

            # to get pervious date of latest_date and format
            pervious_date_format=latest_date - datetime.timedelta(days=1)
            to_date = pervious_date_format.strftime("%Y-%m-%d")
            
            data = data_sentiment_analysis(search_query,from_date,to_date,max_no_tweets)

            for row in data.itertuples():
                cursor.execute(" INSERT INTO tbl_twitter_sentiment_analysis (Date,Time,Location,Likes,Retweet,Tweet_text,Sentiments,Search_topic) VALUES(?,?,?,?,?,?,?,?)",
                    row.date,
                    row.time,
                    row.location,
                    row.favorite_count,
                    row.retweet_count,
                    row.text,
                    row.AnalysisVad,
                    row.Topic
                )
                conn.commit()
         
            return render_template('dashboard.html') 

        else:
            return render_template('dashboard.html') 


if __name__ == '__main__':
    app.run(host='0.0.0.0',port=int(8080), debug=True)




        



    

        










