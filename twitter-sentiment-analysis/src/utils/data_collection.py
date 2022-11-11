'''
    This function fetch the data from the twitter using tweepy library
    Parse the fetched data into structured format and
    save the data in csv file

'''



## importing dependencies
import os
import tweepy as tw  
import pandas as pd
from datetime import datetime
import yaml


#Load config file
with open('config/config.yaml') as file:
  config= yaml.safe_load(file)

  
#Get current working dir
cwd_path=os.getcwd()

# get the authentication config
APIKey = config["APIKey"]
APIsecretKey = config['APIsecretKey']
accessToken = config['accessToken']
accessTokenSecret = config['accessTokenSecret']

#Using our Twitter API credentials to authenticate and connect to the API.
authenticate = tw.OAuthHandler(APIKey,APIsecretKey)
authenticate.set_access_token(accessToken,accessTokenSecret)
api = tw.API(authenticate)


def data_collection(topic,from_date,to_date,count):

  # get tweets from the API 
  tweets = tw.Cursor(api.search_tweets,q=topic,lang="en",since_id=from_date,until=to_date).items(count)
  
  # formatting the parsing data into structured format
  json_data = [r._json for r in tweets]
  raw_data = pd.json_normalize(json_data)

  if len(raw_data) != 0:

    # #to change the format of data 
    raw_data['date'] = pd.to_datetime(raw_data['created_at'])
    
    #to change the format of time
    raw_data['time'] = pd.to_datetime(raw_data['created_at']).dt.time

    # user query data type conversion from str to datetime.date format
    datetime1 = datetime.strptime(from_date, '%Y-%m-%d')
    start_date=datetime1.date()

    # checking whether the fetched are between the user queries dates or not
    filtered_data = []
    for row in raw_data.itertuples():
        if row.date < to_date and row.date > start_date:
            filtered_data.append('Yes')
        else:
            filtered_data.append('No')

    raw_data['date_checking']=filtered_data

    # filtering data only for specified date from the user side 
    raw_data = raw_data[raw_data['date_checking']=='Yes']

    # droping the date_checking col after checking
    raw_data.drop(columns=['date_checking'],axis=1,inplace=True)

    return raw_data

  else: 
    return raw_data
  


