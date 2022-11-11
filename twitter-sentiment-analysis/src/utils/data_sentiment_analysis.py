from flask import render_template
from utils.data_collection import *
from utils.data_preprocessing import *
import warnings
warnings.filterwarnings('ignore')   

    
    
def data_sentiment_analysis(search_query,from_date,to_date,max_no_tweets):

    Topic = search_query[1:]
    
   
    # to get the required data based on the user query
    data = data_collection(search_query,from_date,to_date,max_no_tweets)


    if len(data) == 0:
        return data
        
    else: 

        # taking only required features
        data=data[['date','time','user.location','favorite_count','retweet_count','text']]

        # applying the define function data_cleaning for whole data
        data['cleaned_data']=data['text'].apply(lambda x:data_cleaning(x))

        ## applying the define function contra_expan for whole data
        data['cleaned_data']=data['cleaned_data'].apply(lambda x:contra_expan(x))


        ## applying the define function remove_accented for whole data
        data['cleaned_data']=data['cleaned_data'].apply(lambda x:remove_accented(x))

        #identifying the sentiment of each tweet
        data['vader_analysis']=data['cleaned_data'].apply(lambda x:vader.polarity_scores(x))
        data['comp_score']=data['vader_analysis'].apply(lambda x:x['compound'])


        ## applying the define function to identify the sentiment 
        data['AnalysisVad'] = data['comp_score'].apply(lambda x:getanalysis(x))

        data.drop(columns=['vader_analysis'],axis=1,inplace=True)

        #rename the particular column name
        data.rename(columns={'user.location':'location'},inplace=True)

        data['Topic'] = Topic

        return data


