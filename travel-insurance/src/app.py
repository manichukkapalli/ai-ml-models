from flask import Flask, request,jsonify, render_template,redirect
from werkzeug.utils import secure_filename
import pandas as pd
import pickle
import os
import numpy as np
import pyodbc
import yaml

# Name of model file
prediction_model_file='randomforest_model.pkl'

# Load config file
with open('config/config.yaml') as file:
  config= yaml.safe_load(file)

  # get the db config
db_config=config['db_config']


# Prediction model full path
app_file_path=os.path.abspath(__file__) #current app file path
current_dir=os.path.dirname(app_file_path)  #current app file dir path
prediction_model_file=os.path.join(current_dir,prediction_model_file) # model absolute path

app = Flask(__name__)

#for uploading file
path = os.path.dirname(__file__)
UPLOAD_FOLDER = os.path.join(path, 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('TravelInsurance.html')


@app.route('/predict', methods=['POST'])

def predict():
    # Data from postman

    if request.method == 'POST':

        Age=int(request.form.get('Age'))
        Employment_Type=str(request.form.get('Employment_Type'))
        GraduateOrNot=str(request.form.get('GraduateOrNot'))
        AnnualIncome=int(request.form.get('AnnualIncome'))
        FamilyMembers=int(request.form.get('FamilyMembers'))
        ChronicDiseases=int(request.form.get('ChronicDiseases'))
        FrequentFlyer=str(request.form.get('FrequentFlyer'))
        EverTravelledAbroad=str(request.form.get('EverTravelledAbroad')) 


       # connecting the database 
        conn = pyodbc.connect(**db_config)
        cursor=conn.cursor()

       

    
        # Create user query for prediction
        user_query={'Age':[Age],
                        'Employment Type':[Employment_Type],
                        'GraduateOrNot':[GraduateOrNot],
                        'AnnualIncome':[AnnualIncome],
                        'FamilyMembers':[FamilyMembers],
                        'ChronicDiseases':[ChronicDiseases],
                        'FrequentFlyer':[FrequentFlyer],
                        'EverTravelledAbroad':[EverTravelledAbroad],}
        user_query=pd.DataFrame.from_dict(user_query)
        print(user_query)

        ## Data Pre-processing
        user_query['Employment Type'] = user_query['Employment Type'].map({'Private Sector':0, 'Government Sector':1,'Self Employed':0})
        user_query['GraduateOrNot']  =user_query['GraduateOrNot'].map({'Yes':0, 'No':1})
        user_query['FrequentFlyer'] = user_query['FrequentFlyer'].map({'Yes':0,'No':1})
        user_query['EverTravelledAbroad'] = user_query['EverTravelledAbroad'].map({'Yes':0,'No':1})

        
        # Prediction
        #model=load_model()
        #prediction = model.predict(user_query)
        #predictions_prob = model.predict_proba(user_query)
        # Prediction model filename
    
        #prediction_model = pickle.load(open(prediction_model_file, 'rb'))
        prediction =int( model.predict(user_query))
        print(prediction)
        print(type(prediction))
        prediction_prob = model.predict_proba(user_query)
        print(prediction_prob)

        if prediction == 0:
            prediction_probability=(round((prediction_prob[0,0]*100),2))
        else:
            prediction_probability =(round((prediction_prob[0,1]*100),2))

        print(prediction_probability)    



          
        

         # Inserting the form values to tbl_change_request
        cursor.execute(''' INSERT INTO tbl_user_travel_information(Age,Employment_Type,Graduate_Or_Not,Annual_Income,Family_Members,
    Chronic_Diseases,Frequent_Flyer,Ever_Travelled_Abroad,predictions,predictions_probability)
    values(?,?,?,?,?,?,?,?,?,?)''',
        (Age,Employment_Type,GraduateOrNot,AnnualIncome,FamilyMembers,ChronicDiseases,FrequentFlyer,EverTravelledAbroad,prediction,prediction_probability))     
        conn.commit()
    

    # 
    if prediction == 0:
        prediction_str ='The customer may not buy the product. Prediction probability is '+str(round((prediction_prob[0,0]*100),2))
    else:
        prediction_str ='The customer may buy the product. Prediction probability is '+str(round((prediction_prob[0,1]*100),2))
            
   

    # user_query['prediction']=prediction
    # print(user_query)

    #report_data=user_query.to_dict()

    #print(report_data)
    return render_template('report.html',prediction=prediction_str)


def batch_process(filename):
    #read uploaded file
    Data=pd.read_csv(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    print('Hi')
    #Process categorical variables
   # Data.replace({'MenClothing': 0, 'WomenClothing': 1,'OtherClothing':2},inplace=True)
    ## Data Pre-processing
    Data['Employment Type'] = Data['Employment Type'].map({'Government Sector':1,'Private Sector/Self Employed':0})
    Data['GraduateOrNot']  =Data['GraduateOrNot'].map({'Yes':0, 'No':1})
    Data['FrequentFlyer'] = Data['FrequentFlyer'].map({'Yes':0,'No':1})
    Data['EverTravelledAbroad'] = Data['EverTravelledAbroad'].map({'Yes':0,'No':1})


    #predict the values
    prediction = model.predict(Data)
    # Return the prediction results
    return prediction
    #return jsonify({'prediction': int(prediction)})
    

ALLOWED_EXTENSIONS = set(['csv'])

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/redirect_bulk',methods=['POST'])
def redirect_bulk():
    if request.method =='POST':
        if request.files:
            data_file=  request.files['data_file']
            print(data_file)

    return render_template('bulk.html')

@app.route('/transform',methods=['POST'])
def transform():
    if request.method =='POST':
        if request.files:
            data_file=request.files['data_file']
            #print(data_file)
            #return redirect(request.url)

            if not allowed_file(data_file.filename):
                print('that filename extension is not allowed')
                return redirect(request.url)
            else:
                filename = secure_filename(data_file.filename)# read file


            data_file.save(os.path.join(app.config['UPLOAD_FOLDER'],data_file.filename))
            print('csv file saved')
    print(filename)
    predictions=batch_process(filename) # batch prediction

    predictions_str=predictions.tolist() # convert array to list and then str
    data=pd.read_csv(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    print(data)
    data['Travel_Insurance_Prediction']=predictions
    data.to_csv('batch.csv',index=False)
    
    print(type(predictions_str))
    #print(type((filename)))
    #filename['Prediction']=predictions_str

    return render_template('bulk.html')        
  
    
if __name__ == '__main__':
    with open(prediction_model_file, 'rb') as f:
        model = pickle.load(f)

    from waitress import serve
    serve(app, host="0.0.0.0", port=8080)
