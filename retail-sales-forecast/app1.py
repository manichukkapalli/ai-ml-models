from flask import Flask, request,jsonify,render_template,redirect
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np
import pickle
import os
import yaml
import pyodbc
import xgboost
import waitress
import logging

logging.basicConfig(filename="log.txt", format="%(asctime)s %(message)s",filemode='w')

logger = logging.getLogger()

# Load config file
with open('config/config.yaml') as file:
    config= yaml.safe_load(file)   

# get the db config
db_config=config['db_config']

# Name of model file
prediction_model_file='Updated_TunedXGB_Model.pkl'


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
def home():
    return render_template('index.html')



@app.route('/predict', methods=['POST'])
def predict():
    ProductCategory=str(request.form.get('ProductCategory'))
    MonthlyNominalGDPIndexinMillion=float(request.form.get('MonthlyNominalGDPIndexinMillion'))
    MonthlyRealGDPIndexinMillion=float(request.form.get('MonthlyRealGDPIndexinMillion'))
    CPI=float(request.form.get('CPI'))
    unemploymentrate=float(request.form.get('unemploymentrate'))	
    CommercialBankInterestRateonCreditCardPlans=float(request.form.get('CommercialBankInterestRateonCreditCardPlans'))	
    FinanceRateonPersonalLoansatCommercialBanks24MonthLoan=float(request.form.get('FinanceRateonPersonalLoansatCommercialBanks24MonthLoan'))	
    Earningsorwagesindollarsperhour=float(request.form.get('Earningsorwagesindollarsperhour'))	
    CottonMonthlyPriceUScentsperPoundlbs=float(request.form.get('CottonMonthlyPriceUScentsperPoundlbs'))	
    Changein=float(request.form.get('Changein'))	
    Averageuplandplantedmillionacres=float(request.form.get('Averageuplandplantedmillionacres'))	
    yieldperharvestedacre=float(request.form.get('yieldperharvestedacre'))	
    Millusein480lbnetwerightinmillionbales=float(request.form.get('Millusein480lbnetwerightinmillionbales'))	
    Exports=float(request.form.get('Exports'))	
    SeaLevelPressavg=float(request.form.get('SeaLevelPressavg'))
    Visibilityavg=float(request.form.get('Visibilityavg'))
    Windavg=float(request.form.get('Windavg'))
    Precipsum=float(request.form.get('Precipsum'))
    Event=float(request.form.get('Event'))
    FederalHoliday=float(request.form.get('FederalHoliday'))
    
    # Create user query for prediction
    user_query={'ProductCategory':[ProductCategory],
                    'MonthlyNominalGDPIndexinMillion':[MonthlyNominalGDPIndexinMillion],
                    'MonthlyRealGDPIndexinMillion':[MonthlyRealGDPIndexinMillion],
                    'CPI':[CPI],
                    'unemploymentrate':[unemploymentrate],
                    'CommercialBankInterestRateonCreditCardPlans':[CommercialBankInterestRateonCreditCardPlans],
                    'FinanceRateonPersonalLoansatCommercialBanks24MonthLoan':[FinanceRateonPersonalLoansatCommercialBanks24MonthLoan],
                    'Earningsorwagesindollarsperhour':[Earningsorwagesindollarsperhour],
                    'CottonMonthlyPriceUScentsper':[CottonMonthlyPriceUScentsperPoundlbs],
                    'Changein':[Changein],
                    'Averageuplandplantedmillionacres':[Averageuplandplantedmillionacres],
                    'yieldperharvestedacre':[yieldperharvestedacre],
                    'Millusein480lbnetw':[Millusein480lbnetwerightinmillionbales],
                    'Exports':[Exports],
                    'SeaLevelPressavg':[SeaLevelPressavg],
                    'Visibilityavg':[Visibilityavg],
                    'Windavg':[Windavg],
                    'Precipsum':[Precipsum],
                    'Event':[Event],
                    'FederalHoliday':[FederalHoliday],
        }
    user_query=pd.DataFrame.from_dict(user_query)
        ## Data Pre-processing
    user_query['ProductCategory'] = user_query['ProductCategory'].map({'MenClothing':0, 'WomenClothing':1,'OtherClothing':2})
        

        # Prediction
        #model = pickle.load(open(prediction_model_file, 'rb'))
    prediction = model.predict(user_query)
        #output = np.round(prediction[0], 2)
    Predicted_Sales=str(round(prediction[0],2))
    
        #connecting the database
    try:
        conn = pyodbc.connect(**db_config)
        cursor=conn.cursor()
    except Exception as e:
        logger.exception("Exception occur while code execution"+ str(e))
    
        # Inserting the form values to tbl_retail_Sales
        try:
            cursor.execute(''' INSERT INTO tbl_retail_sales(ProductCategory,MonthlyNominalGDPIndexinMillion,MonthlyRealGDPIndexinMillion,
            CPI,unemploymentrate,CommercialBankInterestRateonCreditCardPlans,FinanceRateonPersonalLoansatCommercialBanks24MonthLoan,
            Earningsorwagesindollarsperhour,CottonMonthlyPriceUScentsperPoundlbs,Changein,
            Averageuplandplantedmillionacres, yieldperharvestedacre,Millusein480lbnetwerightinmillionbales,Exports,SeaLevelPressavg,
            Visibilityavg,Windavg,Precipsum,Event,FederalHoliday,SalesInThousandDollars)
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
            (ProductCategory,MonthlyNominalGDPIndexinMillion,MonthlyRealGDPIndexinMillion,
            CPI,unemploymentrate,CommercialBankInterestRateonCreditCardPlans,FinanceRateonPersonalLoansatCommercialBanks24MonthLoan,
            Earningsorwagesindollarsperhour,CottonMonthlyPriceUScentsperPoundlbs,Changein,
            Averageuplandplantedmillionacres, yieldperharvestedacre,Millusein480lbnetwerightinmillionbales,Exports,SeaLevelPressavg,
            Visibilityavg,Windavg,Precipsum,Event,FederalHoliday,Predicted_Sales))
            conn.commit()
            return render_template('Prediction_Report.html', prediction='The predicted sales value for given input is $ {}'.format(Predicted_Sales))
        except Exception as e:
            logger.exception("Exception occur while predicting the sales value"+ str(e))
            
def batch_process(filename):
    try:
        #read uploaded file
                Data=pd.read_csv(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        #Process categorical variables
                Data.replace({'MenClothing': 0, 'WomenClothing': 1,'OtherClothing':2},inplace=True)
        #predict the values
                prediction = model.predict(Data)
        # Return the prediction results
                return prediction
        #return jsonify({'prediction': int(prediction)})
    except Exception as e: 
        logger.exception("Exception occur while uploading the file "+ str(e))

    

ALLOWED_EXTENSIONS = set(['csv'])
def allowed_file(filename):
    try:
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    except Exception as e:
        logger.exception("Exception occur while uploading the wrong file extension "+ str(e))


@app.route('/redirect_bulk',methods=['POST'])
def redirect_bulk():
    try:
        if request.method =='POST':
            if request.files:
                data_file=  request.files['data_file']
                print(data_file)
    except Exception as e:
        logger.exception("Exception occured during the file upload "+ str(e))  
        return render_template('bulk_prediction.html')  



@app.route('/transform',methods=['POST'])
def transform():
    try:
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
        data['SalesInThousandDollars']=predictions
        data.to_csv('batch.csv',index=False)
        print(type(predictions_str))
        #print(type((filename)))
        #filename['Prediction']=predictions_str
    except Exception as e:
        logger.exception("Exception occured during the predicitng on bulk data "+ str(e))  
        return render_template('bulk_prediction.html') 
if __name__ == '__main__':
    with open(prediction_model_file, 'rb') as f:
       model = pickle.load(f)
    
    app.run(port=8080, debug=True)
    #from waitress import serve
    #serve(app, host="0.0.0.0", port=8080)
