import numpy as np
import pandas as pd
import env
import os

def connect(db):
   
    return f'mysql+pymysql://{env.user}:{env.password}@{env.host}/{db}'

def get_titanic_data():

    filename = 'titanic.csv'
    if os.path.isfile(filename):
        return pd.read_csv(filename)
    else:
        df = pd.read_sql('''SELECT 
                            * FROM passengers;''', connect('titanic_db'))
        df.to_csv(filename)
        return df

def get_iris_data():
  
    filename = 'iris.csv'
    if os.path.isfile(filename):
        return pd.read_csv(filename)
    else:
        df = pd.read_sql('''SELECT * 
                            FROM measurements 
                            JOIN species 
                                USING(species_id);''', connect('iris_db'))
        df.to_csv(filename)
        return df

def get_telco_data():
    
  
    filename = 'telco.csv'
    if os.path.isfile(filename):
        return pd.read_csv(filename)
    else:
        query = '''
            SELECT * FROM customers
            JOIN contract_types 
                USING (contract_type_id)
            JOIN payment_types 
                USING (payment_type_id)
            JOIN internet_service_types 
                USING (internet_service_type_id);  
        '''
        df = pd.read_sql(query, connect('telco_churn')) 
        df.to_csv(filename)
        return df