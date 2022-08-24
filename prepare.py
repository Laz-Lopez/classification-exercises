import numpy as np
import pandas as pd
import os
import acquire
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer


def prep_iris(iris):
    iris = iris.drop(columns=['species_id','measurement_id'])
    iris = iris.rename(columns={'species_name':'species'})
    dummy_iris = pd.get_dummies(iris.species, drop_first=True)
    iris = pd.concat([iris, dummy_iris], axis=1)
    return iris


def prep_titanic(titanic):
    titanic = titanic.drop(columns=['embarked','class', 'age','deck'])
    dummy_df = pd.get_dummies(data=titanic[['sex','embark_town']], drop_first=True)
    titanic = pd.concat([titanic, dummy_df], axis=1)
    
    return titanic

def prep_telco(df):
   ''' this function takes in telco df and removes colums, switches all values to 0/1 and leave 
    some variables with numbers 1-4 , it also removes 11 messed up entries with in the data and gets rid of null values its messy but it works.'''


    #.info shows that test_1 total_charges which shows numbers in .head is listed as an object here we are going to 
    # change it to a float then verify change 
    df['total_charges'] = pd.to_numeric(df['total_charges'], errors='coerce')
        # the change from object to float shows that there are missing values 11 missing values so we are going to go 
        # ahead see what is up with that
    df[df['total_charges'].isnull()]
#added this in after the fact to remove the word automatic from payment type it only impacts on 
# of my test df but to lasy to mess with later 
    df['payment_type'] = df['payment_type'].str.replace(' (automatic)', '', regex=False)
# the change from object to float shows that there are missing values 11 missing values so we are going to go ahead 
# see what is up with that
    df[df['total_charges'].isnull()]
#added this in after the fact to remove the word automatic from payment type it only impacts on of my test df but to 
# lasy to mess with later 
    df['payment_type'] = df['payment_type'].str.replace(' (automatic)', '', regex=False)
#dropped these 11 entries because they make no sense. 
    df.dropna(inplace=True)
# there  appears to be duplicate col Interrnet service type, payment type contract type.
# my hypos also do not focus on any of the customer demoraphic data so i am going to drop col for 
#gender, senior_citizens,partner, dependants. I will also be splitting the data in to 2 groups one keeping _id col and deleting 
# their related obj col and the other will do the inverse for expoloration purposes to see what looks prettier
#I will also be dropping Unamed:0 because i should not have created a new index when pulling from sql and also because it serves me no purpose
#was going to drop customer Id but after rereading the assingment I have to be able to predict which customers are going to churn so it will stay 

######cut the split out found it much easier to model and graph with numbers only 
#    ''' test_no_num=df.drop(['Unnamed: 0' ,'internet_service_type_id','payment_type_id','contract_type_id', 'gender', 'senior_citizen', 
#     'partner', 'dependents'], axis=1)'''
    df=df.drop([ 'Unnamed: 0','gender', 'senior_citizen', 'partner', 'dependents','internet_service_type',
    'payment_type','contract_type'], axis=1)
# figured out the issues so removeing yes and no to 1, 0 and removing no phone service no internet service from num only df
#also adding col for sum of services offered on num only df 




    df = df.replace(to_replace = ['Yes','No'],value = [1,0])
    df = df.replace(to_replace = ['True','False'],value = [1,0])
    df = df.replace(to_replace = ['No phone service','No internet service'],value = [0,0])
#had to add this part in after one of my graphs looked horrible it gives a better read 
    df["internet_service_type_id"] = df["internet_service_type_id"].replace(to_replace = [2,3],value = [1,0])
    df["phone_service"] = df["phone_service"].astype(str).astype(int)
    df["multiple_lines"] = df["multiple_lines"].astype(str).astype(int)
    df["online_security"] = df["online_security"].astype(str).astype(int)
    df["online_backup"] = df["online_backup"].astype(str).astype(int)
    df["device_protection"] = df["device_protection"].astype(str).astype(int)
    df["tech_support"] = df["tech_support"].astype(str).astype(int)
    df["streaming_tv"] = df["streaming_tv"].astype(str).astype(int)
    df["streaming_movies"] = df["streaming_movies"].astype(str).astype(int)
    df["paperless_billing"] = df["paperless_billing"].astype(str).astype(int)

    df['number_services'] = (df["phone_service"] + df["multiple_lines"]
    +df["online_security"]+df["online_backup"]+df["device_protection"]
        +df["tech_support"]+df["streaming_tv"]+df["streaming_movies"]
        +df["paperless_billing"]+df["internet_service_type_id"])

#had to add this part in after one of my graphs looked horrible it gives a better read 

    df['monthly_avg'] = df["monthly_charges"]>df["monthly_charges"].mean()
    df['monthly_avg'] = df['monthly_avg'].replace(to_replace = [True,False],value = [1,0])
    return df




def train_validate_test_split(df, target, seed=123):
    '''
    This function takes in a dataframe, the name of the target variable
    (for stratification purposes), and an integer for a setting a seed
    and splits the data into train, validate and test. 
    Test is 20% of the original dataset, validate is .30*.80= 24% of the 
    original dataset, and train is .70*.80= 56% of the original dataset. 
    The function returns, in this order, train, validate and test dataframes. 
    '''
    train_validate, test = train_test_split(df, test_size=0.2, 
                                            random_state=seed, 
                                            stratify=df[target])
    train, validate = train_test_split(train_validate, test_size=0.3, 
                                       random_state=seed,
                                       stratify=train_validate[target])
    return train, validate, test
