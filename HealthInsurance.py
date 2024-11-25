import os
import pandas as pd
import numpy as np
import pickle

class HealthInsurance:
    
    def __init__(self):
        self.home_path = os.path.join(os.path.abspath(''), 'scalers')
        with open(os.path.join(self.home_path, 'scaler_age.pkl'), 'rb') as file:
            self.age_scaler = pickle.load(file)
            
        with open(os.path.join(self.home_path, 'scaler_annual_premium.pkl'), 'rb') as file:
            self.annual_premium_scaler = pickle.load(file)
        
        with open(os.path.join(self.home_path, 'scaler_vintage.pkl'), 'rb') as file:
            self.vintage_scaler = pickle.load(file)
        
        with open(os.path.join(self.home_path, 'encoder_gender.pkl'), 'rb') as file:
            self.encode_gender_scaler = pickle.load(file)

        with open(os.path.join(self.home_path, 'encoder_veh_dmg.pkl'), 'rb') as file:
            self.encode_vehicle_damage = pickle.load(file)

        with open(os.path.join(self.home_path, 'veh_age_encoder.pkl'), 'rb') as file:
            self.encode_vehicle_age = pickle.load(file)
        
        with open(os.path.join(self.home_path, 'target_region_code_encoder.pkl'), 'rb') as file:
            self.target_region_code_scaler = pickle.load(file)
        
        with open(os.path.join(self.home_path, 'target_policy_sales_channel_encoder.pkl'), 'rb') as file:
            self.target_policy_sales_channel_scaler = pickle.load(file)
              
    def data_cleaning(self, df1):
        #change column names to snake_case
        df1.columns = df1.columns.str.lower().str.replace(' ', '_')
        
        #Remove non-driving license values
        df1 = df1[df1['driving_license'] == 1].drop('driving_license', axis=1)

        return df1   

    def data_preparation(self, df2):

        df2['age'] = self.age_scaler.fit_transform(df2[['age']])
        df2['annual_premium'] = self.annual_premium_scaler.fit_transform(df2[['annual_premium']])
        df2['vintage'] = self.vintage_scaler.fit_transform(df2[['vintage']])
        df2['gender'] = self.encode_gender_scaler.fit_transform(df2[['gender']])
        df2['vehicle_damage_yes'] = self.encode_vehicle_damage.fit_transform(df2[['vehicle_damage']])     
        df2['vehicle_age_numeric'] = df2['vehicle_age'].map(self.encode_vehicle_age)    
        df2.loc[:,'region_code'] = df2['region_code'].map(self.target_region_code_scaler)               
        df2.loc[:,'policy_sales_channel'] = df2['policy_sales_channel'].map(self.target_policy_sales_channel_scaler)             
        cols_selected = ['vintage', 'annual_premium', 'age', 'region_code', 'vehicle_damage_yes', 
                            'policy_sales_channel', 'previously_insured']

        return df2[cols_selected]
    
    def get_prediction( self, model, original_data, test_data, final_data ):
        # model prediction
        pred = model.predict_proba( test_data )

        # join prediction into original data
        original_data['prediction'] = pred[:, 1]

        final_data = final_data.merge(original_data[['id', 'prediction']], on='id', how='left')

        # Rename the 'prediction' column to 'score'
        final_data.rename(columns={'prediction': 'score'}, inplace=True)

        # Set columns names as camelCase
        final_data.columns = final_data.columns.str.lower().str.replace(' ', '_')
        
        # Set 'Score' to 0 where 'Driving_License' is 0
        final_data.loc[final_data['driving_license'] == 0, 'score'] = 0

        return final_data.to_json( orient='records', date_format='iso' )