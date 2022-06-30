# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 19:58:29 2022

@author: 105050802
"""

import json
import joblib
from typing import Dict, List

import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class ModelHelper:
    
    @classmethod
    def convert_json_to_pd_series(cls,
                                  json_string: str,
                                  data_contains_status: bool) -> List[pd.Series]:

        result_data = []
        try:
            json_dict = json.loads(json_string)
        except json.JSONDecodeError:
            cls._logger.exception(
                'Could not parse data to convert to Pandas series.')
            return result_data
        col_names = [*json_dict]

        for _, col_name in enumerate(col_names):
            col_values = []
            col_timestamps = []
            col_status_values = []
            col_data = json_dict[col_name]['data']
            for data in col_data:
                col_values.append(data['val'])
                col_timestamps.append(int((data['ts'] * 1000) + 0.01)) # converting to ms
                if data_contains_status:
                    col_status_values.append(data['status'])
            col_timestamps = pd.to_datetime(col_timestamps,
                                            unit='ms', utc=True)
            col_series = pd.Series(col_values,
                                   index=col_timestamps,
                                   name=col_name)
            if pd.api.types.is_numeric_dtype(col_series) is False:
                raise ValueError(
                    f'ModelHelper: Cannot convert {col_name} value to numeric')

            result_data.append(col_series.replace('', np.nan))
            if data_contains_status:
                col_series = pd.Series(col_status_values,
                                       index=col_timestamps,
                                       name=f'{col_name}_status')
                result_data.append(col_series.replace('', np.nan))
        return result_data
    
    @staticmethod
    def convert_pd_series_list_to_df(series_list: List[pd.Series]) -> pd.DataFrame:
        return pd.concat(series_list, axis=1)
    
    @staticmethod
    def convert_df_to_json(result_df: pd.DataFrame):
        results = {}
        timestamps = result_df.index.view('int64') / 1_000_000_000
        col_names = result_df.columns[~result_df.columns.str.contains('_status')]
        for col_name in col_names:
            data = []
            col_status_exists = False
            if col_name+'_status' in result_df.columns:
                col_status_exists = True
            for i, val in enumerate(result_df[col_name]):
                if np.isnan(val):
                    continue
                result = {'ts': timestamps[i], 'val': val}
                if col_status_exists:
                    result['status'] = int(result_df[col_name+'_status'][i])
                data.append(result)
            results[col_name] = {'data': data}
        return json.dumps(results)


class DummyAD:
    
    def __init__(self):
        self._model = None
        self._scaler = None
    
    def train_model(self, json_string: str):
        pd_series = ModelHelper.convert_json_to_pd_series(json_string, False)
        data_df = ModelHelper.convert_pd_series_list_to_df(pd_series)
        data_df.ffill(inplace=True)
        data_df.dropna(inplace=True)
        if data_df.shape[0] < 100:
            raise ValueError('Too short training data.')
        no_tags = data_df.shape[1]
        train_set, test_set = train_test_split(data_df, test_size=0.2)
        
        scaler = StandardScaler()
        train_set = scaler.fit_transform(train_set.values)
        self._scaler = scaler
        
        model = Sequential()
        model.add(Dense(no_tags/2, input_dim=no_tags, activation='relu'))
        model.add(Dense(no_tags, activation='linear'))
        model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
        
        model.fit(train_set, 
                  train_set,
                  epochs=2,
                  batch_size=32)
        
        self._model = model
    
    def export_model(self,
                     path: str):
        if self._model is None:
            raise ValueError('Model not trained.')
        self._model.save(f'{path}/model/ad_model')
        state = {'scaler': self._scaler}
        joblib.dump(state, f'{path}/model/ad_state.joblib')
    
    def import_model(self,
                     path: str):
        state = joblib.load(f'{path}/ad_state.joblib')
        self._scaler = state['scaler']
        self._model = keras.models.load_model(f'{path}/ad_model')

    def predict(self,model:Sequential, json_str: str) -> str:
#         if self._model is None:
#             raise ValueError('Model is not trained.')
        pd_series = ModelHelper.convert_json_to_pd_series(json_str, False)
        data_df = ModelHelper.convert_pd_series_list_to_df(pd_series)
        data_df.ffill(inplace=True)
        data_df.dropna(inplace=True)
        if data_df.shape[0] < 20:
            raise ValueError('Too short data for prediction.')
        data_df = data_df.iloc[19:]
#         predictions = self._model.predict(data_df.values)
        predictions = model.predict(data_df.values)
        predictions = pd.DataFrame(predictions,
                                   columns=data_df.columns,
                                   index=data_df.index)
        return ModelHelper.convert_df_to_json(predictions) 

if __name__ == '__main__':
    df = pd.DataFrame(np.random.randint(0,100,size=(150, 3)),
                      columns=list('ABC'),
                      index = pd.date_range('2020-01-01', periods=150, freq='5T'))
    for col in df:
        df.loc[df.sample(frac=0.2).index, col] = np.nan
    df_json = ModelHelper.convert_df_to_json(df)
    model = DummyAD()
    model.train_model(df_json)
    model.export_model('.')
    model = DummyAD()
    model.import_model('.')
    result = model.predict(df_json)
