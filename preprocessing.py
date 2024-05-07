from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import ast


class PreProcessing:
    @staticmethod
    def drop_column(data, column_name):
        data.drop(column_name, inplace=True, axis=1)

    @staticmethod
    def label_encode(data, column_name):
        return LabelEncoder().fit_transform(data[column_name])

    @staticmethod
    def get_date(data, column_name):
        date = pd.to_datetime(data[column_name], format='mixed')
        year = date.dt.year
        month = date.dt.month
        return year, month

    @staticmethod
    def encode_lists(data, column_name):
        encoder = LabelEncoder()
        encoded_data = []
        for row in data[column_name]:
            row = ast.literal_eval(row)
            encoded_row = encoder.fit_transform(row)
            encoded_data.append(list(encoded_row))
        return encoded_data

    @staticmethod
    def pipline_encode(data, column_name):
        encoded_data = []
        for row in data[column_name]:
            encoded_string = ''.join(map(str, row))
            if encoded_string == '':
                encoded_string = '0'
            encoded_data.append(int(encoded_string))
        return encoded_data

    @staticmethod
    def normalize(data):
        all_columns = data.columns.tolist()
        features = [col for col in all_columns if col != 'Popularity' and col != 'PopularityLevel']
        x = pd.DataFrame(MinMaxScaler().fit_transform(data[features]), columns=features)
        return x
