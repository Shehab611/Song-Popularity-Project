from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import pandas as pd
import ast


class PreProcessing:

    @staticmethod
    def __save_fill_null_values(fill_values, typee):
        with open(f'{typee}_fill_nulls.txt', 'w') as file:
            file.write(str(fill_values))

    @staticmethod
    def __get_fill_null_values(typee):
        with open(f'{typee}_fill_nulls.txt', 'r') as file:
            fill_values = ast.literal_eval(file.read())
        return fill_values

    @staticmethod
    def handle_nulls_in_train_data(train_data):
        fill_values = {}
        for column in train_data.columns:
            if train_data[column].dtype == 'object':
                fill_value = train_data[column].mode()[0]
                fill_values[column] = fill_value
                train_data.fillna({column: fill_value}, inplace=True)
            else:
                fill_value = train_data[column].median()
                fill_values[column] = fill_value
                train_data.fillna({column: fill_value}, inplace=True)
        if 'PopularityLevel' in train_data.columns:
            model_type = 'classification'
        else:
            model_type = 'regression'
        PreProcessing.__save_fill_null_values(fill_values, model_type)

    @staticmethod
    def handle_nulls_in_test_data(test_data):
        if 'PopularityLevel' in test_data.columns:
            model_type = 'classification'
        else:
            model_type = 'regression'

        fill_values = PreProcessing.__get_fill_null_values(model_type)

        for column in test_data.columns:
            if test_data[column].dtype == 'object':
                test_data[column].fillna(fill_values[column], inplace=True)
            else:
                test_data[column].fillna(fill_values[column], inplace=True)
        return test_data

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
