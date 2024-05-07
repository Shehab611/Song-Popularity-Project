from sklearn.model_selection import train_test_split
import preprocessing as ps
import regressions as rs
import classifications as cs
import pandas as pd


class AppFlow:

    @staticmethod
    def __preProcessing(data):
        if 'PopularityLevel' in data.columns:
            model_type = 'classification'
            y = ps.PreProcessing.label_encode(data, 'PopularityLevel')
        else:
            model_type = 'regression'
            y = data['Popularity']

        data['Song'] = ps.PreProcessing.label_encode(data, 'Song')
        data['Spotify Link'] = ps.PreProcessing.label_encode(data, 'Spotify Link')
        data['Song Image'] = ps.PreProcessing.label_encode(data, 'Song Image')
        data['Spotify URI'] = ps.PreProcessing.label_encode(data, 'Spotify URI')
        data['Album'] = ps.PreProcessing.label_encode(data, 'Album')

        # change dates
        data['year'], data['month'] = ps.PreProcessing.get_date(data, 'Album Release Date')
        ps.PreProcessing.drop_column(data, 'Album Release Date')

        # double encode the lists
        data['Artist Names'] = ps.PreProcessing.encode_lists(data, 'Artist Names')
        data['Artist(s) Genres'] = ps.PreProcessing.encode_lists(data, 'Artist(s) Genres')
        data['Artist Names'] = ps.PreProcessing.pipline_encode(data, 'Artist Names')
        data['Artist(s) Genres'] = ps.PreProcessing.pipline_encode(data, 'Artist(s) Genres')

        # normalize
        x = ps.PreProcessing.normalize(data)

        # feature selection

        # drop uniques and low correlated data
        low_correlated_features = ['Song', 'Album', 'Artist Names', 'Artist(s) Genres', 'Hot100 Rank', 'Spotify Link',
                                   'Spotify URI', 'Song Image', 'Acousticness', 'Instrumentalness', 'Liveness',
                                   'Tempo', 'Valence', 'Key', 'Mode', 'Time Signature', 'year', 'month']

        for feature in low_correlated_features:
            ps.PreProcessing.drop_column(data, feature)

        return x, y, model_type

    @staticmethod
    def __doRegression(x, y):
        # train and test
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, shuffle=True,
                                                            random_state=44)
        # apply regression models
        mse = rs.Regressions.apply_regressions(x_train, y_train, x_test, y_test)

        # plotting and print the results
        y_pred, mses, model_names = [], [], []

        for key, value in mse.items():
            print(f'{key} mse ={value[0]}')
            y_pred.append(value[1])
            mses.append(value[0])
            model_names.append(key)

        rs.Regressions.plotting(y_test, y_pred, mses, model_names)

    @staticmethod
    def __doClassification(x, y):
        # train and test
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True,
                                                            random_state=42)
        # apply classification models
        mse = cs.Classifications.apply_classifications(x_train, y_train, x_test, y_test)

        models, accuracies, test_times, train_times = [], [], [], []
        # plotting and print the results
        for key, value in mse.items():
            print(f'========== {key} ==========')
            print(f"Accuracy: {value[0]}%")
            print(f'MSE = {value[1]}')
            print(f'Total Train Time = {value[2]}')
            print(f'Total Test Time = {value[3]}')
            print('================================')
            accuracies.append(value[0])
            models.append(value[1])
            train_times.append(value[2])
            test_times.append(value[3])

        cs.Classifications.plotting(models, accuracies, train_times, test_times)

    @staticmethod
    def runModel(file):
        # Read the Data
        data = pd.read_csv(file)

        # Apply preprocessing techniques
        x, y, model_type = AppFlow.__preProcessing(data)

        # Check which model type should we run

        if model_type == 'regression':
            AppFlow.__doRegression(x, y)
        else:
            AppFlow.__doClassification(x, y)
