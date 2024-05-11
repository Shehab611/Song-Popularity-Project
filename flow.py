from sklearn.model_selection import train_test_split
import preprocessing as ps
import regressions as rs
import classifications as cs
import pandas as pd


class AppFlow:

    @staticmethod
    def __preprocessing(data, train=True):
        print("start in preprocessing =========")

        # Handle null Data
        if train:

            # Save the fill_values of The null Data to use them in test scenario when handling the null
            ps.PreProcessing.handle_nulls_in_train_data(data)
        else:

            # In Test Scenario handle the null Data by using the saved fill_values in training
            data = ps.PreProcessing.handle_nulls_in_test_data(data)

        print("Null Data Handled")

        # Check the type of data and return the y
        if 'PopularityLevel' in data.columns:
            model_type = 'classification'
            y = ps.PreProcessing.label_encode(data, 'PopularityLevel')
        else:
            model_type = 'regression'
            y = data['Popularity']

        # Encode Labeled Data
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
        print("Finish Pre processing ==============")
        return x, y, model_type

    @staticmethod
    def __do_regression(x, y, train=True):
        print("Start in regression ==============")
        # train and test
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, shuffle=True,
                                                            random_state=44)
        # apply regression models
        model_names, y_pred, mses, r2_scores = rs.Regressions.regression(x_train, y_train, x_test, y_test, train)

        # plot the results
        rs.Regressions.plotting(y_test, y_pred, mses, model_names, r2_scores)

    @staticmethod
    def __do_classification(x, y, train=True):
        print("Start in classification ==============")
        # train and test
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True,
                                                            random_state=42)
        # apply classification models
        models, acc, mse, tests, trains = cs.Classifications.classify(x_train, y_train, x_test, y_test, train)

        # plotting and print the results
        for i in range(len(models)):
            print(f'========== {models[i]} ==========')
            print(f"Accuracy: {acc[i]} %")
            print(f'MSE = {mse[i]}')
            print(f'Total Train Time = {trains[i]}')
            print(f'Total Test Time = {tests[i]}')
            print('================================')

        cs.Classifications.plotting(acc, trains, tests)

    @staticmethod
    def run_model(file, train=True):
        # Read the Data
        data = pd.read_csv(file)

        # Apply preprocessing techniques
        x, y, model_type = AppFlow.__preprocessing(data, train)

        # Check which model type should we run
        if model_type == 'regression':
            AppFlow.__do_regression(x, y, train)
        else:
            AppFlow.__do_classification(x, y, train)
