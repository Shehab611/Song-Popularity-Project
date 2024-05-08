import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import pickle


class Regressions:

    @staticmethod
    def __save_model(model_name, clf):
        with open(f'regressions_models/{model_name}.pkl', 'wb') as file:
            pickle.dump(clf, file)

    @staticmethod
    def __get_saved_model(model_name):
        with open(f'regressions_models/{model_name}.pkl', 'rb') as file:
            return pickle.load(file)

    @staticmethod
    def plotting(y_test, y_pred, mse, model_name):
        fig = plt.figure(figsize=(11, 11))
        rows, columns = 4, 3
        for i in range(1, rows * columns + 1):
            if i == 11:
                break
            ax = fig.add_subplot(rows, columns, i)
            ax.scatter(y_test, y_pred[i - 1])
            ax.plot(np.arange(0, 100), np.arange(0, 100), color='r', linestyle='--')
            ax.set_title(f'{model_name[i - 1]}\nMSE = {mse[i - 1]}')
            ax.set_xlabel('Actual values')
            ax.set_ylabel('Predicted values')

        plt.subplots_adjust(left=0.1, bottom=0.07, right=0.9, top=0.925, wspace=0.4, hspace=1)
        plt.show()

    @staticmethod
    def regression(x_train, y_train, x_test, y_test, train=True):
        models_names = ['Linear Regression', 'Polynomial Regression', 'Ridge Regression', 'Lasso Regression',
                        'ElasticNet Regression', 'Support Vector Regression', 'Decision Tree Regression',
                        'Random Forest Regression', 'Gradient Boosting Regression', 'Polynomial with Lasso']

        models = [LinearRegression(), PolynomialFeatures(degree=2), Ridge(alpha=.005),
                  Lasso(alpha=.005), ElasticNet(alpha=.005, l1_ratio=1.0), SVR(kernel='linear'),
                  DecisionTreeRegressor(),RandomForestRegressor(), GradientBoostingRegressor(),
                  Pipeline([('poly', PolynomialFeatures(degree=2)),('lasso', Lasso(alpha=0.001))])]

        y_preds, mse = [], []

        for i in range(len(models_names)):
            if models_names[i] != 'Polynomial Regression':
                if train:
                    working_clf = models[i]
                    working_clf.fit(x_train, y_train)
                    Regressions.__save_model(models_names[i], working_clf)
                else:
                    working_clf = Regressions.__get_saved_model(models_names[i])

                y_pred = working_clf.predict(x_test)
                mse.append(mean_squared_error(y_test, y_pred))
                y_preds.append(y_pred)
            else:
                working_clf = models[i]
                x_poly_train = working_clf.transform(x_train)
                x_poly_test = working_clf.transform(x_test)
                if train:
                    poly_model = models[i - 1]
                    poly_model.fit(x_poly_train, y_train)
                    Regressions.__save_model(models_names[i], poly_model)
                else:
                    poly_model = Regressions.__get_saved_model(models_names[i])

                y_pred = poly_model.predict(x_poly_test)
                mse.append(mean_squared_error(y_test, y_pred))
                y_preds.append(y_pred)

        return models_names, y_preds, mse,
