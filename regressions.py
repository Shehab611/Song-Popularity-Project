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
    __regressions_types = ['Linear Regression', 'Polynomial Regression', 'Ridge Regression', 'Lasso Regression',
                           'ElasticNet Regression', 'Support Vector Regression', 'Decision Tree Regression',
                           'Random Forest Regression', 'Gradient Boosting Regression',
                           'Polynomial with Lasso']

    @staticmethod
    def apply_regressions(x_train, y_train, x_test, y_test):

        methods = [
            Regressions.__LR, Regressions.__PR, Regressions.__RR, Regressions.__LAR,
            Regressions.__ER, Regressions.__SVR, Regressions.__DT, Regressions.__RF,
            Regressions.__GBR, Regressions.__poly_with_lasso
        ]
        returned_data = dict()

        for i in range(len(Regressions.__regressions_types)):
            returned_data[Regressions.__regressions_types[i]] = methods[i](x_train, y_train, x_test, y_test)

        return returned_data

    @staticmethod
    def __saveModel(model_name, clf):
        with open(f'regressions_models/{model_name}.pkl', 'wb') as file:
            pickle.dump(clf, file)

    @staticmethod
    def __LR(x_train, y_train, x_test, y_test):
        model = LinearRegression()
        model.fit(x_train, y_train)

        # Regressions.__saveModel(Regressions.__regressions_types[0], model)

        y_pred = model.predict(x_test)
        mse = mean_squared_error(y_test, y_pred)
        return mse, y_pred

    @staticmethod
    def __PR(x_train, y_train, x_test, y_test):
        poly = PolynomialFeatures(degree=2)
        x_poly_train = poly.fit_transform(x_train)
        x_poly_test = poly.transform(x_test)
        poly_model = LinearRegression()
        poly_model.fit(x_poly_train, y_train)

        # Regressions.__saveModel(Regressions.__regressions_types[1], poly_model)

        y_pred = poly_model.predict(x_poly_test)
        mse = mean_squared_error(y_test, y_pred)
        return mse, y_pred

    @staticmethod
    def __RR(x_train, y_train, x_test, y_test):
        model = Ridge(alpha=.005)
        model.fit(x_train, y_train)

        # Regressions.__saveModel(Regressions.__regressions_types[2], model)

        y_pred = model.predict(x_test)
        mse = mean_squared_error(y_test, y_pred)
        return mse, y_pred

    @staticmethod
    def __LAR(x_train, y_train, x_test, y_test):
        model = Lasso(alpha=.005)
        model.fit(x_train, y_train)

        # Regressions.__saveModel(Regressions.__regressions_types[3], model)

        y_pred = model.predict(x_test)
        mse = mean_squared_error(y_test, y_pred)
        return mse, y_pred

    @staticmethod
    def __ER(x_train, y_train, x_test, y_test):
        model = ElasticNet(alpha=.005, l1_ratio=1.0)
        model.fit(x_train, y_train)

        # Regressions.__saveModel(Regressions.__regressions_types[4], model)

        y_pred = model.predict(x_test)
        mse = mean_squared_error(y_test, y_pred)
        return mse, y_pred

    @staticmethod
    def __SVR(x_train, y_train, x_test, y_test):
        model = SVR(kernel='linear')
        model.fit(x_train, y_train)

        # Regressions.__saveModel(Regressions.__regressions_types[5], model)

        y_pred = model.predict(x_test)
        mse = mean_squared_error(y_test, y_pred)
        return mse, y_pred

    @staticmethod
    def __DT(x_train, y_train, x_test, y_test):
        model = DecisionTreeRegressor()
        model.fit(x_train, y_train)

        # Regressions.__saveModel(Regressions.__regressions_types[6], model)

        y_pred = model.predict(x_test)
        mse = mean_squared_error(y_test, y_pred)
        return mse, y_pred

    @staticmethod
    def __RF(x_train, y_train, x_test, y_test):
        model = RandomForestRegressor()
        model.fit(x_train, y_train)

        # Regressions.__saveModel(Regressions.__regressions_types[7], model)

        y_pred = model.predict(x_test)
        mse = mean_squared_error(y_test, y_pred)
        return mse, y_pred

    @staticmethod
    def __GBR(x_train, y_train, x_test, y_test):
        model = GradientBoostingRegressor()
        model.fit(x_train, y_train)

        # Regressions.__saveModel(Regressions.__regressions_types[8], model)

        y_pred = model.predict(x_test)
        mse = mean_squared_error(y_test, y_pred)
        return mse, y_pred

    @staticmethod
    def __poly_with_lasso(x_train, y_train, x_test, y_test):
        model = Pipeline([('poly', PolynomialFeatures(degree=2)),
                          ('lasso', Lasso(alpha=0.001))])
        model.fit(x_train, y_train)

        # Regressions.__saveModel(Regressions.__regressions_types[9], model)

        y_pred = model.predict(x_test)
        mse = mean_squared_error(y_test, y_pred)
        return mse, y_pred

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
