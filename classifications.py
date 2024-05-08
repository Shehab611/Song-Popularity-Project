from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import time
from matplotlib import pyplot as plt
import pickle


class Classifications:
    @staticmethod
    def __save_model(model_name, clf, train_time):
        with open(f'classification_models/{model_name}.pkl', 'wb') as file:
            pickle.dump(clf, file)
        with open(f'training_time/{model_name}.txt') as file:
            file.write(train_time)

    @staticmethod
    def __get_saved_model(model_name):
        with open(f'classification_models/{model_name}.pkl', 'rb') as file:
            model = pickle.load(file)
        with open(f'training_time/{model_name}.txt', 'rb') as file:
            train_time = file.read()
        return model, train_time

    @staticmethod
    def gridSearch(params, clf, x_train, y_train, is_cat=False):
        grid_search = GridSearchCV(clf, params, cv=5)
        if is_cat:
            grid_search.fit(x_train, y_train, cat_features=None, verbose=False)
        else:
            grid_search.fit(x_train, y_train)
        print("Best Hyperparameters:", grid_search.best_params_)
        return grid_search.best_estimator_

    @staticmethod
    def Gaussian(x_train, y_train, x_test, y_test):
        train_start_time = time.time()
        clf = GaussianNB()
        param_grid = {
            'priors': [None, [0.25, 0.25, 0.5], [0.1, 0.5, 0.4]],
            'var_smoothing': [1e-9, 1e-8, 1e-7]
        }
        clf = Classifications.gridSearch(param_grid, clf, x_train, y_train)
        clf.fit(x_train, y_train)

        # Classifications.saveModel(Classifications.classifications_types[8], clf)

        # Calculate the total train time
        total_train_time = time.time() - train_start_time

        test_start_time = time.time()
        y_pred = clf.predict(x_test)
        # Calculate the total test time
        total_test_time = time.time() - test_start_time
        mse = mean_squared_error(y_test, y_pred)
        acc = accuracy_score(y_test, y_pred)
        return acc * 100, mse, total_train_time, total_test_time

    @staticmethod
    def XGB(x_train, y_train, x_test, y_test):
        train_start_time = time.time()
        clf = XGBClassifier()
        param_grid = {
            'max_depth': [3, 4, 5],
            'learning_rate': [0.1, 0.01, 0.001],
            'n_estimators': [100, 200, 300],
            'gamma': [0, 0.1, 0.2],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'use_label_encoder': [True, False]
        }
        clf = Classifications.gridSearch(param_grid, clf, x_train, y_train)
        clf.fit(x_train, y_train)

        # Classifications.saveModel(Classifications.classifications_types[9], clf)

        # Calculate the total train time
        total_train_time = time.time() - train_start_time

        test_start_time = time.time()
        y_pred = clf.predict(x_test)
        # Calculate the total test time
        total_test_time = time.time() - test_start_time
        mse = mean_squared_error(y_test, y_pred)
        acc = accuracy_score(y_test, y_pred)
        return acc * 100, mse, total_train_time, total_test_time


    @staticmethod
    def ADABoost(x_train, y_train, x_test, y_test):
        train_start_time = time.time()
        clf = AdaBoostClassifier()
        param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.1, 0.01, 0.001]
        }
        clf = Classifications.gridSearch(param_grid, clf, x_train, y_train)
        clf.fit(x_train, y_train)

        # Classifications.saveModel(Classifications.classifications_types[9], clf)

        # Calculate the total train time
        total_train_time = time.time() - train_start_time

        test_start_time = time.time()
        y_pred = clf.predict(x_test)
        # Calculate the total test time
        total_test_time = time.time() - test_start_time
        mse = mean_squared_error(y_test, y_pred)
        acc = accuracy_score(y_test, y_pred)
        return acc * 100, mse, total_train_time, total_test_time

    @staticmethod
    def GBM(x_train, y_train, x_test, y_test):
        train_start_time = time.time()
        clf = GradientBoostingClassifier()
        param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.1, 0.01, 0.001],
            'max_depth': [3, 4, 5]
        }
        clf = Classifications.gridSearch(param_grid, clf, x_train, y_train)
        clf.fit(x_train, y_train)

        # Classifications.saveModel(Classifications.classifications_types[9], clf)

        # Calculate the total train time
        total_train_time = time.time() - train_start_time

        test_start_time = time.time()
        y_pred = clf.predict(x_test)
        # Calculate the total test time
        total_test_time = time.time() - test_start_time
        mse = mean_squared_error(y_test, y_pred)
        acc = accuracy_score(y_test, y_pred)
        return acc * 100, mse, total_train_time, total_test_time

    @staticmethod
    def LightGBM(x_train, y_train, x_test, y_test):
        train_start_time = time.time()
        clf = LGBMClassifier()
        param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.1, 0.01, 0.001],
            'max_depth': [3, 4, 5]
        }
        clf = Classifications.gridSearch(param_grid, clf, x_train, y_train)
        clf.fit(x_train, y_train)

        # Classifications.saveModel(Classifications.classifications_types[9], clf)

        # Calculate the total train time
        total_train_time = time.time() - train_start_time

        test_start_time = time.time()
        y_pred = clf.predict(x_test)
        # Calculate the total test time
        total_test_time = time.time() - test_start_time
        mse = mean_squared_error(y_test, y_pred)
        acc = accuracy_score(y_test, y_pred)
        return acc * 100, mse, total_train_time, total_test_time

    @staticmethod
    def plotting(model_names, class_accuracy, train_time, test_time):
        num_models = len(model_names)
        fig, axes = plt.subplots(num_models, 3, figsize=(15, 5 * num_models))

        for i, model_name in enumerate(model_names):
            # Plot classification accuracy
            axes[i, 0].bar(model_name, class_accuracy[i], color='skyblue')
            axes[i, 0].set_title(f'{model_name} Classification Accuracy')
            axes[i, 0].set_ylabel('Accuracy')
            axes[i, 0].set_xlabel('Model')

            # Plot total training time
            axes[i, 1].bar(model_name, train_time[i], color='salmon')
            axes[i, 1].set_title(f'{model_name} Total Training Time')
            axes[i, 1].set_ylabel('Time (seconds)')
            axes[i, 1].set_xlabel('Model')

            # Plot total test time
            axes[i, 2].bar(model_name, test_time[i], color='lightgreen')
            axes[i, 2].set_title(f'{model_name} Total Test Time')
            axes[i, 2].set_ylabel('Time (seconds)')
            axes[i, 2].set_xlabel('Model')

        plt.tight_layout()
        plt.show()

    @staticmethod
    def classify(x_train, y_train, x_test, y_test, train=True):
        models = ['Logistic Regression', 'Decision Tree', 'Random Forest', 'SVM',
                  'SVM Linear Kernel', 'SVM RPF Kernel', 'SVM Poly Kernel',
                  'K-Nearest Neighbors', 'Naive Bayes', 'XGBoost',
                  'AdaBoost', 'Gradient Boost', 'Light Gradient Boost']

        clf = [LogisticRegression(max_iter=1000, random_state=42, C=10, penalty='l2', solver='lbfgs'),
               DecisionTreeClassifier(criterion='gini', max_depth=5, min_samples_leaf=2, min_samples_split=2),
               RandomForestClassifier(bootstrap=False, max_depth=15, min_samples_leaf=1, min_samples_split=2,
                                      n_estimators=200),
               SVC(C=100, degree=2, gamma=0.1), SVC(kernel='linear', C=10, degree=2, gamma='scale'),
               SVC(kernel='rbf', C=100, degree=2, gamma=0.1), SVC(kernel='poly'),
               KNeighborsClassifier(), GaussianNB(), XGBClassifier(),
               AdaBoostClassifier(), GradientBoostingClassifier(), LGBMClassifier()]
        train_times, test_times, acc, mse = [], [], [], []
        for i in range(len(models)):
            if train:
                train_start_time = time.time()
                working_clf = clf[i]
                working_clf.fit(x_train, y_train)
                train_time = time.time() - train_start_time
                train_times.append(train_time)
                Classifications.__save_model(models[i], working_clf, train_time)
            else:
                working_clf, train_time = Classifications.__get_saved_model(models[i])
                train_times.append(train_time)

            test_start_time = time.time()
            y_pred = working_clf.predict(x_test)
            test_times.append(time.time() - test_start_time)
            mse.append(mean_squared_error(y_test, y_pred))
            acc.append(f'{accuracy_score(y_test, y_pred) * 100} %')

        return models, acc, mse, test_times, train_times
