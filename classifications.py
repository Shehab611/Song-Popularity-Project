from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import time
from matplotlib import pyplot as plt
import pickle
import numpy as np


class Classifications:
    @staticmethod
    def __save_model(model_name, clf, train_time):
        with open(f'classification_models/{model_name}.pkl', 'wb') as file:
            pickle.dump(clf, file)
        with open(f'training_time/{model_name}.txt', 'w') as file:
            file.write(str(train_time))

    @staticmethod
    def __get_saved_model(model_name):
        with open(f'classification_models/{model_name}.pkl', 'rb') as file:
            model = pickle.load(file)
        with open(f'training_time/{model_name}.txt', 'r') as file:
            train_time = float(file.read())
        return model, train_time

    @staticmethod
    def plotting(class_accuracy, train_time, test_time):
        model_names = ['Logistic\nRegression', 'Decision\nTree', 'Random\nForest', 'SVM',
                       'SVM\nLinear\nKernel', 'SVM\nRPF\nKernel', 'K-Nearest\nNeighbors',
                       'Gaussian\nNaive\nBayes', 'XGBoost', 'AdaBoost',
                       'Gradient\nBoost', 'Light\nGradient\nBoost']

        fig = plt.figure(figsize=(12, 12))
        rows, columns, bar_width = 3, 1, 0.08
        bar_values = [class_accuracy, train_time, test_time]
        bar_titles = ['Accuracy', 'Train Time', 'Test Time']
        bar_colors = ['r', 'g', 'b']
        bar_y_titles = ['%', ' Seconds', 'Seconds']

        for i in range(1, rows * columns + 1):
            ax = fig.add_subplot(rows, columns, i)
            ax.bar(np.arange(len(bar_values[i - 1])), bar_values[i - 1],
                   width=bar_width, edgecolor='grey', color=bar_colors[i - 1])
            plt.ylabel(bar_y_titles[i - 1], fontweight='bold')
            ax.set_title(bar_titles[i - 1])
            plt.xticks([r for r in range(len(bar_values[i - 1]))], model_names)

        plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.95, wspace=0.5, hspace=0.8)
        plt.show()

    @staticmethod
    def classify(x_train, y_train, x_test, y_test, train=True):
        # Identify Models Names
        models = ['Logistic Regression', 'Decision Tree', 'Random Forest', 'SVM',
                  'SVM Linear Kernel', 'SVM RPF Kernel', 'K-Nearest Neighbors',
                  'Gaussian Naive Bayes', 'XGBoost', 'AdaBoost',
                  'Gradient Boost', 'Light Gradient Boost']

        # Identify Models Classifiers
        clf = [LogisticRegression(max_iter=1000, random_state=42, C=10, penalty='l2', solver='lbfgs'),
               DecisionTreeClassifier(criterion='gini', max_depth=5, min_samples_leaf=2, min_samples_split=2),
               RandomForestClassifier(bootstrap=False, max_depth=15, min_samples_leaf=1, min_samples_split=2,
                                      n_estimators=200),
               SVC(C=100, degree=2, gamma=0.1), SVC(kernel='linear', C=10, degree=2, gamma='scale'),
               SVC(kernel='rbf', C=100, degree=2, gamma=0.1),
               KNeighborsClassifier(algorithm='auto', n_neighbors=7, p=1, weights='distance'),
               GaussianNB(priors=None, var_smoothing=1e-08),
               XGBClassifier(gamma=0, colsample_bytree=0.6, learing_rate=0.1, max_depth=5, n_estimators=200,
                             subsample=0.6, use_label_encoder=True),
               AdaBoostClassifier(learning_rate=0.1, n_estimators=200),
               GradientBoostingClassifier(learning_rate=0.1, max_depth=5, n_estimators=50),
               LGBMClassifier(force_col_wise=True, learning_rate=0.01)]

        train_times, test_times, acc, mse = [], [], [], []

        # Loop on Models to Determine the Classifier to Work With
        for i in range(len(models)):

            # Check whether we are in test scenario or train one
            if train:

                # In Train scenario
                # Train the Model and Calculate its train time
                train_start_time = time.time()
                working_clf = clf[i]
                working_clf.fit(x_train, y_train)
                train_time = time.time() - train_start_time
                train_times.append(train_time)

                # Save the Model in pickle and its train time in text file to use them in testing
                Classifications.__save_model(models[i], working_clf, train_time)
            else:

                # In Test scenario
                # Load the Trained Model from pickle and its train time from text file
                working_clf, train_time = Classifications.__get_saved_model(models[i])
                train_times.append(train_time)

            # Calculate Test Time and get the accuracy of the trained model after predicting
            test_start_time = time.time()
            y_pred = working_clf.predict(x_test)
            test_times.append(time.time() - test_start_time)
            mse.append(mean_squared_error(y_test, y_pred))
            acc.append(accuracy_score(y_test, y_pred) * 100)

        return models, acc, mse, test_times, train_times
