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
    def plotting(model_names, class_accuracy, train_time, test_time):
        fig = plt.figure(figsize=(11, 11))
        rows, columns, bar_width = 4, 3, 0.15
        places, labels = [0.1, 0.6, 1], ['Accuracy', 'Train Time', 'Test Time']

        for i in range(1, rows * columns + 1):
            if i == (len(model_names) + 1):
                break
            ax = fig.add_subplot(rows, columns, i)
            ax.bar(places[0], class_accuracy[i - 1], color='r', width=bar_width,
                   edgecolor='grey', label=labels[0])
            ax.bar(places[1], train_time[i - 1], color='g', width=bar_width,
                   edgecolor='grey', label=labels[1])
            ax.bar(places[2], test_time[i - 1], color='b', width=bar_width,
                   edgecolor='grey', label=labels[2])
            plt.ylabel('Values', fontweight='bold')
            plt.xticks(places, labels)
            ax.set_title(model_names[i - 1])

        plt.subplots_adjust(left=0.1, bottom=0.07, right=0.9, top=0.925, wspace=0.4, hspace=1)
        plt.show()

    @staticmethod
    def classify(x_train, y_train, x_test, y_test, train=True):
        models = ['Logistic Regression', 'Decision Tree', 'Random Forest', 'SVM',
                  'SVM Linear Kernel', 'SVM RPF Kernel', 'K-Nearest Neighbors',
                  'Gaussian Naive Bayes', 'XGBoost', 'AdaBoost',
                  'Gradient Boost', 'Light Gradient Boost']

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
            acc.append(accuracy_score(y_test, y_pred) * 100)

        return models, acc, mse, test_times, train_times
