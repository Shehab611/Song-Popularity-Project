from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from xgboost import XGBClassifier
import time
from matplotlib import pyplot as plt
import pickle


class Classifications:
    __classifications_types = ['Logistic Regression', 'Decision Tree', 'Random Forest',
                               'SVM', 'SVM Linear Kernel', 'SVM RPF Kernel',
                               'SVM Poly Kernel', 'K-Nearest Neighbors', 'Naive Bayes',
                               'XGBoost']

    @staticmethod
    def apply_classifications(x_train, y_train, x_test, y_test):
        methods = [
            Classifications.__LR, Classifications.__DT, Classifications.__RF, Classifications.__SVM,
            Classifications.__SVM_Lin, Classifications.__SVM_RPF, Classifications.__SVM_Poly,
            Classifications.__KNeighbors, Classifications.__Gaussian, Classifications.__XGB
        ]
        returned_data = dict()
        for i in range(len(Classifications.__classifications_types)):
            returned_data[Classifications.__classifications_types[i]] = methods[i](x_train, y_train, x_test, y_test)

        return returned_data

    @staticmethod
    def __saveModel(model_name, clf):
        with open(f'classification_models/{model_name}.pkl', 'wb') as file:
            pickle.dump(clf, file)

    @staticmethod
    def __LR(x_train, y_train, x_test, y_test):
        train_start_time = time.time()
        clf = LogisticRegression()
        clf.fit(x_train, y_train)

        # Classifications.__saveModel(Classifications.__classifications_types[0], clf)

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
    def __DT(x_train, y_train, x_test, y_test):
        train_start_time = time.time()
        clf = DecisionTreeClassifier()
        clf.fit(x_train, y_train)

        # Classifications.__saveModel(Classifications.__classifications_types[1], clf)

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
    def __RF(x_train, y_train, x_test, y_test):
        train_start_time = time.time()
        clf = RandomForestClassifier()
        clf.fit(x_train, y_train)

        # Classifications.__saveModel(Classifications.__classifications_types[2], clf)

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
    def __SVM(x_train, y_train, x_test, y_test):
        train_start_time = time.time()
        clf = SVC()
        clf.fit(x_train, y_train)

        # Classifications.__saveModel(Classifications.__classifications_types[3], clf)

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
    def __SVM_Lin(x_train, y_train, x_test, y_test):
        train_start_time = time.time()
        clf = SVC(kernel='linear')
        clf.fit(x_train, y_train)

        # Classifications.__saveModel(Classifications.__classifications_types[4], clf)

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
    def __SVM_RPF(x_train, y_train, x_test, y_test):
        train_start_time = time.time()
        clf = SVC(kernel='rbf')
        grid_search = GridSearchCV(clf, {'C': [0.1, 1, 10], 'gamma': [0.01, 0.1, 1]}, cv=5)
        grid_search.fit(x_train, y_train)
        clf = grid_search.best_estimator_
        clf.fit(x_train, y_train)

        # Classifications.__saveModel(Classifications.__classifications_types[5], clf)

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
    def __SVM_Poly(x_train, y_train, x_test, y_test):
        train_start_time = time.time()
        clf = SVC(kernel='poly')
        grid_search = GridSearchCV(clf, {'C': [0.1, 1, 10], 'degree': [2, 3, 4]}, cv=5)
        grid_search.fit(x_train, y_train)
        clf = grid_search.best_estimator_
        clf.fit(x_train, y_train)

        # Classifications.__saveModel(Classifications.__classifications_types[6], clf)

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
    def __KNeighbors(x_train, y_train, x_test, y_test):
        train_start_time = time.time()
        clf = KNeighborsClassifier()
        grid_search = GridSearchCV(clf, {'n_neighbors': [3, 5, 7]}, cv=5)
        grid_search.fit(x_train, y_train)
        clf = grid_search.best_estimator_
        clf.fit(x_train, y_train)

        # Classifications.__saveModel(Classifications.__classifications_types[7], clf)

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
    def __Gaussian(x_train, y_train, x_test, y_test):
        train_start_time = time.time()
        clf = GaussianNB()
        clf.fit(x_train, y_train)

        # Classifications.__saveModel(Classifications.__classifications_types[8], clf)

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
    def __XGB(x_train, y_train, x_test, y_test):
        train_start_time = time.time()
        clf = XGBClassifier(use_label_encoder=False)
        grid_search = GridSearchCV(clf, {'learning_rate': [0.1, 0.01], 'max_depth': [3, 5, 7]}, cv=5)
        grid_search.fit(x_train, y_train)
        clf = grid_search.best_estimator_
        clf.fit(x_train, y_train)

        # Classifications.__saveModel(Classifications.__classifications_types[9], clf)

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
    def xxx(train, x_train, y_train, x_test, y_test):
        models = ['Logistic Regression', 'Decision Tree', 'Random Forest',
                  'SVM', 'SVM Linear Kernel', 'SVM RPF Kernel',
                  'SVM Poly Kernel', 'K-Nearest Neighbors', 'Naive Bayes',
                  'XGBoost']

        clf = [LogisticRegression(), DecisionTreeClassifier(), RandomForestClassifier(),
               SVC(), SVC(kernel='linear'), SVC(kernel='rbf'), SVC(kernel='poly'), KNeighborsClassifier(),
               GaussianNB(), XGBClassifier(use_label_encoder=False)]

        for i in range(len(models)):
            if train:
                train_start_time = time.time()
                clf[i].fit(x_train, y_train)
                total_train_time = time.time() - train_start_time
                Classifications.__saveModel(models[i], clf[i])
            else:
                clf[i].predict(x_test)
