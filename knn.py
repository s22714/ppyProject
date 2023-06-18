import array

import pandas as pd
import dbinit
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib


class Model:
    def __init__(self, rnd_state=2023, test_size=0.25, name = "", classcolumn = 0, norm = False, n_neigh = 5):
        self.y = None
        self.X = None
        self.n_neigh = n_neigh
        self.y_test = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.knn = None
        self.rnd_state = rnd_state
        self.test_size = test_size
        self.name = name
        self.norm = norm
        self.classcolumn = classcolumn

    def partition_data(self):

        df = dbinit.get_data_from_db(self.name)

        if self.classcolumn != 0:
            self.X = df.iloc[:, :-1].values
            self.y = df.iloc[:, -1].values
        else:
            self.X = df.iloc[:, 1:].values
            self.y = df.iloc[:, 0].values

        if self.norm:
            self.X = preprocessing.normalize(self.X)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=self.test_size, random_state=self.rnd_state, shuffle=True)

    def new_model_train(self):

        self.knn = KNeighborsClassifier(n_neighbors=self.n_neigh)
        self.knn.fit(self.X_train, self.y_train)

    def model_from_saved(self, filename):
        self.knn = joblib.load("saved/" + filename)

    def save_model_to_file(self, filename):
        joblib.dump(self.knn,"saved/" + filename + ".sav")

    def score(self):
        kfold = KFold(n_splits=self.n_neigh, random_state=self.rnd_state, shuffle=True)

        scores = cross_val_score(self.knn, self.X_train, self.y_train, cv=kfold, scoring="accuracy")
        return scores

    def cross_val(self,min_neighbours = 1, max_neighbours = 21):

        kfold = KFold(n_splits=self.n_neigh, random_state=self.rnd_state, shuffle=True)

        param_grid = {
            'n_neighbors': list(range(min_neighbours, max_neighbours)),
            'metric': ['euclidean', 'manhattan']
        }

        grid_search = GridSearchCV(self.knn, param_grid, cv=kfold)

        grid_search.fit(self.X_train, self.y_train)

        results = pd.DataFrame(grid_search.cv_results_)
        print(results["mean_test_score"])

        print("Najlepsze parametry: ", grid_search.best_params_)
        print("Najlepszy wynik: ", grid_search.best_score_)

        self.knn = grid_search.best_estimator_

        best_predict_train = self.knn.predict(self.X_train)
        print("Dokładność na zbiorze treningowym: ", accuracy_score(self.y_train, best_predict_train))
        best_predict = self.knn.predict(self.X_test)
        print("Dokładność na zbiorze testowym: ", accuracy_score(self.y_test, best_predict))

        cm_train = confusion_matrix(self.y_train, best_predict_train)
        print("Macierz pomylek dla zbioru treningowego:")
        print(cm_train)

        cm_test = confusion_matrix(self.y_test, best_predict)
        print("Macierz pomylek dla zbioru testowego:")
        print(cm_test)

    def pred(self,vector):
        return self.knn.predict(vector)[0]