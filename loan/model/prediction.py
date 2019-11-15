import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix


class Models:
    """Class containing all model"""

    def __init__(self):
        self.dataset = pd.read_csv('loan_data_set.csv')
        var_mod = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Status']
        le = LabelEncoder()
        for i in var_mod:
            self.dataset[i] = le.fit_transform(self.dataset[i].astype(str))

        self.X = pd.DataFrame(self.dataset.iloc[:, 1:-1])
        self.y = pd.DataFrame(self.dataset.iloc[:, -1]).values.ravel()

        # Pre Processing
        imputer = SimpleImputer(strategy="mean")
        imputer = imputer.fit(self.X)
        self.X = imputer.transform(self.X)

        self.x_train, self.x_test, self.y_train, self.y_test = model_selection.train_test_split(self.X, self.y,
                                                                                                test_size=0.2,
                                                                                                random_state=7)

        self.logistic_reg_model = LogisticRegression(solver='liblinear')
        self.logistic_reg_model.fit(self.x_train, self.y_train)

        self.decision_tree_model = DecisionTreeClassifier()
        self.decision_tree_model.fit(self.x_train, self.y_train)

        self.random_forest_model = RandomForestClassifier(n_estimators=100)
        self.random_forest_model.fit(self.x_train, self.y_train)

        self.knn_model = KNeighborsClassifier(n_neighbors=9)
        self.knn_model.fit(self.x_train, self.y_train)

        self.smv_model = SVC(gamma='scale', kernel='rbf')
        self.smv_model.fit(self.x_train, self.y_train)

    def clean_input(self, features):
        dataset_test = self.dataset

        features = ['LP001486', 'Male', 'Yes', 1, 'Not Graduate', 'No', 4583, 1508, 128, 360, 1, 'Rural', 'N']

        # create a new DataFrame from user input
        new_customer = pd.DataFrame({
            'Loan_ID': [features[0]],
            'Gender': [features[1]],
            'Married': [features[2]],
            'Dependents': [features[3]],
            'Education': [features[4]],
            'Self_Employed': [features[5]],
            'ApplicantIncome': [features[6]],
            'CoapplicantIncome': [features[7]],
            'LoanAmount': [features[8]],
            'Loan_Amount_Term': [features[9]],
            'Credit_History': [features[10]],
            'Property_Area': [features[11]],
            'Loan_Status': [features[12]],
        })

        # append new user input to original dataset
        dataset_test = dataset_test.append(new_customer)

        # encode the new dataset
        var_mod = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Status']
        le = LabelEncoder()
        for i in var_mod:
            dataset_test[i] = le.fit_transform(dataset_test[i].astype(str))

        # obtain the encoded user data from the encoded dataset
        user = dataset_test[-1:]

        # select the features necessary
        user = pd.DataFrame(user.iloc[:, 1:-1])

        return user.values

    def train_logistic_reg(self, feature):
        eligibility = self.logistic_reg_model.predict(self.clean_input(self, feature))
        return eligibility[0]

    def train_decision_tree(self, feature):
        eligibility = self.decision_tree_model.predict(self.clean_input(self, feature))
        return eligibility[0]

    def train_random_forest(self, feature):
        eligibility = self.random_forest_model.predict(self.clean_input(self, feature))
        return eligibility[0]

    def train_knn(self, feature):
        eligibility = self.knn_model.predict(self.clean_input(self, feature))
        return eligibility[0]

    def train_svm(self, feature):
        eligibility = self.smv_model.predict(self.clean_input(self, feature))
        return eligibility[0]
