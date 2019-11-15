import pandas as pd
import joblib
from sklearn import model_selection
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, NuSVC
from sklearn.tree import DecisionTreeClassifier


import os



def prediction_models():
    dataset = pd.read_csv('loan_data_set.csv')
    dataset = dataset.drop('Loan_ID', axis=1)

    X = dataset.drop('Loan_Status', axis=1)
    y = dataset['Loan_Status']

    x_train, x_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2,
                                                                        random_state=7)

    numeric_features = dataset.select_dtypes(include=['int64', 'float64']).columns
    numeric_features_steps = [('imputer', SimpleImputer(strategy='median')), ('scaler', MinMaxScaler())]
    numeric_transformer = Pipeline(steps=numeric_features_steps)

    categorical_features = dataset.select_dtypes(include=['object']).drop(['Loan_Status'], axis=1).columns
    categorical_features_steps = [('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                                  ('onehot', OneHotEncoder())]
    categorical_transformer = Pipeline(steps=categorical_features_steps)

    preprocessor = ColumnTransformer(
        remainder='passthrough',
        transformers=[
            ('numeric', numeric_transformer, numeric_features),
            ('categorical', categorical_transformer, categorical_features)
        ])

    classifiers = {
        'knn': KNeighborsClassifier(9),
        'logistic': LogisticRegression(solver='liblinear'),
        'smv': SVC(gamma='auto', kernel='rbf'),
        'smv_1': SVC(gamma='auto', kernel="rbf", C=0.025,
                     probability=True),
        'nu_smv': NuSVC(gamma='auto', probability=True),
        'decision_trees': DecisionTreeClassifier(),
        'random_forest': RandomForestClassifier(n_estimators=100),
        'ada_boost': AdaBoostClassifier(),
        'gradient_boost': GradientBoostingClassifier()
    }

    # knn, logistic, smv, smv_1, nu_svm, decision_trees, random_forest, ada_boost, gradient_boost
    pred_models = []

    for name, classifier in classifiers.items():
        pipe = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', classifier)])
        pipe.fit(x_train, y_train)
        joblib.dump(pipe, "%s.pkl" % name)

        pred_models.append(pipe)

    return pred_models


def main():
    # models = prediction_models()

    # features = ['LP001486', 'Male', 'Yes', '1', 'Not Graduate', 'No', 4583, 1508, 128, 360, 1, 'Rural', 'N']
    #
    # new_customer = pd.DataFrame({
    #     'Gender': [features[1]],
    #     'Married': [features[2]],
    #     'Dependents': [features[3]],
    #     'Education': [features[4]],
    #     'Self_Employed': [features[5]],
    #     'ApplicantIncome': [features[6]],
    #     'CoapplicantIncome': [features[7]],
    #     'LoanAmount': [features[8]],
    #     'Loan_Amount_Term': [features[9]],
    #     'Credit_History': [features[10]],
    #     'Property_Area': [features[11]],
    # })
    #
    # y_pred_single = models[1].predict(new_customer)
    # print(y_pred_single)


    # current_model = joblib.load('filename.pkl')
    # current_model.predict()

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_file_path = os.path.join(BASE_DIR, 'model\\')
    print(models_file_path)

if __name__ == '__main__':
    main()


