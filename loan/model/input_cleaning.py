import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler


def clean_input(features=['LP001486', 'Male', 'Yes', '1', 'Not Graduate', 'No', 4583, 1508, 128, 360, 1, 'Rural', 'N']):
    dataset_test = pd.read_csv('loan_data_set.csv')

    numeric_features = dataset_test.select_dtypes(include=['int64', 'float64']).columns
    numeric_features_steps = [('imputer', SimpleImputer(strategy='median')), ('scaler', MinMaxScaler())]
    numeric_transformer = Pipeline(steps=numeric_features_steps)


    categorical_features = dataset_test.select_dtypes(include=['object']).drop(['Loan_Status'], axis=1).columns
    categorical_features_steps = [('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                                  ('onehot', OneHotEncoder())]
    categorical_transformer = Pipeline(steps=categorical_features_steps)

    preprocessor = ColumnTransformer(
        remainder='passthrough',
        transformers=[
            ('numeric', numeric_transformer, numeric_features),
            ('categorical', categorical_transformer, categorical_features)
        ])


    new_customer = {
        'Loan_ID': features[0],
        'Gender': features[1],
        'Married': features[2],
        'Dependents': features[3],
        'Education': features[4],
        'Self_Employed': features[5],
        'ApplicantIncome': features[6],
        'CoapplicantIncome': features[7],
        'LoanAmount': features[8],
        'Loan_Amount_Term': features[9],
        'Credit_History': features[10],
        'Property_Area': features[11],
        'Loan_Status': features[12]
    }

    dataset_test = dataset_test.append(new_customer, ignore_index=True)

    dataset_test = dataset_test.drop('Loan_ID', axis=1)
    dataset_test = dataset_test.drop('Loan_Status', axis=1)

    # encode the new dataset
    encoded_data_set = preprocessor.fit_transform(dataset_test)
    # obtain the encoded user data from the encoded dataset
    customer = encoded_data_set[-1:]

    return customer
