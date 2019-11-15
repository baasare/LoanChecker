import os

import joblib
import pandas as pd
from django.shortcuts import render, redirect
from .forms import InputFeatureForm
from django.contrib.auth.decorators import login_required



# Create your views here.


def eligibility(request, selected_model,
                features={'Customer ID': 'LP001486', 'Gender': 'Male', 'Marital Status': 'Yes', 'Dependents': '1',
                          'Education': 'Not Graduate',
                          'Self Employment': 'No', 'Applicant Income': 4583, 'Co-applicant Income': 1508,
                          'Loan Amount': 128,
                          'Loan Amount Term': 360, 'Credit History': 1, 'Property Location': 'Rural'}):
    selected_model = selected_model + ".pkl"
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_file_path = os.path.join(BASE_DIR, 'loan\\model\\')
    file_path = models_file_path + selected_model
    current_model = joblib.load(file_path)

    new_customer = pd.DataFrame({
        'Gender': [features['Gender']],
        'Married': [features['Marital Status']],
        'Dependents': [features['Dependents']],
        'Education': [features['Education']],
        'Self_Employed': [features['Self Employment']],
        'ApplicantIncome': [features['Applicant Income']],
        'CoapplicantIncome': [features['Co-applicant Income']],
        'LoanAmount': [features['Loan Amount']],
        'Loan_Amount_Term': [features['Loan Amount Term']],
        'Credit_History': [features['Credit History']],
        'Property_Area': [features['Property Location']],
    })

    result = current_model.predict(new_customer)

    return result

@login_required(login_url='/signin')
def check_eligibility(request):
    features = {'Customer ID': '', 'Gender': '', 'Marital Status': '', 'Dependents': '',
                'Education': '', 'Self Employment': '', 'Applicant Income': 0, 'Co-applicant Income': 0,
                'Loan Amount': 0,
                'Loan Amount Term': 0, 'Credit History': 0, 'Property Location': ''}

    if request.method == 'POST':
        form = InputFeatureForm(request.POST)
        if form.is_valid():
            # form.save()
            features['Customer ID'] = form.cleaned_data.get('applicant_id')
            features['Gender'] = form.cleaned_data.get('gender')
            features['Marital Status'] = form.cleaned_data.get('marriage')
            features['Dependents'] = form.cleaned_data.get('dependents')
            features['Education'] = form.cleaned_data.get('education')
            features['Self Employment'] = form.cleaned_data.get('self_employed')
            features['Applicant Income'] = form.cleaned_data.get('income')
            features['Co-applicant Income'] = form.cleaned_data.get('co_income')
            features['Loan Amount'] = form.cleaned_data.get('loan_amount')
            features['Loan Amount Term'] = form.cleaned_data.get('loan_amount_term')
            features['Credit History'] = form.cleaned_data.get('credit_history')
            features['Property Location'] = form.cleaned_data.get('location')
            selected_model = form.cleaned_data.get('ml_model')

            client_eligibility = eligibility(features, selected_model)

            if client_eligibility[0] == 'Y':
                message = "Eligible"
                print(message)
            else:
                message = "Not Eligible"

            return display_results(request=request, output=message, features=features)
    else:
        form = InputFeatureForm()

    return render(request=request,
                  template_name="loan/check_eligibility.html",
                  context={'form': form})

@login_required(login_url='/signin')
def display_results(request, output=None, features=None):

    if features and output is None:
        redirect('check_eligibility', request)

    return render(request=request,
                  template_name="loan/eligibility.html",
                  context={'eligibility': output,
                           'features': features})
