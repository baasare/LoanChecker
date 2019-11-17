import os

import joblib
import pandas as pd
from django.contrib import messages
from django.shortcuts import render, redirect
from .forms import InputFeatureForm
from django.contrib.auth.decorators import login_required


# Create your views here.


def eligibility(selected_model, features):
    selected_model = selected_model + ".pkl"
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_path_1 = os.path.join(BASE_DIR, 'loan')
    file_path = os.path.join(file_path_1, 'model', selected_model)
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

            for key, value in features.items():
                print(key, '->', value)
            print("Selected Model: " + selected_model)

            client_eligibility = eligibility(features=features, selected_model=selected_model)

            if client_eligibility[0] == 'Y':
                message = "Eligible"
                print(message)
            else:
                message = "Not Eligible"

            return display_results(request=request, output=message, features=features, selected_model=selected_model)
        else:
            for field in form:
                for error in field.errors:
                    messages.error(request, error)
                    print("Field: ")
                    print(field)
                    print("Error:")
                    print(error)
            messages.error(request, "Form wasn't processed.")
    else:
        form = InputFeatureForm()

    return render(request=request,
                  template_name="loan/check_eligibility.html",
                  context={'form': form})


@login_required(login_url='/signin')
def display_results(request, output=None, features=None, selected_model=None):
    if features and output is None:
        redirect('check_eligibility', request)

    return render(request=request,
                  template_name="loan/eligibility.html",
                  context={'eligibility': output,
                           'features': features,
                           'selected_model': selected_model})
