from django import forms


class InputFeatureForm(forms.Form):
    applicant_id = forms.CharField(max_length=8)
    gender = forms.CharField(max_length=100)
    marriage = forms.CharField(max_length=100)
    dependents = forms.CharField(max_length=100)
    education = forms.CharField(max_length=100)
    self_employed = forms.CharField(max_length=100)
    income = forms.FloatField()
    co_income = forms.FloatField()
    loan_amount = forms.FloatField()
    loan_amount_term = forms.FloatField()
    credit_history = forms.FloatField()
    location = forms.CharField(max_length=100)
    ml_model = forms.CharField(max_length=100)
