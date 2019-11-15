from django.urls import path
from . import views


urlpatterns = [
    path('eligibility', views.display_results, name='eligibility'),
    path('check_eligibility', views.check_eligibility, name='check_eligibility'),
]
