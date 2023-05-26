from django.contrib import admin
from django.urls import path, include
from glucApi.views import PatientList, PatientCreate, \
    PatientGlucose, PatientForm, GetPatientInfo


urlpatterns = [
    path("", PatientCreate.as_view()),
    path('list/', PatientList.as_view()),
    path('<int:patientId>',  PatientGlucose.as_view()),
    path("PatientForm", PatientForm.as_view()),
    path('GetPatientInfo', GetPatientInfo.as_view(), name='get_patient_info')
]
