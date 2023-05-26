# from django.shortcuts import render
# from django.http import JsonResponse
# from rest_framework.response import Response
# from rest_framework.decorators import api_view
# from rest_framework import status
# from glucApi.models import Glucose
# from glucApi.serializer import GlucoseSerializer
# # Create your views here.


from typing import Any
from rest_framework.views import APIView
from glucApi.models import Glucose
from glucApi.serializer import GlucoseSerializer
from rest_framework.response import Response
from rest_framework import status
from django.shortcuts import render
from django.http import JsonResponse

class PatientForm(APIView):
    def __init__(self):
        self.filename = 'index.html'

    def get(self, request):
        return render(request,template_name=self.filename)
    
class GetPatientInfo(APIView):
    def get(self, request):
        patientId = request.GET['patientId']
        patientPassword = request.GET['patientPassword']

        return JsonResponse({ 'success': True, 'youDidIt': 'yay', 'data': patientPassword})

class PatientList(APIView):
    def get(self, request):
        glucs = Glucose.objects.all() # complex data
        glucSerialized = GlucoseSerializer(glucs, many=True)
        
        return Response(glucSerialized.data)
    
    def post(self, request):
        return Response({ "hello": "friend"})
    

class PatientCreate(APIView):
    def post(self, request):
        #create gluocose values (maybe when we add new user)

        serializer = GlucoseSerializer(data=request.data)

        if serializer.is_valid():
            serializer.save() # save to db
            return Response(serializer.data)
        else:
            return Response(serializer.errors)


class PatientGlucose(APIView):
    def get_gluc_from_pk(self, patientId):
        try:
            return Glucose.objects.get(PatientId=patientId)
        except:
            return Response({
                "error": "Patient is not found"
            }, status=status.HTTP_404_NOT_FOUND)

    def get(self, request, pk):
        glucVals = self.get_gluc_from_pk(pk);
        serializer = GlucoseSerializer(glucVals)
        return Response(serializer.data)


    def post(self, request, pk):
        #create gluocose values (maybe when we add new user)
        
        glucVals = self.get_gluc_from_pk(pk)

        serializer = GlucoseSerializer(glucVals)

        if serializer.is_valid():
            serializer.save() # save to db
            return Response(serializer.data)
        else:
            return Response(serializer.errors)
        
    def put(self, request, pk):
        glucVals = self.get_gluc_from_pk(pk)
        serializer = GlucoseSerializer(glucVals, data=request.data)
        if serializer.is_valid:
            serializer.save()
            return Response(serializer.data)
        
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
    def delete(self, request, pk):
        glucVals = self.get_gluc_from_pk(pk)
        glucVals.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)


