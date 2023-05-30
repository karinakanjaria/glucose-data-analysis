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
from django.core.files.storage import default_storage
import boto3
#from boto.s3.connection import S3Connection
from glucApp import settings 
import pandas as pd
import pyarrow.parquet as pq
import io

class PatientForm(APIView):
    def __init__(self):
        self.filename = 'index.html'

    def get(self, request):

        # conn = S3Connection(settings.AWS_ACCESS_KEY_ID,settings.AWS_SECRET_ACCESS_KEY)
        # bucket = conn.get_bucket(settings.MEDIA_BUCKET)
       
        session = boto3.Session(aws_access_key_id = settings.ACCESS_ID,
                         aws_secret_access_key = settings.SECRET_ACCESS_KEY)

        s3 = session.resource('s3', endpoint_url = settings.ENDPOINT_URL)

      

        my_bucket = s3.Bucket(settings.STORAGE_BUCKET_NAME)

        # buffer = io.BytesIO()
        # data_here = s3.Object(settings.STORAGE_BUCKET_NAME, 'train_data/parquet_3.parquet')
        # data_here.download_fileobj(buffer)
        
        # df = pd.read_parquet('s3://glucose/train_data/parquet_3.parquet')

        # df = pq.ParquetDataset('s3://glucose/train_data/parquet_3.parquet', filesystem=session)\
        #     .read_pandas().to_pandas()

        # print(df.head(10))
        

        # for my_bucket_object in my_bucket.objects.all():
        #     print('OH MY GOD IT DID IT')
        #     print(my_bucket_object.key)
          
        # data_here = object
        # for test in my_bucket.objects.filter(Prefix="train_data/parquet_3.parquet"):
        #     print('maybe this is it???')
        #     data_here = test.get()['Body'].read()
        #     #data_here = test.get()['Body']
        #     # with gzip.open(data_here, 'rt') as gf:
        #     print('did it get here')
        #     does_it_work = pd.read_parquet(data_here, encoding='utf-16')
        #     print('did it work???')
        #     print(does_it_work)

        s3_client = boto3.client('s3', aws_access_key_id = settings.ACCESS_ID,
                        aws_secret_access_key = settings.SECRET_ACCESS_KEY,
                        endpoint_url = settings.ENDPOINT_URL)
        
        obj = s3_client.get_object(Bucket=settings.STORAGE_BUCKET_NAME,\
                                    Key='train_data/parquet_3.parquet')
        
        test = pd.read_parquet(io.BytesIO(obj['Body'].read()))

        print('maybe?')
        print(test.head())        

        print('look here for buckets')
        response = s3.list_objects()
        print(response)
        print('end of buckets')

        for bucket in s3.list_buckets()['Buckets']:
            print('another one -DJ Karina')
            print(bucket['Name'])

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
        #glucVals.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)


