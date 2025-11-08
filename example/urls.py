# example/urls.py
from django.urls import path

from example.views import index, regex_to_dfa


urlpatterns = [
    path('', index),
    path('api/regex-to-dfa/', regex_to_dfa, name='regex_to_dfa'),
]