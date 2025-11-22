# example/urls.py
from django.urls import path

from example.views import index, regex_to_dfa, regex_to_dfa_jff, regex_file_to_csv, transitions_to_dfa_endpoint, regex_to_alphabet, search_regex, acepnet_predict


urlpatterns = [
    path('', index),
    path('api/regex-to-dfa/', regex_to_dfa, name='regex_to_dfa'),
    path('api/regex-to-dfa/jff/', regex_to_dfa_jff, name='regex_to_dfa_jff'),
    path('api/regex-file-to-csv/', regex_file_to_csv, name='regex_file_to_csv'),
    path('api/transitions-to-dfa/', transitions_to_dfa_endpoint, name='transitions_to_dfa'),
    path('api/regex-to-alphabet/', regex_to_alphabet, name='regex_to_alphabet'),
    path('api/search-regex/', search_regex, name='search_regex'),
    path('api/acepnet-predict/', acepnet_predict, name='acepnet_predict'),
]