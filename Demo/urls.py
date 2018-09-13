from django.conf.urls import include, url
from django.contrib import admin
from webapp import views as tv
from MachineLearning import views as ML
from MachineLearning.views import Predict

urlpatterns = [
    # Examples:
    # url(r'^$', 'Demo.views.home', name='home'),
    # url(r'^blog/', include('blog.urls')),


    url(r'^$', tv.upload),
    url(r'^normalmap/', tv.do_noramlmap),
    url(r'^training/', ML.train),
    url(r'^prediction/',ML.prediction),
]
