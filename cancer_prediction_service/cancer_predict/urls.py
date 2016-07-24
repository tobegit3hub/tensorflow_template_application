from django.conf.urls import url
from . import views

urlpatterns = [
    url(r'^predict/', views.index, name='predict'),
    url(r'^$', views.index, name='index'),
]
