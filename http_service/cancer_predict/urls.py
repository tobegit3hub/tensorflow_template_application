from django.conf.urls import url
from . import views

urlpatterns = [
    url(r'^predict/', views.predict,
        name='predict'),
    url(r'^$', views.index, name='index'),
]
