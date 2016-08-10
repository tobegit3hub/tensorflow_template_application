from django.conf.urls import url
from . import views

urlpatterns = [
    url(r'^predict/',
        views.predict,
        name='predict'),
    url(r'^online_train/',
        views.online_train,
        name='oneline_train'),
    url(r'^$',
        views.index,
        name='index'),
]
