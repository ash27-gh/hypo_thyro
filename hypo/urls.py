from django.urls import path
import hypo.views

urlpatterns = [
    path('', hypo.views.index,name='index'),
    path('predict', hypo.views.predict,name='predict'),
]