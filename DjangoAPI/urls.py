from django.contrib import admin
from django.urls import path,include
from . import views
from rest_framework import routers
from django.conf import settings
from django.urls import path, include
from django.conf.urls.static import static

router = routers.DefaultRouter()
urlpatterns = [
                path('api/', include(router.urls)),
                path('', views.FormView, name='form'),
              ]
urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT) #new