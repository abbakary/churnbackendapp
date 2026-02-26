from django.contrib import admin
from django.urls import path, include
from modelapp.views import home_page

urlpatterns = [
    path('', home_page, name='home'),
    path('admin/', admin.site.urls),
    path('stats/', include('modelapp.urls')),
]
