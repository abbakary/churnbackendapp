from django.urls import path
from . import views

urlpatterns = [
    path('predict/', views.predict, name='predict'),
    path('predict/batch/', views.batch_predict, name='batch_predict'),
    path('history/', views.history, name='history'),
    path('stats/', views.stats, name='stats'),
    path('history/<int:pk>/', views.delete_prediction, name='delete_prediction'),
    
    # Customer Tracking
    path('customers/', views.customer_list, name='customer_list'),
    path('customers/<str:customer_id>/', views.customer_detail, name='customer_detail'),
]
