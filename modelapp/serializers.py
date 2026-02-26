from rest_framework import serializers
from .models import ChurnPrediction


class PredictionInputSerializer(serializers.Serializer):
    customer_id = serializers.CharField(required=False, default='', allow_blank=True)
    Age = serializers.IntegerField(min_value=18, max_value=100)
    Subscription_Duration_Months = serializers.IntegerField(min_value=0, max_value=120)
    Contract_Type = serializers.ChoiceField(choices=['Monthly', 'Annual'])
    Monthly_Logins = serializers.IntegerField(min_value=0, max_value=60)
    Last_Purchase_Days_Ago = serializers.IntegerField(min_value=0, max_value=365)
    App_Usage_Time_Min = serializers.FloatField(min_value=0.0, max_value=300.0)
    Monthly_Spend = serializers.FloatField(min_value=0.0, max_value=1000.0)
    Discount_Usage_Percentage = serializers.FloatField(min_value=0.0, max_value=1.0)
    Customer_Support_Calls = serializers.IntegerField(min_value=0, max_value=20)
    Satisfaction_Score = serializers.IntegerField(min_value=1, max_value=5)


class ChurnPredictionSerializer(serializers.ModelSerializer):
    class Meta:
        model = ChurnPrediction
        fields = '__all__'
