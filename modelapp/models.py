from django.db import models


class Customer(models.Model):
    """
    Represents a real customer record imported from the dataset.
    These records are the source of truth for prediction inputs.
    """
    CONTRACT_TYPE_CHOICES = [
        ('Monthly', 'Monthly'),
        ('Annual', 'Annual'),
    ]

    customer_id = models.CharField(max_length=50, unique=True)
    age = models.IntegerField()
    subscription_duration_months = models.IntegerField()
    contract_type = models.CharField(max_length=20, choices=CONTRACT_TYPE_CHOICES)
    monthly_logins = models.IntegerField()
    last_purchase_days_ago = models.IntegerField()
    app_usage_time_min = models.FloatField()
    monthly_spend = models.FloatField()
    discount_usage_percentage = models.FloatField()
    customer_support_calls = models.IntegerField()
    satisfaction_score = models.IntegerField()

    # Ground truth label from the dataset (if available)
    actual_churn = models.BooleanField(null=True, blank=True)

    imported_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['customer_id']

    def __str__(self):
        return f"{self.customer_id} | Age {self.age} | {self.contract_type}"


class ChurnPrediction(models.Model):
    CONTRACT_TYPE_CHOICES = [
        ('Monthly', 'Monthly'),
        ('Annual', 'Annual'),
    ]

    # Link to source customer record (optional)
    customer = models.ForeignKey(
        Customer, null=True, blank=True, on_delete=models.SET_NULL,
        related_name='predictions'
    )

    # Input features (stored for audit, even if linked to Customer)
    age = models.IntegerField()
    subscription_duration_months = models.IntegerField()
    contract_type = models.CharField(max_length=20, choices=CONTRACT_TYPE_CHOICES)
    monthly_logins = models.IntegerField()
    last_purchase_days_ago = models.IntegerField()
    app_usage_time_min = models.FloatField()
    monthly_spend = models.FloatField()
    discount_usage_percentage = models.FloatField()
    customer_support_calls = models.IntegerField()
    satisfaction_score = models.IntegerField()

    # Prediction output
    is_churn = models.BooleanField(db_index=True)
    churn_probability = models.FloatField()
    risk_level = models.CharField(max_length=20, db_index=True)  # Low / Medium / High / Critical

    # Metadata
    customer_ref_id = models.CharField(max_length=50, blank=True, default='', db_index=True)  # renamed to avoid clash
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        status = "CHURN" if self.is_churn else "RETAIN"
        return f"{self.customer_ref_id or 'Customer'} | {status} | {self.churn_probability:.1%}"


from django.db.models.signals import post_save
from django.dispatch import receiver
from .utils import predict_for_customer

@receiver(post_save, sender=Customer)
def auto_predict_churn(sender, instance, created, **kwargs):
    """
    Automatically triggers a churn prediction when a Customer record is created or updated.
    """
    # Prepare data for prediction
    data = {
        'Age': instance.age,
        'Subscription_Duration_Months': instance.subscription_duration_months,
        'Contract_Type': instance.contract_type,
        'Monthly_Logins': instance.monthly_logins,
        'Last_Purchase_Days_Ago': instance.last_purchase_days_ago,
        'App_Usage_Time_Min': instance.app_usage_time_min,
        'Monthly_Spend': instance.monthly_spend,
        'Discount_Usage_Percentage': instance.discount_usage_percentage,
        'Customer_Support_Calls': instance.customer_support_calls,
        'Satisfaction_Score': instance.satisfaction_score,
    }
    
    try:
        result = predict_for_customer(data)
        
        # Create or update prediction record
        # We search for the latest prediction for this customer to determine if we update or create new
        # For "real-time feeding", we usually want to record the state at that moment
        ChurnPrediction.objects.create(
            customer=instance,
            customer_ref_id=instance.customer_id,
            age=instance.age,
            subscription_duration_months=instance.subscription_duration_months,
            contract_type=instance.contract_type,
            monthly_logins=instance.monthly_logins,
            last_purchase_days_ago=instance.last_purchase_days_ago,
            app_usage_time_min=instance.app_usage_time_min,
            monthly_spend=instance.monthly_spend,
            discount_usage_percentage=instance.discount_usage_percentage,
            customer_support_calls=instance.customer_support_calls,
            satisfaction_score=instance.satisfaction_score,
            is_churn=result['is_churn'],
            churn_probability=result['churn_probability'],
            risk_level=result['risk_level']
        )
    except Exception as e:
        print(f"Error in auto_predict_churn: {e}")
