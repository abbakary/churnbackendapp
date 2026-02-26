from django.contrib import admin, messages
from django.urls import path
from django.shortcuts import render, redirect
from django.http import HttpResponseRedirect
import pandas as pd
import io

from .models import Customer, ChurnPrediction
from .forms import CsvImportForm

@admin.register(Customer)
class CustomerAdmin(admin.ModelAdmin):
    list_display = ('customer_id', 'age', 'contract_type', 'is_churn_display', 'risk_level_display', 'imported_at')
    search_fields = ('customer_id',)
    list_filter = ('contract_type',)
    
    def get_urls(self):
        urls = super().get_urls()
        my_urls = [
            path('import-csv/', self.import_csv),
        ]
        return my_urls + urls

    def import_csv(self, request):
        if request.method == "POST":
            csv_file = request.FILES["csv_file"]
            if not csv_file.name.endswith('.csv'):
                messages.error(request, 'This is not a CSV file')
                return HttpResponseRedirect(request.path_info)
            
            try:
                # Read CSV
                df = pd.read_csv(io.StringIO(csv_file.read().decode('utf-8')))
                
                # Expected columns mapping (CSV names -> Model names)
                # Map standard CSV column names to model fields
                col_map = {
                    'CustomerID': 'customer_id',
                    'Age': 'age',
                    'Subscription_Duration_Months': 'subscription_duration_months',
                    'Contract_Type': 'contract_type',
                    'Monthly_Logins': 'monthly_logins',
                    'Last_Purchase_Days_Ago': 'last_purchase_days_ago',
                    'App_Usage_Time_Min': 'app_usage_time_min',
                    'Monthly_Spend': 'monthly_spend',
                    'Discount_Usage_Percentage': 'discount_usage_percentage',
                    'Customer_Support_Calls': 'customer_support_calls',
                    'Satisfaction_Score': 'satisfaction_score',
                }
                
                success_count = 0
                for _, row in df.iterrows():
                    # Check if customer already exists, if so update, else create
                    cid = str(row.get('CustomerID', ''))
                    if not cid: continue
                    
                    customer, created = Customer.objects.update_or_create(
                        customer_id=cid,
                        defaults={
                            'age': row.get('Age', 0),
                            'subscription_duration_months': row.get('Subscription_Duration_Months', 0),
                            'contract_type': row.get('Contract_Type', 'Monthly'),
                            'monthly_logins': row.get('Monthly_Logins', 0),
                            'last_purchase_days_ago': row.get('Last_Purchase_Days_Ago', 0),
                            'app_usage_time_min': row.get('App_Usage_Time_Min', 0.0),
                            'monthly_spend': row.get('Monthly_Spend', 0.0),
                            'discount_usage_percentage': row.get('Discount_Usage_Percentage', 0.0),
                            'customer_support_calls': row.get('Customer_Support_Calls', 0),
                            'satisfaction_score': row.get('Satisfaction_Score', 1),
                        }
                    )
                    # Note: Signal in models.py will automatically trigger prediction
                    success_count += 1
                
                self.message_user(request, f"Successfully imported {success_count} customers.", messages.SUCCESS)
                return redirect("..")
            except Exception as e:
                self.message_user(request, f"Error: {str(e)}", messages.ERROR)
                return HttpResponseRedirect(request.path_info)

        form = CsvImportForm()
        payload = {"form": form}
        return render(
            request, "admin/csv_import.html", payload
        )

    @admin.display(description='Is Churn', boolean=True)
    def is_churn_display(self, obj):
        latest = obj.predictions.order_by('-created_at').first()
        return latest.is_churn if latest else None

    @admin.display(description='Risk Level')
    def risk_level_display(self, obj):
        latest = obj.predictions.order_by('-created_at').first()
        return latest.risk_level if latest else "N/A"


@admin.register(ChurnPrediction)
class ChurnPredictionAdmin(admin.ModelAdmin):
    list_display = ('customer_ref_id', 'is_churn', 'churn_probability', 'risk_level', 'created_at')
    list_filter = ('is_churn', 'risk_level', 'created_at')
    search_fields = ('customer_ref_id',)
    readonly_fields = ('created_at',)
