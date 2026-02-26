"""
API views for customer churn prediction.
Handles single predictions, batch processing, history, and customer tracking.
"""

import os
from datetime import datetime

from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status

from django.db.models import Avg, Count, Q, Max, Subquery, OuterRef
from django.db.models.functions import TruncDate

from .models import ChurnPrediction
from .serializers import PredictionInputSerializer
from .utils import (
    predict_for_customer, 
    get_risk_factors, 
    get_recommendations, 
    compute_risk_level,
    get_model,
    initialize
)

# Try to initialize model on startup
try:
    initialize()
except Exception as e:
    print(f"Model initialization warning: {e}")

# ─────────────────────────────────────────────
#  Single Prediction
# ─────────────────────────────────────────────
@api_view(['POST'])
def predict(request):
    """
    Make a single churn prediction for one customer.
    
    Request body should contain all customer features:
    - Age (int)
    - Subscription_Duration_Months (int)
    - Contract_Type (str: 'Monthly', 'Quarterly', 'Annual')
    - Monthly_Logins (int)
    - Last_Purchase_Days_Ago (int)
    - App_Usage_Time_Min (int)
    - Monthly_Spend (float)
    - Discount_Usage_Percentage (float)
    - Customer_Support_Calls (int)
    - Satisfaction_Score (int, 1-5)
    - customer_id (str, optional)
    """
    # Validate input data
    serializer = PredictionInputSerializer(data=request.data)
    if not serializer.is_valid():
        return Response(
            {'errors': serializer.errors}, 
            status=status.HTTP_400_BAD_REQUEST
        )

    data = serializer.validated_data
    customer_id = data.pop('customer_id', '')

    # Make prediction
    result = predict_for_customer(data)
    
    # Check for errors
    if 'error' in result:
        return Response(
            {'error': result['error']}, 
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

    is_churn = result['is_churn']
    prob = result['churn_probability']
    risk_level = result['risk_level']

    # Save to database
    try:
        record = ChurnPrediction.objects.create(
            customer_ref_id=customer_id,
            age=data['Age'],
            subscription_duration_months=data['Subscription_Duration_Months'],
            contract_type=data['Contract_Type'],
            monthly_logins=data['Monthly_Logins'],
            last_purchase_days_ago=data['Last_Purchase_Days_Ago'],
            app_usage_time_min=data['App_Usage_Time_Min'],
            monthly_spend=data['Monthly_Spend'],
            discount_usage_percentage=data['Discount_Usage_Percentage'],
            customer_support_calls=data['Customer_Support_Calls'],
            satisfaction_score=data['Satisfaction_Score'],
            is_churn=is_churn,
            churn_probability=prob,
            risk_level=risk_level,
        )
    except Exception as e:
        return Response(
            {'error': f'Failed to save prediction: {str(e)}'},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

    # Return response with explanations
    return Response({
        'id': record.id,
        'customer_id': customer_id,
        'is_churn': is_churn,
        'churn_probability': round(prob, 4),
        'churn_probability_pct': round(prob * 100, 1),
        'risk_level': risk_level,
        'risk_factors': get_risk_factors(data, prob),
        'recommendations': get_recommendations(is_churn, prob, data),
        'created_at': record.created_at,
    }, status=status.HTTP_200_OK)


# ─────────────────────────────────────────────
#  Batch Prediction
# ─────────────────────────────────────────────
@api_view(['POST'])
def batch_predict(request):
    """
    Make predictions for multiple customers.
    
    Request body can be:
    - List of customer objects
    - Dictionary with 'customers' key containing list
    """
    # Parse input
    if isinstance(request.data, list):
        customers = request.data
    elif isinstance(request.data, dict):
        customers = request.data.get('customers', [])
    else:
        customers = []

    if not customers:
        return Response(
            {'error': 'No customers provided. Send a list of customers or {"customers": [...]}'},
            status=status.HTTP_400_BAD_REQUEST
        )

    # Check if model is available
    try:
        get_model()
    except Exception as e:
        return Response(
            {'error': f'Model not available: {str(e)}'},
            status=status.HTTP_503_SERVICE_UNAVAILABLE
        )

    results = []
    errors = []
    
    for i, customer_data in enumerate(customers):
        # Validate each customer
        serializer = PredictionInputSerializer(data=customer_data)
        if not serializer.is_valid():
            errors.append({
                'index': i, 
                'errors': serializer.errors,
                'customer_id': customer_data.get('customer_id', f'index_{i}')
            })
            continue

        data = serializer.validated_data
        customer_id = data.pop('customer_id', f'CUST-{i+1:04d}')

        # Make prediction
        result = predict_for_customer(data)
        
        if 'error' in result:
            errors.append({
                'index': i,
                'error': result['error'],
                'customer_id': customer_id
            })
            continue
            
        is_churn = result['is_churn']
        prob = result['churn_probability']
        risk_level = result['risk_level']

        # Save to database
        try:
            record = ChurnPrediction.objects.create(
                customer_ref_id=customer_id,
                age=data['Age'],
                subscription_duration_months=data['Subscription_Duration_Months'],
                contract_type=data['Contract_Type'],
                monthly_logins=data['Monthly_Logins'],
                last_purchase_days_ago=data['Last_Purchase_Days_Ago'],
                app_usage_time_min=data['App_Usage_Time_Min'],
                monthly_spend=data['Monthly_Spend'],
                discount_usage_percentage=data['Discount_Usage_Percentage'],
                customer_support_calls=data['Customer_Support_Calls'],
                satisfaction_score=data['Satisfaction_Score'],
                is_churn=is_churn,
                churn_probability=prob,
                risk_level=risk_level,
            )

            results.append({
                'id': record.id,
                'customer_id': customer_id,
                'is_churn': is_churn,
                'churn_probability': round(prob, 4),
                'churn_probability_pct': round(prob * 100, 1),
                'risk_level': risk_level,
                'index': i
            })
        except Exception as e:
            errors.append({
                'index': i,
                'error': f'Failed to save: {str(e)}',
                'customer_id': customer_id
            })

    return Response({
        'total': len(customers),
        'processed': len(results),
        'failed': len(errors),
        'results': results,
        'errors': errors,
    }, status=status.HTTP_200_OK)


# ─────────────────────────────────────────────
#  Prediction History
# ─────────────────────────────────────────────
@api_view(['GET'])
def history(request):
    """
    Get paginated prediction history with optional filters.
    
    Query parameters:
    - page: Page number (default: 1)
    - limit: Items per page (default: 50)
    - risk_level: Filter by risk level (Low, Medium, High, Critical)
    - is_churn: Filter by churn status (true/false)
    - customer_id: Filter by customer ID (partial match)
    """
    queryset = ChurnPrediction.objects.all().order_by('-created_at')
    
    # Apply filters
    risk_filter = request.query_params.get('risk_level')
    if risk_filter:
        queryset = queryset.filter(risk_level=risk_filter)
        
    churn_filter = request.query_params.get('is_churn')
    if churn_filter:
        queryset = queryset.filter(is_churn=churn_filter.lower() == 'true')
        
    customer_filter = request.query_params.get('customer_id')
    if customer_filter:
        queryset = queryset.filter(customer_ref_id__icontains=customer_filter)
    
    # Pagination
    try:
        page = int(request.query_params.get('page', 1))
        limit = int(request.query_params.get('limit', 50))
    except ValueError:
        page, limit = 1, 50
    
    # Ensure valid values
    page = max(1, page)
    limit = min(100, max(1, limit))  # Limit between 1 and 100
    
    start = (page - 1) * limit
    end = page * limit
    total = queryset.count()
    
    # Get paginated results
    predictions = queryset[start:end]
    
    # Convert to list of dicts for JSON response
    results = []
    for pred in predictions:
        results.append({
            'id': pred.id,
            'customer_id': pred.customer_ref_id,
            'is_churn': pred.is_churn,
            'churn_probability': pred.churn_probability,
            'risk_level': pred.risk_level,
            'age': pred.age,
            'contract_type': pred.contract_type,
            'monthly_spend': pred.monthly_spend,
            'satisfaction_score': pred.satisfaction_score,
            'created_at': pred.created_at
        })
    
    return Response({
        'results': results,
        'pagination': {
            'page': page,
            'limit': limit,
            'total': total,
            'pages': (total + limit - 1) // limit,
            'has_next': end < total,
            'has_previous': page > 1
        }
    }, status=status.HTTP_200_OK)


# ─────────────────────────────────────────────
#  Customer List (Unique Customers with Latest Status)
# ─────────────────────────────────────────────
@api_view(['GET'])
def customer_list(request):
    """
    Get list of unique customers with their latest prediction status.
    
    Query parameters:
    - page: Page number (default: 1)
    - limit: Items per page (default: 50)
    - search: Search by customer ID
    - status: Filter by status (all, churn, safe)
    """
    # Pagination params
    try:
        page = int(request.query_params.get('page', 1))
        limit = int(request.query_params.get('limit', 50))
    except ValueError:
        page, limit = 1, 50
    
    page = max(1, page)
    limit = min(100, max(1, limit))
    
    search_query = request.query_params.get('search', '').strip()
    status_filter = request.query_params.get('status', 'all')

    start = (page - 1) * limit
    end = page * limit

    # Get the latest prediction ID for each customer
    latest_ids_subquery = (
        ChurnPrediction.objects
        .values('customer_ref_id')
        .annotate(max_id=Max('id'))
        .filter(customer_ref_id__isnull=False)
    )
    
    # Apply search filter
    if search_query:
        latest_ids_subquery = latest_ids_subquery.filter(
            customer_ref_id__icontains=search_query
        )

    # Apply status filter on the latest predictions
    if status_filter == 'churn':
        # Get customers whose latest prediction is churn
        churn_customers = ChurnPrediction.objects.filter(
            id__in=latest_ids_subquery.values('max_id'),
            is_churn=True
        ).values('customer_ref_id')
        latest_ids_subquery = latest_ids_subquery.filter(
            customer_ref_id__in=churn_customers
        )
    elif status_filter == 'safe':
        # Get customers whose latest prediction is not churn
        safe_customers = ChurnPrediction.objects.filter(
            id__in=latest_ids_subquery.values('max_id'),
            is_churn=False
        ).values('customer_ref_id')
        latest_ids_subquery = latest_ids_subquery.filter(
            customer_ref_id__in=safe_customers
        )

    # Get total count after filters
    total_count = latest_ids_subquery.count()
    
    # Get paginated latest IDs
    latest_ids_info = latest_ids_subquery.order_by('-max_id')[start:end]
    latest_ids = [item['max_id'] for item in latest_ids_info]
    
    # Fetch the actual prediction records
    latest_predictions = ChurnPrediction.objects.filter(id__in=latest_ids)
    
    # Get prediction counts for these customers
    customer_refs = [p.customer_ref_id for p in latest_predictions]
    counts = (
        ChurnPrediction.objects
        .filter(customer_ref_id__in=customer_refs)
        .values('customer_ref_id')
        .annotate(total=Count('id'))
    )
    counts_map = {item['customer_ref_id']: item['total'] for item in counts}
    
    # Build results
    results = []
    for pred in latest_predictions:
        results.append({
            'customer_id': pred.customer_ref_id,
            'is_churn': pred.is_churn,
            'churn_probability': pred.churn_probability,
            'risk_level': pred.risk_level,
            'last_prediction_date': pred.created_at,
            'total_predictions': counts_map.get(pred.customer_ref_id, 0),
            'age': pred.age,
            'contract_type': pred.contract_type,
            'monthly_spend': pred.monthly_spend
        })
    
    # Get overall statistics for the filter tabs
    all_latest_ids = (
        ChurnPrediction.objects
        .values('customer_ref_id')
        .annotate(max_id=Max('id'))
        .values_list('max_id', flat=True)
    )
    
    stats = ChurnPrediction.objects.filter(id__in=all_latest_ids).aggregate(
        churn_count=Count('id', filter=Q(is_churn=True)),
        safe_count=Count('id', filter=Q(is_churn=False))
    )

    return Response({
        'results': results,
        'pagination': {
            'page': page,
            'limit': limit,
            'total': total_count,
            'pages': (total_count + limit - 1) // limit,
            'has_next': end < total_count,
            'has_previous': page > 1,
            'churn_count': stats['churn_count'],
            'safe_count': stats['safe_count']
        }
    }, status=status.HTTP_200_OK)


# ─────────────────────────────────────────────
#  Customer Detail
# ─────────────────────────────────────────────
@api_view(['GET'])
def customer_detail(request, customer_id):
    """
    Get full history and recommendations for a specific customer.
    """
    predictions = ChurnPrediction.objects.filter(
        customer_ref_id=customer_id
    ).order_by('-created_at')
    
    if not predictions.exists():
        return Response(
            {'error': f'Customer with ID {customer_id} not found'},
            status=status.HTTP_404_NOT_FOUND
        )
    
    latest = predictions.first()
    
    # Prepare history data for charts
    history_data = [
        {
            'date': p.created_at,
            'churn_probability': p.churn_probability,
            'risk_level': p.risk_level,
            'is_churn': p.is_churn
        }
        for p in predictions
    ]
    
    # Reconstruct data dictionary for XAI
    data_dict = {
        'Age': latest.age,
        'Subscription_Duration_Months': latest.subscription_duration_months,
        'Contract_Type': latest.contract_type,
        'Monthly_Logins': latest.monthly_logins,
        'Last_Purchase_Days_Ago': latest.last_purchase_days_ago,
        'App_Usage_Time_Min': latest.app_usage_time_min,
        'Monthly_Spend': float(latest.monthly_spend),
        'Discount_Usage_Percentage': float(latest.discount_usage_percentage),
        'Customer_Support_Calls': latest.customer_support_calls,
        'Satisfaction_Score': latest.satisfaction_score,
    }

    # Get risk factors and recommendations
    risk_factors = get_risk_factors(data_dict, latest.churn_probability)
    recommendations = get_recommendations(
        latest.is_churn, 
        latest.churn_probability, 
        data_dict
    )

    return Response({
        'customer_id': customer_id,
        'latest_prediction': {
            'id': latest.id,
            'is_churn': latest.is_churn,
            'churn_probability': latest.churn_probability,
            'risk_level': latest.risk_level,
            'risk_factors': risk_factors,
            'recommendations': recommendations,
            'date': latest.created_at,
            'data': data_dict
        },
        'history': history_data,
        'total_predictions': predictions.count()
    }, status=status.HTTP_200_OK)


# ─────────────────────────────────────────────
#  Statistics
# ─────────────────────────────────────────────
@api_view(['GET'])
def stats(request):
    """
    Get overall statistics about predictions.
    
    Query parameters:
    - risk_level: Filter by risk level (optional)
    """
    queryset = ChurnPrediction.objects.all()

    # Optional filter by risk level
    risk_filter = request.query_params.get('risk_level')
    if risk_filter:
        queryset = queryset.filter(risk_level=risk_filter)

    total = queryset.count()
    
    if total == 0:
        return Response({
            'total_predictions': 0,
            'churned': 0,
            'not_churned': 0,
            'churn_rate_pct': 0,
            'avg_churn_probability_pct': 0,
            'risk_breakdown': [],
            'daily_trend': []
        })
    
    churned = queryset.filter(is_churn=True).count()
    not_churned = total - churned
    avg_prob = queryset.aggregate(avg=Avg('churn_probability'))['avg'] or 0

    # Breakdown by risk level
    risk_breakdown = (
        queryset.values('risk_level')
        .annotate(count=Count('id'))
        .order_by('risk_level')
    )

    # Daily trend of predictions (last 30 days)
    from django.utils import timezone
    from datetime import timedelta
    
    thirty_days_ago = timezone.now() - timedelta(days=30)
    
    daily_trend = (
        queryset
        .filter(created_at__gte=thirty_days_ago)
        .annotate(date=TruncDate('created_at'))
        .values('date')
        .annotate(
            total=Count('id'),
            churned=Count('id', filter=Q(is_churn=True))
        )
        .order_by('date')
    )

    return Response({
        'total_predictions': total,
        'churned': churned,
        'not_churned': not_churned,
        'churn_rate_pct': round(churned / total * 100, 1),
        'avg_churn_probability_pct': round(avg_prob * 100, 1),
        'risk_breakdown': [
            {'level': item['risk_level'], 'count': item['count']}
            for item in risk_breakdown
        ],
        'daily_trend': [
            {
                'date': str(d['date']),
                'total': d['total'],
                'churned': d['churned'],
            }
            for d in daily_trend
        ],
    }, status=status.HTTP_200_OK)


# ─────────────────────────────────────────────
#  Delete Prediction
# ─────────────────────────────────────────────
@api_view(['DELETE'])
def delete_prediction(request, pk):
    """
    Delete a specific prediction by ID.
    """
    try:
        record = ChurnPrediction.objects.get(pk=pk)
        customer_id = record.customer_ref_id
        record.delete()
        return Response({
            'message': f'Prediction {pk} for customer {customer_id} deleted successfully'
        }, status=status.HTTP_200_OK)
    except ChurnPrediction.DoesNotExist:
        return Response(
            {'error': f'Prediction with ID {pk} not found'},
            status=status.HTTP_404_NOT_FOUND
        )
    except Exception as e:
        return Response(
            {'error': f'Failed to delete: {str(e)}'},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


# ─────────────────────────────────────────────
#  Health Check
# ─────────────────────────────────────────────
@api_view(['GET'])
def health_check(request):
    """
    Check if the API and model are ready.
    """
    from datetime import datetime
    try:
        from .utils import get_model
        model = get_model()
        return Response({
            'status': 'healthy',
            'model_ready': model is not None,
            'timestamp': datetime.now()
        }, status=status.HTTP_200_OK)
    except Exception as e:
        return Response({
            'status': 'unhealthy',
            'error': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

def home_page(request):
    """
    Serve the branded Amazooh.com home page.
    """
    from django.shortcuts import render
    return render(request, 'home.html')