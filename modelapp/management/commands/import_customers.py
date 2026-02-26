"""
Management command: import_customers
Usage:  python manage.py import_customers
        python manage.py import_customers --clear

Reads churn.csv from the modelapp directory and bulk-upserts records
into the Customer table.  Column mapping is tolerant of different
capitalisation styles coming from train_model.py.
"""
import os
import pandas as pd
from django.core.management.base import BaseCommand
from modelapp.models import Customer


# Canonical column map: csv_col -> model_field
COLUMN_MAP = {
    'Age':                        'age',
    'Subscription_Duration_Months': 'subscription_duration_months',
    'Contract_Type':              'contract_type',
    'Monthly_Logins':             'monthly_logins',
    'Last_Purchase_Days_Ago':     'last_purchase_days_ago',
    'App_Usage_Time_Min':         'app_usage_time_min',
    'Monthly_Spend':              'monthly_spend',
    'Discount_Usage_Percentage':  'discount_usage_percentage',
    'Customer_Support_Calls':     'customer_support_calls',
    'Satisfaction_Score':         'satisfaction_score',
    'Is_Churn':                   'actual_churn',
}


class Command(BaseCommand):
    help = 'Import customer records from churn.csv into the Customer database table.'

    def add_arguments(self, parser):
        parser.add_argument(
            '--clear', action='store_true',
            help='Delete all existing Customer records before importing.'
        )
        parser.add_argument(
            '--limit', type=int, default=None,
            help='Only import the first N rows (useful for testing).'
        )

    def handle(self, *args, **options):
        csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'churn.csv')

        if not os.path.exists(csv_path):
            self.stderr.write(self.style.ERROR(f'churn.csv not found at: {csv_path}'))
            self.stderr.write('Run: python modelapp/train_model.py  to generate it first.')
            return

        if options['clear']:
            count, _ = Customer.objects.all().delete()
            self.stdout.write(self.style.WARNING(f'Cleared {count} existing customer records.'))

        self.stdout.write(f'Reading {csv_path} …')
        df = pd.read_csv(csv_path)

        if options['limit']:
            df = df.head(options['limit'])

        # Normalise: strip spaces from column names
        df.columns = [c.strip() for c in df.columns]

        # Validate required columns exist
        missing = [c for c in COLUMN_MAP if c not in df.columns and c != 'Is_Churn']
        if missing:
            self.stderr.write(self.style.ERROR(f'Missing columns in CSV: {missing}'))
            return

        created = updated = skipped = 0
        batch_create = []
        batch_update = []

        for idx, row in df.iterrows():
            cid = f'CUST-{idx + 1:05d}'

            kwargs = {
                'age':                         int(row['Age']),
                'subscription_duration_months': int(row['Subscription_Duration_Months']),
                'contract_type':               str(row['Contract_Type']).strip(),
                'monthly_logins':              int(row['Monthly_Logins']),
                'last_purchase_days_ago':      int(row['Last_Purchase_Days_Ago']),
                'app_usage_time_min':          float(row['App_Usage_Time_Min']),
                'monthly_spend':               float(row['Monthly_Spend']),
                'discount_usage_percentage':   float(row['Discount_Usage_Percentage']),
                'customer_support_calls':      int(row['Customer_Support_Calls']),
                'satisfaction_score':          int(row['Satisfaction_Score']),
                'actual_churn':                bool(row['Is_Churn']) if 'Is_Churn' in df.columns else None,
            }

            try:
                obj, was_created = Customer.objects.update_or_create(
                    customer_id=cid, defaults=kwargs
                )
                if was_created:
                    created += 1
                else:
                    updated += 1
            except Exception as e:
                self.stderr.write(f'Row {idx}: {e}')
                skipped += 1

            if (idx + 1) % 500 == 0:
                self.stdout.write(f'  … processed {idx + 1} rows')

        self.stdout.write(self.style.SUCCESS(
            f'\nDone! Created: {created}  Updated: {updated}  Skipped: {skipped}'
        ))
        self.stdout.write(f'Total customers in DB: {Customer.objects.count()}')
