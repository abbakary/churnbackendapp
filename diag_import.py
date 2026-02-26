import os
import sys
import traceback

# Add the current directory to sys.path
sys.path.append(os.getcwd())

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'customerchurn.settings')

try:
    print("Attempting to import django...")
    import django
    django.setup()
    print("Django setup successful")
    
    print("Attempting to import WSGI application...")
    from customerchurn.wsgi import application
    print("WSGI application imported successfully")
except Exception:
    print("\n--- IMPORT ERROR DETECTED ---")
    traceback.print_exc()
    sys.exit(1)
