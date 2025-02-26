from django.shortcuts import render
from .models import TodoItem
from django.http import JsonResponse
from django.shortcuts import render
from .services import scrape, create_db, perform_query
import threading

def home(request):
    return render(request, 'home.html')

def todos(request):
    items = TodoItem.objects.all() # using the django ORM
    return render(request, 'todos.html', {'todos': items})

def scrape_webpages(request):
    thread = threading.Thread(target=scrape)
    thread.start()
    return JsonResponse({'status': 'scraping sites...'})

def create_database(request):
    thread = threading.Thread(target=create_db)
    thread.start()
    return JsonResponse({'status': 'building database...'}) 

def index(request):
    if request.method == 'POST' and request.headers.get("X-Requested-With") == "XMLHttpRequest":
        query = request.POST.get('query')
        result = perform_query(query)
        return JsonResponse({"result": result}) # result) # 
    return render(request, 'index.html')