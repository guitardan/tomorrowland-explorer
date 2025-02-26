from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('todos/', views.todos, name='todos'),
    path('scrape/', views.scrape_webpages, name='scrape'),
    path('make_db/', views.create_database, name='make_db'),
    path('index/', views.index, name="index"),
]