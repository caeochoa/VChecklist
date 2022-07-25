from django.urls import path

from . import views
app_name = "interface"
urlpatterns = [
    path('<int:pk>', views.HomeView.as_view(), name='image'),
    path('apply', views.select, name='select'),
    path('<str:mode>/<int:prop>/<int:angle>', views.ResultsView, name='results')
]

