from django.urls import path

from . import views
app_name = "interface"
urlpatterns = [
    path('', views.HomeView.as_view(), name='image',kwargs={"pk":1}),
    path('apply', views.select, name='select'),
    path('<str:mode>/<int:prop>/<int:angle>', views.ResultsView, name='results')
]

