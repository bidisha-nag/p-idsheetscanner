from django.urls import path
from .views import generate_result
from django.conf import settings
from django.conf.urls.static import static
from myapp import views

urlpatterns = [
    path('', generate_result, name='generate_result'),
    path('generate_result', generate_result, name='generate_result'),
    path("logout",views.logoutUser,name='logout'),
    path("login",views.loginUser,name='login'),
    # Add other URLs as needed
]+ static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
