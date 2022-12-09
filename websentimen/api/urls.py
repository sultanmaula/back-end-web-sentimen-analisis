from django.urls import include, path
from rest_framework import routers

from . import views

from api.views import preprocessing, preprocessing_post, preprocessing_reset, clustering_post, clustering, clustering_reset, knn_post, knn, knn_reset, smoteknn_post, smoteknn, smoteknn_reset

router = routers.DefaultRouter()
# router.register(r'preprocessing', views.PreprocessingViewSet)

# Wire up our API using automatic URL routing.
# Additionally, we include login URLs for the browsable API.
urlpatterns = [
    path('', include(router.urls)),
    path('api-auth/', include('rest_framework.urls', namespace='rest_framework')),

    # Preprocessing
    path('preprocessing/post/', preprocessing_post),
    path('preprocessing/reset/', preprocessing_reset),
    path('preprocessing/', preprocessing),
    
    # Clustering
    path('clustering/post/', clustering_post),
    path('clustering/reset/', clustering_reset),
    path('clustering/', clustering),
    
    # KNN
    path('knn/post/', knn_post),
    path('knn/reset/', knn_reset),
    path('knn/', knn),
    
    # Smote + KNN
    path('smote-knn/post/', smoteknn_post),
    path('smote-knn/reset/', smoteknn_reset),
    path('smote-knn/', smoteknn),
]