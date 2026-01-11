from django.urls import path
from .views import upload_dataset, visualizar_dataset, process_dataset,prepare_dataset
from .views import pipelines_personalizados_comprimido, evaluar_modelo

urlpatterns = [
    path("upload_dataset/", upload_dataset),
    path("visualizar_dataset/",visualizar_dataset),
    path("process_dataset/",process_dataset),
    path("prepare_dataset/",prepare_dataset),
    path("pipelines_personalizados/",pipelines_personalizados_comprimido),
    path("evaluar_modelo/",evaluar_modelo)
]