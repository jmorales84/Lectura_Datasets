import uuid
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import default_storage
from django.conf import settings
import os,json, base64, gzip, io 
from io import BytesIO
# importaciones para tratamiento de datos y funciones auxiliares 
import pandas as pd 
import arff
import numpy as np 
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt 
import math
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from pandas.plotting import scatter_matrix
# Funciones auxiliares y para los enpoints 
# Paleta de colores para los histogramas para division del dataset 
COLOR_MAP = {
    "train": "#4CAF50",       # verde
    "validation": "#FFC107",  # amarillo
    "test": "#2196F3"         # azul
}
# Evitar errores de GUI en el servidor
matplotlib.use('Agg')
def detect_format(filename, head_text):
    ext = os.path.splitext(filename)[1].lower()

    if ext == ".arff":
        return "arff"

    if ext == ".csv":
        return "csv"

    if ext == ".txt":
        return "txt"

    # fallback por contenido
    if "@relation" in head_text.lower():
        return "arff"

    if "," in head_text:
        return "csv"

    return "unknown"

def load_arff_dataframe(file_path: str) -> pd.DataFrame:
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        dataset = arff.load(f)

    attributes = [attr[0] for attr in dataset["attributes"]]
    df = pd.DataFrame(dataset["data"], columns=attributes)
    return df

def load_csv_dataframe(path):
    return pd.read_csv(path)

def load_txt_dataframe(path):
    return pd.read_csv(
        path,
        sep=None,          # autodetecta separador
        engine="python"
    )

def dataframe_info(df: pd.DataFrame) -> dict:
    return {
        "total_filas": int(df.shape[0]),
        "total_columnas": int(df.shape[1]),
        "columns": [
            {
                "name": col,
                "dtype": str(df[col].dtype),
                "non_null": int(df[col].notnull().sum())
            }
            for col in df.columns
        ]
    }

def plot_histogram_base64(df, column=None, title="Histograma", color="#4CAF50"):
    """Genera un histograma vistoso y devuelve la imagen en base64"""
    plt.figure(figsize=(6,4))
    if column and column in df.columns:
        df[column].value_counts().plot(kind='bar', color=color)
    else:
        df.hist(figsize=(6,4), color=color)
    plt.title(title, fontsize=12, fontweight="bold")
    plt.xlabel("Categorías / Valores")
    plt.ylabel("Frecuencia")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

def generate_plots(df, dataset_id):
    plots_dir = os.path.join(settings.MEDIA_ROOT, "plots", dataset_id)
    os.makedirs(plots_dir, exist_ok=True)
    
    plots = []

    # 1. Configuración de la Rejilla (Grid)
    columnas = df.columns
    num_plots = len(columnas)
    
    # Calculamos cuántas filas necesitamos (asumiendo 3 columnas por fila para que se lean bien los nombres)
    cols_por_fila = 3
    filas = math.ceil(num_plots / cols_por_fila)
    
    # Creamos una figura gigante
    fig, axes = plt.subplots(filas, cols_por_fila, figsize=(20, 5 * filas))
    axes = axes.flatten() # Aplanamos el array para iterar fácil

    # 2. Iterar sobre cada columna de tus datos
    for i, col_name in enumerate(columnas):
        ax = axes[i]
        
        # Detectar si es Numérica o Categórica
        if np.issubdtype(df[col_name].dtype, np.number):
            # --- ES NUMÉRICA (Histograma) ---
            df[col_name].plot(kind='hist', bins=30, ax=ax, color='skyblue', edgecolor='black')
            ax.set_title(f"{col_name} (Numérica)", fontsize=14, fontweight='bold')
            ax.set_xlabel("Valor")
            ax.set_ylabel("Frecuencia")
        else:
            # --- ES CATEGÓRICA (Gráfico de Barras) ---
            # Tomamos el Top 10 para que no se amontone el texto
            conteo = df[col_name].value_counts().head(10)
            conteo.plot(kind='bar', ax=ax, color='salmon', edgecolor='black')
            
            ax.set_title(f"{col_name} (Top 10)", fontsize=14, fontweight='bold')
            ax.set_xlabel("Categoría")
            ax.set_ylabel("Cantidad")
            
            # Rotar los nombres de abajo para que se lean bien
            ax.tick_params(axis='x', rotation=45, labelsize=10)

        ax.grid(axis='y', linestyle='--', alpha=0.7)

    # 3. Ocultar los espacios vacíos si sobran cuadros en la rejilla
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # 4. Ajustar espacios y guardar
    plt.tight_layout()
    
    filename = "analisis_completo.png"
    filepath = os.path.join(plots_dir, filename)
    
    plt.savefig(filepath)
    plt.close('all') # Limpiar memoria

    # 5. Retornar solo esta imagen maestra
    plots.append({
        "column": "Vista General",
        "category": "Todas las variables",
        "url": f"{settings.MEDIA_URL}plots/{dataset_id}/{filename}"
    })

    return plots

def train_val_test_split(df, test_size=0.4, val_size=0.2, stratify=None, rstate=42):
    """
    Versión unificada para división train/val/test.
    """
    # Si stratify es un string (nombre de columna), obtener la serie
    if isinstance(stratify, str):
        stratify_series = df[stratify]
    else:
        stratify_series = stratify
    
    # Primer split: separar test del resto
    df_train_val, df_test = train_test_split(
        df, 
        test_size=test_size, 
        random_state=rstate,
        shuffle=True, 
        stratify=stratify_series
    )
    
    # Si stratify es un string, obtener la serie para el subset train_val
    if isinstance(stratify, str):
        stratify_train_val = df_train_val[stratify]
    else:
        stratify_train_val = stratify_series
    
    # Calcular tamaño relativo para validation
    val_relative_size = val_size / (1 - test_size)
    
    # Segundo split: separar train y validation
    df_train, df_val = train_test_split(
        df_train_val,
        test_size=val_relative_size,
        random_state=rstate,
        shuffle=True,
        stratify=stratify_train_val
    )
    
    return df_train, df_val, df_test

def load_kdd_dataset(data_path):
    """Lectura del DataSet NSL-KDD"""
    with open(data_path, 'r') as f:
        dataset = arff.load(f)
        attributes = [attr[0] for attr in dataset['attributes']]
        return pd.DataFrame(dataset['data'], columns=attributes)

def df_to_json_safe(df, max_rows=10):
    """
    Convierte un DataFrame a JSON seguro para JS:
    """
    df_preview = df.head(max_rows).copy()
    
    # Reemplazar todos los valores problemáticos
    df_preview = df_preview.replace([np.inf, -np.inf], np.nan)
    df_preview = df_preview.where(pd.notnull(df_preview), None)
    
    # Convertir todas las columnas a string primero para evitar problemas
    for col in df_preview.columns:
        df_preview[col] = df_preview[col].apply(
            lambda x: (
                None if pd.isna(x)
                else int(x) if isinstance(x, (np.integer, np.int64))
                else float(x) if isinstance(x, (np.floating, np.float64))
                else str(x).encode('utf-8', 'ignore').decode('utf-8') 
                if isinstance(x, (np.bytes_, bytes, str))
                else str(x)
            )
        )
    
    return df_preview.to_dict(orient="records")

def save_confusion_matrix_image(cm, labels, media_root):
    os.makedirs(os.path.join(media_root, "evaluations"), exist_ok=True)

    filename = f"confusion_{uuid.uuid4().hex}.png"
    filepath = os.path.join(media_root, "evaluations", filename)

    plt.figure(figsize=(5, 4))
    plt.imshow(cm)
    plt.title("Matriz de Confusión")
    plt.xlabel("Predicción")
    plt.ylabel("Real")
    plt.colorbar()

    plt.xticks(range(len(labels)), labels)
    plt.yticks(range(len(labels)), labels)

    for i in range(len(labels)):
        for j in range(len(labels)):
            plt.text(j, i, cm[i, j],
                     ha="center", va="center")

    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()

    return f"/media/evaluations/{filename}"

def generate_correlation_plots(df, dataset_id):
    plots = []

    # Columnas exactas que queremos en el scatter matrix
    attributes = ["same_srv_rate", "dst_host_srv_count", "class", "dst_host_same_srv_rate"]

    # Verificar que existan en el DataFrame
    attributes = [col for col in attributes if col in df.columns]
    if not attributes:
        return plots

    # Crear figura scatter_matrix
    fig = scatter_matrix(
        df[attributes],
        diagonal="hist",
        figsize=(12, 8)
    )

    plt.suptitle("Scatter Matrix", fontsize=16, fontweight="bold")

    # Guardar en buffer y convertir a base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    plt.close()
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')

    plots.append({
        "type": "scatter_matrix",
        "title": "Scatter Matrix",
        "columns": attributes,
        "image_base64": f"data:image/png;base64,{img_base64}"
    })

    return plots

# =========================
# Funciones de visualización
# =========================

def generate_plots_base64(df):
    """Genera histogramas/barras por columna y devuelve base64"""
    plots = []
    for col in df.columns:
        fig, ax = plt.subplots(figsize=(6,4))
        if np.issubdtype(df[col].dtype, np.number):
            df[col].plot(kind='hist', bins=30, ax=ax, color='skyblue', edgecolor='black')
        else:
            df[col].value_counts().head(10).plot(kind='bar', ax=ax, color='salmon', edgecolor='black')
            ax.tick_params(axis='x', rotation=45)
        ax.set_title(col)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()

        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        plt.close()
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plots.append({
            "column": col,
            "title": f"Distribución de {col}",
            "image_base64": f"data:image/png;base64,{img_base64}",
            "url": ""
        })
    return plots


def generate_scatter_matrix_base64(df, attributes=None):
    """Genera scatter_matrix en base64. Si no se pasan atributos, usa columnas numéricas"""
    plots = []
    if attributes is None:
        attributes = df.select_dtypes(include=[np.number]).columns.tolist()
    else:
        attributes = [col for col in attributes if col in df.columns]

    if not attributes:
        return plots

    fig = scatter_matrix(df[attributes], diagonal="hist", figsize=(12, 8))
    plt.suptitle("Scatter Matrix", fontsize=16, fontweight="bold")

    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    plt.close()
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')

    plots.append({
        "type": "scatter_matrix",
        "title": "Scatter Matrix",
        "columns": attributes,
        "image_base64": f"data:image/png;base64,{img_base64}",
        "url": ""
    })
    return plots


def generate_confusion_matrix_base64(y_true, y_pred):
    """Genera matriz de confusión en base64"""
    labels = sorted(np.unique(y_true))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(5,4))
    cax = ax.matshow(cm, cmap="Blues")
    plt.title("Matriz de Confusión")
    plt.xlabel("Predicción")
    plt.ylabel("Real")
    plt.colorbar(cax)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(cm[i,j]), ha="center", va="center")

    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png', dpi=100)
    plt.close()
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')

    return [{
        "type": "confusion_matrix",
        "title": "Matriz de Confusión",
        "image_base64": f"data:image/png;base64,{img_base64}",
        "url": ""
    }]


# Clases para usar en el uso de creacion de los pipelines y transformadores
class DeleteNanRows(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X.dropna()

class CustomScaler(BaseEstimator, TransformerMixin):
    def __init__(self, attributes):
        self.attributes = attributes

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_copy = X.copy()

        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X_copy[self.attributes])

        X_scaled = pd.DataFrame(
            X_scaled,
            columns=self.attributes,
            index=X_copy.index
        )

        for attr in self.attributes:
            X_copy[attr] = X_scaled[attr]

        return X_copy

class CustomOneHotEncoding(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.encoder = OneHotEncoder(sparse_output=False)
        self.columns = None

    def fit(self, X, y=None):
        X_cat = X.select_dtypes(include=['object'])
        self.encoder.fit(X_cat)
        self.columns = self.encoder.get_feature_names_out(X_cat.columns)
        return self

    def transform(self, X, y=None):
        X_copy = X.copy()

        X_cat = X_copy.select_dtypes(include=['object'])
        X_num = X_copy.select_dtypes(exclude=['object'])

        X_cat_encoded = self.encoder.transform(X_cat)
        X_cat_encoded = pd.DataFrame(
            X_cat_encoded,
            columns=self.columns,
            index=X_copy.index
        )

        return pd.concat([X_num, X_cat_encoded], axis=1)

# endpoint general para subir archivo 
@csrf_exempt  # Solo si tu frontend no envía CSRF token; en producción reemplaza con CSRF seguro
def upload_dataset(request):
    if request.method != "POST":
        return JsonResponse({"error": "Método no permitido"}, status=405)
    
    file = request.FILES.get("dataset")
    if not file:
        return JsonResponse({"error": "No se recibió archivo"}, status=400)
    
    # Generar un ID único para cada dataset
    dataset_id = str(uuid.uuid4())

    # Carpeta de destino dentro de MEDIA_ROOT
    datasets_dir = os.path.join(settings.MEDIA_ROOT, "datasets")
    os.makedirs(datasets_dir, exist_ok=True)

    # Nombre de archivo seguro
    filename = f"{dataset_id}_{file.name}"
    file_path = os.path.join("datasets", filename)

    # Guardar archivo
    saved_path = default_storage.save(file_path, file)

    # Respuesta JSON
    return JsonResponse({
        "dataset_id": dataset_id,
        "filename": file.name,
        "path": saved_path
    })

# Enpoints para visualizacion del dataset 
@csrf_exempt
def visualizar_dataset(request):
    if request.method != "POST":
        return JsonResponse({"error": "Método no permitido"}, status=405)

    try:
        body = json.loads(request.body.decode("utf-8"))
    except Exception:
        return JsonResponse({"error": "JSON inválido"}, status=400)

    dataset_id = body.get("dataset_id")
    options = body.get("options", {})

    if not dataset_id:
        return JsonResponse({"error": "dataset_id requerido"}, status=400)

    datasets_dir = os.path.join(settings.MEDIA_ROOT, "datasets")
    # Buscamos el archivo que coincida con el ID
    archivo = next((f for f in os.listdir(datasets_dir) if f.startswith(dataset_id)), None)
    
    if not archivo:
        return JsonResponse({"error": "Dataset no encontrado"}, status=404)

    file_path = os.path.join(datasets_dir, archivo)

    # Cargar dataset (Usando tu función load_kdd_dataset que definiste arriba)
    try:
        df = load_kdd_dataset(file_path)
        df.columns = [str(c).strip().lower() for c in df.columns]
    except Exception as e:
        return JsonResponse({"error": f"Error al cargar: {str(e)}"}, status=500)

    # 1. Info general y Previsualización
    info = dataframe_info(df)
    preview_rows = options.get("preview_rows", 10)
    
    # 2. Generar Gráficos de Distribución (Histogramas/Barras en Base64)
    # Nota: Limitamos a las primeras 12 columnas para no saturar el JSON del front si el dataset es muy grande
    plots = generate_plots_base64(df.iloc[:, :12]) 

    # 3. Gráficos de Correlación (Scatter Matrix)
    # Usamos las columnas específicas que definiste en tu función original
    correlation_cols = ["same_srv_rate", "dst_host_srv_count", "dst_host_same_srv_rate"]
    correlation_plots = generate_scatter_matrix_base64(df, attributes=correlation_cols)

    # 4. Respuesta final estructurada para el Frontend
    response = {
        "dataset_id": dataset_id,
        "filename": archivo,
        "stats": {
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
        },
        "preview": {
            "columns": list(df.columns),
            "rows": df_to_json_safe(df, max_rows=preview_rows) # Usamos tu función safe para evitar errores de NaN en JSON
        },
        "info": info,
        "plots": plots,
        "correlation_plots": correlation_plots,
        # Eliminamos la matriz de confusión de aquí porque no tiene sentido antes de entrenar
    }
    return JsonResponse(response)

# Enpoints para la division del DataSet
@csrf_exempt
def process_dataset(request):
    if request.method != "POST":
        return JsonResponse({"error": "Método no permitido"}, status=405)

    try:
        body = json.loads(request.body.decode("utf-8"))
    except Exception:
        return JsonResponse({"error": "JSON inválido"}, status=400)

    dataset_id = body.get("dataset_id")
    options = body.get("options", {})

    if not dataset_id:
        return JsonResponse({"error": "dataset_id requerido"}, status=400)

    datasets_dir = os.path.join(settings.MEDIA_ROOT, "datasets")
    if not os.path.exists(datasets_dir):
        return JsonResponse({"error": "Directorio datasets no existe"}, status=500)

    archivo = next((f for f in os.listdir(datasets_dir) if f.startswith(dataset_id)), None)
    if not archivo:
        return JsonResponse({"error": "Dataset no encontrado"}, status=404)

    file_path = os.path.join(datasets_dir, archivo)

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        head_text = "".join(f.readlines()[:20])

    file_format = detect_format(archivo, head_text)

    try:
        if file_format == "arff":
            df = load_arff_dataframe(file_path)
        elif file_format == "csv":
            df = load_csv_dataframe(file_path)
        elif file_format == "txt":
            df = load_txt_dataframe(file_path)
        else:
            return JsonResponse({"error": f"Formato no soportado: {file_format}"}, status=400)
    except Exception as e:
        return JsonResponse({"error": f"Error al leer dataset: {str(e)}"}, status=500)

    # --- Normalizar nombres de columna ---
    df.columns = [str(c).strip().lower() for c in df.columns]

    # --- Verificar columna de estratificación ---
    stratify_col = next((c for c in df.columns if c == "protocol_type"), None)
    if not stratify_col:
        return JsonResponse({"error": 'El dataset debe contener "protocol_type"'}, status=400)

    # --- División real usando la función unificada ---
    test_size = options.get("test_size", 0.2)
    val_size = options.get("val_size", 0.2)
    
    df_train, df_val, df_test = train_val_test_split(
        df, 
        test_size=test_size,
        val_size=val_size,
        stratify=stratify_col,  # Ahora acepta string con nombre de columna
        rstate=42
    )

    # --- Histogramas vistosos ---
    train_hist = plot_histogram_base64(df_train, stratify_col, title="Train Set", color=COLOR_MAP["train"])
    val_hist = plot_histogram_base64(df_val, stratify_col, title="Validation Set", color=COLOR_MAP["validation"])
    test_hist = plot_histogram_base64(df_test, stratify_col, title="Test Set", color=COLOR_MAP["test"])

    response = {
        "dataset_id": dataset_id,
        "filename": archivo,
        "format": file_format,
        "split_sizes": {
            "train": df_train.shape[0],
            "validation": df_val.shape[0],
            "test": df_test.shape[0]
        },
        "histograms": {
            "train": train_hist,
            "validation": val_hist,
            "test": test_hist
        }
    }

    return JsonResponse(response)

# Enpoints para la preparacion del DataSet
@csrf_exempt
def prepare_dataset(request):
    if request.method != "POST":
        return JsonResponse({"error": "Método no permitido"}, status=405)

    try:
        body = json.loads(request.body.decode("utf-8"))
        dataset_id = body.get("dataset_id")
    except Exception:
        return JsonResponse({"error": "JSON inválido"}, status=400)

    if not dataset_id:
        return JsonResponse({"error": "dataset_id requerido"}, status=400)

    datasets_dir = os.path.join(settings.MEDIA_ROOT, "datasets")
    archivo = next((f for f in os.listdir(datasets_dir) if f.startswith(dataset_id)), None)
    if not archivo:
        return JsonResponse({"error": "Dataset no encontrado"}, status=404)

    file_path = os.path.join(datasets_dir, archivo)
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        head_text = "".join(f.readlines()[:20])

    file_format = detect_format(archivo, head_text)
    try:
        if file_format == "arff":
            df = load_arff_dataframe(file_path)
        elif file_format == "csv":
            df = load_csv_dataframe(file_path)
        elif file_format == "txt":
            df = load_txt_dataframe(file_path)
        else:
            return JsonResponse({"error": f"Formato no soportado: {file_format}"}, status=400)
    except Exception as e:
        return JsonResponse({"error": f"Error al leer dataset: {str(e)}"}, status=500)

    # Normalizar nombres de columnas
    df.columns = [str(c).strip().lower() for c in df.columns]

    # Validar columna de estratificación
    stratify_col = next((c for c in df.columns if c == "protocol_type"), None)
    if not stratify_col:
        return JsonResponse({"error": 'El dataset debe contener "protocol_type"'}, status=400)

    # --- Split real usando la función unificada ---
    # Reemplaza la lógica anterior por el split con tamaños específicos
    train_set, val_set, test_set = train_val_test_split(
        df, 
        test_size=0.4, 
        val_size=0.2, 
        stratify=stratify_col 
    )

    # --- Limpieza y Preprocesamiento de datos (sobre el conjunto de entrenamiento) ---
    # Nota: Se asume que existe una columna llamada "class" para el target
    if "class" in train_set.columns:
        X_train = train_set.drop("class", axis=1)
        y_train = train_set["class"].copy()
    else:
        X_train = train_set.copy()

    # Introducir nulos como ejemplo pedagógico
    X_train.loc[(X_train['src_bytes'] > 400) & (X_train['src_bytes'] < 800), 'src_bytes'] = np.nan
    X_train.loc[(X_train['dst_bytes'] > 500) & (X_train['dst_bytes'] < 2000), 'dst_bytes'] = np.nan

    # Copia para transformaciones
    X_train_processed = X_train.copy()

    # 1. Rellenar valores nulos con mediana
    imputer = SimpleImputer(strategy='median')
    X_train_num = X_train_processed.select_dtypes(include=[np.number])
    if not X_train_num.empty:
        imputer.fit(X_train_num)
        X_train_num_filled = imputer.transform(X_train_num)
        X_train_processed[X_train_num.columns] = X_train_num_filled

    # 2. One-hot encoding de protocolo
    X_train_processed = pd.get_dummies(X_train_processed, columns=['protocol_type'], prefix='protocol')

    # 3. Escalar atributos numéricos específicos
    scale_attrs = [col for col in ['src_bytes', 'dst_bytes'] if col in X_train_processed.columns]
    robust_scaler = RobustScaler()
    if scale_attrs:
        X_train_processed[scale_attrs] = robust_scaler.fit_transform(X_train_processed[scale_attrs])

    # --- Preparar JSON de resultados ---
    response = {
        "dataset_id": dataset_id,
        "split_sizes": {
            "train": len(train_set),
            "validation": len(val_set),
            "test": len(test_set)
        },
        "imputer_median": dict(zip(X_train_num.columns, imputer.statistics_)) if not X_train_num.empty else {},
        "one_hot_columns": [c for c in X_train_processed.columns if "protocol_" in c],
        "scaled_columns": scale_attrs,
        "X_train_preview": X_train_processed.head(10).to_dict(orient="records")
    }

    return JsonResponse(response)

# Enpoint para la pipelinees y transformadores 
@csrf_exempt
def pipelines_personalizados_comprimido(request):
    if request.method != "POST":
        return JsonResponse({"error": "Método no permitido"}, status=405)

    try:
        body = json.loads(request.body.decode("utf-8"))
        dataset_id = body.get("dataset_id")
        if not dataset_id:
            return JsonResponse({"error": "dataset_id requerido"}, status=400)

        # Buscar dataset
        datasets_dir = os.path.join(settings.MEDIA_ROOT, "datasets")
        archivo = next((f for f in os.listdir(datasets_dir) if f.startswith(dataset_id)), None)
        if not archivo:
            return JsonResponse({"error": "Dataset no encontrado"}, status=404)
        data_path = os.path.join(datasets_dir, archivo)

        # Cargar dataset
        df = load_kdd_dataset(data_path)
        
        # Normalizar nombres para consistencia
        df.columns = [str(c).strip().lower() for c in df.columns]
        
        if 'class' not in df.columns:
            return JsonResponse({"error": 'El dataset debe contener la columna "class"'}, status=400)

        # --- Split real usando la función unificada ---
        # Se reemplaza el split anterior por la configuración solicitada
        train_set, val_set, test_set = train_val_test_split(
            df, 
            test_size=0.4, 
            val_size=0.2, 
            stratify='class' 
        )

        # Preparación de datos para transformaciones manuales
        X_train_original = train_set.drop("class", axis=1)
        X_train = X_train_original.copy()
        
        # Introducir nulos artificiales para demostración
        X_train.loc[(X_train['src_bytes'] > 400) & (X_train['src_bytes'] < 800), 'src_bytes'] = np.nan
        X_train.loc[(X_train['dst_bytes'] > 500) & (X_train['dst_bytes'] < 2000), 'dst_bytes'] = np.nan

        # --- Transformaciones Manuales Paso a Paso ---
        X_after_delete = DeleteNanRows().fit_transform(X_train)
        X_after_scaler = CustomScaler(['src_bytes', 'dst_bytes']).fit_transform(X_after_delete)
        X_after_onehot = CustomOneHotEncoding().fit_transform(X_after_scaler)

        # --- Transformación usando Scikit-Learn Pipeline ---
        num_attribs = list(X_train_original.select_dtypes(exclude=['object']))
        cat_attribs = list(X_train_original.select_dtypes(include=['object']))
        
        num_pipeline = Pipeline([
            ("delete_nan", DeleteNanRows()),
            ("scaler", CustomScaler(["src_bytes", "dst_bytes"]))
        ])
        
        full_pipeline = ColumnTransformer([
            ("num", num_pipeline, num_attribs),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_attribs)
        ])
        
        # Ajustar y transformar con el pipeline completo
        X_pipeline_raw = full_pipeline.fit_transform(X_train_original)
        
        # Intentar reconstruir DataFrame con nombres de columnas si es posible
        try:
            # Intentamos obtener los nombres de las columnas tras OneHot si la versión de sklearn lo permite
            new_cols = list(pd.get_dummies(X_train_original).columns)
            X_pipeline = pd.DataFrame(X_pipeline_raw, columns=new_cols, index=X_train_original.index)
        except:
            X_pipeline = pd.DataFrame(X_pipeline_raw, index=X_train_original.index)

        # 🔹 Preparar respuesta JSON segura
        max_preview = 10
        response_data = {
            "x_train_original": df_to_json_safe(X_train_original, max_preview),
            "x_train_with_nan": df_to_json_safe(X_train, max_preview),
            "after_delete_nan": df_to_json_safe(X_after_delete, max_preview),
            "after_scaler": df_to_json_safe(X_after_scaler, max_preview),
            "after_onehot": df_to_json_safe(X_after_onehot, max_preview),
            "pipeline_final": df_to_json_safe(X_pipeline, max_preview),
            "split_info": {
                "train_size": len(train_set),
                "val_size": len(val_set),
                "test_size": len(test_set)
            }
        }

        # 🔹 Comprimir la respuesta con Gzip y Base64
        json_bytes = json.dumps(response_data).encode("utf-8")
        buf = io.BytesIO()
        with gzip.GzipFile(fileobj=buf, mode="wb") as f:
            f.write(json_bytes)
        compressed_data = base64.b64encode(buf.getvalue()).decode("utf-8")

        return JsonResponse({"compressed": compressed_data})

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)

@csrf_exempt
def evaluar_modelo(request):
    if request.method != "POST":
        return JsonResponse({"error": "Método no permitido"}, status=405)

    try:
        body = json.loads(request.body.decode("utf-8"))
        dataset_id = body.get("dataset_id")
    except Exception:
        return JsonResponse({"error": "JSON inválido"}, status=400)

    if not dataset_id:
        return JsonResponse({"error": "dataset_id requerido"}, status=400)

    # 1. Localizar el archivo en el sistema
    datasets_dir = os.path.join(settings.MEDIA_ROOT, "datasets")
    archivo = next((f for f in os.listdir(datasets_dir) if f.startswith(dataset_id)), None)
    
    if not archivo:
        return JsonResponse({"error": "Dataset no encontrado"}, status=404)

    data_path = os.path.join(datasets_dir, archivo)

    # 2. Cargar dataset
    try:
        df = load_kdd_dataset(data_path)
    except Exception as e:
        return JsonResponse({"error": f"Error al leer el archivo: {str(e)}"}, status=422)

    # Normalizar nombres de columnas
    df.columns = [str(c).strip().lower() for c in df.columns]

    if "class" not in df.columns:
        return JsonResponse({"error": 'El dataset debe contener la columna "class"'}, status=400)

    # 3. División de datos (Split)
    try:
        # Usando la función unificada que definiste previamente
        train_set, val_set, test_set = train_val_test_split(
            df,
            test_size=0.4,
            val_size=0.2,
            stratify="class"
        )
    except Exception as e:
        return JsonResponse({"error": f"Error en la división de datos: {str(e)}"}, status=500)

    # Definir X (características) y y (objetivo)
    X_train = train_set.drop("class", axis=1)
    y_train = train_set["class"]
    X_val = val_set.drop("class", axis=1)
    y_val = val_set["class"]
    X_test = test_set.drop("class", axis=1)
    y_test = test_set["class"]

    # 4. Pipeline de Preprocesamiento
    num_attribs = list(X_train.select_dtypes(exclude=["object"]).columns)
    cat_attribs = list(X_train.select_dtypes(include=["object"]).columns)

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler())
    ])

    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_attribs)
    ])

    # 5. Entrenamiento del Modelo
    try:
        # Fit y Transformación
        X_train_prep = full_pipeline.fit_transform(X_train)
        X_val_prep = full_pipeline.transform(X_val)
        X_test_prep = full_pipeline.transform(X_test)

        clf = LogisticRegression(max_iter=10000)
        clf.fit(X_train_prep, y_train)
    except Exception as e:
        return JsonResponse({"error": f"Error durante el entrenamiento: {str(e)}"}, status=500)

    # 6. Predicciones y Métricas
    y_val_pred = clf.predict(X_val_prep)
    y_test_pred = clf.predict(X_test_prep)

    acc_val = accuracy_score(y_val, y_val_pred)
    acc_test = accuracy_score(y_test, y_test_pred)
    acc_diff = abs(acc_val - acc_test)

    # Reporte de clasificación
    report_test = classification_report(
        y_test,
        y_test_pred,
        output_dict=True
    )

    # 7. Respuesta estructurada para el Front
    response = {
        "dataset_id": dataset_id,
        "model_info": {
            "algorithm": "LogisticRegression",
            "parameters": "max_iter=10000"
        },
        "split_sizes": {
            "train": int(len(train_set)),
            "validation": int(len(val_set)),
            "test": int(len(test_set))
        },
        "results": {
            "accuracy_validation": float(acc_val),
            "accuracy_test": float(acc_test),
            "accuracy_difference": float(acc_diff),
            "classification_report": report_test
        }
    }

    return JsonResponse(response)