# Predicción de Tiempo de Resolución de Issues de GitHub

Modelo de clasificación que predice cuánto tardará en resolverse un issue de GitHub, basándose en el texto del título, la descripción y metadatos del issue.

## Problema

Dado un issue recién abierto, el modelo predice a cuál de estas 3 categorías pertenece:

| Clase | Tiempo de resolución |
|-------|---------------------|
| `rapido` | Menos de 1 día |
| `normal` | Entre 1 y 14 días |
| `lento` | Más de 14 días |

## Dataset

5,442 issues reales extraídos de la API de GitHub de 37 repositorios populares:
`microsoft/vscode`, `golang/go`, `angular/angular`, `symfony/symfony`, `vuejs/vue`, `pallets/flask`, `docker/compose`, `grafana/grafana`, entre otros.

Campos extraídos por issue:
- `title` — título del issue
- `body` — descripción
- `labels` — etiquetas asignadas
- `created_at` / `closed_at` — fechas para calcular tiempo de resolución
- `comments` — número de comentarios
- `has_assignee` — si tiene persona asignada (0/1)
- `has_milestone` — si pertenece a un milestone (0/1)
- `day_of_week` / `hour_of_day` — día y hora de creación

El target se calcula como:
```
resolution_days = (closed_at - created_at).total_seconds() / 86400
```

## Arquitectura del modelo

### Entrada: dos tipos de features

**Texto** — título (repetido ×2 para darle más peso) + body del issue:
- Se procesa con **TF-IDF** (`TfidfVectorizer`)
- 8,000 términos máximo
- Unigramas y bigramas (`ngram_range=(1,2)`)
- `sublinear_tf=True` — usa log(frecuencia) para reducir el peso de términos muy repetidos
- `stop_words='english'` — elimina palabras vacías (the, is, a, etc.)
- `min_df=3` — ignora términos que aparecen en menos de 3 issues

**Numéricas** — 12 features estructuradas:
```
has_assignee, has_milestone, comments, day_of_week, hour_of_day,
title_word_count, body_word_count, has_code_block, has_url,
has_question, label_count, title_len
```
Se normalizan con `StandardScaler` (media 0, desviación estándar 1).

Ambas matrices se combinan con `hstack()` en una sola entrada sparse.

### Modelo: Random Forest

```python
RandomForestClassifier(
    n_estimators=300,        # 300 árboles de decisión
    max_depth=20,
    min_samples_leaf=2,
    class_weight='balanced', # compensa desbalance de clases
    random_state=42,
    n_jobs=-1,
)
```

**Por qué Random Forest:**
- Funciona bien con features mixtas (texto + numéricas)
- Resistente a overfitting al promediar 300 árboles independientes
- `class_weight='balanced'` compensa que hay el doble de issues `rapido` que `lento`
- Produce importancia de features interpretable

### División del dataset

```
80% entrenamiento (4,353 issues) → el modelo aprende
20% test          (1,089 issues) → evaluación con datos nunca vistos
```

`stratify=y` garantiza que la proporción de clases se mantenga en ambos conjuntos.

## Métricas

| Métrica | Valor |
|---------|-------|
| Accuracy | 56.3% |
| F1 weighted | 0.566 |
| Baseline (mayoría) | 47.4% |
| Mejora sobre baseline | +9 puntos porcentuales |

**F1 por clase:**

| Clase | Precision | Recall | F1 |
|-------|-----------|--------|----|
| rapido | 0.662 | 0.638 | 0.650 |
| normal | 0.543 | 0.509 | 0.525 |
| lento  | 0.373 | 0.466 | 0.414 |

**Por qué 56% es un resultado válido:**
El tiempo de resolución de un issue depende de factores que el texto no puede capturar: disponibilidad del mantenedor, prioridad del sprint, tamaño del equipo, carga de trabajo. Con solo texto + metadata básica, superar el baseline en +9 puntos es el comportamiento esperado.

## Estructura del proyecto

```
├── github_issues_model.ipynb   # Notebook completo: EDA → entrenamiento → evaluación → demo
├── presentacion.html           # Presentación de diapositivas del proyecto
├── scripts/
│   ├── fetch_issues.py         # Extrae issues de GitHub API (batch 1)
│   ├── fetch_issues_more.py    # Extrae issues de GitHub API (batch 2)
│   ├── fetch_issues_final.py   # Extrae issues de GitHub API (batch 3)
│   ├── train_model.py          # Entrenamiento v1 (4 clases)
│   └── train_model_v2.py       # Entrenamiento v2 (3 clases) — versión final
└── data/
    ├── eda_stats.json           # Estadísticas del análisis exploratorio
    ├── model_metrics_v2.json    # Métricas del modelo final
    └── model_metrics.json       # Métricas del modelo v1
```

> Los archivos `.csv` y `.pkl` no se incluyen en el repo por tamaño.  
> Para regenerar el dataset ejecutar los scripts de `scripts/fetch_*.py`.

## Cómo ejecutar

### Requisitos
- Docker (sin necesidad de instalar Python localmente)

### Levantar Jupyter

```bash
docker run --rm -p 8888:8888 \
  -v $(pwd):/home/jovyan/work \
  jupyter/scipy-notebook:latest \
  jupyter notebook --ip=0.0.0.0 --no-browser --NotebookApp.token=''
```

Abrir `http://localhost:8888` → navegar a `work/` → abrir `github_issues_model.ipynb`.

### Ejecutar entrenamiento directo

```bash
docker run --rm \
  -e HOME=/tmp \
  -v $(pwd)/scripts:/scripts:ro \
  -v $(pwd)/data:/data \
  python:3.12-slim bash -c "pip install -q pandas numpy scikit-learn scipy && python /scripts/train_model_v2.py"
```

## Flujo completo

```
GitHub API
    ↓
CSV raw (título, body, fechas, labels, metadata)
    ↓
Calcular días → asignar clase (rapido / normal / lento)
    ↓
    ├── texto → TF-IDF (8,000 features)
    └── metadata → StandardScaler (12 features)
                        ↓
                    hstack() — matriz sparse combinada
                        ↓
               Random Forest (300 árboles)
                        ↓
          Train 80%  |  Test 20%
                        ↓
                    Métricas
                        ↓
             modelo guardado (model_artifacts_v2.pkl)
                        ↓
             issue nuevo → predicción + probabilidad
```

## Stack

- Python 3.12
- pandas, numpy
- scikit-learn
- scipy (matrices sparse)
- GitHub REST API v3
