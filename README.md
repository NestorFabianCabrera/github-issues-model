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

#### Parámetros en detalle

**`n_estimators=300`** — número de árboles

Construye 300 árboles de decisión independientes. Cada árbol vota una clase; gana la que más votos acumula. Valor elegido por ser estable sin ser excesivamente costoso:

```python
n_estimators=10   # rápido pero impreciso, resultado varía entre ejecuciones
n_estimators=100  # buen equilibrio velocidad/precisión
n_estimators=300  # ← usado, estable
n_estimators=1000 # mejora marginal, tarda ~3× más
```

Regla: más árboles = más estable, con rendimiento decreciente a partir de ~200.

---

**`max_depth=20`** — profundidad máxima de cada árbol

Limita cuántos niveles de preguntas puede encadenar cada árbol:

```
Nivel 1: ¿comments > 5?
Nivel 2:   ¿has_assignee = 1?
Nivel 3:     ¿title_len > 40?
... hasta nivel 20
```

```python
max_depth=3    # árbol muy simple → underfitting, no aprende suficiente
max_depth=10   # moderado
max_depth=20   # ← usado
max_depth=None # sin límite → overfitting (memoriza el training set, falla en test)
```

Overfitting con `None`: accuracy 99% en train, ~40% en test. El árbol memoriza un caso por hoja en vez de generalizar.

---

**`min_samples_leaf=2`** — mínimo de ejemplos por hoja

Una decisión final (hoja) necesita al menos 2 issues para existir. Evita que el árbol cree ramas para un único caso.

```python
min_samples_leaf=1   # puede crear hojas de 1 solo issue → overfitting
min_samples_leaf=2   # ← usado, protección mínima
min_samples_leaf=10  # hojas más generales → underfitting en datasets pequeños
```

---

**`class_weight='balanced'`** — el más importante

**El problema:** hay 2,577 issues `rapido` y solo 882 `lento`. Sin este parámetro el modelo aprende que equivocarse en `lento` cuesta poco (son pocos ejemplos) y empieza a ignorarla.

`'balanced'` calcula automáticamente un peso inverso a la frecuencia de cada clase:

```
peso_rapido = 5442 / (3 × 2577) = 0.70
peso_normal = 5442 / (3 × 1983) = 0.91
peso_lento  = 5442 / (3 × 882)  = 2.05  ← cada error en lento cuesta 3× más
```

```python
class_weight=None       # ignora desbalance → recall de lento ~10%
class_weight='balanced' # ← usado, fuerza al modelo a aprender todas las clases
```

---

**`random_state=42`** — semilla aleatoria

Random Forest usa aleatoriedad al construir los árboles. Sin fijar la semilla, cada ejecución daría métricas ligeramente distintas. El valor 42 es convención en la comunidad ML; cualquier número entero funciona igual.

```python
random_state=None  # resultados diferentes cada vez → no reproducible
random_state=42    # ← siempre el mismo resultado
```

---

**`n_jobs=-1`** — paralelismo

Usa todos los núcleos del CPU para entrenar los 300 árboles en paralelo.

```python
n_jobs=1    # un solo núcleo → hasta 4× más lento
n_jobs=-1   # ← todos los núcleos disponibles
```

---

### División del dataset

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

```
80% entrenamiento (4,353 issues) → el modelo aprende
20% test          (1,089 issues) → evaluación con datos nunca vistos
```

- `test_size=0.2` → reserva el 20% para test
- `random_state=42` → el corte es siempre el mismo (reproducible)
- `stratify=y` → garantiza que la proporción de clases sea idéntica en train y test. Sin esto podría ocurrir que el test tuviera el doble de `lento` que el train y las métricas serían engañosas.

## Métricas

Las métricas se calculan comparando las predicciones del modelo contra los valores reales del test set (los 1,089 issues que el modelo **nunca vio** durante el entrenamiento).

```python
y_pred = rf.predict(X_test_final)          # el modelo predice las 1,089 filas
acc    = accuracy_score(y_test, y_pred)    # % de aciertos totales
f1     = f1_score(y_test, y_pred, average='weighted')
```

| Métrica | Valor | Cómo se calcula |
|---------|-------|-----------------|
| Accuracy | 56.3% | predicciones correctas / total → 613/1089 |
| F1 weighted | 0.566 | promedio de F1 por clase, pesado por frecuencia |
| Baseline (mayoría) | 47.4% | % de la clase más frecuente (`rapido`) |
| Mejora sobre baseline | +9pp | 56.3% − 47.4% |

**Baseline:** un modelo idiota que siempre prediga `rapido` tendría 47.4% de accuracy sin aprender nada. Nuestro modelo supera ese umbral en +9 puntos, lo que demuestra que sí aprende patrones reales.

**F1 por clase** — generado por `classification_report(y_test, y_pred)`:

| Clase | Precision | Recall | F1 | Qué significa |
|-------|-----------|--------|----|---------------|
| rapido | 0.662 | 0.638 | 0.650 | La clase más fácil, tiene más ejemplos |
| normal | 0.543 | 0.509 | 0.525 | Intermedia |
| lento  | 0.373 | 0.466 | 0.414 | La más difícil, pocos ejemplos y texto similar a `normal` |

**Precision** — de los que el modelo dijo que eran `rapido`, ¿cuántos realmente lo eran?
```
precision_rapido = 329 / (329 + 136 + 32) = 329/497 = 0.662
```

**Recall** — de todos los que SON `rapido`, ¿cuántos detectó el modelo?
```
recall_rapido = 329 / (329 + 108 + 79) = 329/516 = 0.638
```

**F1** — media armónica de precision y recall. Castiga cuando uno de los dos es muy bajo:
```
F1 = 2 × (0.662 × 0.638) / (0.662 + 0.638) = 0.650
```

**Matriz de confusión** — generada por `confusion_matrix(y_test, y_pred)`:
```
              Predicho →
              rapido  normal  lento
Real rapido  [  329     108     79 ]   ← de 516 reales: 329 acertó
Real normal  [  136     202     59 ]   ← de 397 reales: 202 acertó
Real lento   [   32      62     82 ]   ← de 176 reales: solo 82 acertó
```
La diagonal (329, 202, 82) son los aciertos. Todo lo demás son errores.

**Por qué `lento` tiene el F1 más bajo (0.414):**
El texto de un issue que tarda 20 días es casi idéntico al de uno que tarda 10 días. El modelo no puede saber si el mantenedor está de vacaciones, si el repo tiene 2 contribuidores, o si hay un sprint activo. Esos factores determinan el tiempo pero no están en el texto.

**Por qué 56% es un resultado válido:**
El tiempo de resolución depende de factores externos que el texto no captura: disponibilidad del mantenedor, prioridad del sprint, tamaño del equipo, carga de trabajo. Con solo texto + metadata básica, superar el baseline en +9 puntos es el comportamiento esperado y es un resultado honesto.

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
