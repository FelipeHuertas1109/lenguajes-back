# 📊 Análisis Completo del Sistema AcepNet

## 🎯 Propósito General

**AcepNet** es un sistema de aprendizaje automático (deep learning) que resuelve el problema de **clasificación de cadenas en Autómatas Finitos Deterministas (AFDs)**. El modelo aprende a predecir si una cadena es aceptada o rechazada por un AFD específico sin necesidad de simular el autómata directamente.

---

## 🏗️ Arquitectura del Sistema

### 1. **Componentes Principales**

```
acepnet/
├── acepten.py              # Arquitectura del modelo y clases base
├── inferencia_mejorada.py  # Script de inferencia interactiva
├── best_model.pt          # Modelo entrenado (7.3MB)
├── dataset6000.csv        # Dataset con 6000 AFDs (9.6MB)
├── thresholds.json        # Umbrales calibrados (Y1=0.43, Y2=0.53)
└── README.md             # Documentación
```

---

## 📚 Dataset (`dataset6000.csv`)

### Estructura de Datos

El dataset contiene **6000 AFDs** con las siguientes columnas:

- **Regex**: Expresión regular original (ej: `[LCIG]+`, `[GDIK]*`)
- **Alfabeto**: Símbolos del alfabeto separados por espacio (ej: `C G I L`)
- **Estados de aceptación**: Estados finales (ej: `S3 S4 S2 S1`)
- **Estados**: Todos los estados del AFD (ej: `S0 S1 S2 S3 S4`)
- **Transiciones**: Transiciones en formato `S0 --A--> S1 | S1 --B--> S2`
- **Clase**: Diccionario JSON con 100 cadenas (50 aceptadas, 50 rechazadas) y su valor booleano
- **Error**: Campo para errores (vacío si no hay error)

### Ejemplo de Entrada:

```csv
Regex,Alfabeto,Estados de aceptación,Estados,Transiciones,Clase,Error
[LCIG]+,C G I L,S3 S4 S2 S1,S0 S1 S2 S3 S4,S3 --C--> S4 | S3 --G--> S1 | ...,"{'C': true, 'CC': true, ...}", 
```

---

## 🧠 Arquitectura del Modelo (`DualEncoderModel`)

### Visión General

El modelo implementa una arquitectura **Dual-Encoder** con **dos tareas simultáneas**:

1. **Y1**: Predicción de pertenencia (¿la cadena pertenece al AFD específico?)
2. **Y2**: Predicción de cadena compartida (¿la cadena es aceptada por múltiples AFDs?)

### Componentes del Modelo

#### 1. **String Encoder** (Encoder de Cadenas)

```
Cadena → Embedding → BiGRU → h_str
```

- **Embedding Layer**: Convierte índices de símbolos a vectores densos
  - Vocabulario: 12 símbolos (A-L) + 1 token de padding = 13 tokens
  - Dimensión de embedding: 32
  
- **BiGRU** (Bidirectional GRU):
  - 2 capas
  - Hidden dimension: 64
  - Bidireccional: output dimension = 64 × 2 = **128**
  - Dropout: 0.2 entre capas
  
**Manejo especial**: Las cadenas vacías (épsilon) se representan como un vector de ceros.

#### 2. **AFD Encoder** (Encoder de Autómatas)

```
Features del AFD → MLP → h_afd
```

**Representación del AFD**:
- **Matriz de transiciones one-hot**: `[16 estados × 12 símbolos × 16 estados]` = 3072 features
- **Vector de estados de aceptación**: 16 features (uno por estado)
- **Vector de estados válidos**: 16 features (máscara de estados existentes)
- **Total**: 3072 + 16 + 16 = **3104 features**

**MLP del AFD Encoder**:
```
3104 → Linear(512) → ReLU → Dropout(0.3)
     → Linear(256) → ReLU → Dropout(0.3)
     → Linear(128) → ReLU
     → h_afd (128 dimensiones)
```

#### 3. **Head 1: Pertenencia (Y1)**

**Input**: Concatenación de `h_str` + `h_afd` = 128 + 128 = **256 dimensiones**

```
256 → Linear(128) → ReLU → Dropout(0.3)
    → Linear(64) → ReLU
    → Linear(1) → Sigmoid
    → y1_hat (probabilidad)
```

#### 4. **Head 2: Cadena Compartida (Y2)**

**Input**: Solo `h_str` = **128 dimensiones**

```
128 → Linear(64) → ReLU → Dropout(0.2)
    → Linear(32) → ReLU
    → Linear(1) → Sigmoid
    → y2_hat (probabilidad)
```

### Parámetros del Modelo

- **Total de parámetros**: ~500,000-600,000 (estimado)
- **Tamaño del modelo**: 7.3MB (best_model.pt)

---

## 🔄 Pipeline de Entrenamiento (`acepten.py`)

### 1. **Parsing de AFDs** (`AFDParser`)

- **Función**: Extrae información estructurada de los AFDs del CSV
- **Métodos clave**:
  - `parse_states()`: Parsea estados (ej: `"S0 S1 S2"` → `[0, 1, 2]`)
  - `parse_accept_states()`: Parsea estados de aceptación
  - `parse_transitions()`: Parsea transiciones usando regex
  - `get_afd_features()`: Convierte AFD a vector de 3104 features
  - `simulate_afd()`: Simula el AFD para verificar aceptación (ground truth)

### 2. **Generación de Dataset** (`StringDatasetGenerator`)

- **Función**: Genera pares `(dfa_id, string, label)` para entrenamiento

**Muestras Positivas**:
- Extrae cadenas aceptadas de la columna `Clase` del CSV
- Por defecto: 50 cadenas aceptadas por AFD

**Muestras Negativas**:
- Genera cadenas aleatorias usando el alfabeto del AFD
- Verifica que NO sean aceptadas (simulación del AFD)
- Por defecto: 50 cadenas rechazadas por AFD

**Cálculo de Y2**:
- Cuenta cuántos AFDs distintos aceptan cada cadena
- Si una cadena es aceptada por ≥2 AFDs → `y2=1`, sino `y2=0`

### 3. **Dataset de PyTorch** (`AFDStringDataset`)

- **Función**: Dataset personalizado para PyTorch
- **Tokenización**:
  - Mapeo: `'A' → 0, 'B' → 1, ..., 'L' → 11`
  - Padding: `PAD_IDX = 12`
  - Cadenas vacías: `[]`

### 4. **Collate Function**

- **Función**: Maneja secuencias de longitud variable en batches
- **Padding**: Rellena cadenas más cortas con `PAD_IDX` hasta la longitud máxima del batch

### 5. **Entrenamiento** (`Trainer`)

**Loss Function**:
```
Loss = λ1 * BCE(y1_hat, y1_true) + λ2 * BCE(y2_hat, y2_true)
```
- Por defecto: `λ1 = 1.0`, `λ2 = 1.0`

**Optimizador**:
- Adam con learning rate: 0.001
- Weight decay: 1e-5
- Gradiente clipping: max_norm = 5.0

**Scheduler**:
- ReduceLROnPlateau (reduce LR si no mejora en 3 épocas, factor=0.5)

**División del Dataset**:
- Por **dfa_id** (no por ejemplo individual)
- Train: 70% de AFDs
- Val: 15% de AFDs
- Test: 15% de AFDs

**Métricas**:
- Accuracy para Y1 y Y2
- Loss promedio

---

## 🎯 Inferencia (`inferencia_mejorada.py`)

### Clase `Predictor`

**Funcionalidad**:
- Carga el modelo entrenado (`best_model.pt`)
- Carga umbrales calibrados (`thresholds.json`)
- Realiza predicciones para pares `(dfa_id, string)`

**Validación de Alfabeto**:
- Si la cadena contiene símbolos fuera del alfabeto del AFD:
  - `y1_prob = 0.0` (rechazo automático)
  - `y2_prob = 0.0`
  - No se ejecuta el modelo

**Umbrales Calibrados**:
- **Y1**: 0.43 (en lugar de 0.5 estándar)
- **Y2**: 0.53 (en lugar de 0.5 estándar)
- Estos umbrales fueron optimizados para mejor precisión

### Modo Interactivo

El script ofrece 5 opciones:

1. **Probar cadena con AFD (por ID)**: Ingresa ID y cadena, muestra predicción vs ground truth
2. **Buscar AFD por palabra clave**: Busca AFDs cuya regex contenga una palabra
3. **Ver información de AFD**: Muestra detalles sin predicción
4. **Ejemplos predefinidos**: Ejecuta casos de prueba automáticos
5. **Salir**: Cierra el programa

---

## 📊 Métricas y Evaluación

### Métricas Principales

**Para Y1 (Pertenencia)**:
- **Accuracy**: Porcentaje de predicciones correctas
- **F1 Score**: Balance entre precisión y recall
- **Clasificación de rendimiento**:
  - ✅ MUY BUENO: Accuracy ≥ 0.95 y F1 ≥ 0.95
  - ✔️ BUENO: Accuracy ≥ 0.90 y F1 ≥ 0.90
  - ⚠️ REGULAR: Accuracy ≥ 0.85
  - ❌ MALO: Accuracy < 0.85

**Para Y2 (Cadena Compartida)**:
- **Accuracy**: Porcentaje de predicciones correctas
- **F1 Score**: Balance entre precisión y recall
- **PR-AUC** (Precision-Recall Area Under Curve): Área bajo la curva PR

---

## 🔧 Características Técnicas

### 1. **Constantes Globales**

```python
ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L']
MAX_STATES = 16
NUM_SYMBOLS = 12
PAD_IDX = 12
```

### 2. **Manejo de Cadenas Vacías (Épsilon)**

- Las cadenas vacías se representan como `[]` (lista vacía)
- En el modelo, se manejan especialmente: se produce un vector de ceros para `h_str`
- Permite al modelo aprender que algunas cadenas vacías son aceptadas

### 3. **Cache de Features de AFD**

- `AFDParser` mantiene un cache (`afd_cache`) para evitar recalcular features
- Las features se calculan una vez por AFD y se reutilizan

### 4. **Padding Dinámico**

- El padding se hace por batch (no global)
- Cada batch solo paddea hasta la longitud máxima dentro de ese batch
- Más eficiente en memoria

---

## 🎓 Ventajas del Sistema

1. **Aprendizaje Automático**: No requiere simular el AFD para predecir aceptación
2. **Velocidad**: Predicciones rápidas una vez entrenado el modelo
3. **Escalabilidad**: Puede generalizar a AFDs no vistos durante entrenamiento
4. **Multi-tarea**: Aprende dos tareas relacionadas simultáneamente (Y1 y Y2)
5. **Validación Inteligente**: Rechaza automáticamente cadenas con símbolos inválidos

---

## ⚠️ Limitaciones

1. **Alfabeto Fijo**: Solo funciona con símbolos A-L (12 símbolos)
2. **Estados Máximos**: Limitado a 16 estados por AFD
3. **Dependencia del Dataset**: Requiere un dataset de AFDs pre-generado
4. **Umbrales Calibrados**: Los umbrales fueron ajustados para este dataset específico
5. **Modelo Específico**: El modelo está entrenado para este dominio específico

---

## 🔄 Flujo de Datos Completo

```
1. Dataset CSV (6000 AFDs)
   ↓
2. AFDParser → Extrae features (3104 dim)
   ↓
3. StringDatasetGenerator → Genera pares (dfa_id, string, y1, y2)
   ↓
4. AFDStringDataset → Tokeniza cadenas
   ↓
5. DataLoader → Batching con padding
   ↓
6. DualEncoderModel:
   - String → Embedding → BiGRU → h_str (128)
   - AFD → MLP → h_afd (128)
   ↓
7. Head 1: concat(h_str, h_afd) → Y1 (pertenencia)
8. Head 2: h_str → Y2 (compartida)
   ↓
9. Loss → Backpropagation → Optimización
```

---

## 🚀 Casos de Uso

1. **Clasificación Rápida**: Determinar si una cadena pertenece a un AFD sin simular
2. **Búsqueda de Patrones**: Identificar cadenas compartidas entre múltiples AFDs
3. **Validación de Alfabetos**: Detectar automáticamente símbolos inválidos
4. **Investigación**: Estudiar la capacidad de las redes neuronales para aprender lenguajes regulares

---

## 📈 Mejoras Futuras Potenciales

1. **Alfabeto Dinámico**: Soporte para alfabetos de cualquier tamaño
2. **Más Estados**: Aumentar el límite de estados (actualmente 16)
3. **Atención**: Agregar mecanismos de atención para mejor interpretabilidad
4. **Transfer Learning**: Pre-entrenar en un dataset más grande y fine-tune en dominios específicos
5. **Interpretabilidad**: Visualizar qué partes de la cadena y del AFD son más importantes para la decisión

---

## 🔍 Resumen Ejecutivo

**AcepNet** es un sistema de deep learning que aprende a clasificar cadenas en AFDs usando una arquitectura dual-encoder. El modelo procesa tanto la cadena de entrada (usando BiGRU) como la representación del AFD (usando MLP), y produce dos predicciones: pertenencia a un AFD específico (Y1) y si la cadena es compartida entre múltiples AFDs (Y2). El sistema está entrenado en 6000 AFDs y puede hacer predicciones rápidas sin necesidad de simular el autómata.

