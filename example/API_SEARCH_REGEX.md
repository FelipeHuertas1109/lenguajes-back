# 🔍 API Endpoint: Search Regex

Endpoint de búsqueda para consultar el dataset de AcepNet (`dataset6000.csv`) que contiene 6000 expresiones regulares.

## 📋 Información General

**URL:** `/api/search-regex/`

**Métodos:** `GET`, `POST`

**Descripción:** Busca expresiones regulares en el dataset de AcepNet. Soporta búsqueda por texto (query) o por ID específico.

---

## 🎯 Parámetros

### GET Request

| Parámetro | Tipo | Requerido | Descripción |
|-----------|------|-----------|-------------|
| `query` o `q` | string | No* | Palabra clave para buscar en las regex (case-insensitive) |
| `id` | integer | No* | ID específico del regex (0-5999) |
| `limit` | integer | No | Límite de resultados a devolver (1-1000, por defecto: 50) |

\* Al menos uno de `query`/`q` o `id` debe estar presente.

### POST Request

| Parámetro | Tipo | Requerido | Descripción |
|-----------|------|-----------|-------------|
| `query` o `q` | string | No* | Palabra clave para buscar en las regex (case-insensitive) |
| `id` | integer | No* | ID específico del regex (0-5999) |
| `limit` | integer | No | Límite de resultados a devolver (1-1000, por defecto: 50) |

\* Al menos uno de `query`/`q` o `id` debe estar presente.

---

## 📤 Respuesta

### Formato de Respuesta

```json
{
  "success": true/false,
  "query": "palabra_clave" o null,
  "id": <numero> o null,
  "results": [
    {
      "id": 0,
      "regex": "[LCIG]+"
    },
    {
      "id": 1,
      "regex": "[GDIK]*"
    },
    ...
  ],
  "total": 10,
  "limit": 50,
  "error": null o "mensaje de error"
}
```

### Campos de Respuesta

| Campo | Tipo | Descripción |
|-------|------|-------------|
| `success` | boolean | Indica si la petición fue exitosa |
| `query` | string\|null | La query de búsqueda usada (null si se buscó por ID) |
| `id` | integer\|null | El ID usado para la búsqueda (null si se buscó por query) |
| `results` | array | Lista de resultados encontrados (cada uno con `id` y `regex`) |
| `total` | integer | Total de resultados encontrados (antes de aplicar el límite) |
| `limit` | integer | Límite de resultados aplicado |
| `error` | string\|null | Mensaje de error si hubo algún problema |

---

## 💡 Ejemplos de Uso

### 1. Búsqueda por ID (GET)

Buscar un regex específico por su ID:

```bash
# Buscar el regex con ID 0
curl "http://localhost:8000/api/search-regex/?id=0"
```

**Respuesta:**
```json
{
  "success": true,
  "query": null,
  "id": 0,
  "results": [
    {
      "id": 0,
      "regex": "[LCIG]+"
    }
  ],
  "total": 1,
  "limit": 50,
  "error": null
}
```

### 2. Búsqueda por Texto (GET)

Buscar regex que contengan una palabra clave:

```bash
# Buscar regex que contengan "LCIG"
curl "http://localhost:8000/api/search-regex/?query=LCIG"
```

**Respuesta:**
```json
{
  "success": true,
  "query": "LCIG",
  "id": null,
  "results": [
    {
      "id": 0,
      "regex": "[LCIG]+"
    },
    {
      "id": 19,
      "regex": "[LCKH]+"
    }
  ],
  "total": 25,
  "limit": 50,
  "error": null
}
```

### 3. Búsqueda con Límite (GET)

Limitar el número de resultados devueltos:

```bash
# Buscar "LCIG" con máximo 10 resultados
curl "http://localhost:8000/api/search-regex/?query=LCIG&limit=10"
```

### 4. Búsqueda usando 'q' (GET)

Usar el parámetro abreviado `q` en lugar de `query`:

```bash
curl "http://localhost:8000/api/search-regex/?q=GDIK"
```

### 5. Búsqueda por ID (POST)

```bash
curl -X POST http://localhost:8000/api/search-regex/ \
  -H "Content-Type: application/json" \
  -d '{"id": 42}'
```

**Respuesta:**
```json
{
  "success": true,
  "query": null,
  "id": 42,
  "results": [
    {
      "id": 42,
      "regex": "[JCDLB]*"
    }
  ],
  "total": 1,
  "limit": 50,
  "error": null
}
```

### 6. Búsqueda por Texto (POST)

```bash
curl -X POST http://localhost:8000/api/search-regex/ \
  -H "Content-Type: application/json" \
  -d '{
    "query": "LCIG",
    "limit": 20
  }'
```

### 7. Búsqueda Combinada (POST)

Si se proporcionan ambos parámetros, el `id` tiene prioridad:

```bash
curl -X POST http://localhost:8000/api/search-regex/ \
  -H "Content-Type: application/json" \
  -d '{
    "id": 0,
    "query": "LCIG"
  }'
```

En este caso, se buscará **solo por ID** (el query será ignorado).

---

## ⚠️ Casos de Error

### 1. Falta Parámetro Requerido

**Request:**
```bash
curl "http://localhost:8000/api/search-regex/"
```

**Respuesta (400 Bad Request):**
```json
{
  "success": false,
  "query": null,
  "id": null,
  "results": [],
  "total": 0,
  "limit": 50,
  "error": "Se requiere al menos uno de: 'query'/'q' o 'id'"
}
```

### 2. ID Fuera de Rango

**Request:**
```bash
curl "http://localhost:8000/api/search-regex/?id=9999"
```

**Respuesta (400 Bad Request):**
```json
{
  "success": false,
  "query": null,
  "id": 9999,
  "results": [],
  "total": 0,
  "limit": 50,
  "error": "ID fuera de rango. Debe estar entre 0 y 5999"
}
```

### 3. JSON Inválido (POST)

**Request:**
```bash
curl -X POST http://localhost:8000/api/search-regex/ \
  -H "Content-Type: application/json" \
  -d 'invalid json'
```

**Respuesta (400 Bad Request):**
```json
{
  "success": false,
  "query": null,
  "id": null,
  "results": [],
  "total": 0,
  "limit": 50,
  "error": "JSON inválido en el cuerpo de la petición"
}
```

### 4. Archivo CSV No Encontrado

**Respuesta (404 Not Found):**
```json
{
  "success": false,
  "query": "LCIG",
  "id": null,
  "results": [],
  "total": 0,
  "limit": 50,
  "error": "Archivo CSV no encontrado: /ruta/al/dataset6000.csv"
}
```

---

## 🚀 Características

### 1. **Cache de Datos**
- El CSV se carga una vez y se cachea en memoria
- Evita releer el archivo en cada petición
- Mejora significativamente el rendimiento

### 2. **Búsqueda Case-Insensitive**
- La búsqueda por texto no distingue entre mayúsculas y minúsculas
- `"LCIG"`, `"lcig"`, `"LcIg"` encontrarán los mismos resultados

### 3. **Búsqueda por Subcadena**
- Busca cualquier regex que **contenga** la palabra clave
- No requiere coincidencia exacta

### 4. **Límite Configurable**
- Por defecto devuelve máximo 50 resultados
- Se puede ajustar entre 1 y 1000
- Reduce el tamaño de la respuesta para queries muy generales

### 5. **Prioridad de Búsqueda**
- Si se proporcionan ambos `id` y `query`, el `id` tiene prioridad
- Útil para garantizar búsqueda específica cuando ambos están presentes

---

## 📊 Ejemplos Prácticos

### Buscar Todas las Regex que Contengan Kleene Star

```bash
curl "http://localhost:8000/api/search-regex/?query=*&limit=100"
```

### Buscar Regex Específica por ID

```bash
# Obtener el regex con ID 0
curl "http://localhost:8000/api/search-regex/?id=0"
```

### Buscar Regex con Clases de Caracteres

```bash
# Buscar regex que usen clases de caracteres [A-Z]
curl "http://localhost:8000/api/search-regex/?query=["
```

### Buscar Regex con Alternancia

```bash
# Buscar regex que usen el operador |
curl "http://localhost:8000/api/search-regex/?query=|"
```

---

## 🔧 Detalles Técnicos

### Dataset

- **Archivo:** `models/acepnet/dataset6000.csv`
- **Total de Regex:** 6000 (IDs de 0 a 5999)
- **Columnas:** Regex, Alfabeto, Estados de aceptación, Estados, Transiciones, Clase, Error

### Cache

- **Variable global:** `_csv_cache`
- **Inicialización:** Lazy loading (se carga en la primera petición)
- **Invalidación:** Solo si cambia la ruta del archivo

### Rendimiento

- **Búsqueda por ID:** O(1) - Acceso directo al DataFrame
- **Búsqueda por texto:** O(n) - Escaneo lineal con pandas
- **Cache:** Evita I/O en cada petición

---

## 📝 Notas

1. **Rango de IDs:** Los IDs válidos son de 0 a 5999 (inclusive)
2. **Límite por defecto:** Si no se especifica, se devuelven máximo 50 resultados
3. **Búsqueda parcial:** La búsqueda por texto busca subcadenas, no coincidencias exactas
4. **Case-insensitive:** Todas las búsquedas son insensibles a mayúsculas/minúsculas
5. **Cache persistente:** El cache permanece en memoria durante toda la ejecución del servidor

---

## 🔗 Endpoints Relacionados

- [`/api/regex-to-dfa/`](./API_REGEX_TO_DFA.md) - Convertir regex a DFA
- [`/api/regex-to-alphabet/`](./API_REGEX_TO_ALPHABET.md) - Predecir alfabeto de una regex
- [`/api/transitions-to-dfa/`](./API_TRANSITIONS_TO_DFA.md) - Construir DFA desde transiciones

---

## 📄 Ejemplo Completo en JavaScript

```javascript
// Búsqueda por ID
async function buscarPorId(id) {
  const response = await fetch(`http://localhost:8000/api/search-regex/?id=${id}`);
  const data = await response.json();
  
  if (data.success) {
    console.log(`Regex con ID ${id}:`, data.results[0].regex);
    return data.results[0];
  } else {
    console.error('Error:', data.error);
    return null;
  }
}

// Búsqueda por texto
async function buscarPorTexto(query, limit = 50) {
  const response = await fetch(
    `http://localhost:8000/api/search-regex/?query=${encodeURIComponent(query)}&limit=${limit}`
  );
  const data = await response.json();
  
  if (data.success) {
    console.log(`Encontrados ${data.total} resultados`);
    return data.results;
  } else {
    console.error('Error:', data.error);
    return [];
  }
}

// Uso
buscarPorId(0);  // Buscar regex con ID 0
buscarPorTexto("LCIG", 10);  // Buscar regex que contengan "LCIG", máximo 10 resultados
```

---

## 📄 Ejemplo Completo en Python

```python
import requests

def buscar_por_id(id_regex):
    """Busca un regex por su ID"""
    response = requests.get(f"http://localhost:8000/api/search-regex/?id={id_regex}")
    data = response.json()
    
    if data["success"]:
        return data["results"][0] if data["results"] else None
    else:
        print(f"Error: {data['error']}")
        return None

def buscar_por_texto(query, limit=50):
    """Busca regex que contengan el texto especificado"""
    response = requests.get(
        "http://localhost:8000/api/search-regex/",
        params={"query": query, "limit": limit}
    )
    data = response.json()
    
    if data["success"]:
        return data["results"]
    else:
        print(f"Error: {data['error']}")
        return []

# Uso
regex_0 = buscar_por_id(0)
print(f"Regex con ID 0: {regex_0['regex']}")

resultados = buscar_por_texto("LCIG", limit=10)
for r in resultados:
    print(f"ID {r['id']}: {r['regex']}")
```

---

## 🎓 Casos de Uso

1. **Exploración del Dataset**: Buscar regex por patrones comunes
2. **Validación de IDs**: Verificar si un ID existe antes de usarlo
3. **Análisis de Patrones**: Encontrar regex que usen operadores específicos
4. **Integración con AcepNet**: Obtener regex del dataset para pruebas con el modelo
5. **Generación de Reportes**: Listar todas las regex que coinciden con un criterio

---

## 📚 Referencias

- [Dataset AcepNet](./../models/acepnet/README.md)
- [Análisis Completo del Sistema AcepNet](./../models/acepnet/ANALISIS_COMPLETO.md)

---

# 🤖 API Endpoint: AcepNet Predict

Endpoint que usa el modelo AcepNet para predecir si una cadena es aceptada por un AFD (Autómata Finito Determinista).

## 📋 Información General

**URL:** `/api/acepnet-predict/`

**Métodos:** `GET`, `POST`

**Descripción:** Utiliza el modelo de deep learning AcepNet para predecir si una o múltiples cadenas son aceptadas por un AFD específico del dataset. El modelo proporciona dos predicciones:
- **Y1**: Pertenencia a AFD específico (predice si la cadena es aceptada por el AFD)
- **Y2**: Cadena compartida (predice si la cadena es compartida entre múltiples AFDs)

El endpoint también incluye la simulación real del AFD (ground truth) para comparar la precisión del modelo.

---

## 🎯 Parámetros

### GET Request

| Parámetro | Tipo | Requerido | Descripción |
|-----------|------|-----------|-------------|
| `dfa_id` o `id` | integer | Sí | ID del AFD en el dataset (0-5999) |
| `string` | string | No* | Cadena única a evaluar |
| `strings` | string | No* | Cadenas separadas por comas para evaluar múltiples |

\* Al menos uno de `string` o `strings` debe estar presente.

**Nota:** También puedes pasar múltiples parámetros `string` en la URL: `?string=a&string=b&string=c`

### POST Request

| Parámetro | Tipo | Requerido | Descripción |
|-----------|------|-----------|-------------|
| `dfa_id` o `id` | integer | Sí | ID del AFD en el dataset (0-5999) |
| `string` | string | No* | Cadena única a evaluar |
| `strings` | string[] o string | No* | Array de cadenas o cadena separada por comas |

\* Al menos uno de `string` o `strings` debe estar presente.

---

## 📤 Respuesta

### Formato de Respuesta

```json
{
  "success": true/false,
  "dfa_id": <numero>,
  "afd_info": {
    "id": 0,
    "regex": "[LCIG]+",
    "alphabet": "L C I G",
    "states": "4",
    "accepting": "S3"
  },
  "predictions": [
    {
      "string": "C",
      "y1": {
        "probability": 0.95,
        "predicted": true,
        "ground_truth": true,
        "correct": true,
        "alphabet_mismatch": false
      },
      "y2": {
        "probability": 0.42,
        "predicted": false
      }
    },
    ...
  ],
  "error": null o "mensaje de error"
}
```

### Campos de Respuesta

| Campo | Tipo | Descripción |
|-------|------|-------------|
| `success` | boolean | Indica si la petición fue exitosa |
| `dfa_id` | integer | ID del AFD usado para la predicción |
| `afd_info` | object | Información del AFD (regex, alfabeto, estados, aceptación) |
| `predictions` | array | Lista de predicciones (una por cada cadena evaluada) |
| `error` | string\|null | Mensaje de error si hubo algún problema |

#### Campos de `afd_info`

| Campo | Tipo | Descripción |
|-------|------|-------------|
| `id` | integer | ID del AFD |
| `regex` | string | Expresión regular que define el AFD |
| `alphabet` | string | Alfabeto del AFD (caracteres separados por espacios) |
| `states` | string | Número de estados del AFD |
| `accepting` | string | Estados de aceptación del AFD |

#### Campos de `predictions[].y1` (Pertenencia a AFD)

| Campo | Tipo | Descripción |
|-------|------|-------------|
| `probability` | float | Probabilidad de pertenencia (0.0 - 1.0) |
| `predicted` | boolean | Predicción binaria (true = aceptada, false = rechazada) |
| `ground_truth` | boolean\|null | Resultado real de la simulación del AFD (null si hay error) |
| `correct` | boolean\|null | Si la predicción fue correcta comparada con ground truth (null si no hay ground truth) |
| `alphabet_mismatch` | boolean | Si la cadena contiene caracteres que no están en el alfabeto del AFD |

#### Campos de `predictions[].y2` (Cadena Compartida)

| Campo | Tipo | Descripción |
|-------|------|-------------|
| `probability` | float | Probabilidad de que sea cadena compartida (0.0 - 1.0) |
| `predicted` | boolean | Predicción binaria (true = compartida, false = no compartida) |

---

## 💡 Ejemplos de Uso

### 1. Predicción con una Cadena (GET)

Predecir si una cadena es aceptada por un AFD:

```bash
# Predecir si "C" es aceptada por el AFD con ID 0
curl "http://localhost:8000/api/acepnet-predict/?dfa_id=0&string=C"
```

**Respuesta:**
```json
{
  "success": true,
  "dfa_id": 0,
  "afd_info": {
    "id": 0,
    "regex": "[LCIG]+",
    "alphabet": "L C I G",
    "states": "4",
    "accepting": "S3"
  },
  "predictions": [
    {
      "string": "C",
      "y1": {
        "probability": 0.9523,
        "predicted": true,
        "ground_truth": true,
        "correct": true,
        "alphabet_mismatch": false
      },
      "y2": {
        "probability": 0.4215,
        "predicted": false
      }
    }
  ],
  "error": null
}
```

### 2. Predicción con Múltiples Cadenas (GET)

Evaluar múltiples cadenas en una sola petición:

```bash
# Usando múltiples parámetros 'string'
curl "http://localhost:8000/api/acepnet-predict/?dfa_id=0&string=C&string=LC&string=LCIG"

# O usando parámetro 'strings' separado por comas
curl "http://localhost:8000/api/acepnet-predict/?dfa_id=0&strings=C,LC,LCIG"
```

**Respuesta:**
```json
{
  "success": true,
  "dfa_id": 0,
  "afd_info": {
    "id": 0,
    "regex": "[LCIG]+",
    "alphabet": "L C I G",
    "states": "4",
    "accepting": "S3"
  },
  "predictions": [
    {
      "string": "C",
      "y1": {
        "probability": 0.9523,
        "predicted": true,
        "ground_truth": true,
        "correct": true,
        "alphabet_mismatch": false
      },
      "y2": {
        "probability": 0.4215,
        "predicted": false
      }
    },
    {
      "string": "LC",
      "y1": {
        "probability": 0.9876,
        "predicted": true,
        "ground_truth": true,
        "correct": true,
        "alphabet_mismatch": false
      },
      "y2": {
        "probability": 0.5123,
        "predicted": true
      }
    },
    {
      "string": "LCIG",
      "y1": {
        "probability": 0.9987,
        "predicted": true,
        "ground_truth": true,
        "correct": true,
        "alphabet_mismatch": false
      },
      "y2": {
        "probability": 0.6234,
        "predicted": true
      }
    }
  ],
  "error": null
}
```

### 3. Predicción con una Cadena (POST)

```bash
curl -X POST http://localhost:8000/api/acepnet-predict/ \
  -H "Content-Type: application/json" \
  -d '{
    "dfa_id": 0,
    "string": "C"
  }'
```

### 4. Predicción con Múltiples Cadenas (POST)

```bash
curl -X POST http://localhost:8000/api/acepnet-predict/ \
  -H "Content-Type: application/json" \
  -d '{
    "dfa_id": 0,
    "strings": ["C", "LC", "LCIG"]
  }'
```

**Respuesta:** Igual que en el ejemplo 2.

### 5. Usar 'id' en lugar de 'dfa_id'

```bash
curl "http://localhost:8000/api/acepnet-predict/?id=0&string=C"
```

### 6. Cadena Vacía o Épsilon

El modelo también puede evaluar cadenas vacías o épsilon:

```bash
# Cadena vacía
curl "http://localhost:8000/api/acepnet-predict/?dfa_id=0&string="

# Épsilon explícito
curl "http://localhost:8000/api/acepnet-predict/?dfa_id=0&string=<EPS>"
```

---

## ⚠️ Casos de Error

### 1. Falta Parámetro `dfa_id`

**Request:**
```bash
curl "http://localhost:8000/api/acepnet-predict/?string=C"
```

**Respuesta (400 Bad Request):**
```json
{
  "success": false,
  "dfa_id": null,
  "afd_info": null,
  "predictions": [],
  "error": "Parámetro 'dfa_id' o 'id' es requerido"
}
```

### 2. Falta Parámetro `string` o `strings`

**Request:**
```bash
curl "http://localhost:8000/api/acepnet-predict/?dfa_id=0"
```

**Respuesta (400 Bad Request):**
```json
{
  "success": false,
  "dfa_id": 0,
  "afd_info": null,
  "predictions": [],
  "error": "Parámetro 'string' o 'strings' es requerido"
}
```

### 3. ID Fuera de Rango

**Request:**
```bash
curl "http://localhost:8000/api/acepnet-predict/?dfa_id=9999&string=C"
```

**Respuesta (400 Bad Request):**
```json
{
  "success": false,
  "dfa_id": 9999,
  "afd_info": null,
  "predictions": [],
  "error": "dfa_id debe estar entre 0 y 5999"
}
```

### 4. JSON Inválido (POST)

**Request:**
```bash
curl -X POST http://localhost:8000/api/acepnet-predict/ \
  -H "Content-Type: application/json" \
  -d 'invalid json'
```

**Respuesta (400 Bad Request):**
```json
{
  "success": false,
  "dfa_id": null,
  "afd_info": null,
  "predictions": [],
  "error": "JSON inválido en el cuerpo de la petición"
}
```

### 5. Error al Cargar el Modelo

**Respuesta (500 Internal Server Error):**
```json
{
  "success": false,
  "dfa_id": 0,
  "afd_info": null,
  "predictions": [],
  "error": "Error al cargar el modelo: [mensaje de error]"
}
```

---

## 🚀 Características

### 1. **Modelo de Deep Learning**
- Utiliza el modelo **AcepNet** (Dual-Encoder con BiGRU)
- Dos cabezas de salida: Y1 (pertenencia) y Y2 (cadena compartida)
- Umbrales calibrados: Y1=0.43, Y2=0.53

### 2. **Ground Truth Incluido**
- Simula el AFD real para obtener el resultado correcto
- Compara automáticamente predicción vs. realidad
- Indica si la predicción fue correcta (`correct: true/false`)

### 3. **Validación de Alfabeto**
- Detecta si la cadena contiene caracteres fuera del alfabeto del AFD
- Marca `alphabet_mismatch: true` en esos casos
- Devuelve probabilidad 0.0 si hay desajuste de alfabeto

### 4. **Múltiples Cadenas**
- Soporta evaluar múltiples cadenas en una sola petición
- Reduce el número de llamadas al API
- Útil para batch processing

### 5. **Cache del Modelo**
- El modelo se carga una vez y se cachea en memoria (singleton)
- Evita recargar el modelo en cada petición
- Mejora significativamente el rendimiento

### 6. **Información del AFD**
- Incluye información completa del AFD en la respuesta
- Regex, alfabeto, estados, estados de aceptación
- Útil para entender el contexto de la predicción

---

## 📊 Ejemplos Prácticos

### Evaluar Varias Cadenas para el Mismo AFD

```bash
# Evaluar múltiples cadenas de prueba
curl -X POST http://localhost:8000/api/acepnet-predict/ \
  -H "Content-Type: application/json" \
  -d '{
    "dfa_id": 0,
    "strings": ["", "C", "L", "LC", "LCIG", "X", "LCIGX"]
  }'
```

### Verificar Precisión del Modelo

```python
import requests

# Obtener predicciones
response = requests.post("http://localhost:8000/api/acepnet-predict/", json={
    "dfa_id": 0,
    "strings": ["C", "LC", "X"]
})

data = response.json()
if data["success"]:
    correct_count = sum(1 for p in data["predictions"] 
                       if p.get("y1", {}).get("correct") == True)
    total = len(data["predictions"])
    accuracy = correct_count / total if total > 0 else 0
    print(f"Precisión: {accuracy:.2%}")
```

---

## 🔧 Detalles Técnicos

### Modelo AcepNet

- **Arquitectura**: Dual-Encoder (String Encoder + AFD Encoder)
- **String Encoder**: Embedding (32 dim) + BiGRU (64 hidden, 2 layers)
- **AFD Encoder**: MLP (3104 → 512 → 256 → 128)
- **Head 1 (Y1)**: Concat(h_str, h_afd) → MLP → Sigmoid
- **Head 2 (Y2)**: h_str → MLP → Sigmoid

### Umbrales Calibrados

- **Y1 (Pertenencia)**: 0.43
- **Y2 (Cadena Compartida)**: 0.53

### Rendimiento

- **Carga del Modelo**: Una sola vez al iniciar (lazy loading)
- **Predicción Individual**: ~1-5ms (CPU) o ~0.1-1ms (GPU)
- **Batch Processing**: Más eficiente que múltiples peticiones individuales

### Validaciones

- **Alfabeto**: Verifica que todos los caracteres estén en el alfabeto del AFD
- **ID**: Valida que el dfa_id esté en el rango 0-5999
- **Cadena Vacía**: Soporta cadenas vacías (`""`) y épsilon (`"<EPS>"`)

---

## 📝 Notas

1. **Primera Carga**: La primera petición puede tardar más porque carga el modelo y el dataset
2. **Alfabeto del AFD**: Si la cadena contiene caracteres fuera del alfabeto, `alphabet_mismatch` será `true` y la probabilidad será 0.0
3. **Ground Truth**: Se obtiene mediante simulación real del AFD, puede ser `null` si hay error en la simulación
4. **Probabilidades**: Los valores están en el rango [0.0, 1.0], donde valores más altos indican mayor confianza
5. **Predicción Binaria**: Se determina comparando la probabilidad con el umbral calibrado
6. **Múltiples Cadenas**: Todas las cadenas se evalúan con el mismo AFD

---

## 🔗 Endpoints Relacionados

- [`/api/search-regex/`](#-api-endpoint-search-regex) - Buscar regex en el dataset por texto o ID
- [`/api/regex-to-dfa/`](./API_REGEX_TO_DFA.md) - Convertir regex a DFA
- [`/api/regex-to-alphabet/`](./API_REGEX_TO_ALPHABET.md) - Predecir alfabeto de una regex
- [`/api/transitions-to-dfa/`](./API_TRANSITIONS_TO_DFA.md) - Construir DFA desde transiciones

---

## 📄 Ejemplo Completo en JavaScript

```javascript
// Predicción con una cadena
async function predecirCadena(dfaId, string) {
  const response = await fetch(
    `http://localhost:8000/api/acepnet-predict/?dfa_id=${dfaId}&string=${encodeURIComponent(string)}`
  );
  const data = await response.json();
  
  if (data.success) {
    const pred = data.predictions[0];
    console.log(`AFD ${dfaId} (${data.afd_info.regex}):`);
    console.log(`  Cadena: "${pred.string}"`);
    console.log(`  Probabilidad (Y1): ${pred.y1.probability.toFixed(4)}`);
    console.log(`  Predicción: ${pred.y1.predicted ? 'ACEPTA' : 'RECHAZA'}`);
    console.log(`  Ground Truth: ${pred.y1.ground_truth ? 'ACEPTA' : 'RECHAZA'}`);
    console.log(`  Correcto: ${pred.y1.correct ? 'SÍ' : 'NO'}`);
    return pred;
  } else {
    console.error('Error:', data.error);
    return null;
  }
}

// Predicción con múltiples cadenas
async function predecirCadenas(dfaId, strings) {
  const response = await fetch('http://localhost:8000/api/acepnet-predict/', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ dfa_id: dfaId, strings: strings })
  });
  const data = await response.json();
  
  if (data.success) {
    console.log(`AFD ${dfaId} (${data.afd_info.regex}):`);
    data.predictions.forEach(pred => {
      console.log(`  "${pred.string}": ${pred.y1.predicted ? 'ACEPTA' : 'RECHAZA'} (${pred.y1.probability.toFixed(4)}) - ${pred.y1.correct ? '✓' : '✗'}`);
    });
    return data.predictions;
  } else {
    console.error('Error:', data.error);
    return [];
  }
}

// Uso
predecirCadena(0, 'C');
predecirCadenas(0, ['C', 'LC', 'LCIG', 'X']);
```

---

## 📄 Ejemplo Completo en Python

```python
import requests

def predecir_cadena(dfa_id, string):
    """Predice si una cadena es aceptada por un AFD"""
    response = requests.get(
        "http://localhost:8000/api/acepnet-predict/",
        params={"dfa_id": dfa_id, "string": string}
    )
    data = response.json()
    
    if data["success"]:
        pred = data["predictions"][0]
        print(f"AFD {dfa_id} ({data['afd_info']['regex']}):")
        print(f"  Cadena: '{pred['string']}'")
        print(f"  Probabilidad (Y1): {pred['y1']['probability']:.4f}")
        print(f"  Predicción: {'ACEPTA' if pred['y1']['predicted'] else 'RECHAZA'}")
        print(f"  Ground Truth: {'ACEPTA' if pred['y1']['ground_truth'] else 'RECHAZA'}")
        print(f"  Correcto: {'SÍ' if pred['y1']['correct'] else 'NO'}")
        return pred
    else:
        print(f"Error: {data['error']}")
        return None

def predecir_cadenas(dfa_id, strings):
    """Predice si múltiples cadenas son aceptadas por un AFD"""
    response = requests.post(
        "http://localhost:8000/api/acepnet-predict/",
        json={"dfa_id": dfa_id, "strings": strings}
    )
    data = response.json()
    
    if data["success"]:
        print(f"AFD {dfa_id} ({data['afd_info']['regex']}):")
        for pred in data["predictions"]:
            correct = "✓" if pred['y1']['correct'] else "✗"
            print(f"  '{pred['string']}': {'ACEPTA' if pred['y1']['predicted'] else 'RECHAZA'} "
                  f"({pred['y1']['probability']:.4f}) - {correct}")
        return data["predictions"]
    else:
        print(f"Error: {data['error']}")
        return []

# Uso
predecir_cadena(0, "C")
predecir_cadenas(0, ["C", "LC", "LCIG", "X"])

# Calcular precisión
predictions = predecir_cadenas(0, ["C", "LC", "LCIG", "X", "Z"])
correct = sum(1 for p in predictions if p.get('y1', {}).get('correct') == True)
total = len(predictions)
accuracy = correct / total if total > 0 else 0
print(f"\nPrecisión: {accuracy:.2%}")
```

---

## 🎓 Casos de Uso

1. **Evaluación de Modelo**: Verificar la precisión del modelo en diferentes cadenas
2. **Testing de AFDs**: Probar rápidamente si cadenas son aceptadas por un AFD
3. **Batch Processing**: Evaluar múltiples cadenas eficientemente
4. **Integración con Frontend**: Permitir a usuarios probar cadenas en tiempo real
5. **Análisis de Rendimiento**: Comparar predicciones del modelo vs. simulación real

---

## 📚 Referencias

- [Modelo AcepNet](./../models/acepnet/README.md)
- [Análisis Completo del Sistema AcepNet](./../models/acepnet/ANALISIS_COMPLETO.md)
- [Diagrama de Flujo AcepNet](./../models/acepnet/DIAGRAMA_FLUJO.md)

