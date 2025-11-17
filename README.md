[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https%3A%2F%2Fgithub.com%2Fvercel%2Fexamples%2Ftree%2Fmain%2Fpython%2Fdjango&demo-title=Django%20%2B%20Vercel&demo-description=Use%20Django%204%20on%20Vercel%20with%20Serverless%20Functions%20using%20the%20Python%20Runtime.&demo-url=https%3A%2F%2Fdjango-template.vercel.app%2F&demo-image=https://assets.vercel.com/image/upload/v1669994241/random/django.png)

# Django + Vercel

This example shows how to use Django 4 on Vercel with Serverless Functions using the [Python Runtime](https://vercel.com/docs/concepts/functions/serverless-functions/runtimes/python).

## Demo

https://django-template.vercel.app/

## How it Works

Our Django application, `example` is configured as an installed application in `api/settings.py`:

```python
# api/settings.py
INSTALLED_APPS = [
    # ...
    'example',
]
```

We allow "\*.vercel.app" subdomains in `ALLOWED_HOSTS`, in addition to 127.0.0.1:

```python
# api/settings.py
ALLOWED_HOSTS = ['127.0.0.1', '.vercel.app']
```

The `wsgi` module must use a public variable named `app` to expose the WSGI application:

```python
# api/wsgi.py
app = get_wsgi_application()
```

The corresponding `WSGI_APPLICATION` setting is configured to use the `app` variable from the `api.wsgi` module:

```python
# api/settings.py
WSGI_APPLICATION = 'api.wsgi.app'
```

There is a single view which renders the current time in `example/views.py`:

```python
# example/views.py
from datetime import datetime

from django.http import HttpResponse


def index(request):
    now = datetime.now()
    html = f'''
    <html>
        <body>
            <h1>Hello from Vercel!</h1>
            <p>The current time is { now }.</p>
        </body>
    </html>
    '''
    return HttpResponse(html)
```

This view is exposed a URL through `example/urls.py`:

```python
# example/urls.py
from django.urls import path

from example.views import index


urlpatterns = [
    path('', index),
]
```

Finally, it's made accessible to the Django server inside `api/urls.py`:

```python
# api/urls.py
from django.urls import path, include

urlpatterns = [
    ...
    path('', include('example.urls')),
]
```

This example uses the Web Server Gateway Interface (WSGI) with Django to enable handling requests on Vercel with Serverless Functions.

## API Endpoint: Regex to DFA

El proyecto incluye un endpoint que convierte expresiones regulares a autómatas finitos deterministas (DFA) usando el algoritmo de Thompson y la construcción de subconjuntos.

### Endpoint

**URL:** `/api/regex-to-dfa/`

**Métodos:** `GET`, `POST`

### Uso

#### GET Request

```bash
# Convertir una expresión regular a DFA
curl "http://localhost:8000/api/regex-to-dfa/?regex=a*b"

# Convertir y probar una cadena
curl "http://localhost:8000/api/regex-to-dfa/?regex=a*b&test=aaab"
```

#### POST Request

```bash
# Convertir una expresión regular
curl -X POST http://localhost:8000/api/regex-to-dfa/ \
  -H "Content-Type: application/json" \
  -d '{"regex": "a*b"}'

# Convertir y probar una cadena
curl -X POST http://localhost:8000/api/regex-to-dfa/ \
  -H "Content-Type: application/json" \
  -d '{"regex": "a*b", "test": "aaab"}'
```

### Respuesta

```json
{
  "success": true,
  "regex": "a*b",
  "dfa": {
    "alphabet": ["a", "b"],
    "states": ["S0", "S1", "S2"],
    "start": "S0",
    "accepting": ["S2"],
    "transitions": [
      {"from": "S0", "symbol": "a", "to": "S1"},
      {"from": "S0", "symbol": "b", "to": "S2"},
      {"from": "S1", "symbol": "a", "to": "S1"},
      {"from": "S1", "symbol": "b", "to": "S2"}
    ]
  },
  "test_result": {
    "string": "aaab",
    "accepted": true
  },
  "error": null
}
```

### Características

- Soporte para expresiones regulares estándar: `*`, `+`, `?`, `|`, `.`, `()`, y caracteres escapados con `\`
- Conversión automática de Regex → NFA (Thompson) → DFA (Subconjuntos)
- Opción de probar cadenas contra el DFA generado
- Respuestas en formato JSON estructurado
- Manejo de errores con mensajes descriptivos

### Ejemplos de Expresiones Regulares

- `a*b` - Cero o más 'a' seguidas de 'b'
- `(a|b)*` - Cualquier combinación de 'a' y 'b'
- `a+b` - Una o más 'a' seguidas de 'b'
- `a?b` - Opcionalmente 'a' seguida de 'b'
- `a\.b` - Literal 'a.b' (el punto está escapado)

## API Endpoint: Regex to Alphabet (AlphabetNet)

El proyecto incluye un endpoint que predice el alfabeto de una expresión regular usando el modelo de aprendizaje profundo AlphabetNet. Este modelo utiliza una red neuronal entrenada para predecir qué símbolos del alfabeto están presentes en el lenguaje definido por la expresión regular.

### Endpoint

**URL:** `/api/regex-to-alphabet/`

**Métodos:** `GET`, `POST`

### Uso

#### GET Request

```bash
# Predecir alfabeto de una expresión regular
curl "http://localhost:8000/api/regex-to-alphabet/?regex=(AB)*C"

# Con expresión regular codificada en URL
curl "http://localhost:8000/api/regex-to-alphabet/?regex=%28AB%29%2AC"
```

#### POST Request

```bash
# Predecir alfabeto de una expresión regular
curl -X POST http://localhost:8000/api/regex-to-alphabet/ \
  -H "Content-Type: application/json" \
  -d '{"regex": "(AB)*C"}'
```

### Respuesta

**Éxito (200 OK):**
```json
{
  "success": true,
  "regex": "(AB)*C",
  "alphabet": ["A", "B", "C"],
  "probabilities": {
    "A": 0.95,
    "B": 0.87,
    "C": 0.92,
    "D": 0.12,
    "E": 0.08,
    ...
  },
  "error": null
}
```

**Error (400 Bad Request):**
```json
{
  "success": false,
  "error": "Parámetro 'regex' es requerido"
}
```

**Error (500 Internal Server Error):**
```json
{
  "success": false,
  "regex": "(AB)*C",
  "alphabet": null,
  "probabilities": null,
  "error": "Error al cargar el modelo: Archivo no encontrado"
}
```

### Características

- **Predicción basada en ML**: Utiliza un modelo de aprendizaje profundo (AlphabetNet) entrenado para predecir el alfabeto
- **Probabilidades por símbolo**: Devuelve la probabilidad de que cada símbolo pertenezca al alfabeto
- **Alfabeto predicho**: Lista de símbolos con probabilidad mayor al threshold (por defecto 0.5)
- **Lazy loading**: El modelo se carga solo cuando se necesita y se cachea en memoria
- **Manejo de errores**: Respuestas con mensajes descriptivos en caso de error

### Estructura de la Respuesta

- **`success`**: `true` si la predicción fue exitosa, `false` en caso de error
- **`regex`**: La expresión regular proporcionada
- **`alphabet`**: Lista ordenada de símbolos predichos (símbolos con probabilidad ≥ threshold)
- **`probabilities`**: Diccionario con la probabilidad de cada símbolo del alfabeto completo
- **`error`**: `null` si no hay error, o un mensaje descriptivo en caso contrario

### Uso desde el Frontend (JavaScript/TypeScript)

**Con fetch:**
```javascript
async function predictAlphabet(regex) {
  try {
    const response = await fetch('http://localhost:8000/api/regex-to-alphabet/', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ regex: regex })
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.error || 'Error al predecir el alfabeto');
    }

    const data = await response.json();
    console.log('Alfabeto predicho:', data.alphabet);
    console.log('Probabilidades:', data.probabilities);
    return data;
  } catch (error) {
    console.error('Error:', error);
    throw error;
  }
}

// Ejemplo de uso
predictAlphabet('(AB)*C');
```

**Con axios (TypeScript):**
```typescript
import axios from 'axios';

interface AlphabetResponse {
  success: boolean;
  regex: string;
  alphabet: string[];
  probabilities: { [symbol: string]: number };
  error: string | null;
}

async function predictAlphabet(regex: string): Promise<AlphabetResponse> {
  try {
    const response = await axios.post<AlphabetResponse>(
      'http://localhost:8000/api/regex-to-alphabet/',
      { regex }
    );
    return response.data;
  } catch (error) {
    if (axios.isAxiosError(error) && error.response) {
      throw new Error(error.response.data.error || 'Error al predecir el alfabeto');
    }
    throw error;
  }
}

// Ejemplo de uso
predictAlphabet('(AB)*C').then(data => {
  console.log('Alfabeto:', data.alphabet);
  console.log('Probabilidades:', data.probabilities);
});
```

**Ejemplo con React:**
```tsx
import React, { useState } from 'react';

function AlphabetPredictor() {
  const [regex, setRegex] = useState('');
  const [result, setResult] = useState<AlphabetResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handlePredict = async () => {
    if (!regex.trim()) {
      setError('Por favor ingresa una expresión regular');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const response = await fetch('http://localhost:8000/api/regex-to-alphabet/', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ regex: regex })
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Error al predecir el alfabeto');
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Error desconocido');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <input
        type="text"
        value={regex}
        onChange={(e) => setRegex(e.target.value)}
        placeholder="Ingresa una expresión regular (ej: (AB)*C)"
      />
      <button onClick={handlePredict} disabled={loading}>
        {loading ? 'Prediciendo...' : 'Predecir Alfabeto'}
      </button>
      {error && <p style={{ color: 'red' }}>{error}</p>}
      {result && (
        <div>
          <h3>Resultado</h3>
          <p><strong>Regex:</strong> {result.regex}</p>
          <p><strong>Alfabeto predicho:</strong> {result.alphabet.join(', ')}</p>
          <details>
            <summary>Probabilidades por símbolo</summary>
            <ul>
              {Object.entries(result.probabilities)
                .sort(([, a], [, b]) => b - a)
                .map(([symbol, prob]) => (
                  <li key={symbol}>
                    {symbol}: {(prob * 100).toFixed(2)}%
                    {result.alphabet.includes(symbol) && ' ✓'}
                  </li>
                ))}
            </ul>
          </details>
        </div>
      )}
    </div>
  );
}
```

### Requisitos del Modelo

Para que el endpoint funcione correctamente, se necesitan los siguientes archivos:

1. **Modelo entrenado**: `models/alphabetnet/alphabetnet.pt` ✓ (verificado)
2. **Hiperparámetros**: `hparams.json` (opcional - se intentará cargar desde el checkpoint si no existe)
3. **Archivos del modelo**: `model.py` y `train.py` ⚠️ **REQUERIDOS**

#### Ubicación de los Archivos del Modelo

Los archivos `model.py` y `train.py` deben estar en una de estas ubicaciones:

- `models/src/` (recomendado según test_model.py)
- `models/alphabetnet/`
- `src/` (en la raíz del proyecto)

**Importante**: Estos archivos son necesarios porque contienen:
- `model.py`: La clase `AlphabetNet` que define la arquitectura del modelo
- `train.py`: Las constantes `ALPHABET`, `MAX_PREFIX_LEN` y la función `regex_to_indices`

Si no tienes estos archivos, cópialos desde el proyecto donde se entrenó el modelo originalmente.

#### Verificación de Archivos

Puedes ejecutar el script de verificación para verificar qué archivos faltan:

```bash
python check_model_files.py
```

Este script mostrará:
- ✓ Archivos encontrados
- ✗ Archivos faltantes
- Ubicaciones donde se están buscando los archivos

### Thresholds

Por defecto, el modelo usa un threshold de 0.5 para todos los símbolos. Si existe un archivo `thresholds.json` con thresholds personalizados por símbolo, se cargarán automáticamente. El formato esperado es:

```json
{
  "per_symbol": {
    "A": 0.6,
    "B": 0.7,
    "C": 0.5,
    ...
  }
}
```

### Errores Comunes

- **`"Parámetro 'regex' es requerido"`**: Falta el parámetro regex en la petición
- **`"El parámetro 'regex' no puede estar vacío"`**: El regex proporcionado está vacío
- **`"Error al cargar el modelo: Archivo no encontrado"`**: No se encontró el archivo del modelo o los hiperparámetros
- **`"Error al importar el modelo: ..."`**: No se encontraron los archivos `model.py` o `train.py`
- **`"Error al procesar la regex: ..."`**: Error durante la predicción (puede ser un problema con el formato de la regex o con el modelo)

### Notas Importantes

- El modelo se carga de forma lazy (solo cuando se necesita) y se cachea en memoria para mejorar el rendimiento
- Las probabilidades son valores entre 0 y 1, donde valores más altos indican mayor confianza de que el símbolo pertenece al alfabeto
- El alfabeto predicho (`alphabet`) contiene solo los símbolos con probabilidad mayor o igual al threshold
- El diccionario `probabilities` contiene probabilidades para todos los símbolos del alfabeto completo del modelo
- El endpoint es compatible con expresiones regulares estándar, pero el modelo fue entrenado con un conjunto específico de patrones

### Endpoint: Construir DFA desde Transiciones

**URL:** `/api/transitions-to-dfa/`

**Métodos:** `POST`

Este endpoint construye un DFA desde una especificación de transiciones y devuelve la información completa del DFA en el mismo formato que el endpoint `regex-to-dfa`. Útil cuando ya tienes las transiciones del DFA y quieres obtener la representación estructurada, validar el DFA, o probar cadenas.

#### Uso

**POST Request con curl:**
```bash
curl -X POST http://localhost:8000/api/transitions-to-dfa/ \
  -H "Content-Type: application/json" \
  -d '{
    "states": ["S0", "S1", "S2"],
    "start": "S0",
    "accepting": ["S2"],
    "transitions": [
      {"from": "S0", "symbol": "a", "to": "S1"},
      {"from": "S0", "symbol": "b", "to": "S0"},
      {"from": "S1", "symbol": "a", "to": "S1"},
      {"from": "S1", "symbol": "b", "to": "S2"}
    ],
    "test": "aab"
  }'
```

**POST Request con prueba de cadena opcional:**
```bash
curl -X POST http://localhost:8000/api/transitions-to-dfa/ \
  -H "Content-Type: application/json" \
  -d '{
    "states": ["S0", "S1"],
    "start": "S0",
    "accepting": ["S1"],
    "transitions": [
      {"from": "S0", "symbol": "0", "to": "S0"},
      {"from": "S0", "symbol": "1", "to": "S1"},
      {"from": "S1", "symbol": "0", "to": "S1"},
      {"from": "S1", "symbol": "1", "to": "S0"}
    ],
    "test": "101"
  }'
```

#### Uso desde el Frontend (JavaScript/TypeScript)

**Con fetch:**
```javascript
async function buildDFAFromTransitions(states, start, accepting, transitions, testString = null) {
  try {
    const response = await fetch('http://localhost:8000/api/transitions-to-dfa/', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        states: states,
        start: start,
        accepting: accepting,
        transitions: transitions,
        test: testString
      })
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.error || 'Error al construir el DFA');
    }

    const data = await response.json();
    console.log('DFA construido:', data.dfa);
    if (data.test_result) {
      console.log('Resultado de prueba:', data.test_result);
    }
    return data;
  } catch (error) {
    console.error('Error:', error);
    throw error;
  }
}

// Ejemplo de uso
buildDFAFromTransitions(
  ["S0", "S1", "S2"],
  "S0",
  ["S2"],
  [
    {"from": "S0", "symbol": "a", "to": "S1"},
    {"from": "S1", "symbol": "b", "to": "S2"}
  ],
  "ab"
);
```

**Con axios (TypeScript):**
```typescript
import axios from 'axios';

interface Transition {
  from: string;
  symbol: string;
  to: string;
}

interface DFAResponse {
  success: boolean;
  dfa: {
    alphabet: string[];
    states: string[];
    start: string;
    accepting: string[];
    transitions: Transition[];
  };
  test_result: {
    string: string;
    accepted: boolean;
  } | null;
  error: string | null;
}

async function buildDFAFromTransitions(
  states: string[],
  start: string,
  accepting: string[],
  transitions: Transition[],
  testString?: string
): Promise<DFAResponse> {
  try {
    const response = await axios.post<DFAResponse>('http://localhost:8000/api/transitions-to-dfa/', {
      states,
      start,
      accepting,
      transitions,
      test: testString
    });
    return response.data;
  } catch (error) {
    if (axios.isAxiosError(error) && error.response) {
      throw new Error(error.response.data.error || 'Error al construir el DFA');
    }
    throw error;
  }
}
```

**Ejemplo con React:**
```jsx
import React, { useState } from 'react';

function DFABuilder() {
  const [dfa, setDfa] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleBuildDFA = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch('http://localhost:8000/api/transitions-to-dfa/', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          states: ["S0", "S1", "S2"],
          start: "S0",
          accepting: ["S2"],
          transitions: [
            {"from": "S0", "symbol": "a", "to": "S1"},
            {"from": "S1", "symbol": "b", "to": "S2"}
          ],
          test: "ab"
        })
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error);
      }

      const data = await response.json();
      setDfa(data.dfa);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <button onClick={handleBuildDFA} disabled={loading}>
        {loading ? 'Construyendo DFA...' : 'Construir DFA'}
      </button>
      {error && <p style={{color: 'red'}}>Error: {error}</p>}
      {dfa && (
        <div>
          <h3>DFA Construido</h3>
          <p>Alfabeto: {dfa.alphabet.join(', ')}</p>
          <p>Estados: {dfa.states.join(', ')}</p>
          <p>Estado inicial: {dfa.start}</p>
          <p>Estados de aceptación: {dfa.accepting.join(', ')}</p>
          <h4>Transiciones:</h4>
          <ul>
            {dfa.transitions.map((t, i) => (
              <li key={i}>{t.from} --{t.symbol}--> {t.to}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
```

#### Request Body

```json
{
  "states": ["S0", "S1", "S2"],           // Lista de nombres de estados (requerido)
  "start": "S0",                          // Estado inicial (requerido)
  "accepting": ["S2"],                    // Lista de estados de aceptación (requerido, puede ser [])
  "transitions": [                        // Lista de transiciones (requerido)
    {
      "from": "S0",                       // Estado origen (requerido)
      "symbol": "a",                      // Símbolo de la transición (requerido)
      "to": "S1"                          // Estado destino (requerido)
    }
  ],
  "test": "aab"                           // Cadena opcional para probar el DFA
}
```

#### Respuesta

**Éxito (200 OK):**
```json
{
  "success": true,
  "dfa": {
    "alphabet": ["a", "b"],
    "states": ["S0", "S1", "S2"],
    "start": "S0",
    "accepting": ["S2"],
    "transitions": [
      {"from": "S0", "symbol": "a", "to": "S1"},
      {"from": "S0", "symbol": "b", "to": "S0"},
      {"from": "S1", "symbol": "a", "to": "S1"},
      {"from": "S1", "symbol": "b", "to": "S2"}
    ]
  },
  "test_result": {
    "string": "aab",
    "accepted": true
  },
  "error": null
}
```

**Error (400 Bad Request):**
```json
{
  "success": false,
  "dfa": null,
  "test_result": null,
  "error": "Estado inicial 'S5' no está en la lista de estados"
}
```

#### Validaciones

El endpoint valida:

1. **Parámetros requeridos**: `states`, `start`, `accepting`, `transitions` deben estar presentes
2. **Estado inicial**: El estado inicial debe existir en la lista de estados
3. **Estados de aceptación**: Todos los estados de aceptación deben existir en la lista de estados
4. **Transiciones válidas**: 
   - Cada transición debe tener `from`, `symbol`, y `to`
   - Los estados `from` y `to` deben existir en la lista de estados
   - El DFA debe ser determinista (no puede haber múltiples transiciones desde el mismo estado con el mismo símbolo)
5. **Formato JSON**: El cuerpo de la petición debe ser un JSON válido

#### Errores Comunes

- **`"Parámetro 'states' requerido"`**: Falta la lista de estados
- **`"Estado inicial 'X' no está en la lista de estados"`**: El estado inicial no existe en `states`
- **`"Estado de aceptación 'X' no está en la lista de estados"`**: Un estado de aceptación no existe en `states`
- **`"Transición no determinista: desde 'S0' con símbolo 'a' hay múltiples transiciones"`**: El DFA no es determinista (hay múltiples transiciones desde el mismo estado con el mismo símbolo)
- **`"Transición inválida: falta 'from', 'symbol' o 'to'"`**: Una transición no tiene todos los campos requeridos
- **`"Estado origen 'X' en transición no está en la lista de estados"`**: Un estado origen en una transición no existe

#### Notas Importantes

- El DFA debe ser **determinista**: no puede haber múltiples transiciones desde el mismo estado con el mismo símbolo
- Los nombres de estados pueden ser cualquier string (ej: "S0", "q0", "estado1", etc.)
- Los símbolos del alfabeto pueden ser cualquier string (ej: "a", "0", "ε", etc.)
- El campo `test` es opcional: si se proporciona, el DFA probará la cadena y devolverá el resultado en `test_result`
- El formato de respuesta es idéntico al endpoint `regex-to-dfa`, facilitando la interoperabilidad

### Endpoint: Descargar DFA en formato JFLAP

**URL:** `/api/regex-to-dfa/jff/`

**Métodos:** `GET`, `POST`

Este endpoint convierte una expresión regular a DFA y devuelve un archivo `.jff` compatible con JFLAP para visualización y análisis.

#### Uso

**GET Request con curl:**
```bash
# Descargar archivo JFF
curl "http://localhost:8000/api/regex-to-dfa/jff/?regex=a*b" -o dfa.jff

# Con expresión regular codificada en URL
curl "http://localhost:8000/api/regex-to-dfa/jff/?regex=a%2Ab" -o dfa.jff
```

**POST Request con curl:**
```bash
# Descargar archivo JFF usando POST
curl -X POST http://localhost:8000/api/regex-to-dfa/jff/ \
  -H "Content-Type: application/json" \
  -d '{"regex": "a*b"}' \
  -o dfa.jff
```

**En el navegador:**
```
http://localhost:8000/api/regex-to-dfa/jff/?regex=a*b
```
El navegador descargará automáticamente el archivo `dfa_a_b.jff`

#### Uso desde el Frontend (JavaScript/TypeScript)

**Con fetch y GET:**
```javascript
async function downloadJFF(regex) {
  try {
    // Codificar la regex para la URL
    const encodedRegex = encodeURIComponent(regex);
    const url = `http://localhost:8000/api/regex-to-dfa/jff/?regex=${encodedRegex}`;
    
    const response = await fetch(url, {
      method: 'GET',
    });
    
    if (!response.ok) {
      // Si hay error, el servidor devuelve JSON
      const errorData = await response.json();
      throw new Error(errorData.error || 'Error al generar el archivo JFF');
    }
    
    // Obtener el nombre del archivo del header Content-Disposition
    const contentDisposition = response.headers.get('Content-Disposition');
    let filename = 'dfa.jff';
    if (contentDisposition) {
      const filenameMatch = contentDisposition.match(/filename="?(.+)"?/);
      if (filenameMatch) {
        filename = filenameMatch[1];
      }
    }
    
    // Convertir la respuesta a blob y descargar
    const blob = await response.blob();
    const downloadUrl = window.URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = downloadUrl;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    window.URL.revokeObjectURL(downloadUrl);
    
    console.log('Archivo JFF descargado:', filename);
  } catch (error) {
    console.error('Error al descargar JFF:', error);
    throw error;
  }
}

// Uso:
downloadJFF('a*b');
```

**Con fetch y POST:**
```javascript
async function downloadJFFPost(regex) {
  try {
    const response = await fetch('http://localhost:8000/api/regex-to-dfa/jff/', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ regex: regex }),
    });
    
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.error || 'Error al generar el archivo JFF');
    }
    
    // Obtener el nombre del archivo
    const contentDisposition = response.headers.get('Content-Disposition');
    let filename = 'dfa.jff';
    if (contentDisposition) {
      const filenameMatch = contentDisposition.match(/filename="?(.+)"?/);
      if (filenameMatch) {
        filename = filenameMatch[1];
      }
    }
    
    // Descargar el archivo
    const blob = await response.blob();
    const downloadUrl = window.URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = downloadUrl;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    window.URL.revokeObjectURL(downloadUrl);
    
    console.log('Archivo JFF descargado:', filename);
  } catch (error) {
    console.error('Error al descargar JFF:', error);
    throw error;
  }
}

// Uso:
downloadJFFPost('(a|b)*');
```

**Con axios (TypeScript):**
```typescript
import axios from 'axios';

async function downloadJFFAxios(regex: string): Promise<void> {
  try {
    const response = await axios({
      url: 'http://localhost:8000/api/regex-to-dfa/jff/',
      method: 'POST',
      data: { regex },
      responseType: 'blob', // Importante: especificar blob para archivos
    });
    
    // Obtener el nombre del archivo del header
    const contentDisposition = response.headers['content-disposition'];
    let filename = 'dfa.jff';
    if (contentDisposition) {
      const filenameMatch = contentDisposition.match(/filename="?(.+)"?/);
      if (filenameMatch) {
        filename = filenameMatch[1];
      }
    }
    
    // Crear un enlace temporal para descargar
    const url = window.URL.createObjectURL(new Blob([response.data]));
    const link = document.createElement('a');
    link.href = url;
    link.setAttribute('download', filename);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    window.URL.revokeObjectURL(url);
    
    console.log('Archivo JFF descargado:', filename);
  } catch (error) {
    if (axios.isAxiosError(error) && error.response) {
      // Si el error es una respuesta JSON del servidor
      const reader = new FileReader();
      reader.onload = () => {
        try {
          const errorData = JSON.parse(reader.result as string);
          console.error('Error del servidor:', errorData.error);
        } catch (e) {
          console.error('Error al procesar respuesta del servidor');
        }
      };
      reader.readAsText(error.response.data);
    } else {
      console.error('Error al descargar JFF:', error);
    }
    throw error;
  }
}

// Uso:
downloadJFFAxios('a+b');
```

**Ejemplo con React:**
```tsx
import React, { useState } from 'react';

function JFFDownloadButton() {
  const [regex, setRegex] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleDownload = async () => {
    if (!regex.trim()) {
      setError('Por favor ingresa una expresión regular');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const encodedRegex = encodeURIComponent(regex);
      const response = await fetch(
        `http://localhost:8000/api/regex-to-dfa/jff/?regex=${encodedRegex}`
      );

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Error al generar el archivo JFF');
      }

      // Obtener nombre del archivo
      const contentDisposition = response.headers.get('Content-Disposition');
      let filename = 'dfa.jff';
      if (contentDisposition) {
        const match = contentDisposition.match(/filename="?(.+)"?/);
        if (match) filename = match[1];
      }

      // Descargar
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = filename;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      window.URL.revokeObjectURL(url);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Error desconocido');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <input
        type="text"
        value={regex}
        onChange={(e) => setRegex(e.target.value)}
        placeholder="Ingresa una expresión regular (ej: a*b)"
      />
      <button onClick={handleDownload} disabled={loading}>
        {loading ? 'Descargando...' : 'Descargar JFF'}
      </button>
      {error && <p style={{ color: 'red' }}>{error}</p>}
    </div>
  );
}
```

#### Respuesta del Servidor

**Éxito (200 OK):**
- **Content-Type:** `application/xml; charset=utf-8`
- **Content-Disposition:** `attachment; filename="dfa_<regex>.jff"; filename*=UTF-8''<encoded_filename>`
- **Content-Length:** Tamaño del archivo en bytes
- **Body:** Contenido XML del archivo JFF

**Error (400 Bad Request):**
- **Content-Type:** `application/json`
- **Body:**
```json
{
  "success": false,
  "regex": "a*b",
  "error": "Mensaje de error descriptivo"
}
```

#### Manejo de Errores

El endpoint puede retornar errores en los siguientes casos:
- **Regex faltante:** `"Parámetro 'regex' requerido"`
- **JSON inválido (POST):** `"JSON inválido en el cuerpo de la petición"`
- **Regex inválida:** Mensajes de error específicos de la conversión (paréntesis desbalanceados, token inesperado, etc.)

#### Formato del archivo

El archivo generado es un XML compatible con JFLAP que incluye:
- Estados del DFA con nombres (S0, S1, S2, ...)
- Estado inicial marcado
- Estados de aceptación marcados
- Todas las transiciones con sus símbolos

#### Nombre del archivo

El nombre del archivo se genera automáticamente desde la expresión regular:
- Caracteres especiales se reemplazan por guiones bajos
- Formato: `dfa_<regex_sanitizada>.jff`
- Ejemplos:
  - `a*b` → `dfa_a_b.jff`
  - `(a|b)*` → `dfa_a_b__jff`
  - `a+b` → `dfa_a_b.jff`
- El nombre del archivo está codificado correctamente para soportar caracteres especiales mediante RFC 5987

#### Notas Importantes para el Frontend

1. **CORS:** El servidor incluye el header `Access-Control-Expose-Headers` para permitir que el frontend acceda a `Content-Disposition`, `Content-Length` y `Content-Type`
2. **Codificación de URL:** Cuando uses GET, codifica la expresión regular con `encodeURIComponent()`
3. **Tipo de respuesta:** Para archivos, usa `responseType: 'blob'` en axios o `response.blob()` en fetch
4. **Nombre del archivo:** Extrae el nombre del header `Content-Disposition` usando una expresión regular
5. **Errores:** Si la respuesta no es exitosa, el servidor devuelve JSON con el error, no XML
6. **Compatibilidad:** Los archivos generados son compatibles con JFLAP 7.1 y versiones posteriores

### Endpoint: Procesar Archivo de Regex a CSV con Columna Clase

**URL:** `/api/regex-file-to-csv/`

**Métodos:** `POST`

Este endpoint recibe un archivo de texto (`.txt`) o CSV (`.csv`) con expresiones regulares (una por línea) y genera un archivo CSV con toda la información del DFA más una columna adicional llamada "Clase" que contiene un diccionario JSON con 100 cadenas de prueba (50 aceptadas y 50 rechazadas) y sus valores booleanos. **Todas las cadenas usan únicamente símbolos del alfabeto del DFA.**

#### Uso

**POST Request con curl:**
```bash
# Procesar archivo de texto
curl -X POST http://localhost:8000/api/regex-file-to-csv/ \
  -F "file=@regexes.txt" \
  -o resultado.csv

# Procesar archivo CSV
curl -X POST http://localhost:8000/api/regex-file-to-csv/ \
  -F "file=@regexes.csv" \
  -o resultado.csv
```

**POST Request con Python (requests):**
```python
import requests

url = "http://localhost:8000/api/regex-file-to-csv/"
files = {'file': open('regexes.txt', 'rb')}
response = requests.post(url, files=files)

if response.status_code == 200:
    with open('resultado.csv', 'wb') as f:
        f.write(response.content)
    print("CSV generado exitosamente")
else:
    print(f"Error: {response.json()}")
```

**POST Request con JavaScript (fetch):**
```javascript
const formData = new FormData();
formData.append('file', fileInput.files[0]); // fileInput es un elemento <input type="file">

fetch('http://localhost:8000/api/regex-file-to-csv/', {
    method: 'POST',
    body: formData
})
.then(response => response.blob())
.then(blob => {
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'resultado.csv';
    a.click();
});
```

#### Formato del archivo de entrada

**Archivo de texto (`.txt`):**
```
a*b
(a|b)*
a+b
a?b
```

**Archivo CSV (`.csv`):**
```csv
Regex
a*b
(a|b)*
a+b
a?b
```

O también puede tener otras columnas, pero se leerá la columna "Regex" o la primera columna:
```csv
Regex,Descripcion
a*b,Zero or more a followed by b
(a|b)*,Any combination of a and b
```

#### Formato del archivo de salida

El CSV generado contiene las siguientes columnas:

1. **Regex**: La expresión regular original
2. **Alfabeto**: Los símbolos del alfabeto del DFA (separados por espacios)
3. **Estados de aceptación**: Los estados de aceptación del DFA (separados por espacios)
4. **Estados**: Todos los estados del DFA (separados por espacios)
5. **Transiciones**: Las transiciones del DFA en formato `Sx --a--> Sy` (separadas por ` | `)
6. **Clase**: Un diccionario JSON con 100 cadenas de prueba y sus valores booleanos (validado automáticamente)
7. **Error**: Mensaje de error si hubo algún problema al procesar la regex (vacío si fue exitoso)

**Importante:** El archivo CSV se guarda con encoding UTF-8 para evitar problemas con caracteres especiales como `[`, `]`, `→`, etc. que pueden aparecer en las transiciones. Asegúrate de leer el archivo con `encoding='utf-8'` en Python.

#### Ejemplo de archivo CSV de salida

```csv
Regex,Alfabeto,Estados de aceptación,Estados,Transiciones,Clase,Error
a*b,"a b","S2","S0 S1 S2","S0 --a--> S1 | S0 --b--> S2 | S1 --a--> S1 | S1 --b--> S2","{""": false, ""a"": false, ""b"": true, ""aa"": false, ""ab"": true, ...}","
```

#### Columna "Clase"

La columna "Clase" contiene un diccionario JSON con exactamente 100 cadenas de prueba:

- **50 cadenas aceptadas**: Cadenas que son aceptadas por el DFA (valor `true`)
- **50 cadenas rechazadas**: Cadenas que son rechazadas por el DFA (valor `false`)

**IMPORTANTE:** Todas las cadenas (aceptadas y rechazadas) usan **únicamente símbolos del alfabeto del DFA**. Las cadenas rechazadas son válidas sobre el alfabeto, pero el DFA las rechaza porque no llegan a un estado de aceptación.

Ejemplo de contenido de la columna "Clase":
```json
{
  "": false,
  "a": false,
  "b": true,
  "aa": false,
  "ab": true,
  "aaa": false,
  "aab": true,
  "ba": false,
  "bb": false,
  "aba": false,
  ...
}
```

En este ejemplo, todas las cadenas usan solo los símbolos `a` y `b` (el alfabeto del DFA). Las cadenas rechazadas como `"a"`, `"aa"`, `"ba"` son válidas sobre el alfabeto, pero el DFA las rechaza porque no terminan en un estado de aceptación.

**Características de las cadenas generadas:**
- Todas las cadenas (aceptadas y rechazadas) usan **únicamente símbolos del alfabeto del DFA**
- Las cadenas aceptadas son generadas probando diferentes combinaciones del alfabeto hasta encontrar las que el DFA acepta
- Las cadenas rechazadas son generadas probando diferentes combinaciones del alfabeto hasta encontrar las que el DFA rechaza (no terminan en un estado de aceptación)
- Todas las cadenas son únicas en el diccionario
- El diccionario siempre contiene exactamente 100 entradas (50 aceptadas + 50 rechazadas)

**Validación de la columna Clase:**

El JSON en la columna "Clase" es validado automáticamente durante la generación. Si planeas procesar esta columna en Python, asegúrate de parsearla correctamente:

```python
import csv
import json

# Leer el CSV con encoding UTF-8
with open('resultado.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        regex = row['Regex']
        clase_str = row['Clase']
        
        # Validar y parsear la columna Clase
        if clase_str and clase_str.strip():
            try:
                clase_dict = json.loads(clase_str)
                # Verificar que sea un diccionario
                if isinstance(clase_dict, dict):
                    print(f"Regex: {regex}")
                    print(f"  Total de cadenas: {len(clase_dict)}")
                    print(f"  Aceptadas: {sum(1 for v in clase_dict.values() if v)}")
                    print(f"  Rechazadas: {sum(1 for v in clase_dict.values() if not v)}")
                else:
                    print(f"⚠ Advertencia: Clase no es un diccionario para regex: {regex}")
            except json.JSONDecodeError as e:
                print(f"✗ Error al parsear JSON en Clase para regex '{regex}': {e}")
                print(f"  Contenido: {clase_str[:100]}...")
        else:
            print(f"⚠ Clase vacía para regex: {regex} (puede haber un error en la columna Error)")
```

**Nota importante:** El archivo CSV se guarda con encoding UTF-8 para evitar problemas con caracteres especiales como `[`, `]`, `→`, etc. que pueden aparecer en las transiciones. Asegúrate de leer el archivo con `encoding='utf-8'` en Python.

#### Manejo de errores

Si una expresión regular es inválida, el endpoint continuará procesando las demás y registrará el error en la columna "Error":

```csv
Regex,Alfabeto,Estados de aceptación,Estados,Transiciones,Clase,Error
((((,"","","","","","Línea 1: ValueError: Paréntesis desbalanceados."
```

#### Respuestas de error

Si hay un error al procesar el archivo, el endpoint retornará un JSON con el error:

```json
{
  "success": false,
  "error": "No se proporcionó archivo. Use el campo 'file' en el formulario."
}
```

Códigos de estado HTTP:
- `200`: Archivo procesado exitosamente
- `400`: Error en la petición (archivo faltante, formato inválido, etc.)

#### Nombre del archivo de salida

El archivo CSV generado se descarga con el nombre:
- Formato: `regex_dataset_YYYYMMDD_HHMMSS.csv`
- Ejemplo: `regex_dataset_20241107_143022.csv`

### Notas Importantes

- **Hosts permitidos**: El backend acepta peticiones desde `localhost`, `127.0.0.1` y dominios `.vercel.app`
- **URL del endpoint**: Asegúrate de usar la URL completa con la barra final: `/api/regex-to-dfa/`, `/api/regex-to-dfa/jff/`, `/api/regex-file-to-csv/`
- **CORS**: Está habilitado para todos los orígenes, por lo que el frontend puede hacer peticiones sin problemas
- **Archivos JFF**: Los archivos generados son compatibles con JFLAP y se pueden abrir directamente en la herramienta
- **Archivos CSV**: El endpoint de procesamiento de archivos acepta archivos `.txt` y `.csv`. El archivo CSV de salida incluye codificación UTF-8 y puede contener expresiones JSON en la columna "Clase"
- **Límites**: El procesamiento de archivos puede tardar más tiempo dependiendo del número de expresiones regulares y la complejidad de cada una. Se recomienda procesar archivos con menos de 1000 expresiones regulares por lote

## Running Locally

```bash
python manage.py runserver
```

Your Django application is now available at `http://localhost:8000`.

## One-Click Deploy

Deploy the example using [Vercel](https://vercel.com?utm_source=github&utm_medium=readme&utm_campaign=vercel-examples):

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https%3A%2F%2Fgithub.com%2Fvercel%2Fexamples%2Ftree%2Fmain%2Fpython%2Fdjango&demo-title=Django%20%2B%20Vercel&demo-description=Use%20Django%204%20on%20Vercel%20with%20Serverless%20Functions%20using%20the%20Python%20Runtime.&demo-url=https%3A%2F%2Fdjango-template.vercel.app%2F&demo-image=https://assets.vercel.com/image/upload/v1669994241/random/django.png)
