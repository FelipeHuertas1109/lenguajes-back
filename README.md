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

Este endpoint recibe un archivo de texto (`.txt`) o CSV (`.csv`) con expresiones regulares (una por línea) y genera un archivo CSV con toda la información del DFA más una columna adicional llamada "Clase" que contiene un diccionario JSON con 100 cadenas de prueba (50 aceptadas y 50 rechazadas) y sus valores booleanos.

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
  ...
  "__REJECTED_rej_0_45__": false,
  "__REJECTED_rej_1_45__": false
}
```

**Características de las cadenas generadas:**
- Las cadenas aceptadas son generadas probando diferentes combinaciones del alfabeto
- Las cadenas rechazadas incluyen cadenas con símbolos fuera del alfabeto (garantizadas como rechazadas)
- Todas las cadenas son únicas en el diccionario
- El diccionario siempre contiene exactamente 100 entradas

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
