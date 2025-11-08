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

### Notas Importantes

- **Hosts permitidos**: El backend acepta peticiones desde `localhost`, `127.0.0.1` y dominios `.vercel.app`
- **URL del endpoint**: Asegúrate de usar la URL completa con la barra final: `/api/regex-to-dfa/`
- **CORS**: Está habilitado para todos los orígenes, por lo que el frontend puede hacer peticiones sin problemas

## Running Locally

```bash
python manage.py runserver
```

Your Django application is now available at `http://localhost:8000`.

## One-Click Deploy

Deploy the example using [Vercel](https://vercel.com?utm_source=github&utm_medium=readme&utm_campaign=vercel-examples):

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https%3A%2F%2Fgithub.com%2Fvercel%2Fexamples%2Ftree%2Fmain%2Fpython%2Fdjango&demo-title=Django%20%2B%20Vercel&demo-description=Use%20Django%204%20on%20Vercel%20with%20Serverless%20Functions%20using%20the%20Python%20Runtime.&demo-url=https%3A%2F%2Fdjango-template.vercel.app%2F&demo-image=https://assets.vercel.com/image/upload/v1669994241/random/django.png)
