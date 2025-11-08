# example/views.py
from datetime import datetime
import json
import sys

from django.http import HttpResponse, JsonResponse
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt
import re

from example.thompson_nfa import (
    regex_to_nfa,
    nfa_to_dfa,
    alias_dfa,
    get_dfa_alphabet,
    get_dfa_states,
    dfa_accepts,
    dfa_to_jff_string
)


def index(request):
    print("=" * 60)
    print(f"[INDEX] Petición recibida - {datetime.now()}")
    print(f"[INDEX] Método: {request.method}")
    print(f"[INDEX] IP: {request.META.get('REMOTE_ADDR', 'Unknown')}")
    print(f"[INDEX] User-Agent: {request.META.get('HTTP_USER_AGENT', 'Unknown')}")
    
    now = datetime.now()
    html = f'''
    <html>
        <body>
            <h1>Hello from Vercel!</h1>
            <p>The current time is { now }.</p>
        </body>
    </html>
    '''
    
    print("[INDEX] --- RESPUESTA HTML ---")
    print(f"[INDEX] Tamaño: ~{len(html)} bytes")
    print(f"[INDEX] Contenido (primeros 200 chars): {html[:200]}...")
    print("[INDEX] --- FIN RESPUESTA ---")
    print("=" * 60)
    sys.stdout.flush()  # Forzar escritura inmediata
    
    return HttpResponse(html)


def dfa_to_json(dfa):
    """Convierte un DFA a un diccionario JSON serializable."""
    aliases, rev = alias_dfa(dfa)
    
    # Alfabeto
    alphabet = sorted(get_dfa_alphabet(dfa))
    
    # Estados: mantener el orden establecido en alias_dfa (S0 es el inicial)
    # Obtener todos los estados y ordenarlos según sus aliases (S0, S1, S2, ...)
    all_states = get_dfa_states(dfa)
    # Ordenar por el número del alias (S0 < S1 < S2 ...)
    state_names = sorted([aliases[s] for s in all_states], 
                        key=lambda alias: int(alias[1:]) if alias[1:].isdigit() else 999)
    
    # Estado inicial (ahora siempre será S0)
    start = aliases[dfa.start]
    
    # Estados de aceptación: ordenarlos también por su número de alias
    accepts = sorted([aliases[s] for s in dfa.accepts],
                    key=lambda alias: int(alias[1:]) if alias[1:].isdigit() else 999)
    
    # Transiciones: ordenar por el estado origen (S0 primero, luego S1, etc.)
    transitions = []
    # Ordenar las fuentes por su alias
    sorted_sources = sorted(dfa.trans.keys(), 
                           key=lambda s: int(aliases[s][1:]) if aliases[s][1:].isdigit() else 999)
    for src in sorted_sources:
        for sym, dst in sorted(dfa.trans[src].items(), key=lambda x: x[0]):
            transitions.append({
                "from": aliases[src],
                "symbol": sym,
                "to": aliases[dst]
            })
    
    return {
        "alphabet": alphabet,
        "states": state_names,
        "start": start,
        "accepting": accepts,
        "transitions": transitions
    }


@csrf_exempt
@require_http_methods(["GET", "POST"])
def regex_to_dfa(request):
    """
    Endpoint que convierte una expresión regular a DFA.
    
    Parámetros:
    - GET: ?regex=<expresion>
    - POST: {"regex": "<expresion>"} o {"regex": "<expresion>", "test": "<cadena>"}
    
    Retorna JSON con:
    {
        "success": true/false,
        "regex": "<expresion>",
        "dfa": {
            "alphabet": [...],
            "states": [...],
            "start": "S0",
            "accepting": [...],
            "transitions": [...]
        },
        "test_result": null o {"string": "...", "accepted": true/false},
        "error": null o "mensaje de error"
    }
    """
    # ========== LOGS DE ENTRADA ==========
    print("=" * 80)
    print(f"[REGEX_TO_DFA] Petición recibida - {datetime.now()}")
    print(f"[REGEX_TO_DFA] Método HTTP: {request.method}")
    print(f"[REGEX_TO_DFA] IP Cliente: {request.META.get('REMOTE_ADDR', 'Unknown')}")
    print(f"[REGEX_TO_DFA] User-Agent: {request.META.get('HTTP_USER_AGENT', 'Unknown')}")
    print(f"[REGEX_TO_DFA] Origin: {request.META.get('HTTP_ORIGIN', 'None')}")
    print(f"[REGEX_TO_DFA] Referer: {request.META.get('HTTP_REFERER', 'None')}")
    print(f"[REGEX_TO_DFA] Path: {request.path}")
    print(f"[REGEX_TO_DFA] Query String: {request.META.get('QUERY_STRING', 'None')}")
    sys.stdout.flush()
    
    regex = None
    test_string = None
    
    if request.method == "GET":
        regex = request.GET.get("regex")
        test_string = request.GET.get("test")
        print(f"[REGEX_TO_DFA] GET - regex={repr(regex)}, test={repr(test_string)}")
        sys.stdout.flush()
    else:  # POST
        try:
            body_str = request.body.decode('utf-8') if request.body else '{}'
            print(f"[REGEX_TO_DFA] POST Body (raw): {body_str[:200]}...")  # Primeros 200 chars
            sys.stdout.flush()
            
            data = json.loads(body_str)
            regex = data.get("regex")
            test_string = data.get("test")
            print(f"[REGEX_TO_DFA] POST - regex={repr(regex)}, test={repr(test_string)}")
            sys.stdout.flush()
        except json.JSONDecodeError as e:
            print(f"[REGEX_TO_DFA] ERROR - JSON inválido: {e}")
            sys.stdout.flush()
            error_response = {
                "success": False,
                "error": "JSON inválido en el cuerpo de la petición"
            }
            print("[REGEX_TO_DFA] --- RESPUESTA DE ERROR ---")
            print(json.dumps(error_response, ensure_ascii=False, indent=2))
            print("[REGEX_TO_DFA] --- FIN RESPUESTA ---")
            print("=" * 80)
            sys.stdout.flush()
            return JsonResponse(error_response, status=400)
    
    if not regex:
        print("[REGEX_TO_DFA] ERROR - Parámetro 'regex' faltante")
        sys.stdout.flush()
        error_response = {
            "success": False,
            "error": "Parámetro 'regex' requerido"
        }
        print("[REGEX_TO_DFA] --- RESPUESTA DE ERROR ---")
        print(json.dumps(error_response, ensure_ascii=False, indent=2))
        print("[REGEX_TO_DFA] --- FIN RESPUESTA ---")
        print("=" * 80)
        sys.stdout.flush()
        return JsonResponse(error_response, status=400)
    
    try:
        print(f"[REGEX_TO_DFA] Iniciando conversión - Regex: {repr(regex)}")
        sys.stdout.flush()
        
        # Convertir regex a NFA y luego a DFA
        print("[REGEX_TO_DFA] Paso 1: Convirtiendo Regex -> NFA (Thompson)")
        sys.stdout.flush()
        nfa = regex_to_nfa(regex)
        print(f"[REGEX_TO_DFA] NFA creado - Estados: {len(nfa.states)}, Aceptación: {nfa.accepts}")
        sys.stdout.flush()
        
        print("[REGEX_TO_DFA] Paso 2: Convirtiendo NFA -> DFA (Subconjuntos)")
        sys.stdout.flush()
        dfa = nfa_to_dfa(nfa)
        print(f"[REGEX_TO_DFA] DFA creado - Estados: {len(dfa.trans)}, Aceptación: {len(dfa.accepts)}")
        sys.stdout.flush()
        
        # Serializar DFA a JSON
        print("[REGEX_TO_DFA] Paso 3: Serializando DFA a JSON")
        sys.stdout.flush()
        dfa_json = dfa_to_json(dfa)
        print(f"[REGEX_TO_DFA] DFA serializado - Alfabeto: {dfa_json['alphabet']}, Estados: {len(dfa_json['states'])}")
        sys.stdout.flush()
        
        # Si se proporcionó una cadena de prueba, verificar si es aceptada
        test_result = None
        if test_string is not None:
            print(f"[REGEX_TO_DFA] Probando cadena: {repr(test_string)}")
            sys.stdout.flush()
            accepted = dfa_accepts(dfa, test_string)
            test_result = {
                "string": test_string,
                "accepted": accepted
            }
            print(f"[REGEX_TO_DFA] Resultado de prueba: {'ACEPTADA' if accepted else 'RECHAZADA'}")
            sys.stdout.flush()
        
        response_data = {
            "success": True,
            "regex": regex,
            "dfa": dfa_json,
            "test_result": test_result,
            "error": None
        }
        
        # Preparar respuesta JSON para logging
        response_json_str = json.dumps(response_data, ensure_ascii=False, indent=2)
        response_size = len(response_json_str.encode('utf-8'))
        
        print("[REGEX_TO_DFA] ÉXITO - Respuesta generada correctamente")
        print(f"[REGEX_TO_DFA] Tamaño de respuesta: ~{response_size} bytes")
        print("[REGEX_TO_DFA] --- RESUMEN DE RESPUESTA ---")
        print(f"[REGEX_TO_DFA] success: {response_data['success']}")
        print(f"[REGEX_TO_DFA] regex: {response_data['regex']}")
        print(f"[REGEX_TO_DFA] DFA - Alfabeto: {dfa_json['alphabet']}")
        print(f"[REGEX_TO_DFA] DFA - Estados totales: {len(dfa_json['states'])}")
        print(f"[REGEX_TO_DFA] DFA - Estados: {dfa_json['states']}")
        print(f"[REGEX_TO_DFA] DFA - Estado inicial: {dfa_json['start']}")
        print(f"[REGEX_TO_DFA] DFA - Estados de aceptación: {dfa_json['accepting']}")
        print(f"[REGEX_TO_DFA] DFA - Transiciones: {len(dfa_json['transitions'])} transiciones")
        if test_result:
            print(f"[REGEX_TO_DFA] test_result: cadena='{test_result['string']}', accepted={test_result['accepted']}")
        else:
            print(f"[REGEX_TO_DFA] test_result: null")
        print(f"[REGEX_TO_DFA] error: {response_data['error']}")
        print("[REGEX_TO_DFA] --- RESPUESTA JSON COMPLETA ---")
        print(response_json_str)
        print("[REGEX_TO_DFA] --- FIN RESPUESTA ---")
        print("=" * 80)
        sys.stdout.flush()
        
        return JsonResponse(response_data, json_dumps_params={"ensure_ascii": False})
    
    except Exception as e:
        import traceback
        print(f"[REGEX_TO_DFA] ERROR - Excepción capturada:")
        print(f"[REGEX_TO_DFA] Tipo: {type(e).__name__}")
        print(f"[REGEX_TO_DFA] Mensaje: {str(e)}")
        print(f"[REGEX_TO_DFA] Traceback:")
        traceback.print_exc()
        sys.stdout.flush()
        
        error_response = {
            "success": False,
            "regex": regex,
            "dfa": None,
            "test_result": None,
            "error": str(e)
        }
        print("[REGEX_TO_DFA] --- RESPUESTA DE ERROR ---")
        print(json.dumps(error_response, ensure_ascii=False, indent=2))
        print("[REGEX_TO_DFA] --- FIN RESPUESTA ---")
        print("=" * 80)
        sys.stdout.flush()
        
        return JsonResponse(error_response, status=400)


@csrf_exempt
@require_http_methods(["GET", "POST"])
def regex_to_dfa_jff(request):
    """
    Endpoint que convierte una expresión regular a DFA y lo devuelve en formato JFLAP (.jff).
    
    Parámetros:
    - GET: ?regex=<expresion>
    - POST: {"regex": "<expresion>"}
    
    Retorna un archivo .jff descargable compatible con JFLAP.
    """
    # ========== LOGS DE ENTRADA ==========
    print("=" * 80)
    print(f"[REGEX_TO_DFA_JFF] Petición recibida - {datetime.now()}")
    print(f"[REGEX_TO_DFA_JFF] Método HTTP: {request.method}")
    print(f"[REGEX_TO_DFA_JFF] IP Cliente: {request.META.get('REMOTE_ADDR', 'Unknown')}")
    print(f"[REGEX_TO_DFA_JFF] User-Agent: {request.META.get('HTTP_USER_AGENT', 'Unknown')}")
    print(f"[REGEX_TO_DFA_JFF] Origin: {request.META.get('HTTP_ORIGIN', 'None')}")
    print(f"[REGEX_TO_DFA_JFF] Path: {request.path}")
    print(f"[REGEX_TO_DFA_JFF] Query String: {request.META.get('QUERY_STRING', 'None')}")
    sys.stdout.flush()
    
    regex = None
    
    if request.method == "GET":
        regex = request.GET.get("regex")
        print(f"[REGEX_TO_DFA_JFF] GET - regex={repr(regex)}")
        sys.stdout.flush()
    else:  # POST
        try:
            body_str = request.body.decode('utf-8') if request.body else '{}'
            print(f"[REGEX_TO_DFA_JFF] POST Body (raw): {body_str[:200]}...")
            sys.stdout.flush()
            
            data = json.loads(body_str)
            regex = data.get("regex")
            print(f"[REGEX_TO_DFA_JFF] POST - regex={repr(regex)}")
            sys.stdout.flush()
        except json.JSONDecodeError as e:
            print(f"[REGEX_TO_DFA_JFF] ERROR - JSON inválido: {e}")
            sys.stdout.flush()
            return JsonResponse({
                "success": False,
                "error": "JSON inválido en el cuerpo de la petición"
            }, status=400)
    
    if not regex:
        print("[REGEX_TO_DFA_JFF] ERROR - Parámetro 'regex' faltante")
        sys.stdout.flush()
        return JsonResponse({
            "success": False,
            "error": "Parámetro 'regex' requerido"
        }, status=400)
    
    try:
        print(f"[REGEX_TO_DFA_JFF] Iniciando conversión - Regex: {repr(regex)}")
        sys.stdout.flush()
        
        # Convertir regex a NFA y luego a DFA
        print("[REGEX_TO_DFA_JFF] Paso 1: Convirtiendo Regex -> NFA (Thompson)")
        sys.stdout.flush()
        nfa = regex_to_nfa(regex)
        print(f"[REGEX_TO_DFA_JFF] NFA creado - Estados: {len(nfa.states)}, Aceptación: {nfa.accepts}")
        sys.stdout.flush()
        
        print("[REGEX_TO_DFA_JFF] Paso 2: Convirtiendo NFA -> DFA (Subconjuntos)")
        sys.stdout.flush()
        dfa = nfa_to_dfa(nfa)
        print(f"[REGEX_TO_DFA_JFF] DFA creado - Estados: {len(dfa.trans)}, Aceptación: {len(dfa.accepts)}")
        sys.stdout.flush()
        
        # Convertir DFA a formato JFF
        print("[REGEX_TO_DFA_JFF] Paso 3: Convirtiendo DFA a formato JFF")
        sys.stdout.flush()
        jff_content = dfa_to_jff_string(dfa)
        jff_size = len(jff_content.encode('utf-8'))
        print(f"[REGEX_TO_DFA_JFF] Archivo JFF generado - Tamaño: ~{jff_size} bytes")
        sys.stdout.flush()
        
        # Generar nombre de archivo seguro desde la regex
        # Reemplazar caracteres especiales por guiones bajos
        safe_filename = re.sub(r'[^\w\s-]', '_', regex)
        safe_filename = re.sub(r'[-\s]+', '_', safe_filename)
        safe_filename = safe_filename[:50]  # Limitar longitud
        filename = f"dfa_{safe_filename}.jff" if safe_filename else "dfa.jff"
        
        print(f"[REGEX_TO_DFA_JFF] Nombre de archivo: {filename}")
        print(f"[REGEX_TO_DFA_JFF] --- CONTENIDO JFF (primeros 500 chars) ---")
        print(jff_content[:500])
        print("[REGEX_TO_DFA_JFF] --- FIN CONTENIDO ---")
        print("[REGEX_TO_DFA_JFF] ÉXITO - Archivo JFF generado correctamente")
        print("=" * 80)
        sys.stdout.flush()
        
        # Crear respuesta HTTP con el archivo
        response = HttpResponse(jff_content, content_type='application/xml')
        response['Content-Disposition'] = f'attachment; filename="{filename}"'
        response['Content-Length'] = jff_size
        return response
    
    except Exception as e:
        import traceback
        print(f"[REGEX_TO_DFA_JFF] ERROR - Excepción capturada:")
        print(f"[REGEX_TO_DFA_JFF] Tipo: {type(e).__name__}")
        print(f"[REGEX_TO_DFA_JFF] Mensaje: {str(e)}")
        print(f"[REGEX_TO_DFA_JFF] Traceback:")
        traceback.print_exc()
        sys.stdout.flush()
        
        error_response = {
            "success": False,
            "regex": regex,
            "error": str(e)
        }
        print("[REGEX_TO_DFA_JFF] --- RESPUESTA DE ERROR ---")
        print(json.dumps(error_response, ensure_ascii=False, indent=2))
        print("[REGEX_TO_DFA_JFF] --- FIN RESPUESTA ---")
        print("=" * 80)
        sys.stdout.flush()
        
        return JsonResponse(error_response, status=400)