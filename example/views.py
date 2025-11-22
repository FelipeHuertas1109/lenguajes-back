# example/views.py
from datetime import datetime
import json
import sys
import os
import tempfile
from urllib.parse import quote

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
    dfa_to_jff_string,
    process_regex_file_to_csv_with_clase,
    transitions_to_dfa
)
from example.alphabetnet_model import predict_alphabet
from example.acepnet_service import get_acepnet_service
import pandas as pd
from pathlib import Path


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
    - GET: ?regex=<expresion>&test=cadena1&test=cadena2 o ?regex=<expresion>&tests=cadena1,cadena2
    - POST: {"regex": "<expresion>"} o {"regex": "<expresion>", "test": "<cadena>"} o {"regex": "<expresion>", "tests": ["cadena1", "cadena2"]}
    
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
        "test_results": null o [{"string": "...", "accepted": true/false}, ...],
        "error": null o "mensaje de error"
    }
    
    Nota: Puede recibir múltiples cadenas para probar:
    - GET: ?test=cadena1&test=cadena2 o ?tests=cadena1,cadena2
    - POST: {"test": "cadena"} (una cadena) o {"tests": ["cadena1", "cadena2"]} (múltiples)
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
    test_strings = []  # Lista de cadenas a probar
    
    if request.method == "GET":
        regex = request.GET.get("regex")
        # GET puede tener múltiples parámetros "test" o un parámetro "tests" separado por comas
        test_list = request.GET.getlist("test")  # Obtener todos los parámetros "test"
        if test_list:
            test_strings = test_list
        else:
            # Intentar obtener "tests" como cadena separada por comas
            tests_str = request.GET.get("tests")
            if tests_str:
                test_strings = [s.strip() for s in tests_str.split(",") if s.strip()]
        print(f"[REGEX_TO_DFA] GET - regex={repr(regex)}, tests={test_strings}")
        sys.stdout.flush()
    else:  # POST
        try:
            body_str = request.body.decode('utf-8') if request.body else '{}'
            print(f"[REGEX_TO_DFA] POST Body (raw): {body_str[:200]}...")  # Primeros 200 chars
            sys.stdout.flush()
            
            data = json.loads(body_str)
            regex = data.get("regex")
            # POST puede tener "test" (cadena única, para compatibilidad) o "tests" (array)
            if "tests" in data:
                # Array de cadenas
                tests_data = data.get("tests")
                if isinstance(tests_data, list):
                    test_strings = [str(s) for s in tests_data]
                elif isinstance(tests_data, str):
                    # Si es una cadena, separarla por comas
                    test_strings = [s.strip() for s in tests_data.split(",") if s.strip()]
            elif "test" in data:
                # Cadena única (compatibilidad hacia atrás)
                test_str = data.get("test")
                if test_str is not None:
                    if isinstance(test_str, list):
                        test_strings = [str(s) for s in test_str]
                    else:
                        test_strings = [str(test_str)]
            print(f"[REGEX_TO_DFA] POST - regex={repr(regex)}, tests={test_strings}")
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
        
        # Probar cadenas si se proporcionaron
        test_results = None
        if test_strings:
            print(f"[REGEX_TO_DFA] Probando {len(test_strings)} cadena(s): {test_strings}")
            sys.stdout.flush()
            test_results = []
            for test_str in test_strings:
                accepted = dfa_accepts(dfa, test_str)
                test_results.append({
                    "string": test_str,
                    "accepted": accepted
                })
                print(f"[REGEX_TO_DFA] Cadena '{test_str}': {'ACEPTADA' if accepted else 'RECHAZADA'}")
            sys.stdout.flush()
        
        response_data = {
            "success": True,
            "regex": regex,
            "dfa": dfa_json,
            "test_results": test_results,  # Ahora es una lista o None
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
        if test_results:
            print(f"[REGEX_TO_DFA] test_results: {len(test_results)} resultado(s)")
            for i, tr in enumerate(test_results):
                print(f"[REGEX_TO_DFA]   [{i}] cadena='{tr['string']}', accepted={tr['accepted']}")
        else:
            print(f"[REGEX_TO_DFA] test_results: null")
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
            "test_results": None,
            "error": str(e)
        }
        print("[REGEX_TO_DFA] --- RESPUESTA DE ERROR ---")
        print(json.dumps(error_response, ensure_ascii=False, indent=2))
        print("[REGEX_TO_DFA] --- FIN RESPUESTA ---")
        print("=" * 80)
        sys.stdout.flush()
        
        return JsonResponse(error_response, status=400)


@csrf_exempt
@require_http_methods(["POST"])
def transitions_to_dfa_endpoint(request):
    """
    Endpoint que construye un DFA desde transiciones y devuelve la información del DFA.
    
    Parámetros (POST):
    {
        "states": ["S0", "S1", "S2"],           // Lista de estados
        "start": "S0",                          // Estado inicial
        "accepting": ["S2"],                    // Estados de aceptación
        "transitions": [                        // Lista de transiciones
            {"from": "S0", "symbol": "a", "to": "S1"},
            {"from": "S1", "symbol": "b", "to": "S2"},
            {"from": "S0", "symbol": "b", "to": "S0"}
        ],
        "test": "ab",                           // (Opcional) Cadena única para probar (compatibilidad)
        "tests": ["ab", "ba", "a"]              // (Opcional) Array de cadenas para probar
    }
    
    Retorna JSON con:
    {
        "success": true/false,
        "dfa": {
            "alphabet": [...],
            "states": [...],
            "start": "S0",
            "accepting": [...],
            "transitions": [...]
        },
        "test_results": null o [{"string": "...", "accepted": true/false}, ...],
        "error": null o "mensaje de error"
    }
    
    Nota: Puede recibir múltiples cadenas para probar usando "test" (una cadena) o "tests" (array).
    """
    # ========== LOGS DE ENTRADA ==========
    print("=" * 80)
    print(f"[TRANSITIONS_TO_DFA] Petición recibida - {datetime.now()}")
    print(f"[TRANSITIONS_TO_DFA] Método HTTP: {request.method}")
    print(f"[TRANSITIONS_TO_DFA] IP Cliente: {request.META.get('REMOTE_ADDR', 'Unknown')}")
    print(f"[TRANSITIONS_TO_DFA] User-Agent: {request.META.get('HTTP_USER_AGENT', 'Unknown')}")
    print(f"[TRANSITIONS_TO_DFA] Origin: {request.META.get('HTTP_ORIGIN', 'None')}")
    print(f"[TRANSITIONS_TO_DFA] Path: {request.path}")
    sys.stdout.flush()
    
    # Solo aceptar POST
    if request.method != "POST":
        print("[TRANSITIONS_TO_DFA] ERROR - Método no permitido. Solo POST está permitido.")
        sys.stdout.flush()
        return JsonResponse({
            "success": False,
            "error": "Método no permitido. Use POST."
        }, status=405)
    
    # Parsear JSON del cuerpo
    try:
        body_str = request.body.decode('utf-8') if request.body else '{}'
        print(f"[TRANSITIONS_TO_DFA] POST Body (raw): {body_str[:500]}...")  # Primeros 500 chars
        sys.stdout.flush()
        
        data = json.loads(body_str)
        states = data.get("states")
        start = data.get("start")
        accepting = data.get("accepting")
        transitions = data.get("transitions")
        
        # Soporte para múltiples cadenas de prueba
        test_strings = []  # Lista de cadenas a probar
        if "tests" in data:
            # Array de cadenas
            tests_data = data.get("tests")
            if isinstance(tests_data, list):
                test_strings = [str(s) for s in tests_data]
            elif isinstance(tests_data, str):
                # Si es una cadena, separarla por comas
                test_strings = [s.strip() for s in tests_data.split(",") if s.strip()]
        elif "test" in data:
            # Cadena única (compatibilidad hacia atrás) o lista
            test_data = data.get("test")
            if test_data is not None:
                if isinstance(test_data, list):
                    test_strings = [str(s) for s in test_data]
                else:
                    test_strings = [str(test_data)]
        
        print(f"[TRANSITIONS_TO_DFA] POST - states={states}, start={start}, accepting={accepting}, transitions_count={len(transitions) if transitions else 0}, tests={test_strings}")
        sys.stdout.flush()
    except json.JSONDecodeError as e:
        print(f"[TRANSITIONS_TO_DFA] ERROR - JSON inválido: {e}")
        sys.stdout.flush()
        error_response = {
            "success": False,
            "error": "JSON inválido en el cuerpo de la petición"
        }
        print("[TRANSITIONS_TO_DFA] --- RESPUESTA DE ERROR ---")
        print(json.dumps(error_response, ensure_ascii=False, indent=2))
        print("[TRANSITIONS_TO_DFA] --- FIN RESPUESTA ---")
        print("=" * 80)
        sys.stdout.flush()
        return JsonResponse(error_response, status=400)
    
    # Validar parámetros requeridos
    if not states:
        print("[TRANSITIONS_TO_DFA] ERROR - Parámetro 'states' faltante")
        sys.stdout.flush()
        error_response = {
            "success": False,
            "error": "Parámetro 'states' requerido (lista de estados)"
        }
        return JsonResponse(error_response, status=400)
    
    if not start:
        print("[TRANSITIONS_TO_DFA] ERROR - Parámetro 'start' faltante")
        sys.stdout.flush()
        error_response = {
            "success": False,
            "error": "Parámetro 'start' requerido (estado inicial)"
        }
        return JsonResponse(error_response, status=400)
    
    if accepting is None:
        print("[TRANSITIONS_TO_DFA] ERROR - Parámetro 'accepting' faltante")
        sys.stdout.flush()
        error_response = {
            "success": False,
            "error": "Parámetro 'accepting' requerido (lista de estados de aceptación, puede ser [])"
        }
        return JsonResponse(error_response, status=400)
    
    if not transitions:
        print("[TRANSITIONS_TO_DFA] ERROR - Parámetro 'transitions' faltante")
        sys.stdout.flush()
        error_response = {
            "success": False,
            "error": "Parámetro 'transitions' requerido (lista de transiciones)"
        }
        return JsonResponse(error_response, status=400)
    
    try:
        print(f"[TRANSITIONS_TO_DFA] Iniciando construcción de DFA desde transiciones")
        sys.stdout.flush()
        
        # Construir DFA desde transiciones
        print("[TRANSITIONS_TO_DFA] Construyendo DFA desde transiciones...")
        sys.stdout.flush()
        dfa = transitions_to_dfa(states, start, accepting, transitions)
        print(f"[TRANSITIONS_TO_DFA] DFA construido - Estados: {len(dfa.trans)}, Aceptación: {len(dfa.accepts)}")
        sys.stdout.flush()
        
        # Convertir DFA a JSON
        print("[TRANSITIONS_TO_DFA] Convirtiendo DFA a formato JSON...")
        sys.stdout.flush()
        dfa_json = dfa_to_json(dfa)
        
        # Probar cadenas si se proporcionaron
        test_results = None
        if test_strings:
            print(f"[TRANSITIONS_TO_DFA] Probando {len(test_strings)} cadena(s): {test_strings}")
            sys.stdout.flush()
            test_results = []
            for test_str in test_strings:
                accepted = dfa_accepts(dfa, test_str)
                test_results.append({
                    "string": test_str,
                    "accepted": accepted
                })
                print(f"[TRANSITIONS_TO_DFA] Cadena '{test_str}': {'ACEPTADA' if accepted else 'RECHAZADA'}")
            sys.stdout.flush()
        
        # Construir respuesta exitosa
        response_data = {
            "success": True,
            "dfa": dfa_json,
            "test_results": test_results,  # Ahora es una lista o None
            "error": None
        }
        
        print("[TRANSITIONS_TO_DFA] --- RESPUESTA ---")
        print(json.dumps(response_data, ensure_ascii=False, indent=2)[:1000] + "...")
        print("[TRANSITIONS_TO_DFA] --- FIN RESPUESTA ---")
        print("[TRANSITIONS_TO_DFA] ÉXITO - DFA construido correctamente")
        print("=" * 80)
        sys.stdout.flush()
        
        return JsonResponse(response_data)
    
    except ValueError as e:
        # Errores de validación
        print(f"[TRANSITIONS_TO_DFA] ERROR - Error de validación: {e}")
        sys.stdout.flush()
        error_response = {
            "success": False,
            "dfa": None,
            "test_results": None,
            "error": str(e)
        }
        print("[TRANSITIONS_TO_DFA] --- RESPUESTA DE ERROR ---")
        print(json.dumps(error_response, ensure_ascii=False, indent=2))
        print("[TRANSITIONS_TO_DFA] --- FIN RESPUESTA ---")
        print("=" * 80)
        sys.stdout.flush()
        return JsonResponse(error_response, status=400)
    
    except Exception as e:
        import traceback
        print(f"[TRANSITIONS_TO_DFA] ERROR - Excepción capturada:")
        print(f"[TRANSITIONS_TO_DFA] Tipo: {type(e).__name__}")
        print(f"[TRANSITIONS_TO_DFA] Mensaje: {str(e)}")
        print(f"[TRANSITIONS_TO_DFA] Traceback:")
        traceback.print_exc()
        sys.stdout.flush()
        
        error_response = {
            "success": False,
            "dfa": None,
            "test_results": None,
            "error": str(e)
        }
        print("[TRANSITIONS_TO_DFA] --- RESPUESTA DE ERROR ---")
        print(json.dumps(error_response, ensure_ascii=False, indent=2))
        print("[TRANSITIONS_TO_DFA] --- FIN RESPUESTA ---")
        print("=" * 80)
        sys.stdout.flush()
        
        return JsonResponse(error_response, status=500)


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
        # Usar 'application/xml' o 'text/xml' para archivos JFF
        # JFLAP reconoce ambos tipos MIME
        response = HttpResponse(jff_content, content_type='application/xml; charset=utf-8')
        
        # Codificar el nombre del archivo correctamente para Content-Disposition
        # Usar formato RFC 5987 para nombres de archivo con caracteres especiales
        encoded_filename = quote(filename, safe='')
        response['Content-Disposition'] = f'attachment; filename="{filename}"; filename*=UTF-8\'\'{encoded_filename}'
        response['Content-Length'] = jff_size
        response['Access-Control-Expose-Headers'] = 'Content-Disposition, Content-Length, Content-Type'
        
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


@csrf_exempt
@require_http_methods(["POST"])
def regex_file_to_csv(request):
    """
    Endpoint que recibe un archivo txt o csv con expresiones regulares
    y genera un CSV con los datos del DFA más una columna "Clase" que contiene
    un diccionario JSON con 100 cadenas (50 aceptadas, 50 rechazadas) y sus valores.
    
    Parámetros:
    - POST: archivo en el campo 'file'
    
    Retorna un archivo CSV descargable.
    """
    # ========== LOGS DE ENTRADA ==========
    print("=" * 80)
    print(f"[REGEX_FILE_TO_CSV] Petición recibida - {datetime.now()}")
    print(f"[REGEX_FILE_TO_CSV] Método HTTP: {request.method}")
    print(f"[REGEX_FILE_TO_CSV] IP Cliente: {request.META.get('REMOTE_ADDR', 'Unknown')}")
    print(f"[REGEX_FILE_TO_CSV] User-Agent: {request.META.get('HTTP_USER_AGENT', 'Unknown')}")
    sys.stdout.flush()
    
    # Verificar que se envió un archivo
    if 'file' not in request.FILES:
        print("[REGEX_FILE_TO_CSV] ERROR - No se proporcionó archivo")
        sys.stdout.flush()
        return JsonResponse({
            "success": False,
            "error": "No se proporcionó archivo. Use el campo 'file' en el formulario."
        }, status=400)
    
    uploaded_file = request.FILES['file']
    file_name = uploaded_file.name
    file_extension = os.path.splitext(file_name)[1].lower()
    
    print(f"[REGEX_FILE_TO_CSV] Archivo recibido: {file_name}")
    print(f"[REGEX_FILE_TO_CSV] Extensión: {file_extension}")
    sys.stdout.flush()
    
    # Validar extensión
    if file_extension not in ['.txt', '.csv']:
        print(f"[REGEX_FILE_TO_CSV] ERROR - Extensión no válida: {file_extension}")
        sys.stdout.flush()
        return JsonResponse({
            "success": False,
            "error": f"Formato de archivo no soportado. Use .txt o .csv (recibido: {file_extension})"
        }, status=400)
    
    try:
        # Crear archivos temporales
        with tempfile.NamedTemporaryFile(mode='w+b', delete=False, suffix=file_extension) as temp_input:
            temp_input_path = temp_input.name
            # Escribir el contenido del archivo subido
            for chunk in uploaded_file.chunks():
                temp_input.write(chunk)
        
        print(f"[REGEX_FILE_TO_CSV] Archivo temporal creado: {temp_input_path}")
        sys.stdout.flush()
        
        # Crear archivo de salida temporal
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv', encoding='utf-8') as temp_output:
            temp_output_path = temp_output.name
        
        print(f"[REGEX_FILE_TO_CSV] Procesando archivo...")
        sys.stdout.flush()
        
        # OPTIMIZACIÓN: Procesar en paralelo con verbose=False para mejor rendimiento
        # La función ahora procesa en paralelo automáticamente
        process_regex_file_to_csv_with_clase(temp_input_path, temp_output_path, verbose=False)
        
        print(f"[REGEX_FILE_TO_CSV] Archivo procesado exitosamente")
        print(f"[REGEX_FILE_TO_CSV] Archivo de salida: {temp_output_path}")
        sys.stdout.flush()
        
        # Leer el contenido del CSV generado
        with open(temp_output_path, 'r', encoding='utf-8') as f:
            csv_content = f.read()
        
        csv_size = len(csv_content.encode('utf-8'))
        print(f"[REGEX_FILE_TO_CSV] Tamaño del CSV generado: ~{csv_size} bytes")
        print("[REGEX_FILE_TO_CSV] ÉXITO - CSV generado correctamente")
        print("=" * 80)
        sys.stdout.flush()
        
        # Limpiar archivos temporales
        try:
            os.unlink(temp_input_path)
            os.unlink(temp_output_path)
        except Exception as e:
            print(f"[REGEX_FILE_TO_CSV] Advertencia - Error al limpiar archivos temporales: {e}")
            sys.stdout.flush()
        
        # Generar nombre de archivo de salida
        output_filename = f"regex_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        # Crear respuesta HTTP con el archivo CSV
        response = HttpResponse(csv_content, content_type='text/csv; charset=utf-8')
        response['Content-Disposition'] = f'attachment; filename="{output_filename}"'
        response['Content-Length'] = csv_size
        return response
    
    except FileNotFoundError as e:
        print(f"[REGEX_FILE_TO_CSV] ERROR - Archivo no encontrado: {e}")
        sys.stdout.flush()
        error_response = {
            "success": False,
            "error": f"Error al procesar el archivo: {str(e)}"
        }
        return JsonResponse(error_response, status=400)
    
    except Exception as e:
        import traceback
        print(f"[REGEX_FILE_TO_CSV] ERROR - Excepción capturada:")
        print(f"[REGEX_FILE_TO_CSV] Tipo: {type(e).__name__}")
        print(f"[REGEX_FILE_TO_CSV] Mensaje: {str(e)}")
        print(f"[REGEX_FILE_TO_CSV] Traceback:")
        traceback.print_exc()
        sys.stdout.flush()
        
        # Limpiar archivos temporales en caso de error
        try:
            if 'temp_input_path' in locals():
                os.unlink(temp_input_path)
            if 'temp_output_path' in locals():
                os.unlink(temp_output_path)
        except Exception:
            pass
        
        error_response = {
            "success": False,
            "error": str(e)
        }
        print("[REGEX_FILE_TO_CSV] --- RESPUESTA DE ERROR ---")
        print(json.dumps(error_response, ensure_ascii=False, indent=2))
        print("[REGEX_FILE_TO_CSV] --- FIN RESPUESTA ---")
        print("=" * 80)
        sys.stdout.flush()
        
        return JsonResponse(error_response, status=400)


@csrf_exempt
@require_http_methods(["GET", "POST"])
def regex_to_alphabet(request):
    """
    Endpoint que predice el alfabeto de una expresión regular usando el modelo AlphabetNet.
    
    Parámetros:
    - GET: ?regex=<expresion>
    - POST: {"regex": "<expresion>"}
    
    Retorna JSON con:
    {
        "success": true/false,
        "regex": "<expresion>",
        "alphabet": ["A", "B", "C", ...],  // Alfabeto predicho (sigma_hat)
        "probabilities": {                  // Probabilidades por símbolo
            "A": 0.95,
            "B": 0.87,
            ...
        },
        "error": null o "mensaje de error"
    }
    """
    # ========== LOGS DE ENTRADA ==========
    print("=" * 80)
    print(f"[REGEX_TO_ALPHABET] Petición recibida - {datetime.now()}")
    print(f"[REGEX_TO_ALPHABET] Método HTTP: {request.method}")
    print(f"[REGEX_TO_ALPHABET] IP Cliente: {request.META.get('REMOTE_ADDR', 'Unknown')}")
    print(f"[REGEX_TO_ALPHABET] User-Agent: {request.META.get('HTTP_USER_AGENT', 'Unknown')}")
    print(f"[REGEX_TO_ALPHABET] Origin: {request.META.get('HTTP_ORIGIN', 'None')}")
    print(f"[REGEX_TO_ALPHABET] Referer: {request.META.get('HTTP_REFERER', 'None')}")
    print(f"[REGEX_TO_ALPHABET] Path: {request.path}")
    print(f"[REGEX_TO_ALPHABET] Query String: {request.META.get('QUERY_STRING', 'None')}")
    sys.stdout.flush()
    
    regex = None
    
    if request.method == "GET":
        regex = request.GET.get("regex")
        print(f"[REGEX_TO_ALPHABET] GET - regex={repr(regex)}")
        sys.stdout.flush()
    else:  # POST
        try:
            body_str = request.body.decode('utf-8') if request.body else '{}'
            print(f"[REGEX_TO_ALPHABET] POST Body (raw): {body_str[:200]}...")
            sys.stdout.flush()
            
            data = json.loads(body_str)
            regex = data.get("regex")
            print(f"[REGEX_TO_ALPHABET] POST - regex={repr(regex)}")
            sys.stdout.flush()
        except json.JSONDecodeError as e:
            print(f"[REGEX_TO_ALPHABET] ERROR - JSON inválido: {e}")
            sys.stdout.flush()
            error_response = {
                "success": False,
                "error": "JSON inválido en el cuerpo de la petición"
            }
            print("[REGEX_TO_ALPHABET] --- RESPUESTA DE ERROR ---")
            print(json.dumps(error_response, ensure_ascii=False, indent=2))
            print("[REGEX_TO_ALPHABET] --- FIN RESPUESTA ---")
            print("=" * 80)
            sys.stdout.flush()
            return JsonResponse(error_response, status=400)
    
    if not regex:
        print("[REGEX_TO_ALPHABET] ERROR - Parámetro 'regex' faltante")
        sys.stdout.flush()
        error_response = {
            "success": False,
            "error": "Parámetro 'regex' es requerido"
        }
        print("[REGEX_TO_ALPHABET] --- RESPUESTA DE ERROR ---")
        print(json.dumps(error_response, ensure_ascii=False, indent=2))
        print("[REGEX_TO_ALPHABET] --- FIN RESPUESTA ---")
        print("=" * 80)
        sys.stdout.flush()
        return JsonResponse(error_response, status=400)
    
    # Validar que regex sea string
    regex = str(regex).strip()
    if not regex:
        print("[REGEX_TO_ALPHABET] ERROR - Regex vacío")
        sys.stdout.flush()
        error_response = {
            "success": False,
            "error": "El parámetro 'regex' no puede estar vacío"
        }
        print("[REGEX_TO_ALPHABET] --- RESPUESTA DE ERROR ---")
        print(json.dumps(error_response, ensure_ascii=False, indent=2))
        print("[REGEX_TO_ALPHABET] --- FIN RESPUESTA ---")
        print("=" * 80)
        sys.stdout.flush()
        return JsonResponse(error_response, status=400)
    
    # ========== PROCESAR REGEX ==========
    try:
        print(f"[REGEX_TO_ALPHABET] Procesando regex: {repr(regex)}")
        sys.stdout.flush()
        
        # Predecir alfabeto usando el modelo
        result = predict_alphabet(regex)
        
        alphabet = result['sigma_hat']
        probabilities = result['p_sigma']
        
        print(f"[REGEX_TO_ALPHABET] Alfabeto predicho: {alphabet}")
        print(f"[REGEX_TO_ALPHABET] Número de símbolos: {len(alphabet)}")
        sys.stdout.flush()
        
        # ========== CONSTRUIR RESPUESTA ==========
        response = {
            "success": True,
            "regex": regex,
            "alphabet": alphabet,
            "probabilities": probabilities,
            "error": None
        }
        
        print("[REGEX_TO_ALPHABET] --- RESPUESTA EXITOSA ---")
        print(json.dumps(response, ensure_ascii=False, indent=2))
        print("[REGEX_TO_ALPHABET] --- FIN RESPUESTA ---")
        print("=" * 80)
        sys.stdout.flush()
        
        return JsonResponse(response)
        
    except FileNotFoundError as e:
        print(f"[REGEX_TO_ALPHABET] ERROR - Archivo no encontrado: {e}")
        sys.stdout.flush()
        error_response = {
            "success": False,
            "regex": regex,
            "alphabet": None,
            "probabilities": None,
            "error": f"Error al cargar el modelo: {str(e)}"
        }
        print("[REGEX_TO_ALPHABET] --- RESPUESTA DE ERROR ---")
        print(json.dumps(error_response, ensure_ascii=False, indent=2))
        print("[REGEX_TO_ALPHABET] --- FIN RESPUESTA ---")
        print("=" * 80)
        sys.stdout.flush()
        return JsonResponse(error_response, status=500)
        
    except ImportError as e:
        print(f"[REGEX_TO_ALPHABET] ERROR - Error de importación: {e}")
        sys.stdout.flush()
        error_response = {
            "success": False,
            "regex": regex,
            "alphabet": None,
            "probabilities": None,
            "error": f"Error al importar el modelo: {str(e)}"
        }
        print("[REGEX_TO_ALPHABET] --- RESPUESTA DE ERROR ---")
        print(json.dumps(error_response, ensure_ascii=False, indent=2))
        print("[REGEX_TO_ALPHABET] --- FIN RESPUESTA ---")
        print("=" * 80)
        sys.stdout.flush()
        return JsonResponse(error_response, status=500)
        
    except Exception as e:
        print(f"[REGEX_TO_ALPHABET] ERROR - Excepción inesperada: {e}")
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        error_response = {
            "success": False,
            "regex": regex,
            "alphabet": None,
            "probabilities": None,
            "error": f"Error al procesar la regex: {str(e)}"
        }
        print("[REGEX_TO_ALPHABET] --- RESPUESTA DE ERROR ---")
        print(json.dumps(error_response, ensure_ascii=False, indent=2))
        print("[REGEX_TO_ALPHABET] --- FIN RESPUESTA ---")
        print("=" * 80)
        sys.stdout.flush()
        return JsonResponse(error_response, status=500)


# Cache para el DataFrame del CSV
_csv_cache = None
_csv_cache_path = None

def _load_csv_cache(csv_path):
    """Carga y cachea el DataFrame del CSV para evitar leerlo repetidamente"""
    global _csv_cache, _csv_cache_path
    
    # Si el path cambió o no hay cache, recargar
    if _csv_cache is None or _csv_cache_path != csv_path:
        print(f"[SEARCH_REGEX] Cargando CSV desde: {csv_path}")
        sys.stdout.flush()
        _csv_cache = pd.read_csv(csv_path)
        _csv_cache_path = csv_path
        print(f"[SEARCH_REGEX] CSV cargado: {len(_csv_cache)} filas")
        sys.stdout.flush()
    
    return _csv_cache


@csrf_exempt
@require_http_methods(["GET", "POST"])
def search_regex(request):
    """
    Endpoint que busca regex en el dataset de AcepNet (dataset6000.csv).
    
    Parámetros:
    - GET: ?query=<palabra_clave>&limit=<numero> o ?id=<numero>
    - POST: {"query": "<palabra_clave>", "limit": <numero>} o {"id": <numero>}
    
    Retorna JSON con:
    {
        "success": true/false,
        "query": "<palabra_clave>" o null,
        "id": <numero> o null,
        "results": [
            {"id": 0, "regex": "[LCIG]+"},
            {"id": 1, "regex": "[GDIK]*"},
            ...
        ],
        "total": 10,
        "limit": 50,
        "error": null o "mensaje de error"
    }
    """
    # ========== LOGS DE ENTRADA ==========
    print("=" * 80)
    print(f"[SEARCH_REGEX] Petición recibida - {datetime.now()}")
    print(f"[SEARCH_REGEX] Método HTTP: {request.method}")
    print(f"[SEARCH_REGEX] IP Cliente: {request.META.get('REMOTE_ADDR', 'Unknown')}")
    print(f"[SEARCH_REGEX] User-Agent: {request.META.get('HTTP_USER_AGENT', 'Unknown')}")
    print(f"[SEARCH_REGEX] Origin: {request.META.get('HTTP_ORIGIN', 'None')}")
    print(f"[SEARCH_REGEX] Path: {request.path}")
    print(f"[SEARCH_REGEX] Query String: {request.META.get('QUERY_STRING', 'None')}")
    sys.stdout.flush()
    
    query = None
    search_id = None
    limit = 50  # Límite por defecto
    
    if request.method == "GET":
        query = request.GET.get("query") or request.GET.get("q")
        id_str = request.GET.get("id")
        limit_str = request.GET.get("limit")
        
        if id_str:
            try:
                search_id = int(id_str)
            except ValueError:
                pass
        
        if limit_str:
            try:
                limit = int(limit_str)
                limit = max(1, min(limit, 1000))  # Entre 1 y 1000
            except ValueError:
                pass
        print(f"[SEARCH_REGEX] GET - query={repr(query)}, id={search_id}, limit={limit}")
        sys.stdout.flush()
    else:  # POST
        try:
            body_str = request.body.decode('utf-8') if request.body else '{}'
            print(f"[SEARCH_REGEX] POST Body (raw): {body_str[:200]}...")
            sys.stdout.flush()
            
            data = json.loads(body_str)
            query = data.get("query") or data.get("q")
            
            if "id" in data:
                try:
                    search_id = int(data["id"])
                except (ValueError, TypeError):
                    pass
            
            if "limit" in data:
                try:
                    limit = int(data["limit"])
                    limit = max(1, min(limit, 1000))  # Entre 1 y 1000
                except ValueError:
                    pass
            print(f"[SEARCH_REGEX] POST - query={repr(query)}, id={search_id}, limit={limit}")
            sys.stdout.flush()
        except json.JSONDecodeError as e:
            print(f"[SEARCH_REGEX] ERROR - JSON inválido: {e}")
            sys.stdout.flush()
            error_response = {
                "success": False,
                "query": None,
                "id": None,
                "results": [],
                "total": 0,
                "limit": limit,
                "error": "JSON inválido en el cuerpo de la petición"
            }
            print("[SEARCH_REGEX] --- RESPUESTA DE ERROR ---")
            print(json.dumps(error_response, ensure_ascii=False, indent=2))
            print("[SEARCH_REGEX] --- FIN RESPUESTA ---")
            print("=" * 80)
            sys.stdout.flush()
            return JsonResponse(error_response, status=400)
    
    # Validar que al menos uno de los parámetros esté presente
    if search_id is None and not query:
        print("[SEARCH_REGEX] ERROR - Parámetros faltantes")
        sys.stdout.flush()
        error_response = {
            "success": False,
            "query": None,
            "id": None,
            "results": [],
            "total": 0,
            "limit": limit,
            "error": "Se requiere al menos uno de: 'query'/'q' o 'id'"
        }
        print("[SEARCH_REGEX] --- RESPUESTA DE ERROR ---")
        print(json.dumps(error_response, ensure_ascii=False, indent=2))
        print("[SEARCH_REGEX] --- FIN RESPUESTA ---")
        print("=" * 80)
        sys.stdout.flush()
        return JsonResponse(error_response, status=400)
    
    # Validar query si se proporcionó
    if query:
        query = str(query).strip()
        if not query:
            query = None  # Si está vacío, ignorarlo
    
    # ========== BUSCAR EN CSV ==========
    try:
        # Ruta al CSV del dataset
        csv_path = Path(__file__).parent.parent / "models" / "acepnet" / "dataset6000.csv"
        
        if not csv_path.exists():
            print(f"[SEARCH_REGEX] ERROR - Archivo CSV no encontrado: {csv_path}")
            sys.stdout.flush()
            error_response = {
                "success": False,
                "query": query,
                "id": search_id,
                "results": [],
                "total": 0,
                "limit": limit,
                "error": f"Archivo CSV no encontrado: {csv_path}"
            }
            print("[SEARCH_REGEX] --- RESPUESTA DE ERROR ---")
            print(json.dumps(error_response, ensure_ascii=False, indent=2))
            print("[SEARCH_REGEX] --- FIN RESPUESTA ---")
            print("=" * 80)
            sys.stdout.flush()
            return JsonResponse(error_response, status=404)
        
        # Cargar CSV (con cache)
        df = _load_csv_cache(str(csv_path))
        
        results = []
        matches = pd.DataFrame()  # DataFrame vacío para casos de búsqueda por texto
        
        # Búsqueda por ID (tiene prioridad si está presente)
        if search_id is not None:
            print(f"[SEARCH_REGEX] Buscando por ID: {search_id}")
            sys.stdout.flush()
            
            # Validar que el ID esté en el rango válido
            if search_id < 0 or search_id >= len(df):
                print(f"[SEARCH_REGEX] ERROR - ID fuera de rango: {search_id} (rango válido: 0-{len(df)-1})")
                sys.stdout.flush()
                error_response = {
                    "success": False,
                    "query": query,
                    "id": search_id,
                    "results": [],
                    "total": 0,
                    "limit": limit,
                    "error": f"ID fuera de rango. Debe estar entre 0 y {len(df)-1}"
                }
                print("[SEARCH_REGEX] --- RESPUESTA DE ERROR ---")
                print(json.dumps(error_response, ensure_ascii=False, indent=2))
                print("[SEARCH_REGEX] --- FIN RESPUESTA ---")
                print("=" * 80)
                sys.stdout.flush()
                return JsonResponse(error_response, status=400)
            
            # Obtener la fila por ID
            row = df.iloc[search_id]
            results.append({
                "id": int(search_id),
                "regex": str(row['Regex'])
            })
            matches = df.iloc[[search_id]]  # DataFrame con una sola fila para consistencia
            
            print(f"[SEARCH_REGEX] Regex encontrado por ID: {results[0]['regex']}")
            sys.stdout.flush()
        
        # Búsqueda por texto (si no se buscó por ID o se quiere combinar)
        elif query:
            print(f"[SEARCH_REGEX] Buscando regex que contengan: {repr(query)}")
            sys.stdout.flush()
            
            # Buscar regex que contengan la query (case-insensitive)
            print(f"[SEARCH_REGEX] Buscando en {len(df)} filas...")
            sys.stdout.flush()
            mask = df['Regex'].str.contains(query, case=False, na=False)
            matches = df[mask]
            
            print(f"[SEARCH_REGEX] Encontrados {len(matches)} resultados")
            sys.stdout.flush()
            
            # Limitar resultados
            matches_limited = matches.head(limit)
            
            # Construir lista de resultados con id y regex
            for idx, row in matches_limited.iterrows():
                # El ID es el índice original del DataFrame (0-5999)
                results.append({
                    "id": int(idx),
                    "regex": str(row['Regex'])
                })
        
        # ========== CONSTRUIR RESPUESTA ==========
        response_data = {
            "success": True,
            "query": query,
            "id": search_id,
            "results": results,
            "total": len(matches),
            "limit": limit,
            "error": None
        }
        
        print("[SEARCH_REGEX] --- RESPUESTA EXITOSA ---")
        if search_id is not None:
            print(f"[SEARCH_REGEX] Búsqueda por ID: {search_id}")
        if query:
            print(f"[SEARCH_REGEX] Query: {query}")
        print(f"[SEARCH_REGEX] Total encontrados: {len(matches)}")
        print(f"[SEARCH_REGEX] Resultados devueltos: {len(results)}")
        print("[SEARCH_REGEX] --- FIN RESPUESTA ---")
        print("=" * 80)
        sys.stdout.flush()
        
        return JsonResponse(response_data, json_dumps_params={"ensure_ascii": False})
        
    except Exception as e:
        import traceback
        print(f"[SEARCH_REGEX] ERROR - Excepción inesperada: {e}")
        traceback.print_exc()
        sys.stdout.flush()
        error_response = {
            "success": False,
            "query": query,
            "id": search_id,
            "results": [],
            "total": 0,
            "limit": limit,
            "error": f"Error al buscar en el CSV: {str(e)}"
        }
        print("[SEARCH_REGEX] --- RESPUESTA DE ERROR ---")
        print(json.dumps(error_response, ensure_ascii=False, indent=2))
        print("[SEARCH_REGEX] --- FIN RESPUESTA ---")
        print("=" * 80)
        sys.stdout.flush()
        return JsonResponse(error_response, status=500)


@csrf_exempt
@require_http_methods(["GET", "POST"])
def acepnet_predict(request):
    """
    Endpoint que usa el modelo AcepNet para predecir si una cadena es aceptada por un AFD.
    
    Parámetros:
    - GET: ?dfa_id=<id>&string=<cadena> o ?dfa_id=<id>&strings=<cadena1,cadena2>
    - POST: {"dfa_id": <id>, "string": "<cadena>"} o {"dfa_id": <id>, "strings": ["cadena1", "cadena2"]}
    
    Retorna JSON con:
    {
        "success": true/false,
        "dfa_id": <numero>,
        "afd_info": {
            "regex": "...",
            "alphabet": "...",
            "states": "...",
            "accepting": "..."
        },
        "predictions": [
            {
                "string": "...",
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
    """
    # ========== LOGS DE ENTRADA ==========
    print("=" * 80)
    print(f"[ACEPNET_PREDICT] Petición recibida - {datetime.now()}")
    print(f"[ACEPNET_PREDICT] Método HTTP: {request.method}")
    print(f"[ACEPNET_PREDICT] IP Cliente: {request.META.get('REMOTE_ADDR', 'Unknown')}")
    print(f"[ACEPNET_PREDICT] User-Agent: {request.META.get('HTTP_USER_AGENT', 'Unknown')}")
    print(f"[ACEPNET_PREDICT] Origin: {request.META.get('HTTP_ORIGIN', 'None')}")
    print(f"[ACEPNET_PREDICT] Path: {request.path}")
    print(f"[ACEPNET_PREDICT] Query String: {request.META.get('QUERY_STRING', 'None')}")
    sys.stdout.flush()
    
    dfa_id = None
    strings = []  # Lista de cadenas a probar
    
    if request.method == "GET":
        dfa_id_str = request.GET.get("dfa_id") or request.GET.get("id")
        if dfa_id_str:
            try:
                dfa_id = int(dfa_id_str)
            except ValueError:
                pass
        
        # GET puede tener múltiples parámetros "string" o un parámetro "strings" separado por comas
        string_list = request.GET.getlist("string")  # Obtener todos los parámetros "string"
        if string_list:
            strings = string_list
        else:
            # Intentar obtener "strings" como cadena separada por comas
            strings_str = request.GET.get("strings")
            if strings_str:
                strings = [s.strip() for s in strings_str.split(",") if s.strip()]
        
        print(f"[ACEPNET_PREDICT] GET - dfa_id={dfa_id}, strings={strings}")
        sys.stdout.flush()
    else:  # POST
        try:
            body_str = request.body.decode('utf-8') if request.body else '{}'
            print(f"[ACEPNET_PREDICT] POST Body (raw): {body_str[:200]}...")
            sys.stdout.flush()
            
            data = json.loads(body_str)
            dfa_id_str = data.get("dfa_id") or data.get("id")
            if dfa_id_str is not None:
                try:
                    dfa_id = int(dfa_id_str)
                except (ValueError, TypeError):
                    pass
            
            # POST puede tener "string" (cadena única) o "strings" (array)
            if "strings" in data:
                # Array de cadenas
                strings_data = data.get("strings")
                if isinstance(strings_data, list):
                    strings = [str(s) for s in strings_data]
                elif isinstance(strings_data, str):
                    # Si es una cadena, separarla por comas
                    strings = [s.strip() for s in strings_data.split(",") if s.strip()]
            elif "string" in data:
                # Cadena única (compatibilidad hacia atrás)
                string_data = data.get("string")
                if string_data is not None:
                    if isinstance(string_data, list):
                        strings = [str(s) for s in string_data]
                    else:
                        strings = [str(string_data)]
            
            print(f"[ACEPNET_PREDICT] POST - dfa_id={dfa_id}, strings={strings}")
            sys.stdout.flush()
        except json.JSONDecodeError as e:
            print(f"[ACEPNET_PREDICT] ERROR - JSON inválido: {e}")
            sys.stdout.flush()
            error_response = {
                "success": False,
                "dfa_id": None,
                "afd_info": None,
                "predictions": [],
                "error": "JSON inválido en el cuerpo de la petición"
            }
            print("[ACEPNET_PREDICT] --- RESPUESTA DE ERROR ---")
            print(json.dumps(error_response, ensure_ascii=False, indent=2))
            print("[ACEPNET_PREDICT] --- FIN RESPUESTA ---")
            print("=" * 80)
            sys.stdout.flush()
            return JsonResponse(error_response, status=400)
    
    # Validar dfa_id
    if dfa_id is None:
        print("[ACEPNET_PREDICT] ERROR - Parámetro 'dfa_id' faltante")
        sys.stdout.flush()
        error_response = {
            "success": False,
            "dfa_id": None,
            "afd_info": None,
            "predictions": [],
            "error": "Parámetro 'dfa_id' o 'id' es requerido"
        }
        print("[ACEPNET_PREDICT] --- RESPUESTA DE ERROR ---")
        print(json.dumps(error_response, ensure_ascii=False, indent=2))
        print("[ACEPNET_PREDICT] --- FIN RESPUESTA ---")
        print("=" * 80)
        sys.stdout.flush()
        return JsonResponse(error_response, status=400)
    
    # Validar strings
    if not strings:
        print("[ACEPNET_PREDICT] ERROR - Parámetro 'string' o 'strings' faltante")
        sys.stdout.flush()
        error_response = {
            "success": False,
            "dfa_id": dfa_id,
            "afd_info": None,
            "predictions": [],
            "error": "Parámetro 'string' o 'strings' es requerido"
        }
        print("[ACEPNET_PREDICT] --- RESPUESTA DE ERROR ---")
        print(json.dumps(error_response, ensure_ascii=False, indent=2))
        print("[ACEPNET_PREDICT] --- FIN RESPUESTA ---")
        print("=" * 80)
        sys.stdout.flush()
        return JsonResponse(error_response, status=400)
    
    # ========== HACER PREDICCIÓN ==========
    try:
        print(f"[ACEPNET_PREDICT] Haciendo predicción - DFA ID: {dfa_id}, Cadenas: {strings}")
        sys.stdout.flush()
        
        # Obtener servicio AcepNet (singleton)
        service = get_acepnet_service()
        
        # Obtener información del AFD
        afd_info = service.get_afd_info(dfa_id)
        
        # Hacer predicciones para todas las cadenas
        predictions = []
        for string in strings:
            try:
                result = service.predict(dfa_id, string)
                
                predictions.append({
                    "string": result['string'],
                    "y1": {
                        "probability": result['y1']['probability'],
                        "predicted": result['y1']['predicted'],
                        "ground_truth": result['y1']['ground_truth'],
                        "correct": result['y1']['correct'],
                        "alphabet_mismatch": result['y1']['alphabet_mismatch']
                    },
                    "y2": {
                        "probability": result['y2']['probability'],
                        "predicted": result['y2']['predicted']
                    }
                })
                
                print(f"[ACEPNET_PREDICT] Cadena '{string}': Y1={result['y1']['probability']:.4f} → {'ACEPTA' if result['y1']['predicted'] else 'RECHAZA'} (GT={'ACEPTA' if result['y1']['ground_truth'] else 'RECHAZA'}, Correcto={'SÍ' if result['y1']['correct'] else 'NO'})")
                sys.stdout.flush()
            except Exception as e:
                print(f"[ACEPNET_PREDICT] ERROR al predecir cadena '{string}': {e}")
                sys.stdout.flush()
                predictions.append({
                    "string": string,
                    "error": str(e)
                })
        
        # ========== CONSTRUIR RESPUESTA ==========
        response_data = {
            "success": True,
            "dfa_id": dfa_id,
            "afd_info": afd_info,
            "predictions": predictions,
            "error": None
        }
        
        print("[ACEPNET_PREDICT] --- RESPUESTA EXITOSA ---")
        print(f"[ACEPNET_PREDICT] DFA ID: {dfa_id}")
        print(f"[ACEPNET_PREDICT] Regex: {afd_info['regex']}")
        print(f"[ACEPNET_PREDICT] Cadenas evaluadas: {len(predictions)}")
        print("[ACEPNET_PREDICT] --- FIN RESPUESTA ---")
        print("=" * 80)
        sys.stdout.flush()
        
        return JsonResponse(response_data, json_dumps_params={"ensure_ascii": False})
        
    except ValueError as e:
        # Errores de validación (dfa_id fuera de rango, etc.)
        print(f"[ACEPNET_PREDICT] ERROR - Error de validación: {e}")
        sys.stdout.flush()
        error_response = {
            "success": False,
            "dfa_id": dfa_id,
            "afd_info": None,
            "predictions": [],
            "error": str(e)
        }
        print("[ACEPNET_PREDICT] --- RESPUESTA DE ERROR ---")
        print(json.dumps(error_response, ensure_ascii=False, indent=2))
        print("[ACEPNET_PREDICT] --- FIN RESPUESTA ---")
        print("=" * 80)
        sys.stdout.flush()
        return JsonResponse(error_response, status=400)
    
    except FileNotFoundError as e:
        print(f"[ACEPNET_PREDICT] ERROR - Archivo no encontrado: {e}")
        sys.stdout.flush()
        error_response = {
            "success": False,
            "dfa_id": dfa_id,
            "afd_info": None,
            "predictions": [],
            "error": f"Error al cargar el modelo: {str(e)}"
        }
        print("[ACEPNET_PREDICT] --- RESPUESTA DE ERROR ---")
        print(json.dumps(error_response, ensure_ascii=False, indent=2))
        print("[ACEPNET_PREDICT] --- FIN RESPUESTA ---")
        print("=" * 80)
        sys.stdout.flush()
        return JsonResponse(error_response, status=500)
    
    except ImportError as e:
        print(f"[ACEPNET_PREDICT] ERROR - Error de importación: {e}")
        sys.stdout.flush()
        error_response = {
            "success": False,
            "dfa_id": dfa_id,
            "afd_info": None,
            "predictions": [],
            "error": f"Error al importar el modelo: {str(e)}"
        }
        print("[ACEPNET_PREDICT] --- RESPUESTA DE ERROR ---")
        print(json.dumps(error_response, ensure_ascii=False, indent=2))
        print("[ACEPNET_PREDICT] --- FIN RESPUESTA ---")
        print("=" * 80)
        sys.stdout.flush()
        return JsonResponse(error_response, status=500)
    
    except Exception as e:
        import traceback
        print(f"[ACEPNET_PREDICT] ERROR - Excepción inesperada: {e}")
        traceback.print_exc()
        sys.stdout.flush()
        error_response = {
            "success": False,
            "dfa_id": dfa_id,
            "afd_info": None,
            "predictions": [],
            "error": f"Error al hacer predicción: {str(e)}"
        }
        print("[ACEPNET_PREDICT] --- RESPUESTA DE ERROR ---")
        print(json.dumps(error_response, ensure_ascii=False, indent=2))
        print("[ACEPNET_PREDICT] --- FIN RESPUESTA ---")
        print("=" * 80)
        sys.stdout.flush()
        return JsonResponse(error_response, status=500)