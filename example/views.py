# example/views.py
from datetime import datetime
import json
import sys

from django.http import HttpResponse, JsonResponse
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt

from example.thompson_nfa import (
    regex_to_nfa,
    nfa_to_dfa,
    alias_dfa,
    get_dfa_alphabet,
    get_dfa_states,
    dfa_accepts
)


def index(request):
    print("=" * 60)
    print(f"[INDEX] Petición recibida - {datetime.now()}")
    print(f"[INDEX] Método: {request.method}")
    print(f"[INDEX] IP: {request.META.get('REMOTE_ADDR', 'Unknown')}")
    print(f"[INDEX] User-Agent: {request.META.get('HTTP_USER_AGENT', 'Unknown')}")
    print("=" * 60)
    sys.stdout.flush()  # Forzar escritura inmediata
    
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


def dfa_to_json(dfa):
    """Convierte un DFA a un diccionario JSON serializable."""
    aliases, _ = alias_dfa(dfa)
    
    # Alfabeto
    alphabet = sorted(get_dfa_alphabet(dfa))
    
    # Estados
    all_states = get_dfa_states(dfa)
    state_names = sorted([aliases[s] for s in all_states])
    
    # Estado inicial
    start = aliases[dfa.start]
    
    # Estados de aceptación
    accepts = [aliases[s] for s in sorted(dfa.accepts, key=lambda x: sorted(x))]
    
    # Transiciones
    transitions = []
    for src in sorted(dfa.trans.keys(), key=lambda x: sorted(x)):
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
            return JsonResponse({
                "success": False,
                "error": "JSON inválido en el cuerpo de la petición"
            }, status=400)
    
    if not regex:
        print("[REGEX_TO_DFA] ERROR - Parámetro 'regex' faltante")
        sys.stdout.flush()
        return JsonResponse({
            "success": False,
            "error": "Parámetro 'regex' requerido"
        }, status=400)
    
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
        
        print("[REGEX_TO_DFA] ÉXITO - Respuesta generada correctamente")
        print(f"[REGEX_TO_DFA] Tamaño de respuesta: ~{len(json.dumps(response_data))} bytes")
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
        
        return JsonResponse({
            "success": False,
            "regex": regex,
            "dfa": None,
            "test_result": None,
            "error": str(e)
        }, status=400)