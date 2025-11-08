# example/views.py
from datetime import datetime
import json

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
    regex = None
    test_string = None
    
    if request.method == "GET":
        regex = request.GET.get("regex")
        test_string = request.GET.get("test")
    else:  # POST
        try:
            data = json.loads(request.body) if request.body else {}
            regex = data.get("regex")
            test_string = data.get("test")
        except json.JSONDecodeError:
            return JsonResponse({
                "success": False,
                "error": "JSON inválido en el cuerpo de la petición"
            }, status=400)
    
    if not regex:
        return JsonResponse({
            "success": False,
            "error": "Parámetro 'regex' requerido"
        }, status=400)
    
    try:
        # Convertir regex a NFA y luego a DFA
        nfa = regex_to_nfa(regex)
        dfa = nfa_to_dfa(nfa)
        
        # Serializar DFA a JSON
        dfa_json = dfa_to_json(dfa)
        
        # Si se proporcionó una cadena de prueba, verificar si es aceptada
        test_result = None
        if test_string is not None:
            accepted = dfa_accepts(dfa, test_string)
            test_result = {
                "string": test_string,
                "accepted": accepted
            }
        
        response_data = {
            "success": True,
            "regex": regex,
            "dfa": dfa_json,
            "test_result": test_result,
            "error": None
        }
        
        return JsonResponse(response_data, json_dumps_params={"ensure_ascii": False})
    
    except Exception as e:
        return JsonResponse({
            "success": False,
            "regex": regex,
            "dfa": None,
            "test_result": None,
            "error": str(e)
        }, status=400)