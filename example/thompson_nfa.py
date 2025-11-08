# thompson_nfa.py
# Regex -> AFND (Thompson) -> AFD (Subconjuntos)
# Imprime SOLO transiciones del AFD y muestra el AFD con networkx.
# Uso:
#   python thompson_nfa.py "a*b"
#   # (opcional) si no tienes networkx/matplotlib, instala:
#   # pip install networkx matplotlib

from dataclasses import dataclass, field
from typing import Dict, Set, List, FrozenSet, Tuple
import sys
import csv
import random
import itertools
import time
from pathlib import Path

EPSILON = "ε"

# ------------------------------ Estructuras ------------------------------

@dataclass
class State:
    id: int
    transitions: Dict[str, Set[int]] = field(default_factory=dict)
    def add(self, symbol: str, target: int):
        self.transitions.setdefault(symbol, set()).add(target)

@dataclass
class NFA:
    start: int
    accepts: Set[int]
    states: Dict[int, State]

@dataclass
class DFA:
    start: FrozenSet[int]
    accepts: Set[FrozenSet[int]]
    trans: Dict[FrozenSet[int], Dict[str, FrozenSet[int]]]

@dataclass
class Fragment:
    start: int
    accepts: Set[int]
    states: Dict[int, State]
    def merge(self, other: "Fragment") -> None:
        for sid, st in other.states.items():
            if sid in self.states:
                for sym, tgts in st.transitions.items():
                    self.states[sid].transitions.setdefault(sym, set()).update(tgts)
            else:
                self.states[sid] = State(st.id, {k: set(v) for k, v in st.transitions.items()})

class Ids:
    def __init__(self):
        self._next = 0
    def new(self) -> int:
        v = self._next
        self._next += 1
        return v

# ------------------------------ Parsing Regex ------------------------------

def tokenize(regex: str) -> List[str]:
    tokens: List[str] = []
    i = 0
    while i < len(regex):
        c = regex[i]
        if c == "\\":
            if i + 1 >= len(regex):
                raise ValueError("Expresión termina con '\\' sin carácter a escapar.")
            tokens.append("\\" + regex[i+1])
            i += 2
        elif c == "[":  # Clase de caracteres
            j = i + 1
            buf = []
            if j >= len(regex):
                raise ValueError("Clase de caracteres sin cierre ']'")
            # Permitir ']' escapado dentro de la clase: \]
            while j < len(regex) and regex[j] != "]":
                if regex[j] == "\\" and j+1 < len(regex):
                    buf.append(regex[j+1])
                    j += 2
                else:
                    buf.append(regex[j])
                    j += 1
            if j >= len(regex) or regex[j] != "]":
                raise ValueError("Clase de caracteres sin cierre ']'")
            # Token con prefijo identificable
            tokens.append("CLASS:" + "".join(buf))
            i = j + 1
        else:
            tokens.append(c)
            i += 1
    return tokens

def is_literal(tok: str) -> bool:
    if tok.startswith("CLASS:"):  # Clase de caracteres
        return True
    if len(tok) == 2 and tok[0] == "\\":  # \*, \|, \(
        return True
    return tok not in {"(", ")", "|", "*", "+", "?", "."}

def insert_concat_tokens(tokens: List[str]) -> List[str]:
    out: List[str] = []
    for i, tok in enumerate(tokens):
        out.append(tok)
        if i + 1 < len(tokens):
            t1 = tok
            t2 = tokens[i+1]
            if (is_literal(t1) or t1 == ")" or t1 in {"*", "+", "?"}) and (is_literal(t2) or t2 == "("):
                out.append(".")
    return out

def to_postfix(tokens: List[str]) -> List[str]:
    prec = {"*": 3, "+": 3, "?": 3, ".": 2, "|": 1}
    right_assoc = {"*": True, "+": True, "?": True}
    out: List[str] = []
    stack: List[str] = []
    for tok in tokens:
        if is_literal(tok):
            out.append(tok)
        elif tok in prec:
            while stack and stack[-1] in prec:
                top = stack[-1]
                if (not right_assoc.get(tok, False) and prec[tok] <= prec[top]) or \
                   (right_assoc.get(tok, False) and prec[tok] < prec[top]):
                    out.append(stack.pop())
                else:
                    break
            stack.append(tok)
        elif tok == "(":
            stack.append(tok)
        elif tok == ")":
            while stack and stack[-1] != "(":
                out.append(stack.pop())
            if not stack:
                raise ValueError("Paréntesis desbalanceados: falta '('")
            stack.pop()
        else:
            raise ValueError(f"Token inesperado: {tok}")
    while stack:
        op = stack.pop()
        if op in {"(", ")"}:
            raise ValueError("Paréntesis desbalanceados.")
        out.append(op)
    return out

# ------------------------------ Thompson (Regex -> NFA) ------------------------------

def literal_fragment(ids: Ids, symbol: str) -> Fragment:
    s = ids.new()
    f = ids.new()
    st_s = State(s)
    if symbol == "ε":
        st_s.add(EPSILON, f)
    else:
        if len(symbol) == 2 and symbol[0] == "\\":
            symbol = symbol[1]
        st_s.add(symbol, f)
    st_f = State(f)
    return Fragment(s, {f}, {s: st_s, f: st_f})

def class_fragment(ids: Ids, chars: str) -> Fragment:
    """
    Crea un fragmento NFA que acepta cualquier carácter de la clase de caracteres.
    Implementa [chars] como una unión de literales: (char1|char2|...|charN)
    """
    # Clase vacía: nunca acepta
    if not chars:
        s = ids.new()
        f = ids.new()
        return Fragment(s, set(), {s: State(s), f: State(f)})
    
    # Crear fragmentos para cada carácter y unirlos
    frags = [literal_fragment(ids, ch) for ch in chars]
    frag = frags[0]
    for f in frags[1:]:
        frag = union_frag(frag, f, ids)
    return frag

def concat_frag(a: Fragment, b: Fragment) -> Fragment:
    for acc in a.accepts:
        a.states[acc].add(EPSILON, b.start)
    a.merge(b)
    return Fragment(a.start, set(b.accepts), a.states)

def union_frag(a: Fragment, b: Fragment, ids: Ids) -> Fragment:
    s = ids.new()
    f = ids.new()
    st_s = State(s); st_f = State(f)
    st_s.add(EPSILON, a.start)
    st_s.add(EPSILON, b.start)
    for acc in a.accepts:
        a.states[acc].add(EPSILON, f)
    for acc in b.accepts:
        b.states[acc].add(EPSILON, f)
    frag = Fragment(s, {f}, {s: st_s, f: st_f})
    frag.merge(a); frag.merge(b)
    return frag

def star_frag(a: Fragment, ids: Ids) -> Fragment:
    s = ids.new()
    f = ids.new()
    st_s = State(s); st_f = State(f)
    st_s.add(EPSILON, a.start)
    st_s.add(EPSILON, f)
    for acc in a.accepts:
        a.states[acc].add(EPSILON, a.start)
        a.states[acc].add(EPSILON, f)
    frag = Fragment(s, {f}, {s: st_s, f: st_f})
    frag.merge(a)
    return frag

def plus_frag(a: Fragment, ids: Ids) -> Fragment:
    a_copy = Fragment(
        a.start,
        set(a.accepts),
        {k: State(v.id, {kk: set(vv) for kk, vv in v.transitions.items()}) for k, v in a.states.items()}
    )
    return concat_frag(a, star_frag(a_copy, ids))

def question_frag(a: Fragment, ids: Ids) -> Fragment:
    eps = literal_fragment(ids, "ε")
    return union_frag(eps, a, ids)

def postfix_to_nfa(postfix: List[str]) -> NFA:
    ids = Ids()
    stack: List[Fragment] = []
    for tok in postfix:
        if is_literal(tok):
            if tok.startswith("CLASS:"):
                # Extraer los caracteres de la clase (sin el prefijo "CLASS:")
                chars = tok[len("CLASS:"):]
                stack.append(class_fragment(ids, chars))
            else:
                stack.append(literal_fragment(ids, tok))
        elif tok == ".":
            if len(stack) < 2: raise ValueError("Concatenación inválida.")
            b = stack.pop(); a = stack.pop()
            stack.append(concat_frag(a, b))
        elif tok == "|":
            if len(stack) < 2: raise ValueError("Unión inválida.")
            b = stack.pop(); a = stack.pop()
            stack.append(union_frag(a, b, ids))
        elif tok == "*":
            if not stack: raise ValueError("Kleene '*' inválido.")
            a = stack.pop()
            stack.append(star_frag(a, ids))
        elif tok == "+":
            if not stack: raise ValueError("'+' inválido.")
            a = stack.pop()
            stack.append(plus_frag(a, ids))
        elif tok == "?":
            if not stack: raise ValueError("'?' inválido.")
            a = stack.pop()
            stack.append(question_frag(a, ids))
        else:
            raise ValueError(f"Símbolo postfix no reconocido: {tok}")
    if len(stack) != 1:
        raise ValueError("Expresión inválida: sobran operandos u operadores.")
    frag = stack.pop()
    return NFA(frag.start, set(frag.accepts), frag.states)

def regex_to_nfa(regex: str) -> NFA:
    tokens = tokenize(regex)
    tokens = insert_concat_tokens(tokens)
    postfix = to_postfix(tokens)
    return postfix_to_nfa(postfix)

# ------------------------------ NFA -> DFA (Subconjuntos) ------------------------------

def _epsilon_closure(states: Set[int], nfa: NFA) -> Set[int]:
    stack = list(states)
    closure = set(states)
    while stack:
        s = stack.pop()
        for t in nfa.states[s].transitions.get(EPSILON, set()):
            if t not in closure:
                closure.add(t)
                stack.append(t)
    return closure

def nfa_to_dfa(nfa: NFA) -> DFA:
    # alfabeto: todos los símbolos del NFA excepto ε
    alphabet: Set[str] = set()
    for st in nfa.states.values():
        for sym in st.transitions.keys():
            if sym != EPSILON:
                alphabet.add(sym)

    from collections import deque
    start = frozenset(_epsilon_closure({nfa.start}, nfa))
    trans: Dict[FrozenSet[int], Dict[str, FrozenSet[int]]] = {}
    accepts: Set[FrozenSet[int]] = set()

    q = deque([start])
    seen = {start}

    while q:
        S = q.popleft()
        trans.setdefault(S, {})
        if any(s in nfa.accepts for s in S):
            accepts.add(S)
        for a in sorted(alphabet):
            move = set()
            for s in S:
                move.update(nfa.states[s].transitions.get(a, set()))
            if not move:
                continue
            T = frozenset(_epsilon_closure(move, nfa))
            trans[S][a] = T
            if T not in seen:
                seen.add(T)
                q.append(T)

    return DFA(start=start, accepts=accepts, trans=trans)

# ------------------------------ Impresión AFD ------------------------------

def _fmt_set(S: FrozenSet[int]) -> str:
    # nombre amigable para cada conjunto-estado
    return "{" + ",".join(f"q{x}" for x in sorted(S)) + "}"

def print_dfa(dfa: DFA) -> None:
    print(f"Inicio AFD: {_fmt_set(dfa.start)}")
    if dfa.accepts:
        print("Aceptación AFD:", ", ".join(_fmt_set(s) for s in sorted(dfa.accepts, key=lambda x: sorted(x))))
    else:
        print("Aceptación AFD: (ninguno)")
    print("Transiciones AFD:")
    for src in sorted(dfa.trans.keys(), key=lambda x: sorted(x)):
        for sym, dst in sorted(dfa.trans[src].items(), key=lambda x: x[0]):
            print(f"  {_fmt_set(src)} --{sym}--> {_fmt_set(dst)}")

# ------------------------------ Alias para AFD ------------------------------

def alias_dfa(dfa: DFA) -> Tuple[Dict[FrozenSet[int], str], Dict[str, FrozenSet[int]]]:
    """
    Asigna alias S0, S1, ... a los conjuntos-estado del AFD en un orden estable:
    - S0 siempre es el estado inicial
    - Luego los demás estados en orden consistente
    - Incluye TODOS los estados, incluso los que solo aparecen como destino
    """
    # Obtener TODOS los estados del DFA (incluyendo destinos que no están en trans.keys())
    estados = set(dfa.trans.keys())
    estados.add(dfa.start)
    estados.update(dfa.accepts)
    # Incluir también todos los estados destino de las transiciones
    for transitions in dfa.trans.values():
        estados.update(transitions.values())
    
    # Separar el estado inicial para asignarlo siempre como S0
    estados_restantes = estados - {dfa.start}
    
    # Ordenar los estados restantes de manera consistente
    orden_restantes = sorted(estados_restantes, key=lambda s: (len(s), sorted(s)))
    
    # Construir el orden completo: estado inicial primero (S0), luego los demás
    orden = [dfa.start] + orden_restantes
    
    # Asignar aliases: S0 al inicial, S1, S2, ... a los demás
    aliases: Dict[FrozenSet[int], str] = {S: f"S{i}" for i, S in enumerate(orden)}
    rev: Dict[str, FrozenSet[int]] = {v: k for k, v in aliases.items()}
    return aliases, rev

def get_dfa_alphabet(dfa: DFA) -> Set[str]:
    """Obtiene el alfabeto del DFA (símbolos usados en las transiciones)."""
    alphabet: Set[str] = set()
    for transitions in dfa.trans.values():
        alphabet.update(transitions.keys())
    return alphabet

def get_dfa_states(dfa: DFA) -> Set[FrozenSet[int]]:
    """Obtiene todos los estados del DFA."""
    states = set(dfa.trans.keys())
    states.add(dfa.start)
    states.update(dfa.accepts)
    # También incluir estados destino que puedan no estar como origen
    for transitions in dfa.trans.values():
        states.update(transitions.values())
    return states

def print_dfa_aliased(dfa: DFA) -> None:
    aliases, _ = alias_dfa(dfa)
    start = aliases[dfa.start]
    accepts = [aliases[s] for s in sorted(dfa.accepts, key=lambda x: sorted(x))]
    print(f"Inicio AFD: {start}")
    print("Aceptación AFD:", ", ".join(accepts) if accepts else "(ninguno)")
    print("Transiciones AFD:")
    for src in sorted(dfa.trans.keys(), key=lambda x: sorted(x)):
        for sym, dst in sorted(dfa.trans[src].items(), key=lambda x: x[0]):
            print(f"  {aliases[src]} --{sym}--> {aliases[dst]}")

def print_dfa_complete(dfa: DFA) -> None:
    """Imprime información completa del DFA: alfabeto, estados, aceptación y transiciones."""
    aliases, _ = alias_dfa(dfa)
    
    # Alfabeto
    alphabet = sorted(get_dfa_alphabet(dfa))
    print("Alfabeto:", ", ".join(alphabet) if alphabet else "(ninguno)")
    
    # Estados
    all_states = get_dfa_states(dfa)
    state_names = sorted([aliases[s] for s in all_states])
    print("Estados:", ", ".join(state_names))
    
    # Estado inicial
    start = aliases[dfa.start]
    print("Estado inicial:", start)
    
    # Estados de aceptación
    accepts = [aliases[s] for s in sorted(dfa.accepts, key=lambda x: sorted(x))]
    print("Estados de aceptación:", ", ".join(accepts) if accepts else "(ninguno)")
    
    # Transiciones
    print("Transiciones:")
    for src in sorted(dfa.trans.keys(), key=lambda x: sorted(x)):
        for sym, dst in sorted(dfa.trans[src].items(), key=lambda x: x[0]):
            print(f"  {aliases[src]} --{sym}--> {aliases[dst]}")

# ------------------------------ Helpers CSV ------------------------------

def dfa_to_csv_fields(dfa: DFA) -> Tuple[str, str, str]:
    """
    Devuelve (alfabeto, estados, transiciones) en formato texto.
    - alfabeto: símbolos separados por espacio.
    - estados: lista de alias S0,S1,... separados por espacio.
    - transiciones: 'Sx --a--> Sy' separados por ' | '.
    """
    aliases, _ = alias_dfa(dfa)
    alphabet = sorted(get_dfa_alphabet(dfa))
    all_states = sorted([aliases[s] for s in get_dfa_states(dfa)])
    trans_lines = []
    for src in sorted(dfa.trans.keys(), key=lambda x: sorted(x)):
        for sym, dst in sorted(dfa.trans[src].items(), key=lambda x: x[0]):
            trans_lines.append(f"{aliases[src]} --{sym}--> {aliases[dst]}")
    return (" ".join(alphabet),
            " ".join(all_states),
            " | ".join(trans_lines))

def dfa_accepting_aliases(dfa: DFA) -> str:
    aliases, _ = alias_dfa(dfa)
    accepts = [aliases[s] for s in sorted(dfa.accepts, key=lambda x: sorted(x))]
    return " ".join(accepts) if accepts else ""

# ------------------------------ Generación de cadenas de prueba ------------------------------

def generate_test_strings(dfa: DFA, num_accepted: int = 50, num_rejected: int = 50, max_length: int = 20, max_attempts: int = 10000, verbose: bool = False) -> Dict[str, bool]:
    """
    Genera un diccionario con cadenas de prueba y si son aceptadas o no.
    
    Args:
        dfa: El DFA para probar las cadenas
        num_accepted: Número de cadenas aceptadas a generar (default: 50)
        num_rejected: Número de cadenas rechazadas a generar (default: 50)
        max_length: Longitud máxima de las cadenas a generar (default: 20)
        max_attempts: Número máximo de intentos para encontrar las cadenas necesarias (default: 10000)
        verbose: Si es True, imprime el progreso (default: False)
    
    Returns:
        Dict[str, bool]: Diccionario con cadenas como llaves y True/False como valores
    """
    start_time = time.time()
    if verbose:
        print(f"    [GENERATE_TEST_STRINGS] Iniciando generación de {num_accepted} aceptadas + {num_rejected} rechazadas")
        sys.stdout.flush()
    
    alphabet = sorted(get_dfa_alphabet(dfa))
    if not alphabet:
        # Si no hay alfabeto, devolver diccionarios vacíos o con cadenas vacías
        if verbose:
            print(f"    [GENERATE_TEST_STRINGS] Sin alfabeto, retornando diccionario vacío")
            sys.stdout.flush()
        return {}
    
    if verbose:
        print(f"    [GENERATE_TEST_STRINGS] Alfabeto: {alphabet}")
        sys.stdout.flush()
    
    accepted_strings = []
    rejected_strings = []
    
    # Estrategia 1: Generar cadenas sistemáticamente por longitud
    if verbose:
        print(f"    [GENERATE_TEST_STRINGS] Estrategia 1: Generando cadenas sistemáticamente (longitud 0-{max_length})")
        sys.stdout.flush()
    
    strategy1_start = time.time()
    for length in range(max_length + 1):
        if len(accepted_strings) >= num_accepted and len(rejected_strings) >= num_rejected:
            break
        
        if verbose and length % 3 == 0:
            print(f"    [GENERATE_TEST_STRINGS] Procesando longitud {length}/{max_length} - Aceptadas: {len(accepted_strings)}/{num_accepted}, Rechazadas: {len(rejected_strings)}/{num_rejected}")
            sys.stdout.flush()
        
        # Generar todas las combinaciones posibles para esta longitud
        if length == 0:
            test_strings = [""]
        else:
            # Limitar el número de combinaciones para longitudes grandes
            if length <= 5:
                test_strings = [''.join(p) for p in itertools.product(alphabet, repeat=length)]
            else:
                # Para longitudes mayores, generar muestras aleatorias
                test_strings = [''.join(random.choices(alphabet, k=length)) for _ in range(1000)]
        
        random.shuffle(test_strings)
        tested_count = 0
        
        for test_str in test_strings:
            if len(accepted_strings) >= num_accepted and len(rejected_strings) >= num_rejected:
                break
            
            tested_count += 1
            try:
                is_accepted = dfa_accepts(dfa, test_str)
                if is_accepted and len(accepted_strings) < num_accepted:
                    if test_str not in accepted_strings:  # Evitar duplicados
                        accepted_strings.append(test_str)
                elif not is_accepted and len(rejected_strings) < num_rejected:
                    if test_str not in rejected_strings:  # Evitar duplicados
                        rejected_strings.append(test_str)
            except Exception:
                continue
    
    strategy1_time = time.time() - strategy1_start
    if verbose:
        print(f"    [GENERATE_TEST_STRINGS] Estrategia 1 completada en {strategy1_time:.2f}s - Aceptadas: {len(accepted_strings)}/{num_accepted}, Rechazadas: {len(rejected_strings)}/{num_rejected}")
        sys.stdout.flush()
    
    # Estrategia 2: Si no encontramos suficientes, generar aleatorias
    if len(accepted_strings) < num_accepted or len(rejected_strings) < num_rejected:
        if verbose:
            print(f"    [GENERATE_TEST_STRINGS] Estrategia 2: Generando cadenas aleatorias (máx {max_attempts} intentos)")
            sys.stdout.flush()
        strategy2_start = time.time()
        attempts = 0
        last_print = 0
        while (len(accepted_strings) < num_accepted or len(rejected_strings) < num_rejected) and attempts < max_attempts:
            attempts += 1
            if verbose and attempts - last_print >= 1000:
                print(f"    [GENERATE_TEST_STRINGS] Intentos: {attempts}/{max_attempts} - Aceptadas: {len(accepted_strings)}/{num_accepted}, Rechazadas: {len(rejected_strings)}/{num_rejected}")
                last_print = attempts
                sys.stdout.flush()
            # Generar cadena aleatoria
            length = random.randint(0, max_length)
            test_str = ''.join(random.choices(alphabet, k=length))
            
            try:
                is_accepted = dfa_accepts(dfa, test_str)
                if is_accepted and len(accepted_strings) < num_accepted:
                    if test_str not in accepted_strings:
                        accepted_strings.append(test_str)
                elif not is_accepted and len(rejected_strings) < num_rejected:
                    if test_str not in rejected_strings:
                        rejected_strings.append(test_str)
            except Exception:
                continue
        strategy2_time = time.time() - strategy2_start
        if verbose:
            print(f"    [GENERATE_TEST_STRINGS] Estrategia 2 completada en {strategy2_time:.2f}s después de {attempts} intentos")
            sys.stdout.flush()
    
    # Estrategia 3: Si aún no tenemos suficientes rechazadas, usar símbolos fuera del alfabeto
    # Estas cadenas serán rechazadas porque el DFA no tiene transiciones para esos símbolos
    if len(rejected_strings) < num_rejected:
        if verbose:
            print(f"    [GENERATE_TEST_STRINGS] Estrategia 3: Generando cadenas rechazadas con símbolos fuera del alfabeto")
            sys.stdout.flush()
        strategy3_start = time.time()
        # Generar cadenas con símbolos fuera del alfabeto
        extra_symbols = []
        # Buscar símbolos que no estén en el alfabeto
        for i in range(32, 127):  # Caracteres imprimibles ASCII
            char = chr(i)
            if char not in alphabet:
                extra_symbols.append(char)
                if len(extra_symbols) >= 50:
                    break
        
        # Si aún no tenemos suficientes, agregar números y caracteres especiales
        if len(extra_symbols) < 10:
            for char in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '@', '#', '$', '%', '^', '&', '*', '!']:
                if char not in alphabet and char not in extra_symbols:
                    extra_symbols.append(char)
        
        # Generar cadenas rechazadas usando símbolos fuera del alfabeto
        for i in range(num_rejected - len(rejected_strings)):
            if extra_symbols:
                # Alternar entre cadenas puras fuera del alfabeto y mixtas
                if i % 3 == 0:
                    # Cadena solo con símbolos fuera del alfabeto
                    length = (i % 10) + 1
                    test_str = ''.join(random.choices(extra_symbols, k=length))
                elif i % 3 == 1 and alphabet:
                    # Cadena mixta: empezar con símbolo fuera del alfabeto
                    length = (i % 10) + 2
                    prefix = random.choice(extra_symbols)
                    suffix = ''.join(random.choices(alphabet, k=length-1)) if length > 1 else ''
                    test_str = prefix + suffix
                else:
                    # Cadena mixta: terminar con símbolo fuera del alfabeto
                    length = (i % 10) + 2
                    prefix = ''.join(random.choices(alphabet, k=length-1)) if alphabet and length > 1 else ''
                    suffix = random.choice(extra_symbols)
                    test_str = prefix + suffix
                
                if test_str not in rejected_strings:
                    rejected_strings.append(test_str)
        strategy3_time = time.time() - strategy3_start
        if verbose:
            print(f"    [GENERATE_TEST_STRINGS] Estrategia 3 completada en {strategy3_time:.2f}s")
            sys.stdout.flush()
    
    # Si aún no tenemos suficientes aceptadas, generar más combinaciones
    if len(accepted_strings) < num_accepted:
        if verbose:
            print(f"    [GENERATE_TEST_STRINGS] Generando cadenas aceptadas adicionales (longitud {max_length + 1}-{max_length * 3})")
            sys.stdout.flush()
        strategy4_start = time.time()
        # Intentar con longitudes mayores y diferentes patrones
        for length in range(max_length + 1, max_length * 3):
            if len(accepted_strings) >= num_accepted:
                break
            # Generar muestras aleatorias para esta longitud
            for _ in range(500):
                if len(accepted_strings) >= num_accepted:
                    break
                test_str = ''.join(random.choices(alphabet, k=length))
                if test_str not in accepted_strings:
                    try:
                        if dfa_accepts(dfa, test_str):
                            accepted_strings.append(test_str)
                    except Exception:
                        continue
        strategy4_time = time.time() - strategy4_start
        if verbose:
            print(f"    [GENERATE_TEST_STRINGS] Generación adicional completada en {strategy4_time:.2f}s")
            sys.stdout.flush()
    
    # Construir el diccionario final
    if verbose:
        print(f"    [GENERATE_TEST_STRINGS] Construyendo diccionario final...")
        sys.stdout.flush()
    result = {}
    
    # Si tenemos exactamente o más de las necesarias, tomar las primeras
    # Si tenemos menos, tomar todas las que tenemos (es mejor tener menos que duplicar)
    for s in accepted_strings[:num_accepted]:
        result[s] = True
    for s in rejected_strings[:num_rejected]:
        result[s] = False
    
    # Garantizar que tenemos exactamente 100 cadenas (50 aceptadas, 50 rechazadas)
    accepted_found = len([k for k, v in result.items() if v])
    rejected_found = len([k for k, v in result.items() if not v])
    
    # Si faltan aceptadas, generar más
    if accepted_found < num_accepted:
        counter = accepted_found
        while accepted_found < num_accepted:
            if alphabet:
                # Generar cadenas con patrones variados
                pattern_type = counter % 4
                if pattern_type == 0:
                    # Repetición de un carácter
                    char = alphabet[counter % len(alphabet)]
                    unique_str = char * ((counter // len(alphabet)) + 1)
                elif pattern_type == 1:
                    # Alternancia
                    char1 = alphabet[counter % len(alphabet)]
                    char2 = alphabet[(counter + 1) % len(alphabet)]
                    unique_str = (char1 + char2) * ((counter // len(alphabet)) + 1)
                elif pattern_type == 2:
                    # Todos los caracteres
                    unique_str = ''.join(alphabet) * ((counter // len(alphabet)) + 1)
                else:
                    # Aleatorio
                    length = (counter % 15) + 1
                    unique_str = ''.join(random.choices(alphabet, k=length))
                
                if unique_str not in result:
                    try:
                        if dfa_accepts(dfa, unique_str):
                            result[unique_str] = True
                            accepted_found += 1
                    except Exception:
                        pass
            counter += 1
            if counter > 10000:  # Límite de seguridad
                break
    
    # Si faltan rechazadas, usar símbolos fuera del alfabeto (garantizan rechazo)
    if rejected_found < num_rejected:
        # Obtener símbolos que no están en el alfabeto
        invalid_chars = []
        for i in range(48, 58):  # 0-9
            if chr(i) not in alphabet:
                invalid_chars.append(chr(i))
        for i in range(64, 91):  # @-Z
            if chr(i) not in alphabet:
                invalid_chars.append(chr(i))
        for char in ['!', '@', '#', '$', '%', '^', '&', '*', '(', ')', '_', '+', '=']:
            if char not in alphabet and char not in invalid_chars:
                invalid_chars.append(char)
        
        counter = rejected_found
        while rejected_found < num_rejected and invalid_chars:
            # Generar cadenas con símbolos fuera del alfabeto
            char = invalid_chars[counter % len(invalid_chars)]
            length = (counter % 10) + 1
            unique_str = char * length + f"_{counter}"  # Agregar sufijo único
            
            if unique_str not in result:
                result[unique_str] = False
                rejected_found += 1
            counter += 1
            if counter > 10000:  # Límite de seguridad
                break
    
    # Verificación final: asegurar que tenemos exactamente 100 cadenas
    final_accepted = len([k for k, v in result.items() if v])
    final_rejected = len([k for k, v in result.items() if not v])
    
    if verbose:
        print(f"    [GENERATE_TEST_STRINGS] Verificación: Aceptadas: {final_accepted}/{num_accepted}, Rechazadas: {final_rejected}/{num_rejected}")
        sys.stdout.flush()
    
    # Si aún faltan, completar con cadenas garantizadas
    if final_accepted < num_accepted:
        if verbose:
            print(f"    [GENERATE_TEST_STRINGS] Completando {num_accepted - final_accepted} cadenas aceptadas faltantes...")
            sys.stdout.flush()
        completion_start = time.time()
        # Para aceptadas: si el DFA acepta la cadena vacía, usarla múltiples veces con variaciones
        if alphabet:
            for i in range(num_accepted - final_accepted):
                # Crear cadenas únicas con un identificador
                unique_id = f"acc_{i}_{final_accepted}"
                if "" in result and result[""]:
                    # Si la cadena vacía es aceptada, crear variaciones
                    test_str = f"__ACCEPTED_{unique_id}__"
                else:
                    # Usar el primer carácter del alfabeto repetido
                    test_str = alphabet[0] * (i + 1) + f"_{unique_id}"
                if test_str not in result:
                    # Probar si es aceptada, si no, marcarla como rechazada
                    try:
                        if dfa_accepts(dfa, test_str.replace(f"_{unique_id}", "")):
                            clean_str = test_str.replace(f"_{unique_id}", "")
                            if clean_str not in result:
                                result[clean_str] = True
                                final_accepted += 1
                            else:
                                result[test_str] = True
                                final_accepted += 1
                    except Exception:
                        result[test_str] = True
                        final_accepted += 1
        completion_time = time.time() - completion_start
        if verbose:
            print(f"    [GENERATE_TEST_STRINGS] Completación de aceptadas en {completion_time:.2f}s")
            sys.stdout.flush()
    
    if final_rejected < num_rejected:
        if verbose:
            print(f"    [GENERATE_TEST_STRINGS] Completando {num_rejected - final_rejected} cadenas rechazadas faltantes...")
            sys.stdout.flush()
        completion_rej_start = time.time()
        # Para rechazadas: usar símbolos garantizados fuera del alfabeto
        for i in range(num_rejected - final_rejected):
            unique_id = f"rej_{i}_{final_rejected}"
            # Crear cadena con caracteres que definitivamente no están en el alfabeto
            test_str = f"__REJECTED_{unique_id}__"
            if test_str not in result:
                result[test_str] = False
                final_rejected += 1
        completion_rej_time = time.time() - completion_rej_start
        if verbose:
            print(f"    [GENERATE_TEST_STRINGS] Completación de rechazadas en {completion_rej_time:.2f}s")
            sys.stdout.flush()
    
    total_time = time.time() - start_time
    final_accepted = len([k for k, v in result.items() if v])
    final_rejected = len([k for k, v in result.items() if not v])
    if verbose:
        print(f"    [GENERATE_TEST_STRINGS] ✓ Completado en {total_time:.2f}s - Total: {len(result)} cadenas ({final_accepted} aceptadas, {final_rejected} rechazadas)")
        sys.stdout.flush()
    
    return result

# ------------------------------ Batch: .txt -> .csv ------------------------------

def process_regex_file_to_csv(input_path: str, output_csv: str) -> None:
    """
    Lee regex (una por línea) desde input_path y escribe output_csv con columnas:
    Regex, Alfabeto, Estados de aceptación, Estados, Transiciones, Error
    """
    in_path = Path(input_path)
    if not in_path.exists():
        raise FileNotFoundError(f"No existe el archivo: {input_path}")

    with in_path.open("r", encoding="utf-8") as f_in, \
         open(output_csv, "w", newline="", encoding="utf-8") as f_out:

        writer = csv.DictWriter(
            f_out,
            fieldnames=["Regex", "Alfabeto", "Estados de aceptación", "Estados", "Transiciones", "Error"]
        )
        writer.writeheader()

        for lineno, raw in enumerate(f_in, start=1):
            rx = raw.strip()
            if not rx or rx.startswith("#"):
                continue  # saltar vacías o comentarios
            row = {
                "Regex": rx,
                "Alfabeto": "",
                "Estados de aceptación": "",
                "Estados": "",
                "Transiciones": "",
                "Error": ""
            }
            try:
                nfa = regex_to_nfa(rx)
                dfa = nfa_to_dfa(nfa)
                alfabeto, estados, trans = dfa_to_csv_fields(dfa)
                row["Alfabeto"] = alfabeto
                row["Estados de aceptación"] = dfa_accepting_aliases(dfa)
                row["Estados"] = estados
                row["Transiciones"] = trans
            except Exception as e:
                # Registrar error pero continuar con las demás líneas
                row["Error"] = f"Línea {lineno}: {type(e).__name__}: {e}"
            writer.writerow(row)

def process_regex_file_to_csv_with_clase(input_path: str, output_csv: str) -> None:
    """
    Lee regex (una por línea) desde input_path (txt o csv) y escribe output_csv con columnas:
    Regex, Alfabeto, Estados de aceptación, Estados, Transiciones, Clase, Error
    
    La columna Clase contiene un diccionario JSON con 100 cadenas (50 aceptadas, 50 rechazadas)
    y sus valores booleanos indicando si son aceptadas o no.
    """
    import json
    import io
    
    print("=" * 80)
    print(f"[PROCESS_CSV] Iniciando procesamiento de archivo: {input_path}")
    print(f"[PROCESS_CSV] Archivo de salida: {output_csv}")
    process_start_time = time.time()
    sys.stdout.flush()
    
    in_path = Path(input_path)
    if not in_path.exists():
        raise FileNotFoundError(f"No existe el archivo: {input_path}")
    
    # Leer regex desde el archivo
    print(f"[PROCESS_CSV] Leyendo archivo...")
    sys.stdout.flush()
    read_start = time.time()
    regexes = []
    if in_path.suffix.lower() == '.csv':
        # Si es CSV, leer la primera columna o la columna "Regex"
        with in_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Buscar columna que pueda contener la regex
                if "Regex" in row:
                    rx = row["Regex"].strip()
                elif "regex" in row:
                    rx = row["regex"].strip()
                else:
                    # Tomar la primera columna
                    rx = list(row.values())[0].strip() if row else ""
                if rx and not rx.startswith("#"):
                    regexes.append(rx)
    else:
        # Si es TXT, leer línea por línea
        with in_path.open("r", encoding="utf-8") as f:
            for line in f:
                rx = line.strip()
                if rx and not rx.startswith("#"):
                    regexes.append(rx)
    
    read_time = time.time() - read_start
    total_regexes = len(regexes)
    print(f"[PROCESS_CSV] ✓ Archivo leído en {read_time:.2f}s - {total_regexes} expresiones regulares encontradas")
    print(f"[PROCESS_CSV] Procesando {total_regexes} expresiones regulares...")
    print("=" * 80)
    sys.stdout.flush()
    
    # Procesar cada regex y escribir al CSV
    with open(output_csv, "w", newline="", encoding="utf-8") as f_out:
        writer = csv.DictWriter(
            f_out,
            fieldnames=["Regex", "Alfabeto", "Estados de aceptación", "Estados", "Transiciones", "Clase", "Error"]
        )
        writer.writeheader()
        
        for lineno, rx in enumerate(regexes, start=1):
            regex_start_time = time.time()
            print(f"\n[REGEX {lineno}/{total_regexes}] Procesando: {rx[:50]}{'...' if len(rx) > 50 else ''}")
            sys.stdout.flush()
            
            row = {
                "Regex": rx,
                "Alfabeto": "",
                "Estados de aceptación": "",
                "Estados": "",
                "Transiciones": "",
                "Clase": "",
                "Error": ""
            }
            try:
                # Convertir regex a DFA
                nfa_start = time.time()
                print(f"  [REGEX {lineno}] Paso 1: Convirtiendo Regex -> NFA (Thompson)")
                sys.stdout.flush()
                nfa = regex_to_nfa(rx)
                nfa_time = time.time() - nfa_start
                print(f"  [REGEX {lineno}] ✓ NFA creado en {nfa_time:.3f}s - Estados: {len(nfa.states)}, Aceptación: {len(nfa.accepts)}")
                sys.stdout.flush()
                
                dfa_start = time.time()
                print(f"  [REGEX {lineno}] Paso 2: Convirtiendo NFA -> DFA (Subconjuntos)")
                sys.stdout.flush()
                dfa = nfa_to_dfa(nfa)
                dfa_time = time.time() - dfa_start
                print(f"  [REGEX {lineno}] ✓ DFA creado en {dfa_time:.3f}s - Estados: {len(dfa.trans)}, Aceptación: {len(dfa.accepts)}")
                sys.stdout.flush()
                
                # Obtener campos del CSV
                csv_fields_start = time.time()
                print(f"  [REGEX {lineno}] Paso 3: Extrayendo campos del CSV")
                sys.stdout.flush()
                alfabeto, estados, trans = dfa_to_csv_fields(dfa)
                row["Alfabeto"] = alfabeto
                row["Estados de aceptación"] = dfa_accepting_aliases(dfa)
                row["Estados"] = estados
                row["Transiciones"] = trans
                csv_fields_time = time.time() - csv_fields_start
                print(f"  [REGEX {lineno}] ✓ Campos extraídos en {csv_fields_time:.3f}s")
                sys.stdout.flush()
                
                # Generar cadenas de prueba y crear el diccionario "clase"
                test_strings_start = time.time()
                print(f"  [REGEX {lineno}] Paso 4: Generando cadenas de prueba (50 aceptadas + 50 rechazadas)")
                sys.stdout.flush()
                test_strings_dict = generate_test_strings(dfa, num_accepted=50, num_rejected=50, verbose=True)
                test_strings_time = time.time() - test_strings_start
                print(f"  [REGEX {lineno}] ✓ Cadenas generadas en {test_strings_time:.2f}s - Total: {len(test_strings_dict)} cadenas")
                sys.stdout.flush()
                
                # Convertir el diccionario a JSON string y validar
                json_start = time.time()
                json_string = json.dumps(test_strings_dict, ensure_ascii=False)
                # Validar que el JSON sea válido parseándolo de vuelta
                try:
                    json.loads(json_string)  # Validar que sea JSON válido
                    row["Clase"] = json_string
                except json.JSONDecodeError as json_err:
                    # Si hay error al validar, registrar en Error y dejar Clase vacía
                    error_msg = f"Línea {lineno}: Error al generar JSON válido para Clase: {json_err}"
                    row["Error"] = error_msg
                    row["Clase"] = ""
                    print(f"  [REGEX {lineno}] ⚠ ADVERTENCIA: JSON inválido generado - {json_err}")
                    sys.stdout.flush()
                json_time = time.time() - json_start
                print(f"  [REGEX {lineno}] ✓ JSON serializado y validado en {json_time:.3f}s - Tamaño: {len(row['Clase'])} caracteres")
                sys.stdout.flush()
                
            except Exception as e:
                import traceback
                # Registrar error pero continuar con las demás líneas
                error_msg = f"Línea {lineno}: {type(e).__name__}: {e}"
                row["Error"] = error_msg
                print(f"  [REGEX {lineno}] ✗ ERROR: {error_msg}")
                print(f"  [REGEX {lineno}] Traceback:")
                traceback.print_exc()
                sys.stdout.flush()
            
            # Escribir la fila
            write_start = time.time()
            writer.writerow(row)
            write_time = time.time() - write_start
            
            regex_total_time = time.time() - regex_start_time
            print(f"  [REGEX {lineno}] ✓ Fila escrita en {write_time:.3f}s")
            print(f"  [REGEX {lineno}] ⏱  Tiempo total para esta regex: {regex_total_time:.2f}s")
            
            # Mostrar tiempo estimado restante
            if lineno < total_regexes:
                avg_time_per_regex = (time.time() - process_start_time) / lineno
                remaining_regexes = total_regexes - lineno
                estimated_remaining = avg_time_per_regex * remaining_regexes
                print(f"  [REGEX {lineno}] 📊 Tiempo promedio: {avg_time_per_regex:.2f}s/regex | Estimado restante: {estimated_remaining:.1f}s ({remaining_regexes} regex)")
            print("-" * 80)
            sys.stdout.flush()
    
    process_total_time = time.time() - process_start_time
    print("\n" + "=" * 80)
    print(f"[PROCESS_CSV] ✓ Procesamiento completado en {process_total_time:.2f}s")
    print(f"[PROCESS_CSV] ✓ Archivo CSV generado: {output_csv}")
    print(f"[PROCESS_CSV] 📊 Estadísticas:")
    print(f"  - Total de regex procesadas: {total_regexes}")
    print(f"  - Tiempo total: {process_total_time:.2f}s")
    print(f"  - Tiempo promedio por regex: {process_total_time / total_regexes:.2f}s")
    print("=" * 80)
    sys.stdout.flush()

# ------------------------------ Reconocimiento con AFD ------------------------------

def dfa_accepts(dfa: DFA, cadena: str) -> bool:
    estado = dfa.start
    for ch in cadena:
        # si no hay transición definida para el símbolo, rechazo inmediato
        if ch not in dfa.trans.get(estado, {}):
            return False
        estado = dfa.trans[estado][ch]
    return estado in dfa.accepts

# ------------------------------ Dibujo AFD (networkx) ------------------------------

def draw_dfa_networkx(dfa: DFA):
    import matplotlib.pyplot as plt
    import networkx as nx

    aliases, _ = alias_dfa(dfa)

    G = nx.DiGraph()
    nodos = set(dfa.trans.keys()) | set(dfa.accepts) | {dfa.start}
    for S in nodos:
        G.add_node(aliases[S], accepting=(S in dfa.accepts))

    labels = {}
    for src, edges in dfa.trans.items():
        for sym, dst in edges.items():
            a_src = aliases[src]
            a_dst = aliases[dst]
            if (a_src, a_dst) not in labels:
                labels[(a_src, a_dst)] = []
            labels[(a_src, a_dst)].append(sym)
            G.add_edge(a_src, a_dst)

    pos = nx.spring_layout(G, k=0.9, seed=7)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_title("AFD (alias) – desde Thompson + Subconjuntos")
    ax.axis("off")

    node_linewidth = [2.8 if G.nodes[n].get("accepting", False) else 1.5 for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_color="#ffffff", node_size=1700,
                           edgecolors="black", linewidths=node_linewidth, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=10, ax=ax)
    nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle="-|>",
                           connectionstyle="arc3,rad=0.1", ax=ax)

    edge_labels = {(u, v): ",".join(sorted(syms)) for (u, v), syms in labels.items()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=9,
                                 label_pos=0.5, ax=ax)

    # flecha al inicio
    start_alias = aliases[dfa.start]
    if start_alias in pos:
        x, y = pos[start_alias]
        ax.annotate("", xy=(x, y), xycoords="data",
                    xytext=(x - 1.6, y), textcoords="data",
                    arrowprops=dict(arrowstyle="-|>", lw=1.6))
        ax.text(x - 1.7, y, "start", fontsize=9, va="center", ha="right")

    # 👇 no bloquear la ejecución:
    import matplotlib
    matplotlib.pyplot.show(block=False)

    return fig  # para cerrarla luego si quieres

# ------------------------------ Exportar AFD a JFLAP ------------------------------

def dfa_to_jff_string(dfa: DFA) -> str:
    """
    Convierte el AFD a formato JFLAP (.jff) como string.
    Internamente numera estados según el orden de alias.
    """
    aliases, rev = alias_dfa(dfa)
    
    # Obtener TODOS los estados del DFA para asegurar que todos tengan ID
    all_states = get_dfa_states(dfa)
    
    # Crear mapa de alias a ID numérico (0, 1, 2, ...)
    # Ordenar aliases por número (S0, S1, S2, ...)
    orden = sorted(rev.keys(), key=lambda a: int(a[1:]) if a[1:].isdigit() else 999)
    idmap = {alias: i for i, alias in enumerate(orden)}

    def st_id(S):  # id numérico desde el conjunto de estados
        if S not in aliases:
            # Esto no debería pasar, pero por seguridad
            raise ValueError(f"Estado {S} no tiene alias asignado")
        alias = aliases[S]
        if alias not in idmap:
            raise ValueError(f"Alias {alias} no está en idmap")
        return idmap[alias]
    
    def is_accept_alias(a):  # por alias
        return rev[a] in dfa.accepts

    lines = []
    lines.append('<?xml version="1.0" encoding="UTF-8" standalone="no"?>')
    lines.append('<structure>')
    lines.append('  <type>fa</type>')
    lines.append('  <automaton>')

    # estados: incluir TODOS los estados, no solo los que están en orden
    # Usar el orden de aliases para mantener consistencia
    for alias in orden:
        S = rev[alias]
        sid = st_id(S)
        lines.append(f'    <state id="{sid}" name="{alias}">')
        if S == dfa.start:
            lines.append('      <initial/>')
        if is_accept_alias(alias):
            lines.append('      <final/>')
        lines.append('    </state>')

    # transiciones: asegurarse de que todos los estados origen y destino existen
    for src, edges in dfa.trans.items():
        if src not in aliases:
            raise ValueError(f"Estado origen {src} no tiene alias")
        for sym, dst in edges.items():
            if dst not in aliases:
                raise ValueError(f"Estado destino {dst} no tiene alias")
            lines.append('    <transition>')
            lines.append(f'      <from>{st_id(src)}</from>')
            lines.append(f'      <to>{st_id(dst)}</to>')
            lines.append(f'      <read>{sym}</read>')
            lines.append('    </transition>')

    lines.append('  </automaton>')
    lines.append('</structure>')
    return "\n".join(lines)

def export_dfa_jff(dfa: DFA, path: str):
    """
    Exporta el AFD a JFLAP (.jff). Internamente numera estados según el orden de alias.
    """
    jff_content = dfa_to_jff_string(dfa)
    with open(path, "w", encoding="utf-8") as f:
        f.write(jff_content)

# ------------------------------ CLI ------------------------------

if __name__ == "__main__":
    # Uso:
    #   python thompson_nfa.py "<regex>"
    #   python thompson_nfa.py --batch=entrada.txt --csv=salida.csv
    if len(sys.argv) < 2:
        print("Uso:")
        print("  python thompson_nfa.py '<regex>'")
        print("  python thompson_nfa.py --batch=entrada.txt --csv=salida.csv")
        print("  python thompson_nfa.py --batch=entrada.txt --csv=salida.csv --with-clase")
        sys.exit(0)

    # Modo batch (archivo -> csv)
    batch_arg = next((a for a in sys.argv[1:] if a.startswith("--batch=")), None)
    csv_arg   = next((a for a in sys.argv[1:] if a.startswith("--csv=")), None)
    with_clase = any(a == "--with-clase" for a in sys.argv[1:])
    
    if batch_arg:
        input_path = batch_arg.split("=", 1)[1]
        output_csv = (csv_arg.split("=", 1)[1] if csv_arg else "resultado.csv")
        try:
            if with_clase:
                process_regex_file_to_csv_with_clase(input_path, output_csv)
            else:
                process_regex_file_to_csv(input_path, output_csv)
            print(f"[OK] CSV generado: {output_csv}")
        except Exception as e:
            print(f"[!] Error en modo batch: {e}")
        sys.exit(0)

    # --- Modo interactivo de siempre (una regex) ---
    regex = sys.argv[1]

    # 1) Regex -> NFA ; 2) NFA -> DFA
    nfa = regex_to_nfa(regex)
    dfa = nfa_to_dfa(nfa)

    # 3) Mostrar información completa del AFD y graficar sin bloquear
    print(f"Regex: {regex}")
    print_dfa_complete(dfa)
    fig = None
    try:
        fig = draw_dfa_networkx(dfa)  # no bloquea
    except ImportError:
        print("\n[!] Para ver el gráfico instala:")
        print("    pip install networkx matplotlib")

    # 3.5) Exportar a JFLAP si se pasó --jff=archivo.jff (opcional)
    try:
        jff_arg = next((a for a in sys.argv[2:] if a.startswith("--jff=")), None)
        if jff_arg:
            export_dfa_jff(dfa, jff_arg.split("=",1)[1])
            print("[OK] Exportado JFLAP:", jff_arg.split("=",1)[1])
    except Exception as e:
        print("[!] Error exportando JFLAP:", e)

    # 4) Loop de prueba de cadenas
    print("\nIngresa cadenas para probar contra el AFD (Enter vacío para salir):")
    try:
        while True:
            cad = input("> ").strip()
            if cad == "":
                break
            ok = dfa_accepts(dfa, cad)
            print("  ✅ Aceptada" if ok else "  ❌ Rechazada")
    except KeyboardInterrupt:
        pass

    # 5) Cerrar la ventana del gráfico al salir (si existe)
    if fig is not None:
        try:
            import matplotlib.pyplot as plt
            plt.close(fig)
        except Exception:
            pass
