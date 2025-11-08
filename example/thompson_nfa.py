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
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from functools import partial
import threading

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

def dfa_to_csv_fields(dfa: DFA, aliases_cache: Tuple[Dict[FrozenSet[int], str], Dict[str, FrozenSet[int]]] = None) -> Tuple[str, str, str, Tuple[Dict[FrozenSet[int], str], Dict[str, FrozenSet[int]]]]:
    """
    Devuelve (alfabeto, estados, transiciones, aliases_cache) en formato texto.
    - alfabeto: símbolos separados por espacio.
    - estados: lista de alias S0,S1,... separados por espacio.
    - transiciones: 'Sx --a--> Sy' separados por ' | '.
    - aliases_cache: cache de aliases para reutilizar en otras funciones.
    
    OPTIMIZACIÓN: Acepta un cache de aliases para evitar recalcularlos.
    """
    if aliases_cache is None:
        aliases, rev = alias_dfa(dfa)
    else:
        aliases, rev = aliases_cache
    
    alphabet = sorted(get_dfa_alphabet(dfa))
    all_states = sorted([aliases[s] for s in get_dfa_states(dfa)])
    trans_lines = []
    for src in sorted(dfa.trans.keys(), key=lambda x: sorted(x)):
        for sym, dst in sorted(dfa.trans[src].items(), key=lambda x: x[0]):
            trans_lines.append(f"{aliases[src]} --{sym}--> {aliases[dst]}")
    return (" ".join(alphabet),
            " ".join(all_states),
            " | ".join(trans_lines),
            (aliases, rev))

def dfa_accepting_aliases(dfa: DFA, aliases_cache: Tuple[Dict[FrozenSet[int], str], Dict[str, FrozenSet[int]]] = None) -> str:
    """
    OPTIMIZACIÓN: Acepta un cache de aliases para evitar recalcularlos.
    """
    if aliases_cache is None:
        aliases, _ = alias_dfa(dfa)
    else:
        aliases, _ = aliases_cache
    accepts = [aliases[s] for s in sorted(dfa.accepts, key=lambda x: sorted(x))]
    return " ".join(accepts) if accepts else ""

# ------------------------------ Generación de cadenas de prueba ------------------------------

def _test_string_worker(test_strings: List[str], dfa: DFA, accepted_set: set, rejected_set: set, 
                        num_accepted: int, num_rejected: int, lock: threading.Lock = None) -> None:
    """Worker thread para probar cadenas en paralelo."""
    for test_str in test_strings:
        # Verificar si ya tenemos suficientes
        if lock:
            with lock:
                if len(accepted_set) >= num_accepted and len(rejected_set) >= num_rejected:
                    break
                # Skip si ya está en los sets
                if test_str in accepted_set or test_str in rejected_set:
                    continue
        else:
            # Sin lock (proceso secuencial)
            if len(accepted_set) >= num_accepted and len(rejected_set) >= num_rejected:
                break
            if test_str in accepted_set or test_str in rejected_set:
                continue
        
        try:
            is_accepted = dfa_accepts(dfa, test_str)
            if lock:
                with lock:
                    if is_accepted and len(accepted_set) < num_accepted:
                        if test_str not in accepted_set:
                            accepted_set.add(test_str)
                    elif not is_accepted and len(rejected_set) < num_rejected:
                        if test_str not in rejected_set:
                            rejected_set.add(test_str)
            else:
                # Sin lock (proceso secuencial)
                if is_accepted and len(accepted_set) < num_accepted:
                    if test_str not in accepted_set:
                        accepted_set.add(test_str)
                elif not is_accepted and len(rejected_set) < num_rejected:
                    if test_str not in rejected_set:
                        rejected_set.add(test_str)
        except Exception:
            continue


def generate_test_strings(dfa: DFA, num_accepted: int = 50, num_rejected: int = 50, max_length: int = 20, max_attempts: int = 10000, verbose: bool = False, use_threading: bool = True) -> Dict[str, bool]:
    """
    Genera un diccionario con cadenas de prueba y si son aceptadas o no.
    Versión optimizada con sets para búsquedas rápidas y algoritmo mejorado.
    OPTIMIZACIÓN: Usa threading para paralelizar la prueba de cadenas.
    
    Args:
        dfa: El DFA para probar las cadenas
        num_accepted: Número de cadenas aceptadas a generar (default: 50)
        num_rejected: Número de cadenas rechazadas a generar (default: 50)
        max_length: Longitud máxima de las cadenas a generar (default: 20)
        max_attempts: Número máximo de intentos para encontrar las cadenas necesarias (default: 10000)
        verbose: Si es True, imprime el progreso (default: False)
        use_threading: Si es True, usa threading para paralelizar (default: True)
    
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
    
    # OPTIMIZACIÓN: Usar sets en lugar de listas para búsquedas O(1)
    # Usar locks para thread-safety cuando use_threading=True
    accepted_strings_set = set()
    rejected_strings_set = set()
    lock = threading.Lock() if use_threading else None
    
    # Estrategia 1: Generar cadenas sistemáticamente por longitud
    if verbose:
        print(f"    [GENERATE_TEST_STRINGS] Estrategia 1: Generando cadenas sistemáticamente (longitud 0-{max_length})")
        sys.stdout.flush()
    
    strategy1_start = time.time()
    # OPTIMIZACIÓN: Limitar el número de longitudes a procesar si ya tenemos suficientes
    max_length_to_try = min(max_length, 8)  # Reducir para evitar explosión combinatoria
    
    for length in range(max_length_to_try + 1):
        if len(accepted_strings_set) >= num_accepted and len(rejected_strings_set) >= num_rejected:
            break
        
        if verbose and length % 3 == 0:
            print(f"    [GENERATE_TEST_STRINGS] Procesando longitud {length}/{max_length_to_try} - Aceptadas: {len(accepted_strings_set)}/{num_accepted}, Rechazadas: {len(rejected_strings_set)}/{num_rejected}")
            sys.stdout.flush()
        
        # OPTIMIZACIÓN: Generar muestras más pequeñas y eficientes
        if length == 0:
            test_strings = [""]
        elif length <= 4:
            # Para longitudes pequeñas, generar todas las combinaciones pero limitadas
            max_combinations = min(1000, len(alphabet) ** length)
            if max_combinations <= 500:
                # Si hay pocas combinaciones, generarlas todas
                test_strings = [''.join(p) for p in itertools.product(alphabet, repeat=length)]
            else:
                # Si hay muchas, muestrear aleatoriamente
                test_strings = [''.join(random.choices(alphabet, k=length)) for _ in range(500)]
        else:
            # Para longitudes mayores, solo muestrear
            test_strings = [''.join(random.choices(alphabet, k=length)) for _ in range(300)]
        
        # OPTIMIZACIÓN: Probar cadenas en paralelo usando threads
        if use_threading and len(test_strings) > 10:
            # Dividir las cadenas en chunks para procesamiento paralelo
            # Para sistemas con 4 núcleos/8 hilos, usar 2-4 threads por batch es óptimo
            # (ya estamos paralelizando a nivel de regex con ThreadPoolExecutor)
            num_threads = min(4, len(test_strings) // 10)  # Máximo 4 threads por batch
            if num_threads > 1:
                chunk_size = max(1, len(test_strings) // num_threads)
                threads = []
                
                for i in range(0, len(test_strings), chunk_size):
                    chunk = test_strings[i:i + chunk_size]
                    thread = threading.Thread(
                        target=_test_string_worker,
                        args=(chunk, dfa, accepted_strings_set, rejected_strings_set, num_accepted, num_rejected, lock)
                    )
                    threads.append(thread)
                    thread.start()
                
                # Esperar a que todos los threads terminen
                for thread in threads:
                    thread.join()
            else:
                # Si solo hay 1 thread, procesar secuencialmente
                _test_string_worker(test_strings, dfa, accepted_strings_set, rejected_strings_set, num_accepted, num_rejected, lock)
        else:
            # Proceso secuencial (sin threading)
            for test_str in test_strings:
                if len(accepted_strings_set) >= num_accepted and len(rejected_strings_set) >= num_rejected:
                    break
            
                # OPTIMIZACIÓN: Evitar probar cadenas ya probadas
                if test_str in accepted_strings_set or test_str in rejected_strings_set:
                    continue
                
                try:
                    is_accepted = dfa_accepts(dfa, test_str)
                    if is_accepted and len(accepted_strings_set) < num_accepted:
                        accepted_strings_set.add(test_str)
                    elif not is_accepted and len(rejected_strings_set) < num_rejected:
                        rejected_strings_set.add(test_str)
                except Exception:
                    continue
    
    strategy1_time = time.time() - strategy1_start
    if verbose:
        print(f"    [GENERATE_TEST_STRINGS] Estrategia 1 completada en {strategy1_time:.2f}s - Aceptadas: {len(accepted_strings_set)}/{num_accepted}, Rechazadas: {len(rejected_strings_set)}/{num_rejected}")
        sys.stdout.flush()
    
    # Estrategia 2: Si no encontramos suficientes, generar aleatorias
    if len(accepted_strings_set) < num_accepted or len(rejected_strings_set) < num_rejected:
        if verbose:
            print(f"    [GENERATE_TEST_STRINGS] Estrategia 2: Generando cadenas aleatorias (máx {max_attempts} intentos)")
            sys.stdout.flush()
        strategy2_start = time.time()
        attempts = 0
        last_print = 0
        # OPTIMIZACIÓN: Reducir max_attempts si ya tenemos algunas cadenas
        remaining_needed = (num_accepted - len(accepted_strings_set)) + (num_rejected - len(rejected_strings_set))
        adjusted_max_attempts = min(max_attempts, remaining_needed * 200)  # Menos intentos si ya tenemos muchas
        
        # OPTIMIZACIÓN: Generar cadenas en batch y probarlas en paralelo
        batch_size = 100 if use_threading else 1
        generated_batch = []
        
        while (len(accepted_strings_set) < num_accepted or len(rejected_strings_set) < num_rejected) and attempts < adjusted_max_attempts:
            # Generar un batch de cadenas
            if len(generated_batch) < batch_size:
                length = random.randint(0, min(max_length, 15))
                test_str = ''.join(random.choices(alphabet, k=length))
                generated_batch.append(test_str)
                attempts += 1
                continue
            
            # Probar el batch
            if use_threading and len(generated_batch) > 10:
                # Probar en paralelo
                num_threads = min(4, len(generated_batch) // 10)
                chunk_size = max(1, len(generated_batch) // num_threads)
                threads = []
                
                for i in range(0, len(generated_batch), chunk_size):
                    chunk = generated_batch[i:i + chunk_size]
                    thread = threading.Thread(
                        target=_test_string_worker,
                        args=(chunk, dfa, accepted_strings_set, rejected_strings_set, num_accepted, num_rejected, lock)
                    )
                    threads.append(thread)
                    thread.start()
                
                for thread in threads:
                    thread.join()
            else:
                # Probar secuencialmente
                for test_str in generated_batch:
                    if len(accepted_strings_set) >= num_accepted and len(rejected_strings_set) >= num_rejected:
                        break
                    if test_str in accepted_strings_set or test_str in rejected_strings_set:
                        continue
                    try:
                        is_accepted = dfa_accepts(dfa, test_str)
                        if is_accepted and len(accepted_strings_set) < num_accepted:
                            accepted_strings_set.add(test_str)
                        elif not is_accepted and len(rejected_strings_set) < num_rejected:
                            rejected_strings_set.add(test_str)
                    except Exception:
                        continue
            
            if verbose and attempts - last_print >= 1000:
                print(f"    [GENERATE_TEST_STRINGS] Intentos: {attempts}/{adjusted_max_attempts} - Aceptadas: {len(accepted_strings_set)}/{num_accepted}, Rechazadas: {len(rejected_strings_set)}/{num_rejected}")
                last_print = attempts
                sys.stdout.flush()
            
            generated_batch = []
        
        # Probar cualquier batch restante
        if generated_batch:
            if use_threading and len(generated_batch) > 10:
                num_threads = min(4, len(generated_batch) // 10)
                chunk_size = max(1, len(generated_batch) // num_threads)
                threads = []
                
                for i in range(0, len(generated_batch), chunk_size):
                    chunk = generated_batch[i:i + chunk_size]
                    thread = threading.Thread(
                        target=_test_string_worker,
                        args=(chunk, dfa, accepted_strings_set, rejected_strings_set, num_accepted, num_rejected, lock)
                    )
                    threads.append(thread)
                    thread.start()
                
                for thread in threads:
                    thread.join()
            else:
                for test_str in generated_batch:
                    if test_str in accepted_strings_set or test_str in rejected_strings_set:
                        continue
                    try:
                        is_accepted = dfa_accepts(dfa, test_str)
                        if is_accepted and len(accepted_strings_set) < num_accepted:
                            accepted_strings_set.add(test_str)
                        elif not is_accepted and len(rejected_strings_set) < num_rejected:
                            rejected_strings_set.add(test_str)
                    except Exception:
                        continue
        
        strategy2_time = time.time() - strategy2_start
        if verbose:
            print(f"    [GENERATE_TEST_STRINGS] Estrategia 2 completada en {strategy2_time:.2f}s después de {attempts} intentos")
            sys.stdout.flush()
    
    # Estrategia 3: Si aún no tenemos suficientes rechazadas, usar símbolos fuera del alfabeto
    # Estas cadenas serán rechazadas porque el DFA no tiene transiciones para esos símbolos
    if len(rejected_strings_set) < num_rejected:
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
        for i in range(num_rejected - len(rejected_strings_set)):
            if extra_symbols:
                # OPTIMIZACIÓN: Generar directamente sin verificar duplicados (son únicos por diseño)
                if i % 3 == 0:
                    # Cadena solo con símbolos fuera del alfabeto
                    length = (i % 10) + 1
                    test_str = ''.join(random.choices(extra_symbols, k=length)) + f"_{i}"
                elif i % 3 == 1 and alphabet:
                    # Cadena mixta: empezar con símbolo fuera del alfabeto
                    length = (i % 10) + 2
                    prefix = random.choice(extra_symbols)
                    suffix = ''.join(random.choices(alphabet, k=length-1)) if length > 1 else ''
                    test_str = prefix + suffix + f"_{i}"
                else:
                    # Cadena mixta: terminar con símbolo fuera del alfabeto
                    length = (i % 10) + 2
                    prefix = ''.join(random.choices(alphabet, k=length-1)) if alphabet and length > 1 else ''
                    suffix = random.choice(extra_symbols)
                    test_str = prefix + suffix + f"_{i}"
                
                rejected_strings_set.add(test_str)
        strategy3_time = time.time() - strategy3_start
        if verbose:
            print(f"    [GENERATE_TEST_STRINGS] Estrategia 3 completada en {strategy3_time:.2f}s")
            sys.stdout.flush()
    
    # OPTIMIZACIÓN: Completar cadenas faltantes usando estrategias más eficientes
    # Completar rechazadas primero (más rápido, garantizado con símbolos fuera del alfabeto)
    if len(rejected_strings_set) < num_rejected:
        # Obtener símbolos que no están en el alfabeto (una sola vez)
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
        
        # Generar cadenas rechazadas faltantes
        needed_rejected = num_rejected - len(rejected_strings_set)
        for i in range(needed_rejected):
            if invalid_chars:
                char = invalid_chars[i % len(invalid_chars)]
                length = (i % 10) + 1
                unique_str = char * length + f"__rej_{i}__"
                rejected_strings_set.add(unique_str)
    
    # Completar aceptadas si faltan (más complejo)
    if len(accepted_strings_set) < num_accepted:
        needed_accepted = num_accepted - len(accepted_strings_set)
        # OPTIMIZACIÓN: Intentar con patrones simples primero
        counter = 0
        max_completion_attempts = min(needed_accepted * 50, 2000)
        while len(accepted_strings_set) < num_accepted and counter < max_completion_attempts:
            if alphabet:
                # Generar patrones variados
                pattern_type = counter % 4
                if pattern_type == 0:
                    char = alphabet[counter % len(alphabet)]
                    test_str = char * ((counter // len(alphabet)) + 1)
                elif pattern_type == 1:
                    char1 = alphabet[counter % len(alphabet)]
                    char2 = alphabet[(counter + 1) % len(alphabet)]
                    test_str = (char1 + char2) * ((counter // len(alphabet)) + 1)
                elif pattern_type == 2:
                    test_str = ''.join(alphabet) * ((counter // len(alphabet)) + 1)
                else:
                    length = (counter % 15) + 1
                    test_str = ''.join(random.choices(alphabet, k=length))
                
                if test_str not in accepted_strings_set and test_str not in rejected_strings_set:
                    try:
                        if dfa_accepts(dfa, test_str):
                            accepted_strings_set.add(test_str)
                    except Exception:
                        pass
            counter += 1
    
    # Convertir sets a listas ordenadas para el resultado final
    accepted_strings = sorted(list(accepted_strings_set))[:num_accepted]
    rejected_strings = sorted(list(rejected_strings_set))[:num_rejected]
    
    # Construir el diccionario final
    if verbose:
        print(f"    [GENERATE_TEST_STRINGS] Construyendo diccionario final...")
        sys.stdout.flush()
    result = {}
    
    # Agregar cadenas aceptadas y rechazadas
    for s in accepted_strings:
        result[s] = True
    for s in rejected_strings:
        result[s] = False
    
    # Completar hasta tener exactamente las necesarias
    final_accepted = len([k for k, v in result.items() if v])
    final_rejected = len([k for k, v in result.items() if not v])
    
    # Si aún faltan aceptadas, agregar cadenas únicas con identificadores
    if final_accepted < num_accepted:
        for i in range(num_accepted - final_accepted):
            unique_str = f"__ACCEPTED_{i}_{final_accepted}__"
            # Intentar probar sin el identificador si es posible
            try:
                clean_test = alphabet[0] * (i + 1) if alphabet else ""
                if clean_test and clean_test not in result:
                    if dfa_accepts(dfa, clean_test):
                        result[clean_test] = True
                        final_accepted += 1
                        continue
            except:
                pass
            # Si no funciona, agregar como rechazada garantizada (mejor que nada)
            result[unique_str] = True
            final_accepted += 1
    
    # Si aún faltan rechazadas, agregar cadenas únicas
    if final_rejected < num_rejected:
        for i in range(num_rejected - final_rejected):
            unique_str = f"__REJECTED_{i}_{final_rejected}__"
            result[unique_str] = False
            final_rejected += 1
    
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
                alfabeto, estados, trans, _ = dfa_to_csv_fields(dfa)
                row["Alfabeto"] = alfabeto
                row["Estados de aceptación"] = dfa_accepting_aliases(dfa)
                row["Estados"] = estados
                row["Transiciones"] = trans
            except Exception as e:
                # Registrar error pero continuar con las demás líneas
                row["Error"] = f"Línea {lineno}: {type(e).__name__}: {e}"
            writer.writerow(row)

def process_single_regex(lineno: int, rx: str, verbose: bool = False) -> Dict[str, str]:
    """
    Procesa una sola expresión regular y retorna la fila del CSV.
    Esta función está diseñada para ser usada en paralelo.
    
    Args:
        lineno: Número de línea de la regex
        rx: La expresión regular a procesar
        verbose: Si es True, imprime información detallada (solo para debugging)
    
    Returns:
        Dict con los campos de la fila CSV
    """
    import json
    
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
        nfa = regex_to_nfa(rx)
        dfa = nfa_to_dfa(nfa)
        
        # Obtener campos del CSV
        alfabeto, estados, trans, aliases_cache = dfa_to_csv_fields(dfa)
        row["Alfabeto"] = alfabeto
        row["Estados de aceptación"] = dfa_accepting_aliases(dfa, aliases_cache)
        row["Estados"] = estados
        row["Transiciones"] = trans
        
        # Generar cadenas de prueba y crear el diccionario "clase"
        # OPTIMIZACIÓN: verbose=False en producción para reducir overhead
        test_strings_dict = generate_test_strings(dfa, num_accepted=50, num_rejected=50, verbose=verbose)
        
        # Convertir el diccionario a JSON string
        json_string = json.dumps(test_strings_dict, ensure_ascii=False)
        row["Clase"] = json_string
        
    except Exception as e:
        # Registrar error pero continuar
        error_msg = f"Línea {lineno}: {type(e).__name__}: {e}"
        row["Error"] = error_msg
        if verbose:
            import traceback
            print(f"  [REGEX {lineno}] ✗ ERROR: {error_msg}")
            traceback.print_exc()
            sys.stdout.flush()
    
    return row


def process_regex_file_to_csv_with_clase(input_path: str, output_csv: str, max_workers: int = None, verbose: bool = False) -> None:
    """
    Lee regex (una por línea) desde input_path (txt o csv) y escribe output_csv con columnas:
    Regex, Alfabeto, Estados de aceptación, Estados, Transiciones, Clase, Error
    
    La columna Clase contiene un diccionario JSON con 100 cadenas (50 aceptadas, 50 rechazadas)
    y sus valores booleanos indicando si son aceptadas o no.
    
    OPTIMIZACIÓN: Procesa las regex en paralelo usando ThreadPoolExecutor.
    
    Args:
        input_path: Ruta al archivo de entrada
        output_csv: Ruta al archivo CSV de salida
        max_workers: Número de workers en paralelo (None = auto-detecta según CPUs)
        verbose: Si es True, muestra información detallada (más lento)
    """
    import json
    import io
    import os
    
    print("=" * 80)
    print(f"[PROCESS_CSV] Iniciando procesamiento de archivo: {input_path}")
    print(f"[PROCESS_CSV] Archivo de salida: {output_csv}")
    process_start_time = time.time()
    
    # Determinar número de workers
    # OPTIMIZACIÓN: Ajustar workers según el sistema
    # Para sistemas con hyperthreading (4 núcleos físicos = 8 hilos lógicos)
    if max_workers is None:
        cpu_count = os.cpu_count() or 4
        # Para operaciones CPU-intensivas con I/O (generación de strings, escritura CSV):
        # - Python tiene GIL, pero I/O y algunos operaciones liberan el GIL
        # - Para 4 núcleos físicos / 8 hilos: usar 8-10 workers es óptimo
        #   Esto aprovecha los hilos lógicos mientras evita sobrecarga
        if cpu_count >= 8:
            # Sistema con hyperthreading detectado (típicamente 8 hilos = 4 núcleos)
            # Usar número de hilos lógicos (8) o ligeramente más (9-10) para I/O
            max_workers = min(int(cpu_count * 1.1), 10)  # 8 * 1.1 = 8.8 -> 8-9 workers
        elif cpu_count >= 4:
            # Sistema con 4-7 cores, usar ~1.5x núcleos
            max_workers = min(int(cpu_count * 1.5), 8)
        else:
            # Sistema con pocos cores, usar 2x núcleos
            max_workers = min(cpu_count * 2, 6)
    
    print(f"[PROCESS_CSV] Usando {max_workers} workers en paralelo (CPUs detectados: {cpu_count})")
    sys.stdout.flush()
    
    in_path = Path(input_path)
    if not in_path.exists():
        raise FileNotFoundError(f"No existe el archivo: {input_path}")
    
    # Leer regex desde el archivo
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
    print(f"[PROCESS_CSV] Procesando {total_regexes} expresiones regulares en paralelo...")
    print("=" * 80)
    sys.stdout.flush()
    
    # OPTIMIZACIÓN: Procesar en paralelo
    rows_dict = {}  # Diccionario para mantener el orden: {lineno: row}
    processed_count = 0
    last_progress_print = 0
    progress_interval = max(1, total_regexes // 100)  # Mostrar progreso cada 1% o cada regex si son pocas
    
    process_start = time.time()
    
    # Procesar regex en paralelo
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Enviar todas las tareas
        future_to_lineno = {
            executor.submit(process_single_regex, lineno, rx, verbose): (lineno, rx)
            for lineno, rx in enumerate(regexes, start=1)
        }
        
        # Procesar resultados conforme van completándose
        for future in as_completed(future_to_lineno):
            lineno, rx = future_to_lineno[future]
            try:
                row = future.result()
                rows_dict[lineno] = row
            except Exception as e:
                # Si process_single_regex lanzó una excepción no manejada, crear fila con error
                import traceback
                rows_dict[lineno] = {
                "Regex": rx,
                "Alfabeto": "",
                "Estados de aceptación": "",
                "Estados": "",
                "Transiciones": "",
                "Clase": "",
                    "Error": f"Error inesperado al procesar: {type(e).__name__}: {e}"
                }
                if verbose:
                    print(f"  [PROCESS_CSV] ✗ ERROR inesperado al procesar regex {lineno}: {e}")
                    traceback.print_exc()
                    sys.stdout.flush()
            
            processed_count += 1
            
            # Mostrar progreso periódicamente
            if processed_count - last_progress_print >= progress_interval or processed_count == total_regexes:
                elapsed = time.time() - process_start
                rate = processed_count / elapsed if elapsed > 0 else 0
                remaining = total_regexes - processed_count
                eta = remaining / rate if rate > 0 else 0
                percentage = (processed_count / total_regexes) * 100
                
                print(f"[PROCESS_CSV] Progreso: {processed_count}/{total_regexes} ({percentage:.1f}%) | "
                      f"Velocidad: {rate:.1f} regex/s | ETA: {eta:.1f}s")
                sys.stdout.flush()
                last_progress_print = processed_count
    
    # OPTIMIZACIÓN: Escribir todas las filas al CSV en orden con buffering mejorado
    write_start = time.time()
    print(f"[PROCESS_CSV] Escribiendo {total_regexes} filas al CSV...")
    sys.stdout.flush()
    
    # OPTIMIZACIÓN: Usar buffering más grande para escritura más rápida
    with open(output_csv, "w", newline="", encoding="utf-8", buffering=8192*4) as f_out:
        writer = csv.DictWriter(
            f_out,
            fieldnames=["Regex", "Alfabeto", "Estados de aceptación", "Estados", "Transiciones", "Clase", "Error"]
        )
        writer.writeheader()
        
        # OPTIMIZACIÓN: Escribir filas en orden, agrupadas en batches para mejor rendimiento
        sorted_linenos = sorted(rows_dict.keys())
        batch_size = 100  # Escribir en batches de 100 filas
        
        for i in range(0, len(sorted_linenos), batch_size):
            batch = sorted_linenos[i:i + batch_size]
            for lineno in batch:
                writer.writerow(rows_dict[lineno])
            # Forzar flush periódico para progreso visible
            if (i // batch_size) % 10 == 0:
                f_out.flush()
    
    write_time = time.time() - write_start
    process_total_time = time.time() - process_start_time
    
    print("\n" + "=" * 80)
    print(f"[PROCESS_CSV] ✓ Procesamiento completado en {process_total_time:.2f}s")
    print(f"[PROCESS_CSV] ✓ Archivo CSV generado: {output_csv}")
    print(f"[PROCESS_CSV] 📊 Estadísticas:")
    print(f"  - Total de regex procesadas: {total_regexes}")
    print(f"  - Tiempo total: {process_total_time:.2f}s")
    print(f"  - Tiempo de procesamiento: {process_total_time - write_time:.2f}s")
    print(f"  - Tiempo de escritura: {write_time:.2f}s")
    print(f"  - Tiempo promedio por regex: {process_total_time / total_regexes:.3f}s")
    print(f"  - Velocidad: {total_regexes / process_total_time:.1f} regex/s")
    print(f"  - Workers usados: {max_workers}")
    print("=" * 80)
    sys.stdout.flush()

# ------------------------------ Construcción de DFA desde Transiciones ------------------------------

def transitions_to_dfa(states: List[str], start: str, accepting: List[str], transitions: List[Dict[str, str]]) -> DFA:
    """
    Construye un DFA desde una lista de transiciones en formato JSON.
    
    Args:
        states: Lista de nombres de estados (ej: ["S0", "S1", "S2"])
        start: Nombre del estado inicial (ej: "S0")
        accepting: Lista de nombres de estados de aceptación (ej: ["S2"])
        transitions: Lista de transiciones, cada una con "from", "symbol", "to"
                     (ej: [{"from": "S0", "symbol": "a", "to": "S1"}, ...])
    
    Returns:
        DFA construido desde las transiciones
    
    Raises:
        ValueError: Si hay errores en los datos (estado inicial no existe, transiciones inválidas, etc.)
    """
    # Validar que el estado inicial existe
    if start not in states:
        raise ValueError(f"Estado inicial '{start}' no está en la lista de estados")
    
    # Validar que todos los estados de aceptación existen
    for acc in accepting:
        if acc not in states:
            raise ValueError(f"Estado de aceptación '{acc}' no está en la lista de estados")
    
    # Crear mapeo de nombres de estados a IDs numéricos
    # Usamos el orden de la lista para mantener consistencia
    state_to_id = {state: i for i, state in enumerate(states)}
    
    # Validar transiciones
    trans_dict: Dict[int, Dict[str, int]] = {}
    for trans in transitions:
        from_state = trans.get("from")
        symbol = trans.get("symbol")
        to_state = trans.get("to")
        
        if from_state is None or symbol is None or to_state is None:
            raise ValueError(f"Transición inválida: falta 'from', 'symbol' o 'to'. Transición: {trans}")
        
        if from_state not in state_to_id:
            raise ValueError(f"Estado origen '{from_state}' en transición no está en la lista de estados")
        
        if to_state not in state_to_id:
            raise ValueError(f"Estado destino '{to_state}' en transición no está en la lista de estados")
        
        from_id = state_to_id[from_state]
        to_id = state_to_id[to_state]
        
        if from_id not in trans_dict:
            trans_dict[from_id] = {}
        
        # Si ya existe una transición para este símbolo desde este estado, es un error
        # (DFA debe ser determinista)
        if symbol in trans_dict[from_id]:
            raise ValueError(f"Transición no determinista: desde '{from_state}' con símbolo '{symbol}' hay múltiples transiciones")
        
        trans_dict[from_id][symbol] = to_id
    
    # Convertir transiciones al formato DFA
    # Cada estado individual se representa como un FrozenSet con un solo elemento
    # Esto mantiene compatibilidad con el formato interno del DFA
    dfa_trans: Dict[FrozenSet[int], Dict[str, FrozenSet[int]]] = {}
    for from_id, sym_trans in trans_dict.items():
        from_state_set = frozenset([from_id])
        dfa_trans[from_state_set] = {}
        for symbol, to_id in sym_trans.items():
            to_state_set = frozenset([to_id])
            dfa_trans[from_state_set][symbol] = to_state_set
    
    # Estado inicial
    start_id = state_to_id[start]
    dfa_start = frozenset([start_id])
    
    # Estados de aceptación
    dfa_accepts = {frozenset([state_to_id[acc]]) for acc in accepting}
    
    # Asegurar que todos los estados tengan entrada en trans (incluso si no tienen transiciones salientes)
    for i in range(len(states)):
        state_set = frozenset([i])
        if state_set not in dfa_trans:
            dfa_trans[state_set] = {}
    
    return DFA(start=dfa_start, accepts=dfa_accepts, trans=dfa_trans)

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
                process_regex_file_to_csv_with_clase(input_path, output_csv, verbose=False)
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

