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
        else:
            tokens.append(c)
            i += 1
    return tokens

def is_literal(tok: str) -> bool:
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
    """
    estados = set(dfa.trans.keys()) | set(dfa.accepts) | {dfa.start}
    
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
    # mapa alias -> id numérico jflap
    orden = sorted(rev.keys(), key=lambda a: int(a[1:]))  # S0,S1,...
    idmap = {alias: i for i, alias in enumerate(orden)}

    def st_id(S):  # id numérico desde el conjunto
        return idmap[aliases[S]]
    def is_accept_alias(a):  # por alias
        return rev[a] in dfa.accepts

    lines = []
    lines.append('<?xml version="1.0" encoding="UTF-8" standalone="no"?>')
    lines.append('<structure>')
    lines.append('  <type>fa</type>')
    lines.append('  <automaton>')

    # estados
    for alias in orden:
        S = rev[alias]
        sid = st_id(S)
        lines.append(f'    <state id="{sid}" name="{alias}">')
        if S == dfa.start:
            lines.append('      <initial/>')
        if is_accept_alias(alias):
            lines.append('      <final/>')
        lines.append('    </state>')

    # transiciones
    for src, edges in dfa.trans.items():
        for sym, dst in edges.items():
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
        sys.exit(0)

    # Modo batch (archivo -> csv)
    batch_arg = next((a for a in sys.argv[1:] if a.startswith("--batch=")), None)
    csv_arg   = next((a for a in sys.argv[1:] if a.startswith("--csv=")), None)
    if batch_arg:
        input_path = batch_arg.split("=", 1)[1]
        output_csv = (csv_arg.split("=", 1)[1] if csv_arg else "resultado.csv")
        try:
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
