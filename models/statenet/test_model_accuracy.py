#!/usr/bin/env python3
"""
Prueba la precisión del modelo comparándolo con el DFA real.

Muestra una tabla con cadenas de prueba y si coinciden o no.
"""

import argparse
import json
import re
from pathlib import Path

import pandas as pd
import torch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STATES = ROOT / "artifacts" / "statenet" / "states_for_acceptnet.pt"
DEFAULT_DATASET = ROOT / "data" / "dataset6000.csv"

SYMBOLS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"]
SYMBOL_TO_IDX = {sym: idx for idx, sym in enumerate(SYMBOLS)}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prueba la precisión del modelo comparándolo con el DFA real.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python scripts/test_model_accuracy.py "[LCIG]+" "C" "G" "CG" "IL" "CGIL"
  python scripts/test_model_accuracy.py "[ABCH]*" "A" "B" "AB" "ABC" "ABCH"
  python scripts/test_model_accuracy.py "[LCIG]+" --dfa-id 0
        """
    )
    parser.add_argument(
        "regex",
        nargs="?",
        type=str,
        help="Regex a buscar en dataset6000.csv",
    )
    parser.add_argument(
        "strings",
        nargs="*",
        metavar="CADENA",
        help="Cadenas a probar con el modelo (se esperan 5 cadenas, o se usan del dataset si no se especifican)",
    )
    
    parser.add_argument(
        "--dfa-id",
        type=int,
        default=None,
        help="ID numérico del DFA (0-5999). Si se especifica, ignora regex y usa cadenas del dataset",
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=DEFAULT_DATASET,
        help="Ruta a dataset6000.csv",
    )
    parser.add_argument(
        "--states-path",
        type=Path,
        default=DEFAULT_STATES,
        help="Ruta a states_for_acceptnet.pt",
    )
    return parser.parse_args()


def normalize_regex(regex: str) -> str:
    """Normaliza la regex eliminando espacios."""
    return re.sub(r"\s+", "", regex.strip())


def find_dfa_id(regex: str, df: pd.DataFrame) -> int:
    """Encuentra el dfa_id de una regex."""
    regex_norm = normalize_regex(regex)
    df["Regex_normalized"] = df["Regex"].apply(normalize_regex)
    
    matches = df[df["Regex_normalized"] == regex_norm]
    if not matches.empty:
        return int(matches.iloc[0]["dfa_id"])
    
    matches = df[df["Regex"] == regex]
    if not matches.empty:
        return int(matches.iloc[0]["dfa_id"])
    
    matches = df[df["Regex"].str.contains(regex_norm, case=False, na=False, regex=False)]
    if len(matches) == 1:
        return int(matches.iloc[0]["dfa_id"])
    elif len(matches) > 1:
        print(f"⚠️  Múltiples coincidencias para '{regex}':")
        for idx, row in matches.head(5).iterrows():
            print(f"   [{row['dfa_id']}] {row['Regex']}")
        raise ValueError("Especifica --dfa-id para desambiguar")
    
    raise ValueError(f"No se encontró la regex '{regex}' en el dataset")


def parse_real_transitions(transitions_str: str) -> dict:
    """Parsea las transiciones del DFA real. {(estado, símbolo): estado_destino}"""
    transitions = {}
    parts = transitions_str.split("|")
    for part in parts:
        part = part.strip()
        match = re.match(r"S(\d+)\s*--([A-L])-->\s*S(\d+)", part)
        if match:
            from_state = int(match.group(1))
            symbol = match.group(2)
            to_state = int(match.group(3))
            transitions[(from_state, symbol)] = to_state
    return transitions


def simulate_real_dfa(transitions: dict, accepting_states: set, string: str) -> bool:
    """Simula el DFA real. Retorna True si acepta la cadena."""
    current_state = 0
    for char in string:
        key = (current_state, char)
        if key not in transitions:
            return False
        current_state = transitions[key]
    return current_state in accepting_states


def simulate_generated_dfa(dfa_data: dict, string: str) -> bool:
    """Simula el DFA generado. Retorna True si acepta la cadena."""
    m_use = dfa_data["m_use"]
    m_accept = dfa_data["m_accept"]
    delta = dfa_data["delta"]
    
    if isinstance(m_use, torch.Tensor):
        m_use = m_use.numpy()
    if isinstance(m_accept, torch.Tensor):
        m_accept = m_accept.numpy()
    if isinstance(delta, torch.Tensor):
        delta = delta.numpy()
    
    used_indices = np.where(m_use)[0]
    if len(used_indices) == 0:
        return False
    
    current_state = int(used_indices[0])
    
    for char in string:
        if char not in SYMBOL_TO_IDX:
            return False
        
        symbol_idx = SYMBOL_TO_IDX[char]
        if current_state >= len(delta) or symbol_idx >= delta.shape[1]:
            return False
        
        next_state = int(delta[current_state, symbol_idx])
        if next_state < 0 or not m_use[next_state]:
            return False
        
        current_state = next_state
    
    return bool(m_accept[current_state]) if current_state < len(m_accept) else False


def get_test_strings_from_dataset(row: pd.Series) -> list[str]:
    """Extrae cadenas de ejemplo del dataset."""
    test_strings = []
    try:
        clase_dict = json.loads(row["Clase"])
        accepted = [s for s, v in clase_dict.items() if v][:5]
        rejected = [s for s, v in clase_dict.items() if not v][:5]
        test_strings = accepted + rejected
    except:
        pass
    
    return test_strings if test_strings else ["C", "G", "I", "L", "CG", "IL", "LCIG", "ABC"]


def main():
    args = parse_args()
    
    # Validar argumentos
    if args.dfa_id is None and not args.regex:
        raise ValueError("Debe especificar regex o --dfa-id")
    
    if args.dfa_id is None and (not args.strings or len(args.strings) != 5):
        raise ValueError("Debe proporcionar exactamente 5 cadenas como argumentos")
    
    # Cargar dataset
    df = pd.read_csv(args.dataset_path)
    df = df.reset_index().rename(columns={"index": "dfa_id"})
    
    # Resolver dfa_id
    if args.dfa_id is not None:
        dfa_id = args.dfa_id
        if dfa_id < 0 or dfa_id >= len(df):
            raise ValueError(f"dfa_id fuera de rango (0-{len(df)-1}): {dfa_id}")
        row = df.iloc[dfa_id]
        regex = row["Regex"]
        # Si se usa --dfa-id, usar cadenas del dataset (o las proporcionadas si hay)
        if args.strings and len(args.strings) == 5:
            test_strings = args.strings
        else:
            test_strings = get_test_strings_from_dataset(row)
    else:
        dfa_id = find_dfa_id(args.regex, df)
        row = df.iloc[dfa_id]
        regex = row["Regex"]
        test_strings = args.strings  # Usar las 5 cadenas pasadas como argumentos
    
    print("=" * 70)
    print("🔄 COMPARACIÓN FUNCIONAL: DFA REAL vs MODELO GENERADO")
    print("=" * 70)
    print(f"DFA ID: {dfa_id}")
    print(f"Regex: {regex}")
    print(f"Alfabeto: {row['Alfabeto']}")
    print()
    
    # Parsear DFA real
    transitions_real = parse_real_transitions(row["Transiciones"])
    accepting_states_real = {int(s.replace("S", "")) for s in row["Estados de aceptación"].split()}
    
    # Cargar DFA generado
    states_dict = torch.load(args.states_path, map_location="cpu", weights_only=False)
    if dfa_id not in states_dict:
        dfa_key = str(dfa_id)
        if dfa_key not in states_dict:
            raise KeyError(f"DFA {dfa_id} no encontrado en {args.states_path}")
        dfa_data = states_dict[dfa_key]
    else:
        dfa_data = states_dict[dfa_id]
    
    # Probar cadenas
    print("=" * 70)
    print("🧪 PROBANDO CADENAS")
    print("=" * 70)
    print(f"{'Cadena':<15} {'DFA Real':<12} {'Modelo':<12} {'¿Coincide?':<12}")
    print("-" * 70)
    
    matches = 0
    total = len(test_strings)
    
    for test_str in test_strings:
        # Simular DFA real
        real_accept = simulate_real_dfa(transitions_real, accepting_states_real, test_str)
        real_result = "✓ ACEPTA" if real_accept else "✗ RECHAZA"
        
        # Simular DFA generado
        gen_accept = simulate_generated_dfa(dfa_data, test_str)
        gen_result = "✓ ACEPTA" if gen_accept else "✗ RECHAZA"
        
        # Comparar
        match = "✓ SÍ" if real_accept == gen_accept else "✗ NO"
        if real_accept == gen_accept:
            matches += 1
        
        print(f"{test_str:<15} {real_result:<12} {gen_result:<12} {match:<12}")
    
    print("-" * 70)
    accuracy = (matches / total) * 100 if total > 0 else 0
    print(f"Precisión: {matches}/{total} ({accuracy:.1f}%)")
    print()
    
    print("=" * 70)
    print("📊 RESUMEN")
    print("=" * 70)
    print(f"✓ Coincidencias: {matches}/{total}")
    print(f"✗ Desacuerdos: {total - matches}/{total}")
    print()
    
    if accuracy >= 80:
        print("✓ El modelo funciona bien (≥80% precisión)")
    elif accuracy >= 50:
        print("⚠️  El modelo tiene algunos errores (50-80% precisión)")
    else:
        print("❌ El modelo tiene muchos errores (<50% precisión)")
    print()


if __name__ == "__main__":
    main()

