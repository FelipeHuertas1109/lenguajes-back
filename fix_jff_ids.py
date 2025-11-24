#!/usr/bin/env python3
"""
Script para corregir IDs de estado en archivos JFF
Convierte IDs tipo "S0", "S1" a IDs numéricos 0, 1, etc.
"""
import sys
import re
import xml.etree.ElementTree as ET
from pathlib import Path

def fix_jff_file(input_path, output_path=None):
    """
    Corrige los IDs de estado en un archivo JFF.
    
    Args:
        input_path: Ruta al archivo JFF de entrada
        output_path: Ruta al archivo de salida (opcional, sobrescribe si no se especifica)
    """
    if output_path is None:
        output_path = input_path
    
    print(f"Procesando: {input_path}")
    
    try:
        # Leer el archivo
        tree = ET.parse(input_path)
        root = tree.getroot()
        
        # Buscar el automaton
        automaton = root.find('automaton')
        if automaton is None:
            print("  ERROR: No se encontró el elemento 'automaton'")
            return False
        
        # Obtener todos los estados
        states = automaton.findall('state')
        
        # Crear un mapeo de IDs antiguos a nuevos
        old_to_new = {}
        
        print(f"  Estados encontrados: {len(states)}")
        
        # Asignar nuevos IDs numéricos
        for i, state in enumerate(states):
            old_id = state.get('id')
            new_id = str(i)
            old_to_new[old_id] = new_id
            
            # Actualizar el ID del estado
            state.set('id', new_id)
            print(f"    {old_id} -> {new_id}")
        
        # Actualizar las transiciones
        transitions = automaton.findall('transition')
        print(f"  Transiciones encontradas: {len(transitions)}")
        
        for trans in transitions:
            from_elem = trans.find('from')
            to_elem = trans.find('to')
            
            if from_elem is not None and from_elem.text in old_to_new:
                old_from = from_elem.text
                from_elem.text = old_to_new[old_from]
                print(f"    Transición FROM: {old_from} -> {old_to_new[old_from]}")
            
            if to_elem is not None and to_elem.text in old_to_new:
                old_to = to_elem.text
                to_elem.text = old_to_new[old_to]
                print(f"    Transición TO: {old_to} -> {old_to_new[old_to]}")
        
        # Guardar el archivo corregido
        tree.write(output_path, encoding='UTF-8', xml_declaration=True)
        
        print(f"  [OK] Archivo corregido guardado en: {output_path}")
        return True
        
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    if len(sys.argv) < 2:
        print("Uso:")
        print("  python fix_jff_ids.py <archivo.jff>")
        print("  python fix_jff_ids.py <archivo.jff> <salida.jff>")
        print()
        print("Ejemplos:")
        print("  python fix_jff_ids.py dfa_A_B.jff                    # Sobrescribe el archivo")
        print("  python fix_jff_ids.py dfa_A_B.jff dfa_A_B_fixed.jff  # Crea nuevo archivo")
        sys.exit(1)
    
    input_path = Path(sys.argv[1])
    
    if not input_path.exists():
        print(f"ERROR: Archivo no encontrado: {input_path}")
        sys.exit(1)
    
    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else input_path
    
    print("="*60)
    print("CORRECTOR DE IDs EN ARCHIVOS JFF")
    print("="*60)
    print()
    
    if fix_jff_file(input_path, output_path):
        print()
        print("="*60)
        print("[OK] PROCESO COMPLETADO CON EXITO")
        print("="*60)
        sys.exit(0)
    else:
        print()
        print("="*60)
        print("[ERROR] ERROR EN EL PROCESO")
        print("="*60)
        sys.exit(1)

if __name__ == "__main__":
    main()

