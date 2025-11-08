#!/usr/bin/env python3
"""Script para probar la generación de archivos JFF"""
import requests
import xml.etree.ElementTree as ET

def test_jff_generation(regex):
    print(f"\n{'='*60}")
    print(f"Probando regex: {regex}")
    print('='*60)
    
    try:
        r = requests.get('http://localhost:8000/api/regex-to-dfa/jff/', params={'regex': regex})
        print(f"Status: {r.status_code}")
        
        if r.status_code != 200:
            print(f"Error: {r.text}")
            return False
        
        # Parsear XML
        try:
            root = ET.fromstring(r.text)
            
            # Verificar estructura
            if root.tag != 'structure':
                print("ERROR: El elemento raíz no es 'structure'")
                return False
            
            automaton = root.find('automaton')
            if automaton is None:
                print("ERROR: No se encuentra el elemento 'automaton'")
                return False
            
            # Obtener todos los estados
            states = automaton.findall('state')
            print(f"\nEstados encontrados: {len(states)}")
            
            # Verificar IDs de estados
            state_ids = []
            state_names = []
            for state in states:
                state_id = state.get('id')
                state_name = state.get('name')
                state_ids.append(state_id)
                state_names.append(state_name)
                print(f"  - Estado: id={state_id}, name={state_name}")
            
            # Verificar que los IDs sean únicos
            if len(state_ids) != len(set(state_ids)):
                print(f"\nERROR: IDs duplicados encontrados!")
                print(f"IDs: {state_ids}")
                return False
            
            # Verificar que no haya IDs negativos o inválidos
            for state_id in state_ids:
                try:
                    id_num = int(state_id)
                    if id_num < 0:
                        print(f"\nERROR: ID negativo encontrado: {id_num}")
                        return False
                except ValueError:
                    print(f"\nERROR: ID no numérico: {state_id}")
                    return False
            
            # Obtener todas las transiciones
            transitions = automaton.findall('transition')
            print(f"\nTransiciones encontradas: {len(transitions)}")
            
            # Verificar que todas las transiciones referencien estados válidos
            valid_state_ids = set(state_ids)
            for trans in transitions:
                from_id = trans.find('from').text
                to_id = trans.find('to').text
                symbol = trans.find('read').text
                
                if from_id not in valid_state_ids:
                    print(f"\nERROR: Transición desde estado inválido: {from_id}")
                    return False
                if to_id not in valid_state_ids:
                    print(f"\nERROR: Transición hacia estado inválido: {to_id}")
                    return False
                print(f"  - {from_id} --{symbol}--> {to_id}")
            
            print(f"\n[OK] Archivo JFF valido!")
            print(f"   - {len(states)} estados con IDs únicos")
            print(f"   - {len(transitions)} transiciones válidas")
            return True
            
        except ET.ParseError as e:
            print(f"ERROR: XML malformado: {e}")
            print(f"Contenido: {r.text[:500]}")
            return False
            
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_cases = [
        "a*b",
        "(a|b)*",
        "(a|b)*c",
        "a+",
        "a?b",
    ]
    
    print("="*60)
    print("PRUEBAS DE GENERACIÓN DE ARCHIVOS JFF")
    print("="*60)
    
    passed = 0
    failed = 0
    
    for regex in test_cases:
        if test_jff_generation(regex):
            passed += 1
        else:
            failed += 1
    
    print("\n" + "="*60)
    print("RESUMEN")
    print("="*60)
    print(f"Pasadas: {passed}")
    print(f"Fallidas: {failed}")
    print(f"Total: {len(test_cases)}")

