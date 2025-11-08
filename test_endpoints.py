#!/usr/bin/env python3
"""
Script de prueba para los endpoints de la API de Regex to DFA
"""
import requests
import json
import sys
from datetime import datetime

BASE_URL = "http://127.0.0.1:8000"

def print_test_header(test_name):
    print("\n" + "=" * 80)
    print(f"TEST: {test_name}")
    print("=" * 80)

def print_response(response):
    print(f"Status Code: {response.status_code}")
    try:
        data = response.json()
        print("Response JSON:")
        print(json.dumps(data, indent=2, ensure_ascii=False))
    except:
        print("Response Text:")
        print(response.text)

def test_get_basic():
    """Test 1: GET básico con regex simple"""
    print_test_header("GET básico - regex=a*b")
    response = requests.get(f"{BASE_URL}/api/regex-to-dfa/", params={"regex": "a*b"})
    print_response(response)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] == True
    assert data["regex"] == "a*b"
    assert "dfa" in data
    print("[OK] PASSED")

def test_get_with_test():
    """Test 2: GET con regex y cadena de prueba"""
    print_test_header("GET con test - regex=a*b&test=aaab")
    response = requests.get(
        f"{BASE_URL}/api/regex-to-dfa/",
        params={"regex": "a*b", "test": "aaab"}
    )
    print_response(response)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] == True
    assert data["test_result"] is not None
    assert data["test_result"]["string"] == "aaab"
    assert data["test_result"]["accepted"] == True
    print("[OK] PASSED")

def test_post_basic():
    """Test 3: POST básico"""
    print_test_header("POST básico - regex=(a|b)*")
    response = requests.post(
        f"{BASE_URL}/api/regex-to-dfa/",
        json={"regex": "(a|b)*"}
    )
    print_response(response)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] == True
    assert data["regex"] == "(a|b)*"
    print("[OK] PASSED")

def test_post_with_test():
    """Test 4: POST con regex y cadena de prueba"""
    print_test_header("POST con test - regex=a?b&test=b")
    response = requests.post(
        f"{BASE_URL}/api/regex-to-dfa/",
        json={"regex": "a?b", "test": "b"}
    )
    print_response(response)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] == True
    assert data["test_result"]["accepted"] == True
    print("[OK] PASSED")

def test_post_complex():
    """Test 5: POST con regex compleja"""
    print_test_header("POST complejo - regex=a+")
    response = requests.post(
        f"{BASE_URL}/api/regex-to-dfa/",
        json={"regex": "a+"}
    )
    print_response(response)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] == True
    assert len(data["dfa"]["states"]) > 0
    print("[OK] PASSED")

def test_error_missing_regex():
    """Test 6: Error - falta parámetro regex"""
    print_test_header("Error - Falta parámetro regex")
    response = requests.get(f"{BASE_URL}/api/regex-to-dfa/")
    print_response(response)
    assert response.status_code == 400
    data = response.json()
    assert data["success"] == False
    assert "error" in data
    print("[OK] PASSED")

def test_error_invalid_regex():
    """Test 7: Error - regex inválida"""
    print_test_header("Error - Regex inválida (paréntesis desbalanceados)")
    response = requests.get(
        f"{BASE_URL}/api/regex-to-dfa/",
        params={"regex": "((((("}
    )
    print_response(response)
    assert response.status_code == 400
    data = response.json()
    assert data["success"] == False
    assert "error" in data
    print("[OK] PASSED")

def test_error_invalid_json():
    """Test 8: Error - JSON inválido en POST"""
    print_test_header("Error - JSON inválido")
    response = requests.post(
        f"{BASE_URL}/api/regex-to-dfa/",
        data="invalid json",
        headers={"Content-Type": "application/json"}
    )
    print_response(response)
    assert response.status_code == 400
    data = response.json()
    assert data["success"] == False
    print("[OK] PASSED")

def test_index_endpoint():
    """Test 9: Endpoint index"""
    print_test_header("Endpoint index (/)")
    response = requests.get(f"{BASE_URL}/")
    print(f"Status Code: {response.status_code}")
    assert response.status_code == 200
    assert "Hello from Vercel" in response.text
    print("[OK] PASSED")

def test_cors_headers():
    """Test 10: Verificar headers CORS"""
    print_test_header("Verificar headers CORS")
    response = requests.options(
        f"{BASE_URL}/api/regex-to-dfa/",
        headers={"Origin": "https://example.com"}
    )
    print(f"Status Code: {response.status_code}")
    print("CORS Headers:")
    for key, value in response.headers.items():
        if "access-control" in key.lower():
            print(f"  {key}: {value}")
    # Verificar que CORS está habilitado
    assert "Access-Control-Allow-Origin" in response.headers or response.headers.get("Access-Control-Allow-Origin") == "*"
    print("[OK] PASSED")

def main():
    print("=" * 80)
    print("PRUEBAS DE ENDPOINTS - API Regex to DFA")
    print(f"Base URL: {BASE_URL}")
    print(f"Inicio: {datetime.now()}")
    print("=" * 80)
    
    tests = [
        test_index_endpoint,
        test_get_basic,
        test_get_with_test,
        test_post_basic,
        test_post_with_test,
        test_post_complex,
        test_error_missing_regex,
        test_error_invalid_regex,
        test_error_invalid_json,
        test_cors_headers,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"[FAIL] FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"[ERROR] ERROR: {e}")
            failed += 1
    
    print("\n" + "=" * 80)
    print("RESUMEN DE PRUEBAS")
    print("=" * 80)
    print(f"Total: {len(tests)}")
    print(f"[OK] Pasadas: {passed}")
    print(f"[FAIL] Fallidas: {failed}")
    print(f"Final: {datetime.now()}")
    print("=" * 80)
    
    if failed > 0:
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == "__main__":
    main()

