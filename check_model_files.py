"""
Script de ayuda para verificar la ubicación de los archivos del modelo AlphabetNet.
"""

from pathlib import Path
import sys

def check_model_files():
    """Verifica dónde deberían estar los archivos del modelo."""
    base_dir = Path(__file__).resolve().parent
    
    print("=" * 70)
    print("VERIFICACIÓN DE ARCHIVOS DEL MODELO ALPHABETNET")
    print("=" * 70)
    print()
    
    # Ubicaciones posibles
    possible_paths = [
        base_dir / 'models' / 'alphabetnet' / 'standalone_inference' / 'src',  # Nueva ubicación
        base_dir / 'models' / 'src',
        base_dir / 'models' / 'alphabetnet',
        base_dir / 'src',
        base_dir.parent / 'src',
        base_dir.parent / 'models' / 'src',
        base_dir,
    ]
    
    print("Buscando archivos model.py y train.py en las siguientes ubicaciones:")
    print()
    
    found_any = False
    for path in possible_paths:
        model_py = path / 'model.py'
        train_py = path / 'train.py'
        
        model_exists = model_py.exists()
        train_exists = train_py.exists()
        
        status = "✓" if (model_exists and train_exists) else "✗"
        print(f"{status} {path}")
        
        if model_exists:
            print(f"    ✓ model.py encontrado")
        else:
            print(f"    ✗ model.py NO encontrado")
        
        if train_exists:
            print(f"    ✓ train.py encontrado")
        else:
            print(f"    ✗ train.py NO encontrado")
        
        if model_exists and train_exists:
            found_any = True
            print(f"\n✓✓✓ ARCHIVOS ENCONTRADOS EN: {path} ✓✓✓")
            break
        
        print()
    
    if not found_any:
        print()
        print("=" * 70)
        print("❌ NO SE ENCONTRARON LOS ARCHIVOS")
        print("=" * 70)
        print()
        print("INSTRUCCIONES:")
        print()
        print("1. Los archivos model.py y train.py deben estar en una de estas ubicaciones:")
        print("   - models/src/")
        print("   - models/alphabetnet/")
        print("   - src/ (en la raíz del proyecto)")
        print()
        print("2. Si tienes los archivos en otro lugar, cópialos a una de las ubicaciones arriba.")
        print()
        print("3. Los archivos deben contener:")
        print("   - model.py: Clase AlphabetNet")
        print("   - train.py: ALPHABET, MAX_PREFIX_LEN, función regex_to_indices")
        print()
        print("4. Según test_model.py, los archivos deberían estar en: models/src/")
        print("   Puedes crear ese directorio y copiar los archivos allí.")
        print()
    
    # Verificar checkpoint
    print()
    print("=" * 70)
    print("VERIFICACIÓN DEL CHECKPOINT")
    print("=" * 70)
    print()
    
    checkpoint_path = base_dir / 'models' / 'alphabetnet' / 'alphabetnet.pt'
    if checkpoint_path.exists():
        print(f"✓ Checkpoint encontrado: {checkpoint_path}")
        size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
        print(f"  Tamaño: {size_mb:.2f} MB")
    else:
        print(f"✗ Checkpoint NO encontrado: {checkpoint_path}")
    
    # Verificar thresholds
    print()
    print("=" * 70)
    print("VERIFICACIÓN DE THRESHOLDS")
    print("=" * 70)
    print()
    
    thresholds_path = base_dir / 'models' / 'alphabetnet' / 'thresholds.json'
    if thresholds_path.exists():
        print(f"✓ thresholds.json encontrado: {thresholds_path}")
    else:
        print(f"⚠ thresholds.json NO encontrado (se usarán valores por defecto)")

if __name__ == '__main__':
    check_model_files()

