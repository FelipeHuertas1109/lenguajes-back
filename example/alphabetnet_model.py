"""
Módulo auxiliar para cargar y usar el modelo AlphabetNet en Django.
Adaptado desde test_model.py para uso en endpoints.
"""

import json
import sys
import re
from pathlib import Path
from typing import Dict, Optional
import os

import torch
import numpy as np

# Variable global para cachear el modelo cargado
_model_cache = None
_model_device = None
_thresholds_cache = None


def _find_model_files():
    """
    Busca los archivos necesarios del modelo (model.py, train.py, hparams.json).
    Intenta diferentes ubicaciones posibles.
    """
    base_dir = Path(__file__).resolve().parent.parent
    possible_paths = [
        base_dir / 'models' / 'alphabetnet' / 'standalone_inference' / 'src',  # Nueva ubicación
        base_dir / 'models' / 'src',  # Según test_model.py: root / 'src' donde root = models/
        base_dir / 'models' / 'alphabetnet',
        base_dir / 'src',
        base_dir.parent / 'src',
        base_dir.parent / 'models' / 'src',
        base_dir,
    ]
    
    for path in possible_paths:
        model_py = path / 'model.py'
        train_py = path / 'train.py'
        if model_py.exists() and train_py.exists():
            print(f"[ALPHABETNET] Archivos del modelo encontrados en: {path}")
            return path
    
    # Si no se encuentran, mostrar todas las ubicaciones buscadas
    print(f"[ALPHABETNET] ❌ No se encontraron model.py y train.py en ninguna de estas ubicaciones:")
    for path in possible_paths:
        print(f"[ALPHABETNET]    - {path}")
    
    return None


def _import_model_classes():
    """
    Importa las clases y funciones necesarias del modelo.
    """
    model_path = _find_model_files()
    if model_path is None:
        error_msg = (
            "No se encontraron los archivos model.py y train.py.\n"
            "Estos archivos son necesarios para cargar el modelo AlphabetNet.\n\n"
            "Opciones:\n"
            "1. Copia los archivos model.py y train.py a una de estas ubicaciones:\n"
            "   - models/src/\n"
            "   - models/alphabetnet/\n"
            "   - src/ (en la raíz del proyecto)\n\n"
            "2. O crea un enlace simbólico desde donde están los archivos originales.\n\n"
            "Los archivos deben contener:\n"
            "  - model.py: Clase AlphabetNet\n"
            "  - train.py: ALPHABET, MAX_PREFIX_LEN, función regex_to_indices"
        )
        raise ImportError(error_msg)
    
    # Agregar al path si no está
    if str(model_path) not in sys.path:
        sys.path.insert(0, str(model_path))
    
    try:
        from model import AlphabetNet
        from train import ALPHABET, MAX_PREFIX_LEN, regex_to_indices
        print(f"[ALPHABETNET] ✓ Clases importadas correctamente desde {model_path}")
        return AlphabetNet, ALPHABET, MAX_PREFIX_LEN, regex_to_indices
    except ImportError as e:
        raise ImportError(
            f"No se pudieron importar las clases del modelo desde {model_path}.\n"
            f"Error: {e}\n\n"
            f"Asegúrate de que los archivos model.py y train.py estén en {model_path} "
            f"y contengan las clases/funciones necesarias."
        )


def load_model(checkpoint_path: Path, hparams_path: Optional[Path], device: torch.device):
    """
    Carga el modelo desde checkpoint.
    
    Args:
        checkpoint_path: Path al archivo .pt del modelo
        hparams_path: Path al archivo JSON con hiperparámetros (opcional)
        device: Dispositivo torch (cpu/cuda)
    
    Returns:
        Tupla (model, metrics)
    """
    # Importar clases
    AlphabetNet, _, _, _ = _import_model_classes()
    
    # Cargar checkpoint primero para verificar si tiene hiperparámetros
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint no encontrado: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Intentar cargar hiperparámetros
    hparams = None
    
    # Opción 1: Desde archivo hparams.json
    if hparams_path and hparams_path.exists():
        with open(hparams_path, 'r') as f:
            hparams = json.load(f)
        print(f"[ALPHABETNET] Hiperparámetros cargados desde: {hparams_path}")
    # Opción 2: Desde el checkpoint (si están guardados)
    elif 'hparams' in checkpoint:
        hparams = checkpoint['hparams']
        print(f"[ALPHABETNET] Hiperparámetros cargados desde el checkpoint")
    elif 'model' in checkpoint and isinstance(checkpoint['model'], dict):
        # Algunos checkpoints guardan los hparams en checkpoint['model']
        if 'hparams' in checkpoint['model']:
            hparams = checkpoint['model']['hparams']
            print(f"[ALPHABETNET] Hiperparámetros cargados desde checkpoint['model']")
    
    # Si no se encontraron hiperparámetros, intentar inferirlos del state_dict
    if hparams is None:
        print(f"[ALPHABETNET] ⚠️  No se encontraron hiperparámetros, intentando inferirlos del modelo...")
        state_dict = checkpoint.get('model_state_dict', {})
        
        # Intentar inferir algunos parámetros del state_dict
        try:
            # Intentar inferir emb_dim desde la capa de embedding
            emb_dim = 128  # default
            for key in state_dict.keys():
                if 'embedding' in key.lower() or 'emb' in key.lower():
                    if 'weight' in key:
                        shape = state_dict[key].shape
                        if len(shape) == 2:
                            emb_dim = shape[1]
                            break
            
            # Intentar inferir hidden_dim desde las capas RNN
            hidden_dim = 256  # default
            for key in state_dict.keys():
                if 'rnn' in key.lower() or 'lstm' in key.lower() or 'gru' in key.lower():
                    if 'weight_ih' in key or 'weight_hh' in key:
                        shape = state_dict[key].shape
                        if len(shape) >= 2:
                            # Para LSTM/GRU, weight_ih tiene shape [4*hidden_dim, input_dim] o [3*hidden_dim, input_dim]
                            # Para RNN, weight_ih tiene shape [hidden_dim, input_dim]
                            if 'lstm' in key.lower():
                                hidden_dim = shape[0] // 4
                            elif 'gru' in key.lower():
                                hidden_dim = shape[0] // 3
                            else:
                                hidden_dim = shape[0]
                            break
            
            # Intentar inferir num_layers contando las capas
            num_layers = 2  # default
            layer_keys = [k for k in state_dict.keys() if any(x in k.lower() for x in ['rnn', 'lstm', 'gru'])]
            if layer_keys:
                # Contar capas únicas (asumiendo formato layer0, layer1, etc.)
                layer_nums = set()
                for key in layer_keys:
                    parts = key.split('.')
                    for part in parts:
                        if part.startswith('layer') or part.startswith('rnn'):
                            try:
                                # Intentar extraer número de capa
                                match = re.search(r'(\d+)', part)
                                if match:
                                    layer_nums.add(int(match.group(1)))
                            except:
                                pass
                if layer_nums:
                    num_layers = max(layer_nums) + 1
            
            # Intentar inferir vocab_size desde embedding
            vocab_size = 100  # default
            for key in state_dict.keys():
                if 'embedding' in key.lower() or 'emb' in key.lower():
                    if 'weight' in key:
                        shape = state_dict[key].shape
                        if len(shape) == 2:
                            vocab_size = shape[0]
                            break
            
            # Intentar detectar rnn_type
            rnn_type = 'LSTM'  # default
            for key in state_dict.keys():
                if 'lstm' in key.lower():
                    rnn_type = 'LSTM'
                    break
                elif 'gru' in key.lower():
                    rnn_type = 'GRU'
                    break
                elif 'rnn' in key.lower():
                    rnn_type = 'RNN'
                    break
            
            # Valores por defecto para otros parámetros
            hparams = {
                'model': {
                    'vocab_size': vocab_size,
                    'alphabet_size': 26,  # A-Z (común para este tipo de modelos)
                    'emb_dim': emb_dim,
                    'hidden_dim': hidden_dim,
                    'rnn_type': rnn_type,
                    'num_layers': num_layers,
                    'dropout': 0.2,
                    'padding_idx': 0,
                    'use_automata_conditioning': False,
                    'automata_emb_dim': 64
                }
            }
            print(f"[ALPHABETNET] ⚠️  Hiperparámetros inferidos del modelo:")
            print(f"[ALPHABETNET]    vocab_size={vocab_size}, emb_dim={emb_dim}, hidden_dim={hidden_dim}")
            print(f"[ALPHABETNET]    rnn_type={rnn_type}, num_layers={num_layers}")
            print(f"[ALPHABETNET] ⚠️  Se recomienda crear un archivo hparams.json con los valores correctos.")
        except Exception as e:
            print(f"[ALPHABETNET] ❌ Error al inferir hiperparámetros: {e}")
            raise ValueError(
                f"No se pudieron inferir los hiperparámetros del modelo. "
                f"Por favor, crea un archivo hparams.json en models/alphabetnet/ con los valores correctos. "
                f"Error: {e}"
            )
    
    # Extraer los hiperparámetros del modelo
    model_hparams = hparams.get('model', hparams) if isinstance(hparams, dict) else hparams
    
    # Crear modelo
    model = AlphabetNet(
        vocab_size=model_hparams['vocab_size'],
        alphabet_size=model_hparams['alphabet_size'],
        emb_dim=model_hparams['emb_dim'],
        hidden_dim=model_hparams['hidden_dim'],
        rnn_type=model_hparams['rnn_type'],
        num_layers=model_hparams['num_layers'],
        dropout=model_hparams['dropout'],
        padding_idx=model_hparams['padding_idx'],
        use_automata_conditioning=model_hparams['use_automata_conditioning'],
        num_automata=model_hparams.get('num_automata'),
        automata_emb_dim=model_hparams['automata_emb_dim']
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Obtener métricas del checkpoint
    metrics = {
        'epoch': checkpoint.get('epoch', 'N/A'),
        'f1_macro': checkpoint.get('f1_macro', 'N/A'),
        'f1_min': checkpoint.get('f1_min', 'N/A'),
        'ece': checkpoint.get('ece', 'N/A')
    }
    
    return model, metrics


def load_thresholds(thresholds_path: Optional[Path]) -> Optional[Dict[str, float]]:
    """
    Carga thresholds desde archivo JSON.
    
    Args:
        thresholds_path: Path al archivo JSON con thresholds (opcional)
    
    Returns:
        Diccionario con thresholds por símbolo o None
    """
    if thresholds_path is None or not thresholds_path.exists():
        return None
    
    with open(thresholds_path, 'r') as f:
        data = json.load(f)
        return data.get('per_symbol', {})


def get_or_load_model():
    """
    Obtiene el modelo desde cache o lo carga si no está en cache.
    Usa lazy loading para cargar solo cuando se necesite.
    
    Returns:
        Tupla (model, device, thresholds)
    """
    global _model_cache, _model_device, _thresholds_cache
    
    if _model_cache is not None:
        return _model_cache, _model_device, _thresholds_cache
    
    # Configurar paths
    base_dir = Path(__file__).resolve().parent.parent
    
    # Buscar checkpoint en diferentes ubicaciones
    checkpoint_paths = [
        base_dir / 'models' / 'alphabetnet' / 'standalone_inference' / 'novTest' / 'alphabetnet.pt',  # Nueva ubicación
        base_dir / 'models' / 'alphabetnet' / 'alphabetnet.pt',
        base_dir / 'models' / 'alphabetnet' / 'standalone_inference' / 'alphabetnet.pt',
    ]
    checkpoint_path = None
    for path in checkpoint_paths:
        if path.exists():
            checkpoint_path = path
            break
    
    if checkpoint_path is None:
        # Usar la ruta por defecto si no se encuentra
        checkpoint_path = base_dir / 'models' / 'alphabetnet' / 'alphabetnet.pt'
        print(f"[ALPHABETNET] ⚠️  Checkpoint no encontrado en ubicaciones conocidas, usando: {checkpoint_path}")
    
    # Buscar hparams.json en diferentes ubicaciones (opcional)
    hparams_paths = [
        base_dir / 'models' / 'alphabetnet' / 'standalone_inference' / 'hparams.json',  # Nueva ubicación
        base_dir / 'models' / 'alphabetnet' / 'hparams.json',
        base_dir / 'hparams.json',
        base_dir.parent / 'hparams.json',
    ]
    hparams_path = None
    for path in hparams_paths:
        if path.exists():
            hparams_path = path
            break
    
    # Si no se encuentra hparams.json, intentaremos cargar desde el checkpoint
    if hparams_path is None:
        print(f"[ALPHABETNET] ⚠️  No se encontró hparams.json, intentando cargar desde el checkpoint...")
    
    # Buscar thresholds.json (opcional)
    thresholds_paths = [
        base_dir / 'models' / 'alphabetnet' / 'standalone_inference' / 'novTest' / 'thresholds.json',  # Nueva ubicación
        base_dir / 'models' / 'alphabetnet' / 'thresholds.json',
        base_dir / 'thresholds.json',
    ]
    thresholds_path = None
    for path in thresholds_paths:
        if path.exists():
            thresholds_path = path
            break
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Cargar modelo (hparams_path puede ser None si no existe el archivo)
    model, metrics = load_model(checkpoint_path, hparams_path, device)
    print(f"[ALPHABETNET] Modelo cargado - Época: {metrics['epoch']}, F1: {metrics['f1_macro']}")
    
    # Cargar thresholds
    if thresholds_path:
        thresholds = load_thresholds(thresholds_path)
        print(f"[ALPHABETNET] Thresholds cargados desde: {thresholds_path}")
    else:
        # Usar thresholds por defecto (0.5 para todos)
        _, ALPHABET, _, _ = _import_model_classes()
        thresholds = {sym: 0.5 for sym in ALPHABET}
        print(f"[ALPHABETNET] Usando thresholds por defecto (0.5)")
    
    # Cachear
    _model_cache = model
    _model_device = device
    _thresholds_cache = thresholds
    
    return model, device, thresholds


def predict_alphabet(regex: str, threshold_per_symbol: Optional[Dict[str, float]] = None) -> Dict:
    """
    Predice alfabeto desde regex usando el modelo AlphabetNet.
    
    Args:
        regex: Expresión regular a procesar
        threshold_per_symbol: Thresholds por símbolo (opcional, usa cache si no se proporciona)
    
    Returns:
        Diccionario con:
        - 'p_sigma': probabilidades por símbolo
        - 'sigma_hat': alfabeto predicho (lista de símbolos)
    """
    # Obtener modelo (carga si es necesario)
    model, device, default_thresholds = get_or_load_model()
    
    # Importar funciones necesarias
    _, ALPHABET, MAX_PREFIX_LEN, regex_to_indices = _import_model_classes()
    
    # Usar thresholds proporcionados o los del cache
    if threshold_per_symbol is None:
        threshold_per_symbol = default_thresholds
    
    # Convertir regex a índices
    regex_indices, length = regex_to_indices(regex, MAX_PREFIX_LEN)
    regex_indices = regex_indices.unsqueeze(0).to(device)
    lengths = torch.tensor([length], dtype=torch.long).to(device)
    
    # Forward pass
    with torch.no_grad():
        logits = model(regex_indices, lengths, return_logits=True)
        probs = torch.sigmoid(logits).cpu().numpy().flatten()
    
    # Construir sigma_hat
    sigma_hat = []
    p_sigma_dict = {}
    
    for i, symbol in enumerate(ALPHABET):
        prob = float(probs[i])
        threshold = threshold_per_symbol.get(symbol, 0.5)
        p_sigma_dict[symbol] = prob
        if prob >= threshold:
            sigma_hat.append(symbol)
    
    return {
        'p_sigma': p_sigma_dict,
        'sigma_hat': sorted(sigma_hat)  # Ordenar alfabeto predicho
    }

