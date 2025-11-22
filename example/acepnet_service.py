"""
Servicio para usar el modelo AcepNet desde Django
Wrapper para cargar el modelo y hacer predicciones
"""

import sys
import os
from pathlib import Path
import torch
import json

# Añadir la ruta del modelo al path
models_path = Path(__file__).parent.parent / "models" / "acepnet"
sys.path.insert(0, str(models_path))

try:
    from acepten import AFDParser, DualEncoderModel, CHAR_TO_IDX
except ImportError as e:
    print(f"[ACEPNET_SERVICE] ERROR - No se pudo importar el modelo: {e}")
    raise


class AcepNetService:
    """
    Servicio singleton para cargar y usar el modelo AcepNet
    Cachea el modelo y el parser para evitar recargarlos
    """
    
    _instance = None
    _model = None
    _parser = None
    _thresholds = None
    _device = None
    _loaded = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AcepNetService, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._loaded:
            self._load_model()
            self._loaded = True
    
    def _load_model(self):
        """Carga el modelo, parser y umbrales"""
        try:
            # Rutas a los archivos del modelo - usar rutas absolutas
            # __file__ está en example/acepnet_service.py
            # Queremos llegar a models/acepnet/
            service_file = Path(__file__).resolve()  # Ruta absoluta del archivo actual
            base_path = service_file.parent.parent / "models" / "acepnet"
            
            # Convertir a rutas absolutas
            base_path = base_path.resolve()
            model_path = (base_path / "best_model.pt").resolve()
            dataset_path = (base_path / "dataset6000.csv").resolve()
            thresholds_path = (base_path / "thresholds.json").resolve()
            
            print(f"[ACEPNET_SERVICE] Service file: {service_file}")
            print(f"[ACEPNET_SERVICE] Base path: {base_path}")
            print(f"[ACEPNET_SERVICE] Model path: {model_path}")
            print(f"[ACEPNET_SERVICE] Dataset path: {dataset_path}")
            print(f"[ACEPNET_SERVICE] Thresholds path: {thresholds_path}")
            sys.stdout.flush()
            
            # Verificar que los archivos existan
            if not model_path.exists():
                raise FileNotFoundError(
                    f"Modelo no encontrado: {model_path}\n"
                    f"Ruta base buscada: {base_path}\n"
                    f"Archivo del servicio: {service_file}"
                )
            
            if not dataset_path.exists():
                raise FileNotFoundError(
                    f"Dataset no encontrado: {dataset_path}\n"
                    f"Ruta base buscada: {base_path}\n"
                    f"Archivo del servicio: {service_file}\n"
                    f"Verifica que el archivo 'dataset6000.csv' esté en: {base_path}"
                )
            
            print(f"[ACEPNET_SERVICE] ✅ Todos los archivos encontrados")
            sys.stdout.flush()
            
            # Cargar modelo
            self._model = DualEncoderModel()
            self._model.load_state_dict(
                torch.load(str(model_path), map_location='cpu', weights_only=False)
            )
            self._model.eval()
            
            # Cargar parser con ruta absoluta
            print(f"[ACEPNET_SERVICE] Cargando dataset desde: {dataset_path}")
            sys.stdout.flush()
            
            # Verificar nuevamente antes de cargar (por si acaso)
            if not dataset_path.exists():
                raise FileNotFoundError(
                    f"Dataset no encontrado después de verificación: {dataset_path}\n"
                    f"Ruta absoluta buscada: {dataset_path.resolve()}\n"
                    f"Directorio existe: {dataset_path.parent.exists()}"
                )
            
            try:
                # Asegurar que la ruta sea absoluta y exista antes de pasar a AFDParser
                absolute_path = dataset_path.resolve()
                absolute_path_str = str(absolute_path)
                
                # Verificar una vez más que el archivo existe
                if not absolute_path.exists():
                    current_working_dir = Path.cwd()
                    error_rutas_buscadas = [
                        absolute_path_str,
                        str(base_path / "dataset6000.csv"),
                        str(service_file.parent / "dataset6000.csv"),
                    ]
                    
                    raise FileNotFoundError(
                        f"Archivo CSV no encontrado. Rutas buscadas: {', '.join(error_rutas_buscadas)}\n"
                        f"Directorio de trabajo actual: {current_working_dir}\n"
                        f"Verifica que el archivo 'dataset6000.csv' esté en: {base_path}"
                    )
                
                print(f"[ACEPNET_SERVICE] ✅ Archivo CSV verificado: {absolute_path_str}")
                print(f"[ACEPNET_SERVICE] Cargando CSV con pandas...")
                sys.stdout.flush()
                
                # Pasar la ruta absoluta como string a AFDParser
                self._parser = AFDParser(absolute_path_str)
            except FileNotFoundError as e:
                # Si pandas o Path lanza un FileNotFoundError, dar un mensaje más claro
                current_dir = Path.cwd()
                error_msg = (
                    f"Error al cargar el dataset CSV.\n"
                    f"Ruta absoluta buscada: {dataset_path.resolve()}\n"
                    f"Archivo existe: {dataset_path.exists()}\n"
                    f"Directorio de trabajo actual: {current_dir}\n"
                    f"Archivo del servicio: {service_file}\n"
                    f"Error original: {str(e)}\n\n"
                    f"Verifica que el archivo 'dataset6000.csv' esté en:\n"
                    f"  {base_path}"
                )
                print(f"[ACEPNET_SERVICE] ERROR - {error_msg}")
                sys.stdout.flush()
                raise FileNotFoundError(error_msg) from e
            except Exception as e:
                # Cualquier otro error al cargar el CSV
                error_msg = (
                    f"Error inesperado al cargar el dataset CSV:\n"
                    f"Ruta: {dataset_path.resolve()}\n"
                    f"Ruta absoluta: {str(dataset_path.resolve())}\n"
                    f"Error: {str(e)}\n"
                    f"Tipo de error: {type(e).__name__}"
                )
                print(f"[ACEPNET_SERVICE] ERROR - {error_msg}")
                import traceback
                traceback.print_exc()
                sys.stdout.flush()
                raise
            
            # Detectar device
            self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self._model = self._model.to(self._device)
            
            # Cargar umbrales calibrados
            self._thresholds = {'y1': 0.43, 'y2': 0.53}  # Valores por defecto
            if thresholds_path.exists():
                try:
                    with open(thresholds_path, 'r') as f:
                        self._thresholds = json.load(f)
                    print(f"[ACEPNET_SERVICE] Umbrales calibrados: Y1={self._thresholds['y1']}, Y2={self._thresholds['y2']}")
                except Exception as e:
                    print(f"[ACEPNET_SERVICE] ⚠️  Error cargando umbrales, usando por defecto: {e}")
            else:
                print(f"[ACEPNET_SERVICE] ⚠️  Umbrales no encontrados, usando por defecto")
            
            print(f"[ACEPNET_SERVICE] ✅ Modelo cargado correctamente en {self._device}")
            print(f"[ACEPNET_SERVICE] ✅ Dataset cargado: {len(self._parser.df)} AFDs")
            sys.stdout.flush()
            
        except Exception as e:
            print(f"[ACEPNET_SERVICE] ERROR - Error al cargar el modelo: {e}")
            import traceback
            traceback.print_exc()
            sys.stdout.flush()
            raise
    
    def predict(self, dfa_id: int, string: str):
        """
        Hace una predicción para una cadena y un AFD
        
        Args:
            dfa_id: ID del AFD (0-5999)
            string: Cadena a evaluar
        
        Returns:
            dict con predicción, probabilidades y ground truth
        """
        if self._model is None or self._parser is None:
            raise RuntimeError("Modelo no cargado. Llama a _load_model() primero.")
        
        # Validar dfa_id
        if dfa_id < 0 or dfa_id >= len(self._parser.df):
            raise ValueError(f"dfa_id debe estar entre 0 y {len(self._parser.df)-1}")
        
        # Obtener información del AFD
        row = self._parser.df.iloc[dfa_id]
        afd_info = {
            'regex': row['Regex'],
            'alphabet': row['Alfabeto'],
            'states': row['Estados'],
            'accepting': row['Estados de aceptación'],
        }
        
        # Verificar alfabeto del AFD
        alfabeto = set(row['Alfabeto'].split())
        alphabet_mismatch = False
        if string not in ("", "<EPS>"):
            for ch in string:
                if ch not in alfabeto:
                    alphabet_mismatch = True
                    break
        
        if alphabet_mismatch:
            y1_prob = 0.0
            y1_pred = False
            y2_prob = 0.0
            y2_pred = False
        else:
            # Obtener features del AFD
            afd_features = torch.tensor(
                self._parser.get_afd_features(dfa_id),
                dtype=torch.float32
            ).unsqueeze(0).to(self._device)
            
            # Tokenizar cadena
            if string == '<EPS>' or string == '':
                tokens = []
            else:
                tokens = [CHAR_TO_IDX.get(c, 12) for c in string if c in CHAR_TO_IDX]
            
            # Preparar tensors
            if len(tokens) == 0:
                string_tokens = torch.zeros((1, 1), dtype=torch.long).to(self._device)
                string_lengths = torch.tensor([0], dtype=torch.long).to(self._device)
            else:
                string_tokens = torch.tensor([tokens], dtype=torch.long).to(self._device)
                string_lengths = torch.tensor([len(tokens)], dtype=torch.long).to(self._device)
            
            # Predecir con el modelo
            with torch.no_grad():
                y1_prob, y2_prob = self._model(string_tokens, string_lengths, afd_features)
                y1_prob = y1_prob.item()
                y2_prob = y2_prob.item()
            
            y1_pred = y1_prob >= self._thresholds['y1']
            y2_pred = y2_prob >= self._thresholds['y2']
        
        # Simular AFD para comparar (ground truth)
        try:
            ground_truth = self._parser.simulate_afd(dfa_id, string)
        except Exception as e:
            print(f"[ACEPNET_SERVICE] ⚠️  Error al simular AFD: {e}")
            ground_truth = None
        
        # Determinar si la predicción fue correcta
        prediction_correct = None
        if ground_truth is not None:
            prediction_correct = (y1_pred == ground_truth)
        
        return {
            'dfa_id': dfa_id,
            'string': string,
            'afd_info': afd_info,
            'y1': {
                'probability': float(y1_prob),
                'predicted': bool(y1_pred),
                'ground_truth': ground_truth,
                'correct': prediction_correct,
                'alphabet_mismatch': alphabet_mismatch
            },
            'y2': {
                'probability': float(y2_prob),
                'predicted': bool(y2_pred)
            }
        }
    
    def get_afd_info(self, dfa_id: int):
        """Obtiene información de un AFD sin hacer predicción"""
        if self._parser is None:
            raise RuntimeError("Parser no cargado")
        
        if dfa_id < 0 or dfa_id >= len(self._parser.df):
            raise ValueError(f"dfa_id debe estar entre 0 y {len(self._parser.df)-1}")
        
        row = self._parser.df.iloc[dfa_id]
        return {
            'id': dfa_id,
            'regex': row['Regex'],
            'alphabet': row['Alfabeto'],
            'states': row['Estados'],
            'accepting': row['Estados de aceptación'],
        }


def get_acepnet_service():
    """Función helper para obtener la instancia singleton del servicio"""
    return AcepNetService()

