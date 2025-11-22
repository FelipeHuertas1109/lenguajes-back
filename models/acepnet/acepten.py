"""
Modelo Multi-Tarea para Clasificación de Cadenas en Autómatas Finitos Deterministas

Arquitectura:
- Encoder de cadena: BiGRU sobre símbolos
- Encoder de autómata: MLP sobre representación del AFD
- Dos cabezas de salida:
  1. Pertenencia a este AFD específico
  2. Si la cadena es aceptada por múltiples AFDs
"""

import re
import json
import pandas as pd
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Set
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_recall_curve, auc
import matplotlib.pyplot as plt

# Constantes globales
ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L']
CHAR_TO_IDX = {char: idx for idx, char in enumerate(ALPHABET)}
MAX_STATES = 16
NUM_SYMBOLS = len(ALPHABET)
PAD_IDX = len(ALPHABET)  # Índice 12 para padding


class AFDParser:
    """Parser para extraer información estructurada de los AFDs del dataset"""
    
    def __init__(self, csv_path: str):
        self.df = pd.read_csv(csv_path)
        self.afd_cache = {}
        
    def parse_states(self, estados_str: str) -> List[int]:
        """Parsea 'S0 S1 S2 S3' -> [0, 1, 2, 3]"""
        if pd.isna(estados_str):
            return []
        return [int(s.replace('S', '')) for s in estados_str.split()]
    
    def parse_accept_states(self, accept_str: str) -> Set[int]:
        """Parsea estados de aceptación"""
        if pd.isna(accept_str):
            return set()
        return set(int(s.replace('S', '')) for s in accept_str.split())
    
    def parse_transitions(self, trans_str: str, alphabet_str: str) -> Dict[Tuple[int, str], int]:
        """
        Parsea transiciones: 'S0 --A--> S1 | S0 --B--> S2' 
        -> {(0, 'A'): 1, (0, 'B'): 2}
        """
        transitions = {}
        if pd.isna(trans_str):
            return transitions
            
        # Regex para capturar: S0 --A--> S1
        pattern = r'S(\d+)\s*--(\w+)-->\s*S(\d+)'
        matches = re.findall(pattern, trans_str)
        
        for from_state, symbol, to_state in matches:
            transitions[(int(from_state), symbol)] = int(to_state)
            
        return transitions
    
    def get_afd_features(self, dfa_id: int) -> np.ndarray:
        """
        Extrae representación vectorial del AFD:
        - one_hot_T[16, 12, 16]: matriz de transiciones one-hot
        - accept_vec[16]: vector de estados de aceptación
        - valid_states[16]: máscara de estados válidos
        
        Total: 16*12*16 + 16 + 16 = 3104 features
        """
        if dfa_id in self.afd_cache:
            return self.afd_cache[dfa_id]
            
        row = self.df.iloc[dfa_id]
        
        # Parsear información del AFD
        estados = self.parse_states(row['Estados'])
        accept_states = self.parse_accept_states(row['Estados de aceptación'])
        transitions = self.parse_transitions(row['Transiciones'], row['Alfabeto'])
        
        # Crear vectores
        accept_vec = np.zeros(MAX_STATES, dtype=np.float32)
        valid_states = np.zeros(MAX_STATES, dtype=np.float32)
        one_hot_T = np.zeros((MAX_STATES, NUM_SYMBOLS, MAX_STATES), dtype=np.float32)
        
        # Rellenar vectores
        for state in estados:
            if state < MAX_STATES:
                valid_states[state] = 1.0
                
        for state in accept_states:
            if state < MAX_STATES:
                accept_vec[state] = 1.0
        
        # Rellenar matriz de transiciones
        for (from_state, symbol), to_state in transitions.items():
            if symbol in CHAR_TO_IDX and from_state < MAX_STATES and to_state < MAX_STATES:
                symbol_idx = CHAR_TO_IDX[symbol]
                one_hot_T[from_state, symbol_idx, to_state] = 1.0
        
        # Concatenar todo en un vector plano
        features = np.concatenate([
            one_hot_T.flatten(),  # 3072
            accept_vec,           # 16
            valid_states          # 16
        ])
        
        self.afd_cache[dfa_id] = features
        return features
    
    def simulate_afd(self, dfa_id: int, string: str) -> bool:
        """Simula el AFD para verificar si acepta una cadena"""
        row = self.df.iloc[dfa_id]
        
        # Casos especiales
        if string == "<EPS>" or string == "":
            string = ""
            
        estados = self.parse_states(row['Estados'])
        accept_states = self.parse_accept_states(row['Estados de aceptación'])
        transitions = self.parse_transitions(row['Transiciones'], row['Alfabeto'])
        
        # Simular desde S0
        current_state = 0
        
        for char in string:
            if (current_state, char) not in transitions:
                return False  # Transición no definida
            current_state = transitions[(current_state, char)]
        
        return current_state in accept_states


class StringDatasetGenerator:
    """Generador de dataset de pares (dfa_id, string, label)"""
    
    def __init__(self, parser: AFDParser):
        self.parser = parser
        
    def generate_positive_samples(self, dfa_id: int, max_samples: int = 50) -> List[str]:
        """
        Genera cadenas aceptadas por el AFD usando la columna 'Clase'
        """
        row = self.parser.df.iloc[dfa_id]
        clase = row['Clase']
        
        if pd.isna(clase):
            return []
            
        try:
            # Parsear JSON de Clase
            clase_dict = json.loads(clase.replace("'", '"'))
            # Filtrar solo cadenas aceptadas (value == True)
            accepted = [k for k, v in clase_dict.items() if v is True]
            # Limitar número de muestras
            return accepted[:max_samples]
        except:
            return []
    
    def generate_negative_samples(self, dfa_id: int, num_samples: int = 50, 
                                  max_len: int = 5) -> List[str]:
        """
        Genera cadenas aleatorias que probablemente no son aceptadas
        """
        row = self.parser.df.iloc[dfa_id]
        alfabeto_str = row['Alfabeto']
        
        if pd.isna(alfabeto_str):
            return []
            
        # Extraer símbolos del alfabeto del AFD
        simbolos = alfabeto_str.split()
        
        negative_samples = []
        attempts = 0
        max_attempts = num_samples * 10
        
        while len(negative_samples) < num_samples and attempts < max_attempts:
            attempts += 1
            # Generar cadena aleatoria
            length = np.random.randint(1, max_len + 1)
            string = ''.join(np.random.choice(simbolos, size=length))
            
            # Verificar que no sea aceptada
            if not self.parser.simulate_afd(dfa_id, string):
                negative_samples.append(string)
        
        return negative_samples
    
    def generate_full_dataset(self, pos_samples_per_dfa: int = 50,
                             neg_samples_per_dfa: int = 50) -> pd.DataFrame:
        """
        Genera dataset completo con pares (dfa_id, string, label)
        """
        data = []
        num_dfas = len(self.parser.df)
        
        print(f"Generando dataset desde {num_dfas} AFDs...")
        
        for dfa_id in range(num_dfas):
            if dfa_id % 500 == 0:
                print(f"  Procesando AFD {dfa_id}/{num_dfas}...")
            
            # Muestras positivas
            pos_strings = self.generate_positive_samples(dfa_id, pos_samples_per_dfa)
            for string in pos_strings:
                data.append({
                    'dfa_id': dfa_id,
                    'string': string if string != "" else "<EPS>",
                    'label': 1
                })
            
            # Muestras negativas
            neg_strings = self.generate_negative_samples(dfa_id, neg_samples_per_dfa)
            for string in neg_strings:
                data.append({
                    'dfa_id': dfa_id,
                    'string': string,
                    'label': 0
                })
        
        df = pd.DataFrame(data)
        print(f"Dataset generado: {len(df)} ejemplos")
        return df
    
    def compute_shared_label(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula y2: si una cadena es aceptada por múltiples AFDs
        """
        # Agrupar por string y contar cuántos AFDs distintos la aceptan
        string_counts = defaultdict(set)
        
        for _, row in df.iterrows():
            if row['label'] == 1:  # Solo cadenas aceptadas
                string_counts[row['string']].add(row['dfa_id'])
        
        # Crear columna y2
        df['y2'] = df['string'].apply(
            lambda s: 1 if len(string_counts.get(s, set())) >= 2 else 0
        )
        
        num_shared = df['y2'].sum()
        print(f"Cadenas compartidas: {num_shared}/{len(df)} ({num_shared/len(df)*100:.1f}%)")
        
        return df


class AFDStringDataset(Dataset):
    """Dataset de PyTorch para pares (dfa_id, string, y1, y2)"""
    
    def __init__(self, df: pd.DataFrame, parser: AFDParser):
        self.df = df.reset_index(drop=True)
        self.parser = parser
        
    def __len__(self):
        return len(self.df)
    
    def string_to_indices(self, string: str) -> List[int]:
        """Convierte cadena a índices: 'ABC' -> [0, 1, 2]"""
        if string == "<EPS>" or string == "":
            return []
        return [CHAR_TO_IDX[char] for char in string if char in CHAR_TO_IDX]
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Features del AFD
        afd_features = self.parser.get_afd_features(row['dfa_id'])
        
        # Tokenización de la cadena
        string_tokens = self.string_to_indices(row['string'])
        
        # Labels
        y1 = float(row['label'])
        y2 = float(row['y2'])
        
        return {
            'afd_features': torch.tensor(afd_features, dtype=torch.float32),
            'string_tokens': torch.tensor(string_tokens, dtype=torch.long),
            'string_length': len(string_tokens),
            'y1': torch.tensor(y1, dtype=torch.float32),
            'y2': torch.tensor(y2, dtype=torch.float32)
        }


def collate_fn(batch):
    """Collate function para manejar secuencias de longitud variable"""
    afd_features = torch.stack([item['afd_features'] for item in batch])
    y1 = torch.stack([item['y1'] for item in batch])
    y2 = torch.stack([item['y2'] for item in batch])
    
    # Padding de secuencias
    string_lengths = [item['string_length'] for item in batch]
    max_len = max(string_lengths) if string_lengths else 1
    
    padded_strings = []
    for item in batch:
        tokens = item['string_tokens']
        # Pad con PAD_IDX
        padded = F.pad(tokens, (0, max_len - len(tokens)), value=PAD_IDX)
        padded_strings.append(padded)
    
    string_tokens = torch.stack(padded_strings)
    string_lengths = torch.tensor(string_lengths, dtype=torch.long)
    
    return {
        'afd_features': afd_features,
        'string_tokens': string_tokens,
        'string_lengths': string_lengths,
        'y1': y1,
        'y2': y2
    }


class DualEncoderModel(nn.Module):
    """
    Modelo dual-encoder con dos cabezas de salida
    
    Arquitectura:
    - String Encoder: Embedding + BiGRU -> h_str
    - AFD Encoder: MLP -> h_afd
    - Head 1: concat(h_str, h_afd) -> MLP -> y1_hat (pertenencia)
    - Head 2: h_str -> MLP -> y2_hat (cadena compartida)
    """
    
    def __init__(self, 
                 vocab_size: int = NUM_SYMBOLS + 1,  # +1 para PAD
                 embed_dim: int = 32,
                 rnn_hidden_dim: int = 64,
                 afd_input_dim: int = 3104,
                 afd_hidden_dim: int = 128,
                 combined_hidden_dim: int = 128):
        super().__init__()
        
        # String Encoder
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        self.rnn = nn.GRU(embed_dim, rnn_hidden_dim, 
                         num_layers=2, 
                         bidirectional=True, 
                         batch_first=True,
                         dropout=0.2)
        self.rnn_output_dim = rnn_hidden_dim * 2  # Bidireccional
        
        # AFD Encoder
        self.afd_encoder = nn.Sequential(
            nn.Linear(afd_input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, afd_hidden_dim),
            nn.ReLU()
        )
        
        # Head 1: Pertenencia (usa string + AFD)
        self.head1 = nn.Sequential(
            nn.Linear(self.rnn_output_dim + afd_hidden_dim, combined_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(combined_hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Head 2: Cadena compartida (solo usa string)
        self.head2 = nn.Sequential(
            nn.Linear(self.rnn_output_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, string_tokens, string_lengths, afd_features):
        """
        Args:
            string_tokens: [batch_size, max_len]
            string_lengths: [batch_size]
            afd_features: [batch_size, 3104]
        
        Returns:
            y1_hat: [batch_size, 1] - pertenencia a AFD
            y2_hat: [batch_size, 1] - cadena compartida
        """
        batch_size = string_tokens.size(0)
        device = string_tokens.device
        
        # Manejar cadenas vacías (longitud == 0)
        # Separar índices de cadenas vacías y no vacías
        empty_mask = string_lengths == 0
        non_empty_mask = ~empty_mask
        
        # Inicializar h_str con ceros
        h_str = torch.zeros(batch_size, self.rnn_output_dim, device=device)
        
        # Procesar solo cadenas no vacías
        if non_empty_mask.any():
            non_empty_tokens = string_tokens[non_empty_mask]
            non_empty_lengths = string_lengths[non_empty_mask]
            
            # Encode string
            embedded = self.embedding(non_empty_tokens)  # [n_non_empty, max_len, embed_dim]
            
            # Pack para RNN
            packed = nn.utils.rnn.pack_padded_sequence(
                embedded, 
                non_empty_lengths.cpu(), 
                batch_first=True, 
                enforce_sorted=False
            )
            
            _, hidden = self.rnn(packed)  # hidden: [num_layers*2, n_non_empty, hidden_dim]
            
            # Concatenar direcciones forward y backward de la última capa
            h_str_non_empty = torch.cat([hidden[-2], hidden[-1]], dim=1)  # [n_non_empty, rnn_output_dim]
            
            # Asignar de vuelta a h_str
            h_str[non_empty_mask] = h_str_non_empty
        
        # Si hay cadenas vacías, usar un embedding especial (vector de ceros ya inicializado)
        # Esto representa la cadena épsilon
        
        # Encode AFD
        h_afd = self.afd_encoder(afd_features)  # [batch, afd_hidden_dim]
        
        # Concatenar para Head 1
        h_combined = torch.cat([h_str, h_afd], dim=1)
        
        # Predicciones
        y1_hat = self.head1(h_combined).squeeze(1)  # [batch]
        y2_hat = self.head2(h_str).squeeze(1)  # [batch]
        
        return y1_hat, y2_hat


class Trainer:
    """Entrenador del modelo dual-task"""
    
    def __init__(self, model, train_loader, val_loader, 
                 lambda1=1.0, lambda2=1.0, lr=0.001, device='cuda'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.device = device
        
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=3, factor=0.5
        )
        self.criterion = nn.BCELoss()
        
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_acc_y1': [], 'val_acc_y1': [],
            'train_acc_y2': [], 'val_acc_y2': []
        }
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        all_y1_pred, all_y1_true = [], []
        all_y2_pred, all_y2_true = [], []
        
        for batch in self.train_loader:
            # Mover a device
            string_tokens = batch['string_tokens'].to(self.device)
            string_lengths = batch['string_lengths'].to(self.device)
            afd_features = batch['afd_features'].to(self.device)
            y1_true = batch['y1'].to(self.device)
            y2_true = batch['y2'].to(self.device)
            
            # Forward
            y1_hat, y2_hat = self.model(string_tokens, string_lengths, afd_features)
            
            # Loss
            loss1 = self.criterion(y1_hat, y1_true)
            loss2 = self.criterion(y2_hat, y2_true)
            loss = self.lambda1 * loss1 + self.lambda2 * loss2
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Acumular predicciones para accuracy
            all_y1_pred.extend((y1_hat > 0.5).cpu().numpy())
            all_y1_true.extend(y1_true.cpu().numpy())
            all_y2_pred.extend((y2_hat > 0.5).cpu().numpy())
            all_y2_true.extend(y2_true.cpu().numpy())
        
        avg_loss = total_loss / len(self.train_loader)
        acc_y1 = accuracy_score(all_y1_true, all_y1_pred)
        acc_y2 = accuracy_score(all_y2_true, all_y2_pred)
        
        return avg_loss, acc_y1, acc_y2
    
    def validate(self):
        self.model.eval()
        total_loss = 0
        all_y1_pred, all_y1_true = [], []
        all_y2_pred, all_y2_true = [], []
        
        with torch.no_grad():
            for batch in self.val_loader:
                string_tokens = batch['string_tokens'].to(self.device)
                string_lengths = batch['string_lengths'].to(self.device)
                afd_features = batch['afd_features'].to(self.device)
                y1_true = batch['y1'].to(self.device)
                y2_true = batch['y2'].to(self.device)
                
                y1_hat, y2_hat = self.model(string_tokens, string_lengths, afd_features)
                
                loss1 = self.criterion(y1_hat, y1_true)
                loss2 = self.criterion(y2_hat, y2_true)
                loss = self.lambda1 * loss1 + self.lambda2 * loss2
                
                total_loss += loss.item()
                
                all_y1_pred.extend((y1_hat > 0.5).cpu().numpy())
                all_y1_true.extend(y1_true.cpu().numpy())
                all_y2_pred.extend((y2_hat > 0.5).cpu().numpy())
                all_y2_true.extend(y2_true.cpu().numpy())
        
        avg_loss = total_loss / len(self.val_loader)
        acc_y1 = accuracy_score(all_y1_true, all_y1_pred)
        acc_y2 = accuracy_score(all_y2_true, all_y2_pred)
        
        return avg_loss, acc_y1, acc_y2
    
    def train(self, num_epochs):
        best_val_loss = float('inf')
        
        print(f"\nEntrenando modelo durante {num_epochs} épocas...")
        print("-" * 70)
        
        for epoch in range(num_epochs):
            train_loss, train_acc_y1, train_acc_y2 = self.train_epoch()
            val_loss, val_acc_y1, val_acc_y2 = self.validate()
            
            # Guardar historial
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc_y1'].append(train_acc_y1)
            self.history['val_acc_y1'].append(val_acc_y1)
            self.history['train_acc_y2'].append(train_acc_y2)
            self.history['val_acc_y2'].append(val_acc_y2)
            
            # Scheduler
            self.scheduler.step(val_loss)
            
            # Guardar mejor modelo
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), 'best_model.pt')
            
            # Log
            print(f"Época {epoch+1}/{num_epochs}")
            print(f"  Train - Loss: {train_loss:.4f} | Y1_Acc: {train_acc_y1:.4f} | Y2_Acc: {train_acc_y2:.4f}")
            print(f"  Val   - Loss: {val_loss:.4f} | Y1_Acc: {val_acc_y1:.4f} | Y2_Acc: {val_acc_y2:.4f}")
            print()


class Evaluator:
    """Evaluador completo con métricas detalladas"""
    
    def __init__(self, model, test_loader, device='cuda'):
        self.model = model.to(device)
        self.test_loader = test_loader
        self.device = device
    
    def evaluate(self):
        self.model.eval()
        
        all_y1_pred, all_y1_true, all_y1_scores = [], [], []
        all_y2_pred, all_y2_true, all_y2_scores = [], [], []
        
        with torch.no_grad():
            for batch in self.test_loader:
                string_tokens = batch['string_tokens'].to(self.device)
                string_lengths = batch['string_lengths'].to(self.device)
                afd_features = batch['afd_features'].to(self.device)
                y1_true = batch['y1'].cpu().numpy()
                y2_true = batch['y2'].cpu().numpy()
                
                y1_hat, y2_hat = self.model(string_tokens, string_lengths, afd_features)
                
                y1_scores = y1_hat.cpu().numpy()
                y2_scores = y2_hat.cpu().numpy()
                y1_pred = (y1_scores > 0.5).astype(int)
                y2_pred = (y2_scores > 0.5).astype(int)
                
                all_y1_pred.extend(y1_pred)
                all_y1_true.extend(y1_true)
                all_y1_scores.extend(y1_scores)
                
                all_y2_pred.extend(y2_pred)
                all_y2_true.extend(y2_true)
                all_y2_scores.extend(y2_scores)
        
        # Métricas para Y1 (pertenencia)
        print("\n" + "="*70)
        print("EVALUACIÓN EN TEST SET")
        print("="*70)
        
        print("\n📊 TAREA 1: Pertenencia a AFD (Y1)")
        print("-" * 70)
        acc_y1 = accuracy_score(all_y1_true, all_y1_pred)
        f1_y1 = f1_score(all_y1_true, all_y1_pred, average='binary')
        print(f"  Accuracy: {acc_y1:.4f}")
        print(f"  F1 Score: {f1_y1:.4f}")
        
        # Clasificación de rendimiento Y1
        if acc_y1 >= 0.95 and f1_y1 >= 0.95:
            print("  ✅ Rendimiento: MUY BUENO")
        elif acc_y1 >= 0.90 and f1_y1 >= 0.90:
            print("  ✔️  Rendimiento: BUENO")
        elif acc_y1 >= 0.85:
            print("  ⚠️  Rendimiento: REGULAR")
        else:
            print("  ❌ Rendimiento: MALO")
        
        # Métricas para Y2 (cadena compartida)
        print("\n📊 TAREA 2: Cadena compartida entre AFDs (Y2)")
        print("-" * 70)
        acc_y2 = accuracy_score(all_y2_true, all_y2_pred)
        f1_y2 = f1_score(all_y2_true, all_y2_pred, average='binary', zero_division=0)
        
        # PR-AUC para Y2
        precision, recall, _ = precision_recall_curve(all_y2_true, all_y2_scores)
        pr_auc_y2 = auc(recall, precision)
        
        print(f"  Accuracy: {acc_y2:.4f}")
        print(f"  F1 Score: {f1_y2:.4f}")
        print(f"  PR-AUC:   {pr_auc_y2:.4f}")
        
        # Clasificación de rendimiento Y2
        if f1_y2 >= 0.9 and pr_auc_y2 >= 0.9:
            print("  ✅ Rendimiento: BUENO")
        elif f1_y2 >= 0.8 and pr_auc_y2 >= 0.8:
            print("  ⚠️  Rendimiento: REGULAR")
        else:
            print("  ❌ Rendimiento: MALO")
        
        print("\n" + "="*70)
        
        return {
            'y1_accuracy': acc_y1,
            'y1_f1': f1_y1,
            'y2_accuracy': acc_y2,
            'y2_f1': f1_y2,
            'y2_pr_auc': pr_auc_y2
        }


def plot_training_history(history):
    """Visualiza el historial de entrenamiento"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Loss
    axes[0].plot(history['train_loss'], label='Train')
    axes[0].plot(history['val_loss'], label='Val')
    axes[0].set_title('Loss')
    axes[0].set_xlabel('Época')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Y1 Accuracy
    axes[1].plot(history['train_acc_y1'], label='Train')
    axes[1].plot(history['val_acc_y1'], label='Val')
    axes[1].set_title('Y1 Accuracy (Pertenencia)')
    axes[1].set_xlabel('Época')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    # Y2 Accuracy
    axes[2].plot(history['train_acc_y2'], label='Train')
    axes[2].plot(history['val_acc_y2'], label='Val')
    axes[2].set_title('Y2 Accuracy (Compartida)')
    axes[2].set_xlabel('Época')
    axes[2].set_ylabel('Accuracy')
    axes[2].legend()
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150)
    print("📈 Gráficas guardadas en 'training_history.png'")


def main():
    """Pipeline completo de entrenamiento y evaluación"""
    
    print("="*70)
    print("MODELO DUAL-ENCODER PARA CLASIFICACIÓN DE CADENAS EN AFDs")
    print("="*70)
    
    # 1. Cargar y parsear AFDs
    print("\n1️⃣  Cargando y parseando AFDs...")
    parser = AFDParser('dataset6000.csv')
    print(f"   ✓ {len(parser.df)} AFDs cargados")
    
    # 2. Generar dataset
    print("\n2️⃣  Generando dataset de pares (dfa_id, string, label)...")
    generator = StringDatasetGenerator(parser)
    df = generator.generate_full_dataset(pos_samples_per_dfa=30, neg_samples_per_dfa=30)
    
    # 3. Calcular y2
    print("\n3️⃣  Calculando etiqueta Y2 (cadenas compartidas)...")
    df = generator.compute_shared_label(df)
    
    # Guardar dataset generado
    df.to_csv('dataset_generated.csv', index=False)
    print(f"   ✓ Dataset guardado en 'dataset_generated.csv'")
    
    # 4. Split por dfa_id
    print("\n4️⃣  Dividiendo dataset (train/val/test por dfa_id)...")
    unique_dfas = df['dfa_id'].unique()
    train_dfas, temp_dfas = train_test_split(unique_dfas, test_size=0.3, random_state=42)
    val_dfas, test_dfas = train_test_split(temp_dfas, test_size=0.5, random_state=42)
    
    train_df = df[df['dfa_id'].isin(train_dfas)]
    val_df = df[df['dfa_id'].isin(val_dfas)]
    test_df = df[df['dfa_id'].isin(test_dfas)]
    
    print(f"   Train: {len(train_df)} ejemplos ({len(train_dfas)} AFDs)")
    print(f"   Val:   {len(val_df)} ejemplos ({len(val_dfas)} AFDs)")
    print(f"   Test:  {len(test_df)} ejemplos ({len(test_dfas)} AFDs)")
    
    # 5. Crear dataloaders
    print("\n5️⃣  Creando dataloaders...")
    train_dataset = AFDStringDataset(train_df, parser)
    val_dataset = AFDStringDataset(val_df, parser)
    test_dataset = AFDStringDataset(test_df, parser)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)
    
    # 6. Crear modelo
    print("\n6️⃣  Creando modelo dual-encoder...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"   Device: {device}")
    
    model = DualEncoderModel()
    num_params = sum(p.numel() for p in model.parameters())
    print(f"   Parámetros: {num_params:,}")
    
    # 7. Entrenar
    print("\n7️⃣  Entrenando modelo...")
    trainer = Trainer(model, train_loader, val_loader, device=device)
    trainer.train(num_epochs=30)
    
    # 8. Cargar mejor modelo y evaluar
    print("\n8️⃣  Evaluando en test set...")
    model.load_state_dict(torch.load('best_model.pt'))
    evaluator = Evaluator(model, test_loader, device=device)
    metrics = evaluator.evaluate()
    
    # 9. Visualizar
    print("\n9️⃣  Generando visualizaciones...")
    plot_training_history(trainer.history)
    
    print("\n✅ Pipeline completo finalizado!")
    print("="*70)


if __name__ == "__main__":
    main()

