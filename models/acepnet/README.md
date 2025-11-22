# 🤖 AcepNet - Inferencia de Modelos AFD

Paquete standalone para ejecutar inferencia con el modelo mejorado de clasificación de cadenas en Autómatas Finitos Deterministas.

## 📦 Contenido del Paquete

```
acepnet/
├── inferencia_mejorada.py    # Script principal de inferencia
├── acepten.py                 # Módulo con clases del modelo
├── best_model.pt              # Modelo entrenado (mejorado con augmentación)
├── thresholds.json            # Umbrales calibrados (Y1=0.43, Y2=0.53)
├── dataset6000.csv            # Dataset con 6000 AFDs
└── README.md                  # Este archivo
```

## 🚀 Uso Rápido

### Requisitos
```bash
pip install torch pandas numpy scikit-learn
```

### Ejecutar
```bash
cd acepnet
python inferencia_mejorada.py
```

## 🎮 Opciones del Menú

### 1. 🎯 Probar cadena con un AFD (por ID)
Ingresa el ID del AFD (0-5999) y una cadena para ver:
- Información del AFD (regex, alfabeto, estados)
- Predicción del modelo (probabilidad y veredicto)
- Simulación real del AFD (ground truth)
- Comparación: si el modelo acertó o falló

**Ejemplo:**
```
ID del AFD: 0
Cadena: C
→ Muestra si la cadena es aceptada por el AFD #0
```

### 2. 🔍 Buscar AFD por palabra clave
Busca AFDs que contengan una palabra en su regex y luego prueba cadenas.

**Ejemplo:**
```
Palabra clave: AB
→ Muestra AFDs cuya regex contenga "AB"
```

### 3. 📋 Ver información de un AFD
Muestra detalles del AFD sin hacer predicción.

### 4. 🎲 Ejemplos predefinidos
Ejecuta casos de prueba automáticos para verificar el modelo.

### 5. 🚪 Salir
Cierra el programa.

## 📊 Características del Modelo

- **Arquitectura**: Dual-Encoder (String + AFD)
- **Tareas**:
  - Y1: Pertenencia a AFD específico
  - Y2: Cadena compartida entre múltiples AFDs
- **Entrenamiento**: Con augmentación de datos (positivos y negativos)
- **Umbrales calibrados**: Y1=0.43, Y2=0.53
- **Validación de alfabeto**: Rechaza automáticamente cadenas con símbolos fuera del alfabeto del AFD

## 🎯 Ejemplo de Salida

```
📋 AFD SELECCIONADO: #0
  📌 Regex: [LCIG]+
  🔤 Alfabeto: C G I L
  🔢 Estados: S0 S1 S2 S3 S4
  ✅ Estados de aceptación: S3 S4 S2 S1

✍️  Ingresa la cadena: C

📊 RESULTADO DE LA PREDICCIÓN
Cadena evaluada: 'C'

🤖 PREDICCIÓN DEL MODELO:
   Probabilidad: 0.9719
   Veredicto: ✅ ACEPTA

🎯 SIMULADOR REAL (Ground Truth):
   Veredicto: ✅ ACEPTA

🎉 ¡CORRECTO! El modelo predijo correctamente
```

## 📝 Notas

- El modelo usa umbrales calibrados para mejor precisión
- Valida automáticamente el alfabeto del AFD
- Compara predicción vs simulación real en tiempo real
- Dataset: 6000 AFDs con diferentes expresiones regulares

## 🔧 Troubleshooting

**Error: "No module named 'acepten'"**
- Asegúrate de estar en la carpeta `acepnet/`

**Error: "File not found: best_model.pt"**
- Verifica que todos los archivos estén en la misma carpeta

**Error: "CUDA out of memory"**
- El modelo usa CPU por defecto, no requiere GPU

## 📄 Licencia

Proyecto de investigación en aprendizaje automático aplicado a teoría de autómatas.

