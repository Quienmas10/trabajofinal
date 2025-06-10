import pandas as pd
import json
import os
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pytesseract
from PIL import Image
try:
    from pdf2image import convert_from_path
except ImportError:
    convert_from_path = None

# -----------------------------------
# Configuración inicial
# -----------------------------------
app = Flask(__name__)

# Directorios y archivos
DOCS_FOLDER = "documentos"
TRAMITES_FILE = "tramites.csv"
DOC_METADATA_FILE = "documentos.json"
HISTORICAL_DATA_FILE = "datos_historicos.csv"
CONGESTION_FILE = "analisis_congestion.json"

# Credenciales de administrador
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "admin123"

# Crear directorios si no existen
if not os.path.exists(DOCS_FOLDER):
    os.makedirs(DOCS_FOLDER)
if not os.path.exists('css'):
    os.makedirs('css')
if not os.path.exists('img'):
    os.makedirs('img')

# -----------------------------------
# Funciones auxiliares
# -----------------------------------

def init_files():
    """Inicializa los archivos CSV y JSON si no existen."""
    if not os.path.exists(TRAMITES_FILE):
        df = pd.DataFrame(columns=[
            "ID_Tramite", "Tipo", "Fecha", "Estado", "Nombre_Ciudadano",
            "Correo", "Telefono", "Prioridad", "Errores_Detectados"
        ])
        df.to_csv(TRAMITES_FILE, index=False)
    if not os.path.exists(DOC_METADATA_FILE):
        with open(DOC_METADATA_FILE, 'w') as f:
            json.dump([], f)
    if not os.path.exists(HISTORICAL_DATA_FILE):
        df = pd.DataFrame(columns=[
            "ID", "Tipo", "Tiempo_Procesamiento", "Errores", "Prioridad"
        ])
        df.to_csv(HISTORICAL_DATA_FILE, index=False)

def generate_historical_data():
    """Genera datos históricos simulados para el modelo de ML."""
    data = [
        {"ID": f"{i:03d}", "Tipo": t, "Tiempo_Procesamiento": tp, "Errores": e, "Prioridad": p}
        for i, (t, tp, e, p) in enumerate([
            ("Licencia", 5, "Ninguno", "Alta"),
            ("Permiso", 3, "Falta_Dato", "Media"),
            ("Servicio", 7, "Ninguno", "Baja"),
            ("Licencia", 4, "Falta_Firma", "Alta"),
            ("Permiso", 2, "Ninguno", "Media")
        ])
    ]
    df = pd.DataFrame(data)
    df.to_csv(HISTORICAL_DATA_FILE, index=False)

def process_document(file_path, tramite_id):
    """Procesa un documento (PDF o imagen) con OCR y guarda metadatos."""
    try:
        if file_path.endswith('.pdf'):
            if convert_from_path is None:
                return "Error: pdf2image no está instalado; no se pueden procesar PDFs"
            images = convert_from_path(file_path)
            text = "".join(pytesseract.image_to_string(img) + " " for img in images)
        else:
            img = Image.open(file_path)
            text = pytesseract.image_to_string(img)
        metadata = {"ID_Tramite": tramite_id, "Texto_OCR": text.strip()}
        with open(DOC_METADATA_FILE, 'r') as f:
            docs = json.load(f)
        docs.append(metadata)
        with open(DOC_METADATA_FILE, 'w') as f:
            json.dump(docs, f, indent=2)
        return text.strip()
    except Exception as e:
        return f"Error en OCR: {str(e)}"

def train_priority_model():
    """Entrena un modelo de Random Forest para predecir la prioridad de trámites."""
    df = pd.read_csv(HISTORICAL_DATA_FILE)
    le_tipo = LabelEncoder()
    le_prioridad = LabelEncoder()
    df["Tipo_Encoded"] = le_tipo.fit_transform(df["Tipo"])
    df["Prioridad_Encoded"] = le_prioridad.fit_transform(df["Prioridad"])
    X = df[["Tipo_Encoded", "Tiempo_Procesamiento"]]
    y = df["Prioridad_Encoded"]
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    return model, le_tipo, le_prioridad

def predict_priority(model, le_tipo, le_prioridad, tipo, tiempo_procesamiento):
    """Predice la prioridad de un trámite usando el modelo entrenado."""
    input_data = pd.DataFrame(
        [[le_tipo.transform([tipo])[0], tiempo_procesamiento]],
        columns=["Tipo_Encoded", "Tiempo_Procesamiento"]
    )
    prediction = model.predict(input_data)
    return le_prioridad.inverse_transform(prediction)[0]

def detect_errors(text):
    """Detecta errores en el texto extraído (simulación)."""
    if "firma" not in text.lower() and "solicitud" in text.lower():
        return "Falta_Firma"
    elif len(text.strip()) < 20:
        return "Falta_Dato"
    return "Ninguno"

def send_notification(nombre, correo, tramite_id, estado):
    """Simula el envío de una notificación al ciudadano."""
    msg = f"Estimado {nombre}, su trámite {tramite_id} está: {estado}"
    print(f"SIMULACIÓN - Enviando notificación a {correo}: {msg}")

def analyze_congestion():
    """Analiza la congestión por tipo de trámite y guarda resultados."""
    df = pd.read_csv(HISTORICAL_DATA_FILE)
    congestion = df.groupby("Tipo").agg({"Tiempo_Procesamiento": "mean"}).to_dict()
    with open(CONGESTION_FILE, 'w') as f:
        json.dump({"Congestion_Por_Tipo": congestion}, f, indent=2)

# -----------------------------------
# Rutas de la aplicación
# -----------------------------------

@app.route('/')
def index():
    """Sirve la página inicial."""
    init_files()
    generate_historical_data()
    print("Serving index.html")
    return send_from_directory('.', 'index.html')

@app.route('/tramites')
def tramites():
    """Sirve la página de trámites para ciudadanos."""
    print("Serving tramites.html")
    return send_from_directory('.', 'tramites.html')

@app.route('/admin')
def admin():
    """Sirve la página de administración."""
    print("Serving admin.html")
    return send_from_directory('.', 'admin.html')

@app.route('/css/<path:filename>')
def css_files(filename):
    """Sirve archivos desde la carpeta css."""
    print(f"Serving css file: {filename}")
    return send_from_directory('css', filename)

@app.route('/img/<path:filename>')
def img_files(filename):
    """Sirve archivos desde la carpeta img."""
    print(f"Serving img file: {filename}")
    return send_from_directory('img', filename)

@app.route('/submit', methods=['POST'])
def submit_tramite():
    """Procesa y registra un nuevo trámite enviado por un ciudadano."""
    init_files()
    df = pd.read_csv(TRAMITES_FILE, dtype={'ID_Tramite': str})
    tramite_id = f"{len(df) + 1:03d}"
    fecha = datetime.now().strftime("%Y-%m-%d")
    estado = "Recibido"

    tipo = request.form['tipo']
    nombre = request.form['nombre']
    correo = request.form['correo']
    telefono = request.form['telefono']
    documento = request.files['documento']

    # Guardar documento
    file_path = os.path.join(DOCS_FOLDER, f"temp_{tramite_id}")
    documento.save(file_path)

    # Procesar documento
    text = process_document(file_path, tramite_id)
    os.rename(file_path, os.path.join(DOCS_FOLDER, f"Tramite_{tramite_id}{os.path.splitext(documento.filename)[1]}"))

    # Predecir prioridad
    model, le_tipo, le_prioridad = train_priority_model()
    tiempo_procesamiento = 5  # Simulación
    prioridad = predict_priority(model, le_tipo, le_prioridad, tipo, tiempo_procesamiento)

    # Detectar errores
    errores = detect_errors(text)
    estado = "En_Proceso" if errores == "Ninguno" else "Pendiente_Correccion"

    # Registrar trámite
    new_row = {
        "ID_Tramite": tramite_id, "Tipo": tipo, "Fecha": fecha, "Estado": estado,
        "Nombre_Ciudadano": nombre, "Correo": correo, "Telefono": telefono,
        "Prioridad": prioridad, "Errores_Detectados": errores
    }
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(TRAMITES_FILE, index=False)

    # Enviar notificación
    send_notification(nombre, correo, tramite_id, estado)

    # Analizar congestión
    analyze_congestion()

    return jsonify({"tramite_id": tramite_id, "prioridad": prioridad, "errores": errores})

@app.route('/status')
def check_status():
    """Consulta el estado de un trámite por su ID."""
    tramite_id = request.args.get('tramite_id')
    if not tramite_id:
        return jsonify({"error": "No se proporcionó ID de trámite"})

    try:
        cleaned_id = tramite_id.strip()
        formatted_id = f"{int(cleaned_id):03d}"
    except ValueError:
        return jsonify({"error": "ID de trámite inválido, debe ser un número"})

    df = pd.read_csv(TRAMITES_FILE, dtype={'ID_Tramite': str})
    result = df[df['ID_Tramite'] == formatted_id].to_dict('records')

    if result:
        return jsonify(result[0])
    return jsonify({"error": "Trámite no encontrado"})

@app.route('/api/tramites')
def get_tramites():
    """Devuelve todos los trámites registrados (API para admin)."""
    df = pd.read_csv(TRAMITES_FILE, dtype={'ID_Tramite': str})
    return jsonify(df.to_dict('records'))

@app.route('/admin_login', methods=['POST'])
def admin_login():
    """Autentica al administrador."""
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
        return jsonify({"success": True})
    return jsonify({"error": "Usuario o contraseña incorrectos"})

@app.route('/update_state', methods=['POST'])
def update_state():
    """Actualiza el estado de un trámite."""
    data = request.get_json()
    tramite_id = data.get('tramite_id')
    new_state = data.get('new_state')

    if not tramite_id or not new_state:
        return jsonify({"error": "Falta ID de trámite o nuevo estado"})

    valid_states = ["Recibido", "En_Proceso", "Pendiente_Correccion", "Finalizado"]
    if new_state not in valid_states:
        return jsonify({"error": "Estado inválido"})

    try:
        cleaned_id = tramite_id.strip()
        formatted_id = f"{int(cleaned_id):03d}"
    except ValueError:
        return jsonify({"error": "ID de trámite inválido, debe ser un número"})

    df = pd.read_csv(TRAMITES_FILE, dtype={'ID_Tramite': str})
    if formatted_id not in df['ID_Tramite'].values:
        return jsonify({"error": "Trámite no encontrado"})

    # Actualizar estado
    df.loc[df['ID_Tramite'] == formatted_id, 'Estado'] = new_state
    df.to_csv(TRAMITES_FILE, index=False)

    # Simular notificación al ciudadano
    tramite = df[df['ID_Tramite'] == formatted_id].iloc[0]
    send_notification(tramite['Nombre_Ciudadano'], tramite['Correo'], formatted_id, new_state)

    return jsonify({"success": True})

@app.route('/congestion')
def get_congestion():
    """Devuelve el análisis de congestión."""
    with open(CONGESTION_FILE, 'r') as f:
        congestion = json.load(f)
    return jsonify(congestion)

# -----------------------------------
# Iniciar la aplicación
# -----------------------------------
if __name__ == '__main__':
    app.run(debug=True)