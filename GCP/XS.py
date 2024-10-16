import torch
import torch.nn as nn
import spacy
import numpy as np

# Cargar modelo de spaCy para obtener embeddings en español
nlp = spacy.load('es_core_news_lg')

# Clase del modelo básico para la predicción de verbos modales
class FraseClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FraseClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Función para obtener embeddings de spaCy
def obtener_embeddings(frase):
    doc = nlp(frase)
    return doc.vector  # Devuelve el vector de la frase completa

# Cargar el modelo guardado
def cargar_modelo(filepath):
    input_dim = 300  # Dimensión del vector de embeddings de spaCy
    hidden_dim = 256
    output_dim = 1  # Para clasificación binaria

    modelo = FraseClassifier(input_dim, hidden_dim, output_dim)
    modelo.load_state_dict(torch.load(filepath))
    modelo.eval()  # Cambia el modelo a modo de evaluación
    return modelo

# Predecir si se necesita un verbo modal
def predecir(frase, modelo_modales):
    with torch.no_grad():
        embeddings = torch.tensor(obtener_embeddings(frase), dtype=torch.float32).unsqueeze(0)
        salida_modal = torch.sigmoid(modelo_modales(embeddings)).item()
        return salida_modal > 0.5

def obtener_verbo_modal_sugerido(frase, modales_sugeridos):
    # Obtener el embedding de la frase
    embedding_frase = obtener_embeddings(frase)

    # Diccionario para almacenar las similitudes
    similitudes = {}

    for verbo in modales_sugeridos:
        # Obtener el embedding del verbo
        embedding_verbo = obtener_embeddings(verbo)

        # Calcular la similitud semántica entre la frase y el verbo
        similitud = np.dot(embedding_frase, embedding_verbo) / (np.linalg.norm(embedding_frase) * np.linalg.norm(embedding_verbo))  # Cosine similarity
        similitudes[verbo] = similitud

    # Ajustar el criterio para seleccionar el mejor verbo modal
    mejor_verbo_modal = max(similitudes, key=lambda k: (similitudes[k], k), default=None)

    return mejor_verbo_modal  # Retornar el verbo con la mayor similitud


def insertar_verbo_modal(frase, verbo_modal):
    doc = nlp(frase)
    
    # Encontrar el verbo principal en la frase
    verbo_principal = None
    for token in doc:
        if token.pos_ == "VERB":
            verbo_principal = token
            break
    
    # Si se encuentra un verbo principal, insertar el modal antes de él
    if verbo_principal:
        posicion = verbo_principal.i
        # Crear una lista de palabras para reensamblar la frase
        palabras = [token.text for token in doc]
        # Insertar el verbo modal
        palabras.insert(posicion, verbo_modal)
        # Unir las palabras en una nueva frase
        return ' '.join(palabras)
    else:
        # Si no hay verbo principal, simplemente colocarlo al principio
        return f"{verbo_modal} {frase}"

# Ejecución principal
if __name__ == '__main__':
    # Cargar el modelo guardado
    modelo_modales = cargar_modelo("modelo_verbos_modales.pth")

    # Lista de verbos modales sugeridos (modificada según tus datos JSON)
    modales_sugeridos = [
        "quiero",
        "debo",
        "puedo",
        "necesito",
        "debes",
        "puedes",
        "tienes que"
    ]

    # Frase a analizar
    frase_prueba = "Yo ir a casa"
    # Predecir si la frase necesita un verbo modal
    necesita_modal = predecir(frase_prueba, modelo_modales)

    if necesita_modal:
        verbo_modal = obtener_verbo_modal_sugerido(frase_prueba, modales_sugeridos)
        
        # Insertar el verbo modal en la posición correcta
        frase_mejorada = insertar_verbo_modal(frase_prueba, verbo_modal)
        
        print(f"Frase original: '{frase_prueba}'")
        print(f"Frase mejorada: '{frase_mejorada}'")
    else:
        print(f"La frase '{frase_prueba}' no necesita un verbo modal.")
