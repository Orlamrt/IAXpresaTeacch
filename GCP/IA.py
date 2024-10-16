import torch
import torch.nn as nn
import torch.optim as optim
import spacy
import json
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Cargar modelo de spaCy para obtener embeddings en español
nlp = spacy.load('es_core_news_lg')  # O 'es_core_news_lg' para más precisión

# Clase del modelo básico para la predicción de verbos modales
class FraseClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.2):
        super(FraseClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)  # Capa de Dropout
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)  # Aplicar Dropout
        out = self.fc2(out)
        return out

# Función para obtener embeddings de spaCy
def obtener_embeddings(frase):
    doc = nlp(frase)
    print(f"Dimensiones del vector de embeddings: {doc.vector.shape}")  # Mostrar la frase y las dimensiones
    return doc.vector  # Devuelve el vector de la frase completa

# Cargar los datos desde un archivo JSON
def cargar_datos(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

# Preparar los datos para entrenamiento
def preparar_datos(data):
    X = []
    y_modales = []
    modales_sugeridos = []

    for item in data:
        X.append(obtener_embeddings(item['frase']))
        y_modales.append(item['requiere_modal'])
        modales_sugeridos.append(item.get('verbo_modal_sugerido', ''))

    return torch.tensor(X, dtype=torch.float32), torch.tensor(y_modales, dtype=torch.float32), modales_sugeridos

# Definir el modelo de entrenamiento
def entrenar_modelo(X_train, y_train, modelo, X_val, y_val, epochs=600, lr=0.001, batch_size=32, patience=10):
    optimizer = optim.Adam(modelo.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    
    # Listas para almacenar las pérdidas
    train_losses = []
    val_losses = []

    best_val_loss = float('inf')
    patience_counter = 0

    # Entrenamiento del modelo
    for epoch in range(epochs):
        modelo.train()
        permuted_indices = torch.randperm(X_train.size(0))  # Mezclar índices para el tamaño de lote
        for i in range(0, X_train.size(0), batch_size):
            indices = permuted_indices[i:i + batch_size]
            batch_X = X_train[indices]
            batch_y = y_train[indices]

            optimizer.zero_grad()
            outputs = modelo(batch_X)
            loss = criterion(outputs.squeeze(), batch_y)
            loss.backward()
            optimizer.step()

        # Almacenar la pérdida de entrenamiento
        train_losses.append(loss.item())

        # Calcular la pérdida de validación
        modelo.eval()
        with torch.no_grad():
            val_outputs = modelo(X_val)
            val_loss = criterion(val_outputs.squeeze(), y_val).item()
            val_losses.append(val_loss)

        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}, Val Loss: {val_loss}')

        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0  # Reiniciar el contador de paciencia
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered!")
                break

    # Graficar pérdidas
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Pérdida de Entrenamiento', color='blue')
    plt.plot(val_losses, label='Pérdida de Validación', color='orange')
    plt.title('Pérdida durante el Entrenamiento')
    plt.xlabel('Época')
    plt.ylabel('Pérdida')
    plt.legend()
    plt.grid()
    plt.show()

# Ejecución principal
if __name__ == '__main__':
    # Cargar los datos de entrenamiento desde un archivo JSON
    data = cargar_datos('datos_frases.json')

    # Preparar los datos (X: features, y_modales: labels)
    X, y_modales, modales_sugeridos = preparar_datos(data)

    # Dividir los datos en entrenamiento y validación
    X_train, X_val, y_train, y_val = train_test_split(X, y_modales, test_size=0.2, random_state=42)

    # Dimensiones para el modelo
    input_dim = 300  # Cambia esto si el tamaño de tu embedding es 96
    hidden_dim =  256 # Puedes experimentar con este valor
    output_dim = 1  # Para clasificación binaria

    # Crear el modelo para verbos modales
    modelo_modales = FraseClassifier(input_dim, hidden_dim, output_dim)
    
    # Entrenar el modelo para verbos modales
    print("Entrenando el modelo para verbos modales...")
    entrenar_modelo(X_train, y_train, modelo_modales, X_val, y_val, batch_size=32)  # Ajustar el tamaño de lote aquí
   
    # Guardar el modelo entrenado
    torch.save(modelo_modales.state_dict(), "modelo_verbos_modales.pth")

    # Evaluar el modelo en el conjunto de validación
    modelo_modales.eval()
    with torch.no_grad():
        outputs = modelo_modales(X_val)
        val_loss = nn.BCEWithLogitsLoss()(outputs.squeeze(), y_val).item()
        print(f'Validation Loss: {val_loss}')
