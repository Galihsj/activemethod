# Import modul yang diperlukan
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# 1. Memuat dataset Iris dan melakukan preprocessing
iris = load_iris()
X = iris.data  # Fitur
y = iris.target  # Label

# Standardisasi data agar sesuai dengan jaringan saraf
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Membagi dataset menjadi train dan test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Definisi model jaringan saraf
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)  # Lapisan pertama dengan 64 neuron
        self.relu = nn.ReLU()  # Fungsi aktivasi ReLU
        self.fc2 = nn.Linear(64, 32)  # Lapisan kedua dengan 32 neuron
        self.fc3 = nn.Linear(32, num_classes)  # Lapisan keluaran
        self.softmax = nn.Softmax(dim=1)  # Softmax untuk prediksi probabilitas
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.softmax(x)
        return x

# Inisialisasi model, loss function, dan optimizer
input_size = X_train.shape[1]
num_classes = len(set(y))
model = NeuralNetwork(input_size, num_classes)
criterion = nn.CrossEntropyLoss()  # Fungsi loss untuk klasifikasi
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Optimizer Adam

# 3. Pelatihan model
X_train_tensor = torch.FloatTensor(X_train)  # Konversi data ke tensor
y_train_tensor = torch.LongTensor(y_train)  # Konversi label ke tensor

# Latihan selama 100 epoch
for epoch in range(100):
    model.train()  # Mengubah model ke mode pelatihan
    optimizer.zero_grad()  # Reset gradien
    outputs = model(X_train_tensor)  # Prediksi model
    loss = criterion(outputs, y_train_tensor)  # Hitung loss
    loss.backward()  # Backpropagation
    optimizer.step()  # Update bobot
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

# 4. Evaluasi model
model.eval()  # Mengubah model ke mode evaluasi
X_test_tensor = torch.FloatTensor(X_test)  # Konversi data ke tensor
y_test_tensor = torch.LongTensor(y_test)  # Konversi label ke tensor

# Prediksi pada data uji
with torch.no_grad():
    y_pred_probs = model(X_test_tensor)  # Prediksi probabilitas
    y_pred = torch.argmax(y_pred_probs, axis=1).numpy()  # Prediksi label

# Hitung akurasi
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy on test set: {accuracy:.2f}")
# Import modul yang diperlukan
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# 1. Memuat dataset Iris dan melakukan preprocessing
iris = load_iris()
X = iris.data  # Fitur
y = iris.target  # Label

# Standardisasi data agar sesuai dengan jaringan saraf
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Membagi dataset menjadi train dan test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Definisi model jaringan saraf
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)  # Lapisan pertama dengan 64 neuron
        self.relu = nn.ReLU()  # Fungsi aktivasi ReLU
        self.fc2 = nn.Linear(64, 32)  # Lapisan kedua dengan 32 neuron
        self.fc3 = nn.Linear(32, num_classes)  # Lapisan keluaran
        self.softmax = nn.Softmax(dim=1)  # Softmax untuk prediksi probabilitas
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.softmax(x)
        return x

# Inisialisasi model, loss function, dan optimizer
input_size = X_train.shape[1]
num_classes = len(set(y))
model = NeuralNetwork(input_size, num_classes)
criterion = nn.CrossEntropyLoss()  # Fungsi loss untuk klasifikasi
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Optimizer Adam

# 3. Pelatihan model
X_train_tensor = torch.FloatTensor(X_train)  # Konversi data ke tensor
y_train_tensor = torch.LongTensor(y_train)  # Konversi label ke tensor

# Latihan selama 100 epoch
for epoch in range(100):
    model.train()  # Mengubah model ke mode pelatihan
    optimizer.zero_grad()  # Reset gradien
    outputs = model(X_train_tensor)  # Prediksi model
    loss = criterion(outputs, y_train_tensor)  # Hitung loss
    loss.backward()  # Backpropagation
    optimizer.step()  # Update bobot
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

# 4. Evaluasi model
model.eval()  # Mengubah model ke mode evaluasi
X_test_tensor = torch.FloatTensor(X_test)  # Konversi data ke tensor
y_test_tensor = torch.LongTensor(y_test)  # Konversi label ke tensor

# Prediksi pada data uji
with torch.no_grad():
    y_pred_probs = model(X_test_tensor)  # Prediksi probabilitas
    y_pred = torch.argmax(y_pred_probs, axis=1).numpy()  # Prediksi label

# Hitung akurasi
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy on test set: {accuracy:.2f}")
