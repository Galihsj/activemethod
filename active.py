# Import modul yang diperlukan
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans

# 1. Memuat dataset Iris dan melakukan preprocessing
iris = load_iris()
X = iris.data  # Fitur
y = iris.target  # Label

# Standardisasi data agar sesuai dengan jaringan saraf
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Membagi dataset menjadi labeled (X_train) dan unlabeled (X_pool)
X_train, X_pool, y_train, y_pool = train_test_split(X, y, test_size=0.8, random_state=42)

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

# 3. Metode sampling untuk Active Learning
# Uncertainty Sampling
def uncertainty_sampling(model, X_pool, n_queries=5):
    model.eval()  # Mengubah model ke mode evaluasi
    with torch.no_grad():  # Mematikan perhitungan gradien
        X_pool_tensor = torch.FloatTensor(X_pool)  # Konversi ke tensor
        probs = model(X_pool_tensor).numpy()  # Prediksi probabilitas
        uncertainty = -np.sum(probs * np.log(probs + 1e-10), axis=1)  # Entropi sebagai ukuran ketidakpastian
    query_idx = np.argsort(uncertainty)[-n_queries:]  # Pilih data dengan ketidakpastian tertinggi
    return query_idx

# Query by Committee (QBC)
def query_by_committee(models, X_pool, n_queries=5):
    disagreements = []  # Untuk menyimpan jumlah ketidaksepakatan antar model
    X_pool_tensor = torch.FloatTensor(X_pool)  # Konversi ke tensor
    for x in X_pool_tensor:
        preds = [torch.argmax(model(x.unsqueeze(0))).item() for model in models]  # Prediksi dari setiap model
        disagreements.append(len(set(preds)))  # Hitung jumlah prediksi yang berbeda
    query_idx = np.argsort(disagreements)[-n_queries:]  # Pilih data dengan ketidaksepakatan tertinggi
    return query_idx

# Diversity Sampling
def diversity_sampling(X_pool, n_queries=5):
    kmeans = KMeans(n_clusters=n_queries, random_state=42).fit(X_pool)  # Klasterisasi K-Means
    query_idx = []
    for center in kmeans.cluster_centers_:
        distances = np.linalg.norm(X_pool - center, axis=1)  # Hitung jarak ke pusat klaster
        query_idx.append(np.argmin(distances))  # Pilih data terdekat ke pusat
    return np.array(query_idx)

# 4. Proses Deep Active Learning
# Inisialisasi model dan optimizer
input_size = X_train.shape[1]
num_classes = len(set(y))
model = NeuralNetwork(input_size, num_classes)  # Model utama
criterion = nn.CrossEntropyLoss()  # Fungsi loss untuk klasifikasi
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Optimizer Adam

# Membuat komite model untuk QBC
committee = [NeuralNetwork(input_size, num_classes) for _ in range(3)]

# Iterasi Active Learning
for iteration in range(5):  # 5 iterasi
    # Latih model pada data labeled
    model.train()  # Mengubah model ke mode pelatihan
    X_train_tensor = torch.FloatTensor(X_train)  # Konversi data ke tensor
    y_train_tensor = torch.LongTensor(y_train)  # Konversi label ke tensor
    for epoch in range(50):  # Latihan selama 50 epoch
        optimizer.zero_grad()  # Reset gradien
        outputs = model(X_train_tensor)  # Prediksi model
        loss = criterion(outputs, y_train_tensor)  # Hitung loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update bobot
    
    # Evaluasi akurasi pada data yang dilabeli
    model.eval()  # Ubah ke mode evaluasi
    preds = torch.argmax(model(torch.FloatTensor(X_train)), axis=1).numpy()  # Prediksi label
    print(f"Iteration {iteration + 1}, Accuracy: {accuracy_score(y_train, preds):.2f}")
    
    # Pilih metode sampling berdasarkan iterasi
    if iteration % 3 == 0:
        query_idx = uncertainty_sampling(model, X_pool, n_queries=5)
        print("Using Uncertainty Sampling")
    elif iteration % 3 == 1:
        query_idx = query_by_committee(committee, X_pool, n_queries=5)
        print("Using Query by Committee")
    else:
        query_idx = diversity_sampling(X_pool, n_queries=5)
        print("Using Diversity Sampling")
    
    # Tambahkan data yang dipilih ke labeled dataset
    X_train = np.vstack([X_train, X_pool[query_idx]])  # Tambah data ke X_train
    y_train = np.hstack([y_train, y_pool[query_idx]])  # Tambah label ke y_train
    
    # Hapus data yang dipilih dari pool
    X_pool = np.delete(X_pool, query_idx, axis=0)  # Hapus data dari X_pool
    y_pool = np.delete(y_pool, query_idx, axis=0)  # Hapus label dari y_pool
