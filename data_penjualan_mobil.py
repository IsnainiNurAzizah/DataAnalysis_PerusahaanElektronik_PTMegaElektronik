import pandas as pd

# Mengimpor data dari file CSV
data = pd.read_csv('data_penjualan_mobil.csv')

# Menampilkan beberapa baris pertama data
print(data.head())
# Menghapus baris dengan nilai yang hilang
data = data.dropna()

# Menghapus duplikat
data = data.drop_duplicates()

# Menampilkan info ringkasan data
print(data.info())
# Mengubah jenis data kolom jika diperlukan
data['faktur'] = data['faktur'].astype(str)
data['merek'] = data['merek'].astype('category')
data['tanggal'] = pd.to_datetime(data['tanggal'])

# Menambahkan kolom pendapatan
data['pendapatan'] = data['jumlah'] * data['harga_per_unit']

# Menampilkan data yang sudah ditransformasi
print(data.head())
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib_venn import venn3

# Histogram penjualan per merek
plt.figure(figsize=(10,6))
sns.histplot(data, x='merek', kde=False, bins=10)
plt.title('Histogram Penjualan per Merek')
plt.xlabel('Merek')
plt.ylabel('Jumlah Penjualan')
plt.show()

# Diagram Venn penjualan (contoh sederhana, menggunakan beberapa library seperti matplotlib_venn)
# Contoh data Venn
venn_data = {'Toyota': 10, 'Honda': 8, 'BMW': 5, 'Toyota & Honda': 3, 'Toyota & BMW': 2, 'Honda & BMW': 1, 'Toyota & Honda & BMW': 1}

# Membuat diagram Venn
plt.figure(figsize=(8,8))
venn3(subsets=(venn_data['Toyota'], venn_data['Honda'], venn_data['Toyota & Honda'], venn_data['BMW'], venn_data['Toyota & BMW'], venn_data['Honda & BMW'], venn_data['Toyota & Honda & BMW']),
      set_labels=('Toyota', 'Honda', 'BMW'))
plt.title('Diagram Venn Penjualan')
plt.show()

# Hubungan antara penjualan dan pendapatan
plt.figure(figsize=(10,6))
sns.scatterplot(x='jumlah', y='pendapatan', data=data, hue='merek')
plt.title('Hubungan antara Penjualan dan Pendapatan')
plt.xlabel('Jumlah Penjualan')
plt.ylabel('Pendapatan')
plt.show()
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Memisahkan fitur dan target
X = data[['merek', 'jumlah']]
y = data['pendapatan']

# Mengubah data kategori ke dalam bentuk numerik
X = pd.get_dummies(X, drop_first=True)

# Memisahkan data latih dan uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Membangun model regresi linier
model = LinearRegression()
model.fit(X_train, y_train)
from sklearn.metrics import mean_squared_error

# Memprediksi pendapatan pada data uji
y_pred = model.predict(X_test)

# Menghitung Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print('MSE:', mse)
# Menampilkan koefisien model
coefficients = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])
print(coefficients)

# Menyajikan hasil prediksi vs aktual
results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(results.head())
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    prediction = model.predict(pd.DataFrame([data]))
    return jsonify(prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
# Membuat fungsi untuk memperbarui model dengan data baru
def update_model(new_data):
    global model, X_train, y_train
    
    # Mengimpor data baru
    new_data = pd.read_csv(new_data)
    
    # Menggabungkan dengan data lama
    X_new = new_data[['merek', 'jumlah']]
    y_new = new_data['pendapatan']
    X_new = pd.get_dummies(X_new, drop_first=True)
    
    X_train = pd.concat([X_train, X_new])
    y_train = pd.concat([y_train, y_new])
    
    # Melatih ulang model
    model.fit(X_train, y_train)
    print('Model updated with new data')

# Contoh penggunaan fungsi update_model
update_model('data_penjualan_mobil_baru.csv')
import pandas as pd

