from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd

# Inisialisasi FastAPI
app = FastAPI(title="Covid Impacted Country GDP Cluster API")

# Load model dan scaler
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Skema input
class PDB(BaseModel):
    GDP : float
def preprocess_input(data: PDB):
    # Buat DataFrame dari input
    df = pd.DataFrame([{
        "GDP per Capita": data.GDP,
    }])

    # Normalisasi
    df_scaled = scaler.transform(df)
    return df_scaled

@app.get("/")
def read_root():
    return {"message": "GDP Cluster API is running"}

# Endpoint prediksi
@app.post("/predict")
def cluster_gdp(data: PDB):
    try:
        if data.GDP <= 0:
            raise ValueError("GDP harus lebih besar dari 0")
        processed = preprocess_input(data)
        print(f"Processed Data: {processed}")  # Melihat data yang diproses
        prediction = model.predict(processed)
        print(f"Prediction: {prediction}")  # Melihat hasil prediksi model
        result = int(prediction[0])  # Pastikan result adalah integer, bisa diubah ke string jika perlu
        if result == 0:
            result = "Ekonomi Kuat"
        elif result == 1:
            result = "Ekonomi Sedang"
        elif result == 2:
            result = "Ekonomi Lemah"
        return {"GDP": data.GDP, "Kondisi Ekonomi": result}
    except Exception as e:
        return {"error": str(e), "details": repr(e)}  # Tampilkan pesan kesalahan lebih rinci

