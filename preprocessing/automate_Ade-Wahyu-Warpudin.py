import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import os
import argparse

def load_data(filepath):
    df = pd.read_csv(filepath)
    print(f"[INFO] Data loaded: {df.shape}")
    return df

def handle_missing_values(df):
    # Menggunakan inplace=True atau assignment untuk memastikan data terhapus
    df.dropna(inplace=True)
    print(f"[INFO] Missing values handled (Dropped rows with NaN)")
    return df

def remove_duplicates(df):
    before = len(df)
    df.drop_duplicates(inplace=True)
    print(f"[INFO] Removed {before - len(df)} duplicates")
    return df

def remove_outliers_iqr(df, columns):
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        df = df[(df[col] >= Q1 - 1.5*IQR) & (df[col] <= Q3 + 1.5*IQR)]
    print(f"[INFO] Outliers removed. Shape: {df.shape}")
    return df

def encode_features(df):
    le = LabelEncoder()
    # Mencari kolom bertipe objek (teks) secara otomatis
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])
    print(f"[INFO] Encoded categorical columns: {categorical_cols}")
    return df

def scale_features(df, target_col='price_usd'):
    scaler = StandardScaler()
    feature_cols = [c for c in df.columns if c != target_col]
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    print(f"[INFO] Numerical features scaled")
    return df

def split_and_save(df, output_dir, target_col='price_usd'):
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    # Membagi data 80:20
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Membuat direktori jika belum ada
    os.makedirs(output_dir, exist_ok=True)
    
    # Menggabungkan kembali untuk disimpan ke CSV
    train_df = X_train.copy()
    train_df[target_col] = y_train.values
    test_df = X_test.copy()
    test_df[target_col] = y_test.values
    
    # Simpan sesuai kriteria tugas
    train_df.to_csv(os.path.join(output_dir, 'data_train_ready.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'data_test_ready.csv'), index=False)
    
    print(f"[INFO] Files saved to: {output_dir}")
    print(f"[INFO] Final Train Shape: {train_df.shape}, Test Shape: {test_df.shape}")

def preprocess(input_path, output_dir):
    # Urutan eksekusi yang rapi
    df = load_data(input_path)
    df = handle_missing_values(df)
    df = remove_duplicates(df)
    # Kita hanya membersihkan outlier pada harga (target) sesuai diskusi sebelumnya
    df = remove_outliers_iqr(df, ['price_usd'])
    df = encode_features(df)
    df = scale_features(df)
    split_and_save(df, output_dir)
    print("[SUCCESS] Preprocessing Ade-Wahyu complete!")

if __name__ == "__main__":
    # Menyesuaikan argument default dengan nama dataset Anda
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='used_car_price_dataset_extended.csv')
    parser.add_argument('--output', type=str, default='preprocessing/used_car_price_dataset_extended_preprocessing')
    args = parser.parse_args()
    
    preprocess(args.input, args.output)