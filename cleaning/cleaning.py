import pandas as pd
import os

# Get absolute path
base_path = r"C:\coding cihuy\pygwalker"
data_path = os.path.join(base_path, "data", "datamahasiswa.csv")
output_path = os.path.join(base_path, "data", "datamahasiswa_clean.csv")

# Load data
df = pd.read_csv(data_path)

print(" Data SEBELUM Cleaning:")
print(f"Total rows: {len(df)}")
print(f"Missing values:\n{df.isnull().sum()}")
print(f"Duplicates: {df.duplicated().sum()}")
print("\n" + "="*50 + "\n")

# CLEANING
# 1. Hapus missing values
df_clean = df.dropna()

# 2. Hapus duplikat
df_clean = df_clean.drop_duplicates()

# 3. Reset index
df_clean = df_clean.reset_index(drop=True)

# 4. Standarisasi text (uppercase, trim whitespace)
for col in df_clean.select_dtypes(include='object').columns:
    df_clean[col] = df_clean[col].str.strip().str.upper()
    
# 5. Standarisasi Jenis Kelamin
print("\n Kolom yang ada:")
print(df_clean.columns.tolist())

# Ganti 'JENIS_KELAMIN' sesuai nama kolom di dataset
jenis_kelamin_map = {
    'LAKI-LAKI': 'LAKI-LAKI',
    'LAKI - LAKI': 'LAKI-LAKI',
    'LAKI LAKI': 'LAKI-LAKI',
    'PRIA': 'LAKI-LAKI',
    'MALE': 'LAKI-LAKI',
    'COWOK': 'LAKI-LAKI',
    'L': 'LAKI-LAKI',
    
    'PEREMPUAN': 'PEREMPUAN',
    'WANITA': 'PEREMPUAN',
    'CEWEK': 'PEREMPUAN',
    'FEMALE': 'PEREMPUAN',
    'P': 'PEREMPUAN',
}

df_clean['JENIS_KELAMIN'] = df_clean['JENIS_KELAMIN'].replace(jenis_kelamin_map)


# Save data bersih
df_clean.to_csv(output_path, index=False)
print(f" Data bersih tersimpan di: {output_path}")
