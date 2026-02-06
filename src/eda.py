
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Setup
base_path = r"C:\coding cihuy\pygwalker"
data_path = os.path.join(base_path, "data", "datamahasiswa_clean.csv")
output_dir = os.path.join(base_path, "results", "visualizations")

# Buat folder output
os.makedirs(output_dir, exist_ok=True)

# Load data
df = pd.read_csv(data_path)

print("="*60)
print(" EXPLORATORY DATA ANALYSIS (EDA)")
print("="*60)

# 1. INFO DASAR
print("\n INFO DATASET:")
print(f"Total data: {len(df)} rows, {len(df.columns)} columns")

# 2. ENCODE TARGET
df['target_encoded'] = df['STATUS_KELULUSAN'].map({
    'LULUS TEPAT WAKTU': 1,
    'TEPAT WAKTU': 1,
    'TEPAT': 1,
    'TERLAMBAT': 0,
    'TELAT': 0,
    'TIDAK TEPAT WAKTU': 0
})

# ========================================
# PIE CHART 1: DISTRIBUSI STATUS KELULUSAN (dengan jumlah mahasiswa)
# ========================================
print("\n DISTRIBUSI STATUS KELULUSAN:")
target_counts = df['STATUS_KELULUSAN'].value_counts()
print(target_counts)

plt.figure(figsize=(10, 7))
colors = ['#2ecc71', '#e74c3c']
explode = (0.05, 0)

def make_autopct(values):
    def my_autopct(pct):
        total = sum(values)
        val = int(round(pct*total/100.0))
        return f'{pct:.1f}%\n({val} mahasiswa)'
    return my_autopct

plt.pie(target_counts, labels=target_counts.index, autopct=make_autopct(target_counts),
        startangle=90, colors=colors, explode=explode, shadow=True,
        textprops={'fontsize': 11, 'fontweight': 'bold'})
plt.title('Distribusi Status Kelulusan', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, '01_distribusi_kelulusan.png'), dpi=300, bbox_inches='tight')
print(f" Grafik tersimpan:Distribusi_kelulusan.png")
plt.close()

# ========================================
# PIE CHART 2: DISTRIBUSI JENIS KELAMIN 
# ========================================
print("\n DISTRIBUSI JENIS KELAMIN:")
gender_counts = df['JENIS_KELAMIN'].value_counts()
print(gender_counts)

plt.figure(figsize=(10, 7))
colors_gender = ['#3498db', '#e91e63']
explode_gender = (0.05, 0)

plt.pie(gender_counts, labels=gender_counts.index, autopct=make_autopct(gender_counts),
        startangle=90, colors=colors_gender, explode=explode_gender, shadow=True,
        textprops={'fontsize': 11, 'fontweight': 'bold'})
plt.title('Distribusi Jenis Kelamin', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, '02_distribusi_gender.png'), dpi=300, bbox_inches='tight')
print(f" Grafik tersimpan: Distribusi_gender.png")
plt.close()

# ========================================
# LINE CHART: TREN IPS PER SEMESTER (2 garis, warna beda)
# ========================================
print("\n TREN IPS PER SEMESTER")
ips_cols = ['IPS_1', 'IPS_2', 'IPS_3', 'IPS_4', 'IPS_5', 'IPS_6', 'IPS_7', 'IPS_8']

# Hitung rata-rata per semester untuk yang lulus dan terlambat
lulus_avg = df[df['target_encoded'] == 1][ips_cols].mean()
terlambat_avg = df[df['target_encoded'] == 0][ips_cols].mean()

plt.figure(figsize=(12, 6))
semesters = list(range(1, 9))

# Garis 1: Lulus Tepat Waktu (hijau)
plt.plot(semesters, lulus_avg, marker='o', linewidth=2.5, markersize=10, 
         color='#2ecc71', label='Lulus Tepat Waktu', markeredgewidth=2, 
         markeredgecolor='white')

# Garis 2: Terlambat (merah)
plt.plot(semesters, terlambat_avg, marker='s', linewidth=2.5, markersize=10, 
         color='#e74c3c', label='Terlambat', markeredgewidth=2, 
         markeredgecolor='white')

plt.title('Tren Rata-rata IPS Per Semester', fontsize=16, fontweight='bold', pad=15)
plt.xlabel('Semester', fontsize=12, fontweight='bold')
plt.ylabel('Rata-rata IPS', fontsize=12, fontweight='bold')
plt.xticks(semesters)
plt.legend(fontsize=12, loc='best', framealpha=0.9, shadow=True)
plt.grid(True, alpha=0.3, linestyle='--')
plt.ylim(0, 4.0)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, '03_tren_ips_semester.png'), dpi=300, bbox_inches='tight')
print(f" Grafik tersimpan: Tren_ips_semester.png")
plt.close()

# ========================================
# BOXPLOT: PERBANDINGAN IPS PER SEMESTER 
# ========================================
print("\n PERBANDINGAN IPS: LULUS vs TERLAMBAT")
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle('Perbandingan IPS: Lulus Tepat Waktu vs Terlambat', fontsize=16, fontweight='bold')

for idx, col in enumerate(ips_cols):
    ax = axes[idx // 4, idx % 4]
    df.boxplot(column=col, by='STATUS_KELULUSAN', ax=ax, patch_artist=True,
               boxprops=dict(facecolor='lightblue', color='blue'),
               medianprops=dict(color='red', linewidth=2))
    ax.set_title(f'Semester {idx + 1}', fontsize=12, fontweight='bold')
    ax.set_xlabel('')
    ax.set_ylabel('IPS', fontsize=10)
    ax.get_figure().suptitle('')  # Hapus title otomatis
    
fig.suptitle('Perbandingan IPS: Lulus Tepat Waktu vs Terlambat', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, '04_boxplot_ips_comparison.png'), dpi=300, bbox_inches='tight')
print(f" Grafik tersimpan: Boxplot_ips_comparison.png")
plt.close()


# ========================================
# DISTRIBUSI GENDER BERDASARKAN STATUS KELULUSAN
# ========================================
print("\n DISTRIBUSI GENDER BERDASARKAN STATUS KELULUSAN")

# Pisahkan data lulus dan terlambat
lulus_df = df[df['target_encoded'] == 1]
terlambat_df = df[df['target_encoded'] == 0]

# Hitung gender untuk masing-masing status
lulus_gender = lulus_df['JENIS_KELAMIN'].value_counts()
terlambat_gender = terlambat_df['JENIS_KELAMIN'].value_counts()

print("\nLulus Tepat Waktu:")
print(lulus_gender)
print("\nTerlambat:")
print(terlambat_gender)

# Buat 2 pie chart side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle('Distribusi Gender Berdasarkan Status Kelulusan', fontsize=18, fontweight='bold', y=1.02)

colors_gender = ['#3498db', '#e91e63']

def make_autopct_gender(values):
    def my_autopct(pct):
        total = sum(values)
        val = int(round(pct*total/100.0))
        return f'{pct:.1f}%\n({val} mahasiswa)'
    return my_autopct

# Pie 1: Lulus Tepat Waktu
wedges1, texts1, autotexts1 = ax1.pie(lulus_gender, labels=lulus_gender.index, 
                                        autopct=make_autopct_gender(lulus_gender),
                                        startangle=90, colors=colors_gender, 
                                        explode=(0.05, 0), shadow=True,
                                        textprops={'fontsize': 11, 'fontweight': 'bold'})
ax1.set_title('Lulus Tepat Waktu', fontsize=14, fontweight='bold', pad=15, 
              bbox=dict(boxstyle='round', facecolor='#2ecc71', alpha=0.3))

# Pie 2: Terlambat
wedges2, texts2, autotexts2 = ax2.pie(terlambat_gender, labels=terlambat_gender.index,
                                        autopct=make_autopct_gender(terlambat_gender),
                                        startangle=90, colors=colors_gender,
                                        explode=(0.05, 0), shadow=True,
                                        textprops={'fontsize': 11, 'fontweight': 'bold'})
ax2.set_title('Terlambat', fontsize=14, fontweight='bold', pad=15,
              bbox=dict(boxstyle='round', facecolor='#e74c3c', alpha=0.3))

plt.tight_layout()
plt.savefig(os.path.join(output_dir, '05_distribusi_gender_per_status.png'), dpi=300, bbox_inches='tight')
print(f" Grafik tersimpan: Distribusi_gender_per_status.png")
plt.close()


# ========================================
# DIAGRAM BATANG: STATUS PEKERJAAN BERDASARKAN KELULUSAN + IPK
# ========================================
print("\n STATUS PEKERJAAN BERDASARKAN KELULUSAN & RATA-RATA IPK")

# Group by status kelulusan dan status mahasiswa (bekerja/tidak)
pekerjaan_kelulusan = df.groupby(['STATUS_KELULUSAN', 'STATUS_MAHASISWA']).agg({
    'IPK': 'mean',
    'NAMA': 'count'
}).rename(columns={'NAMA': 'Jumlah'}).reset_index()

print("\nData Status Pekerjaan:")
print(pekerjaan_kelulusan)

# Siapkan data untuk plotting
lulus_data = pekerjaan_kelulusan[pekerjaan_kelulusan['STATUS_KELULUSAN'].str.contains('TEPAT', na=False)]
terlambat_data = pekerjaan_kelulusan[pekerjaan_kelulusan['STATUS_KELULUSAN'].str.contains('TERLAMBAT|TELAT', na=False)]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle('Status Pekerjaan & Rata-rata IPK Berdasarkan Kelulusan', 
             fontsize=18, fontweight='bold', y=1.02)

# === GRAFIK KIRI: Lulus Tepat Waktu ===
if len(lulus_data) > 0:
    categories_lulus = lulus_data['STATUS_MAHASISWA'].tolist()
    jumlah_lulus = lulus_data['Jumlah'].tolist()
    ipk_lulus = lulus_data['IPK'].tolist()
    
    colors_lulus = ['#27ae60', '#2ecc71']
    bars1 = ax1.bar(categories_lulus, jumlah_lulus, color=colors_lulus, 
                    edgecolor='black', linewidth=2, alpha=0.8)
    
    # Tambahkan nilai di atas bar dengan detail lengkap
    for i, (bar, jumlah, ipk) in enumerate(zip(bars1, jumlah_lulus, ipk_lulus)):
        height = bar.get_height()
        # Text utama: jumlah mahasiswa
        ax1.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'{int(jumlah)} mahasiswa',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
        # Text IPK di tengah bar
        ax1.text(bar.get_x() + bar.get_width()/2., height/2,
                f'IPK: {ipk:.2f}',
                ha='center', va='center', fontsize=10, fontweight='bold',
                color='white', bbox=dict(boxstyle='round', facecolor='black', alpha=0.6))
    
    ax1.set_title('Lulus Tepat Waktu', fontsize=14, fontweight='bold', pad=15,
                  bbox=dict(boxstyle='round', facecolor='#2ecc71', alpha=0.3))
    ax1.set_ylabel('Jumlah Mahasiswa', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Status Pekerjaan', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax1.set_ylim(0, max(jumlah_lulus) * 1.2)

# === GRAFIK KANAN: Terlambat ===
if len(terlambat_data) > 0:
    categories_terlambat = terlambat_data['STATUS_MAHASISWA'].tolist()
    jumlah_terlambat = terlambat_data['Jumlah'].tolist()
    ipk_terlambat = terlambat_data['IPK'].tolist()
    
    colors_terlambat = ['#c0392b', '#e74c3c']
    bars2 = ax2.bar(categories_terlambat, jumlah_terlambat, color=colors_terlambat,
                    edgecolor='black', linewidth=2, alpha=0.8)
    
    # Tambahkan nilai di atas bar dengan detail lengkap
    for i, (bar, jumlah, ipk) in enumerate(zip(bars2, jumlah_terlambat, ipk_terlambat)):
        height = bar.get_height()
        # Text utama: jumlah mahasiswa
        ax2.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'{int(jumlah)} mahasiswa',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
        # Text IPK di tengah bar
        ax2.text(bar.get_x() + bar.get_width()/2., height/2,
                f'IPK: {ipk:.2f}',
                ha='center', va='center', fontsize=10, fontweight='bold',
                color='white', bbox=dict(boxstyle='round', facecolor='black', alpha=0.6))
    
    ax2.set_title('Terlambat', fontsize=14, fontweight='bold', pad=15,
                  bbox=dict(boxstyle='round', facecolor='#e74c3c', alpha=0.3))
    ax2.set_ylabel('Jumlah Mahasiswa', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Status Pekerjaan', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax2.set_ylim(0, max(jumlah_terlambat) * 1.2)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, '06_status_pekerjaan_ipk.png'), dpi=300, bbox_inches='tight')
print(f" Grafik tersimpan: status_pekerjaan_ipk.png")
plt.close()
