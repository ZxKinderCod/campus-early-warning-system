import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, 
                             roc_curve, roc_auc_score)
import pickle
import os

# Setup paths
base_path = r"C:\coding cihuy\pygwalker"
model_dir = os.path.join(base_path, "results", "models")
viz_dir = os.path.join(base_path, "results", "visualizations")
data_path = os.path.join(base_path, "data", "datamahasiswa_clean.csv")
os.makedirs(viz_dir, exist_ok=True)

print("="*70)
print("📊 MODEL EVALUATION - CAMPUS EARLY WARNING SYSTEM")
print("="*70)

# ========================================
# 1. LOAD MODEL & RESULTS
# ========================================
print("\n1️⃣ LOADING MODELS & DATA...")

with open(os.path.join(model_dir, 'best_model.pkl'), 'rb') as f:
    best_model = pickle.load(f)

with open(os.path.join(model_dir, 'model_results.pkl'), 'rb') as f:
    data = pickle.load(f)
    results = data['results']
    X_test = data['X_test']
    y_test = data['y_test']
    feature_names = data['feature_names']

# Load full dataset for additional analysis
df_full = pd.read_csv(data_path)

print("✅ Models and data loaded successfully")

# ========================================
# 2. FEATURE IMPORTANCE (DIPERBAIKI - TANPA ENCODE)
# ========================================
print("\n2️⃣ CREATING FEATURE IMPORTANCE CHART...")

importances = best_model.feature_importances_

# Map feature names ke nama yang lebih readable
feature_name_mapping = {
    'IPS_1': 'IPS Semester 1',
    'IPS_2': 'IPS Semester 2',
    'IPS_3': 'IPS Semester 3',
    'IPS_4': 'IPS Semester 4',
    'IPS_5': 'IPS Semester 5',
    'IPS_6': 'IPS Semester 6',
    'IPS_7': 'IPS Semester 7',
    'IPS_8': 'IPS Semester 8',
    'UMUR': 'Umur Mahasiswa',
    'JENIS_KELAMIN_ENCODED': 'Jenis Kelamin',
    'STATUS_NIKAH_ENCODED': 'Status Pernikahan',
    'STATUS_MAHASISWA_ENCODED': 'Status Pekerjaan'
}

readable_features = [feature_name_mapping.get(f, f) for f in feature_names]

feature_importance_df = pd.DataFrame({
    'Faktor': readable_features,
    'Tingkat Pengaruh': importances
}).sort_values('Tingkat Pengaruh', ascending=True)

plt.figure(figsize=(12, 8))
colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(feature_importance_df)))
bars = plt.barh(feature_importance_df['Faktor'], feature_importance_df['Tingkat Pengaruh'], 
                color=colors, edgecolor='black', linewidth=1.5)

plt.title('Faktor yang Mempengaruhi Kelulusan Mahasiswa', 
          fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Tingkat Pengaruh (0-1)', fontsize=13, fontweight='bold')
plt.ylabel('Faktor', fontsize=13, fontweight='bold')
plt.grid(True, alpha=0.3, axis='x', linestyle='--')

# Add value labels
for i, (bar, importance) in enumerate(zip(bars, feature_importance_df['Tingkat Pengaruh'])):
    plt.text(importance + 0.005, bar.get_y() + bar.get_height()/2, 
             f'{importance:.3f}',
             va='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(viz_dir, '07_feature_importance.png'), dpi=300, bbox_inches='tight')
print(f"✅ Feature Importance saved: 07_feature_importance.png")
plt.close()

# ========================================
# 3. RISK SCORECARD (PERBAIKAN WARNA)
# ========================================
print("\n3️⃣ CREATING RISK SCORECARD...")

# Encode target untuk full dataset
df_full['target'] = df_full['STATUS_KELULUSAN'].map({
    'LULUS TEPAT WAKTU': 1,
    'TEPAT WAKTU': 1,
    'TEPAT': 1,
    'TERLAMBAT': 0,
    'TELAT': 0,
    'TIDAK TEPAT WAKTU': 0
})

# Prepare features untuk prediksi
from sklearn.preprocessing import LabelEncoder

le_gender = LabelEncoder()
le_nikah = LabelEncoder()
le_mahasiswa = LabelEncoder()

df_full['JENIS_KELAMIN_ENCODED'] = le_gender.fit_transform(df_full['JENIS_KELAMIN'])
df_full['STATUS_NIKAH_ENCODED'] = le_nikah.fit_transform(df_full['STATUS_NIKAH'])
df_full['STATUS_MAHASISWA_ENCODED'] = le_mahasiswa.fit_transform(df_full['STATUS_MAHASISWA'])

X_full = df_full[feature_names]
df_full['Probability_Lulus'] = best_model.predict_proba(X_full)[:, 1]
df_full['Risk_Score'] = (1 - df_full['Probability_Lulus']) * 100  # 0-100 scale

# Calculate risk level (PERBAIKAN URUTAN)
def get_risk_category(score):
    if score >= 70:
        return 'Sangat Tinggi 🔴'  # MERAH
    elif score >= 50:
        return 'Tinggi 🟠'  # ORANGE
    elif score >= 30:
        return 'Sedang 🟡'  # KUNING
    else:
        return 'Rendah 🟢'  # HIJAU

df_full['Risk_Category'] = df_full['Risk_Score'].apply(get_risk_category)

# Visualisasi Risk Distribution
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Risk Scorecard - Distribusi Risiko Mahasiswa', fontsize=18, fontweight='bold')

# Chart 1: Risk Score Distribution
ax1.hist(df_full['Risk_Score'], bins=20, color='#e74c3c', edgecolor='black', alpha=0.7)
ax1.axvline(x=30, color='green', linestyle='--', linewidth=2, label='Batas Risiko Rendah')
ax1.axvline(x=50, color='orange', linestyle='--', linewidth=2, label='Batas Risiko Sedang')
ax1.axvline(x=70, color='red', linestyle='--', linewidth=2, label='Batas Risiko Tinggi')
ax1.set_xlabel('Risk Score (0-100)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Jumlah Mahasiswa', fontsize=12, fontweight='bold')
ax1.set_title('Distribusi Skor Risiko', fontsize=14, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3, axis='y')

# Chart 2: Risk Category Pie (PERBAIKAN WARNA - urutan yang benar)
risk_counts = df_full['Risk_Category'].value_counts()

# Urutan yang benar: Rendah, Sedang, Tinggi, Sangat Tinggi
category_order = ['Rendah 🟢', 'Sedang 🟡', 'Tinggi 🟠', 'Sangat Tinggi 🔴']
risk_counts_ordered = risk_counts.reindex(category_order, fill_value=0)

colors_risk = ['#27ae60', '#f1c40f', '#e67e22', '#e74c3c']  # Hijau, Kuning, Orange, Merah
explode = tuple([0.05] * len(risk_counts_ordered))

def make_autopct_risk(values):
    def my_autopct(pct):
        total = sum(values)
        val = int(round(pct*total/100.0))
        return f'{pct:.1f}%\n({val} mhs)'
    return my_autopct

ax2.pie(risk_counts_ordered, labels=risk_counts_ordered.index, autopct=make_autopct_risk(risk_counts_ordered),
        colors=colors_risk, explode=explode, shadow=True,
        textprops={'fontsize': 11, 'fontweight': 'bold'}, startangle=90)
ax2.set_title('Kategori Risiko Mahasiswa', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(viz_dir, '08_risk_scorecard.png'), dpi=300, bbox_inches='tight')
print(f"✅ Risk Scorecard saved: 08_risk_scorecard.png")
plt.close()

# Save risk scorecard to CSV
risk_df = df_full[['NAMA', 'Risk_Score', 'Risk_Category', 'Probability_Lulus']].sort_values('Risk_Score', ascending=False)
risk_df.to_csv(os.path.join(model_dir, 'risk_scorecard.csv'), index=False)
print(f"✅ Risk scores saved: risk_scorecard.csv")

# ========================================
# 4. AUTOMATED ALERT SYSTEM (PERBAIKAN WARNA & LABEL)
# ========================================
print("\n4️⃣ CREATING AUTOMATED ALERT SYSTEM...")

# Define alert criteria
ips_cols = ['IPS_1', 'IPS_2', 'IPS_3', 'IPS_4', 'IPS_5', 'IPS_6', 'IPS_7', 'IPS_8']
df_full['Avg_IPS'] = df_full[ips_cols].mean(axis=1)

# Alert conditions
alert_details = []
for _, student in df_full.iterrows():
    student_alerts = []
    
    # Alert 1: IPS rata-rata rendah
    if student['Avg_IPS'] < 2.5:
        student_alerts.append('IPS Rendah')
    
    # Alert 2: IPS menurun
    if len(ips_cols) >= 4:
        recent = [student[f'IPS_{i}'] for i in range(5, 9)]
        if np.mean(recent) < 2.3:
            student_alerts.append('Performa Menurun')
    
    # Alert 3: Risk Score tinggi
    if student['Risk_Score'] >= 50:
        student_alerts.append('Risiko Tinggi')
    
    alert_details.append({
        'count': len(student_alerts),
        'details': ', '.join(student_alerts) if student_alerts else 'Tidak Ada Alert'
    })

df_full['Alert_Count'] = [a['count'] for a in alert_details]
df_full['Alert_Details'] = [a['details'] for a in alert_details]

# Kategorisasi berdasarkan alert count (PERBAIKAN LABEL)
def get_alert_category(count):
    if count == 0:
        return 'Aman (0 Alert) 🟢'
    elif count == 1:
        return 'Perlu Perhatian (1 Alert) 🟡'
    elif count == 2:
        return 'Waspada (2 Alert) 🟠'  # ORANGE
    else:
        return 'Prioritas Tinggi (3+ Alert) 🔴'  # MERAH

df_full['Alert_Category'] = df_full['Alert_Count'].apply(get_alert_category)

# Visualisasi
fig = plt.figure(figsize=(18, 7))
gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1.5, 1.3])

ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])
ax3 = fig.add_subplot(gs[2])

fig.suptitle('Automated Alert System - Notifikasi Dini ke Dosen Wali', 
             fontsize=18, fontweight='bold', y=0.98)

# Chart 1: Alert Category Distribution (PERBAIKAN WARNA)
category_counts = df_full['Alert_Category'].value_counts()

# Urutan yang benar
category_order_alert = ['Aman (0 Alert) 🟢', 'Perlu Perhatian (1 Alert) 🟡', 
                        'Waspada (2 Alert) 🟠', 'Prioritas Tinggi (3+ Alert) 🔴']
category_counts_ordered = category_counts.reindex(category_order_alert, fill_value=0)

colors_cat = ['#27ae60', '#f1c40f', '#e67e22', '#e74c3c']  # Hijau, Kuning, Orange, Merah

bars1 = ax1.barh(range(len(category_counts_ordered)), category_counts_ordered.values, 
                 color=colors_cat, edgecolor='black', linewidth=2)

ax1.set_yticks(range(len(category_counts_ordered)))
ax1.set_yticklabels(category_counts_ordered.index, fontsize=10, fontweight='bold')
ax1.set_xlabel('Jumlah Mahasiswa', fontsize=12, fontweight='bold')
ax1.set_title('Kategori Alert Mahasiswa', fontsize=13, fontweight='bold', pad=10)
ax1.grid(True, alpha=0.3, axis='x')

# Add value labels
for i, (bar, count) in enumerate(zip(bars1, category_counts_ordered.values)):
    ax1.text(count + 3, i, f'{count} mhs',
             va='center', fontweight='bold', fontsize=11)

# Chart 2: Alert Distribution Detail (PERBAIKAN WARNA)
alert_distribution = df_full['Alert_Count'].value_counts().sort_index()
colors_alert = ['#27ae60', '#f1c40f', '#e67e22', '#e74c3c']  # Hijau, Kuning, Orange, Merah

bars2 = ax2.bar(alert_distribution.index, alert_distribution.values, 
                color=[colors_alert[min(int(i), 3)] for i in alert_distribution.index],
                edgecolor='black', linewidth=2.5, width=0.6)

# Add labels on bars
for bar, count in zip(bars2, alert_distribution.values):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
             f'{count} mhs',
             ha='center', fontweight='bold', fontsize=12)

ax2.set_xlabel('Jumlah Alert per Mahasiswa', fontsize=12, fontweight='bold')
ax2.set_ylabel('Jumlah Mahasiswa', fontsize=12, fontweight='bold')
ax2.set_title('Distribusi Alert per Mahasiswa', fontsize=13, fontweight='bold', pad=10)
ax2.set_xticks(alert_distribution.index)
ax2.set_xticklabels([f'{int(x)} Alert' for x in alert_distribution.index], fontsize=11)
ax2.grid(True, alpha=0.3, axis='y')

# Add legend untuk penjelasan (PERBAIKAN LABEL)
legend_labels = [
    '🟢 0 Alert = Aman (Hijau)',
    '🟡 1 Alert = Perlu Perhatian (Kuning)',
    '🟠 2 Alert = Waspada (Orange)',
    '🔴 3 Alert = Prioritas Tinggi (Merah)'
]
ax2.text(0.5, 0.97, '\n'.join(legend_labels), 
         transform=ax2.transAxes, fontsize=9,
         verticalalignment='top', horizontalalignment='center',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Chart 3: Top Priority Students
high_priority = df_full[df_full['Alert_Count'] >= 2].sort_values('Risk_Score', ascending=False).head(10)

if len(high_priority) > 0:
    ax3.barh(range(len(high_priority)), high_priority['Risk_Score'], 
             color='#e74c3c', edgecolor='black', linewidth=1.5)
    ax3.set_yticks(range(len(high_priority)))
    ax3.set_yticklabels([name[:25] for name in high_priority['NAMA']], fontsize=9)
    ax3.set_xlabel('Risk Score', fontsize=12, fontweight='bold')
    ax3.set_title('Top 10 Mahasiswa Prioritas Tinggi\n(Perlu Notifikasi ke Doswal)', 
                  fontsize=12, fontweight='bold', pad=10)
    ax3.grid(True, alpha=0.3, axis='x')

    for i, score in enumerate(high_priority['Risk_Score']):
        ax3.text(score + 1, i, f'{score:.0f}',
                 va='center', fontweight='bold', fontsize=9)
else:
    ax3.text(0.5, 0.5, 'Tidak ada mahasiswa\nprioritas tinggi',
             ha='center', va='center', fontsize=12, transform=ax3.transAxes)
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)

plt.tight_layout()
plt.savefig(os.path.join(viz_dir, '09_automated_alert.png'), dpi=300, bbox_inches='tight')
print(f"✅ Automated Alert saved: 09_automated_alert.png")
plt.close()

# Save alert list with details
alert_df = df_full[['NAMA', 'Avg_IPS', 'Risk_Score', 'Alert_Count', 'Alert_Category', 'Alert_Details']].sort_values('Alert_Count', ascending=False)
alert_df.to_csv(os.path.join(model_dir, 'alert_list.csv'), index=False)
print(f"✅ Alert list saved: alert_list.csv")

# Print alert summary
print("\n📊 ALERT SUMMARY:")
print(f"   Total mahasiswa: {len(df_full)}")
for category in category_order_alert:
    count = category_counts_ordered.get(category, 0)
    pct = (count / len(df_full)) * 100
    print(f"   {category}: {count} mahasiswa ({pct:.1f}%)")

# ========================================
# 5. CLASSIFICATION REPORT
# ========================================
print("\n5️⃣ GENERATING CLASSIFICATION REPORT...")

y_pred = results['Best Random Forest']['predictions']
accuracy = results['Best Random Forest']['accuracy']

report = classification_report(y_test, y_pred, 
                               target_names=['Terlambat', 'Lulus Tepat Waktu'],
                               digits=4)
print(report)

# Save report
with open(os.path.join(model_dir, 'classification_report.txt'), 'w') as f:
    f.write("CLASSIFICATION REPORT - CAMPUS EARLY WARNING SYSTEM\n")
    f.write("="*70 + "\n\n")
    f.write(report)
    f.write(f"\nOverall Accuracy: {accuracy*100:.2f}%\n")

print(f"\n✅ Classification report saved: classification_report.txt")

# ========================================
# 6. SUMMARY
# ========================================
print("\n" + "="*70)
print("📈 EVALUATION SUMMARY")
print("="*70)

print(f"\n🎯 MODEL PERFORMANCE:")
print(f"   Accuracy: {accuracy*100:.2f}%")

print(f"\n📊 TOP 3 MOST IMPORTANT FACTORS:")
top_features = feature_importance_df.tail(3)
for idx, row in top_features.iterrows():
    print(f"   {row['Faktor']}: {row['Tingkat Pengaruh']:.3f}")

print(f"\n⚠️ RISK DISTRIBUTION:")
for category, count in risk_counts.items():
    pct = (count / len(df_full)) * 100
    print(f"   {category}: {count} mahasiswa ({pct:.1f}%)")

print(f"\n🚨 AUTOMATED ALERTS:")
alert_students = len(df_full[df_full['Alert_Count'] >= 1])
print(f"   Mahasiswa memerlukan perhatian: {alert_students} ({alert_students/len(df_full)*100:.1f}%)")
high_priority_count = len(df_full[df_full['Alert_Count'] >= 2])
print(f"   Mahasiswa prioritas tinggi: {high_priority_count}")

print("\n✅ EVALUATION COMPLETE!")
print(f"📁 Visualizations saved in: {viz_dir}")
print(f"📁 Reports saved in: {model_dir}")