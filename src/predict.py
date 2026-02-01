import pandas as pd
import numpy as np
import pickle
import os

# Setup paths
base_path = r"C:\coding cihuy\pygwalker"
model_dir = os.path.join(base_path, "results", "models")

print("="*70)
print("🔮 STUDENT GRADUATION PREDICTION SYSTEM")
print("="*70)

# ========================================
# 1. LOAD MODEL & TOOLS
# ========================================
print("\n1️⃣ LOADING MODEL & PREPROCESSING TOOLS...")

with open(os.path.join(model_dir, 'best_model.pkl'), 'rb') as f:
    model = pickle.load(f)

with open(os.path.join(model_dir, 'scaler.pkl'), 'rb') as f:
    scaler = pickle.load(f)

with open(os.path.join(model_dir, 'label_encoders.pkl'), 'rb') as f:
    encoders = pickle.load(f)

with open(os.path.join(model_dir, 'feature_names.pkl'), 'rb') as f:
    feature_names = pickle.load(f)

print("✅ All models and tools loaded successfully")

# ========================================
# 2. PREDICTION FUNCTION
# ========================================

def predict_student(student_data):
    """
    Prediksi kelulusan mahasiswa
    
    Parameters:
    student_data (dict): Data mahasiswa dengan keys:
        - IPS_1 to IPS_8: float
        - UMUR: int
        - JENIS_KELAMIN: str ('LAKI-LAKI' or 'PEREMPUAN')
        - STATUS_NIKAH: str
        - STATUS_MAHASISWA: str ('BEKERJA' or 'TIDAK BEKERJA')
    
    Returns:
    dict: Hasil prediksi
    """
    
    # Create DataFrame
    df_input = pd.DataFrame([student_data])
    
    # Encode categorical variables
    df_input['JENIS_KELAMIN_ENCODED'] = encoders['gender'].transform([student_data['JENIS_KELAMIN']])[0]
    df_input['STATUS_NIKAH_ENCODED'] = encoders['nikah'].transform([student_data['STATUS_NIKAH']])[0]
    df_input['STATUS_MAHASISWA_ENCODED'] = encoders['mahasiswa'].transform([student_data['STATUS_MAHASISWA']])[0]
    
    # Select features
    X = df_input[feature_names]
    
    # Predict
    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0]
    
    # Result
    result = {
        'prediction': 'LULUS TEPAT WAKTU' if prediction == 1 else 'TERLAMBAT',
        'confidence': max(probability) * 100,
        'probability_lulus': probability[1] * 100,
        'probability_terlambat': probability[0] * 100,
        'risk_level': get_risk_level(probability[1]),
        'recommendation': get_recommendation(student_data, probability[1])
    }
    
    return result

# ========================================
# 3. EARLY WARNING SYSTEM
# ========================================

def get_risk_level(prob_lulus):
    """Determine risk level based on probability"""
    if prob_lulus >= 0.7:
        return "RENDAH ✅"
    elif prob_lulus >= 0.5:
        return "SEDANG ⚠️"
    else:
        return "TINGGI 🚨"

def get_recommendation(student_data, prob_lulus):
    """Generate recommendation based on student data"""
    recommendations = []
    
    # Check IPS trends
    ips_values = [student_data[f'IPS_{i}'] for i in range(1, 9)]
    avg_ips = np.mean(ips_values)
    
    if avg_ips < 2.5:
        recommendations.append("⚠️ IPS rata-rata rendah! Tingkatkan performa akademik")
    elif avg_ips < 3.0:
        recommendations.append("📚 IPS cukup baik, pertahankan dan tingkatkan lagi")
    else:
        recommendations.append("✅ IPS sangat baik! Pertahankan prestasi")
    
    # Check IPS trend
    if len(ips_values) >= 4:
        recent_trend = np.mean(ips_values[-4:]) - np.mean(ips_values[:4])
        if recent_trend < -0.2:
            recommendations.append("📉 Performa menurun! Segera konsultasi dengan pembimbing akademik")
        elif recent_trend > 0.2:
            recommendations.append("📈 Performa meningkat! Terus pertahankan")
    
    # Risk-based recommendations
    if prob_lulus < 0.5:
        recommendations.append("🚨 RISIKO TINGGI! Butuh bimbingan intensif dan action plan")
        recommendations.append("💡 Pertimbangkan mengikuti remedial atau kursus tambahan")
    elif prob_lulus < 0.7:
        recommendations.append("⚠️ Perlu peningkatan! Fokus pada mata kuliah dengan IPS rendah")
    
    # Work status recommendation
    if student_data['STATUS_MAHASISWA'] == 'BEKERJA' and prob_lulus < 0.6:
        recommendations.append("⏰ Pertimbangkan untuk mengurangi jam kerja dan fokus pada kuliah")
    
    return recommendations

# ========================================
# 4. BATCH PREDICTION
# ========================================

def predict_batch(csv_file_path):
    """
    Prediksi untuk banyak mahasiswa sekaligus
    
    Parameters:
    csv_file_path (str): Path to CSV file
    
    Returns:
    DataFrame: Hasil prediksi untuk semua mahasiswa
    """
    df = pd.read_csv(csv_file_path)
    
    results = []
    for idx, row in df.iterrows():
        student_data = row.to_dict()
        pred = predict_student(student_data)
        
        results.append({
            'NAMA': row.get('NAMA', f'Student_{idx+1}'),
            'Prediksi': pred['prediction'],
            'Confidence': f"{pred['confidence']:.2f}%",
            'Risk_Level': pred['risk_level'],
            'Prob_Lulus': f"{pred['probability_lulus']:.2f}%",
            'Prob_Terlambat': f"{pred['probability_terlambat']:.2f}%"
        })
    
    return pd.DataFrame(results)

# ========================================
# 5. EXAMPLE USAGE
# ========================================

print("\n" + "="*70)
print("📝 CONTOH PREDIKSI MAHASISWA BARU")
print("="*70)

# Example student 1: Good performance
mahasiswa_1 = {
    'IPS_1': 3.5,
    'IPS_2': 3.6,
    'IPS_3': 3.7,
    'IPS_4': 3.8,
    'IPS_5': 3.7,
    'IPS_6': 3.6,
    'IPS_7': 3.5,
    'IPS_8': 3.4,
    'UMUR': 20,
    'JENIS_KELAMIN': 'LAKI-LAKI',
    'STATUS_NIKAH': 'BELUM MENIKAH',
    'STATUS_MAHASISWA': 'TIDAK BEKERJA'
}

print("\n👨‍🎓 MAHASISWA 1 (Performa Baik):")
print("-" * 70)
result_1 = predict_student(mahasiswa_1)
print(f"🎯 Prediksi: {result_1['prediction']}")
print(f"📊 Confidence: {result_1['confidence']:.2f}%")
print(f"   • Probabilitas Lulus: {result_1['probability_lulus']:.2f}%")
print(f"   • Probabilitas Terlambat: {result_1['probability_terlambat']:.2f}%")
print(f"⚠️ Risk Level: {result_1['risk_level']}")
print(f"\n💡 Recommendations:")
for rec in result_1['recommendation']:
    print(f"   {rec}")

# Example student 2: At risk
mahasiswa_2 = {
    'IPS_1': 2.5,
    'IPS_2': 2.3,
    'IPS_3': 2.4,
    'IPS_4': 2.2,
    'IPS_5': 2.1,
    'IPS_6': 2.0,
    'IPS_7': 1.9,
    'IPS_8': 1.8,
    'UMUR': 22,
    'JENIS_KELAMIN': 'PEREMPUAN',
    'STATUS_NIKAH': 'BELUM MENIKAH',
    'STATUS_MAHASISWA': 'BEKERJA'
}

print("\n👩‍🎓 MAHASISWA 2 (Berisiko):")
print("-" * 70)
result_2 = predict_student(mahasiswa_2)
print(f"🎯 Prediksi: {result_2['prediction']}")
print(f"📊 Confidence: {result_2['confidence']:.2f}%")
print(f"   • Probabilitas Lulus: {result_2['probability_lulus']:.2f}%")
print(f"   • Probabilitas Terlambat: {result_2['probability_terlambat']:.2f}%")
print(f"⚠️ Risk Level: {result_2['risk_level']}")
print(f"\n💡 Recommendations:")
for rec in result_2['recommendation']:
    print(f"   {rec}")

print("\n" + "="*70)
print("✅ PREDICTION SYSTEM READY!")
print("="*70)
print("\n📌 How to use:")
print("   1. Single prediction: Call predict_student(student_data)")
print("   2. Batch prediction: Call predict_batch('path/to/file.csv')")
print("\n💡 Features:")
print("   ✅ Prediction with confidence score")
print("   ✅ Risk level assessment")
print("   ✅ Personalized recommendations")
print("   ✅ Early warning detection")
print("   ✅ Batch processing capability")