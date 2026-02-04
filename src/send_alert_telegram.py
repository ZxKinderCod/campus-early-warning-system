import requests
import pandas as pd

# ========================================
# KONFIGURASI TELEGRAM
# ========================================

BOT_TOKEN = "8557343292:AAG3c8qplC-nsL2izwo5Ro4UE4BRsL3797U"  # ← Token dari @BotFather
CHAT_ID = "5751871840"             # ← Chat ID Anda

ALERT_FILE = r"C:\coding cihuy\pygwalker\results\models\alert_list.csv"
DATA_FILE = r"C:\coding cihuy\pygwalker\data\datamahasiswa_clean.csv"

# HANYA KIRIM YANG KATEGORI MERAH (Prioritas Tinggi)
# Alert_Count >= 3 = Prioritas Tinggi 

# ========================================
# FUNGSI KIRIM TELEGRAM
# ========================================

def send_telegram_message(message):
    """Kirim pesan ke Telegram"""
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    
    data = {
        'chat_id': CHAT_ID,
        'text': message,
    }
    
    try:
        response = requests.post(url, data=data)
        result = response.json()
        
        if result['ok']:
            print(" Pesan berhasil terkirim!")
            return True
        else:
            print(f" Error: {result.get('description', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f" Error: {e}")
        return False

# ========================================
# KIRIM ALERT MAHASISWA PRIORITAS TINGGI
# ========================================

def send_alert_telegram():
    """Kirim daftar mahasiswa PRIORITAS TINGGI via Telegram"""
    
    print("="*70)
    print(" TELEGRAM ALERT SYSTEM - campusAlert_bot")
    print("="*70)
    
    # Load data alert
    try:
        df_alerts = pd.read_csv(ALERT_FILE)
        print(f"\n Data alert loaded: {len(df_alerts)} mahasiswa")
    except FileNotFoundError:
        print(f"\n File tidak ditemukan: {ALERT_FILE}")
        print("   Jalankan evaluasi_model.py terlebih dahulu!")
        return
    
    # Load data asli untuk ambil IPK
    try:
        df_full = pd.read_csv(DATA_FILE)
        print(f" Data lengkap loaded: {len(df_full)} mahasiswa")
    except FileNotFoundError:
        print(f"\n File tidak ditemukan: {DATA_FILE}")
        return
    
    # FILTER HANYA YANG PRIORITAS TINGGI (MERAH)
    # Berdasarkan kategori dari evaluasi_model.py:
    # - Alert_Count >= 3 = Prioritas Tinggi 
    # - Alert_Count == 2 = Waspada 
    # - Alert_Count == 1 = Perlu Perhatian 
    # - Alert_Count == 0 = Aman 
    
    high_priority = df_alerts[df_alerts['Alert_Count'] >= 3].copy()
    
    # Urutkan berdasarkan Risk Score (tertinggi dulu)
    high_priority = high_priority.sort_values('Risk_Score', ascending=False)
    
    print(f"\n KATEGORI MAHASISWA:")
    print(f"   🔴 Prioritas Tinggi (3+ alert): {len(high_priority)} mahasiswa")
    print(f"   🟠 Waspada (2 alert): {len(df_alerts[df_alerts['Alert_Count'] == 2])} mahasiswa")
    print(f"   🟡 Perlu Perhatian (1 alert): {len(df_alerts[df_alerts['Alert_Count'] == 1])} mahasiswa")
    print(f"   🟢 Aman (0 alert): {len(df_alerts[df_alerts['Alert_Count'] == 0])} mahasiswa")
    
    if len(high_priority) == 0:
        print("\n Tidak ada mahasiswa kategori PRIORITAS TINGGI!")
        print("   Semua mahasiswa dalam kondisi baik atau kategori di bawah merah.")
        
        # Opsional: kirim notifikasi ke Telegram
        message = "*CAMPUS EARLY WARNING SYSTEM*\n\nTidak ada mahasiswa kategori PRIORITAS TINGGI saat ini.\nSemua mahasiswa dalam kondisi terkendali."
        send_telegram_message(message)
        return
    
    # Merge dengan data lengkap untuk dapat IPK
    high_priority_full = high_priority.merge(df_full[['NAMA', 'IPK']], on='NAMA', how='left')
    
    # BUILD MESSAGE
    message = f"""*CAMPUS EARLY WARNING SYSTEM*

*DAFTAR MAHASISWA PRIORITAS TINGGI*
Total: {len(high_priority_full)} mahasiswa

"""
    
    # Tambahkan setiap mahasiswa
    for idx, (_, student) in enumerate(high_priority_full.iterrows(), 1):
        avg_ips = student['Avg_IPS']
        ipk = student['IPK']
        
        message += f"""{idx}. {student['NAMA']}
   Rata-rata IPS: {avg_ips:.2f}
   IPK: {ipk:.2f}

"""
    
    message += """Mahasiswa di atas memerlukan PERHATIAN SEGERA.
Segera panggil untuk konseling akademik intensif.

_Dikirim otomatis oleh campusAlert\\_bot_"""
    
    # Preview
    print("\n" + "="*70)
    print(" PREVIEW PESAN:")
    print("="*70)
    print(message.replace('*', '').replace('_', ''))
    print("="*70)
    
    print(f"\n DAFTAR {len(high_priority_full)} MAHASISWA PRIORITAS TINGGI:")
    for idx, (_, student) in enumerate(high_priority_full.iterrows(), 1):
        print(f"   {idx}. {student['NAMA']}")
        print(f"      IPS: {student['Avg_IPS']:.2f}, IPK: {student['IPK']:.2f}")
        print(f"      Alert: {student['Alert_Details']}")
        print()
    
    print(f"📱 Akan dikirim ke Chat ID: {CHAT_ID}")
    
    # Konfirmasi
    confirm = input("\n  Kirim pesan ke Telegram? (y/n): ")
    
    if confirm.lower() != 'y':
        print("\n Pengiriman dibatalkan.")
        return
    
    # KIRIM
    print("\n Mengirim pesan ke Telegram...")
    
    # Telegram limit: 4096 karakter per pesan
    if len(message) > 4096:
        print("  Pesan terlalu panjang, akan dipecah...")
        
        # Split message
        parts = []
        current_part = ""
        
        for line in message.split('\n'):
            if len(current_part) + len(line) + 1 < 4000:
                current_part += line + '\n'
            else:
                parts.append(current_part)
                current_part = line + '\n'
        
        if current_part:
            parts.append(current_part)
        
        for idx, part in enumerate(parts, 1):
            print(f"   Mengirim bagian {idx}/{len(parts)}...")
            send_telegram_message(part)
            
            if idx < len(parts):
                import time
                time.sleep(1)
    else:
        send_telegram_message(message)
    
    print("\n SELESAI!")
    print(f" {len(high_priority_full)} mahasiswa prioritas tinggi telah dikirim ke Telegram")
    print(" Cek Telegram Anda sekarang!")

# ========================================
# JALANKAN
# ========================================

if __name__ == "__main__":
    send_alert_telegram()