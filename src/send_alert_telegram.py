import requests
import pandas as pd

# ========================================
# KONFIGURASI TELEGRAM
# ========================================

BOT_TOKEN = "8557343292:AAG3c8qplC-nsL2izwo5Ro4UE4BRsL3797U"  # ← Dari @BotFather
CHAT_ID = "5751871840"  # ← Chat ID Anda 

ALERT_FILE = r"C:\coding cihuy\pygwalker\results\models\alert_list.csv"
TOP_N_STUDENTS = 10

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
# KIRIM ALERT MAHASISWA
# ========================================

def send_alert_telegram():
    """Kirim daftar mahasiswa berisiko via Telegram"""
    
    print("="*70)
    print(" TELEGRAM ALERT SYSTEM - campusAlert_bot")
    print("="*70)
    
    # Load data
    try:
        df_alerts = pd.read_csv(ALERT_FILE)
        print(f"\n Data loaded: {len(df_alerts)} mahasiswa")
    except FileNotFoundError:
        print(f"\n File tidak ditemukan: {ALERT_FILE}")
        print("   Jalankan evaluasi_model.py terlebih dahulu!")
        return
    
    # Filter mahasiswa berisiko
    students_with_alerts = df_alerts[df_alerts['Alert_Count'] >= 1].copy()
    top_students = students_with_alerts.sort_values('Risk_Score', ascending=False).head(TOP_N_STUDENTS)
    
    if len(top_students) == 0:
        print("\n Tidak ada mahasiswa berisiko!")
        message = " *CAMPUS EARLY WARNING SYSTEM*\n\nTidak ada mahasiswa yang memerlukan alert saat ini.\nSemua mahasiswa dalam kondisi baik!"
        send_telegram_message(message)
        return
    
    # BUILD MESSAGE
    message = f""" *CAMPUS EARLY WARNING SYSTEM*
━━━━━━━━━━━━━━━━━━━━━━━━━━

 *DAFTAR MAHASISWA BERISIKO*
Total: {len(top_students)} mahasiswa

"""
    
    # Tambahkan setiap mahasiswa
    for idx, (_, student) in enumerate(top_students.iterrows(), 1):
        message += f"""{idx}. *{student['NAMA']}*
     Risk Score: {student['Risk_Score']:.0f}/100
     Rata-rata IPS: {student['Avg_IPS']:.2f}
   
"""
    
    message += """━━━━━━━━━━━━━━━━━━━━━━━━━━
  *Tindakan yang disarankan:*
Segera panggil mahasiswa di atas untuk konseling akademik.

_Dikirim otomatis oleh campusAlert\\_bot_"""
    
    # Preview
    print("\n" + "="*70)
    print(" PREVIEW PESAN:")
    print("="*70)
    print(message.replace('*', '').replace('_', ''))  # Remove markdown untuk preview
    print("="*70)
    
    print(f"\n DAFTAR {len(top_students)} MAHASISWA:")
    for idx, (_, student) in enumerate(top_students.iterrows(), 1):
        print(f"   {idx}. {student['NAMA']} (Risk: {student['Risk_Score']:.0f})")
    
    print(f"\n Akan dikirim ke Chat ID: {CHAT_ID}")
    print(f" Menggunakan bot: campusAlert_bot")
    
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
                time.sleep(1)  # Delay 1 detik antar pesan
    else:
        send_telegram_message(message)
    
    print("\n SELESAI!")
    print("  Cek Telegram Anda sekarang!")
    print(f"   Pesan dari: CampusAlert_bot")

# ========================================
# JALANKAN
# ========================================

if __name__ == "__main__":
    send_alert_telegram()