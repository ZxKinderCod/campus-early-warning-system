#control utama untuk menjalankan seluruh pipeline Campus Early Warning System
print(" MENJALANKAN CAMPUS EARLY WARNING SYSTEM\n")

import cleaning
import eda
import models
import evaluasi_model
import predict
from send_alert_telegram import send_alert_telegram
send_alert_telegram()

print("\n SEMUA PROSES SELESAI")
