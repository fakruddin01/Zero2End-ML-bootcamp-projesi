# -*- coding: utf-8 -*-
# Dataset İndirme Scripti
# Bu script Kaggle API kullanarak dataset'i otomatik indirir

import os
import subprocess
import sys

def check_kaggle_api():
    """Kaggle API'nin kurulu olup olmadığını kontrol et"""
    try:
        import kaggle
        print("[OK] Kaggle API kurulu")
        return True
    except ImportError:
        print("[ERROR] Kaggle API kurulu degil")
        return False

def install_kaggle():
    """Kaggle API'yi kur"""
    print("Kaggle API kuruluyor...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "kaggle"])
    print("[OK] Kaggle API kuruldu")

def check_kaggle_credentials():
    """Kaggle credentials'ın olup olmadığını kontrol et"""
    kaggle_dir = os.path.expanduser("~/.kaggle")
    kaggle_json = os.path.join(kaggle_dir, "kaggle.json")
    
    if os.path.exists(kaggle_json):
        print(f"[OK] Kaggle credentials bulundu: {kaggle_json}")
        return True
    else:
        print(f"[ERROR] Kaggle credentials bulunamadi: {kaggle_json}")
        print("\nKaggle API token almak için:")
        print("1. https://www.kaggle.com/account adresine gidin")
        print("2. 'Create New API Token' butonuna tıklayın")
        print("3. İndirilen kaggle.json dosyasını şu konuma kopyalayın:")
        print(f"   {kaggle_json}")
        return False

def download_dataset():
    """Dataset'i Kaggle'dan indir"""
    print("\nDataset indiriliyor...")
    
    # Data klasörünü oluştur
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    # Dataset'i indir
    try:
        subprocess.check_call([
            "kaggle", "datasets", "download",
            "-d", "mlg-ulb/creditcardfraud",
            "-p", data_dir,
            "--unzip"
        ])
        print("[OK] Dataset basariyla indirildi!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Dataset indirme hatasi: {e}")
        return False

def verify_dataset():
    """Dataset'in doğru indirildiğini kontrol et"""
    csv_file = "data/creditcard.csv"
    
    if os.path.exists(csv_file):
        # Dosya boyutunu kontrol et
        size_mb = os.path.getsize(csv_file) / (1024 * 1024)
        print(f"\n[OK] Dataset dogrulandi!")
        print(f"   Dosya: {csv_file}")
        print(f"   Boyut: {size_mb:.2f} MB")
        
        # İlk birkaç satırı oku
        try:
            import pandas as pd
            df = pd.read_csv(csv_file, nrows=5)
            print(f"   Satır sayısı (ilk kontrol): {len(df)}")
            print(f"   Sütunlar: {list(df.columns)}")
            return True
        except Exception as e:
            print(f"[WARNING] Dataset okunurken hata: {e}")
            return False
    else:
        print(f"[ERROR] Dataset dosyasi bulunamadi: {csv_file}")
        return False

def main():
    print("=" * 60)
    print("CREDIT CARD FRAUD DETECTION - DATASET KURULUMU")
    print("=" * 60)
    
    # 1. Kaggle API kontrolü
    if not check_kaggle_api():
        install_kaggle()
    
    # 2. Credentials kontrolü
    if not check_kaggle_credentials():
        print("\n[WARNING] Lutfen once Kaggle API token'inizi ayarlayin.")
        print("Detaylı talimatlar için data/README.md dosyasına bakın.")
        return
    
    # 3. Dataset'i indir
    if download_dataset():
        # 4. Dataset'i doğrula
        if verify_dataset():
            print("\n" + "=" * 60)
            print("[SUCCESS] KURULUM TAMAMLANDI!")
            print("=" * 60)
            print("\nŞimdi notebook'ları çalıştırabilirsiniz:")
            print("  jupyter notebook")
        else:
            print("\n[WARNING] Dataset dogrulamasi basarisiz oldu.")
    else:
        print("\n[ERROR] Dataset indirilemedi.")
        print("Manuel indirme için data/README.md dosyasına bakın.")

if __name__ == "__main__":
    main()
