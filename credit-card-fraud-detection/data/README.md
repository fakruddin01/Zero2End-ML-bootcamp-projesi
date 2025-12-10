# Dataset Kurulum Talimatları

## Kaggle Dataset İndirme

Bu proje için Kaggle'dan Credit Card Fraud Detection dataset'ini indirmeniz gerekmektedir.

### Adım 1: Kaggle Hesabı

1. [Kaggle.com](https://www.kaggle.com) adresinden hesap oluşturun (ücretsiz)
2. Hesabınıza giriş yapın

### Adım 2: Dataset Sayfası

1. [Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud) sayfasına gidin
2. "Download" butonuna tıklayın
3. `creditcard.csv` dosyasını indirin

### Adım 3: Dosyayı Yerleştirme

İndirilen `creditcard.csv` dosyasını proje klasörünüzdeki `data/` dizinine kopyalayın:

```
credit-card-fraud-detection/
└── data/
    └── creditcard.csv  <-- Buraya kopyalayın
```

### Alternatif: Kaggle API ile İndirme

Kaggle API kullanarak otomatik indirebilirsiniz:

#### 1. Kaggle API Kurulumu
```bash
pip install kaggle
```

#### 2. API Token Alma
1. Kaggle hesabınızda: Account → API → Create New API Token
2. `kaggle.json` dosyası indirilecek
3. Dosyayı şu konuma taşıyın:
   - Windows: `C:\Users\<username>\.kaggle\kaggle.json`
   - Linux/Mac: `~/.kaggle/kaggle.json`

#### 3. Dataset İndirme
```bash
cd credit-card-fraud-detection/data
kaggle datasets download -d mlg-ulb/creditcardfraud
unzip creditcardfraud.zip
```

### Dataset Bilgileri

- **Dosya Adı**: creditcard.csv
- **Boyut**: ~150 MB
- **Satır Sayısı**: 284,807
- **Sütun Sayısı**: 31 (Time, V1-V28, Amount, Class)

### Doğrulama

Dataset'in doğru yüklendiğini kontrol etmek için:

```python
import pandas as pd

df = pd.read_csv('data/creditcard.csv')
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
```

Beklenen çıktı:
```
Shape: (284807, 31)
Columns: ['Time', 'V1', 'V2', ..., 'V28', 'Amount', 'Class']
```

## Sorun Giderme

### Dosya Bulunamadı Hatası
- Dataset'in `data/` klasöründe olduğundan emin olun
- Dosya adının tam olarak `creditcard.csv` olduğunu kontrol edin

### İndirme Sorunu
- İnternet bağlantınızı kontrol edin
- Kaggle hesabınıza giriş yaptığınızdan emin olun
- Tarayıcınızı değiştirmeyi deneyin

### API Hatası
- `kaggle.json` dosyasının doğru konumda olduğundan emin olun
- API token'ınızın geçerli olduğunu kontrol edin

## Veri Gizliliği

⚠️ **Önemli**: Bu dataset halka açık ve anonim hale getirilmiştir. PCA dönüşümü uygulanmış olduğu için orijinal feature isimleri gizlenmiştir.
