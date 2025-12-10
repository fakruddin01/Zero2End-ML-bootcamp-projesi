# 💳 Credit Card Fraud Detection

> **Zero2End Machine Learning Bootcamp - Uçtan Uca ML Projesi**

Kredi kartı işlemlerinde dolandırıcılık tespiti için geliştirilmiş end-to-end machine learning projesi.

## 📋 Proje Hakkında

Bu proje, bankacılık sektöründe kredi kartı dolandırıcılığı tespiti problemine makine öğrenmesi çözümü sunmaktadır. Proje, veri keşfinden model deployment'a kadar tüm ML pipeline aşamalarını içermektedir.

### 🎯 Hedef
Kredi kartı işlemlerini analiz ederek dolandırıcılık vakalarını yüksek doğrulukla tespit etmek.

### 📊 Dataset
- **Kaynak**: Kaggle - Credit Card Fraud Detection
- **Boyut**: 284,807 işlem
- **Özellikler**: 28 PCA dönüştürülmüş feature + Time + Amount
- **Hedef Değişken**: Class (0: Normal, 1: Fraud)
- **Challenge**: Yüksek class imbalance (%0.172 fraud)

## 🗂️ Proje Yapısı

```
credit-card-fraud-detection/
├── data/                          # Dataset dosyaları
├── notebooks/                     # Jupyter notebooks
│   ├── 01_EDA.ipynb              # Keşifsel Veri Analizi
│   ├── 02_baseline.ipynb         # Baseline Model
│   ├── 03_feature_engineering.ipynb  # Feature Engineering
│   ├── 04_model_optimization.ipynb   # Model Optimizasyonu
│   ├── 05_model_evaluation.ipynb     # Model Değerlendirme
│   └── 06_final_pipeline.ipynb       # Final Pipeline
├── src/                           # Kaynak kodlar
│   ├── config.py                 # Konfigürasyon
│   ├── inference.py              # Tahmin fonksiyonları
│   ├── app.py                    # Streamlit uygulaması
│   └── pipeline.py               # ML Pipeline
├── models/                        # Eğitilmiş modeller
├── docs/                          # Dokümantasyon
├── tests/                         # Test dosyaları
├── requirements.txt               # Python bağımlılıkları
└── README.md                      # Bu dosya
```

## 🚀 Kurulum

### 1. Repository'yi klonlayın
```bash
git clone <repository-url>
cd credit-card-fraud-detection
```

### 2. Virtual environment oluşturun
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3. Bağımlılıkları yükleyin
```bash
pip install -r requirements.txt
```

### 4. Dataset'i indirin
Kaggle'dan [Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud) dataset'ini indirip `data/` klasörüne yerleştirin.

## 📓 Notebook'ları Çalıştırma

Notebook'lar sırasıyla çalıştırılmalıdır:

```bash
jupyter notebook
```

1. **01_EDA.ipynb**: Veri keşfi ve görselleştirme
2. **02_baseline.ipynb**: Baseline model oluşturma
3. **03_feature_engineering.ipynb**: Feature engineering
4. **04_model_optimization.ipynb**: Model optimizasyonu
5. **05_model_evaluation.ipynb**: Model değerlendirme
6. **06_final_pipeline.ipynb**: Final pipeline ve model kaydetme

## 🌐 Deployment

### Streamlit Uygulaması

```bash
streamlit run src/app.py
```

Uygulama `http://localhost:8501` adresinde çalışacaktır.

### FastAPI (REST API)

```bash
uvicorn src.app:app --reload
```

API dokümantasyonu: `http://localhost:8000/docs`

## 🧪 Test

```bash
pytest tests/ -v
```

## 📈 Kullanılan Teknolojiler

- **Python 3.10+**
- **Pandas & NumPy**: Veri manipülasyonu
- **Scikit-learn**: ML modelleri ve preprocessing
- **XGBoost & LightGBM**: Gradient boosting modelleri
- **Imbalanced-learn**: SMOTE ve class balancing
- **Matplotlib & Seaborn**: Veri görselleştirme
- **Streamlit**: Web uygulaması
- **FastAPI**: REST API

## 🎯 Model Performansı

| Metric | Score |
|--------|-------|
| Accuracy | TBD |
| Precision | TBD |
| Recall | TBD |
| F1-Score | TBD |
| ROC-AUC | TBD |

*Not: Değerler model eğitimi tamamlandıktan sonra güncellenecektir.*

## 📝 Proje Aşamaları

- [x] Proje yapısı oluşturma
- [ ] EDA ve veri keşfi
- [ ] Baseline model
- [ ] Feature engineering
- [ ] Model optimizasyonu
- [ ] Model değerlendirme
- [ ] Pipeline oluşturma
- [ ] Deployment
- [ ] Dokümantasyon

## 👥 Katkıda Bulunma

1. Fork edin
2. Feature branch oluşturun (`git checkout -b feature/amazing-feature`)
3. Commit edin (`git commit -m 'Add amazing feature'`)
4. Push edin (`git push origin feature/amazing-feature`)
5. Pull Request açın

## 📄 Lisans

Bu proje eğitim amaçlıdır.

## 📧 İletişim

Proje Sahibi - Zero2End ML Bootcamp

---

**MultiGroup - Zero2End Machine Learning Bootcamp**
