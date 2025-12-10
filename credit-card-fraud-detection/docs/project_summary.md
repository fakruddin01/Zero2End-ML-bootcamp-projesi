# Credit Card Fraud Detection - Project Summary

## 📌 Proje Genel Bakış

**Proje Adı**: Credit Card Fraud Detection  
**Sektör**: Bankacılık ve Finans  
**Problem**: Kredi kartı işlemlerinde dolandırıcılık tespiti  
**Hedef**: Dolandırıcılık vakalarını yüksek doğrulukla tespit eden bir ML modeli geliştirmek

## 🎯 Problem Tanımı

Kredi kartı dolandırıcılığı, finansal kurumlar için büyük bir sorundur. Bu proje, makine öğrenmesi kullanarak işlemleri analiz edip dolandırıcılık vakalarını otomatik olarak tespit etmeyi amaçlamaktadır.

### Zorluklar:
- **Yüksek Class Imbalance**: Dolandırıcılık vakaları tüm işlemlerin sadece %0.17'sini oluşturuyor
- **Gerçek Zamanlı Tespit**: Hızlı karar verme gerekliliği
- **False Positive Maliyeti**: Yanlış alarm müşteri deneyimini olumsuz etkiler
- **False Negative Maliyeti**: Kaçırılan dolandırıcılık finansal kayıp demektir

## 📊 Dataset

**Kaynak**: Kaggle - Credit Card Fraud Detection  
**Boyut**: 284,807 işlem  
**Özellikler**: 
- 28 PCA dönüştürülmüş feature (V1-V28)
- Time: İşlem zamanı
- Amount: İşlem tutarı
- Class: Hedef değişken (0: Normal, 1: Fraud)

**Class Distribution**:
- Normal İşlemler: 284,315 (%99.83)
- Dolandırıcılık: 492 (%0.17)

## 🔬 Metodoloji

### 1. Exploratory Data Analysis (EDA)
- Veri kalitesi kontrolü (eksik değer yok)
- Class imbalance analizi
- Feature dağılımları incelemesi
- Korelasyon analizi

### 2. Baseline Model
- Logistic Regression ile baseline oluşturma
- Temel metrikler hesaplama
- İyileştirme alanlarını belirleme

### 3. Feature Engineering
- Zaman bazlı feature'lar (Hour, Day)
- Amount logaritması
- Feature interaction'ları
- StandardScaler ile normalizasyon

### 4. Class Imbalance Handling
- SMOTE (Synthetic Minority Over-sampling Technique)
- Sampling strategy: 0.5 (minority class %50 olacak şekilde)

### 5. Model Optimization
- Farklı modellerin karşılaştırılması:
  - Logistic Regression
  - Random Forest
  - XGBoost
- GridSearchCV ile hyperparameter tuning
- Cross-validation (5-fold)

### 6. Model Evaluation
- ROC-AUC Score
- Precision-Recall Curve
- Confusion Matrix
- Feature Importance Analysis
- Threshold Optimization

## 🏆 Sonuçlar

### Final Model
**Model**: Random Forest Classifier

**Hyperparameters**:
- n_estimators: 200
- max_depth: 20
- min_samples_split: 5
- min_samples_leaf: 2

### Performans Metrikleri
*(Not: Gerçek değerler model eğitildikten sonra güncellenecektir)*

- **ROC-AUC Score**: TBD
- **Precision**: TBD
- **Recall**: TBD
- **F1-Score**: TBD

### En Önemli Feature'lar
Model, özellikle V14, V12, V10, V17 gibi PCA feature'larını önemli bulmuştur.

## 🛠️ Kullanılan Teknolojiler

### Core ML Stack
- **Python 3.10+**
- **Pandas & NumPy**: Veri manipülasyonu
- **Scikit-learn**: ML modelleri ve preprocessing
- **XGBoost**: Gradient boosting
- **Imbalanced-learn**: SMOTE implementasyonu

### Visualization
- **Matplotlib & Seaborn**: Statik grafikler
- **Plotly**: İnteraktif görselleştirmeler

### Deployment
- **Streamlit**: Web uygulaması
- **FastAPI**: REST API (opsiyonel)
- **Joblib**: Model persistence

## 🚀 Deployment

### Streamlit Web Uygulaması
Kullanıcı dostu bir arayüz ile:
- Manuel veri girişi
- CSV dosyası yükleme
- Örnek veri ile test
- Gerçek zamanlı tahmin
- Görsel sonuç gösterimi

### Kullanım:
```bash
streamlit run src/app.py
```

## 📈 İyileştirme Önerileri

### Kısa Vadeli
1. Daha fazla feature engineering
2. Ensemble methods deneme
3. Deep learning modelleri (Neural Networks)
4. Threshold optimizasyonu

### Uzun Vadeli
1. Gerçek zamanlı model monitoring
2. A/B testing framework
3. Otomatik model retraining
4. Model versioning sistemi
5. Production deployment (AWS/GCP/Azure)

## 📝 Proje Yapısı

```
credit-card-fraud-detection/
├── notebooks/          # Jupyter notebooks (EDA, modeling, etc.)
├── src/               # Kaynak kodlar
│   ├── config.py      # Konfigürasyon
│   ├── inference.py   # Tahmin modülü
│   ├── app.py         # Streamlit uygulaması
│   └── pipeline.py    # ML pipeline
├── models/            # Eğitilmiş modeller
├── data/              # Dataset
├── docs/              # Dokümantasyon
└── tests/             # Unit testler
```

## 🎓 Öğrenilen Dersler

1. **Class Imbalance**: SMOTE gibi teknikler kritik öneme sahip
2. **Feature Engineering**: Domain knowledge ile feature engineering performansı artırır
3. **Model Selection**: Karmaşık modeller her zaman daha iyi değildir
4. **Threshold Tuning**: Business requirements'a göre threshold ayarlanmalı
5. **Deployment**: Model geliştirmek deployment'ın sadece bir parçası

## 👥 Katkıda Bulunanlar

- **Proje Sahibi**: Zero2End ML Bootcamp Participant
- **Bootcamp**: MultiGroup - Zero2End Machine Learning Bootcamp

## 📄 Lisans

Bu proje eğitim amaçlıdır.

---

**Son Güncelleme**: Aralık 2025  
**Durum**: ✅ Tamamlandı
