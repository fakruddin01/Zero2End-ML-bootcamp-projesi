"""
Streamlit web application for Credit Card Fraud Detection
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent))

from config import APP_TITLE, APP_DESCRIPTION, ALL_FEATURES, FEATURES_TO_SCALE
from inference import FraudDetector

# Page configuration
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .fraud-alert {
        background-color: #ff4b4b;
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        font-weight: bold;
        text-align: center;
    }
    .normal-alert {
        background-color: #00cc00;
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        font-weight: bold;
        text-align: center;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


def main():
    """Main application function"""
    
    # Header
    st.markdown(f'<div class="main-header">{APP_TITLE}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sub-header">{APP_DESCRIPTION}</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Ayarlar")
        
        # Model threshold
        threshold = st.slider(
            "Fraud Eşik Değeri",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Dolandırıcılık tespiti için olasılık eşiği"
        )
        
        st.divider()
        
        # Input method
        input_method = st.radio(
            "Veri Girişi Yöntemi",
            ["Manuel Giriş", "CSV Yükleme", "Örnek Veri"]
        )
        
        st.divider()
        
        st.info("""
        **Nasıl Kullanılır?**
        
        1. Veri girişi yöntemini seçin
        2. İşlem bilgilerini girin
        3. 'Tahmin Yap' butonuna tıklayın
        4. Sonuçları inceleyin
        """)
    
    # Main content
    if input_method == "Manuel Giriş":
        show_manual_input(threshold)
    elif input_method == "CSV Yükleme":
        show_csv_upload(threshold)
    else:
        show_sample_data(threshold)


def show_manual_input(threshold: float):
    """Show manual input form"""
    st.header("📝 Manuel Veri Girişi")
    
    col1, col2 = st.columns(2)
    
    with col1:
        time = st.number_input("Time (saniye)", value=0.0, help="İşlem zamanı")
        amount = st.number_input("Amount (TL)", value=100.0, min_value=0.0, help="İşlem tutarı")
    
    with col2:
        st.info("PCA feature'ları için varsayılan değerler kullanılacaktır.")
    
    # Create sample data with default PCA values
    data = {
        'Time': time,
        'Amount': amount,
    }
    
    # Add default PCA features
    for i in range(1, 29):
        data[f'V{i}'] = 0.0
    
    if st.button("🔍 Tahmin Yap", type="primary"):
        make_prediction(data, threshold)


def show_csv_upload(threshold: float):
    """Show CSV upload interface"""
    st.header("📁 CSV Dosyası Yükleme")
    
    uploaded_file = st.file_uploader(
        "CSV dosyasını yükleyin",
        type=['csv'],
        help="Dosya tüm gerekli feature'ları içermelidir"
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            st.success(f"✅ {len(df)} işlem yüklendi")
            
            # Show data preview
            with st.expander("📊 Veri Önizleme"):
                st.dataframe(df.head(10))
            
            if st.button("🔍 Toplu Tahmin Yap", type="primary"):
                make_batch_prediction(df, threshold)
                
        except Exception as e:
            st.error(f"❌ Dosya yükleme hatası: {str(e)}")


def show_sample_data(threshold: float):
    """Show sample data prediction"""
    st.header("🎯 Örnek Veri ile Test")
    
    # Sample transactions
    sample_normal = {
        'Time': 0, 'Amount': 149.62,
        'V1': -1.359807, 'V2': -0.072781, 'V3': 2.536347, 'V4': 1.378155,
        'V5': -0.338321, 'V6': 0.462388, 'V7': 0.239599, 'V8': 0.098698,
        'V9': 0.363787, 'V10': 0.090794, 'V11': -0.551600, 'V12': -0.617801,
        'V13': -0.991390, 'V14': -0.311169, 'V15': 1.468177, 'V16': -0.470401,
        'V17': 0.207971, 'V18': 0.025791, 'V19': 0.403993, 'V20': 0.251412,
        'V21': -0.018307, 'V22': 0.277838, 'V23': -0.110474, 'V24': 0.066928,
        'V25': 0.128539, 'V26': -0.189115, 'V27': 0.133558, 'V28': -0.021053
    }
    
    sample_type = st.selectbox(
        "Örnek İşlem Seç",
        ["Normal İşlem", "Şüpheli İşlem"]
    )
    
    if sample_type == "Normal İşlem":
        data = sample_normal
    else:
        # Modify for suspicious transaction
        data = sample_normal.copy()
        data['Amount'] = 5000.0
        data['V1'] = -3.5
        data['V2'] = 2.8
    
    # Show sample data
    with st.expander("📊 Örnek Veri Detayları"):
        st.json(data)
    
    if st.button("🔍 Tahmin Yap", type="primary"):
        make_prediction(data, threshold)


def make_prediction(data: dict, threshold: float):
    """Make single prediction"""
    try:
        # Initialize detector
        detector = FraudDetector()
        
        # Check if model exists
        if not detector.model_path.exists():
            st.error("❌ Model dosyası bulunamadı! Lütfen önce modeli eğitin.")
            st.info("💡 `notebooks/06_final_pipeline.ipynb` notebook'unu çalıştırarak modeli eğitebilirsiniz.")
            return
        
        # Make prediction
        result = detector.predict_with_details(data, threshold)[0]
        
        # Display results
        st.divider()
        st.header("📊 Tahmin Sonuçları")
        
        # Alert box
        if result['prediction'] == 1:
            st.markdown(
                f'<div class="fraud-alert">⚠️ DOLANDIRICILIK TESPİT EDİLDİ!</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="normal-alert">✅ NORMAL İŞLEM</div>',
                unsafe_allow_html=True
            )
        
        st.divider()
        
        # Metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Tahmin",
                result['prediction_label'],
                delta=None
            )
        
        with col2:
            st.metric(
                "Dolandırıcılık Olasılığı",
                f"{result['fraud_probability']:.2%}",
                delta=None
            )
        
        with col3:
            st.metric(
                "Risk Seviyesi",
                result['risk_level'],
                delta=None
            )
        
        # Probability gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=result['fraud_probability'] * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Dolandırıcılık Skoru"},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkred"},
                'steps': [
                    {'range': [0, 30], 'color': "lightgreen"},
                    {'range': [30, 60], 'color': "yellow"},
                    {'range': [60, 80], 'color': "orange"},
                    {'range': [80, 100], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': threshold * 100
                }
            }
        ))
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"❌ Tahmin hatası: {str(e)}")


def make_batch_prediction(df: pd.DataFrame, threshold: float):
    """Make batch predictions"""
    try:
        # Initialize detector
        detector = FraudDetector()
        
        if not detector.model_path.exists():
            st.error("❌ Model dosyası bulunamadı!")
            return
        
        # Make predictions
        results = detector.predict_with_details(df, threshold)
        
        # Add results to dataframe
        df_results = df.copy()
        df_results['Prediction'] = [r['prediction_label'] for r in results]
        df_results['Fraud_Probability'] = [r['fraud_probability'] for r in results]
        df_results['Risk_Level'] = [r['risk_level'] for r in results]
        
        # Display summary
        st.divider()
        st.header("📊 Toplu Tahmin Sonuçları")
        
        fraud_count = sum(r['prediction'] for r in results)
        normal_count = len(results) - fraud_count
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Toplam İşlem", len(results))
        with col2:
            st.metric("Normal İşlem", normal_count)
        with col3:
            st.metric("Dolandırıcılık", fraud_count)
        
        # Results table
        st.dataframe(
            df_results,
            use_container_width=True,
            height=400
        )
        
        # Download button
        csv = df_results.to_csv(index=False)
        st.download_button(
            label="📥 Sonuçları İndir (CSV)",
            data=csv,
            file_name="fraud_detection_results.csv",
            mime="text/csv"
        )
        
    except Exception as e:
        st.error(f"❌ Toplu tahmin hatası: {str(e)}")


if __name__ == "__main__":
    main()
