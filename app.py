import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import re
import string
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split # Diperlukan untuk Performance Model

# --- Konfigurasi Halaman dan Styling Kustom ---
st.set_page_config(
    page_title="Rekomendasi Parfum Cerdas",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
    <style>
        .main > div {
            padding: 2rem 3rem;
        }
        .stButton>button {
            width: 100%;
            background-color: #f7a8b8; /* Soft pink */
            color: white;
            font-weight: bold;
            padding: 0.5rem 1rem;
            border-radius: 0.5rem;
            border: none;
            cursor: pointer;
            transition: background-color 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #e67c8e; /* Darker pink */
        }
        .stat-box {
            background-color: rgba(247, 168, 184, 0.1); /* Light pink background */
            border-left: 5px solid #f7a8b8; /* Pink border */
            border-radius: 0.5rem;
            padding: 1rem;
            margin: 0.5rem 0;
        }
        h1, h2, h3, .st-bm {
            color: #f7a8b8; /* Pink headers */
        }
        .stAlert {
            background-color: rgba(247, 168, 184, 0.1);
            border-left-color: #f7a8b8;
            color: #d14760;
        }
        .sidebar .sidebar-content {
            background-color: #fdf2f4; /* Very light pink sidebar */
        }
        .css-vk326u { /* Streamlit's default info box */
            background-color: rgba(247, 168, 184, 0.1);
            border-left-color: #f7a8b8;
        }
        .css-1aum76r { /* Streamlit's default warning box */
            background-color: rgba(247, 168, 184, 0.1);
            border-left-color: #f7a8b8;
        }
        /* Style for individual perfume recommendation cards */
        .perfume-card {
            background-color: rgba(255, 255, 255, 0.9);
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            display: flex;
            align-items: center;
            gap: 20px;
        }
        .perfume-card img {
            border-radius: 5px;
            object-fit: cover;
        }
        .perfume-details {
            flex-grow: 1;
        }
        .perfume-details h4 {
            color: #f7a8b8;
            margin-bottom: 5px;
        }
        .perfume-details p {
            margin-bottom: 5px;
            color: #333;
        }
    </style>
""", unsafe_allow_html=True)

# --- Fungsi Pra-pemrosesan Data (dari notebook Anda) ---
# Pindahkan blok aroma_categories, categorize_notes, dan clean_text ke bagian awal
# agar didefinisikan sebelum dipanggil oleh fungsi lain
aroma_categories = {
    'Floral': ['rose', 'jasmine', 'lily', 'violet', 'gardenia', 'tuberose', 'orchid', 'peony', 'honeysuckle', 'orange flower', 'neroli', 'muguet', 'freesia'],
    'Fruity': ['apple', 'peach', 'berry', 'pear', 'pineapple', 'mango', 'cherry', 'grape', 'yuzu', 'lemon', 'lime', 'bergamot', 'tangerine', 'mandarin', 'orange', 'grapefruit', 'currant', 'plum', 'fig', 'raspberry'],
    'Woody': ['sandalwood', 'cedar', 'patchouli', 'vetiver', 'oak', 'wood', 'pine', 'oud', 'agarwood', 'cypress', 'guaiac wood', 'ebony', 'balsam', 'fir'],
    'Fresh': ['lemon', 'lime', 'bergamot', 'mint', 'green tea', 'citrus', 'grapefruit', 'aquatic', 'ozone', 'clean', 'aldehydes', 'sea notes', 'water notes', 'ginger'],
    'Oriental': ['vanilla', 'amber', 'musk', 'incense', 'cinnamon', 'clove', 'nutmeg', 'tonka', 'saffron', 'spice', 'resin', 'myrrh', 'frankincense', 'benzoin', 'labdanum', 'opoponax'],
    'Gourmand': ['chocolate', 'caramel', 'coffee', 'honey', 'milk', 'praline', 'almond', 'cookie', 'sugar', 'marshmallow', 'cream', 'liqueur'],
    'Spicy': ['pepper', 'cinnamon', 'clove', 'nutmeg', 'cardamom', 'ginger', 'pink pepper', 'cumin', 'coriander'],
    'Earthy': ['moss', 'soil', 'earth', 'dirt', 'mushroom', 'truffle', 'oakmoss'],
    'Resinous': ['amber', 'frankincense', 'myrrh', 'benzoin', 'elemi', 'labdanum'],
    'Animalic': ['musk', 'leather', 'castoreum', 'civet', 'ambergris'],
    'Green': ['grass', 'leaf', 'galbanum', 'herbal', 'tea', 'fig leaf'],
    'Powdery': ['iris', 'violet', 'heliotrope', 'musk', 'vanilla', 'orris'],
    'Leather': ['leather', 'suede'],
    'Tobacco': ['tobacco', 'hay'],
    'Alcoholic': ['cognac', 'rum', 'whiskey', 'wine', 'champagne'],
    'Aldehydic': ['aldehydes'],
    'Herbal': ['lavender', 'rosemary', 'thyme', 'sage', 'basil', 'mint', 'camphor'],
    'Metallic': ['metallic notes'],
    'Ozonic': ['ozone', 'air notes'],
    'Marine': ['sea notes', 'marine notes', 'salt'],
    'Smoky': ['smoke', 'incense', 'tar', 'birch'],
    'Warm Spicy': ['cinnamon', 'clove', 'nutmeg', 'cardamom', 'vanilla', 'amber']
}

def categorize_notes(notes):
    if pd.isna(notes):
        return 'Other'
    notes = str(notes).lower()
    found = [cat for cat, keywords in aroma_categories.items() if any(k in notes for k in keywords)]
    return ', '.join(found) if found else 'Other'

def clean_text(text):
    text = str(text).lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def assign_characteristics_to_df(df_input):
    df_input['characteristics'] = df_input['Notes'].astype(str).apply(categorize_notes)
    return df_input

def get_primary_characteristic(characteristics_str):
    """Extract the primary (first) characteristic from a comma-separated string."""
    if pd.isna(characteristics_str):
        return 'Other'
    characteristics_str = str(characteristics_str)
    if ',' in characteristics_str:
        return characteristics_str.split(',')[0].strip()
    return characteristics_str.strip()

# --- Fungsi Pemuatan Data dan Model ---
@st.cache_data
def load_data_and_models():
    try:
        df_perfume = pd.read_csv('final_perfume_data.csv', encoding='latin-1')
        best_model_svm = joblib.load('svm_perfume_model.pkl')
        vectorizer = joblib.load('tfidf_vectorizer.pkl')
        label_encoder = joblib.load('label_encoder.pkl')
        return df_perfume, best_model_svm, vectorizer, label_encoder
    except FileNotFoundError as e:
        st.error(f"❌ Error: File penting tidak ditemukan! Pastikan Anda memiliki 'final_perfume_data.csv', 'svm_perfume_model.pkl', 'tfidf_vectorizer.pkl', dan 'label_encoder.pkl' di direktori yang sama. Detail: {e}")
        st.stop()

# --- Fungsi Rekomendasi ---
def get_perfume_recommendations(input_notes, df_original, vectorizer_obj, model_obj, label_encoder_obj, top_n=5):
    clean_input_notes = clean_text(input_notes)
    
    # Periksa apakah vectorizer_obj memiliki vocabulary yang sesuai
    if not hasattr(vectorizer_obj, 'vocabulary_') or not vectorizer_obj.vocabulary_:
        # Fallback: jika vectorizer kosong, kita tidak bisa melakukan prediksi berbasis model
        # Coba cari berdasarkan kata kunci langsung
        st.warning("Vectorizer belum dilatih atau tidak memiliki vocabulary. Mencoba mencari berdasarkan kata kunci saja.")
        keywords = clean_input_notes.split()
        if keywords:
            search_pattern = '|'.join(keywords)
            initial_recommendations = df_original[
                df_original['Description'].str.contains(search_pattern, case=False, na=False) |
                df_original['Notes'].str.contains(search_pattern, case=False, na=False)
            ].drop_duplicates(subset=['Name', 'Brand']).head(top_n)
            return initial_recommendations if not initial_recommendations.empty else "Maaf, tidak ada rekomendasi parfum berdasarkan kata kunci yang Anda masukkan."
        else:
            return "Mohon masukkan kata kunci aroma yang valid."

    input_vector = vectorizer_obj.transform([clean_input_notes])
    
    # Prediksi kategori aroma
    predicted_label_index = model_obj.predict(input_vector)[0]
    
    try:
        predicted_characteristic = label_encoder_obj.inverse_transform([predicted_label_index])[0]
    except ValueError:
        st.warning(f"Label {predicted_label_index} tidak ditemukan di encoder. Ini mungkin terjadi jika data training model berbeda atau kategori sangat jarang.")
        predicted_characteristic = "Other" # Fallback category

    # Cari parfum berdasarkan karakteristik yang diprediksi
    # Gunakan str.contains untuk mencari di kolom 'characteristics' yang bisa memiliki multiple categories
    initial_recommendations = df_original[
        df_original['characteristics'].str.contains(predicted_characteristic, case=False, na=False)
    ].drop_duplicates(subset=['Name', 'Brand']) # Hapus duplikat
    
    if initial_recommendations.empty:
        # Jika tidak ada yang cocok persis, coba cari berdasarkan kata kunci dari input
        st.info(f"Tidak ada parfum yang cocok persis dengan kategori '{predicted_characteristic}'. Mencoba mencari berdasarkan kata kunci dalam deskripsi atau notes.")
        
        keywords = clean_input_notes.split()
        if keywords:
            search_pattern = '|'.join(keywords)
            fallback_recommendations = df_original[
                df_original['Description'].str.contains(search_pattern, case=False, na=False) |
                df_original['Notes'].str.contains(search_pattern, case=False, na=False)
            ].drop_duplicates(subset=['Name', 'Brand']).head(top_n)
            
            return fallback_recommendations if not fallback_recommendations.empty else "Maaf, tidak ada rekomendasi parfum berdasarkan kata kunci yang Anda masukkan."
        else:
            return "Maaf, tidak ada rekomendasi parfum berdasarkan karakteristik yang Anda masukkan."
    
    return initial_recommendations.head(top_n)

# --- Sidebar Navigasi ---
with st.sidebar:
    st.title("🌸 Navigasi Aplikasi")
    st.markdown("---")
    page = st.radio(
        "Pilih Halaman",
        ["Home", "Eksplorasi Data", "Rekomendasi Parfum", "Performa Model"],
        format_func=lambda x: f"✨ {x}"
    )
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; padding: 1rem;'>
            <h4>Tentang Aplikasi</h4>
            <p>Aplikasi ini merekomendasikan parfum berdasarkan aroma yang Anda inginkan menggunakan model klasifikasi.</p>
            <p>Model dilatih dengan data parfum dan menggunakan algoritma Support Vector Machine (SVM) dengan fitur TF-IDF.</p>
        </div>
    """, unsafe_allow_html=True)
    st.sidebar.write("---")
    st.sidebar.write("Dataset: `final_perfume_data.csv`")
    st.sidebar.write("Model: Support Vector Machine (SVM)")

# --- Memuat Data dan Model ---
df, model_svm, vectorizer_obj, label_encoder_obj = load_data_and_models()

# Pastikan kolom 'characteristics' sudah ada di df yang dimuat
df = assign_characteristics_to_df(df)

# --- Halaman Home ---
if page == "Home":
    st.title("Selamat Datang di Rekomendasi Parfum Cerdas 🌸")
    st.markdown("---")
    
    st.markdown("""
        <div style=' padding: 2rem; border-radius: 0.5rem; margin-bottom: 2rem;'>
            <h3 style='margin-top: 0;'>Temukan Aroma Ideal Anda!</h3>
            <p>Aplikasi ini dirancang untuk membantu Anda menemukan parfum yang paling sesuai dengan preferensi aroma Anda. 
            Cukup masukkan deskripsi aroma yang Anda inginkan (misalnya, floral, woody, vanilla, fresh), dan kami akan merekomendasikan beberapa pilihan terbaik untuk Anda.</p>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
            <div class='stat-box'>
                <h3>📊 Total Parfum</h3>
                <h2>{len(df):,}</h2>
                <p>dalam database</p>
            </div>
        """, unsafe_allow_html=True)
        
    with col2:
        num_brands = df['Brand'].nunique()
        st.markdown(f"""
            <div class='stat-box'>
                <h3>🏭 Total Brand</h3>
                <h2>{num_brands}</h2>
                <p>parfum</p>
            </div>
        """, unsafe_allow_html=True)
        
    with col3:
        num_categories = len(label_encoder_obj.classes_)
        st.markdown(f"""
            <div class='stat-box'>
                <h3>🏷️ Total Kategori Aroma</h3>
                <h2>{num_categories}</h2>
                <p>yang terdeteksi</p>
            </div>
        """, unsafe_allow_html=True)
        
    st.markdown("---")
    st.subheader("Bagaimana Cara Kerjanya?")
    st.markdown("""
    1.  **Masukkan Aroma:** Anda menuliskan karakteristik aroma yang Anda inginkan (misalnya, "sweet floral with a hint of vanilla").
    2.  **Analisis AI:** Model kecerdasan buatan kami akan menganalisis input Anda untuk memahami preferensi aroma utama.
    3.  **Rekomendasi Cerdas:** Aplikasi akan mencocokkan preferensi Anda dengan database parfum dan menampilkan rekomendasi yang paling relevan.
    """)

# --- Halaman Eksplorasi Data ---
elif page == "Eksplorasi Data":
    st.title("📊 Eksplorasi Data Parfum")
    st.markdown("---")
    
    st.subheader("Ringkasan Data")
    st.write("Berikut adalah beberapa baris pertama dari dataset parfum:")
    st.dataframe(df.head(), use_container_width=True)
    
    st.subheader("Statistik Deskriptif")
    st.write(df.describe(include='all'), use_container_width=True)
    
    st.subheader("Distribusi Kategori Aroma")
    # Create a series of all individual categories
    all_categories = []
    for characteristics in df['characteristics'].dropna():
        if ',' in str(characteristics):
            categories = [cat.strip() for cat in str(characteristics).split(',')]
            all_categories.extend(categories)
        else:
            all_categories.append(str(characteristics))
    
    category_counts = pd.Series(all_categories).value_counts()
    
    # Plotting with Matplotlib/Seaborn for better control
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.barplot(x=category_counts.values, y=category_counts.index, palette='RdPu', ax=ax)
    ax.set_title('Distribusi Kategori Aroma Parfum', fontsize=16)
    ax.set_xlabel('Jumlah Parfum', fontsize=12)
    ax.set_ylabel('Kategori Aroma', fontsize=12)
    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("Top 10 Brand Parfum Terbanyak")
    top_brands = df['Brand'].value_counts().head(10)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=top_brands.values, y=top_brands.index, palette='mako', ax=ax)
    ax.set_title('Top 10 Brand Parfum Terbanyak', fontsize=16)
    ax.set_xlabel('Jumlah Parfum', fontsize=12)
    ax.set_ylabel('Brand', fontsize=12)
    plt.tight_layout()
    st.pyplot(fig)

# --- Halaman Rekomendasi Parfum ---
elif page == "Rekomendasi Parfum":
    st.title("Dapatkan Rekomendasi Parfum Anda! 💖")
    st.markdown("---")
    
    st.markdown("""
    Cukup ketikkan kata kunci aroma yang Anda inginkan (misalnya: *floral, vanilla, woody, fresh, citrus, amber, spicy, gourmand, powdery, leather, tobacco*). 
    Anda bisa memasukkan beberapa kata kunci yang dipisahkan koma atau spasi.
    """)

    user_input = st.text_area(
        "Ketikkan karakteristik aroma parfum yang Anda inginkan:", 
        "sweet floral with a hint of vanilla and musk"
    )

    if st.button("Dapatkan Rekomendasi Parfum"):
        if user_input:
            with st.spinner("Menganalisis preferensi aroma Anda dan mencari rekomendasi..."):
                recommendations = get_perfume_recommendations(user_input, df, vectorizer_obj, model_svm, label_encoder_obj)
                
                if isinstance(recommendations, str):
                    st.warning(recommendations)
                elif not recommendations.empty:
                    st.subheader(f"✨ Parfum yang Direkomendasikan ✨")
                    
                    # Tambahkan bagian ini untuk menampilkan kategori aroma yang diprediksi
                    try:
                        predicted_category_from_input = label_encoder_obj.inverse_transform(model_svm.predict(vectorizer_obj.transform([clean_text(user_input)])))[0]
                        st.info(f"Rekomendasi ini didasarkan pada input Anda yang diprediksi masuk dalam kategori aroma: **{predicted_category_from_input}**")
                    except Exception as e:
                        st.warning(f"Tidak dapat menentukan kategori prediksi utama dari input Anda. Detail: {e}")
                    
                    for index, row in recommendations.iterrows():
                        st.markdown(f"""
                            <div class="perfume-card">
                                {"<img src='" + row['Image URL'] + "' width='100' height='100' alt='Gambar Parfum'>" if pd.notna(row['Image URL']) and row['Image URL'].startswith('http') else ""}
                                <div class="perfume-details">
                                    <h4>{row['Name']}</h4>
                                    <p><strong>Brand:</strong> {row['Brand']}</p>
                                    <p><strong>Karakteristik Aroma:</strong> {row['characteristics']}</p>
                                    <p><strong>Notes Utama:</strong> {row['Notes']}</p>
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
                        
                else:
                    st.info("Maaf, tidak ada rekomendasi yang ditemukan untuk preferensi aroma Anda.")
        else:
            st.warning("Mohon masukkan karakteristik aroma untuk mendapatkan rekomendasi.")

# --- Halaman Performa Model ---
# --- Halaman Performa Model ---
elif page == "Performa Model":
    st.title("📈 Performa Model Klasifikasi")
    st.markdown("---")
    
    st.markdown("""
    Halaman ini menampilkan metrik evaluasi untuk model klasifikasi yang digunakan dalam aplikasi ini. 
    Kami menggunakan metrik standar untuk model klasifikasi multi-kelas.
    """)

    # --- Persiapan Data untuk Evaluasi Model ---
    # Di sini kita perlu melakukan pra-pemrosesan yang sama seperti saat melatih model
    # agar data test cocok dengan ekspektasi model.
    
    # Pastikan df_perfume memiliki 'clean_notes' dan 'label' untuk evaluasi
    df_eval = df.copy() # Gunakan salinan DataFrame
    # df_eval['characteristics'] sudah dibuat di load_data_and_models
    df_eval['clean_notes'] = df_eval['Notes'].astype(str).apply(clean_text)
    
    # Handle multi-label characteristics by taking only the primary (first) characteristic
    df_eval['primary_characteristic'] = df_eval['characteristics'].apply(get_primary_characteristic)
    
    # Encode label untuk dataset evaluasi
    # Pastikan 'primary_characteristic' tidak kosong atau NaN sebelum transform
    df_eval['primary_characteristic'] = df_eval['primary_characteristic'].fillna('Other')
    
    # Filter df_eval agar hanya menyertakan kategori yang dikenal oleh label_encoder
    known_classes = set(label_encoder_obj.classes_)
    df_eval = df_eval[df_eval['primary_characteristic'].isin(known_classes)].copy()
    
    if df_eval.empty:
        st.warning("Tidak ada data yang valid untuk evaluasi model setelah pemrosesan karakteristik. Pastikan dataset memiliki notes yang dapat dikategorikan dan sesuai dengan model yang dilatih.")
        
        # Show what classes are available vs what's in the data
        st.subheader("Informasi Debug")
        st.write("**Kelas yang dikenal oleh model:**")
        st.write(list(label_encoder_obj.classes_))
        
        # Show sample of characteristics from data
        sample_chars = df['characteristics'].value_counts().head(10)
        st.write("**Sample karakteristik dalam data:**")
        st.write(sample_chars)
        
    else:
        # Check if vectorizer_obj has been fit
        if not hasattr(vectorizer_obj, 'vocabulary_') or not vectorizer_obj.vocabulary_:
            st.warning("Vectorizer belum dilatih atau tidak memiliki vocabulary. Tidak dapat melakukan evaluasi model.")
        else:
            X_eval = vectorizer_obj.transform(df_eval['clean_notes'])
            y_eval = label_encoder_obj.transform(df_eval['primary_characteristic'])

            # Bagi data untuk evaluasi (sesuai dengan split saat training)
            # Pastikan ada cukup sampel untuk split jika df_eval menjadi sangat kecil
            if len(df_eval) > 10: # Minimal 10 sampel untuk split yang masuk akal
                try:
                    _, X_test_model, _, y_test_model = train_test_split(
                        X_eval, y_eval, 
                        test_size=0.2, 
                        random_state=42, 
                        stratify=y_eval if len(np.unique(y_eval)) > 1 else None
                    )
                except ValueError:
                    # If stratify fails, do random split
                    _, X_test_model, _, y_test_model = train_test_split(
                        X_eval, y_eval, 
                        test_size=0.2, 
                        random_state=42
                    )
            else:
                X_test_model = X_eval
                y_test_model = y_eval

            if X_test_model.shape[0] == 0:
                st.warning("Tidak ada data uji yang cukup untuk evaluasi model setelah pembagian data.")
            else:
                # Prediksi menggunakan model terbaik (SVM)
                y_pred_model = model_svm.predict(X_test_model)

                # --- Menampilkan Metrik Evaluasi ---
                accuracy = accuracy_score(y_test_model, y_pred_model)
                
                st.subheader("Metrik Akurasi Keseluruhan")
                st.markdown(f"""
                    <div class='stat-box' style='background-color: rgba(144, 238, 144, 0.1); border-left: 5px solid lightgreen;'>
                        <h3>🎯 Akurasi Model</h3>
                        <h2>{accuracy:.4f}</h2>
                        <p>semakin tinggi semakin baik</p>
                    </div>
                """, unsafe_allow_html=True)
                
                st.markdown("---")

                st.subheader("Laporan Klasifikasi (Precision, Recall, F1-Score)")
                
                # Get all class names from the label encoder
                all_class_names = label_encoder_obj.classes_
                
                # Generate classification report with labels parameter to specify all possible labels
                try:
                    report = classification_report(
                        y_test_model, y_pred_model, 
                        labels=range(len(all_class_names)),  # Specify all possible label indices
                        target_names=all_class_names,        # All class names
                        output_dict=True, 
                        zero_division=0
                    )
                    
                    # Mengonversi report ke DataFrame untuk tampilan yang lebih baik
                    report_df = pd.DataFrame(report).transpose()
                    st.dataframe(report_df.style.format('{:.2f}'), use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Error generating classification report: {e}")
                    
                    # Alternative: Show only classes present in test set
                    st.info("Menampilkan laporan untuk kelas yang ada dalam data uji saja...")
                    unique_y_test = np.unique(y_test_model)
                    actual_target_names = [label_encoder_obj.inverse_transform([lbl])[0] for lbl in unique_y_test]
                    
                    # Use only the classes present in test set
                    report_subset = classification_report(
                        y_test_model, y_pred_model, 
                        labels=unique_y_test,
                        target_names=actual_target_names, 
                        output_dict=True, 
                        zero_division=0
                    )
                    
                    report_df_subset = pd.DataFrame(report_subset).transpose()
                    st.dataframe(report_df_subset.style.format('{:.2f}'), use_container_width=True)
                
                st.markdown("---")
                
                st.subheader("Confusion Matrix")
                st.write("Visualisasi seberapa baik model mengklasifikasikan setiap kategori.")
                
                # For confusion matrix, use only classes present in test set
                unique_y_test = np.unique(y_test_model)
                unique_y_pred = np.unique(y_pred_model)
                
                # Get all unique labels that appear in either test or prediction
                all_unique_labels = np.unique(np.concatenate([unique_y_test, unique_y_pred]))
                
                # Get class names for these labels only
                confusion_class_names = [label_encoder_obj.inverse_transform([lbl])[0] for lbl in all_unique_labels]
                
                # Generate confusion matrix with specified labels
                cm = confusion_matrix(y_test_model, y_pred_model, labels=all_unique_labels)
                
                # Only show confusion matrix if it's not too large
                if len(confusion_class_names) <= 20:  # Limit to 20 classes for readability
                    # Menggambar heatmap confusion matrix
                    fig_cm, ax_cm = plt.subplots(figsize=(12, 10))
                    sns.heatmap(
                        cm, 
                        annot=True, 
                        fmt='d', 
                        cmap='PuRd', # Red-Purple colormap
                        xticklabels=confusion_class_names,
                        yticklabels=confusion_class_names,
                        linewidths=.5,
                        linecolor='lightgray',
                        cbar_kws={'shrink': 0.75}
                    )
                    ax_cm.set_title('Confusion Matrix', fontsize=18, pad=20)
                    ax_cm.set_xlabel('Label Prediksi', fontsize=14)
                    ax_cm.set_ylabel('Label Sebenarnya', fontsize=14)
                    plt.xticks(rotation=45, ha='right', fontsize=10)
                    plt.yticks(rotation=0, fontsize=10)
                    plt.tight_layout()
                    st.pyplot(fig_cm)
                else:
                    st.info(f"Confusion matrix memiliki {len(confusion_class_names)} kelas, terlalu banyak untuk ditampilkan dengan jelas. Menampilkan informasi ringkas saja.")
                    st.write(f"**Akurasi per kelas (berdasarkan diagonal confusion matrix):**")
                    
                    # Calculate per-class accuracy from confusion matrix diagonal
                    class_accuracies = {}
                    for i, class_name in enumerate(confusion_class_names):
                        if i < len(cm) and cm[i, i] > 0:
                            class_total = cm[i, :].sum()
                            if class_total > 0:
                                class_acc = cm[i, i] / class_total
                                class_accuracies[class_name] = class_acc
                    
                    if class_accuracies:
                        acc_df = pd.DataFrame(list(class_accuracies.items()), columns=['Kelas', 'Akurasi'])
                        acc_df = acc_df.sort_values('Akurasi', ascending=False)
                        st.dataframe(acc_df.style.format({'Akurasi': '{:.2f}'}), use_container_width=True)

                st.markdown("---")
                
                # Additional information about the evaluation
                st.subheader("Informasi Evaluasi")
                st.write(f"**Jumlah sampel untuk evaluasi:** {X_test_model.shape[0]}")
                st.write(f"**Jumlah total kelas dalam model:** {len(all_class_names)}")
                st.write(f"**Jumlah kelas yang muncul dalam data uji:** {len(np.unique(y_test_model))}")
                st.write(f"**Jumlah kelas yang diprediksi:** {len(np.unique(y_pred_model))}")
                
                # Show class distribution in test set
                st.write("**Distribusi kelas dalam data uji:**")
                test_class_names = [label_encoder_obj.inverse_transform([lbl])[0] for lbl in y_test_model]
                test_class_dist = pd.Series(test_class_names).value_counts()
                st.write(test_class_dist)
                
                st.markdown("---")
                st.info("""
                **Interpretasi Metrik:**
                * **Akurasi:** Proporsi prediksi yang benar secara keseluruhan.
                * **Precision (Presisi):** Dari semua yang diprediksi sebagai kelas X, berapa banyak yang sebenarnya kelas X.
                * **Recall:** Dari semua yang sebenarnya kelas X, berapa banyak yang berhasil diprediksi sebagai kelas X.
                * **F1-Score:** Rata-rata harmonik dari presisi dan recall, berguna untuk menyeimbangkan keduanya.
                * **Confusion Matrix:** Menunjukkan jumlah *true positives*, *true negatives*, *false positives*, dan *false negatives* untuk setiap kelas. Diagonal menunjukkan prediksi yang benar.
                
                **Catatan:** Evaluasi ini menggunakan karakteristik primer (pertama) dari setiap parfum untuk menghindari masalah multi-label dalam evaluasi model klasifikasi single-label.
                
                **Penting:** Model Anda dilatih dengan {len(all_class_names)} kelas, namun tidak semua kelas mungkin muncul dalam setiap evaluasi. Ini normal dalam dataset dengan distribusi kelas yang tidak seimbang.
                """)