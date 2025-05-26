import streamlit as st
import pandas as pd
import json # Import for handling JSON (ipynb file)
import io # Import for treating string as file-like object
import math # Import for math.ceil
import plotly.express as px # For interactive charts
import numpy as np # For numerical operations

# Import ML components
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier # Added RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt

# Custom styling
st.set_page_config(
    page_title="Klasifikasi Aroma Parfum",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
        .main > div {
            padding: 2rem 3rem;
        }
        .stButton>button {
            width: 100%;
            background-color: #FF4B4B;
            color: white;
            font-weight: bold;
            padding: 0.5rem 1rem;
            border-radius: 0.5rem;
        }
        .stButton>button:hover {
            background-color: #FF3333;
        }
        .stat-box {
            border-radius: 0.5rem;
            padding: 1rem;
            margin: 0.5rem 0;
        }
        h1, h2, h3 {
            color: #FF4B4B;
        }
        .stAlert {
            background-color: rgba(255, 75, 75, 0.1);
            border-left-color: #FF4B4B;
        }
        .sidebar .sidebar-content {
            background-color: #f0f2f6;
        }
    </style>
""", unsafe_allow_html=True)

# Define aroma keywords (from your bismillah_remontada.ipynb)
aroma_keywords = {
    'floral': ['rose', 'jasmine', 'lily', 'violet', 'orchid', 'tuberose', 'orange blossom', 'ylang-ylang', 'peony', 'freesia', 'magnolia', 'hyacinth', 'gardenia', 'mimosa', 'honeysuckle', 'narcissus', 'iris', 'cherry blossom', 'frangipani', 'tiare', 'wisteria', 'lilac', 'carnation', 'lotus', 'osmanthus', 'champaca', 'reseda'],
    'woody': ['cedar', 'sandalwood', 'patchouli', 'oak', 'vetiver', 'wood', 'cypress', 'hinoki', 'gaiac', 'agarwood', 'oud', 'mahogany', 'birch', 'pine', 'fir', 'ebony', 'teak', 'rosewood', 'cypriol', 'amyris', 'cashmere wood', 'driftwood', 'spruce'],
    'oriental': ['vanilla', 'amber', 'musk', 'incense', 'resin', 'spice', 'balsam', 'tonka bean', 'benzoin', 'myrrh', 'saffron', 'cinnamon', 'clove', 'nutmeg', 'cardamom', 'ginger', 'opoponax', 'styrax', 'labdanum', 'cumin', 'frankincense', 'castoreum', 'civet', 'animalic', 'oud'],
    'fresh': ['citrus', 'lemon', 'bergamot', 'lime', 'grapefruit', 'mint', 'green', 'aquatic', 'ozone', 'fresh air', 'sea salt', 'water', 'cucumber', 'eucalyptus', 'verbena', 'rosemary', 'lavender', 'petitgrain', 'aldehydes', 'clean', 'soap'],
    'fruity': ['apple', 'pear', 'peach', 'berry', 'pineapple', 'fruit', 'blackcurrant', 'raspberry', 'plum', 'mango', 'lychee', 'strawberry', 'cherry', 'guava', 'melon', 'apricot', 'fig', 'pomegranate', 'passionfruit', 'quince', 'grape', 'kumquat', 'dragon fruit', 'rhubarb', 'banana'],
    'gourmand': ['chocolate', 'caramel', 'coffee', 'honey', 'sugar', 'sweet', 'praline', 'hazelnut', 'toffee', 'milk', 'cream', 'biscuit', 'cookie', 'whipped cream', 'marzipan', 'licorice', 'cocoa', 'rum', 'whiskey', 'brandy', 'confectionary', 'pastry', 'marshmallow', 'candied', 'syrup', 'dates', 'maple', 'pistachio', 'popcorn', 'cereal', 'rice', 'bread', 'cake']
}

def classify_aroma(note_text):
    note_text = str(note_text).lower()
    found_categories = []
    for category, keywords in aroma_keywords.items():
        if any(keyword in note_text for keyword in keywords):
            found_categories.append(category)
    
    # Prioritize specific categories over 'other'
    if 'floral' in found_categories: return 'Floral'
    if 'fresh' in found_categories: return 'Fresh'
    if 'woody' in found_categories: return 'Woody'
    if 'oriental' in found_categories: return 'Oriental'
    
    if found_categories and not any(cat in found_categories for cat in ['floral', 'fresh', 'woody', 'oriental']):
        return 'Other' 
    if not found_categories: 
        return 'Other'
    
    return 'Other' 

# Function to load and prepare data
@st.cache_data
def load_perfume_data(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            notebook_content = json.load(f)
        
        csv_data = ""
        for cell in notebook_content.get('cells', []):
            if cell.get('cell_type') == 'code' and 'outputs' in cell:
                for output in cell.get('outputs', []):
                    if output.get('output_type') == 'stream' and output.get('name') == 'stdout':
                        csv_data_lines = output.get('text', [])
                        if csv_data_lines and csv_data_lines[-1].strip() == '':
                            csv_data_lines = csv_data_lines[:-1]
                        csv_data = "".join(csv_data_lines)
                        if csv_data:
                            break
                if csv_data:
                    break

        if not csv_data:
            st.error("❌ Error: Tidak dapat menemukan data CSV di dalam file .ipynb.")
            st.stop()

        df = pd.read_csv(io.StringIO(csv_data))
        
        df['Notes'] = df['Notes'].fillna('')
        df['Description'] = df['Description'].fillna('')
        df['combined_text'] = df['Notes'] + ' ' + df['Description']
        
        # Apply rule-based classification to create the 'aroma_category' label
        df['aroma_category'] = df['combined_text'].apply(classify_aroma)
        
        # Filter to only the 5 main categories for ML model training,
        # ensuring the target labels are consistent with what we display.
        valid_categories = ['Floral', 'Fresh', 'Woody', 'Oriental', 'Other']
        df = df[df['aroma_category'].isin(valid_categories)].copy() 

        return df
    except FileNotFoundError:
        st.error(f"❌ Error: File '{file_path}' tidak ditemukan!")
        st.stop()
    except json.JSONDecodeError:
        st.error(f"❌ Error: File '{file_path}' bukan file JSON yang valid.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Terjadi error saat memuat data: {e}")
        st.stop()

# Cache the ML model and vectorizer so they are trained only once
@st.cache_resource
def train_ml_model(df):

    
    # Use only the 5 valid categories as target classes
    valid_categories = ['Floral', 'Fresh', 'Woody', 'Oriental', 'Other']
    
    # Filter data to include only recognized categories for training
    df_ml = df[df['aroma_category'].isin(valid_categories)].copy()

    if df_ml.empty:
        st.warning("Tidak ada data yang cukup untuk melatih model setelah penyaringan kategori. Pastikan data memiliki kategori yang valid.")
        return None, None, None, None, None, None

    # Prepare data for ML
    X = df_ml['combined_text']
    y = df_ml['aroma_category']

    # Encode target labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

    # TF-IDF Vectorization
    tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english') 
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)

    # Train RandomForestClassifier model (changed from LogisticRegression)
    model = RandomForestClassifier(n_estimators=100, random_state=42) # Added n_estimators for RandomForest
    model.fit(X_train_tfidf, y_train)

    # Store labels for plotting confusion matrix later
    class_names = label_encoder.classes_

    return model, tfidf_vectorizer, X_test_tfidf, y_test, class_names, label_encoder

# --- Page Functions ---

def home_page(df_perfume):
    st.title("🌸 Klasifikasi Parfum Berdasarkan Kategori Aroma")
    st.markdown("---")

    # Filter out categories that are not among the 5 main ones if they were generated
    valid_categories = ['Floral', 'Fresh', 'Woody', 'Oriental', 'Other']
    df_perfume_filtered = df_perfume[df_perfume['aroma_category'].isin(valid_categories)]

    st.subheader("Pilih Kategori Aroma:")
    selected_category = st.selectbox(
        "Kategori Aroma",
        options=valid_categories,
        index=0, 
        key='home_category_select' 
    )

    st.markdown("---")
    st.subheader(f"Parfum dalam Kategori: {selected_category}")

    # Filter perfumes by selected category
    filtered_perfumes = df_perfume_filtered[df_perfume_filtered['aroma_category'] == selected_category]

    if not filtered_perfumes.empty:
        # Pagination logic
        perfumes_per_page = 9
        total_perfumes = len(filtered_perfumes)
        total_pages = math.ceil(total_perfumes / perfumes_per_page)

        if 'home_current_page' not in st.session_state:
            st.session_state.home_current_page = 1
        
        if st.session_state.get('home_last_selected_category') != selected_category:
            st.session_state.home_current_page = 1
            st.session_state.home_last_selected_category = selected_category


        # Calculate start and end index for the current page
        start_index = (st.session_state.home_current_page - 1) * perfumes_per_page
        end_index = start_index + perfumes_per_page
        
        # Get perfumes for the current page
        display_perfumes = filtered_perfumes.iloc[start_index:end_index]

        # Display perfumes in a grid or cards
        # Iterasi untuk setiap baris dari 3 kolom
        for i in range(0, len(display_perfumes), 3):
            row_perfumes = display_perfumes.iloc[i:i+3]
            cols = st.columns(3) # Buat 3 kolom untuk setiap baris baru
            for j, (idx, row) in enumerate(row_perfumes.iterrows()):
                with cols[j]:
                    st.markdown(f"**{row['Name']}**")
                    st.markdown(f"*{row['Brand']}*")
                    st.image(row['Image URL'], caption=row['Name'], use_container_width=True)
                    with st.expander("Lihat Deskripsi"):
                        st.write(row['Description'])
                    st.markdown("---")
        
        # Display page navigation buttons at the bottom
        st.markdown("---") 
        col_prev, col_page_info, col_next = st.columns([1, 2, 1])

        with col_prev:
            if st.button("Halaman Sebelumnya", disabled=(st.session_state.home_current_page == 1), key='home_prev_btn'):
                st.session_state.home_current_page -= 1
                st.rerun() 

        with col_page_info:
            st.markdown(f"<h4 style='text-align: center;'>Halaman {st.session_state.home_current_page} dari {total_pages}</h4>", unsafe_allow_html=True)
            
        with col_next:
            if st.button("Halaman Selanjutnya", disabled=(st.session_state.home_current_page == total_pages), key='home_next_btn'):
                st.session_state.home_current_page += 1
                st.rerun() 

    else:
        st.info(f"Tidak ada parfum yang ditemukan dalam kategori '{selected_category}'.")

def data_page(df_perfume):
    st.title("📊 Data Parfum dan Distribusi Kategori")
    st.markdown("---")

    st.subheader("Data Mentah Parfum")
    st.dataframe(df_perfume)

<<<<<<< HEAD
                # --- Menampilkan Metrik Evaluasi ---
                accuracy = accuracy_score(y_test_model, y_pred_model)
                
                st.subheader("Metrik Akurasi Keseluruhan")
                st.markdown(f"""
                    <div class='stat-box' style='background-color: #7F8CAA; border-left: 5px solid lightgreen;'>
                        <h3>🎯 Akurasi Model</h3>
                        <h2>{accuracy:.4f}</h2>
                        <p>semakin tinggi semakin baik</p>
                    </div>
                """, unsafe_allow_html=True)
                
                st.markdown("---")
=======
    st.subheader("Distribusi Parfum per Kategori Aroma")
    category_counts = df_perfume['aroma_category'].value_counts().reset_index()
    category_counts.columns = ['Kategori Aroma', 'Jumlah Parfum']
>>>>>>> 0c7ce14068263ff44e53eb28fd319dac8395e453

    st.dataframe(category_counts)

    fig = px.bar(category_counts, x='Kategori Aroma', y='Jumlah Parfum', 
                 title='Distribusi Jumlah Parfum per Kategori Aroma',
                 color='Kategori Aroma',
                 labels={'Kategori Aroma': 'Kategori Aroma', 'Jumlah Parfum': 'Jumlah Parfum'},
                 template='plotly_white')
    st.plotly_chart(fig, use_container_width=True)

def performance_page(df_perfume):
    st.title("📈 Performa Klasifikasi Aroma")
    st.markdown("---")

    st.write("""
    Bagian ini menampilkan metrik kinerja dari model klasifikasi aroma. 
    Model dilatih menggunakan `Random Forest Classifier` pada deskripsi parfum yang telah diubah 
    menjadi fitur numerik menggunakan `TF-IDF Vectorizer`.
    """)

    # Train/load the model
    model, tfidf_vectorizer, X_test_tfidf, y_test_encoded, class_names, label_encoder = train_ml_model(df_perfume)

    if model is None:
        st.warning("Model tidak dapat dilatih. Pastikan data cukup dan memiliki kategori yang valid.")
        return

    # Make predictions
    y_pred_encoded = model.predict(X_test_tfidf)
    y_pred_labels = label_encoder.inverse_transform(y_pred_encoded)
    y_test_labels = label_encoder.inverse_transform(y_test_encoded)

    st.subheader("Metrik Kinerja Model")
    
    # Accuracy
    accuracy = accuracy_score(y_test_encoded, y_pred_encoded)
    st.markdown(f"""
    <div style='background-color: #7F8CAA; padding: 1rem; border-radius: 0.5rem; text-align: center; margin-bottom: 1rem;'>
        <h4>🎯 Akurasi Model:</h4>
        <h2>{accuracy:.2f}</h2>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Laporan Klasifikasi")
    st.text("Precision, Recall, F1-score per Kategori:")
    report_dict = classification_report(y_test_encoded, y_pred_encoded, target_names=class_names, output_dict=True)
    
    # Convert report to DataFrame for better display
    report_df = pd.DataFrame(report_dict).transpose()
    st.dataframe(report_df.round(2))

    st.subheader("Confusion Matrix")
    st.write("Visualisasi perbandingan antara kategori aroma aktual dan yang diprediksi:")

    cm = confusion_matrix(y_test_encoded, y_pred_encoded)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel('Prediksi')
    ax.set_ylabel('Aktual')
    ax.set_title('Confusion Matrix')
    st.pyplot(fig) 

# Main application logic
def main():
    # Load data once
    df_perfume = load_perfume_data("bismillah_remontada.ipynb")

    if df_perfume is None:
        return 

    st.sidebar.title("Navigasi")
    page_selection = st.sidebar.radio(
        "Pilih Halaman",
        ["Halaman Utama", "Data & Distribusi", "Performa Klasifikasi"]
    )

    if page_selection == "Halaman Utama":
        home_page(df_perfume)
    elif page_selection == "Data & Distribusi":
        data_page(df_perfume)
    elif page_selection == "Performa Klasifikasi":
        performance_page(df_perfume)

if __name__ == "__main__":
    main()