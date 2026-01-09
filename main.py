import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from google_play_scraper import Sort, reviews

# Machine Learning & Deep Learning Imports
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# --- KONFIGURASI GLOBAL ---
APP_ID = 'com.shopee.id'
JUMLAH_DATA = 3000
MAX_WORDS = 5000
MAX_LEN = 100
EMBEDDING_DIM = 64
FILTER_SIZE = 128
KERNEL_SIZE = 5
RANDOM_STATE = 42

# Setup Style Visualisasi
sns.set_style("whitegrid")
plt.rcParams.update({'font.size': 11, 'font.family': 'sans-serif'})

# ==========================================
# BAGIAN 1: PENGOLAHAN DATA & MODEL
# ==========================================

def scrape_google_play_data(app_id, count=1000):
    print(f"[INFO] Memulai scraping {count} ulasan untuk {app_id}...")
    result, _ = reviews(app_id, lang='id', country='id', sort=Sort.NEWEST, count=count)
    df = pd.DataFrame(result)
    return df

def preprocess_and_label(df):
    # Rating 4-5 = Positif (1), Rating 1-2 = Negatif (0), Rating 3 = Drop
    df = df[df['score'] != 3].copy()
    df['label'] = df['score'].apply(lambda x: 1 if x > 3 else 0)
    
    def clean_text(text):
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    df['clean_ulasan'] = df['content'].apply(clean_text)
    return df

def build_cnn_model():
    model = Sequential([
        Embedding(MAX_WORDS, EMBEDDING_DIM, input_length=MAX_LEN),
        Conv1D(FILTER_SIZE, KERNEL_SIZE, activation='relu'),
        GlobalMaxPooling1D(),
        Dense(32, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# ==========================================
# BAGIAN 2: FUNGSI VISUALISASI LENGKAP
# ==========================================

def plot_class_distribution(y_original, y_resampled):
    """Visualisasi 1: Distribusi Data Sebelum vs Sesudah SMOTE"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot Sebelum SMOTE
    unique, counts = np.unique(y_original, return_counts=True)
    sns.barplot(x=['Negatif (0)', 'Positif (1)'], y=counts, ax=axes[0], palette='Reds')
    axes[0].set_title(f'Distribusi Asli (Imbalanced)\nTotal: {len(y_original)}')
    axes[0].bar_label(axes[0].containers[0])

    # Plot Sesudah SMOTE
    unique_sm, counts_sm = np.unique(y_resampled, return_counts=True)
    sns.barplot(x=['Negatif (0)', 'Positif (1)'], y=counts_sm, ax=axes[1], palette='Greens')
    axes[1].set_title(f'Distribusi Setelah SMOTE (Balanced)\nTotal: {len(y_resampled)}')
    axes[1].bar_label(axes[1].containers[0])
    
    plt.tight_layout()
    plt.show()

def plot_side_by_side_confusion_matrix(y_true, y_pred_no, y_pred_smote):
    """Visualisasi 2: Komparasi Confusion Matrix"""
    cm_no = confusion_matrix(y_true, y_pred_no)
    cm_smote = confusion_matrix(y_true, y_pred_smote)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Heatmap 1: Tanpa SMOTE
    sns.heatmap(cm_no, annot=True, fmt='d', cmap='Reds', cbar=False, ax=axes[0],
                xticklabels=['Negatif', 'Positif'], yticklabels=['Negatif', 'Positif'], annot_kws={"size": 14})
    axes[0].set_title('TANPA SMOTE\n(Cenderung Bias ke Positif)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Prediksi')
    axes[0].set_ylabel('Aktual')

    # Heatmap 2: Dengan SMOTE
    sns.heatmap(cm_smote, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[1],
                xticklabels=['Negatif', 'Positif'], yticklabels=['Negatif', 'Positif'], annot_kws={"size": 14})
    axes[1].set_title('DENGAN SMOTE\n(Lebih Sensitif thd Negatif)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Prediksi')
    axes[1].set_ylabel('Aktual')
    
    plt.tight_layout()
    plt.show()

def plot_metric_comparison(metrics_no, metrics_smote):
    """Visualisasi 3: Grafik Batang Performa (Fokus Kelas Negatif)"""
    labels = ['Akurasi Total', 'Recall (Negatif)', 'F1-Score (Negatif)']
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, metrics_no, width, label='Tanpa SMOTE', color='tab:red')
    rects2 = ax.bar(x + width/2, metrics_smote, width, label='Dengan SMOTE', color='tab:blue')
    
    ax.set_ylabel('Skor (0-1)')
    ax.set_title('Bukti Efektivitas SMOTE pada Kelas Minoritas')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_ylim(0, 1.15)
    
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontweight='bold')
    autolabel(rects1)
    autolabel(rects2)
    plt.tight_layout()
    plt.show()

def plot_learning_curve_pro(history):
    """Visualisasi 4: Learning Curve Model Terbaik"""
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)
    
    best_epoch = val_acc.index(max(val_acc)) + 1
    best_val_acc = max(val_acc)

    plt.figure(figsize=(16, 6))

    # Grafik Akurasi
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'o-', label='Training Acc', color='tab:blue')
    plt.plot(epochs, val_acc, 's--', label='Validation Acc', color='tab:orange')
    plt.annotate(f'Best: {best_val_acc*100:.2f}%',
                 xy=(best_epoch, best_val_acc), xycoords='data',
                 xytext=(best_epoch, best_val_acc - 0.05), textcoords='data',
                 arrowprops=dict(facecolor='green', shrink=0.05),
                 horizontalalignment='center', verticalalignment='top',
                 fontsize=12, fontweight='bold', color='green')
    plt.title('Learning Curve: Akurasi')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Grafik Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'o-', label='Training Loss', color='tab:red')
    plt.plot(epochs, val_loss, 's--', label='Validation Loss', color='tab:green')
    plt.title('Learning Curve: Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_wordcloud(df):
    """Visualisasi 5: Word Cloud"""
    plt.figure(figsize=(16, 8))
    
    # Positif
    subset_pos = df[df['label'] == 1]['clean_ulasan']
    if len(subset_pos) > 0:
        text_pos = ' '.join(subset_pos.tolist())
        wc_pos = WordCloud(width=800, height=400, background_color='white', colormap='Greens').generate(text_pos)
        plt.subplot(1, 2, 1)
        plt.imshow(wc_pos, interpolation='bilinear')
        plt.title('Word Cloud - Sentimen POSITIF')
        plt.axis('off')
    
    # Negatif
    subset_neg = df[df['label'] == 0]['clean_ulasan']
    if len(subset_neg) > 0:
        text_neg = ' '.join(subset_neg.tolist())
        wc_neg = WordCloud(width=800, height=400, background_color='white', colormap='Reds').generate(text_neg)
        plt.subplot(1, 2, 2)
        plt.imshow(wc_neg, interpolation='bilinear')
        plt.title('Word Cloud - Sentimen NEGATIF')
        plt.axis('off')
    
    plt.show()


# ==========================================
# EKSEKUSI PROGRAM UTAMA
# ==========================================

if __name__ == "__main__":
    # 1. SCRAPING
    df_raw = scrape_google_play_data(APP_ID, count=JUMLAH_DATA)
    df_clean = preprocess_and_label(df_raw)

    # 2. TOKENIZING
    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token='<OOV>')
    tokenizer.fit_on_texts(df_clean['clean_ulasan'])
    sequences = tokenizer.texts_to_sequences(df_clean['clean_ulasan'])
    X = pad_sequences(sequences, maxlen=MAX_LEN, padding='post', truncating='post')
    y = df_clean['label'].values

    # 3. SPLIT DATA
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

    # 4. DATA BALANCING (SMOTE)
    print("\n[INFO] Menyiapkan data SMOTE...")
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    # >>> VISUALISASI 1: Cek Distribusi
    plot_class_distribution(y_train, y_train_smote)

    # ---------------------------------------------------------
    # SKENARIO 1: MODEL TANPA SMOTE (DATA ASLI/TIMPANG)
    # ---------------------------------------------------------
    print("\n--- [SCENARIO 1] Melatih Model TANPA SMOTE ---")
    model_no = build_cnn_model()
    history_no = model_no.fit(
        X_train, y_train, 
        epochs=10, validation_data=(X_test, y_test), verbose=1, batch_size=32,
        callbacks=[EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)]
    )

    # ---------------------------------------------------------
    # SKENARIO 2: MODEL DENGAN SMOTE (DATA SEIMBANG)
    # ---------------------------------------------------------
    print("\n--- [SCENARIO 2] Melatih Model DENGAN SMOTE ---")
    model_smote = build_cnn_model() # Reset model baru
    
    callbacks_smote = [
        EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
        ModelCheckpoint('best_model_smote.h5', monitor='val_accuracy', save_best_only=True)
    ]
    
    history_smote = model_smote.fit(
        X_train_smote, y_train_smote,
        epochs=15, validation_data=(X_test, y_test), verbose=1, batch_size=32,
        callbacks=callbacks_smote
    )

    # ---------------------------------------------------------
    # EVALUASI & KOMPARASI
    # ---------------------------------------------------------
    # Prediksi Skenario 1
    y_pred_no = (model_no.predict(X_test) > 0.5).astype(int)
    # Prediksi Skenario 2
    y_pred_smote = (model_smote.predict(X_test) > 0.5).astype(int)

    # Hitung Metrik Komparasi
    metrics_no = [
        accuracy_score(y_test, y_pred_no),
        recall_score(y_test, y_pred_no, pos_label=0), # Recall Negatif
        f1_score(y_test, y_pred_no, pos_label=0)      # F1 Negatif
    ]
    metrics_smote = [
        accuracy_score(y_test, y_pred_smote),
        recall_score(y_test, y_pred_smote, pos_label=0),
        f1_score(y_test, y_pred_smote, pos_label=0)
    ]

    print("\n=== HASIL AKHIR KOMPARASI ===")
    print(f"Recall Kelas Negatif (Tanpa SMOTE): {metrics_no[1]:.4f}")
    print(f"Recall Kelas Negatif (Dengan SMOTE): {metrics_smote[1]:.4f}")

    # >>> VISUALISASI 2 & 3: Komparasi Head-to-Head
    plot_side_by_side_confusion_matrix(y_test, y_pred_no, y_pred_smote)
    plot_metric_comparison(metrics_no, metrics_smote)

    # ---------------------------------------------------------
    # VISUALISASI MENDALAM (KHUSUS MODEL TERBAIK/SMOTE)
    # ---------------------------------------------------------
    print("\n[INFO] Menampilkan visualisasi detail untuk Model Terbaik...")
    
    # >>> VISUALISASI 4: Learning Curve
    plot_learning_curve_pro(history_smote)
    
    # >>> VISUALISASI 5: Word Cloud
    plot_wordcloud(df_clean)
    
    
    print("\n[INFO] Selesai. Laporan Klasifikasi Model Terbaik:")
    print(classification_report(y_test, y_pred_smote, target_names=['Negatif', 'Positif']))

