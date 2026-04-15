"""
MoodSpace — LSTM Mood Classifier Training (Fixed)
==================================================
Fixes class imbalance issue from previous training.
Run: python train_mood_model_v2.py
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os, json

print("TensorFlow:", tf.__version__)
print("="*50)

# ── CONFIG ────────────────────────────────────────────────────
WINDOW_SIZE = 30
EPOCHS      = 80
BATCH_SIZE  = 64
TEST_SIZE   = 0.2
TFLITE_PATH = 'mood_model.tflite'
MODEL_PATH  = 'mood_model.h5'
DATA_CSV    = 'mood_data.csv'
MOODS       = ['Focused', 'Relaxed', 'Stressed', 'Fatigued']

# ── Mood profiles — tighter, more distinct ranges ─────────────
MOOD_PROFILES = {
    'Focused':  {'hr':(78,88,4),  'temp':(33.8,34.8,0.3), 'gsr':(280,480,40)},
    'Relaxed':  {'hr':(62,72,3),  'temp':(32.8,33.4,0.25),'gsr':(140,240,35)},
    'Stressed': {'hr':(95,125,7), 'temp':(35.2,37.5,0.4), 'gsr':(520,780,70)},
    'Fatigued': {'hr':(50,60,4),  'temp':(30.8,31.8,0.35),'gsr':(110,195,28)},
}

# ── Generate balanced synthetic data ─────────────────────────
def generate_data(n_per_mood=3000):
    print("\n[1/6] Generating balanced synthetic dataset...")
    rows = []

    for mood, p in MOOD_PROFILES.items():
        for _ in range(n_per_mood):
            hr   = np.random.uniform(p['hr'][0],   p['hr'][1])   + np.random.normal(0, p['hr'][2])
            temp = np.random.uniform(p['temp'][0],  p['temp'][1]) + np.random.normal(0, p['temp'][2])
            gsr  = np.random.uniform(p['gsr'][0],   p['gsr'][1]) + np.random.normal(0, p['gsr'][2])

            # Sensor noise
            hr   += np.random.normal(0, 1.2)
            temp += np.random.normal(0, 0.2)
            gsr  += np.random.normal(0, 12)

            # Motion artifact (8% chance)
            if np.random.random() < 0.08:
                gsr += np.random.uniform(30, 120)

            rows.append({
                'hr':   np.clip(hr,   40, 150),
                'temp': np.clip(temp, 28, 40),
                'gsr':  np.clip(gsr,  50, 900),
                'mood': mood,
            })

    # Smooth mood transitions — prevents sharp boundary bias
    transitions = [
        ('Relaxed', 'Focused'),
        ('Focused', 'Stressed'),
        ('Stressed', 'Fatigued'),
        ('Fatigued', 'Relaxed'),
    ]
    for m_from, m_to in transitions:
        pf = MOOD_PROFILES[m_from]
        pt = MOOD_PROFILES[m_to]
        for _ in range(n_per_mood // 6):
            for t in range(WINDOW_SIZE):
                frac = t / WINDOW_SIZE
                hr   = (1-frac)*np.mean(pf['hr'][:2])   + frac*np.mean(pt['hr'][:2])   + np.random.normal(0,2)
                temp = (1-frac)*np.mean(pf['temp'][:2]) + frac*np.mean(pt['temp'][:2]) + np.random.normal(0,0.2)
                gsr  = (1-frac)*np.mean(pf['gsr'][:2])  + frac*np.mean(pt['gsr'][:2])  + np.random.normal(0,15)
                mood = m_to if frac > 0.65 else m_from
                rows.append({'hr':round(hr,1), 'temp':round(temp,2), 'gsr':round(gsr,1), 'mood':mood})

    df = pd.DataFrame(rows).sample(frac=1).reset_index(drop=True)
    df.to_csv(DATA_CSV, index=False)
    print(f"    Total samples: {len(df)}")
    print(f"    Distribution:\n{df['mood'].value_counts().to_string()}")
    return df

df = generate_data(3000)

# ── Preprocess ────────────────────────────────────────────────
print("\n[2/6] Preprocessing...")

le = LabelEncoder()
le.fit(MOODS)
df['label'] = le.transform(df['mood'])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[['hr','temp','gsr']].values)
y_raw    = df['label'].values

# Save scaler
with open('scaler_params.json','w') as f:
    json.dump({'mean':scaler.mean_.tolist(),
               'scale':scaler.scale_.tolist(),
               'classes':MOODS}, f, indent=2)

# Sliding windows — CRITICAL FIX: stride=5 reduces window overlap
# Previous version: stride=1 → massive Stressed overrepresentation
def create_windows(X, y, window_size, stride=5):
    Xw, yw = [], []
    for i in range(0, len(X)-window_size+1, stride):
        Xw.append(X[i:i+window_size])
        yw.append(np.bincount(y[i:i+window_size]).argmax())
    return np.array(Xw, dtype=np.float32), np.array(yw)

X_win, y_win = create_windows(X_scaled, y_raw, WINDOW_SIZE, stride=5)
print(f"    Windows: {X_win.shape}")
print(f"    Window distribution: {dict(zip(MOODS, np.bincount(y_win)))}")

# One-hot encode
y_cat = tf.keras.utils.to_categorical(y_win, 4)

# Train/test split — stratified
X_tr, X_te, y_tr, y_te = train_test_split(
    X_win, y_cat, test_size=TEST_SIZE,
    random_state=42, stratify=y_win
)
print(f"    Train: {X_tr.shape[0]} · Test: {X_te.shape[0]}")

# ── Class weights — KEY FIX for imbalance ─────────────────────
y_int = np.argmax(y_tr, axis=1)
cw = compute_class_weight('balanced', classes=np.unique(y_int), y=y_int)
class_weights = dict(enumerate(cw))
print(f"\n    Class weights: {class_weights}")

# ── Build improved model ──────────────────────────────────────
print("\n[3/6] Building model...")

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(WINDOW_SIZE, 3)),

    # Bidirectional LSTM — reads sequence forward AND backward
    tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(64, return_sequences=True, dropout=0.2)
    ),
    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(32, return_sequences=False, dropout=0.2)
    ),
    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.35),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(4, activation='softmax'),
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy'],
)
model.summary()

# ── Train with class weights ──────────────────────────────────
print(f"\n[4/6] Training {EPOCHS} epochs with class balancing...")

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy', patience=12,
        restore_best_weights=True, verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.4,
        patience=6, min_lr=1e-6, verbose=1
    ),
]

history = model.fit(
    X_tr, y_tr,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.15,
    class_weight=class_weights,   # ← fixes the bias
    callbacks=callbacks,
    verbose=1,
)

# ── Evaluate ──────────────────────────────────────────────────
print("\n[5/6] Evaluating...")
loss, acc = model.evaluate(X_te, y_te, verbose=0)
print(f"\n    Accuracy: {acc*100:.2f}%")

y_pred = np.argmax(model.predict(X_te, verbose=0), axis=1)
y_true = np.argmax(y_te, axis=1)
print("\n" + classification_report(y_true, y_pred, target_names=MOODS))

# Plot
cm = confusion_matrix(y_true, y_pred)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=MOODS, yticklabels=MOODS, ax=axes[0])
axes[0].set_title(f'Confusion Matrix v2 (acc={acc*100:.1f}%)')
axes[0].set_ylabel('True'); axes[0].set_xlabel('Predicted')

axes[1].plot(history.history['accuracy'],     label='Train acc')
axes[1].plot(history.history['val_accuracy'], label='Val acc')
axes[1].plot(history.history['loss'],         label='Train loss', linestyle='--')
axes[1].plot(history.history['val_loss'],     label='Val loss',   linestyle='--')
axes[1].set_title('Training History v2')
axes[1].set_xlabel('Epoch'); axes[1].legend(); axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig('training_plot_v2.png', dpi=120, bbox_inches='tight')
print("    Plot saved → training_plot_v2.png")

# ── Export TFLite ─────────────────────────────────────────────
print("\n[6/6] Exporting TFLite...")
model.save(MODEL_PATH)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite = converter.convert()
with open(TFLITE_PATH, 'wb') as f:
    f.write(tflite)

print(f"\n{'='*50}")
print(f"  Accuracy:  {acc*100:.2f}%")
print(f"  Model:     {TFLITE_PATH} ({len(tflite)/1024:.1f} KB)")
print(f"  Scaler:    scaler_params.json")
print(f"  Plot:      training_plot_v2.png")
print(f"{'='*50}")
print("\nKey fixes applied:")
print("  ✓ Balanced class weights")
print("  ✓ Stride=5 windows (reduces overlap bias)")
print("  ✓ Bidirectional LSTM")
print("  ✓ Tighter mood separation in profiles")
print("  ✓ Mood transition sequences")
