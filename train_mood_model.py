"""
MoodSpace — LSTM Mood Classifier Training
==========================================
Run in WSL:
    python3 train_mood_model.py

What this does:
    1. Generates synthetic biometric dataset (or loads real CSV)
    2. Creates 30-sample sliding windows
    3. Trains LSTM model
    4. Evaluates accuracy + confusion matrix
    5. Exports mood_model.tflite for Raspberry Pi

Output files:
    mood_model.tflite   → copy to Pi, load in classifier.py
    mood_model.h5       → full Keras model for further training
    training_plot.png   → accuracy/loss curves
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')  # WSL has no display — save plots to file
import matplotlib.pyplot as plt
import seaborn as sns
import os, json

print("TensorFlow version:", tf.__version__)
print("="*50)

# ── CONFIG — edit these ───────────────────────────────────────
WINDOW_SIZE   = 30      # samples per window (30 = 3 seconds at 10Hz)
EPOCHS        = 60      # training epochs
BATCH_SIZE    = 32
TEST_SIZE     = 0.2     # 20% for testing
MODEL_PATH    = 'mood_model.h5'
TFLITE_PATH   = 'mood_model.tflite'
DATA_CSV      = 'mood_data.csv'  # if exists, use it; else generate synthetic

MOODS = ['Focused', 'Relaxed', 'Stressed', 'Fatigued']

# ── Mood biometric ranges ─────────────────────────────────────
# Edit these to match YOUR body's sensor readings
MOOD_PROFILES = {
    'Focused': {
        'hr':   (75, 90,  5),   # (mean, max, std)
        'temp': (33.5, 35, 0.4),
        'gsr':  (300, 500, 50),
    },
    'Relaxed': {
        'hr':   (62, 75,  4),
        'temp': (33.0, 33.5, 0.3),
        'gsr':  (150, 250, 40),
    },
    'Stressed': {
        'hr':   (92, 130, 8),
        'temp': (35.0, 38, 0.5),
        'gsr':  (500, 800, 80),
    },
    'Fatigued': {
        'hr':   (48, 60,  5),
        'temp': (30.5, 32, 0.4),
        'gsr':  (100, 200, 30),
    },
}

# ── Step 1: Generate or load dataset ─────────────────────────
def generate_synthetic_data(n_samples_per_mood=5000):
    rows = []
    for mood, profile in MOOD_PROFILES.items():
        for _ in range(n_samples_per_mood):
            hr   = np.random.normal(
                (profile['hr'][0]+profile['hr'][1])/2,
                profile['hr'][2])
            temp = np.random.normal(
                (profile['temp'][0]+profile['temp'][1])/2,
                profile['temp'][2])
            gsr  = np.random.normal(
                (profile['gsr'][0]+profile['gsr'][1])/2,
                profile['gsr'][2])

            # Add real sensor noise
            hr   += np.random.normal(0, 1.5)   # MAX30102 ±1.5bpm
            temp += np.random.normal(0, 0.3)   # LM35 ±0.3°C
            gsr  += np.random.normal(0, 15)    # GSR noise

            # Random motion artifact on GSR (10% chance)
            if np.random.random() < 0.1:
                gsr += np.random.uniform(50, 200)

            hr   = np.clip(hr,   40, 150)
            temp = np.clip(temp, 28, 40)
            gsr  = np.clip(gsr,  50, 900)
            rows.append({'hr':round(hr,1),
                         'temp':round(temp,2),
                         'gsr':round(gsr,1),
                         'mood':mood})

    # Add mood transitions (Relaxed → Stressed)
    for _ in range(n_samples_per_mood // 5):
        for t in range(30):
            frac = t / 30
            hr   = 65 + frac * 35 + np.random.normal(0,2)
            temp = 33 + frac * 2.5 + np.random.normal(0,0.2)
            gsr  = 200 + frac * 400 + np.random.normal(0,20)
            mood = 'Stressed' if frac > 0.6 else 'Relaxed'
            rows.append({'hr':round(hr,1),
                         'temp':round(temp,2),
                         'gsr':round(gsr,1),
                         'mood':mood})

    df = pd.DataFrame(rows).sample(frac=1).reset_index(drop=True)
    df.to_csv(DATA_CSV, index=False)
    print(f"Generated {len(df)} samples with noise + transitions")
    return df

def load_real_data(csv_path):
    """
    Load real sensor data from your Flask backend CSV export.
    Expected columns: hr, temp, gsr, mood
    Also handles: heart_rate, skin_temp, skin_conductance
    """
    df = pd.read_csv(csv_path)
    # Normalize column names
    rename_map = {
        'heart_rate': 'hr', 'skin_temp': 'temp',
        'skin_conductance': 'gsr', 'label': 'mood',
    }
    df = df.rename(columns=rename_map)
    df = df[['hr', 'temp', 'gsr', 'mood']].dropna()
    df = df[df['mood'].isin(MOODS)]
    print(f"    Loaded {len(df)} real samples from {csv_path}")
    return df

# Load or generate
if os.path.exists(DATA_CSV):
    print(f"\n[1/6] Found {DATA_CSV} — loading real data...")
    df = load_real_data(DATA_CSV)
    if len(df) < 200:
        print("    Too few real samples — augmenting with synthetic...")
        df_syn = generate_synthetic_data(200)
        df = pd.concat([df, df_syn], ignore_index=True)
else:
  df = generate_synthetic_data(5000)

# ── Step 2: Preprocess ────────────────────────────────────────
print("\n[2/6] Preprocessing data...")

# Encode mood labels
le = LabelEncoder()
le.fit(MOODS)
df['label'] = le.transform(df['mood'])
print(f"    Label encoding: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# Normalize features
scaler = StandardScaler()
features = df[['hr', 'temp', 'gsr']].values
features_scaled = scaler.fit_transform(features)

# Save scaler params for Pi inference
scaler_params = {
    'mean': scaler.mean_.tolist(),
    'scale': scaler.scale_.tolist(),
    'classes': MOODS,
}
with open('scaler_params.json', 'w') as f:
    json.dump(scaler_params, f, indent=2)
print("    Scaler params saved → scaler_params.json")

# Create sliding windows
def create_windows(X, y, window_size):
    X_win, y_win = [], []
    for i in range(len(X) - window_size + 1):
        X_win.append(X[i:i+window_size])
        # Label = most common mood in window
        y_win.append(np.bincount(y[i:i+window_size]).argmax())
    return np.array(X_win), np.array(y_win)

labels = df['label'].values
X_win, y_win = create_windows(features_scaled, labels, WINDOW_SIZE)
print(f"    Windows created: {X_win.shape} → {len(X_win)} samples of {WINDOW_SIZE} timesteps × 3 features")

# One-hot encode labels
y_cat = tf.keras.utils.to_categorical(y_win, num_classes=4)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_win, y_cat, test_size=TEST_SIZE, random_state=42, stratify=y_win
)
print(f"    Train: {X_train.shape[0]} · Test: {X_test.shape[0]}")

# ── Step 3: Build LSTM model ──────────────────────────────────
print("\n[3/6] Building LSTM model...")

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(WINDOW_SIZE, 3)),

    # First LSTM layer — extracts temporal patterns
    tf.keras.layers.LSTM(64, return_sequences=True,
                         dropout=0.2, recurrent_dropout=0.1),
    tf.keras.layers.BatchNormalization(),

    # Second LSTM layer — higher-level patterns
    tf.keras.layers.LSTM(32, return_sequences=False,
                         dropout=0.2, recurrent_dropout=0.1),
    tf.keras.layers.BatchNormalization(),

    # Dense layers
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(16, activation='relu'),

    # Output — 4 mood classes
    tf.keras.layers.Dense(4, activation='softmax'),
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy'],
)

model.summary()
print(f"\n    Parameters: {model.count_params():,}")

# ── Step 4: Train ─────────────────────────────────────────────
print(f"\n[4/6] Training for {EPOCHS} epochs...")

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy', patience=10,
        restore_best_weights=True, verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5,
        patience=5, min_lr=1e-6, verbose=1
    ),
]

history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.15,
    callbacks=callbacks,
    verbose=1,
)

# ── Step 5: Evaluate ──────────────────────────────────────────
print("\n[5/6] Evaluating model...")

loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\n    Test accuracy: {accuracy*100:.2f}%")
print(f"    Test loss:     {loss:.4f}")

y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
y_true = np.argmax(y_test, axis=1)

print("\n    Classification report:")
print(classification_report(y_true, y_pred, target_names=MOODS))

# Confusion matrix plot
cm = confusion_matrix(y_true, y_pred)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=MOODS, yticklabels=MOODS, ax=axes[0])
axes[0].set_title(f'Confusion Matrix (acc={accuracy*100:.1f}%)')
axes[0].set_ylabel('True'); axes[0].set_xlabel('Predicted')

axes[1].plot(history.history['accuracy'],    label='Train acc')
axes[1].plot(history.history['val_accuracy'], label='Val acc')
axes[1].plot(history.history['loss'],        label='Train loss', linestyle='--')
axes[1].plot(history.history['val_loss'],    label='Val loss',   linestyle='--')
axes[1].set_title('Training History')
axes[1].set_xlabel('Epoch'); axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('training_plot.png', dpi=120, bbox_inches='tight')
print("\n    Plot saved → training_plot.png")

# ── Step 6: Export to TFLite ──────────────────────────────────
print("\n[6/6] Exporting to TFLite...")

model.save(MODEL_PATH)
print(f"    Keras model saved → {MODEL_PATH}")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # quantize = smaller + faster on Pi
tflite_model = converter.convert()

with open(TFLITE_PATH, 'wb') as f:
    f.write(tflite_model)

size_kb = len(tflite_model) / 1024
print(f"    TFLite model saved → {TFLITE_PATH} ({size_kb:.1f} KB)")

print("\n" + "="*50)
print("TRAINING COMPLETE")
print("="*50)
print(f"  Accuracy:      {accuracy*100:.2f}%")
print(f"  TFLite model:  {TFLITE_PATH}  ({size_kb:.1f} KB)")
print(f"  Scaler params: scaler_params.json")
print(f"  Training plot: training_plot.png")
print()
print("Next steps:")
print("  1. Copy mood_model.tflite to your Pi")
print("  2. Copy scaler_params.json to your Pi")
print("  3. Update classifier.py to use TFLite inference")
print("  4. Replace rule-based classifier in app.py")
print("="*50)
