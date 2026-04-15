"""
MoodSpace — Mood Classifier v4 (CNN + LSTM hybrid)
====================================================
Why this works better:
  - CNN extracts local features first (pattern within 5 samples)
  - LSTM then learns temporal patterns across those features
  - Much faster convergence than pure LSTM on synthetic data
  - TFLite compatible — no Bidirectional issues
  - Target: 90%+ accuracy on all 4 moods

Run: python train_mood_model_v4.py
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import json

print("TensorFlow:", tf.__version__)
print("="*55)

# ── CONFIG ────────────────────────────────────────────────────
WINDOW_SIZE = 30
EPOCHS      = 100
BATCH_SIZE  = 64
N_PER_MOOD  = 4000     # 4000 × 4 moods = 16,000 samples
MOODS       = ['Focused', 'Relaxed', 'Stressed', 'Fatigued']
TFLITE_PATH = 'mood_model.tflite'
MODEL_PATH  = 'mood_model.keras'

# ── Well-separated biometric profiles ────────────────────────
# Key: make ranges non-overlapping for strong signal
PROFILES = {
    'Focused': {
        'hr':   (76, 90),   # clearly above resting
        'temp': (33.6, 34.8),
        'gsr':  (280, 470),  # moderate arousal
    },
    'Relaxed': {
        'hr':   (60, 73),   # clearly low
        'temp': (32.5, 33.5),
        'gsr':  (100, 230),  # clearly low
    },
    'Stressed': {
        'hr':   (93, 125),  # clearly high
        'temp': (35.0, 37.5),
        'gsr':  (510, 800),  # clearly high
    },
    'Fatigued': {
        'hr':   (48, 61),   # lowest
        'temp': (30.5, 31.8),# lowest
        'gsr':  (80, 190),   # lowest
    },
}

# ── Step 1: Generate data ─────────────────────────────────────
print(f"\n[1/6] Generating {N_PER_MOOD*4:,} samples...")

def generate():
    rows = []

    # Core samples — one mood at a time, exactly N_PER_MOOD each
    for label, (mood, p) in enumerate(PROFILES.items()):
        for _ in range(N_PER_MOOD):
            hr   = np.random.uniform(*p['hr'])   + np.random.normal(0, 1.5)
            temp = np.random.uniform(*p['temp']) + np.random.normal(0, 0.2)
            gsr  = np.random.uniform(*p['gsr'])  + np.random.normal(0, 20)
            # Random motion spike (6%)
            if np.random.random() < 0.06:
                gsr += np.random.uniform(20, 80)
            rows.append([
                np.clip(hr,   42, 148),
                np.clip(temp, 29,  39),
                np.clip(gsr,  50, 880),
                label
            ])

    # Transition sequences — Relaxed→Stressed, Focused→Fatigued etc.
    pairs = [('Relaxed','Stressed'), ('Stressed','Fatigued'),
             ('Fatigued','Focused'), ('Focused','Relaxed')]
    for m_from, m_to in pairs:
        pf, pt = PROFILES[m_from], PROFILES[m_to]
        lf, lt = MOODS.index(m_from), MOODS.index(m_to)
        n_seq = N_PER_MOOD // 5
        for _ in range(n_seq):
            for t in range(WINDOW_SIZE):
                frac = t / WINDOW_SIZE
                hr   = np.interp(frac,[0,1],[np.mean(pf['hr']),  np.mean(pt['hr'])])  + np.random.normal(0,2)
                temp = np.interp(frac,[0,1],[np.mean(pf['temp']),np.mean(pt['temp'])]) + np.random.normal(0,.2)
                gsr  = np.interp(frac,[0,1],[np.mean(pf['gsr']), np.mean(pt['gsr'])])  + np.random.normal(0,18)
                label = lt if frac > 0.65 else lf
                rows.append([np.clip(hr,42,148), np.clip(temp,29,39), np.clip(gsr,50,880), label])

    arr = np.array(rows)
    np.random.shuffle(arr)
    return arr

data = generate()
X_raw = data[:, :3].astype(np.float32)
y_raw = data[:,  3].astype(np.int32)

counts = np.bincount(y_raw, minlength=4)
print(f"    Total: {len(data):,}")
print(f"    Balance: { {MOODS[i]:int(counts[i]) for i in range(4)} }")

# ── Step 2: Normalize ─────────────────────────────────────────
print("\n[2/6] Normalizing + windowing...")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

with open('scaler_params.json','w') as f:
    json.dump({'mean': scaler.mean_.tolist(),
               'scale': scaler.scale_.tolist(),
               'classes': MOODS}, f, indent=2)
print("    scaler_params.json saved")

# Create windows with stride=10 — maximum class balance
def make_windows(X, y, win=30, stride=10):
    Xw, yw = [], []
    for i in range(0, len(X)-win+1, stride):
        Xw.append(X[i:i+win])
        yw.append(np.bincount(y[i:i+win], minlength=4).argmax())
    return np.array(Xw, np.float32), np.array(yw, np.int32)

X_win, y_win = make_windows(X_scaled, y_raw)
counts_w = np.bincount(y_win, minlength=4)
print(f"    Windows: {X_win.shape}")
print(f"    Window balance: { {MOODS[i]:int(counts_w[i]) for i in range(4)} }")

y_cat = tf.keras.utils.to_categorical(y_win, 4)
X_tr, X_te, y_tr, y_te = train_test_split(
    X_win, y_cat, test_size=0.2,
    random_state=42, stratify=y_win)
print(f"    Train: {len(X_tr):,}  Test: {len(X_te):,}")

# ── Step 3: CNN + LSTM model (best of both) ───────────────────
print("\n[3/6] Building CNN+LSTM model...")

inp = tf.keras.Input(shape=(WINDOW_SIZE, 3), name='sensor_input')

# CNN block — extracts local temporal features
x = tf.keras.layers.Conv1D(64, kernel_size=5, activation='relu', padding='same')(inp)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Conv1D(64, kernel_size=3, activation='relu', padding='same')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
x = tf.keras.layers.Dropout(0.2)(x)

# LSTM block — learns temporal dynamics
x = tf.keras.layers.LSTM(64, return_sequences=True, dropout=0.15)(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.LSTM(32, return_sequences=False, dropout=0.1)(x)
x = tf.keras.layers.BatchNormalization()(x)

# Classifier head
x = tf.keras.layers.Dense(64, activation='relu')(x)
x = tf.keras.layers.Dropout(0.3)(x)
x = tf.keras.layers.Dense(32, activation='relu')(x)
x = tf.keras.layers.Dropout(0.15)(x)
out = tf.keras.layers.Dense(4, activation='softmax', name='mood_output')(x)

model = tf.keras.Model(inp, out)
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy'],
)
model.summary()
print(f"    Parameters: {model.count_params():,}")

# ── Step 4: Train ─────────────────────────────────────────────
print(f"\n[4/6] Training up to {EPOCHS} epochs...")

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=20,
        restore_best_weights=True,
        verbose=1,
        min_delta=0.0005,
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=8,
        min_lr=1e-7,
        verbose=1,
    ),
]

history = model.fit(
    X_tr, y_tr,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.15,
    callbacks=callbacks,
    verbose=1,
)

# ── Step 5: Evaluate ──────────────────────────────────────────
print("\n[5/6] Evaluating...")
loss, acc = model.evaluate(X_te, y_te, verbose=0)
print(f"\n    ✓ Test accuracy: {acc*100:.2f}%")
print(f"    ✓ Test loss:     {loss:.4f}")
print(f"    ✓ Epochs run:    {len(history.history['accuracy'])}")

y_pred = np.argmax(model.predict(X_te, verbose=0), axis=1)
y_true = np.argmax(y_te, axis=1)

print("\n" + classification_report(y_true, y_pred, target_names=MOODS))

print("    Per-class accuracy:")
for i, mood in enumerate(MOODS):
    mask  = (y_true == i)
    pacc  = (y_pred[mask] == i).mean() * 100 if mask.sum() > 0 else 0
    bar   = '█' * int(pacc // 5)
    print(f"      {mood:10s} {pacc:5.1f}% {bar}")

# Plots
cm = confusion_matrix(y_true, y_pred)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=MOODS, yticklabels=MOODS, ax=axes[0])
axes[0].set_title(f'Confusion Matrix v4 — CNN+LSTM  (acc={acc*100:.1f}%)')
axes[0].set_ylabel('True label')
axes[0].set_xlabel('Predicted label')

ep = range(len(history.history['accuracy']))
axes[1].plot(ep, history.history['accuracy'],     label='Train acc',  lw=2)
axes[1].plot(ep, history.history['val_accuracy'], label='Val acc',    lw=2)
axes[1].plot(ep, history.history['loss'],         label='Train loss', lw=1, linestyle='--', alpha=.6)
axes[1].plot(ep, history.history['val_loss'],     label='Val loss',   lw=1, linestyle='--', alpha=.6)
axes[1].axhline(0.9, color='gray', linestyle=':', alpha=.5, label='90% target')
axes[1].set_title(f'Training History — {len(ep)} epochs')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy / Loss')
axes[1].legend()
axes[1].grid(alpha=0.3)
axes[1].set_ylim(0, 1.05)

plt.tight_layout()
plt.savefig('training_plot_v4.png', dpi=130, bbox_inches='tight')
print("\n    Plot saved → training_plot_v4.png")

# ── Step 6: Export TFLite ─────────────────────────────────────
print("\n[6/6] Exporting TFLite...")
model.save(MODEL_PATH)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

try:
    tflite = converter.convert()
    export_type = "Standard"
except Exception:
    print("    Standard failed — using SELECT_TF_OPS...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]
    converter._experimental_lower_tensor_list_ops = False
    tflite = converter.convert()
    export_type = "SELECT_TF_OPS"

with open(TFLITE_PATH, 'wb') as f:
    f.write(tflite)

size_kb = len(tflite) / 1024
print(f"    Export type: {export_type}")
print(f"    Saved → {TFLITE_PATH} ({size_kb:.1f} KB)")

# ── Summary ───────────────────────────────────────────────────
print(f"\n{'='*55}")
print(f"  TRAINING COMPLETE — CNN+LSTM v4")
print(f"{'='*55}")
print(f"  Accuracy:    {acc*100:.2f}%")
print(f"  Epochs:      {len(history.history['accuracy'])}")
print(f"  TFLite size: {size_kb:.1f} KB")
print(f"  Export type: {export_type}")
print(f"{'='*55}")
print(f"\n  Copy these 3 files to your Raspberry Pi:")
print(f"  1. {TFLITE_PATH}")
print(f"  2. scaler_params.json")
print(f"  3. classifier.py")
print(f"{'='*55}")
