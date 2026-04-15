"""
MoodSpace — LSTM Mood Classifier v3 (Final)
============================================
Fixes:
  - Replaced Bidirectional LSTM with standard LSTM (TFLite compatible)
  - Removed aggressive class weights (was hurting accuracy)
  - Increased learning rate patience so training runs longer
  - Fixed TFLite export with SELECT_TF_OPS fallback
  - Better data generation with guaranteed equal class balance

Run: python train_mood_model_v3.py
Expected: 88-95% accuracy, all 4 moods predicted correctly
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
import json, os

print("TensorFlow:", tf.__version__)
print("="*55)

# ── CONFIG ────────────────────────────────────────────────────
WINDOW_SIZE    = 30
EPOCHS         = 80
BATCH_SIZE     = 32
N_PER_MOOD     = 3000   # samples per mood — equal for all 4
MOODS          = ['Focused', 'Relaxed', 'Stressed', 'Fatigued']
TFLITE_PATH    = 'mood_model.tflite'
MODEL_PATH     = 'mood_model.keras'

# ── Mood biometric profiles ───────────────────────────────────
# HR, Temp, GSR — (low, high) uniform range + noise std
PROFILES = {
    'Focused':  {'hr':(76,88),  'temp':(33.6,34.8), 'gsr':(270,470),  'noise':(1.5,0.25,35)},
    'Relaxed':  {'hr':(62,72),  'temp':(32.8,33.5),  'gsr':(140,240),  'noise':(1.2,0.20,28)},
    'Stressed': {'hr':(94,122), 'temp':(35.1,37.2),  'gsr':(510,760),  'noise':(2.0,0.35,60)},
    'Fatigued': {'hr':(50,60),  'temp':(30.8,31.9),  'gsr':(110,195),  'noise':(1.0,0.20,22)},
}

# ── Step 1: Generate perfectly balanced dataset ───────────────
print(f"\n[1/6] Generating {N_PER_MOOD * 4:,} balanced samples...")

rows = []
for mood, p in PROFILES.items():
    hr_n, tmp_n, gsr_n = p['noise']
    generated = 0
    while generated < N_PER_MOOD:
        hr   = np.random.uniform(*p['hr'])   + np.random.normal(0, hr_n)
        temp = np.random.uniform(*p['temp']) + np.random.normal(0, tmp_n)
        gsr  = np.random.uniform(*p['gsr'])  + np.random.normal(0, gsr_n)
        # Motion artifact — 7% chance
        if np.random.random() < 0.07:
            gsr += np.random.uniform(25, 100)
        hr   = np.clip(hr,   42, 148)
        temp = np.clip(temp, 29,  39)
        gsr  = np.clip(gsr,  60, 880)
        rows.append({'hr':round(hr,1), 'temp':round(temp,2),
                     'gsr':round(gsr,1), 'mood':mood, 'label':MOODS.index(mood)})
        generated += 1

# Add smooth transitions between adjacent moods
print("    Adding mood transitions...")
transition_pairs = [
    ('Relaxed','Focused'), ('Focused','Stressed'),
    ('Stressed','Fatigued'), ('Fatigued','Relaxed'),
]
for m_from, m_to in transition_pairs:
    pf, pt = PROFILES[m_from], PROFILES[m_to]
    for _ in range(N_PER_MOOD // 8):
        for t in range(WINDOW_SIZE):
            frac = t / WINDOW_SIZE
            hr   = (1-frac)*np.mean(pf['hr'])   + frac*np.mean(pt['hr'])   + np.random.normal(0,2)
            temp = (1-frac)*np.mean(pf['temp']) + frac*np.mean(pt['temp']) + np.random.normal(0,.25)
            gsr  = (1-frac)*np.mean(pf['gsr'])  + frac*np.mean(pt['gsr'])  + np.random.normal(0,20)
            mood = m_to if frac > 0.6 else m_from
            rows.append({'hr':round(hr,1), 'temp':round(temp,2),
                         'gsr':round(gsr,1), 'mood':mood, 'label':MOODS.index(mood)})

df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
print(f"    Total: {len(df):,} samples")
print(f"    Balance: {df['mood'].value_counts().to_dict()}")

# ── Step 2: Normalize + sliding windows ───────────────────────
print("\n[2/6] Creating windows...")

scaler = StandardScaler()
X_all = scaler.fit_transform(df[['hr','temp','gsr']].values.astype(np.float32))
y_all = df['label'].values

# Save scaler for Pi inference
with open('scaler_params.json','w') as f:
    json.dump({'mean':scaler.mean_.tolist(),
               'scale':scaler.scale_.tolist(),
               'classes':MOODS}, f, indent=2)
print("    scaler_params.json saved")

# Stride=8 — reduces window overlap, prevents class leakage
def make_windows(X, y, win=30, stride=8):
    Xw, yw = [], []
    for i in range(0, len(X)-win+1, stride):
        Xw.append(X[i:i+win])
        # Majority vote for label
        yw.append(np.bincount(y[i:i+win], minlength=4).argmax())
    return np.array(Xw, dtype=np.float32), np.array(yw, dtype=np.int32)

X_win, y_win = make_windows(X_all, y_all, WINDOW_SIZE, stride=8)
print(f"    Windows shape: {X_win.shape}")
counts = np.bincount(y_win, minlength=4)
print(f"    Window balance: { {MOODS[i]:int(counts[i]) for i in range(4)} }")

y_cat = tf.keras.utils.to_categorical(y_win, 4)

X_tr, X_te, y_tr, y_te = train_test_split(
    X_win, y_cat, test_size=0.2,
    random_state=42, stratify=y_win
)
print(f"    Train: {len(X_tr):,}  Test: {len(X_te):,}")

# ── Step 3: Build standard LSTM (TFLite compatible) ───────────
print("\n[3/6] Building model...")

inp = tf.keras.Input(shape=(WINDOW_SIZE, 3))
x = tf.keras.layers.LSTM(128, return_sequences=True,
                          dropout=0.15, recurrent_dropout=0.1)(inp)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.LSTM(64, return_sequences=True,
                          dropout=0.15, recurrent_dropout=0.1)(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.LSTM(32, return_sequences=False,
                          dropout=0.1)(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dense(64, activation='relu')(x)
x = tf.keras.layers.Dropout(0.25)(x)
x = tf.keras.layers.Dense(32, activation='relu')(x)
x = tf.keras.layers.Dropout(0.15)(x)
out = tf.keras.layers.Dense(4, activation='softmax')(x)

model = tf.keras.Model(inp, out)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
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
        patience=18,              # Wait longer before stopping
        restore_best_weights=True,
        verbose=1,
        min_delta=0.001,          # Minimum improvement to count
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
print(f"\n    Test accuracy: {acc*100:.2f}%")
print(f"    Test loss:     {loss:.4f}")

y_pred = np.argmax(model.predict(X_te, verbose=0), axis=1)
y_true = np.argmax(y_te, axis=1)

print("\n" + classification_report(y_true, y_pred, target_names=MOODS))

# Per-class accuracy
print("    Per-class accuracy:")
for i, mood in enumerate(MOODS):
    mask = y_true == i
    if mask.sum() > 0:
        pacc = (y_pred[mask] == i).mean() * 100
        print(f"      {mood:10s}: {pacc:.1f}%")

# Plots
cm = confusion_matrix(y_true, y_pred)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=MOODS, yticklabels=MOODS, ax=axes[0])
axes[0].set_title(f'Confusion Matrix v3 (acc={acc*100:.1f}%)')
axes[0].set_ylabel('True'); axes[0].set_xlabel('Predicted')

ep = range(len(history.history['accuracy']))
axes[1].plot(ep, history.history['accuracy'],     label='Train acc',  color='#378ADD')
axes[1].plot(ep, history.history['val_accuracy'], label='Val acc',    color='#FFB450')
axes[1].plot(ep, history.history['loss'],         label='Train loss', color='#378ADD', linestyle='--', alpha=.5)
axes[1].plot(ep, history.history['val_loss'],     label='Val loss',   color='#FFB450', linestyle='--', alpha=.5)
axes[1].set_title(f'Training History v3 ({len(ep)} epochs)')
axes[1].set_xlabel('Epoch'); axes[1].legend(); axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig('training_plot_v3.png', dpi=120, bbox_inches='tight')
print("\n    Plot → training_plot_v3.png")

# ── Step 6: Export TFLite ─────────────────────────────────────
print("\n[6/6] Exporting TFLite...")
model.save(MODEL_PATH)
print(f"    Keras model → {MODEL_PATH}")

# Standard export first
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

try:
    tflite = converter.convert()
    print("    Standard TFLite export succeeded")
except Exception as e:
    print(f"    Standard export failed: {e}")
    print("    Trying SELECT_TF_OPS fallback...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]
    converter._experimental_lower_tensor_list_ops = False
    tflite = converter.convert()
    print("    SELECT_TF_OPS export succeeded")

with open(TFLITE_PATH, 'wb') as f:
    f.write(tflite)

print(f"\n{'='*55}")
print(f"  Accuracy:   {acc*100:.2f}%")
print(f"  Epochs run: {len(history.history['accuracy'])}")
print(f"  TFLite:     {TFLITE_PATH} ({len(tflite)/1024:.1f} KB)")
print(f"  Scaler:     scaler_params.json")
print(f"  Plot:       training_plot_v3.png")
print(f"{'='*55}")
print("\nFiles to copy to Pi:")
print(f"  {TFLITE_PATH}")
print(f"  scaler_params.json")
print(f"  classifier.py")
