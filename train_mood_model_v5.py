"""
MoodSpace — Mood Classifier v5 (Random Forest + MLP)
=====================================================
Why this works:
  Synthetic biometric data = each sample is independent.
  LSTM needs REAL sequential dependencies to work.
  Random Forest + MLP classify per-sample perfectly.
  Expected: 95-99% accuracy.

Run: python train_mood_model_v5.py
Outputs: mood_model.tflite, scaler_params.json, rf_model.pkl
"""

import numpy as np
import pandas as pd
import json
import pickle
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

print("="*55)
print("  MoodSpace Mood Classifier v5 — RF + MLP")
print("="*55)

MOODS = ['Focused', 'Relaxed', 'Stressed', 'Fatigued']
N_PER_MOOD = 5000   # 5000 × 4 = 20,000 samples

# ── Well-separated profiles ───────────────────────────────────
PROFILES = {
    'Focused':  {'hr':(76,90),   'temp':(33.6,34.8), 'gsr':(280,470)},
    'Relaxed':  {'hr':(60,73),   'temp':(32.5,33.5), 'gsr':(100,230)},
    'Stressed': {'hr':(93,125),  'temp':(35.0,37.5), 'gsr':(510,800)},
    'Fatigued': {'hr':(48,61),   'temp':(30.5,31.8), 'gsr':(80,190)},
}

# ── Step 1: Generate data ─────────────────────────────────────
print(f"\n[1/5] Generating {N_PER_MOOD*4:,} samples...")

rows = []
for label_idx, (mood, p) in enumerate(PROFILES.items()):
    for _ in range(N_PER_MOOD):
        hr   = np.random.uniform(*p['hr'])   + np.random.normal(0, 1.5)
        temp = np.random.uniform(*p['temp']) + np.random.normal(0, 0.2)
        gsr  = np.random.uniform(*p['gsr'])  + np.random.normal(0, 20)
        # Motion spike 6%
        if np.random.random() < 0.06:
            gsr += np.random.uniform(20, 80)
        rows.append([
            round(np.clip(hr,   42, 148), 1),
            round(np.clip(temp, 29,  39), 2),
            round(np.clip(gsr,  50, 880), 1),
            label_idx
        ])

# Add engineered features rows
arr = np.array(rows)
np.random.shuffle(arr)
X_raw = arr[:, :3].astype(np.float32)
y_raw = arr[:,  3].astype(np.int32)

print(f"    Balance: { {MOODS[i]: int((y_raw==i).sum()) for i in range(4)} }")

# ── Step 2: Feature engineering ───────────────────────────────
print("\n[2/5] Engineering features...")

def engineer_features(X):
    """
    Add derived features that make mood separation clearer.
    These are the features medical literature uses for stress detection.
    """
    hr, temp, gsr = X[:,0], X[:,1], X[:,2]
    features = np.column_stack([
        hr,                          # raw HR
        temp,                        # raw temp
        gsr,                         # raw GSR
        hr / 70,                     # HR ratio (normalised to resting)
        gsr / 300,                   # GSR ratio
        hr * gsr / 10000,            # HR × GSR interaction (stress indicator)
        temp - 33,                   # temp deviation from baseline
        (hr - 70) * (gsr - 200) / 1000,  # combined arousal score
        np.where(hr > 90, 1, 0),    # high HR binary flag
        np.where(gsr > 500, 1, 0),  # high GSR binary flag
        np.where(hr < 62, 1, 0),    # low HR binary flag
        np.where(temp > 35, 1, 0),  # high temp binary flag
    ])
    return features.astype(np.float32)

X_feat = engineer_features(X_raw)
print(f"    Features: {X_feat.shape[1]} (3 raw + 9 engineered)")

# ── Step 3: Scale + split ─────────────────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_feat)

# Save scaler — only first 3 features (raw sensors) for Pi inference
scaler_raw = StandardScaler()
scaler_raw.fit(X_raw)
with open('scaler_params.json','w') as f:
    json.dump({
        'mean':    scaler_raw.mean_.tolist(),
        'scale':   scaler_raw.scale_.tolist(),
        'classes': MOODS,
        'note':    'Apply to [hr, temp, gsr] before inference'
    }, f, indent=2)
print("    scaler_params.json saved")

X_tr, X_te, y_tr, y_te = train_test_split(
    X_scaled, y_raw, test_size=0.2,
    random_state=42, stratify=y_raw
)
print(f"    Train: {len(X_tr):,}  Test: {len(X_te):,}")

# ── Step 4a: Random Forest ────────────────────────────────────
print("\n[3/5] Training Random Forest...")

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_leaf=2,
    n_jobs=-1,
    random_state=42,
    class_weight='balanced',
)
rf.fit(X_tr, y_tr)
rf_acc = accuracy_score(y_te, rf.predict(X_te))
print(f"    Random Forest accuracy: {rf_acc*100:.2f}%")

# Save RF model
with open('rf_model.pkl', 'wb') as f:
    pickle.dump(rf, f)
print("    rf_model.pkl saved")

# ── Step 4b: MLP (TFLite compatible) ─────────────────────────
print("\n[4/5] Training MLP neural network...")

inp = tf.keras.Input(shape=(X_scaled.shape[1],), name='sensor_input')
x = tf.keras.layers.Dense(256, activation='relu')(inp)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dropout(0.3)(x)
x = tf.keras.layers.Dense(128, activation='relu')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.Dense(64, activation='relu')(x)
x = tf.keras.layers.Dropout(0.1)(x)
x = tf.keras.layers.Dense(32, activation='relu')(x)
out = tf.keras.layers.Dense(4, activation='softmax', name='mood')(x)

mlp = tf.keras.Model(inp, out)
mlp.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy', patience=15,
        restore_best_weights=True, verbose=0
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5,
        patience=7, min_lr=1e-7, verbose=0
    ),
]

history = mlp.fit(
    X_tr, y_tr,
    epochs=100,
    batch_size=64,
    validation_split=0.15,
    callbacks=callbacks,
    verbose=1,
)

_, mlp_acc = mlp.evaluate(X_te, y_te, verbose=0)
print(f"\n    MLP accuracy: {mlp_acc*100:.2f}%")

# ── Step 5: Evaluate both + export ───────────────────────────
print("\n[5/5] Final evaluation...")

y_rf  = rf.predict(X_te)
y_mlp = np.argmax(mlp.predict(X_te, verbose=0), axis=1)

# Ensemble — majority vote RF + MLP
y_ens = np.array([
    np.bincount([y_rf[i], y_mlp[i]], minlength=4).argmax()
    for i in range(len(y_te))
])
ens_acc = accuracy_score(y_te, y_ens)

print(f"\n    Random Forest: {rf_acc*100:.2f}%")
print(f"    MLP:           {mlp_acc*100:.2f}%")
print(f"    Ensemble:      {ens_acc*100:.2f}%  ← use this one")

best_pred = y_ens if ens_acc >= max(rf_acc, mlp_acc) else (y_rf if rf_acc > mlp_acc else y_mlp)
best_acc  = max(ens_acc, rf_acc, mlp_acc)

print(f"\n    Best model: {best_acc*100:.2f}% accuracy")
print("\n" + classification_report(y_te, best_pred, target_names=MOODS))

print("    Per-class accuracy:")
for i, mood in enumerate(MOODS):
    mask = (y_te == i)
    pacc = (best_pred[mask] == i).mean() * 100 if mask.sum() > 0 else 0
    bar  = '█' * int(pacc // 5)
    print(f"      {mood:10s} {pacc:5.1f}%  {bar}")

# Plots
cm = confusion_matrix(y_te, best_pred)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=MOODS, yticklabels=MOODS, ax=axes[0])
axes[0].set_title(f'Confusion Matrix v5 RF+MLP (acc={best_acc*100:.1f}%)')
axes[0].set_ylabel('True'); axes[0].set_xlabel('Predicted')

ep = range(len(history.history['accuracy']))
axes[1].plot(ep, history.history['accuracy'],     label='Train acc', lw=2)
axes[1].plot(ep, history.history['val_accuracy'], label='Val acc',   lw=2)
axes[1].axhline(0.9, color='gray', linestyle=':', alpha=.5, label='90% line')
axes[1].set_title(f'MLP Training ({len(ep)} epochs)')
axes[1].set_xlabel('Epoch'); axes[1].legend(); axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig('training_plot_v5.png', dpi=130, bbox_inches='tight')
print("\n    Plot → training_plot_v5.png")

# Export MLP to TFLite (MLP is always TFLite compatible — no LSTM!)
print("\n    Exporting MLP to TFLite...")
converter = tf.lite.TFLiteConverter.from_keras_model(mlp)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite = converter.convert()
with open('mood_model.tflite', 'wb') as f:
    f.write(tflite)
print(f"    mood_model.tflite saved ({len(tflite)/1024:.1f} KB) — standard export, no SELECT_TF_OPS needed!")

print(f"\n{'='*55}")
print(f"  COMPLETE — v5 RF+MLP")
print(f"{'='*55}")
print(f"  RF accuracy:  {rf_acc*100:.2f}%")
print(f"  MLP accuracy: {mlp_acc*100:.2f}%")
print(f"  Ensemble:     {ens_acc*100:.2f}%")
print(f"  TFLite:       mood_model.tflite (clean export!)")
print(f"{'='*55}")
print(f"\n  Copy to Pi:")
print(f"  1. mood_model.tflite   (MLP — runs on Pi)")
print(f"  2. rf_model.pkl        (Random Forest — faster on Pi)")
print(f"  3. scaler_params.json")
print(f"  4. classifier.py")
print(f"{'='*55}")
