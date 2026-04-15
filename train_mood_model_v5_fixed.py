"""
train_mood_model_v5_fixed.py
=============================
Same as v5 but saves the FULL 12-feature scaler correctly
so classifier.py gets the right predictions.

Run: python train_mood_model_v5_fixed.py
"""

import numpy as np
import json, pickle
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

print("="*55)
print("  MoodSpace v5 FIXED — correct scaler")
print("="*55)

MOODS      = ['Focused', 'Relaxed', 'Stressed', 'Fatigued']
N_PER_MOOD = 5000

PROFILES = {
    'Focused':  {'hr':(76,90),  'temp':(33.6,34.8),'gsr':(280,470)},
    'Relaxed':  {'hr':(60,73),  'temp':(32.5,33.5),'gsr':(100,230)},
    'Stressed': {'hr':(93,125), 'temp':(35.0,37.5),'gsr':(510,800)},
    'Fatigued': {'hr':(48,61),  'temp':(30.5,31.8),'gsr':(80,190)},
}

# ── Generate data ─────────────────────────────────────────────
print(f"\n[1/5] Generating {N_PER_MOOD*4:,} samples...")
rows = []
for label_idx, (mood, p) in enumerate(PROFILES.items()):
    for _ in range(N_PER_MOOD):
        hr   = np.random.uniform(*p['hr'])   + np.random.normal(0, 1.5)
        temp = np.random.uniform(*p['temp']) + np.random.normal(0, 0.2)
        gsr  = np.random.uniform(*p['gsr'])  + np.random.normal(0, 20)
        if np.random.random() < 0.06:
            gsr += np.random.uniform(20, 80)
        rows.append([
            np.clip(hr,   42, 148),
            np.clip(temp, 29,  39),
            np.clip(gsr,  50, 880),
            label_idx
        ])

arr = np.array(rows); np.random.shuffle(arr)
X_raw = arr[:, :3].astype(np.float32)
y_raw = arr[:,  3].astype(np.int32)
print(f"    Balance: { {MOODS[i]: int((y_raw==i).sum()) for i in range(4)} }")

# ── Engineer features ─────────────────────────────────────────
def engineer(X):
    """12 features from raw [hr, temp, gsr]"""
    hr, temp, gsr = X[:,0], X[:,1], X[:,2]
    return np.column_stack([
        hr,
        temp,
        gsr,
        hr / 70,
        gsr / 300,
        hr * gsr / 10000,
        temp - 33,
        (hr - 70) * (gsr - 200) / 1000,
        (hr > 90).astype(float),
        (gsr > 500).astype(float),
        (hr < 62).astype(float),
        (temp > 35).astype(float),
    ]).astype(np.float32)

X_feat = engineer(X_raw)
print(f"\n[2/5] Features: {X_feat.shape}")

# ── Scale ALL 12 features and save scaler ────────────────────
print("    Fitting scaler on all 12 features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_feat)

# Save scaler for the FULL 12-feature vector
with open('scaler_params.json', 'w') as f:
    json.dump({
        'mean':    scaler.mean_.tolist(),   # 12 values
        'scale':   scaler.scale_.tolist(),  # 12 values
        'classes': MOODS,
        'n_features': 12,
        'raw_mean':  X_raw.mean(axis=0).tolist(),   # for reference
        'raw_scale': X_raw.std(axis=0).tolist(),
    }, f, indent=2)
print("    scaler_params.json saved (12-feature scaler)")

X_tr, X_te, y_tr, y_te = train_test_split(
    X_scaled, y_raw, test_size=0.2, random_state=42, stratify=y_raw)
print(f"    Train: {len(X_tr):,}  Test: {len(X_te):,}")

# ── Random Forest ─────────────────────────────────────────────
print("\n[3/5] Training Random Forest...")
rf = RandomForestClassifier(n_estimators=200, max_depth=20,
                             min_samples_leaf=2, n_jobs=-1, random_state=42)
rf.fit(X_tr, y_tr)
rf_acc = accuracy_score(y_te, rf.predict(X_te))
print(f"    RF accuracy: {rf_acc*100:.2f}%")
with open('rf_model.pkl', 'wb') as f:
    pickle.dump({'model': rf, 'scaler': scaler, 'classes': MOODS}, f)
print("    rf_model.pkl saved (includes scaler)")

# ── MLP ───────────────────────────────────────────────────────
print("\n[4/5] Training MLP...")
y_cat = tf.keras.utils.to_categorical(y_tr, 4)
y_cat_te = tf.keras.utils.to_categorical(y_te, 4)

inp = tf.keras.Input(shape=(12,))
x = tf.keras.layers.Dense(256, activation='relu')(inp)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dropout(0.3)(x)
x = tf.keras.layers.Dense(128, activation='relu')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.Dense(64, activation='relu')(x)
x = tf.keras.layers.Dropout(0.1)(x)
x = tf.keras.layers.Dense(32, activation='relu')(x)
out = tf.keras.layers.Dense(4, activation='softmax')(x)
mlp = tf.keras.Model(inp, out)
mlp.compile(optimizer=tf.keras.optimizers.Adam(0.001),
            loss='categorical_crossentropy', metrics=['accuracy'])

history = mlp.fit(X_tr, y_cat, epochs=100, batch_size=64,
    validation_split=0.15, verbose=1,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',
            patience=15, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
            factor=0.5, patience=7, min_lr=1e-7, verbose=0),
    ])

_, mlp_acc = mlp.evaluate(X_te, y_cat_te, verbose=0)
print(f"    MLP accuracy: {mlp_acc*100:.2f}%")

# ── Evaluate ──────────────────────────────────────────────────
print("\n[5/5] Evaluating...")
y_pred = np.argmax(mlp.predict(X_te, verbose=0), axis=1)
print(classification_report(y_te, y_pred, target_names=MOODS))

for i, mood in enumerate(MOODS):
    mask = (y_te == i)
    pacc = (y_pred[mask] == i).mean() * 100 if mask.sum() > 0 else 0
    print(f"  {mood:10s} {pacc:5.1f}%  {'█'*int(pacc//5)}")

# Plot
cm = confusion_matrix(y_te, y_pred)
fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=MOODS, yticklabels=MOODS, ax=ax)
ax.set_title(f'v5 Fixed (acc={mlp_acc*100:.1f}%)')
plt.tight_layout(); plt.savefig('training_plot_v5_fixed.png', dpi=120)
print("\n    Plot → training_plot_v5_fixed.png")

# Export TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(mlp)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite = converter.convert()
with open('mood_model.tflite', 'wb') as f: f.write(tflite)
print(f"    mood_model.tflite saved ({len(tflite)//1024} KB)")

print(f"\n{'='*55}")
print(f"  MLP: {mlp_acc*100:.2f}%  RF: {rf_acc*100:.2f}%")
print(f"  Scaler: 12-feature (FIXED)")
print(f"{'='*55}")
