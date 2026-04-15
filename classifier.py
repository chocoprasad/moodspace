"""
classifier.py — MoodSpace v5 Fixed
"""
import numpy as np
import json, os

class MoodClassifier:
    MOODS = ['Focused', 'Relaxed', 'Stressed', 'Fatigued']

    def __init__(self, tflite_path='mood_model.tflite',
                 rf_path='rf_model.pkl', scaler_path='scaler_params.json'):
        self._interpreter = None
        self._rf = None
        self._scaler_mean = None
        self._scaler_std  = None

        if os.path.exists(scaler_path):
            with open(scaler_path) as f: p = json.load(f)
            self._scaler_mean = np.array(p['mean'],  dtype=np.float32)
            self._scaler_std  = np.array(p['scale'], dtype=np.float32)
            print(f"[Classifier] Scaler loaded ✓ ({len(self._scaler_mean)} features)")

        if os.path.exists(tflite_path):
            try:
                try:
                    import tflite_runtime.interpreter as tflite
                    self._interpreter = tflite.Interpreter(model_path=tflite_path)
                except ImportError:
                    import tensorflow as tf
                    self._interpreter = tf.lite.Interpreter(model_path=tflite_path)
                self._interpreter.allocate_tensors()
                self._inp = self._interpreter.get_input_details()
                self._out = self._interpreter.get_output_details()
                print(f"[Classifier] TFLite loaded ✓ input:{self._inp[0]['shape']}")
            except Exception as e:
                print(f"[Classifier] TFLite error: {e}")

        if os.path.exists(rf_path):
            try:
                import pickle
                with open(rf_path,'rb') as f: bundle = pickle.load(f)
                self._rf = bundle['model'] if isinstance(bundle,dict) else bundle
                print("[Classifier] RF loaded ✓")
            except Exception as e:
                print(f"[Classifier] RF error: {e}")

        mode = 'TFLite' if self._interpreter else ('RF' if self._rf else 'Rule-based')
        print(f"[Classifier] Mode: {mode}")

    def _engineer(self, hr, temp, gsr):
        f = np.array([[
            hr, temp, gsr,
            hr/70, gsr/300,
            hr*gsr/10000,
            temp-33,
            (hr-70)*(gsr-200)/1000,
            1.0 if hr>90   else 0.0,
            1.0 if gsr>500 else 0.0,
            1.0 if hr<62   else 0.0,
            1.0 if temp>35 else 0.0,
        ]], dtype=np.float32)
        if self._scaler_mean is not None:
            f = (f - self._scaler_mean) / self._scaler_std
        return f

    def predict(self, hr, temp, gsr):
        f = self._engineer(hr, temp, gsr)
        if self._interpreter:
            try:
                self._interpreter.set_tensor(self._inp[0]['index'], f)
                self._interpreter.invoke()
                p = self._interpreter.get_tensor(self._out[0]['index'])[0]
                i = int(np.argmax(p))
                return self.MOODS[i], round(float(p[i]),4)
            except Exception as e:
                print(f"[Classifier] TFLite predict error: {e}")
        if self._rf:
            try:
                p = self._rf.predict_proba(f)[0]
                i = int(np.argmax(p))
                return self.MOODS[i], round(float(p[i]),4)
            except: pass
        return self._rule_based(hr, temp, gsr), 1.0

    def _rule_based(self, hr, temp, gsr):
        s = {m:0 for m in self.MOODS}
        if 75<=hr<=90:       s['Focused']+=35
        if 60<=hr<75:        s['Relaxed']+=35
        if hr>90:            s['Stressed']+=40
        if hr<60:            s['Fatigued']+=35
        if 33.5<=temp<=35:   s['Focused']+=20
        if 32.5<=temp<33.5:  s['Relaxed']+=25
        if temp>35:          s['Stressed']+=30
        if temp<32:          s['Fatigued']+=20
        if gsr<250:          s['Relaxed']+=30; s['Fatigued']+=15
        if 250<=gsr<500:     s['Focused']+=30
        if gsr>=500:         s['Stressed']+=35
        return max(s, key=s.get)

if __name__ == '__main__':
    clf = MoodClassifier()
    print()
    tests = [(83,34.2,360,'Focused'),(67,33.1,190,'Relaxed'),
             (105,35.8,630,'Stressed'),(55,31.5,155,'Fatigued')]
    ok = 0
    for hr,temp,gsr,exp in tests:
        mood,conf = clf.predict(hr,temp,gsr)
        tick = '✓' if mood==exp else '✗'
        print(f"  {tick} HR:{hr} Temp:{temp} GSR:{gsr} → {mood} ({conf*100:.1f}%) [expected {exp}]")
        if mood==exp: ok+=1
    print(f"\nResult: {ok}/{len(tests)} correct")
