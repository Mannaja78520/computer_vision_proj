import os
import pandas as pd
import numpy as np
import keras
from concurrent.futures import ThreadPoolExecutor
from keras import applications

# ==========================================
# CONFIGURATION
# ==========================================
IMG_SIZE = (300, 300)
IMAGE_FOLDER = 'all_train_pic/'
TEST_CSV = 'hidden_test.csv'
OUTPUT_CSV = 'final_submission.csv'
K_FOLDS = 5

# 1. Load All Models
print(f"🚀 Loading {K_FOLDS} models for Ensemble...")
models = []
for i in range(K_FOLDS):
    model_path = f'food_expert_fold_{i+1}.keras'
    if os.path.exists(model_path):
        models.append(keras.models.load_model(model_path))
        print(f"Loaded: {model_path}")

def load_and_preprocess_img(name):
    try:
        path = os.path.join(IMAGE_FOLDER, str(name))
        img = keras.utils.load_img(path, target_size=IMG_SIZE)
        img = keras.utils.img_to_array(img)
        return applications.efficientnet.preprocess_input(img)
    except:
        return np.zeros((*IMG_SIZE, 3))

# 2. Predict Function
def predict_pair(img1_name, img2_name):
    i1 = load_and_preprocess_img(img1_name)
    i2 = load_and_preprocess_img(img2_name)
    i1 = np.expand_dims(i1, axis=0)
    i2 = np.expand_dims(i2, axis=0)
    
    # รวบรวมผลลัพธ์ (Summing predictions)
    preds = []
    for model in models:
        p = model.predict([i1, i2], verbose=0)[0][0]
        preds.append(p)
    
    avg_score = np.mean(preds)
    winner = 1 if avg_score > 0.5 else 0
    return winner, avg_score

# 3. Execution
df_test = pd.read_csv(TEST_CSV)
results = []

print(f"\nEnsemble Predicting {len(df_test)} pairs...")
for idx, row in df_test.iterrows():
    winner, confidence = predict_pair(row['Image 1'], row['Image 2'])
    results.append(winner)
    if (idx + 1) % 10 == 0:
        print(f"Processed {idx + 1}/{len(df_test)}")

df_test['Winner'] = results
df_test.to_csv(OUTPUT_CSV, index=False)
print(f"\n✅ Final Result saved as {OUTPUT_CSV}")