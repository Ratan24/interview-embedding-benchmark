import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_sample_weight
from mord import LogisticAT
import warnings
warnings.filterwarnings('ignore')
BASE_DIR = Path(__file__).resolve().parent
VECTORS_DIR = BASE_DIR / "vectors"
PARSED_DIR = BASE_DIR / "parsed"

def load_data(skill):
    df = pd.read_csv(PARSED_DIR / "labels.csv")
    embeds = np.load(VECTORS_DIR / "voyage_C2a.npy")
    import json
    with open(VECTORS_DIR / "voyage_C2a_ids.json") as f:
        ids = json.load(f)
    
    id_to_idx = {aid: i for i, aid in enumerate(df["job_application_id"])}
    indices = [id_to_idx[aid] for aid in ids]
    df = df.iloc[indices].reset_index(drop=True)
    
    # C2a: 0=react, 1=js, 2=html
    skill_idx = {"react": 0, "javascript": 1, "html_css": 2}[skill]
    X = embeds[:, skill_idx, :]
    y = df[f"{skill}_grade"].values
    return X, y

skills = ["react", "javascript", "html_css"]
models = {
    "LogisticAT_Baseline": lambda: LogisticAT(alpha=1.0, max_iter=1000),
    "Lasso_L1_Balanced": lambda: OneVsRestClassifier(LogisticRegression(penalty='l1', solver='liblinear', class_weight='balanced', C=1.0, max_iter=2000, random_state=42))
}

print("="*60)
print("SENIOR CLASS F1 EXPERIMENT (voyage / C2a)")
print("="*60)

for skill in skills:
    print(f"\n--- SKILL: {skill.upper()} ---")
    X, y = load_data(skill)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for m_name, m_func in models.items():
        senior_f1s = []
        macro_f1s = []
        
        for train_idx, test_idx in skf.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            
            clf = m_func()            
            clf.fit(X_train, y_train)
                
            y_pred = clf.predict(X_test)
            y_pred = np.clip(y_pred, 0, 3).astype(int)
            
            per_class = f1_score(y_test, y_pred, average=None, labels=[0, 1, 2, 3], zero_division=0)
            senior_f1s.append(per_class[3])
            macro_f1s.append(f1_score(y_test, y_pred, average='macro', zero_division=0))
            
        print(f"{m_name:<25} | Macro F1: {np.mean(macro_f1s):.3f} | Senior F1: {np.mean(senior_f1s):.3f}")

print("\n--- Testing 'class_weight' workaround for current LogisticAT ---")
for skill in skills:
    X, y = load_data(skill)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    senior_f1s = []
    macro_f1s = []
    supported = True
    
    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        sw = compute_sample_weight('balanced', y_train)
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        clf = LogisticAT(alpha=1.0, max_iter=1000)
        try:
            clf.fit(X_train, y_train, sample_weight=sw)
            y_pred = clf.predict(X_test)
            y_pred = np.clip(y_pred, 0, 3).astype(int)
            per_class = f1_score(y_test, y_pred, average=None, labels=[0, 1, 2, 3], zero_division=0)
            senior_f1s.append(per_class[3])
            macro_f1s.append(f1_score(y_test, y_pred, average='macro', zero_division=0))
        except TypeError:
            supported = False
            break
            
    if not supported:
        print(f"{skill:<15}: LogisticAT does not support sample_weight.")
    else:
        print(f"LogisticAT_Weighted       | Skill: {skill:<10} | Macro F1: {np.mean(macro_f1s):.3f} | Senior F1: {np.mean(senior_f1s):.3f}")

print("="*60)
