import numpy as np
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

ROOT = Path('/Users/shivam/Desktop/btech-project')
payload = np.load(ROOT / 'data/processed/model_input/hybrid_late_fusion_input.npz')
X = payload['X']
y = payload['y']
window_size = int(payload['window_size'])
price_feature_dim = int(payload['price_feature_dim'])
sentiment_feature_dim = int(payload['sentiment_feature_dim'])

price_end = window_size * price_feature_dim
sentiment_end = price_end + (window_size * sentiment_feature_dim)

X_price_only = X[:, :price_end]

print("TEST: Simple Ridge regression on price features only\n")

# Ridge with additional StandardScaler
model = Pipeline([
    ('scaler', StandardScaler()),
    ('ridge', Ridge(alpha=1.0))
])

print(f"Current baseline (from your notes):")
print(f"  Mean R²: 0.833626")
print(f"  R² folds: 0.882523, 0.837788, 0.780566\n")

scores = cross_val_score(model, X_price_only, y, cv=3, scoring='r2', n_jobs=1)
print(f"Current price-only features (with per-company norm):")
print(f"  R² folds: {', '.join(f'{s:.6f}' for s in scores)}")
print(f"  Mean R²:  {scores.mean():.6f}")
print(f"  Std R²:   {scores.std():.6f}\n")

if scores.mean() < 0.5:
    print("❌ FINDING: Per-company normalization or data alignment issue!")
    print(f"   R² collapsed from 0.83 to {scores.mean():.3f}")
else:
    print("✓ Price signal still strong")

# Now test with sentiment features
print(f"\n{'='*60}")
X_with_sent = X[:, :sentiment_end]
model2 = Pipeline([
    ('scaler', StandardScaler()),
    ('ridge', Ridge(alpha=1.0))
])

scores2 = cross_val_score(model2, X_with_sent, y, cv=3, scoring='r2', n_jobs=1)
print(f"Price + Sentiment features:")
print(f"  R² folds: {', '.join(f'{s:.6f}' for s in scores2)}")
print(f"  Mean R²:  {scores2.mean():.6f}\n")

delta = scores.mean() - scores2.mean()
if delta > 0.1:
    print(f"⚠️  Sentiment features degrade performance by {abs(delta):.4f}")
else:
    print(f"✓ Sentiment adds minimal benefit/harm")
