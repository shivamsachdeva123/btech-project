#!/usr/bin/env python3
"""Diagnostic: isolate which improvement broke baseline performance"""
import json
import sys
from pathlib import Path
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

ROOT = Path('/Users/shivam/Desktop/btech-project')
sys.path.insert(0, str(ROOT / 'src'))

# Load current model input data
payload = np.load(ROOT / 'data/processed/model_input/hybrid_late_fusion_input.npz')
X = payload['X']
y = payload['y']
window_size = int(payload['window_size'])
price_feature_dim = int(payload['price_feature_dim'])
sentiment_feature_dim = int(payload['sentiment_feature_dim'])

print("=" * 70)
print("ABLATION STUDY: Which improvement broke baseline?")
print("=" * 70)
print(f"\nData shape: X={X.shape}, y={y.shape}")
print(f"Window size: {window_size}")
print(f"Price features: {window_size} × {price_feature_dim} = {window_size * price_feature_dim}")
print(f"Sentiment features: {window_size} × {sentiment_feature_dim} = {window_size * sentiment_feature_dim}")
print(f"Company ID: 1 feature")
print(f"Total features: {X.shape[1]}")

# Extract feature slices
price_end = window_size * price_feature_dim
sentiment_end = price_end + (window_size * sentiment_feature_dim)
X_price_only = X[:, :price_end]
X_sentiment_only = X[:, price_end:sentiment_end]
X_company_only = X[:, sentiment_end:]

print(f"\n--- Feature ranges ---")
print(f"Price only: [{X_price_only.min():.4f}, {X_price_only.max():.4f}]")
print(f"Sentiment only: [{X_sentiment_only.min():.4f}, {X_sentiment_only.max():.4f}]")
print(f"Company ID: {X_company_only.shape}")

# Define baseline model (Ridge with scaling, simple but stable)
def get_ridge_model():
    return Pipeline([
        ('scaler', StandardScaler()),
        ('ridge', Ridge(alpha=1.0))
    ])

# Test 1: Price features only (no sentiment, no company)
print("\n" + "=" * 70)
print("TEST 1: Price features ONLY (pre-FinBERT baseline scenario)")
print("=" * 70)
model1 = get_ridge_model()
scores1 = cross_val_score(model1, X_price_only, y, cv=3, scoring='r2', n_jobs=1)
print(f"R² folds:  {', '.join(f'{s:.6f}' for s in scores1)}")
print(f"R² mean:   {scores1.mean():.6f}")
print(f"R² std:    {scores1.std():.6f}")

# Test 2: Price + Sentiment features (no company)
print("\n" + "=" * 70)
print("TEST 2: Price + Sentiment features (FinBERT + normalization)")
print("=" * 70)
X_price_sent = X[:, :sentiment_end]
model2 = get_ridge_model()
scores2 = cross_val_score(model2, X_price_sent, y, cv=3, scoring='r2', n_jobs=1)
print(f"R² folds:  {', '.join(f'{s:.6f}' for s in scores2)}")
print(f"R² mean:   {scores2.mean():.6f}")
print(f"R² std:    {scores2.std():.6f}")

# Test 3: All features (Price + Sentiment + Company)
print("\n" + "=" * 70)
print("TEST 3: All features (+ company embeddings)")
print("=" * 70)
model3 = get_ridge_model()
scores3 = cross_val_score(model3, X, y, cv=3, scoring='r2', n_jobs=1)
print(f"R² folds:  {', '.join(f'{s:.6f}' for s in scores3)}")
print(f"R² mean:   {scores3.mean():.6f}")
print(f"R² std:    {scores3.std():.6f}")

# Diagnosis
print("\n" + "=" * 70)
print("DIAGNOSIS")
print("=" * 70)
print(f"\nExpected baseline (price-only, from your notes): 0.833626")
print(f"Current price-only (with per-company norm): {scores1.mean():.6f}")

if scores1.mean() < 0.5:
    print("\n⚠️  CRITICAL: Per-company normalization seems to have BROKEN the price features!")
    print(f"   Price-only R² dropped from 0.83 → {scores1.mean():.6f}")
elif scores2.mean() < scores1.mean():
    print("\n⚠️  CRITICAL: FinBERT sentiment features are HURTING performance!")
    print(f"   Without sentiment: {scores1.mean():.6f}")
    print(f"   With sentiment:    {scores2.mean():.6f}")
    print(f"   Loss: {scores1.mean() - scores2.mean():.6f}")
else:
    print("\n✓ Sentiment features help slightly (or neutral)")

print(f"\nHybrid model with LSTM (current): {-1.539:.6f}")
print(f"Ridge baseline with all features: {scores3.mean():.6f}")
print(f"\nConclusion: Even simple Ridge struggles, suggesting issue is in DATA not MODEL ARCHITECTURE.")

