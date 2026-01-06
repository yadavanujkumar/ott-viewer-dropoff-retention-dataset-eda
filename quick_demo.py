"""
Quick Demo - Essential Data Science, AI & ML Capabilities
==========================================================

This script demonstrates the key capabilities in just a few minutes.
Perfect for a quick overview before diving into the full analysis.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("🎬 OTT DATASET - QUICK DEMO")
print("="*70)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n[1/5] Loading data...")
df = pd.read_csv('ott_viewer_dropoff_retention_us_v1.0.csv')
print(f"✓ Loaded {len(df):,} episodes from {df['show_id'].nunique()} shows")
print(f"  Platforms: {df['platform'].nunique()} | Genres: {df['genre'].nunique()}")
print(f"  Drop-off rate: {df['drop_off'].mean()*100:.2f}%")

# ============================================================================
# 2. EXPLORATORY DATA ANALYSIS
# ============================================================================
print("\n[2/5] Exploratory Data Analysis...")
print(f"\n📊 Key Statistics:")
print(f"  Average watch completion: {df['avg_watch_percentage'].mean():.1f}%")
print(f"  Average hook strength: {df['hook_strength'].mean():.2f}/10")
print(f"  Average cognitive load: {df['cognitive_load'].mean():.2f}/10")
print(f"  Episodes per show: {len(df) / df['show_id'].nunique():.1f}")

print(f"\n📺 Top 5 Platforms:")
for idx, (platform, count) in enumerate(df['platform'].value_counts().head(5).items(), 1):
    print(f"  {idx}. {platform}: {count:,} episodes")

print(f"\n🎭 Top 5 Genres:")
for idx, (genre, count) in enumerate(df['genre'].value_counts().head(5).items(), 1):
    print(f"  {idx}. {genre}: {count:,} episodes")

# ============================================================================
# 3. MACHINE LEARNING - DROP-OFF PREDICTION
# ============================================================================
print("\n[3/5] Training ML model for drop-off prediction...")

# Prepare features
df_ml = df.copy()
df_ml['is_premiere'] = (df_ml['episode_number'] == 1).astype(int)
df_ml['engagement_score'] = (df_ml['avg_watch_percentage'] * df_ml['hook_strength']) / 100

# Encode categoricals
le_platform = LabelEncoder()
le_genre = LabelEncoder()
df_ml['platform_encoded'] = le_platform.fit_transform(df_ml['platform'])
df_ml['genre_encoded'] = le_genre.fit_transform(df_ml['genre'])

# Select features
feature_cols = ['avg_watch_percentage', 'hook_strength', 'cognitive_load',
               'pacing_score', 'visual_intensity', 'pause_count', 'rewind_count',
               'platform_encoded', 'genre_encoded', 'is_premiere', 'engagement_score']

X = df_ml[feature_cols]
y = df_ml['drop_off']

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestClassifier(n_estimators=50, max_depth=8, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

# Evaluate
y_pred = rf_model.predict(X_test)
y_pred_proba = rf_model.predict_proba(X_test)[:, 1]

print(f"\n🤖 Model Performance:")
print(f"  Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
print(f"  ROC-AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\n🎯 Top 5 Important Features:")
for idx, row in feature_importance.head(5).iterrows():
    print(f"  {row['feature']:30s}: {row['importance']:.4f}")

# ============================================================================
# 4. CLUSTERING ANALYSIS
# ============================================================================
print("\n[4/5] Performing clustering analysis...")

cluster_features = ['avg_watch_percentage', 'hook_strength', 'cognitive_load']
X_cluster = df_ml[cluster_features]

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df_ml['cluster'] = kmeans.fit_predict(X_cluster)

print(f"\n📊 Viewer Segments Identified:")
for i in range(3):
    cluster_data = df_ml[df_ml['cluster'] == i]
    print(f"\n  Cluster {i} ({len(cluster_data):,} episodes):")
    print(f"    Avg watch: {cluster_data['avg_watch_percentage'].mean():.1f}%")
    print(f"    Drop-off rate: {cluster_data['drop_off'].mean()*100:.1f}%")
    print(f"    Risk level: {cluster_data['retention_risk'].mode()[0]}")

# ============================================================================
# 5. BUSINESS INSIGHTS
# ============================================================================
print("\n[5/5] Extracting business insights...")

# High-risk episodes
high_risk = df_ml[df_ml['retention_risk'] == 'high']
print(f"\n⚠️  High-Risk Episodes:")
print(f"  Total: {len(high_risk):,} ({len(high_risk)/len(df_ml)*100:.1f}%)")
print(f"  Avg watch: {high_risk['avg_watch_percentage'].mean():.1f}%")
print(f"  Avg cognitive load: {high_risk['cognitive_load'].mean():.2f}")

# Best performing shows
top_shows = df_ml.groupby('title').agg({
    'avg_watch_percentage': 'mean',
    'drop_off': 'mean',
    'episode_number': 'count'
}).rename(columns={'episode_number': 'episodes'})
top_shows = top_shows[top_shows['episodes'] >= 5].sort_values('avg_watch_percentage', ascending=False)

print(f"\n🌟 Top 5 Shows by Engagement (min 5 episodes):")
for idx, (title, row) in enumerate(top_shows.head(5).iterrows(), 1):
    print(f"  {idx}. {title}")
    print(f"     Watch: {row['avg_watch_percentage']:.1f}% | Drop-off: {row['drop_off']*100:.1f}% | Episodes: {int(row['episodes'])}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*70)
print("✅ QUICK DEMO COMPLETE")
print("="*70)

print(f"""
📊 What We Demonstrated:

1. DATA ANALYSIS
   ✓ Loaded and analyzed 33K+ episodes
   ✓ Generated key statistics and distributions
   ✓ Identified top platforms and genres

2. MACHINE LEARNING
   ✓ Trained Random Forest classifier
   ✓ Achieved {accuracy_score(y_test, y_pred)*100:.1f}% accuracy
   ✓ Identified top predictive features

3. CLUSTERING
   ✓ Segmented viewers into 3 groups
   ✓ Profiled each cluster's behavior
   ✓ Identified high-risk segments

4. BUSINESS INTELLIGENCE
   ✓ Flagged high-risk episodes
   ✓ Identified top-performing shows
   ✓ Generated actionable insights

🚀 Next Steps:
   • Run the full analysis: python ott_comprehensive_analysis.py
   • Explore interactively: jupyter notebook OTT_Analysis_Notebook.ipynb
   • Read documentation: ANALYSIS_DOCUMENTATION.md

""")

print("="*70)
print("Thank you! 🎉")
print("="*70 + "\n")
