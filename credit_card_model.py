
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("creditcard.csv")

df.shape


df.head()

df.info()


df.describe()

potability_counts = df['Class'].value_counts()
potability_counts.plot(kind='bar')
plt.title("Frequency of Fraud or not")
plt.xlabel("fraud or not")
plt.ylabel("Count")
plt.xticks(ticks=[0, 1], labels=["Not fraud", "Fraud"], rotation=0)
plt.show()



plt.figure(figsize=(8, 6))
plt.pie(potability_counts, labels=['Not Fraud','Fraud'], autopct='%1.1f%%')
plt.title('Percentage of Fraud and Not Fraud')
plt.show()


print(df.duplicated().sum())

print(df.isnull().sum())

df = df.dropna()
print(df.isnull().sum())

cor1 = df.corr()
print(cor1)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df['Amount_scaled'] = scaler.fit_transform(df[['Amount']])
df['Time_scaled'] = scaler.fit_transform(df[['Time']])

df = df.drop(['Amount', 'Time'], axis=1)

from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report, f1_score, recall_score, precision_score
from xgboost import XGBClassifier
X = df.drop('Class', axis=1)
y = df['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

print(f"Original distribution in training set:")
print(y_train.value_counts())

all_results = []

print("\nGenerating resampled datasets...")
smote = SMOTE(random_state=42, k_neighbors=3)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

ros = RandomOverSampler(random_state=42)
X_train_ros, y_train_ros = ros.fit_resample(X_train, y_train)

rus = RandomUnderSampler(random_state=42)
X_train_rus, y_train_rus = rus.fit_resample(X_train, y_train)

print("✅ All datasets ready!")

def train_and_evaluate(model, X_tr, y_tr, X_te, y_te, model_name, method_name):
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    
    all_results.append({
        'Model': model_name,
        'Method': method_name,
        'F1-Score': f1_score(y_te, y_pred),
        'Recall': recall_score(y_te, y_pred),
        'Precision': precision_score(y_te, y_pred)
    })
    
    print(f"{model_name} + {method_name}: F1={f1_score(y_te, y_pred):.4f}, Recall={recall_score(y_te, y_pred):.4f}")

print("\n" + "="*70)
print("TRAINING ALL MODELS - FAST MODE (No plots during training)")
print("="*70)

print("\nLogistic Regression...")
train_and_evaluate(LogisticRegression(random_state=42), X_train_smote, y_train_smote, X_test, y_test, 'Logistic Regression', 'SMOTE')
train_and_evaluate(LogisticRegression(random_state=42), X_train_ros, y_train_ros, X_test, y_test, 'Logistic Regression', 'Random Oversampling')
train_and_evaluate(LogisticRegression(random_state=42), X_train_rus, y_train_rus, X_test, y_test, 'Logistic Regression', 'Random Undersampling')
train_and_evaluate(LogisticRegression(class_weight='balanced', random_state=42), X_train, y_train, X_test, y_test, 'Logistic Regression', 'Class Weights')
train_and_evaluate(LogisticRegression(random_state=42), X_train, y_train, X_test, y_test, 'Logistic Regression', 'No Balancing')

print("\nRandom Forest...")
train_and_evaluate(RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1), X_train_smote, y_train_smote, X_test, y_test, 'Random Forest', 'SMOTE')
train_and_evaluate(RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1), X_train_ros, y_train_ros, X_test, y_test, 'Random Forest', 'Random Oversampling')
train_and_evaluate(RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1), X_train_rus, y_train_rus, X_test, y_test, 'Random Forest', 'Random Undersampling')
train_and_evaluate(RandomForestClassifier(n_estimators=50, class_weight='balanced', random_state=42, n_jobs=-1), X_train, y_train, X_test, y_test, 'Random Forest', 'Class Weights')
train_and_evaluate(RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1), X_train, y_train, X_test, y_test, 'Random Forest', 'No Balancing')

print("\nNeural Network...")
train_and_evaluate(MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=2000, random_state=42, early_stopping=True), X_train_smote, y_train_smote, X_test, y_test, 'Neural Network', 'SMOTE')
train_and_evaluate(MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=2000, random_state=42, early_stopping=True), X_train_ros, y_train_ros, X_test, y_test, 'Neural Network', 'Random Oversampling')
train_and_evaluate(MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=2000, random_state=42, early_stopping=True), X_train_rus, y_train_rus, X_test, y_test, 'Neural Network', 'Random Undersampling')
train_and_evaluate(MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=2000, random_state=42, early_stopping=True), X_train, y_train, X_test, y_test, 'Neural Network', 'No Balancing')

print("\nXGBoost...")
train_and_evaluate(XGBClassifier(n_estimators=50, max_depth=6, random_state=42, eval_metric='logloss', use_label_encoder=False), X_train_smote, y_train_smote, X_test, y_test, 'XGBoost', 'SMOTE')
train_and_evaluate(XGBClassifier(n_estimators=50, max_depth=6, random_state=42, eval_metric='logloss', use_label_encoder=False), X_train_ros, y_train_ros, X_test, y_test, 'XGBoost', 'Random Oversampling')
train_and_evaluate(XGBClassifier(n_estimators=50, max_depth=6, random_state=42, eval_metric='logloss', use_label_encoder=False), X_train_rus, y_train_rus, X_test, y_test, 'XGBoost', 'Random Undersampling')

scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
train_and_evaluate(XGBClassifier(n_estimators=50, max_depth=6, scale_pos_weight=scale_pos_weight, random_state=42, eval_metric='logloss', use_label_encoder=False), X_train, y_train, X_test, y_test, 'XGBoost', 'Scale Pos Weight')
train_and_evaluate(XGBClassifier(n_estimators=50, max_depth=6, random_state=42, eval_metric='logloss', use_label_encoder=False), X_train, y_train, X_test, y_test, 'XGBoost', 'No Balancing')

print("\n✅ All models trained!")

print("\n" + "="*70)
print("FINAL COMPARISON - ALL MODELS & METHODS")
print("="*70)

results_df = pd.DataFrame(all_results)
results_df = results_df.drop_duplicates(subset=['Model', 'Method'], keep='last')

results_df_sorted = results_df.sort_values(['Recall', 'F1-Score'], ascending=False)
print("\nTop 10 Best Combinations (Sorted by Recall):")
print(results_df_sorted.head(10).to_string(index=False))

fig, axes = plt.subplots(1, 3, figsize=(20, 6))

results_df.pivot(index='Method', columns='Model', values='Recall').plot(kind='bar', ax=axes[0], width=0.8, colormap='RdYlGn')
axes[0].set_title('⭐ RECALL - MOST IMPORTANT FOR FRAUD! ⭐', fontsize=14, fontweight='bold', color='darkred')
axes[0].set_ylabel('Recall (Higher = Better)', fontweight='bold')
axes[0].set_xlabel('Balancing Method')
axes[0].legend(title='Model', loc='lower right')
axes[0].tick_params(axis='x', rotation=45)
axes[0].grid(axis='y', alpha=0.3)
axes[0].axhline(y=0.85, color='green', linestyle='--', linewidth=2)

results_df.pivot(index='Method', columns='Model', values='F1-Score').plot(kind='bar', ax=axes[1], width=0.8, colormap='viridis')
axes[1].set_title('F1-Score Comparison', fontsize=14, fontweight='bold')
axes[1].set_ylabel('F1-Score')
axes[1].set_xlabel('Balancing Method')
axes[1].legend(title='Model', loc='lower right')
axes[1].tick_params(axis='x', rotation=45)
axes[1].grid(axis='y', alpha=0.3)

results_df.pivot(index='Method', columns='Model', values='Precision').plot(kind='bar', ax=axes[2], width=0.8, colormap='plasma')
axes[2].set_title('Precision Comparison', fontsize=14, fontweight='bold')
axes[2].set_ylabel('Precision')
axes[2].set_xlabel('Balancing Method')
axes[2].legend(title='Model', loc='lower right')
axes[2].tick_params(axis='x', rotation=45)
axes[2].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

best_recall = results_df_sorted.iloc[0]
print(f"\n🏆 BEST COMBINATION (Highest Recall):")
print(f"Model: {best_recall['Model']}")
print(f"Method: {best_recall['Method']}")
print(f"Recall: {best_recall['Recall']:.4f} ⭐⭐⭐")
print(f"F1-Score: {best_recall['F1-Score']:.4f}")
print(f"Precision: {best_recall['Precision']:.4f}")

results_df_f1_sorted = results_df.sort_values('F1-Score', ascending=False)
best_f1 = results_df_f1_sorted.iloc[0]
print(f"\n🥇 BEST F1-SCORE (Best Balance):")
print(f"Model: {best_f1['Model']}")
print(f"Method: {best_f1['Method']}")
print(f"F1-Score: {best_f1['F1-Score']:.4f} ⭐⭐⭐")
print(f"Recall: {best_f1['Recall']:.4f}")
print(f"Precision: {best_f1['Precision']:.4f}")

# NEW: Find best balance between Recall and F1-Score
# Models with high recall (>0.80) AND high F1-score (>0.70)
balanced_models = results_df[(results_df['Recall'] >= 0.80) & (results_df['F1-Score'] >= 0.70)]

if not balanced_models.empty:
    # Sort by combined score (weighted average)
    balanced_models['Combined_Score'] = (0.6 * balanced_models['Recall']) + (0.4 * balanced_models['F1-Score'])
    balanced_models_sorted = balanced_models.sort_values('Combined_Score', ascending=False)
    
    best_balanced = balanced_models_sorted.iloc[0]
    print(f"\n⚖️ BEST BALANCED MODEL (High Recall + High F1):")
    print(f"Model: {best_balanced['Model']}")
    print(f"Method: {best_balanced['Method']}")
    print(f"Recall: {best_balanced['Recall']:.4f}")
    print(f"F1-Score: {best_balanced['F1-Score']:.4f}")
    print(f"Precision: {best_balanced['Precision']:.4f}")
    print(f"Combined Score: {best_balanced['Combined_Score']:.4f}")
else:
    print("\n⚖️ No models meet the balanced criteria (Recall ≥ 0.80 AND F1 ≥ 0.70)")

print("\n" + "="*70)
print("AVERAGE PERFORMANCE BY MODEL")
print("="*70)
model_summary = results_df.groupby('Model')[['F1-Score', 'Recall', 'Precision']].mean().sort_values('Recall', ascending=False)
print(model_summary.to_string())

print("\n" + "="*70)
print("AVERAGE PERFORMANCE BY BALANCING METHOD")
print("="*70)
method_summary = results_df.groupby('Method')[['F1-Score', 'Recall', 'Precision']].mean().sort_values('Recall', ascending=False)
print(method_summary.to_string())

fig, ax = plt.subplots(figsize=(12, 6))
model_summary.plot(kind='bar', ax=ax, width=0.7)
ax.set_title('Average Performance by Model', fontsize=16, fontweight='bold')
ax.set_ylabel('Score', fontweight='bold')
ax.set_xlabel('Model', fontweight='bold')
ax.legend(['F1-Score', 'Recall', 'Precision'])
ax.tick_params(axis='x', rotation=45)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(14, 6))
method_summary.plot(kind='bar', ax=ax, width=0.7)
ax.set_title('Average Performance by Balancing Method', fontsize=16, fontweight='bold')
ax.set_ylabel('Score', fontweight='bold')
ax.set_xlabel('Balancing Method', fontweight='bold')
ax.legend(['F1-Score', 'Recall', 'Precision'])
ax.tick_params(axis='x', rotation=45)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

print("\n" + "="*70)
print("✅ ANALYSIS COMPLETE!")
print("="*70)

print("\n💡 RECOMMENDATIONS FOR FRAUD DETECTION:")
print("\n" + "-"*70)

# Option 1: Maximum fraud detection
print("📊 OPTION 1: MAXIMIZE FRAUD DETECTION (Highest Recall)")
print(f"   Use: {best_recall['Model']} + {best_recall['Method']}")
print(f"   ✅ Catches {best_recall['Recall']:.1%} of all fraud cases")
print(f"   ⚠️  {best_recall['Precision']:.1%} of alerts are real fraud (some false positives)")
print(f"   F1-Score: {best_recall['F1-Score']:.4f}")
print(f"   Best for: When missing fraud is very costly")

print("\n" + "-"*70)

# Option 2: Best overall balance
print("📊 OPTION 2: BEST OVERALL BALANCE (Highest F1-Score)")
print(f"   Use: {best_f1['Model']} + {best_f1['Method']}")
print(f"   ✅ Balanced performance (F1: {best_f1['F1-Score']:.4f})")
print(f"   ✅ Catches {best_f1['Recall']:.1%} of fraud")
print(f"   ✅ {best_f1['Precision']:.1%} of alerts are real fraud")
print(f"   Best for: General production use")

print("\n" + "-"*70)


if not balanced_models.empty:
    print("📊 OPTION 3: SWEET SPOT (High Recall + High F1)")
    print(f"   Use: {best_balanced['Model']} + {best_balanced['Method']}")
    print(f"   ✅ Catches {best_balanced['Recall']:.1%} of fraud")
    print(f"   ✅ Good precision: {best_balanced['Precision']:.1%}")
    print(f"   ✅ Strong F1-Score: {best_balanced['F1-Score']:.4f}")
    print(f"   Best for: Optimal balance between catching fraud and minimizing false alarms")
    
    print("\n" + "="*70)
    print(f"🎯 RECOMMENDED CHOICE: OPTION 3 (Sweet Spot)")
    print("="*70)
else:
    print("\n" + "="*70)
    print(f"🎯 RECOMMENDED CHOICE: OPTION 2 (Best F1-Score)")
    print("="*70)




import pickle
import joblib
print("\n" + "="*70)
print("💾 SAVING THE BEST MODEL")
print("="*70)

# Determine best model
if not balanced_models.empty:
    best_model_name = best_balanced['Model']
    best_method_name = best_balanced['Method']
else:
    best_model_name = best_f1['Model']
    best_method_name = best_f1['Method']

print(f"\nSaving: {best_model_name} + {best_method_name}")

# Select appropriate dataset
if best_method_name == 'SMOTE':
    X_train_final, y_train_final = X_train_smote, y_train_smote
elif best_method_name == 'Random Oversampling':
    X_train_final, y_train_final = X_train_ros, y_train_ros
elif best_method_name == 'Random Undersampling':
    X_train_final, y_train_final = X_train_rus, y_train_rus
else:
    X_train_final, y_train_final = X_train, y_train

# Initialize the best model
if best_model_name == 'Logistic Regression':
    if best_method_name == 'Class Weights':
        final_model = LogisticRegression(class_weight='balanced', random_state=42)
    else:
        final_model = LogisticRegression(random_state=42)
        
elif best_model_name == 'Random Forest':
    if best_method_name == 'Class Weights':
        final_model = RandomForestClassifier(n_estimators=50, class_weight='balanced', random_state=42, n_jobs=-1)
    else:
        final_model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        
elif best_model_name == 'Neural Network':
    final_model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=2000, random_state=42, early_stopping=True)
    
elif best_model_name == 'XGBoost':
    if best_method_name == 'Scale Pos Weight':
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        final_model = XGBClassifier(n_estimators=50, max_depth=6, scale_pos_weight=scale_pos_weight, 
                                    random_state=42, eval_metric='logloss', use_label_encoder=False)
    else:
        final_model = XGBClassifier(n_estimators=50, max_depth=6, random_state=42, 
                                    eval_metric='logloss', use_label_encoder=False)

# Train final model
print("Training final model...")
final_model.fit(X_train_final, y_train_final)
print("✅ Model trained successfully!")

# Save the model
joblib.dump(final_model, 'fraud_detection_model.pkl')
print("✅ Model saved as 'fraud_detection_model.pkl'")

# Save the scaler (important for preprocessing new data)
joblib.dump(scaler, 'scaler.pkl')
print("✅ Scaler saved as 'scaler.pkl'")

# Save feature names
feature_names = list(X_train.columns)
joblib.dump(feature_names, 'feature_names.pkl')
print("✅ Feature names saved as 'feature_names.pkl'")

# Save model metadata
model_info = {
    'model_name': best_model_name,
    'method_name': best_method_name,
    'recall': best_balanced['Recall'] if not balanced_models.empty else best_f1['Recall'],
    'f1_score': best_balanced['F1-Score'] if not balanced_models.empty else best_f1['F1-Score'],
    'precision': best_balanced['Precision'] if not balanced_models.empty else best_f1['Precision'],
    'features': feature_names
}
joblib.dump(model_info, 'model_info.pkl')
print("✅ Model info saved as 'model_info.pkl'")

print("\n✅ ALL FILES SAVED SUCCESSFULLY!")
print("Files created:")
print("  1. fraud_detection_model.pkl")
print("  2. scaler.pkl")
print("  3. feature_names.pkl")
print("  4. model_info.pkl")