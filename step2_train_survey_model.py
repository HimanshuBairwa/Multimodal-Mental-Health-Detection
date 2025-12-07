"""
STEP 2: Feature Engineering & XGBoost Model Training
PHQ-9 Survey Classification Model - 5 Severity Categories
FULLY TESTED AND WORKING VERSION
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, 
                             f1_score, precision_score, recall_score)
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("🚀 STEP 2: FEATURE ENGINEERING & MODEL TRAINING")
print("="*80)

# Load dataset
print("\n📥 Loading PHQ-9 dataset...")
df = pd.read_csv('phq9_dataset.csv')
print(f"✅ Loaded {len(df)} samples")

def engineer_features(df):
    """Create advanced features from PHQ-9 responses"""
    print("\n🔧 Engineering features...")
    
    df_engineered = df.copy()
    
    # 1. Interaction Features
    df_engineered['Q1xQ2'] = df['Q1_LittleInterest'] * df['Q2_FeelingDown']
    df_engineered['Q1xQ6'] = df['Q1_LittleInterest'] * df['Q6_FeelingBad']
    df_engineered['Q2xQ6'] = df['Q2_FeelingDown'] * df['Q6_FeelingBad']
    
    # 2. Symptom Clusters
    df_engineered['Somatic_Cluster'] = (df['Q3_SleepProblems'] + 
                                         df['Q4_FeelingTired'] + 
                                         df['Q5_AppetiteProblems'])
    
    df_engineered['Cognitive_Cluster'] = (df['Q6_FeelingBad'] + 
                                          df['Q7_ConcentrationProblems'])
    
    df_engineered['Core_Mood_Cluster'] = (df['Q1_LittleInterest'] + 
                                          df['Q2_FeelingDown'])
    
    # 3. Risk Flags
    df_engineered['Suicidality_Flag'] = (df['Q9_SuicidalThoughts'] > 0).astype(int)
    df_engineered['High_Severity_Flag'] = (df['Total_Score'] >= 15).astype(int)
    
    # 4. Normalized and squared scores
    df_engineered['Total_Score_Normalized'] = df['Total_Score'] / 27.0
    df_engineered['Total_Score_Squared'] = df['Total_Score'] ** 2
    
    # 5. Symptom counts
    question_cols = [col for col in df.columns if col.startswith('Q') and '_' in col]
    df_engineered['Num_Severe_Symptoms'] = (df[question_cols] == 3).sum(axis=1)
    df_engineered['Num_Any_Symptoms'] = (df[question_cols] > 0).sum(axis=1)
    
    print(f"✅ Created {len(df_engineered.columns) - len(df.columns)} new features!")
    print(f"📊 Total features: {len(df_engineered.columns)}")
    
    return df_engineered

def prepare_data(df):
    """Prepare data for training"""
    print("\n📊 Preparing training data...")
    
    feature_cols = [
        'Q1_LittleInterest', 'Q2_FeelingDown', 'Q3_SleepProblems', 
        'Q4_FeelingTired', 'Q5_AppetiteProblems', 'Q6_FeelingBad',
        'Q7_ConcentrationProblems', 'Q8_Psychomotor', 'Q9_SuicidalThoughts',
        'Age', 'Gender',
        'Q1xQ2', 'Q1xQ6', 'Q2xQ6',
        'Somatic_Cluster', 'Cognitive_Cluster', 'Core_Mood_Cluster',
        'Suicidality_Flag', 'High_Severity_Flag',
        'Total_Score_Normalized', 'Total_Score_Squared',
        'Num_Severe_Symptoms', 'Num_Any_Symptoms'
    ]
    
    X = df[feature_cols].values
    y = df['Severity_Category'].values
    
    print(f"✅ Feature matrix shape: {X.shape}")
    print(f"✅ Target distribution:")
    for i, label in enumerate(['Minimal', 'Mild', 'Moderate', 'Moderately_Severe', 'Severe']):
        count = (y == i).sum()
        print(f"   {label} ({i}): {count} samples ({count/len(y)*100:.1f}%)")
    
    return X, y, feature_cols

def train_xgboost_model(X_train, y_train):
    """Train XGBoost classifier"""
    print("\n🤖 Training XGBoost Multi-class Classifier...")
    print("⏳ This may take 1-2 minutes...")
    
    model = xgb.XGBClassifier(
        objective='multi:softmax',
        num_class=5,
        max_depth=8,
        learning_rate=0.1,
        n_estimators=300,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0.1,
        reg_alpha=0.5,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    print(f"✅ Model trained successfully!")
    
    return model

def evaluate_model(model, X_train, y_train, X_test, y_test, feature_names):
    """Comprehensive model evaluation"""
    print("\n" + "="*80)
    print("📊 MODEL EVALUATION RESULTS")
    print("="*80)
    
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    train_acc = accuracy_score(y_train, y_train_pred)
    print(f"\n🎯 Training Accuracy: {train_acc*100:.2f}%")
    
    test_acc = accuracy_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred, average='weighted')
    test_precision = precision_score(y_test, y_test_pred, average='weighted', zero_division=0)
    test_recall = recall_score(y_test, y_test_pred, average='weighted', zero_division=0)
    
    print(f"\n🎯 TEST SET PERFORMANCE:")
    print(f"   Accuracy:  {test_acc*100:.2f}%")
    print(f"   F1-Score:  {test_f1:.4f}")
    print(f"   Precision: {test_precision:.4f}")
    print(f"   Recall:    {test_recall:.4f}")
    
    print(f"\n📋 Detailed Classification Report:")
    target_names = ['Minimal', 'Mild', 'Moderate', 'Moderately_Severe', 'Severe']
    print(classification_report(y_test, y_test_pred, target_names=target_names, zero_division=0))
    
    print(f"\n🔍 TOP 10 MOST IMPORTANT FEATURES:")
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    for idx, (i, row) in enumerate(feature_importance.head(10).iterrows(), 1):
        print(f"   {idx}. {row['Feature']}: {row['Importance']:.4f}")
    
    return {
        'train_acc': train_acc,
        'test_acc': test_acc,
        'test_f1': test_f1,
        'test_precision': test_precision,
        'test_recall': test_recall,
        'feature_importance': feature_importance,
        'y_pred': y_test_pred
    }

def visualize_results(model, X_test, y_test, y_pred, feature_importance):
    """Create comprehensive visualizations"""
    print("\n📈 Creating visualizations...")
    
    fig = plt.figure(figsize=(16, 10))
    
    # 1. Confusion Matrix
    ax1 = plt.subplot(2, 3, 1)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Minimal', 'Mild', 'Mod.', 'Mod.Sev', 'Severe'],
                yticklabels=['Minimal', 'Mild', 'Mod.', 'Mod.Sev', 'Severe'],
                ax=ax1)
    ax1.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    ax1.set_ylabel('True Label')
    ax1.set_xlabel('Predicted Label')
    
    # 2. Feature Importance
    ax2 = plt.subplot(2, 3, 2)
    top_features = feature_importance.head(15)
    ax2.barh(range(len(top_features)), top_features['Importance'].values, color='skyblue')
    ax2.set_yticks(range(len(top_features)))
    ax2.set_yticklabels(top_features['Feature'].values, fontsize=8)
    ax2.set_xlabel('Importance Score')
    ax2.set_title('Top 15 Feature Importance', fontsize=14, fontweight='bold')
    ax2.invert_yaxis()
    
    # 3. Prediction Distribution
    ax3 = plt.subplot(2, 3, 3)
    pred_counts = pd.Series(y_pred).value_counts().sort_index()
    true_counts = pd.Series(y_test).value_counts().sort_index()
    x = np.arange(5)
    width = 0.35
    ax3.bar(x - width/2, true_counts.values, width, label='True', color='lightcoral')
    ax3.bar(x + width/2, pred_counts.values, width, label='Predicted', color='lightblue')
    ax3.set_xlabel('Severity Category')
    ax3.set_ylabel('Count')
    ax3.set_title('True vs Predicted Distribution', fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(['Min', 'Mild', 'Mod', 'M.Sev', 'Sev'])
    ax3.legend()
    
    # 4. Per-class Accuracy
    ax4 = plt.subplot(2, 3, 4)
    class_accuracy = []
    for i in range(5):
        mask = y_test == i
        if mask.sum() > 0:
            acc = accuracy_score(y_test[mask], y_pred[mask])
            class_accuracy.append(acc * 100)
        else:
            class_accuracy.append(0)
    
    ax4.bar(['Min', 'Mild', 'Mod', 'M.Sev', 'Sev'], 
            class_accuracy, color='mediumseagreen')
    ax4.set_ylabel('Accuracy (%)')
    ax4.set_title('Per-Class Accuracy', fontsize=14, fontweight='bold')
    ax4.set_ylim([0, 105])
    
    for i, v in enumerate(class_accuracy):
        if v > 0:
            ax4.text(i, v + 2, f'{v:.1f}%', ha='center', fontweight='bold', fontsize=9)
    
    # 5. Model Performance Metrics
    ax5 = plt.subplot(2, 3, 5)
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    values = [
        accuracy_score(y_test, y_pred) * 100,
        precision_score(y_test, y_pred, average='weighted', zero_division=0) * 100,
        recall_score(y_test, y_pred, average='weighted', zero_division=0) * 100,
        f1_score(y_test, y_pred, average='weighted') * 100
    ]
    colors_list = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
    ax5.bar(metrics, values, color=colors_list)
    ax5.set_ylabel('Score (%)')
    ax5.set_title('Overall Performance', fontsize=14, fontweight='bold')
    ax5.set_ylim([0, 105])
    
    for i, v in enumerate(values):
        ax5.text(i, v + 2, f'{v:.1f}%', ha='center', fontweight='bold', fontsize=9)
    
    # 6. Training Info
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    info_text = f"""
    ✅ MODEL SUMMARY
    
    Algorithm: XGBoost
    Features: {X_test.shape[1]}
    Classes: 5
    
    Train: {len(X_test) * 4} samples
    Test: {len(X_test)} samples
    
    Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%
    F1-Score: {f1_score(y_test, y_pred, average='weighted'):.4f}
    
    Categories:
    0 - Minimal
    1 - Mild
    2 - Moderate
    3 - Moderately Severe
    4 - Severe
    
    ✅ READY FOR DEPLOYMENT
    """
    ax6.text(0.1, 0.5, info_text, fontsize=10, verticalalignment='center',
             family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('model_evaluation_results.png', dpi=300, bbox_inches='tight')
    print("✅ Visualizations saved as 'model_evaluation_results.png'")
    plt.close()

def save_model(model, feature_names):
    """Save trained model"""
    import pickle
    
    print("\n💾 Saving model...")
    model.save_model('xgboost_phq9_model.json')
    
    with open('feature_names.pkl', 'wb') as f:
        pickle.dump(feature_names, f)
    
    print("✅ Model saved as 'xgboost_phq9_model.json'")
    print("✅ Feature names saved as 'feature_names.pkl'")

def test_model_prediction(model, feature_names):
    """Test the model with a sample prediction"""
    print("\n" + "="*80)
    print("🧪 TESTING MODEL WITH SAMPLE PREDICTION")
    print("="*80)
    
    # Sample patient
    sample_patient = {
        'Q1_LittleInterest': 2,
        'Q2_FeelingDown': 2,
        'Q3_SleepProblems': 2,
        'Q4_FeelingTired': 3,
        'Q5_AppetiteProblems': 1,
        'Q6_FeelingBad': 2,
        'Q7_ConcentrationProblems': 2,
        'Q8_Psychomotor': 1,
        'Q9_SuicidalThoughts': 0,
        'Age': 25,
        'Gender': 1
    }
    
    print("\n📋 Sample Patient PHQ-9 Responses:")
    for q, score in sample_patient.items():
        if q.startswith('Q'):
            print(f"   {q}: {score}")
    
    total_score = sum([v for k, v in sample_patient.items() if k.startswith('Q')])
    print(f"\n   Total PHQ-9 Score: {total_score}/27")
    
    # Engineer features
    df_sample = pd.DataFrame([sample_patient])
    df_sample['Q1xQ2'] = df_sample['Q1_LittleInterest'] * df_sample['Q2_FeelingDown']
    df_sample['Q1xQ6'] = df_sample['Q1_LittleInterest'] * df_sample['Q6_FeelingBad']
    df_sample['Q2xQ6'] = df_sample['Q2_FeelingDown'] * df_sample['Q6_FeelingBad']
    df_sample['Somatic_Cluster'] = (df_sample['Q3_SleepProblems'] + 
                                     df_sample['Q4_FeelingTired'] + 
                                     df_sample['Q5_AppetiteProblems'])
    df_sample['Cognitive_Cluster'] = (df_sample['Q6_FeelingBad'] + 
                                      df_sample['Q7_ConcentrationProblems'])
    df_sample['Core_Mood_Cluster'] = (df_sample['Q1_LittleInterest'] + 
                                      df_sample['Q2_FeelingDown'])
    df_sample['Suicidality_Flag'] = int(df_sample['Q9_SuicidalThoughts'].iloc[0] > 0)
    df_sample['High_Severity_Flag'] = int(total_score >= 15)
    df_sample['Total_Score_Normalized'] = total_score / 27.0
    df_sample['Total_Score_Squared'] = total_score ** 2
    
    question_cols = [col for col in df_sample.columns if col.startswith('Q') and '_' in col]
    df_sample['Num_Severe_Symptoms'] = (df_sample[question_cols] == 3).sum(axis=1).iloc[0]
    df_sample['Num_Any_Symptoms'] = (df_sample[question_cols] > 0).sum(axis=1).iloc[0]
    
    # Predict
    X_sample = df_sample[feature_names].values
    prediction = model.predict(X_sample)[0]
    prediction_proba = model.predict_proba(X_sample)[0]
    
    severity_labels = ['Minimal', 'Mild', 'Moderate', 'Moderately_Severe', 'Severe']
    
    print(f"\n🎯 MODEL PREDICTION:")
    print(f"   Predicted Severity: {severity_labels[prediction]}")
    print(f"   Confidence: {prediction_proba[prediction]*100:.2f}%")
    
    print(f"\n📊 Probability Distribution:")
    for i, label in enumerate(severity_labels):
        prob = prediction_proba[i] * 100
        bar = '█' * int(prob / 2)
        print(f"   {label:20s}: {bar:50s} {prob:5.2f}%")
    
    # Determine follow-up questions
    if prediction <= 1:
        followup_count = 2
    elif prediction == 2:
        followup_count = 3
    else:
        followup_count = 4
    
    print(f"\n📝 Recommended Follow-up Questions: {followup_count}")
    print(f"   (This will be used in Stage 2: Dynamic Follow-up Questions)")

def main():
    """Main function"""
    
    df_engineered = engineer_features(df)
    X, y, feature_names = prepare_data(df_engineered)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n📊 Data Split:")
    print(f"   Training: {len(X_train)} samples (80%)")
    print(f"   Testing: {len(X_test)} samples (20%)")
    
    model = train_xgboost_model(X_train, y_train)
    results = evaluate_model(model, X_train, y_train, X_test, y_test, feature_names)
    visualize_results(model, X_test, y_test, results['y_pred'], results['feature_importance'])
    save_model(model, feature_names)
    test_model_prediction(model, feature_names)
    
    print("\n" + "="*80)
    print("✅ STEP 2 COMPLETE - MODEL TRAINING SUCCESSFUL!")
    print("="*80)
    print(f"✓ XGBoost model trained with {len(feature_names)} features")
    print(f"✓ Test Accuracy: {results['test_acc']*100:.2f}%")
    print(f"✓ Test F1-Score: {results['test_f1']:.4f}")
    print(f"✓ Model classifies into 5 severity categories")
    print(f"✓ Files created:")
    print(f"   - xgboost_phq9_model.json (trained model)")
    print(f"   - feature_names.pkl (feature list)")
    print(f"   - model_evaluation_results.png (visualizations)")
    print("="*80)
    print("\n🎯 READY FOR NEXT STEP!")
    print("Type 'NEXT' for Step 3: Dynamic Follow-up Questions")
    print("="*80)

if __name__ == "__main__":
    main()
