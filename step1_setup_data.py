"""
STEP 1: Environment Setup & PHQ-9 Dataset Generation
Depression Detection System - National Level Project
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("🚀 DEPRESSION DETECTION SYSTEM - STEP 1: DATA PREPARATION")
print("="*70)

# Set random seed for reproducibility
np.random.seed(42)

def generate_phq9_dataset(n_samples=1000):
    """
    Generate realistic PHQ-9 dataset for training
    
    PHQ-9 Questions:
    Q1: Little interest or pleasure in doing things
    Q2: Feeling down, depressed, or hopeless
    Q3: Trouble falling/staying asleep or sleeping too much
    Q4: Feeling tired or having little energy
    Q5: Poor appetite or overeating
    Q6: Feeling bad about yourself or that you're a failure
    Q7: Trouble concentrating on things
    Q8: Moving or speaking slowly or being fidgety
    Q9: Thoughts of being better off dead or hurting yourself
    
    Each question scored 0-3:
    0 = Not at all
    1 = Several days
    2 = More than half the days
    3 = Nearly every day
    """
    
    print(f"\n📊 Generating {n_samples} PHQ-9 samples...")
    
    data = []
    
    for i in range(n_samples):
        # Randomly decide if person is depressed (60% depressed, 40% not)
        is_depressed = np.random.choice([0, 1], p=[0.4, 0.6])
        
        if is_depressed:
            # Depressed individuals: higher scores (typically 10-27)
            q1 = np.random.choice([1, 2, 3], p=[0.2, 0.4, 0.4])  # Low mood
            q2 = np.random.choice([1, 2, 3], p=[0.3, 0.4, 0.3])  # Hopelessness
            q3 = np.random.choice([1, 2, 3], p=[0.3, 0.4, 0.3])  # Sleep issues
            q4 = np.random.choice([1, 2, 3], p=[0.2, 0.4, 0.4])  # Low energy
            q5 = np.random.choice([0, 1, 2, 3], p=[0.3, 0.3, 0.3, 0.1])  # Appetite
            q6 = np.random.choice([1, 2, 3], p=[0.3, 0.4, 0.3])  # Guilt/failure
            q7 = np.random.choice([1, 2, 3], p=[0.3, 0.4, 0.3])  # Concentration
            q8 = np.random.choice([0, 1, 2], p=[0.5, 0.3, 0.2])  # Psychomotor
            q9 = np.random.choice([0, 1, 2, 3], p=[0.7, 0.2, 0.07, 0.03])  # Suicidality
        else:
            # Non-depressed: lower scores (typically 0-9)
            q1 = np.random.choice([0, 1, 2], p=[0.6, 0.3, 0.1])
            q2 = np.random.choice([0, 1, 2], p=[0.6, 0.3, 0.1])
            q3 = np.random.choice([0, 1, 2], p=[0.5, 0.3, 0.2])
            q4 = np.random.choice([0, 1, 2], p=[0.5, 0.3, 0.2])
            q5 = np.random.choice([0, 1], p=[0.7, 0.3])
            q6 = np.random.choice([0, 1], p=[0.7, 0.3])
            q7 = np.random.choice([0, 1, 2], p=[0.6, 0.3, 0.1])
            q8 = np.random.choice([0, 1], p=[0.8, 0.2])
            q9 = np.random.choice([0, 1], p=[0.95, 0.05])
        
        # Calculate total PHQ-9 score (0-27)
        total_score = q1 + q2 + q3 + q4 + q5 + q6 + q7 + q8 + q9
        
        # Determine severity category (5 categories)
        if total_score <= 4:
            severity = 0
            severity_label = "Minimal"
        elif total_score <= 9:
            severity = 1
            severity_label = "Mild"
        elif total_score <= 14:
            severity = 2
            severity_label = "Moderate"
        elif total_score <= 19:
            severity = 3
            severity_label = "Moderately_Severe"
        else:  # 20-27
            severity = 4
            severity_label = "Severe"
        
        # Binary classification (for ML model)
        # Clinical cutoff: PHQ-9 ≥ 10 indicates depression
        depression_binary = 1 if total_score >= 10 else 0
        
        # Additional demographics
        age = np.random.randint(18, 65)
        gender = np.random.choice([0, 1])  # 0=Male, 1=Female
        
        data.append({
            'Q1_LittleInterest': q1,
            'Q2_FeelingDown': q2,
            'Q3_SleepProblems': q3,
            'Q4_FeelingTired': q4,
            'Q5_AppetiteProblems': q5,
            'Q6_FeelingBad': q6,
            'Q7_ConcentrationProblems': q7,
            'Q8_Psychomotor': q8,
            'Q9_SuicidalThoughts': q9,
            'Total_Score': total_score,
            'Severity_Category': severity,
            'Severity_Label': severity_label,
            'Depression_Binary': depression_binary,
            'Age': age,
            'Gender': gender
        })
    
    return pd.DataFrame(data)


def visualize_data(df):
    """Create visualizations to understand the data"""
    print("\n📈 Creating visualizations...")
    
    # Set style
    sns.set_style("whitegrid")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Severity Distribution
    severity_counts = df['Severity_Label'].value_counts()
    axes[0, 0].bar(severity_counts.index, severity_counts.values, color='skyblue', edgecolor='black')
    axes[0, 0].set_title('Distribution of Severity Categories', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Severity Level')
    axes[0, 0].set_ylabel('Number of Patients')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # 2. Binary Depression Distribution
    binary_counts = df['Depression_Binary'].value_counts()
    axes[0, 1].pie(binary_counts.values, 
                   labels=['Not Depressed', 'Depressed'],
                   autopct='%1.1f%%',
                   colors=['lightgreen', 'salmon'],
                   startangle=90)
    axes[0, 1].set_title('Binary Depression Classification', fontsize=14, fontweight='bold')
    
    # 3. PHQ-9 Score Distribution
    axes[1, 0].hist(df['Total_Score'], bins=28, color='coral', edgecolor='black', alpha=0.7)
    axes[1, 0].axvline(x=10, color='red', linestyle='--', linewidth=2, label='Clinical Cutoff (≥10)')
    axes[1, 0].set_title('Distribution of PHQ-9 Total Scores', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Total PHQ-9 Score')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].legend()
    
    # 4. Question-wise Mean Scores
    question_cols = [f'Q{i}_{name}' for i, name in enumerate([
        'LittleInterest', 'FeelingDown', 'SleepProblems', 'FeelingTired',
        'AppetiteProblems', 'FeelingBad', 'ConcentrationProblems', 
        'Psychomotor', 'SuicidalThoughts'
    ], 1)]
    
    mean_scores = df[question_cols].mean()
    axes[1, 1].barh(range(len(mean_scores)), mean_scores.values, color='mediumpurple', edgecolor='black')
    axes[1, 1].set_yticks(range(len(mean_scores)))
    axes[1, 1].set_yticklabels([f'Q{i+1}' for i in range(9)])
    axes[1, 1].set_title('Average Score per PHQ-9 Question', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Average Score')
    axes[1, 1].set_ylabel('Question')
    
    plt.tight_layout()
    plt.savefig('phq9_data_visualization.png', dpi=300, bbox_inches='tight')
    print("✅ Visualization saved as 'phq9_data_visualization.png'")
    plt.close()


def main():
    """Main function for Step 1"""
    
    # Generate dataset (1000 samples for better training)
    df = generate_phq9_dataset(n_samples=1000)
    
    print(f"✅ Dataset created with {len(df)} samples")
    print(f"\n📋 Dataset Preview:")
    print(df.head(10))
    
    print(f"\n📊 Dataset Statistics:")
    print(df.describe())
    
    print(f"\n🎯 Severity Distribution:")
    print(df['Severity_Label'].value_counts().sort_index())
    
    print(f"\n🎯 Binary Depression Labels:")
    depression_counts = df['Depression_Binary'].value_counts()
    print(f"Not Depressed (0): {depression_counts.get(0, 0)} ({depression_counts.get(0, 0)/len(df)*100:.1f}%)")
    print(f"Depressed (1): {depression_counts.get(1, 0)} ({depression_counts.get(1, 0)/len(df)*100:.1f}%)")
    
    # Save dataset
    df.to_csv('phq9_dataset.csv', index=False)
    print(f"\n💾 Dataset saved as 'phq9_dataset.csv'")
    
    # Create visualizations
    visualize_data(df)
    
    # Summary
    print("\n" + "="*70)
    print("✅ STEP 1 COMPLETE - Environment Setup Successful!")
    print("="*70)
    print(f"✓ Dataset created: {len(df)} samples")
    print(f"✓ Features: {len(df.columns)} columns")
    print(f"✓ PHQ-9 questions: Q1-Q9 (each scored 0-3)")
    print(f"✓ 5 Severity categories: Minimal, Mild, Moderate, Moderately_Severe, Severe")
    print(f"✓ Binary labels: 0 (Not Depressed), 1 (Depressed)")
    print(f"✓ Files created:")
    print(f"   - phq9_dataset.csv")
    print(f"   - phq9_data_visualization.png")
    print("="*70)
    print("\n🎯 NEXT STEP: Feature Engineering & Model Training")
    print("Type 'NEXT' when ready to proceed to Step 2!")
    print("="*70)


if __name__ == "__main__":
    main()
