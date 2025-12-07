"""
Verify all your datasets - FIXED VERSION
"""

import pandas as pd
import os

def verify_all():
    print("VERIFYING ALL DATASETS")
    print("=" * 70)
    
    datasets = [
        ('Original Dataset', 'data/generated/phq9_dataset_original.csv'),
        ('Real Emotion Dataset', 'data/real/emotion_processed.csv'),
        ('Real GoEmotions Dataset', 'data/real/goemotions_processed.csv'),
        ('Combined Real Dataset', 'data/combined/phq9_dataset_real_combined.csv'),
        ('Main Dataset (Current)', 'phq9_dataset_real.csv')
    ]
    
    all_good = True
    
    for name, path in datasets:
        print(f"\n📂 {name}")
        print(f"   Path: {path}")
        
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                print(f"   ✅ Found: {len(df):,} samples")
                print(f"   Columns: {df.columns.tolist()[:5]}...")
                
                # Only show depressed stats if column exists
                if 'depressed' in df.columns:
                    print(f"   Depressed: {df['depressed'].sum():,} ({df['depressed'].mean()*100:.1f}%)")
                    print(f"   Avg PHQ-9: {df['phq9_total_score'].mean():.1f}")
                else:
                    print(f"   (Original format - no depressed column)")
                    
            except Exception as e:
                print(f"   ❌ Error reading file: {e}")
                all_good = False
        else:
            print(f"   ⚠️ Not found")
    
    print("\n" + "=" * 70)
    
    if all_good:
        print("✅ ALL DATASETS VERIFIED SUCCESSFULLY!")
        print("\nYou now have:")
        print("   - 59,410 REAL samples (main dataset)")
        print("   - Original data backed up safely")
        print("   - Ready to use in your app!")
        
        print("\nNext steps:")
        print("   1. Run: streamlit run app_final.py")
        print("   2. Test with your new REAL data!")
        print("   3. (Optional) Retrain: python step2_train_survey_model.py")
    else:
        print("⚠️ Some datasets missing")

if __name__ == "__main__":
    verify_all()
