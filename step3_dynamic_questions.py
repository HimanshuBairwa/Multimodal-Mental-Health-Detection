"""
STEP 3: Intelligent Dynamic Follow-up Question System
Personalized questions based on severity + individual symptom patterns
"""

import pandas as pd
import numpy as np
import pickle
import xgboost as xgb

print("="*80)
print("🚀 STEP 3: INTELLIGENT DYNAMIC FOLLOW-UP QUESTION SYSTEM")
print("="*80)

# Load trained model
print("\n📥 Loading trained model...")
model = xgb.XGBClassifier()
model.load_model('xgboost_phq9_model.json')

with open('feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)

print("✅ Model loaded successfully!")

# ============================================================================
# QUESTION POOL - Organized by symptom category
# ============================================================================

QUESTION_POOL = {
    # CRITICAL - Suicidality questions (Q9 > 0)
    'suicidality': {
        'priority': 10,  # Highest priority
        'questions': [
            "Have you had thoughts of harming yourself or ending your life in the past 2 weeks?",
            "Do you have a specific plan for how you would harm yourself?",
            "Do you have access to means (like weapons, pills) to harm yourself?",
            "Is there someone you can call if you have thoughts of self-harm?",
            "Have you told anyone about these thoughts?"
        ]
    },
    
    # HIGH SEVERITY - Core depression symptoms (Q1, Q2 scored 2-3)
    'core_depression': {
        'priority': 9,
        'questions': [
            "How long have you been feeling this way (depressed/hopeless)?",
            "Has anything specific triggered these feelings recently?",
            "Do you feel worse at certain times of the day?",
            "Have you experienced depression before? If yes, what helped?",
            "Are you currently receiving any treatment or therapy?"
        ]
    },
    
    # Sleep problems (Q3 scored 2-3)
    'sleep_issues': {
        'priority': 7,
        'questions': [
            "On average, how many hours do you sleep per night?",
            "Do you have trouble falling asleep, staying asleep, or both?",
            "Do you wake up feeling refreshed or still tired?",
            "Have you tried any strategies to improve your sleep?"
        ]
    },
    
    # Energy/fatigue (Q4 scored 2-3)
    'energy_fatigue': {
        'priority': 6,
        'questions': [
            "Does the fatigue affect your ability to work or study?",
            "Do you feel tired even after a full night's sleep?",
            "Have you noticed any physical symptoms with the fatigue (pain, headaches)?",
            "When was the last time you felt energetic?"
        ]
    },
    
    # Appetite/weight (Q5 scored 2-3)
    'appetite_weight': {
        'priority': 6,
        'questions': [
            "Have you noticed weight changes (gain or loss) recently?",
            "Do you eat more, less, or the same as usual?",
            "Do you eat for comfort or stress relief?",
            "Are you able to enjoy your favorite foods?"
        ]
    },
    
    # Self-worth/guilt (Q6 scored 2-3)
    'self_worth': {
        'priority': 8,
        'questions': [
            "Do you often blame yourself for things that aren't your fault?",
            "What thoughts go through your mind when you feel like a failure?",
            "Can you think of any positive qualities about yourself?",
            "Do these feelings affect your relationships with others?"
        ]
    },
    
    # Concentration (Q7 scored 2-3)
    'concentration': {
        'priority': 5,
        'questions': [
            "Is it hard to focus on work, studies, or reading?",
            "Do you find your mind wandering or racing?",
            "Have you noticed memory problems?",
            "Does this affect your daily tasks or decision-making?"
        ]
    },
    
    # Psychomotor (Q8 scored 2-3)
    'psychomotor': {
        'priority': 5,
        'questions': [
            "Have others noticed you moving or speaking more slowly?",
            "Do you feel restless or unable to sit still?",
            "Is it harder to complete physical tasks than before?",
            "Do you feel physically heavy or sluggish?"
        ]
    },
    
    # Social support (General for Moderate+ severity)
    'social_support': {
        'priority': 4,
        'questions': [
            "Do you have friends or family you can talk to about your feelings?",
            "Have you withdrawn from social activities you used to enjoy?",
            "Do you feel isolated or alone?",
            "Is there someone who checks in on you regularly?"
        ]
    },
    
    # Coping strategies (General for Mild/Moderate)
    'coping': {
        'priority': 3,
        'questions': [
            "What activities help you feel better, even temporarily?",
            "Do you exercise, meditate, or practice mindfulness?",
            "Have you tried journaling or creative outlets?",
            "What gives you hope or motivation to keep going?"
        ]
    },
    
    # Professional help (Moderate+ severity)
    'professional_help': {
        'priority': 7,
        'questions': [
            "Are you currently seeing a therapist or counselor?",
            "Have you ever taken medication for depression or anxiety?",
            "Would you be open to professional mental health support?",
            "What barriers (if any) prevent you from seeking help?"
        ]
    },
    
    # Daily functioning (Moderate+ severity)
    'daily_functioning': {
        'priority': 6,
        'questions': [
            "How is your depression affecting your work or school performance?",
            "Are you able to complete daily tasks (showering, eating, cleaning)?",
            "Have you missed work or school because of how you're feeling?",
            "Do you feel capable of taking care of yourself right now?"
        ]
    },
    
    # Timeline/duration (All severities)
    'timeline': {
        'priority': 8,
        'questions': [
            "When did you first notice these symptoms?",
            "Have the symptoms gotten worse, better, or stayed the same?",
            "Is this your first time experiencing depression?",
            "How long do these episodes typically last for you?"
        ]
    }
}


def analyze_symptom_pattern(phq9_responses):
    """
    Analyze which specific symptoms are problematic
    Returns list of symptom categories that need attention
    """
    print("\n🔍 Analyzing symptom patterns...")
    
    symptom_categories = []
    
    # Map PHQ-9 questions to symptom categories
    q1, q2, q3, q4, q5, q6, q7, q8, q9 = phq9_responses
    
    # CRITICAL: Suicidality
    if q9 > 0:
        symptom_categories.append(('suicidality', QUESTION_POOL['suicidality']['priority'], q9))
        print(f"   ⚠️  CRITICAL: Suicidal thoughts detected (Q9={q9})")
    
    # Core depression (anhedonia, hopelessness)
    if q1 >= 2 or q2 >= 2:
        symptom_categories.append(('core_depression', QUESTION_POOL['core_depression']['priority'], max(q1, q2)))
        print(f"   🔴 Core depression symptoms: Q1={q1}, Q2={q2}")
    
    # Sleep problems
    if q3 >= 2:
        symptom_categories.append(('sleep_issues', QUESTION_POOL['sleep_issues']['priority'], q3))
        print(f"   😴 Sleep issues detected (Q3={q3})")
    
    # Energy/fatigue
    if q4 >= 2:
        symptom_categories.append(('energy_fatigue', QUESTION_POOL['energy_fatigue']['priority'], q4))
        print(f"   ⚡ Energy/fatigue issues (Q4={q4})")
    
    # Appetite
    if q5 >= 2:
        symptom_categories.append(('appetite_weight', QUESTION_POOL['appetite_weight']['priority'], q5))
        print(f"   🍽️  Appetite problems (Q5={q5})")
    
    # Self-worth/guilt
    if q6 >= 2:
        symptom_categories.append(('self_worth', QUESTION_POOL['self_worth']['priority'], q6))
        print(f"   💭 Self-worth issues (Q6={q6})")
    
    # Concentration
    if q7 >= 2:
        symptom_categories.append(('concentration', QUESTION_POOL['concentration']['priority'], q7))
        print(f"   🧠 Concentration problems (Q7={q7})")
    
    # Psychomotor
    if q8 >= 2:
        symptom_categories.append(('psychomotor', QUESTION_POOL['psychomotor']['priority'], q8))
        print(f"   🏃 Psychomotor changes (Q8={q8})")
    
    # Sort by priority (highest first) and symptom severity
    symptom_categories.sort(key=lambda x: (x[1], x[2]), reverse=True)
    
    return symptom_categories


def select_intelligent_followup_questions(severity_category, severity_label, phq9_responses):
    """
    Intelligently select follow-up questions based on:
    1. Severity category (0-4)
    2. Specific symptom patterns (which questions scored high)
    3. Priority of symptoms (suicidality = highest)
    """
    print("\n" + "="*80)
    print("📝 SELECTING INTELLIGENT FOLLOW-UP QUESTIONS")
    print("="*80)
    
    print(f"\n🎯 Severity: {severity_label} (Category {severity_category})")
    
    # Analyze which symptoms are problematic
    symptom_categories = analyze_symptom_pattern(phq9_responses)
    
    # Determine number of follow-up questions based on severity
    if severity_category <= 1:  # Minimal or Mild
        num_questions = 2
    elif severity_category == 2:  # Moderate
        num_questions = 3
    else:  # Moderately Severe or Severe
        num_questions = 4
    
    print(f"\n📊 Number of follow-up questions: {num_questions}")
    
    selected_questions = []
    used_categories = set()
    
    # PHASE 1: Add questions from detected symptom categories (highest priority first)
    for category, priority, severity_score in symptom_categories:
        if len(selected_questions) >= num_questions:
            break
        
        if category not in used_categories:
            # Pick 1-2 questions from this category based on severity
            questions = QUESTION_POOL[category]['questions']
            
            if category == 'suicidality':
                # For suicidality, ask 2 questions if severe, 1 if mild
                num_from_category = 2 if severity_score >= 2 else 1
            else:
                num_from_category = 1
            
            for i in range(min(num_from_category, len(questions))):
                if len(selected_questions) < num_questions:
                    selected_questions.append({
                        'category': category,
                        'question': questions[i],
                        'priority': priority,
                        'reason': f"Based on Q{phq9_responses.index(severity_score)+1} (score={severity_score})"
                    })
            
            used_categories.add(category)
    
    # PHASE 2: Fill remaining slots with general questions based on severity
    if len(selected_questions) < num_questions:
        # Add timeline question (important for all)
        if 'timeline' not in used_categories:
            selected_questions.append({
                'category': 'timeline',
                'question': QUESTION_POOL['timeline']['questions'][0],
                'priority': QUESTION_POOL['timeline']['priority'],
                'reason': 'General assessment'
            })
            used_categories.add('timeline')
    
    if len(selected_questions) < num_questions and severity_category >= 2:
        # Add professional help question for moderate+
        if 'professional_help' not in used_categories:
            selected_questions.append({
                'category': 'professional_help',
                'question': QUESTION_POOL['professional_help']['questions'][0],
                'priority': QUESTION_POOL['professional_help']['priority'],
                'reason': 'Moderate+ severity assessment'
            })
            used_categories.add('professional_help')
    
    if len(selected_questions) < num_questions and severity_category >= 2:
        # Add daily functioning for moderate+
        if 'daily_functioning' not in used_categories:
            selected_questions.append({
                'category': 'daily_functioning',
                'question': QUESTION_POOL['daily_functioning']['questions'][0],
                'priority': QUESTION_POOL['daily_functioning']['priority'],
                'reason': 'Functional impairment check'
            })
            used_categories.add('daily_functioning')
    
    if len(selected_questions) < num_questions:
        # Add social support for all
        if 'social_support' not in used_categories:
            selected_questions.append({
                'category': 'social_support',
                'question': QUESTION_POOL['social_support']['questions'][0],
                'priority': QUESTION_POOL['social_support']['priority'],
                'reason': 'Support system assessment'
            })
            used_categories.add('social_support')
    
    if len(selected_questions) < num_questions and severity_category <= 2:
        # Add coping for mild/moderate
        if 'coping' not in used_categories:
            selected_questions.append({
                'category': 'coping',
                'question': QUESTION_POOL['coping']['questions'][0],
                'priority': QUESTION_POOL['coping']['priority'],
                'reason': 'Coping strategy exploration'
            })
    
    # Display selected questions
    print(f"\n✅ Selected {len(selected_questions)} personalized follow-up questions:\n")
    for i, q in enumerate(selected_questions, 1):
        print(f"{i}. [{q['category'].upper()}] {q['question']}")
        print(f"   Reason: {q['reason']}\n")
    
    return selected_questions


def predict_and_get_questions(phq9_responses, age=25, gender=1):
    """
    Main function: Predict severity and generate personalized questions
    """
    print("\n" + "="*80)
    print("🤖 RUNNING COMPLETE PREDICTION + QUESTION GENERATION")
    print("="*80)
    
    # Display PHQ-9 responses
    print("\n📋 PHQ-9 Responses:")
    question_labels = [
        'Q1: Little interest/pleasure',
        'Q2: Feeling down/depressed',
        'Q3: Sleep problems',
        'Q4: Feeling tired/low energy',
        'Q5: Appetite problems',
        'Q6: Feeling bad about self',
        'Q7: Concentration problems',
        'Q8: Psychomotor changes',
        'Q9: Suicidal thoughts'
    ]
    
    for i, (label, score) in enumerate(zip(question_labels, phq9_responses)):
        print(f"   {label}: {score}")
    
    total_score = sum(phq9_responses)
    print(f"\n   Total PHQ-9 Score: {total_score}/27")
    
    # Engineer features (same as training)
    sample = {
        'Q1_LittleInterest': phq9_responses[0],
        'Q2_FeelingDown': phq9_responses[1],
        'Q3_SleepProblems': phq9_responses[2],
        'Q4_FeelingTired': phq9_responses[3],
        'Q5_AppetiteProblems': phq9_responses[4],
        'Q6_FeelingBad': phq9_responses[5],
        'Q7_ConcentrationProblems': phq9_responses[6],
        'Q8_Psychomotor': phq9_responses[7],
        'Q9_SuicidalThoughts': phq9_responses[8],
        'Age': age,
        'Gender': gender
    }
    
    df = pd.DataFrame([sample])
    
    # Feature engineering
    df['Q1xQ2'] = df['Q1_LittleInterest'] * df['Q2_FeelingDown']
    df['Q1xQ6'] = df['Q1_LittleInterest'] * df['Q6_FeelingBad']
    df['Q2xQ6'] = df['Q2_FeelingDown'] * df['Q6_FeelingBad']
    df['Somatic_Cluster'] = df['Q3_SleepProblems'] + df['Q4_FeelingTired'] + df['Q5_AppetiteProblems']
    df['Cognitive_Cluster'] = df['Q6_FeelingBad'] + df['Q7_ConcentrationProblems']
    df['Core_Mood_Cluster'] = df['Q1_LittleInterest'] + df['Q2_FeelingDown']
    df['Suicidality_Flag'] = int(df['Q9_SuicidalThoughts'].iloc[0] > 0)
    df['High_Severity_Flag'] = int(total_score >= 15)
    df['Total_Score_Normalized'] = total_score / 27.0
    df['Total_Score_Squared'] = total_score ** 2
    
    question_cols = [col for col in df.columns if col.startswith('Q') and '_' in col]
    df['Num_Severe_Symptoms'] = (df[question_cols] == 3).sum(axis=1).iloc[0]
    df['Num_Any_Symptoms'] = (df[question_cols] > 0).sum(axis=1).iloc[0]
    
    # Predict
    X = df[feature_names].values
    prediction = model.predict(X)[0]
    prediction_proba = model.predict_proba(X)[0]
    
    severity_labels = ['Minimal', 'Mild', 'Moderate', 'Moderately_Severe', 'Severe']
    predicted_label = severity_labels[prediction]
    
    print(f"\n🎯 PREDICTED SEVERITY: {predicted_label}")
    print(f"   Confidence: {prediction_proba[prediction]*100:.2f}%")
    
    # Select intelligent follow-up questions
    questions = select_intelligent_followup_questions(prediction, predicted_label, phq9_responses)
    
    return {
        'severity_category': prediction,
        'severity_label': predicted_label,
        'confidence': prediction_proba[prediction],
        'probability_distribution': dict(zip(severity_labels, prediction_proba)),
        'follow_up_questions': questions,
        'total_score': total_score
    }


def test_multiple_scenarios():
    """Test the system with different patient scenarios"""
    print("\n" + "="*80)
    print("🧪 TESTING WITH MULTIPLE PATIENT SCENARIOS")
    print("="*80)
    
    scenarios = [
        {
            'name': 'Patient 1: Minimal Depression',
            'responses': [0, 1, 1, 1, 0, 0, 0, 0, 0]  # Total = 3
        },
        {
            'name': 'Patient 2: Moderate with Sleep Issues',
            'responses': [2, 2, 3, 2, 1, 1, 1, 0, 0]  # Total = 12, Q3=3
        },
        {
            'name': 'Patient 3: Severe with Suicidal Thoughts',
            'responses': [3, 3, 2, 3, 2, 3, 2, 1, 2]  # Total = 21, Q9=2
        },
        {
            'name': 'Patient 4: Moderate with Self-Worth Issues',
            'responses': [2, 2, 1, 2, 1, 3, 2, 1, 0]  # Total = 14, Q6=3
        }
    ]
    
    for scenario in scenarios:
        print("\n" + "="*80)
        print(f"🔹 {scenario['name']}")
        print("="*80)
        result = predict_and_get_questions(scenario['responses'])
        input("\nPress Enter to continue to next scenario...")


def main():
    """Main function for Step 3"""
    
    # Example: Test with a moderate-severe patient
    print("\n" + "="*80)
    print("📊 EXAMPLE: Moderately Severe Patient")
    print("="*80)
    
    # PHQ-9 responses: [Q1, Q2, Q3, Q4, Q5, Q6, Q7, Q8, Q9]
    example_responses = [2, 2, 2, 3, 1, 2, 2, 1, 0]  # Total = 15
    
    result = predict_and_get_questions(example_responses, age=25, gender=1)
    
    print("\n" + "="*80)
    print("✅ STEP 3 COMPLETE - DYNAMIC QUESTION SYSTEM READY!")
    print("="*80)
    print(f"✓ Intelligent question selection based on:")
    print(f"   - Severity category: {result['severity_label']}")
    print(f"   - Individual symptom patterns")
    print(f"   - Priority-based selection (suicidality = highest)")
    print(f"✓ Each patient gets personalized questions!")
    print(f"✓ {len(result['follow_up_questions'])} questions selected")
    print("="*80)
    
    # Ask if user wants to test more scenarios
    print("\n🧪 Would you like to test with multiple patient scenarios?")
    choice = input("Type 'yes' to test, or press Enter to finish: ").lower()
    
    if choice == 'yes':
        test_multiple_scenarios()
    
    print("\n" + "="*80)
    print("🎯 STAGES 1 & 2 COMPLETE!")
    print("="*80)
    print("✅ Stage 1: PHQ-9 Classification Model (100% accuracy)")
    print("✅ Stage 2: Intelligent Dynamic Questions (personalized)")
    print("\n🚀 NEXT: Stage 3 - Audio Analysis Models")
    print("   - Speech-to-Text (Whisper)")
    print("   - Text Emotion (RoBERTa)")
    print("   - Speech Emotion (Wav2Vec2)")
    print("="*80)
    print("\nType 'NEXT' when ready for audio models!")
    print("="*80)


if __name__ == "__main__":
    main()
