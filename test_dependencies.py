#test_dependencies.py
import sys

def test_imports():
    print("Testing dependencies...")
    print("-" * 60)

    # Test 1: Whisper
    try:
        import whisper
        print("✅ openai-whisper: OK")
    except ImportError:
        print("❌ openai-whisper: MISSING")
        print("   Install: pip install openai-whisper")

    # Test 2: Librosa
    try:
        import librosa
        print("✅ librosa: OK")
    except ImportError:
        print("❌ librosa: MISSING")
        print("   Install: pip install librosa")

    # Test 3: Soundfile
    try:
        import soundfile
        print("✅ soundfile: OK")
    except ImportError:
        print("❌ soundfile: MISSING")
        print("   Install: pip install soundfile")

    # Test 4: Transformers
    try:
        import transformers
        print("✅ transformers: OK")
    except ImportError:
        print("❌ transformers: MISSING")
        print("   Install: pip install transformers")

    # Test 5: XGBoost
    try:
        import xgboost
        print("✅ xgboost: OK")
    except ImportError:
        print("❌ xgboost: MISSING")
        print("   Install: pip install xgboost")

    # Test 6: PyTorch
    try:
        import torch
        print(f"✅ torch: OK (version {torch.__version__})")
    except ImportError:
        print("❌ torch: MISSING")
        print("   Install: pip install torch")

    print("-" * 60)
    print("Testing complete!")

if __name__ == "__main__":
    test_imports()