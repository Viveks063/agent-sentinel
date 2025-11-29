import pandas as pd
import requests
from tqdm import tqdm
import time

API_URL = "http://localhost:8000/analyze-text"

def call_api(text, title, retry=3):
    """Call API with retry logic and error handling."""
    for attempt in range(retry):
        try:
            res = requests.post(
                API_URL, 
                json={"text": text, "title": title},
                timeout=30  # 30 second timeout
            )
            if res.status_code == 200:
                return res.json().get("falsehood_score", 0.5)
            else:
                print(f"\n⚠️ API returned status {res.status_code}")
                time.sleep(1)
        except requests.exceptions.Timeout:
            print(f"\n⏱️ Timeout on attempt {attempt + 1}/{retry}")
            time.sleep(2)
        except Exception as e:
            print(f"\n❌ Error on attempt {attempt + 1}/{retry}: {e}")
            time.sleep(1)
    
    print("\n⚠️ Failed after all retries, using neutral score 0.5")
    return 0.5


def evaluate(df, actual_label, desc="Evaluating"):
    """Evaluate a dataframe of articles."""
    tp = tn = fp = fn = 0
    errors = 0

    for idx, row in tqdm(df.iterrows(), total=len(df), desc=desc):
        text = str(row.get("text", ""))
        title = str(row.get("title", text[:80]))
        
        # Skip empty rows
        if not text or len(text) < 10:
            errors += 1
            continue

        score = call_api(text, title)
        predicted = "FAKE" if score >= 0.5 else "REAL"

        if actual_label == "FAKE" and predicted == "FAKE":
            tp += 1
        elif actual_label == "REAL" and predicted == "REAL":
            tn += 1
        elif actual_label == "REAL" and predicted == "FAKE":
            fp += 1
        elif actual_label == "FAKE" and predicted == "REAL":
            fn += 1

    if errors > 0:
        print(f"\n⚠️ Skipped {errors} empty/invalid rows")

    return tp, tn, fp, fn


def main():
    print("="*70)
    print("🧪 AGENT SENTINEL - DATASET EVALUATION")
    print("="*70)
    
    # Load datasets
    print("\n📂 Loading datasets...")
    try:
        fake_df = pd.read_csv("Fake.csv")
        real_df = pd.read_csv("True.csv")
        
        print(f"✅ Loaded {len(fake_df)} fake articles")
        print(f"✅ Loaded {len(real_df)} real articles")
        
        # Sample 200 from each (or use all if you want)
        sample_size = 200
        fake_sample = fake_df.sample(min(sample_size, len(fake_df)), random_state=42)
        real_sample = real_df.sample(min(sample_size, len(real_df)), random_state=42)
        
        print(f"\n🎲 Testing with {len(fake_sample)} fake + {len(real_sample)} real articles")
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("Make sure Fake.csv and True.csv are in the current directory!")
        return
    
    # Check if API is running
    print("\n🔗 Checking if API is running...")
    try:
        response = requests.get("http://localhost:8000/status", timeout=5)
        if response.status_code == 200:
            print("✅ API is online!")
        else:
            print("⚠️ API returned unexpected status")
    except:
        print("❌ ERROR: API is not running!")
        print("Start the server first: uvicorn main:app --reload")
        return
    
    # Initialize counters
    tp = tn = fp = fn = 0
    
    # Evaluate FAKE articles
    print("\n" + "="*70)
    print("📰 Evaluating FAKE NEWS articles...")
    print("="*70)
    start_time = time.time()
    
    a, b, c, d = evaluate(fake_sample, "FAKE", desc="Testing Fake News")
    tp += a; tn += b; fp += c; fn += d
    
    fake_time = time.time() - start_time
    print(f"⏱️ Fake news evaluation took {fake_time:.1f} seconds")
    
    # Evaluate REAL articles
    print("\n" + "="*70)
    print("📰 Evaluating REAL NEWS articles...")
    print("="*70)
    start_time = time.time()
    
    a, b, c, d = evaluate(real_sample, "REAL", desc="Testing Real News")
    tp += a; tn += b; fp += c; fn += d
    
    real_time = time.time() - start_time
    print(f"⏱️ Real news evaluation took {real_time:.1f} seconds")
    
    # Calculate metrics
    total = tp + tn + fp + fn
    
    if total == 0:
        print("\n❌ No articles were processed!")
        return
    
    accuracy = (tp + tn) / total
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Print results
    print("\n" + "="*70)
    print("📊 FINAL RESULTS")
    print("="*70)
    
    print(f"\n📈 Confusion Matrix:")
    print(f"   True Positive  (Fake detected as Fake): {tp}")
    print(f"   True Negative  (Real detected as Real): {tn}")
    print(f"   False Positive (Real detected as Fake): {fp}")
    print(f"   False Negative (Fake detected as Real): {fn}")
    
    print(f"\n🎯 Performance Metrics:")
    print(f"   Total Test Articles: {total}")
    print(f"   Accuracy:  {accuracy*100:.2f}%")
    print(f"   Precision: {precision*100:.2f}%")
    print(f"   Recall:    {recall*100:.2f}%")
    print(f"   F1 Score:  {f1:.3f}")
    
    print(f"\n⏱️ Total Time: {(fake_time + real_time):.1f} seconds")
    print(f"   Average per article: {(fake_time + real_time)/total:.2f} seconds")
    
    # Rating
    if accuracy >= 0.90:
        rating = "🌟 EXCELLENT"
    elif accuracy >= 0.80:
        rating = "✅ GOOD"
    elif accuracy >= 0.70:
        rating = "⚠️ NEEDS IMPROVEMENT"
    else:
        rating = "❌ POOR"
    
    print(f"\n🏆 Overall Rating: {rating}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()