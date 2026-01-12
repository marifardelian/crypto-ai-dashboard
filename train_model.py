import pandas as pd
import requests
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Configuration
COIN_ID = "bitcoin"
DAYS = "365"

def fetch_data():
    print(f"Fetching {DAYS} days of data for {COIN_ID}...")
    url = f"https://api.coingecko.com/api/v3/coins/{COIN_ID}/market_chart"
    params = {'vs_currency': 'usd', 'days': DAYS, 'interval': 'daily'}
    
    response = requests.get(url, params=params)
    data = response.json()
    
    df = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

def train():
    df = fetch_data()
    
    # --- Feature Engineering ---
    df['MA_7'] = df['price'].rolling(window=7).mean()
    df['MA_30'] = df['price'].rolling(window=30).mean()
    df['Volatility'] = df['price'].rolling(window=7).std()
    
    # Target: 1 if Price tomorrow > Price today, else 0
    df['Target'] = (df['price'].shift(-1) > df['price']).astype(int)
    
    df = df.dropna()
    
    features = ['price', 'MA_7', 'MA_30', 'Volatility']
    X = df[features]
    y = df['Target']
    
    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # --- Define 3 Models ---
    models = {
        "Random_Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Logistic_Regression": LogisticRegression(),
        "SVM": SVC(probability=True) # probability=True needed for confidence score
    }
    
    results = {}
    
    # --- Train & Save Loop ---
    print("\nTRAINING RESULTS:")
    print("-" * 30)
    
    for name, model in models.items():
        # Train
        model.fit(X_train, y_train)
        
        # Test
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        results[name] = acc
        
        # Save
        filename = f"{name}_model.pkl"
        joblib.dump(model, filename)
        print(f"✅ {name}: {acc:.2%} accuracy (Saved to {filename})")

    print("-" * 30)
    best_model = max(results, key=results.get)
    print(f"🏆 Winner: {best_model} with {results[best_model]:.2%} accuracy")

if __name__ == "__main__":
    train()