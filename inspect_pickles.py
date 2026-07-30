import joblib
import os

files = ["top5_features.pkl", "threshold.pkl", "scaler.pkl", "cvd_model.pkl"]

print("=== CVD CDSS Pickle Files Inspector ===\n")

for f in files:
    if not os.path.exists(f):
        print(f"File {f} not found.")
        continue
        
    print(f"Reading file: {f}")
    data = joblib.load(f)
    print(f"  Type: {type(data)}")
    
    if f == "top5_features.pkl":
        print(f"  Content (Selected Features): {data}")
    elif f == "threshold.pkl":
        print(f"  Content (Classification Threshold): {data}")
    elif f == "scaler.pkl":
        print(f"  Fitted features in order: {list(data.feature_names_in_)}")
        print(f"  Means: {data.mean_}")
        print(f"  Scales: {data.scale_}")
    elif f == "cvd_model.pkl":
        print(f"  Model Parameters: {data.get_params()}")
        if hasattr(data, "feature_names_in_"):
            print(f"  Expected Features: {list(data.feature_names_in_)}")
            
    print("-" * 50)
