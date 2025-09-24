import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pickle

# Load the Excel dataset
df = pd.read_excel("flightpredection-main/Data_Train.xlsx")

# =======================
# 1. Parse Dates and Times
# =======================

# Journey Date
df["Journey_day"] = pd.to_datetime(df["Date_of_Journey"], dayfirst=True).dt.day
df["Journey_month"] = pd.to_datetime(df["Date_of_Journey"], dayfirst=True).dt.month
df.drop(["Date_of_Journey"], axis=1, inplace=True)

# Departure Time
df["Dep_hour"] = pd.to_datetime(df["Dep_Time"], errors='coerce').dt.hour
df["Dep_min"] = pd.to_datetime(df["Dep_Time"], errors='coerce').dt.minute
df.drop(["Dep_Time"], axis=1, inplace=True)

# Arrival Time
df["Arrival_hour"] = pd.to_datetime(df["Arrival_Time"], errors='coerce').dt.hour
df["Arrival_min"] = pd.to_datetime(df["Arrival_Time"], errors='coerce').dt.minute
df.drop(["Arrival_Time"], axis=1, inplace=True)

# =======================
# 2. Convert Duration
# =======================

# Normalize duration text
df["Duration"] = df["Duration"].apply(lambda x: x.strip())

# Extract hours and minutes safely
df["Dur_hour"] = df["Duration"].apply(lambda x: int(x.split('h')[0].strip()) if 'h' in x else 0)
df["Dur_min"] = df["Duration"].apply(lambda x: int(x.split('m')[0].split()[-1].strip()) if 'm' in x else 0)
df.drop(["Duration"], axis=1, inplace=True)

# =======================
# 3. Clean Total Stops
# =======================

stop_map = {
    "non-stop": 0,
    "1 stop": 1,
    "2 stops": 2,
    "3 stops": 3,
    "4 stops": 4
}
df["Total_Stops"] = df["Total_Stops"].map(stop_map)
df["Total_Stops"] = df["Total_Stops"].fillna(0).astype(int)

# =======================
# 4. Drop Unused Columns
# =======================

df.drop(["Route", "Additional_Info"], axis=1, inplace=True)

# =======================
# 5. One-Hot Encoding
# =======================

df = pd.get_dummies(df, columns=["Airline", "Source", "Destination"], drop_first=True)

# =======================
# 6. Train Model
# =======================

X = df.drop("Price", axis=1)
y = df["Price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor()
model.fit(X_train, y_train)

# =======================
# 7. Save Model
# =======================

# Save feature names
with open("model_features.pkl", "wb") as f:
    pickle.dump(X.columns.tolist(), f)


print("✅ Model trained and saved as flight_rf.pkl")
