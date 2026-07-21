import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load Dataset
df = pd.read_csv("Ecommerce.csv")

# ----------------------------------
# Create Intent Pathway Target
# ----------------------------------

def create_intent(row):
    if row['purchased'] == 1:
        return "Purchase"
    elif row['added_to_cart'] == 1:
        return "Add_to_Cart"
    elif row['cart_abandoned'] == 1:
        return "Cart_Abandonment"
    else:
        return "Browsing"

df['intent_pathway'] = df.apply(create_intent, axis=1)

# ----------------------------------
# Features
# ----------------------------------

features = [
    'device_type',
    'user_type',
    'marketing_channel',
    'product_category',
    'unit_price',
    'quantity',
    'discount_percent',
    'pages_viewed',
    'time_on_site_sec',
    'rating',
    'payment_method',
    'visit_day',
    'visit_month',
    'visit_weekday',
    'location'
]

X = df[features]
y = df['intent_pathway']

# ----------------------------------
# Train Test Split
# ----------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ----------------------------------
# Train Model
# ----------------------------------

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

# ----------------------------------
# Evaluation
# ----------------------------------

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# ----------------------------------
# Save Model
# ----------------------------------

joblib.dump(model, "intent_model.pkl")

print("Model Saved Successfully")