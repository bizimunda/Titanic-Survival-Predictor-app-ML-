import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import plotly.express as px

# -----------------------
# Load Data
# -----------------------
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    return pd.read_csv(url)

df = load_data()

st.title("🚢 Titanic Survival Predictor")
st.write("A simple machine learning app to predict passenger survival.")

# -----------------------
# Explore Data
# -----------------------
st.subheader("🔎 Data Preview")
st.write(df.head())

# -----------------------
# Feature Selection
# -----------------------
st.subheader("⚙️ Select Features for Training")

all_features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
target = "Survived"

selected_features = st.multiselect(
    "Choose features:", all_features, default=["Pclass", "Sex", "Age", "Fare"]
)

if selected_features:
    data = df[selected_features + [target]].dropna()

    # Encode categorical variables
    data = pd.get_dummies(data, drop_first=True)

    X = data.drop(target, axis=1)
    y = data[target]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # -----------------------
    # Results
    # -----------------------
    st.subheader("📊 Model Performance")
    acc = accuracy_score(y_test, y_pred)
    st.write(f"**Accuracy:** {acc:.2f}")

    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=["Died", "Survived"], columns=["Predicted Died", "Predicted Survived"])
    fig = px.imshow(cm_df, text_auto=True, color_continuous_scale="Blues", title="Confusion Matrix")
    st.plotly_chart(fig)

    # -----------------------
    # Make Predictions
    # -----------------------
    st.subheader("🧑 Predict Survival for a Passenger")

    input_data = {}
    for feature in selected_features:
        if feature in ["Sex", "Embarked", "Pclass"]:
            val = st.selectbox(f"{feature}", df[feature].dropna().unique())
        else:
            val = st.number_input(f"{feature}", float(df[feature].min()), float(df[feature].max()))
        input_data[feature] = val

    if st.button("Predict"):
        input_df = pd.DataFrame([input_data])
        input_df = pd.get_dummies(input_df)
        input_df = input_df.reindex(columns=X.columns, fill_value=0)

        prediction = model.predict(input_df)[0]
        if prediction == 1:
            st.success("✅ This passenger would likely **Survive**!")
        else:
            st.error("❌ This passenger would likely **Not Survive**.")
