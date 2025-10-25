import streamlit as st
import pandas as pd
import pickle

# -----------------------------
# 1️⃣ Load the trained pipeline
# -----------------------------
with open('model.pkl', 'rb') as f:
    pipeline = pickle.load(f)

st.title("🌲 Forest Cover Type Prediction App")
st.write("Enter the environmental features below to predict the forest cover type.")

# -----------------------------
# 2️⃣ Define input fields
# -----------------------------
Elevation = st.number_input("Elevation", min_value=0, max_value=4000, value=2500)
Aspect = st.number_input("Aspect", min_value=0, max_value=360, value=90)
Slope = st.number_input("Slope", min_value=0, max_value=90, value=10)
Horizontal_Distance_To_Hydrology = st.number_input("Horizontal Distance To Hydrology", min_value=0, value=100)
Vertical_Distance_To_Hydrology = st.number_input("Vertical Distance To Hydrology", min_value=-100, value=30)
Horizontal_Distance_To_Roadways = st.number_input("Horizontal Distance To Roadways", min_value=0, value=2000)
Hillshade_9am = st.number_input("Hillshade 9am", min_value=0, max_value=255, value=220)
Hillshade_Noon = st.number_input("Hillshade Noon", min_value=0, max_value=255, value=230)
Hillshade_3pm = st.number_input("Hillshade 3pm", min_value=0, max_value=255, value=210)
Horizontal_Distance_To_Fire_Points = st.number_input("Horizontal Distance To Fire Points", min_value=0, value=1000)
Wilderness_Area_4 = st.number_input("Wilderness_Area_4", min_value=0, max_value=1)
Wilderness_Area_3 = st.number_input("Wilderness_Area_3", min_value=0, max_value=1)
Wilderness_Area_1 = st.number_input("Wilderness_Area_1", min_value=0, max_value=1)
Soil_Type_12 = st.number_input("Soil_Type_12", min_value=0, max_value=1)
Soil_Type_23 = st.number_input("Soil_Type_23", min_value=0, max_value=1)
Soil_Type_10 = st.number_input("Soil_Type_10", min_value=0, max_value=1)
Soil_Type_29 = st.number_input("Soil_Type_29", min_value=0, max_value=1)
Soil_Type_39 = st.number_input("Soil_Type_39", min_value=0, max_value=1)
Soil_Type_30 = st.number_input("Soil_Type_30", min_value=0, max_value=1)
Soil_Type_38 = st.number_input("Soil_Type_38", min_value=0, max_value=1)

# -----------------------------
# 3️⃣ Create input dataframe
# -----------------------------
input_data = pd.DataFrame([{
    "Elevation": Elevation,
    "Horizontal_Distance_To_Roadways": Horizontal_Distance_To_Roadways,
    "Horizontal_Distance_To_Fire_Points": Horizontal_Distance_To_Fire_Points,
    "Horizontal_Distance_To_Hydrology": Horizontal_Distance_To_Hydrology,
    "Vertical_Distance_To_Hydrology": Vertical_Distance_To_Hydrology,
    "Wilderness_Area_1": Wilderness_Area_1,
    "Aspect": Aspect,
    "Hillshade_3pm": Hillshade_3pm,
    "Hillshade_Noon": Hillshade_Noon,
    "Hillshade_9am": Hillshade_9am,
    "Slope": Slope,
    "Wilderness_Area_4": Wilderness_Area_4,
    "Wilderness_Area_3": Wilderness_Area_3,
    "Soil_Type_12": Soil_Type_12,
    "Soil_Type_23": Soil_Type_23,
    "Soil_Type_10": Soil_Type_10,
    "Soil_Type_29": Soil_Type_29,
    "Soil_Type_39": Soil_Type_39,
    "Soil_Type_30": Soil_Type_30,
    "Soil_Type_38": Soil_Type_38
}])

# -----------------------------
# 4️⃣ Define mapping for numeric predictions
# -----------------------------
cover_type_mapping = {
    1: "Spruce/Fir",
    2: "Lodgepole Pine",
    3: "Ponderosa Pine",
    4: "Cottonwood/Willow",
    5: "Aspen",
    6: "Douglas-fir",
    7: "Krummholz"
}

# -----------------------------
# 5️⃣ Predict button
# -----------------------------
if st.button("🔍 Predict"):
    # Predict using the pipeline
    prediction_numeric = pipeline.predict(input_data)[0]
    prediction_label = cover_type_mapping.get(prediction_numeric, "Unknown")
    st.success(f"Predicted Forest Cover Type: **{prediction_label}**")

st.markdown("---")
st.caption("Created with ❤️ using Streamlit and your trained Random Forest pipeline.")