import streamlit as st
import pandas as pd
import requests
import json
import plotly.graph_objects as go
from typing import Dict, Tuple

# District name to district code mapping
district_mapping = {
    "Bishan, Ang Mo Kio": 20,
    "Clementi, Queenstown, Southern Islands": 5,
    "Geylang, Eunos": 14,
    "Serangoon, Hougang, Punggol": 19,
    "Changi, Pasir Ris, Paya Lebar, Simei, Tampines": 17,
    "Balestier, Toa Payoh, Serangoon": 12,
    "Bedok, Upper East Coast, Eastwood, Kew Drive": 16,
    "Orchard, Cairnhill, River Valley": 9,
    "Ardmore, Bukit Timah, Holland Road, Tanglin": 10,
    "Boon Lay, Jurong, Tuas": 22,
    "City, Cecil, Marina, People's Park": 1,
    "Keppel, Mount Faber, Sentosa, Telok Blangah": 4,
    "Macpherson, Braddell": 13,
    "Bukit Batok, Choa Chu Kang, Hillview, Dairy Farm, Bukit Panjang, Lim Chu Kang": 23,
    "Katong, Marine Parade, Siglap, Tanjong Rhu": 15,
    "Upper Bukit Timah, Ulu Pandan, Clementi Park": 21,
    "Seletar, Yio Chu Kang": 28,
    "Kranji, Woodgrove, Woodlands": 25,
    "Pioneer, Tuas": 24,
    "Jurong East": 7,
    "Eunos, Geylang, Kembangan": 14,
    "East Coast, Meyer, Mountbatten": 15,
    "Leonie Hill, Orchard, Oxley": 9,
    "Springleaf, Upper Thomson, Yishun, Sembawang": 26,
    "Adam Park, Watten Estate, Novena, Thomson": 11,
}

# Trained district names
trained_district_names = [
    "Ang Mo Kio",
    "Bedok",
    # Add all district names that are in the training data
]

floor_ranges = ["01-05", "06-10", "11-15", "16-20", "21-25", "26-30", "31-35", "36-40", "41-45", "46-50", "51-55"]

api_url = "http://localhost:8000/predict/"


def get_coordinates(postal_code: str) -> Tuple[float, float]:
    url = f"https://developers.onemap.sg/commonapi/search?searchVal={postal_code}&returnGeom=Y&getAddrDetails=Y&pageNum=1"
    response = requests.get(url)

    if response.status_code != 200:
        raise ValueError(f"Error {response.status_code}: {response.text}")

    data = response.json()

    if "found" not in data:
        raise ValueError("The 'found' key is missing in the API response")

    if data["found"] > 0:
        x = float(data["results"][0]["X"])
        y = float(data["results"][0]["Y"])
        return x, y

    raise ValueError(f"No coordinates found for postal code {postal_code}")


st.title("EC Price Prediction and Explainability")

# Input fields
postal_code = st.text_input("Postal Code", "639798")
floor_range = st.selectbox("Floor Range", floor_ranges)
area = st.number_input("Area (sqm)", value=122.0)
type_of_sale = st.selectbox("Type of Sale", ["Resale", "New Sale"])
district_name = st.selectbox("District Name", sorted(district_mapping.keys()))
remaining_lease = st.number_input("Remaining Lease (years)", value=72.31)
price_index = st.number_input("Price Index", value=168.1)

if district_name not in trained_district_names:
    st.warning("The selected district is not in the training data, and the prediction might not be accurate.")

coordinates = get_coordinates(postal_code)

if st.button("Predict"):
    input_data = {
        "x": coordinates[0],
        "y": coordinates[1],
        "area": area,
        "floor_range": floor_range,
        "type_of_sale": type_of_sale,
        "district": district_mapping[district_name],
        "district_name": district_name,
        "remaining_lease": remaining_lease,
        "price_index": price_index
    }

    response = requests.post(api_url, json=input_data)
    if response.status_code == 200:
        prediction_data = response.json()
        price_psm = prediction_data["prediction"]
        price = price_psm * area

        st.success(f"Predicted EC price per square meter (psm): {price_psm:.2f}")
        st.success(f"Predicted EC sales price: {price:.2f}")

        st.subheader("SHAP Values")
        shap_df = pd.DataFrame(prediction_data["shap_values"]["prediction"], index=[0])
        shap_df = shap_df.T
        shap_df.columns = ["SHAP Value"]
        shap_df = shap_df.sort_values("SHAP Value", ascending=False)

        # Create a waterfall chart
        fig = go.Figure(go.Waterfall(
            x=shap_df.index,
            y=shap_df["SHAP Value"],
            textposition="outside",
            connector={"line": {"color": "rgb(63, 63, 63)"}},
        ))

        fig.update_layout(title="SHAP Values Waterfall Chart", showlegend=False)
        st.plotly_chart(fig)
    else:
        st.error(f"Error {response.status_code}: {response.text}")
