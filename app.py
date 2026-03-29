import streamlit as st
import pandas as pd
import joblib
import numpy as np
from pathlib import Path
import requests
from urllib.parse import quote

############################################################
# Page & Style Config                                      #
############################################################

st.set_page_config(
    page_title="Disease Prediction App",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -- Custom CSS ---------------------------------------------------------
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        margin-bottom: 1rem;
    }
    .disclaimer {
        font-size: 0.8rem;
        color: #757575;
        font-style: italic;
        margin-top: 2rem;
    }
    .provider-name {
        font-weight: bold;
        color: #1E88E5;
    }
    .provider-address {
        font-size: 0.9rem;
        margin-top: 5px;
    }
    .provider-contact {
        font-size: 0.9rem;
        color: #424242;
    }
    .provider-taxonomy {
        font-size: 0.8rem;
        color: #616161;
        font-style: italic;
    }
</style>
""",
    unsafe_allow_html=True,
)

# -- Title & Description ------------------------------------------------
st.markdown(
    "<h1 class='main-header'>Disease Prediction System</h1>", unsafe_allow_html=True
)
st.markdown(
    "<p class='sub-header'>Select your symptoms to get a prediction</p>",
    unsafe_allow_html=True,
)

############################################################
# Data Loaders                                             #
############################################################


@st.cache_resource(show_spinner=False)
def load_model_features_mapping():
    """Return trained model, feature list and disease→specialty mapping"""
    model_path = Path("model/disease_prediction_multinomial_naive_bayes.joblib")
    feature_path = Path("model/model_features.csv")
    mapping_path = Path("model/disease_specialty_mapping.tsv")

    # ---- Model
    model = joblib.load(model_path)

    # ---- Feature List
    if feature_path.exists():
        features = pd.read_csv(feature_path)["feature"].tolist()
    else:
        # Fallback to model.coef_ shape if csv missing (keeps previous behaviour)
        features = [f"symptom_{i}" for i in range(model.coef_.shape[1])]

    # ---- Disease→Specialty Mapping
    if mapping_path.exists():
        mapping_df = pd.read_csv(mapping_path, sep="\t").fillna("")
        mapping = mapping_df.set_index("Disease")[
            ["Specialization", "Classification", "Taxonomy Code"]
        ].to_dict(orient="index")
    else:
        mapping = {}
        st.warning("Specialty mapping file not found – specialties will be omitted.")

    return model, features, mapping


model, all_symptoms, disease_to_specialty = load_model_features_mapping()

############################################################
# Helper Functions                                         #
############################################################


def one_hot(symptoms, all_feats):
    """Return DataFrame with one‑hot encoded symptoms."""
    df = pd.DataFrame(0, index=[0], columns=all_feats)
    df.loc[0, symptoms] = 1
    return df


def fetch_providers_by_specialty(specialty, limit=5, location=None):
    """Fetch healthcare providers from NPPES API based on specialty/taxonomy."""
    base_url = "https://npiregistry.cms.hhs.gov/api/"

    # Encode the specialty for URL
    encoded_specialty = quote(specialty)

    # Build query parameters
    params = {
        "taxonomy_description": encoded_specialty,
        "limit": limit,
        "version": "2.1",
        "pretty": "",
    }

    # Add location parameters if provided
    if location and isinstance(location, dict):
        for key, value in location.items():
            if value:
                params[key] = value

    # Build the query string
    query_parts = []
    for key, value in params.items():
        query_parts.append(f"{key}={value}")

    query_string = "&".join(query_parts)
    full_url = f"{base_url}?{query_string}"

    try:
        response = requests.get(full_url)
        if response.status_code == 200:
            return response.json()
        else:
            return {
                "error": f"API request failed with status code {response.status_code}"
            }
    except Exception as e:
        return {"error": str(e)}


def render_provider_results(results):
    """Render the provider results from NPPES API in a readable format."""
    if not results or "results" not in results:
        st.warning("No provider results found.")
        return

    if "error" in results:
        st.error(f"Error fetching providers: {results['error']}")
        return

    providers = results.get("results", [])
    result_count = results.get("result_count", 0)

    if result_count == 0:
        st.info("No providers found for this specialty in the selected location.")
        return

    st.write(f"Found {result_count} providers. Showing top {len(providers)}:")

    for provider in providers:
        # Get basic information
        if provider.get("enumeration_type") == "NPI-1":
            # Individual provider
            name = f"{provider.get('basic', {}).get('first_name', '')} {provider.get('basic', {}).get('last_name', '')}"
        else:
            # Organization
            name = provider.get("basic", {}).get(
                "organization_name", "Unknown Provider"
            )

        # Get address
        addresses = provider.get("addresses", [])
        location_address = next(
            (a for a in addresses if a.get("address_purpose") == "LOCATION"), None
        )

        # Get taxonomy
        taxonomies = provider.get("taxonomies", [])
        primary_taxonomy = next(
            (t for t in taxonomies if t.get("primary") is True), None
        )

        # Render provider card
        with st.container():
            st.markdown(
                f"<div class='provider-name'>{name}</div>", unsafe_allow_html=True
            )

            if location_address:
                address_line = f"{location_address.get('address_1', '')}"
                if location_address.get("address_2"):
                    address_line += f", {location_address.get('address_2')}"
                city_state = f"{location_address.get('city', '')}, {location_address.get('state', '')} {location_address.get('postal_code', '')}"

                st.markdown(
                    f"<div class='provider-address'>{address_line}<br>{city_state}</div>",
                    unsafe_allow_html=True,
                )

                if location_address.get("telephone_number"):
                    st.markdown(
                        f"<div class='provider-contact'>📞 {location_address.get('telephone_number')}</div>",
                        unsafe_allow_html=True,
                    )

            if primary_taxonomy:
                taxonomy_desc = primary_taxonomy.get("desc", "")
                st.markdown(
                    f"<div class='provider-taxonomy'>{taxonomy_desc}</div>",
                    unsafe_allow_html=True,
                )

            st.markdown("</div>", unsafe_allow_html=True)


############################################################
# Main Layout                                              #
############################################################

# Initialize session state variables if they don't exist
if "last_prediction" not in st.session_state:
    st.session_state["last_prediction"] = None
if "recommended_specialty" not in st.session_state:
    st.session_state["recommended_specialty"] = None
if "taxonomy_code" not in st.session_state:
    st.session_state["taxonomy_code"] = None
if "prediction_probs" not in st.session_state:
    st.session_state["prediction_probs"] = None
if "show_providers" not in st.session_state:
    st.session_state["show_providers"] = False

col_left, col_right = st.columns([1, 1])

# -- Symptoms Input ---------------------------------------
with col_left:
    st.markdown("### Select Your Symptoms")
    selected_symptoms = st.multiselect(
        "Start typing to search…", options=sorted(all_symptoms), default=[]
    )

    st.write(f"Selected: **{len(selected_symptoms)}** symptom(s)")

    predict_btn = st.button("Predict Disease", disabled=not selected_symptoms)

# -- Prediction Output ------------------------------------
with col_right:
    st.markdown("### Prediction Results")

    # Process prediction when button is clicked
    if predict_btn and selected_symptoms:
        try:
            X = one_hot(selected_symptoms, all_symptoms)
            with st.spinner("Analyzing symptoms…"):
                y_pred = model.predict(X)[0]

                # Store in session for downstream use and future displays
                st.session_state["last_prediction"] = y_pred
                st.session_state["show_providers"] = False

                # Get probabilities if available
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(X)[0]
                    top_idx = np.argsort(probs)[-3:][::-1]
                    st.session_state["prediction_probs"] = {
                        "probs": probs,
                        "top_idx": top_idx,
                    }

                # Get specialty info
                info = disease_to_specialty.get(
                    y_pred.lower()
                ) or disease_to_specialty.get(y_pred)
                if info:
                    spec = (
                        info.get("Specialization") or info.get("Classification") or ""
                    )
                    taxonomy = info.get("Taxonomy Code") or ""
                    st.session_state["recommended_specialty"] = spec
                    st.session_state["taxonomy_code"] = taxonomy

        except Exception as e:
            st.error(f"Prediction failed: {e}")

    # Always display the prediction results if they exist in session state
    if st.session_state["last_prediction"]:
        y_pred = st.session_state["last_prediction"]
        st.markdown(f"## Likely Disease: **{y_pred}**")

        # Show probabilities if available
        if "prediction_probs" in st.session_state:
            probs = st.session_state["prediction_probs"]["probs"]
            top_idx = st.session_state["prediction_probs"]["top_idx"]

            st.markdown("### Other possibilities")
            for i in top_idx:
                if probs[i] < 0.05:
                    continue
                st.markdown(f"• {model.classes_[i]} — {probs[i]*100:.1f}%")

        # Show specialty information
        if st.session_state["recommended_specialty"]:
            st.markdown("### Recommended Specialty")
            st.success(f"**{st.session_state['recommended_specialty']}**")

            if (
                "taxonomy_code" in st.session_state
                and st.session_state["taxonomy_code"]
            ):
                st.caption(f"NPI Taxonomy Code: {st.session_state['taxonomy_code']}")

    # Show "Find Providers" button when we have a specialty
    if st.session_state["recommended_specialty"]:
        if st.button("Find Providers for this Specialty"):
            st.session_state["show_providers"] = True

# -- Provider Search Section -----------------------------
if st.session_state["show_providers"] and st.session_state["recommended_specialty"]:
    st.markdown("---")
    st.markdown("## Healthcare Providers")

    # Location filters
    col1, col2, col3 = st.columns(3)
    with col1:
        city = st.text_input("City", "Seattle")
    with col2:
        state = st.selectbox(
            "State",
            options=[
                "",
                "AL",
                "AK",
                "AZ",
                "AR",
                "CA",
                "CO",
                "CT",
                "DE",
                "FL",
                "GA",
                "HI",
                "ID",
                "IL",
                "IN",
                "IA",
                "KS",
                "KY",
                "LA",
                "ME",
                "MD",
                "MA",
                "MI",
                "MN",
                "MS",
                "MO",
                "MT",
                "NE",
                "NV",
                "NH",
                "NJ",
                "NM",
                "NY",
                "NC",
                "ND",
                "OH",
                "OK",
                "OR",
                "PA",
                "RI",
                "SC",
                "SD",
                "TN",
                "TX",
                "UT",
                "VT",
                "VA",
                "WA",
                "WV",
                "WI",
                "WY",
            ],
            index=47,  # Default to Washington
        )
    with col3:
        zip_code = st.text_input("ZIP Code")

    # Set up location parameters
    location_params = {"city": city, "state": state, "postal_code": zip_code}

    # Provider limit slider
    limit = st.slider("Number of providers to show", min_value=1, max_value=20, value=5)

    search_btn = st.button("Search for Healthcare Providers")

    if search_btn:
        specialty = st.session_state["recommended_specialty"]
        with st.spinner(f"Searching for {specialty} providers..."):
            results = fetch_providers_by_specialty(specialty, limit, location_params)
            render_provider_results(results)


############################################################
# Sidebar Information                                      #
############################################################
with st.sidebar:
    st.markdown("<h1 style='text-align: center'>🏥</h1>", unsafe_allow_html=True)
    st.title("About This App")
    st.info(
        "This application uses a trained machine‑learning model to predict "
        "possible diseases from selected symptoms. Results are for educational "
        "purposes only and not a substitute for professional medical advice."
    )

    st.markdown("### How to use:")
    st.markdown(
        """
        1. Search & select your symptoms (multi‑select).  
        2. Click **Predict Disease** to run the model.  
        3. Review the likely disease, alternative possibilities, and suggested specialty.  
        4. Find healthcare providers who specialize in treating your condition.
        5. Consult a qualified healthcare professional for clinical concerns.
        """
    )

    st.markdown("### Model Information:")
    st.markdown(
        f"""- **Algorithm**: Multinomial Naïve Bayes  
- **Number of symptoms**: {len(all_symptoms)}  
- **Specialty mapping**: {'✔️' if disease_to_specialty else '❌'}
- **Provider lookup**: NPPES NPI Registry API"""
    )

############################################################
# Footer                                                   #
############################################################

st.markdown(
    """<div class='disclaimer'>Disclaimer: This tool is for educational purposes only. It is not a substitute for medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider with questions about a medical condition.</div>""",
    unsafe_allow_html=True,
)
