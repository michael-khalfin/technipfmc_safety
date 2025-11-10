# app.py
import streamlit as st
from handler.handler import generate_chart

st.set_page_config(page_title="Safety Event Dashboard", layout="wide")

st.title("🚧 Safety Event Dashboard")

# --- Sidebar filters ---
st.sidebar.header("Filter Settings")

year_range = st.sidebar.slider("Year Range", 2015, 2025, (2020, 2024))
incident_types = st.sidebar.multiselect("Incident Types", ["Accident", "Near Miss", "Injury"])

filter_requirements = {
    "YEAR_OF_INCIDENT": {"min": year_range[0], "max": year_range[1]},
}
if incident_types:
    filter_requirements["INCIDENT_TYPE"] = incident_types

# --- Big Numbers ---
st.subheader("📊 Statistical Summary")
summary = generate_chart("statistical_summary", filter_requirements)
col1, col2, col3 = st.columns(3)
col1.metric("Total Incidents", summary["total_incidents"])
col2.metric("High Risk (%)", f"{summary['high_risk_percentage']:.1f}%")
col3.metric("Open Actions", summary["open_actions"])

# --- Temporal Distribution ---
st.subheader("📅 Incident Frequency by Year")
temporal_chart = generate_chart("temporal_distribution", filter_requirements)
st.plotly_chart(temporal_chart, use_container_width=True)

# --- Top Locations ---
st.subheader("🏙️ Top Locations")
top_locations = generate_chart("top_locations", filter_requirements)
st.plotly_chart(top_locations, use_container_width=True)

# --- Word Cloud (still Matplotlib) ---
st.subheader("☁️ Word Cloud: Titles")
wordcloud = generate_chart("wordcloud", filter_requirements)
st.pyplot(wordcloud)
