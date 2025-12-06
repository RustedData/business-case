# =============================================================================
# IMPORTS (all at top, grouped: stdlib, third-party, local)
# =============================================================================
import os
import re
import sqlite3
import tempfile
from math import radians, sin, cos, sqrt, atan2

import altair as alt
import folium
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import openai
import pandas as pd
import requests
import streamlit as st
import streamlit_folium as st_folium
from dotenv import load_dotenv
from pandas.api.types import is_numeric_dtype, is_datetime64_any_dtype

# =============================================================================
# CONFIGURATION
# =============================================================================
load_dotenv()

try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
except Exception:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    st.error("OpenAI API key not found. Set `OPENAI_API_KEY` in Streamlit Secrets or as an environment variable.")
    st.stop()

openai.api_key = OPENAI_API_KEY

CSV_URL = "https://www.dropbox.com/scl/fi/9k53mrii5tf35r4443cmv/RideAustin_Weather.csv?rlkey=bg0pl1cmn542ypxzt92y16wxt&st=koq4dks4&dl=1"
MAX_ROWS = int(os.getenv("SAMPLE_FULL_MAX_ROWS", "200000"))

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

# Brand canonicalization mapping
MAKE_MAPPING = {
    "toyota": "Toyota", "toytoa": "Toyota",
    "honda": "Honda",
    "nissan": "Nissan",
    "chevrolet": "Chevrolet", "chevy": "Chevrolet",
    "chrysler": "Chrysler",
    "ford": "Ford",
    "fiat": "Fiat",
    "volkwagen": "Volkswagen", "volkswagen": "Volkswagen",
    "infinity": "Infiniti", "infiniti": "Infiniti",
    "merceded benz": "Mercedes-Benz", "mercedes benz": "Mercedes-Benz",
    "mercedes-benz": "Mercedes-Benz", "mercedes": "Mercedes-Benz",
    "licoln": "Lincoln", "lincoln": "Lincoln",
    "tessla": "Tesla", "tesla": "Tesla",
    "mini cooper": "MINI", "mini": "MINI",
    "bmw": "BMW", "kia": "Kia", "gmc": "GMC", "ram": "Ram",
    "acura": "Acura", "lexus": "Lexus",
}


def canonicalize_make(s: str) -> str:
    """Normalize car make strings to canonical form."""
    if not s:
        return ""
    norm = re.sub(r'[^a-z0-9\s\-]', '', str(s).strip().lower())
    norm = re.sub(r'\s+', ' ', norm).strip()
    if norm in MAKE_MAPPING:
        return MAKE_MAPPING[norm]
    return norm.title() if norm else ""


def haversine_distance(lat1, lon1, lat2, lon2) -> float:
    """Calculate distance in km between two lat/lon points."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


def get_column_types(df: pd.DataFrame) -> dict:
    """Return dict mapping column name to type string."""
    return {
        col: ("numeric" if is_numeric_dtype(df[col]) else "date" if is_datetime64_any_dtype(df[col]) else "text")
        for col in df.columns
    }


def bin_to_time(b: int) -> str:
    """Convert half-hour bin (0-47) to HH:MM string."""
    return f"{b // 2:02d}:{30 if b % 2 else 0:02d}"


# =============================================================================
# DATA LOADING
# =============================================================================

@st.cache_data(show_spinner=True)
def load_data_sample(nrows: int = 1000) -> pd.DataFrame:
    df = pd.read_csv(CSV_URL, encoding="utf-8", delimiter=",", on_bad_lines="skip", low_memory=False, nrows=nrows)
    if "make" in df.columns:
        df["make"] = df["make"].fillna("").astype(str).apply(canonicalize_make)
    return df


@st.cache_data(show_spinner=True)
def load_data_full() -> pd.DataFrame:
    df = pd.read_csv(CSV_URL, encoding="utf-8", delimiter=",", on_bad_lines="skip", low_memory=False)
    if "make" in df.columns:
        df["make"] = df["make"].fillna("").astype(str).apply(canonicalize_make)
    if len(df) > MAX_ROWS:
        df = df.sample(n=MAX_ROWS, random_state=42).reset_index(drop=True)
    return df


# =============================================================================
# CACHED COMPUTATIONS
# =============================================================================

@st.cache_data(show_spinner=True)
def find_longest_trip(df: pd.DataFrame):
    """Find the longest trip by haversine distance. Returns (trip_row, distance_km)."""
    latlon_cols = ["start_location_lat", "start_location_long", "end_location_lat", "end_location_long"]
    clean_df = df.dropna(subset=latlon_cols).copy()
    clean_df = clean_df[(clean_df[latlon_cols] != 0.0).all(axis=1)]
    clean_df["distance_km"] = clean_df.apply(
        lambda r: haversine_distance(r["start_location_lat"], r["start_location_long"],
                                     r["end_location_lat"], r["end_location_long"]), axis=1
    )
    idx = clean_df["distance_km"].idxmax()
    return clean_df.loc[idx], clean_df.loc[idx, "distance_km"]


@st.cache_data(show_spinner=True)
def create_surge_by_hour_chart(df: pd.DataFrame):
    """Create circular chart showing surge % and rides % above/below average by half-hour."""
    work_df = df.copy()
    work_df["started_on"] = pd.to_datetime(work_df["started_on"], errors="coerce")
    work_df = work_df.dropna(subset=["started_on"])
    work_df["half_hour_bin"] = ((work_df["started_on"].dt.hour * 60 + work_df["started_on"].dt.minute) // 30).astype(int)

    surge_by_half_hour = work_df.groupby("half_hour_bin")["surge_factor"].mean().reindex(range(48), fill_value=np.nan)
    rides_by_half_hour = work_df.groupby("half_hour_bin").size().reindex(range(48), fill_value=0)

    avg_surge = float(np.nanmean(surge_by_half_hour)) if surge_by_half_hour.notna().any() else 0.0
    avg_rides = float(rides_by_half_hour.mean())

    surge_pct = ((surge_by_half_hour - avg_surge) / avg_surge * 100).fillna(0) if avg_surge else surge_by_half_hour.fillna(0) * 0
    rides_pct = ((rides_by_half_hour - avg_rides) / avg_rides * 100) if avg_rides else rides_by_half_hour * 0

    # Polar plot - dynamic range based on actual data
    max_abs = max(np.abs(surge_pct.values).max(), np.abs(rides_pct.values).max(), 10)
    target_range = np.ceil(max_abs / 10) * 10  # Round up to nearest 10
    surge_clamped = surge_pct.values
    rides_clamped = rides_pct.values
    angles = np.linspace(0, 2 * np.pi, 48, endpoint=False)
    width = 2 * np.pi / 48 * 0.9

    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, projection="polar")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    ax.bar(angles, surge_clamped, width=width, bottom=target_range, color="#d7191c", edgecolor="black", linewidth=0.3, alpha=0.9)
    ax.bar(angles, rides_clamped , width=width * 0.6, bottom=target_range, color="#2c7bb6", edgecolor="black", linewidth=0.3, alpha=0.8)

    ax.set_xticks(np.linspace(0, 2 * np.pi, 24, endpoint=False))
    ax.set_xticklabels([f"{i:02d}:00" for i in range(24)], size=10)
    ax.set_ylim(0, 2 * target_range)
    ax.set_yticks(np.linspace(0, 2 * target_range, 5))
    ax.set_yticklabels([f"{int(x)}%" for x in np.linspace(-target_range, target_range, 5)])

    ax.legend(handles=[
        mpatches.Patch(color="#d7191c", label="Surge (red)"),
        mpatches.Patch(color="#2c7bb6", label="Rides (blue)")
    ], bbox_to_anchor=(1.05, 1.0), loc="upper left")
    ax.set_title("Surge % and Rides % above/below average by half-hour bin", size=13, pad=20)
    plt.tight_layout()

    return fig, surge_by_half_hour, rides_by_half_hour, surge_pct, rides_pct


# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================

sample_df = load_data_sample()
st.session_state.setdefault("df", sample_df)
st.session_state.setdefault("db_path", None)
st.session_state.setdefault("columns", list(sample_df.columns))
st.session_state.setdefault("column_types", get_column_types(sample_df))
st.session_state.setdefault("loaded_full", False)
st.session_state.setdefault("messages", [])


def ensure_full_loaded():
    """Lazy-load full dataset and create SQLite DB for chatbot."""
    if st.session_state.get("loaded_full"):
        return
    full_df = load_data_full()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
    db_path = tmp.name
    tmp.close()
    conn = sqlite3.connect(db_path)
    full_df.to_sql("rides", conn, index=False, if_exists="replace")
    conn.close()

    st.session_state["df"] = full_df
    st.session_state["db_path"] = db_path
    st.session_state["columns"] = list(full_df.columns)
    st.session_state["column_types"] = get_column_types(full_df)
    st.session_state["loaded_full"] = True


# =============================================================================
# CHATBOT FUNCTIONS
# =============================================================================

def extract_sql(text: str) -> str:
    """Extract SQL from GPT response, handle median syntax."""
    lines = [l for l in text.strip().splitlines() if not l.strip().startswith("```")]
    sql = "\n".join(lines).strip()
    def median_sql(col):
        return (f"(SELECT AVG(val) FROM (SELECT {col} AS val FROM rides WHERE {col} IS NOT NULL "
                f"ORDER BY {col} LIMIT 2 - (SELECT COUNT(*) FROM rides WHERE {col} IS NOT NULL) % 2 "
                f"OFFSET (SELECT (COUNT(*) - 1) / 2 FROM rides WHERE {col} IS NOT NULL)))")
    sql = re.sub(r"PERCENTILE_CONT\(0\.5\) WITHIN GROUP \(ORDER BY ([a-zA-Z0-9_]+)\)",
                 lambda m: median_sql(m.group(1)), sql, flags=re.IGNORECASE)
    return sql


def nl_to_sql(question: str, columns: list, column_types: dict) -> str:
    col_desc = ", ".join([f"{col} ({column_types.get(col, 'text')})" for col in columns])
    prompt = f"""You are a helpful assistant that translates natural language questions into SQL queries for a table called 'rides'.
The table has the following columns and types: {col_desc}.
If the user asks for min/max/average/sum, only use numeric or date columns.
Translate the following question into a valid SQLite SQL query. Only return the SQL query, nothing else.
Question: {question}"""
    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo", messages=[{"role": "user", "content": prompt}], max_tokens=1024)
        return response.choices[0].message.content.strip()
    except Exception as e:
        raise RuntimeError(f"OpenAI error during SQL generation: {e}")


def answer_from_sql(question: str) -> str:
    """Generate answer from user question using SQL + GPT."""
    ensure_full_loaded()
    df = st.session_state["df"]
    db_path = st.session_state["db_path"]
    columns = st.session_state["columns"]
    column_types = st.session_state["column_types"]

    try:
        raw_sql = nl_to_sql(question, columns, column_types)
    except RuntimeError as e:
        st.error(str(e))
        return "Error: failed to generate SQL from your question."

    sql = extract_sql(raw_sql)
    conn = None
    try:
        conn = sqlite3.connect(db_path) if db_path else sqlite3.connect(":memory:")
        if not db_path:
            df.to_sql("rides", conn, index=False, if_exists="replace")

        result_df = None
        for attempt in range(2):
            try:
                result_df = pd.read_sql_query(sql, conn)
                break
            except Exception as e:
                if attempt == 0:
                    try:
                        fix_prompt = f"Fix this SQL query for SQLite:\nOriginal: {sql}\nError: {e}"
                        resp = openai.chat.completions.create(
                            model="gpt-3.5-turbo", messages=[{"role": "user", "content": fix_prompt}], max_tokens=1024)
                        sql = extract_sql(resp.choices[0].message.content)
                    except Exception:
                        return "Error: could not repair SQL."
                else:
                    return f"SQL Error: {e}"

        if result_df is None or result_df.empty:
            return "No results returned."

        answer_prompt = f"""Use the following data to answer the user's question concisely:
Data:
{result_df.head(50).to_string(index=False)}

Question: {question}"""
        try:
            resp = openai.chat.completions.create(
                model="gpt-3.5-turbo", messages=[{"role": "user", "content": answer_prompt}], max_tokens=512)
            return resp.choices[0].message.content.strip()
        except Exception as e:
            return f"Error generating answer: {e}"
    finally:
        if conn:
            conn.close()


# =============================================================================
# UI: INSIGHTS TAB
# =============================================================================

st.title("RideAustin Data Explorer")
tabs = st.tabs(["Interesting insights", "Chatbot"])

with tabs[0]:
    ensure_full_loaded()
    full_df = st.session_state["df"]

    # --- Distance Distribution ---
    st.subheader("Distance Distribution (Boxplot)")
    distances = full_df["distance_travelled"].dropna().astype(float) / 1000.0  # meters to km

    trim_pct = 0.01
    lower, upper = np.quantile(distances, trim_pct), np.quantile(distances, 1 - trim_pct)
    trimmed = distances[(distances >= lower) & (distances <= upper)]

    fig_dist, ax_dist = plt.subplots(figsize=(8, 3))
    ax_dist.boxplot(trimmed, vert=False, patch_artist=True, boxprops=dict(facecolor="steelblue", color="black"))
    ax_dist.set_xlabel("Distance (km)")
    ax_dist.set_yticks([])
    ax_dist.set_title("Distance distribution (boxplot) — trimmed to 1st–99th percentiles")
    st.pyplot(fig_dist)
    st.caption(f"Showing distances between {lower:.2f} km and {upper:.2f} km")

    c1, c2, c3 = st.columns(3)
    c1.metric("Median distance", f"{np.nanmedian(trimmed):.2f} km")
    c2.metric("Mean distance", f"{np.nanmean(trimmed):.2f} km")
    c3.metric("Max distance (trimmed)", f"{np.nanmax(trimmed):.2f} km")

    # --- Longest Trip ---
    st.subheader("Longest Trip Visualization")
    long_tabs = st.tabs(["Map", "Long Rides (>100 km)"])

    with long_tabs[0]:
        longest_trip, distance_km = find_longest_trip(full_df)
        st.write(f"**Longest Trip Distance:** {distance_km:.2f} km")

        start_lat, start_lon = longest_trip["start_location_lat"], longest_trip["start_location_long"]
        end_lat, end_lon = longest_trip["end_location_lat"], longest_trip["end_location_long"]

        m = folium.Map(location=[(start_lat + end_lat) / 2, (start_lon + end_lon) / 2], zoom_start=6)
        folium.Marker([start_lat, start_lon], popup="Start", icon=folium.Icon(color="green")).add_to(m)
        folium.Marker([end_lat, end_lon], popup="End", icon=folium.Icon(color="red")).add_to(m)

        # Try OSRM route, fallback to straight line
        try:
            osrm_url = f"http://router.project-osrm.org/route/v1/driving/{start_lon},{start_lat};{end_lon},{end_lat}?overview=full&geometries=geojson"
            resp = requests.get(osrm_url, timeout=10)
            resp.raise_for_status()
            route_coords = resp.json()["routes"][0]["geometry"]["coordinates"]
            route_latlon = [[lat, lon] for lon, lat in route_coords]
            folium.PolyLine(route_latlon, color="blue", weight=4, opacity=0.9).add_to(m)
        except Exception:
            folium.PolyLine([[start_lat, start_lon], [end_lat, end_lon]], color="blue", weight=2).add_to(m)

        st_folium.st_folium(m, width=800, height=500)

    with long_tabs[1]:
        df_with_dist = full_df.copy()
        df_with_dist["_distance_km"] = full_df["distance_travelled"].dropna().astype(float) / 1000.0
        long_rides = df_with_dist[df_with_dist["_distance_km"] > 100.0]
        st.write(f"Found {len(long_rides)} rides with distance > 100 km.")

        if not long_rides.empty:
            m2 = folium.Map(location=[long_rides.iloc[0]["start_location_lat"], long_rides.iloc[0]["start_location_long"]], zoom_start=6)
            for _, row in long_rides.head(500).iterrows():
                folium.CircleMarker([row["start_location_lat"], row["start_location_long"]], radius=3, color="green", fill=True).add_to(m2)
                folium.CircleMarker([row["end_location_lat"], row["end_location_long"]], radius=3, color="red", fill=True).add_to(m2)
            st_folium.st_folium(m2, width=800, height=500)

    # --- Correlation Matrix ---
    st.subheader("Correlation Matrix")
    rename_map = {"PRCP": "precipitation", "Year": "Car Year", "AWND": "average wind speed"}
    exclude_cols = {"end_location_lat", "end_location_long", "start_location_lat", "start_location_long", "charity_id", "free_credit_used"}
    numeric_cols = [c for c in full_df.select_dtypes(include=[np.number]).columns if c not in exclude_cols]

    display_to_col = {rename_map.get(c, c): c for c in numeric_cols}
    chosen = st.multiselect("Choose 2+ numeric columns:", list(display_to_col.keys()), key="corr_cols")

    if len(chosen) >= 2:
        selected_cols = [display_to_col[c] for c in chosen]
        corr = full_df[selected_cols].apply(pd.to_numeric, errors="coerce").corr()

        fig, ax = plt.subplots(figsize=(max(4, len(selected_cols)), max(4, len(selected_cols))))
        im = ax.imshow(corr.values, cmap="RdBu", vmin=-1, vmax=1)
        ax.set_xticks(range(len(selected_cols)))
        ax.set_yticks(range(len(selected_cols)))
        ax.set_xticklabels([rename_map.get(c, c) for c in selected_cols], rotation=45, ha="right")
        ax.set_yticklabels([rename_map.get(c, c) for c in selected_cols])
        for (i, j), val in np.ndenumerate(corr.values):
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        st.pyplot(fig)
    elif chosen:
        st.info("Please select at least two numeric columns.")

    # --- Surge Factor by Time ---
    st.subheader("Surge Factor and Average Rides by Time of Day")
    fig, surge_by_half_hour, rides_by_half_hour, surge_pct, rides_pct = create_surge_by_hour_chart(full_df)
    st.pyplot(fig)

    max_surge_idx, min_surge_idx = int(surge_pct.idxmax()), int(surge_pct.idxmin())
    max_rides_idx, min_rides_idx = int(rides_pct.idxmax()), int(rides_pct.idxmin())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Peak Surge Time", bin_to_time(max_surge_idx), f"{surge_pct.max():.1f}%")
    c2.metric("Peak Rides Time", bin_to_time(max_rides_idx), f"{rides_pct.max():.1f}%")
    c3.metric("Lowest Surge Time", bin_to_time(min_surge_idx), f"{surge_pct.min():.1f}%")
    c4.metric("Lowest Rides Time", bin_to_time(min_rides_idx), f"{rides_pct.min():.1f}%")

    st.write(f"Average surge: {np.nanmean(surge_by_half_hour):.3f}. Average rides per half-hour: {int(rides_by_half_hour.mean())}.")

    # --- Car Brands ---
    st.subheader("Car Brands — Average Driver Rating")
    rating_col = "driver_rating" if "driver_rating" in full_df.columns else "rating"
    tmp = full_df[["make", rating_col]].dropna()
    tmp[rating_col] = pd.to_numeric(tmp[rating_col], errors="coerce")
    tmp = tmp.dropna()

    overall = tmp[rating_col].mean()
    grp = tmp.groupby("make").agg(rides=(rating_col, "count"), avg_rating=(rating_col, "mean")).reset_index()
    grp = grp[grp["rides"] >= 5].sort_values("avg_rating", ascending=False)
    grp["pct_diff"] = (grp["avg_rating"] - overall) / overall * 100

    top_n = st.number_input("Show top N brands", 1, min(len(grp), 200), 20, key="brands_top_n")
    asc = st.checkbox("Reverse order (ascending)", key="brands_asc")

    display = grp.sort_values("pct_diff", ascending=asc).head(int(top_n))

    chart = alt.Chart(display).mark_bar().encode(
        x=alt.X("pct_diff:Q", title="Pct diff from average (%)"),
        y=alt.Y("make:N", sort=alt.EncodingSortField(field="pct_diff", order="ascending" if asc else "descending"), title="Make"),
        color=alt.condition(alt.datum.pct_diff > 0, alt.value("#d7191c"), alt.value("#2c7bb6"))
    ).properties(height=min(600, 25 * len(display)), width=700)
    st.altair_chart(chart, width="stretch")

    best_row, worst_row = grp.loc[grp["avg_rating"].idxmax()], grp.loc[grp["avg_rating"].idxmin()]
    c1, c2, c3 = st.columns(3)
    c1.metric("Average driver rating", f"{overall:.2f}")
    c2.metric("Best brand", best_row["make"], f"{best_row['avg_rating']:.2f}")
    c3.metric("Worst brand", worst_row["make"], f"{worst_row['avg_rating']:.2f}")

# =============================================================================
# UI: CHATBOT TAB
# =============================================================================

with tabs[1]:
    st.header("Data Chatbot (Data-Only Answers)")
    st.subheader("Ask anything about RideAustin trips, drivers, and fares")

    chat_container = st.container()
    with chat_container:
        for msg in st.session_state["messages"]:
            st.chat_message(msg["role"]).write(msg["content"])

    spinner_container = st.container()
    user_input = st.chat_input("Ask a question about the data...")

    if user_input:
        st.session_state["messages"].append({"role": "user", "content": user_input})
        with chat_container:
            st.chat_message("user").write(user_input)

        with spinner_container:
            with st.spinner("Thinking..."):
                answer = answer_from_sql(user_input)

        spinner_container.empty()
        st.session_state["messages"].append({"role": "assistant", "content": answer})
        with chat_container:
            st.chat_message("assistant").write(answer)
