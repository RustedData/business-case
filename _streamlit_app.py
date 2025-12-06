import streamlit as st
import pandas as pd
import openai
import os
import sqlite3
import tempfile
import matplotlib.pyplot as plt
import numpy as np
import folium
import streamlit_folium as st_folium
import altair as alt
import requests
import re
from math import radians, sin, cos, sqrt, atan2
from pandas.api.types import is_numeric_dtype, is_datetime64_any_dtype

# Load API keys from .env for local development
from dotenv import load_dotenv
load_dotenv()

# Load OpenAI key from Streamlit secrets or environment
try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
except Exception:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    st.error("OpenAI API key not found. Set `OPENAI_API_KEY` in Streamlit Secrets or as an environment variable.")
    st.stop()

openai.api_key = OPENAI_API_KEY


# New: canonicalize car make strings (fix duplicates, trailing spaces, misspellings)
def _normalize_make_str(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip()
    # keep empty as empty
    if s == "":
        return ""
    # lower + remove punctuation except hyphen, collapse spaces
    norm = s.lower()
    norm = re.sub(r'[^a-z0-9\s\-]', '', norm)
    norm = re.sub(r'\s+', ' ', norm).strip()
    return norm

def canonicalize_make(s: str) -> str:
    norm = _normalize_make_str(s)
    # known mapping from normalized -> canonical label
    mapping = {
        # common duplicates / spacing / case
        "toyota": "Toyota",
        "toyota ": "Toyota",
        "toytoa": "Toyota",
        "toyota  ": "Toyota",
        "toyota  ": "Toyota",
        "toyota": "Toyota",
        "toyota": "Toyota",
        "toyota": "Toyota",

        "honda": "Honda",
        "honda ": "Honda",

        "nissan": "Nissan",
        "nissan ": "Nissan",

        "chevrolet": "Chevrolet",
        "chevrolet ": "Chevrolet",
        "chevy": "Chevrolet",

        "chrysler": "Chrysler",
        "chrysler ": "Chrysler",

        "ford": "Ford",
        "ford ": "Ford",

        "fiat": "Fiat",
        "fiat ": "Fiat",
        "fiat": "Fiat",
        "fiat": "Fiat",
        "fiat": "Fiat",

        # misspellings / near-duplicates
        "volkwagen": "Volkswagen",
        "volkswagen": "Volkswagen",

        "infinity": "Infiniti",
        "infiniti": "Infiniti",

        "merceded benz": "Mercedes-Benz",
        "mercedes benz": "Mercedes-Benz",
        "mercedes-benz": "Mercedes-Benz",
        "mercedes": "Mercedes-Benz",

        "licoln": "Lincoln",
        "lincoln": "Lincoln",

        "tessla": "Tesla",
        "tesla": "Tesla",

        "mini cooper": "MINI",
        "mini": "MINI",
        "mini ": "MINI",

        # a few others / common forms
        "bmw": "BMW",
        "kia": "Kia",
        "gmc": "GMC",
        "ram": "Ram",
        "acura": "Acura",
        "lexus": "Lexus",
    }
    if norm in mapping:
        return mapping[norm]
    # fallback: title-case but keep known acronyms uppercase
    acronyms = {"bmw", "gmc", "ram", "mini", "tesla", "kia"}
    if norm in acronyms:
        return norm.upper()
    # default fallback: title case with single-word hyphen preserved
    return " ".join([p.title() for p in norm.split(" ")]) if norm else ""


@st.cache_data(show_spinner=True)
def load_data_sample(nrows: int = 1000):
    CSV_URL = "https://www.dropbox.com/scl/fi/9k53mrii5tf35r4443cmv/RideAustin_Weather.csv?rlkey=bg0pl1cmn542ypxzt92y16wxt&st=koq4dks4&dl=1"
    df = pd.read_csv(
        CSV_URL,
        encoding="utf-8",
        delimiter=",",
        on_bad_lines="skip",
        low_memory=False,
        nrows=nrows,
    )
    # canonicalize 'make' column if present
    if "make" in df.columns:
        df["make"] = df["make"].fillna("").astype(str).apply(canonicalize_make)
    return df


@st.cache_data(show_spinner=True)
def load_data_full():
    CSV_URL = "https://www.dropbox.com/scl/fi/9k53mrii5tf35r4443cmv/RideAustin_Weather.csv?rlkey=bg0pl1cmn542ypxzt92y16wxt&st=koq4dks4&dl=1"
    # Read full CSV then take a random sample to limit memory usage on Streamlit Cloud
    df = pd.read_csv(
        CSV_URL,
        encoding="utf-8",
        delimiter=",",
        on_bad_lines="skip",
        low_memory=False,
    )
    if "make" in df.columns:
        df["make"] = df["make"].fillna("").astype(str).apply(canonicalize_make)
    try:
        max_rows = int(os.getenv("SAMPLE_FULL_MAX_ROWS", "200000"))
    except Exception:
        max_rows = 200000
    if len(df) > max_rows:
        # reproducible sample
        df = df.sample(n=max_rows, random_state=42).reset_index(drop=True)
    return df


def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


@st.cache_data(show_spinner=True)
def find_longest_trip(df: pd.DataFrame):
    """
    Find the longest trip in the dataframe.
    Filters out invalid lat/long values (0.0, None, NaN, NA, etc).
    Returns the trip entry and its distance.
    """
    # Create a clean copy without modifying the original
    clean_df = df[
        (df['start_location_lat'].notna()) &
        (df['start_location_long'].notna()) &
        (df['end_location_lat'].notna()) &
        (df['end_location_long'].notna()) &
        (df['start_location_lat'] != 0.0) &
        (df['start_location_long'] != 0.0) &
        (df['end_location_lat'] != 0.0) &
        (df['end_location_long'] != 0.0)
    ].copy()
    
    # Calculate distance for each ride
    clean_df['distance_km'] = clean_df.apply(
        lambda row: haversine_distance(
            row['start_location_lat'], 
            row['start_location_long'],
            row['end_location_lat'], 
            row['end_location_long']
        ),
        axis=1
    )
    
    # Find the entry with the most distance
    max_distance_idx = clean_df['distance_km'].idxmax()
    max_distance_entry = clean_df.loc[max_distance_idx]
    
    return max_distance_entry, max_distance_entry['distance_km']


@st.cache_data(show_spinner=True)
def create_surge_by_hour_chart(df: pd.DataFrame):
    """
    Create a circular histogram showing:
     - surge_percent = (surge - avg_surge) / avg_surge * 100  (signed)
     - rides_percent = (count - avg_count) / avg_count * 100  (signed)
    Returns:
      fig (matplotlib.figure.Figure),
      surge_by_half_hour (Series raw average surge per half-hour),
      rides_by_half_hour (Series count per half-hour),
      surge_pct (Series percent above/below avg surge),
      rides_pct (Series percent above/below avg rides)
    """
    work_df = df.copy()
    work_df['started_on'] = pd.to_datetime(work_df['started_on'], errors='coerce')
    work_df = work_df.dropna(subset=['started_on']).copy()

    # half-hour bins 0..47
    work_df['hour_minute'] = work_df['started_on'].dt.hour * 60 + work_df['started_on'].dt.minute
    work_df['half_hour_bin'] = (work_df['hour_minute'] // 30).astype(int)

    # surge average per bin (may have NaNs if no data in bin)
    surge_by_half_hour = work_df.groupby('half_hour_bin')['surge_factor'].mean().reindex(range(48), fill_value=np.nan)
    # ride counts per bin
    rides_by_half_hour = work_df.groupby('half_hour_bin').size().reindex(range(48), fill_value=0)

    # averages
    avg_surge = float(np.nanmean(surge_by_half_hour)) if surge_by_half_hour.notna().any() else 0.0
    avg_rides = float(rides_by_half_hour.mean())  # number windows = 48

    # percentages signed
    # protect division by zero
    if avg_surge == 0:
        surge_pct = (surge_by_half_hour.fillna(0) * 0.0).astype(float)
    else:
        surge_pct = ((surge_by_half_hour - avg_surge) / avg_surge * 100).astype(float)

    if avg_rides == 0:
        rides_pct = (rides_by_half_hour * 0.0).astype(float)
    else:
        rides_pct = ((rides_by_half_hour - avg_rides) / avg_rides * 100).astype(float)

    # Create a polar (circular) plot with two concentric bar rings:
    # - outer ring: surge % above/below avg (absolute radial length, color encodes sign)
    # - inner ring: rides % above/below avg (absolute radial length, color encodes sign)
    # The radial scale is shared and set to the maximum absolute percent across both series so both rings use the same visual range.
    surge_vals = surge_pct.fillna(0).values
    rides_vals = rides_pct.values

    # We want a radial range from -60..60. We'll map that to polar radii by placing the zero at `target_range`.
    # This lets positive values extend outward from center and negative values extend inward relative to that center.
    target_range = 60.0

    # Clamp or scale the plotted values to the target_range so everything fits in -target_range..target_range
    surge_clamped = np.clip(surge_vals, -target_range, target_range)
    rides_clamped = np.clip(rides_vals, -target_range, target_range)

    angles = np.linspace(0, 2 * np.pi, 48, endpoint=False)
    width = 2 * np.pi / 48 * 0.9

    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, projection='polar')
    # Make the polar plot behave like a clock: 12:00 at top and clockwise direction
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)

    # Colors: make all surge bars red, all rides bars blue per user request
    colors_surge = ['#d7191c'] * len(surge_clamped)
    colors_rides = ['#2c7bb6'] * len(rides_clamped)

    # Map signed values to polar radii by offsetting bottom to target_range (so 0 maps to target_range)
    bottom = target_range

    bars_surge = ax.bar(angles, surge_clamped, width=width, bottom=bottom, color=colors_surge, edgecolor='black', linewidth=0.3, alpha=0.9)
    # Draw rides as a slightly shorter ring so it visually nests inside the surge ring
    bars_rides = ax.bar(angles, rides_clamped * 0.7, width=width * 0.6, bottom=bottom, color=colors_rides, edgecolor='black', linewidth=0.3, alpha=0.8)

    # Hour labels at the perimeter (every 2 bins = hourly)
    hour_angles = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    hour_labels = [f"{i:02d}:00" for i in range(24)]
    ax.set_xticks(hour_angles)
    ax.set_xticklabels(hour_labels, size=10)

    # Set radial limits from 0..2*target_range and map ticks to -target..target
    ax.set_ylim(0, 2 * target_range)
    ticks = np.linspace(0, 2 * target_range, 5)
    tick_labels = [f"{int(x)}%" for x in np.linspace(-target_range, target_range, 5)]
    ax.set_yticks(ticks)
    ax.set_yticklabels(tick_labels)

    # Legend: only two patches (one for surge, one for rides) as requested
    import matplotlib.patches as mpatches
    patch_surge = mpatches.Patch(color='#d7191c', label='Surge (red)')
    patch_rides = mpatches.Patch(color='#2c7bb6', label='Rides (blue)')
    ax.legend(handles=[patch_surge, patch_rides], bbox_to_anchor=(1.05, 1.0), loc='upper left')

    ax.set_title('Surge % and Rides % above/below average by half-hour bin', size=13, pad=20)
    plt.tight_layout()

    return fig, surge_by_half_hour, rides_by_half_hour, surge_pct, rides_pct


# Load a small sample immediately so the app starts quickly for health checks
sample_df = load_data_sample()

# Session state for data + DB path (full DB stored in a temp file when loaded)
st.session_state.setdefault("df", sample_df)
st.session_state.setdefault("db_path", None)
st.session_state.setdefault("columns", list(sample_df.columns))
st.session_state.setdefault("column_types", {})
st.session_state.setdefault("loaded_full", False)


def get_column_types(df: pd.DataFrame) -> dict:
    return {col: ("numeric" if is_numeric_dtype(df[col]) else "date" if is_datetime64_any_dtype(df[col]) else "text") for col in df.columns}


# Ensure session column types are initialized from the sample
if not st.session_state.get("column_types"):
    st.session_state["column_types"] = get_column_types(st.session_state["df"])


def ensure_full_loaded():
    """Load the full CSV and write it to a temporary SQLite file (once).
    We store the temp DB path in `st.session_state['db_path']` and the full df in `st.session_state['df']`.
    Writing to a file lets each request open its own sqlite3 connection to avoid cross-thread errors.
    """
    if st.session_state.get("loaded_full"):
        return
    try:
        full_df = load_data_full()
    except Exception as e:
        st.error(f"Failed to load full dataset: {e}")
        return

    try:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
        db_path = tmp.name
        tmp.close()
        wconn = sqlite3.connect(db_path)
        full_df.to_sql("rides", wconn, index=False, if_exists="replace")
        wconn.close()
    except Exception as e:
        st.error(f"Failed to write data to temporary DB file: {e}")
        return

    st.session_state["df"] = full_df
    st.session_state["db_path"] = db_path
    st.session_state["columns"] = list(full_df.columns)
    st.session_state["column_types"] = get_column_types(full_df)
    st.session_state["loaded_full"] = True
    

def extract_sql(text: str) -> str:
    import re
    lines = text.strip().splitlines()
    sql_lines = []
    for line in lines:
        if line.strip().lower().startswith("corrected sql:"):
            line = line.split(":", 1)[-1].strip()
            if line:
                sql_lines.append(line)
            continue
        if line.strip().startswith("```"):
            continue
        sql_lines.append(line)
    sql = "\n".join(sql_lines).strip()

    def median_sql(col: str) -> str:
        return (
            f"(SELECT AVG(val) FROM (SELECT {col} AS val FROM rides WHERE {col} IS NOT NULL"
            f" ORDER BY {col} LIMIT 2 - (SELECT COUNT(*) FROM rides WHERE {col} IS NOT NULL) % 2"
            f" OFFSET (SELECT (COUNT(*) - 1) / 2 FROM rides WHERE {col} IS NOT NULL)))"
        )

    sql = re.sub(
        r"PERCENTILE_CONT\(0\.5\) WITHIN GROUP \(ORDER BY ([a-zA-Z0-9_]+)\)",
        lambda m: median_sql(m.group(1)),
        sql,
        flags=re.IGNORECASE,
    )
    return sql


def nl_to_sql(question: str, columns: list, column_types: dict) -> str:
    col_desc = ", ".join([f"{col} ({column_types.get(col,'text')})" for col in columns])
    prompt = f"""
You are a helpful assistant that translates natural language questions into SQL queries for a table called 'rides'.
The table has the following columns and types: {col_desc}.
If the user asks for min/max/average/sum, only use numeric or date columns. Do not use text columns for these operations.
If the user asks for the median, use only SQLite-compatible SQL. For the median, use a subquery like:
    (SELECT AVG(val) FROM (
        SELECT column AS val FROM rides WHERE column IS NOT NULL ORDER BY column LIMIT 2 - (SELECT COUNT(*) FROM rides WHERE column IS NOT NULL) % 2 OFFSET (SELECT (COUNT(*) - 1) / 2 FROM rides WHERE column IS NOT NULL)
    ))
Replace 'column' with the actual column name.
If the question cannot be answered with the data, return a SQL query that returns an empty result (e.g., SELECT * FROM rides WHERE 1=0).
Translate the following question into a valid SQLite SQL query. Only return the SQL query, nothing else.
Question: {question}
"""
    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
        )
        sql = response.choices[0].message.content.strip()
        return sql
    except Exception as e:
        raise RuntimeError(f"OpenAI error during SQL generation: {e}")


def answer_from_sql(question: str) -> str:
    # Ensure full data available (lazy load)
    ensure_full_loaded()

    df = st.session_state.get("df")
    db_path = st.session_state.get("db_path")
    columns = st.session_state.get("columns", list(df.columns))
    column_types = st.session_state.get("column_types", get_column_types(df))

    try:
        raw_sql = nl_to_sql(question, columns, column_types)
    except RuntimeError as e:
        st.error(str(e))
        return "Error: failed to generate SQL from your question. Check API key and logs."

    sql = extract_sql(raw_sql)

    # Open a fresh connection for this execution to avoid cross-thread sqlite errors
    exec_conn = None
    opened_here = False
    if db_path:
        try:
            exec_conn = sqlite3.connect(db_path)
            opened_here = True
        except Exception as e:
            return f"Failed to open DB at {db_path}: {e}"
    else:
        # use an in-memory db built from the sample
        try:
            exec_conn = sqlite3.connect(":memory:")
            df.to_sql("rides", exec_conn, index=False, if_exists="replace")
            opened_here = True
        except Exception as e:
            if exec_conn:
                exec_conn.close()
            return f"Failed to prepare in-memory DB for sample execution: {e}"

    # Try executing and attempt a single automatic fix if execution fails
    result_df = None
    for attempt in range(2):
        try:
            result_df = pd.read_sql_query(sql, exec_conn)
            break
        except Exception as e:
            if attempt == 0:
                fix_prompt = f"""
You are a SQL expert. The following SQL query resulted in an error when run on a SQLite database. Please fix the query. Only return the corrected SQL query, nothing else.

Original question: {question}
Original SQL: {sql}
Error: {e}
"""
                try:
                    resp = openai.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[{"role": "user", "content": fix_prompt}],
                        max_tokens=1024,
                    )
                    sql = extract_sql(resp.choices[0].message.content)
                    continue
                except Exception as oe:
                    if opened_here and exec_conn:
                        exec_conn.close()
                    st.error(f"OpenAI error while attempting to fix SQL: {oe}")
                    return "Error: could not repair SQL. See logs."
            else:
                if opened_here and exec_conn:
                    exec_conn.close()
                return f"SQL Error: {e}\nSQL: {sql}"

    if result_df is None:
        if opened_here and exec_conn:
            exec_conn.close()
        return "No results returned."

    result_sample = result_df.head(50)

    # Generate a final conversational answer strictly based on the returned data
    answer_prompt = f"""
You are a helpful assistant. Use the following data (from a SQL query) to answer the user's question in a clear, conversational way. If the answer is not in the data, say 'I'm only able to answer questions about the RideAustin data.'

Data:
{result_sample.to_string(index=False)}

Question: {question}

Please provide a concise, conversational answer based only on the data above.
"""
    try:
        resp = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": answer_prompt}],
            max_tokens=512,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"OpenAI error while generating answer: {e}")
        return "Error: failed to generate a natural language answer from the query results."
    finally:
        if opened_here and exec_conn:
            exec_conn.close()


# --- UI: two tabs (default: Interesting insights) ---
st.title("RideAustin Data Explorer")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

tabs = st.tabs(["Interesting insights", "Chatbot"])

with tabs[0]:
    # Distance distribution boxplot
    st.subheader("Distance Distribution (Boxplot)")
    try:
        ensure_full_loaded()
        full_df = st.session_state["df"]

        # Try common distance column names first
        dist_col = None
        for c in ("distance_travelled", "distance_traveled", "distance_km", "distance"):
            if c in full_df.columns:
                dist_col = c
                break

        if dist_col is not None:
            # Source distances are in meters — convert to kilometers for display
            distances = full_df[dist_col].dropna().astype(float) / 1000.0
            st.caption(f"Note: values from column '{dist_col}' converted from meters to kilometers for display.")
        else:
            # Attempt to compute distances from lat/long if available
            latlon_cols = ["start_location_lat", "start_location_long", "end_location_lat", "end_location_long"]
            if all(col in full_df.columns for col in latlon_cols):
                clean = full_df.dropna(subset=latlon_cols).copy()
                distances = clean.apply(lambda r: haversine_distance(r["start_location_lat"], r["start_location_long"], r["end_location_lat"], r["end_location_long"]), axis=1)
            else:
                st.info("No distance column found and insufficient lat/long columns to compute distances.")
                distances = pd.Series([], dtype=float)

        if distances.empty:
            st.write("No distance data available to plot.")
        else:
            # Trim extreme outliers using percentile trimming (remove top/bottom 1% by default)
            try:
                trim_pct = 0.01
                lower = float(np.quantile(distances, trim_pct))
                upper = float(np.quantile(distances, 1.0 - trim_pct))
                trimmed = distances[(distances >= lower) & (distances <= upper)]
            except Exception:
                # Fallback if quantile fails
                trimmed = distances

            if trimmed.empty:
                st.write("No distance data available after trimming outliers.")
            else:
                fig_dist, ax_dist = plt.subplots(figsize=(8, 3))
                ax_dist.boxplot(trimmed, vert=False, patch_artist=True, boxprops=dict(facecolor='steelblue', color='black'))
                ax_dist.set_xlabel("Distance (km)")
                ax_dist.set_yticks([])
                ax_dist.set_title("Distance distribution (boxplot) — trimmed to 1st–99th percentiles")
                st.pyplot(fig_dist)

                st.caption(f"Showing distances between {lower:.2f} km and {upper:.2f} km (trimmed {int(trim_pct*100)}% on each tail).")

                # Show simple metrics on the trimmed data
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Median distance", f"{float(np.nanmedian(trimmed)):.2f} km")
                with col2:
                    st.metric("Mean distance", f"{float(np.nanmean(trimmed)):.2f} km")
                with col3:
                    st.metric("Max distance (trimmed)", f"{float(np.nanmax(trimmed)):.2f} km")
    except Exception as e:
        st.error(f"Error creating distance boxplot: {e}")
        
    
    st.subheader("Longest Trip Visualization")
    try:
        ensure_full_loaded()
        full_df = st.session_state["df"]

        long_tabs = st.tabs(["Map", "Long Rides (>100 km)"])

        # Map subtab: show the single longest trip
        with long_tabs[0]:
            try:
                longest_trip, distance_km = find_longest_trip(full_df)
                st.write(f"**Longest Trip Distance:** {distance_km:.2f} km")

                start_lat = longest_trip['start_location_lat']
                start_lon = longest_trip['start_location_long']
                end_lat = longest_trip['end_location_lat']
                end_lon = longest_trip['end_location_long']

                center_lat = (start_lat + end_lat) / 2
                center_lon = (start_lon + end_lon) / 2

                m = folium.Map(location=[center_lat, center_lon], zoom_start=6, tiles="OpenStreetMap")
                folium.Marker(location=[start_lat, start_lon], popup="Start Location", tooltip="Start", icon=folium.Icon(color="green", icon="info-sign")).add_to(m)
                folium.Marker(location=[end_lat, end_lon], popup="End Location", tooltip="End", icon=folium.Icon(color="red", icon="info-sign")).add_to(m)

                # Try to fetch route from OSRM public server (no API key). Fallback to straight line if it fails.
                try:
                    coords = f"{start_lon},{start_lat};{end_lon},{end_lat}"
                    osrm_url = f"http://router.project-osrm.org/route/v1/driving/{coords}?overview=full&geometries=geojson"
                    resp = requests.get(osrm_url, timeout=10)
                    resp.raise_for_status()
                    data = resp.json()
                    if "routes" in data and data["routes"]:
                        route_coords = data["routes"][0]["geometry"]["coordinates"]
                        # OSRM returns [lon, lat] pairs — convert to [lat, lon] for folium
                        route_latlon = [[lat, lon] for lon, lat in route_coords]
                        folium.PolyLine(locations=route_latlon, color="blue", weight=4, opacity=0.9, popup=f"Route (OSRM) — {distance_km:.2f} km").add_to(m)
                    else:
                        # no route returned -> draw simple straight line
                        folium.PolyLine(locations=[[start_lat, start_lon], [end_lat, end_lon]], color="blue", weight=2, opacity=0.8, popup=f"Distance: {distance_km:.2f} km").add_to(m)
                except Exception:
                    # if OSRM fails (rate-limit, network), fallback to straight line
                    folium.PolyLine(locations=[[start_lat, start_lon], [end_lat, end_lon]], color="blue", weight=2, opacity=0.8, popup=f"Distance: {distance_km:.2f} km").add_to(m)

                st_folium.st_folium(m, width=800, height=500)
            except Exception as me:
                st.error(f"Error loading longest-trip map: {me}")

        # Long Rides subtab: show all rides with distance > 100 km
        with long_tabs[1]:
            try:
                # Determine distances (look for distance column or compute from lat/lon)
                dist_col = None
                for c in ("distance_travelled", "distance_traveled", "distance_km", "distance"):
                    if c in full_df.columns:
                        dist_col = c
                        break

                if dist_col is not None:
                    # convert meters -> kilometers
                    distances_km = full_df[dist_col].dropna().astype(float) / 1000.0
                    df_with_dist = full_df.loc[distances_km.index].copy()
                    df_with_dist["_distance_km"] = distances_km
                else:
                    latlon_cols = ["start_location_lat", "start_location_long", "end_location_lat", "end_location_long"]
                    if all(col in full_df.columns for col in latlon_cols):
                        clean = full_df.dropna(subset=latlon_cols).copy()
                        clean["_distance_km"] = clean.apply(lambda r: haversine_distance(r["start_location_lat"], r["start_location_long"], r["end_location_lat"], r["end_location_long"]), axis=1)
                        df_with_dist = clean
                    else:
                        st.info("No distance column and insufficient lat/long columns to compute distances.")
                        df_with_dist = pd.DataFrame()

                if df_with_dist.empty:
                    st.write("No rides available to show for > 100 km filter.")
                else:
                    long_rides = df_with_dist[df_with_dist["_distance_km"] > 100.0].copy()
                    st.write(f"Found {len(long_rides)} rides with distance > 100 km.")
                    if long_rides.empty:
                        st.write("No rides exceed 100 km.")
                    else:
                        #st.dataframe(long_rides.head(200), width='stretch')
                        try:
                            m2 = folium.Map(location=[long_rides.iloc[0]["start_location_lat"], long_rides.iloc[0]["start_location_long"]], zoom_start=6, tiles="OpenStreetMap")
                            for _, row in long_rides.head(500).iterrows():
                                folium.CircleMarker(location=[row["start_location_lat"], row["start_location_long"]], radius=3, color="green", fill=True, fill_opacity=0.7).add_to(m2)
                                folium.CircleMarker(location=[row["end_location_lat"], row["end_location_long"]], radius=3, color="red", fill=True, fill_opacity=0.7).add_to(m2)
                            st_folium.st_folium(m2, width=800, height=500)
                        except Exception:
                            pass
            except Exception as le:
                st.error(f"Error building long rides list: {le}")
    except Exception as e:
        st.error(f"Error loading longest trip visualization: {e}")

    # Correlation matrix: allow user to choose numeric columns
    st.subheader("Correlation Matrix")
    try:
        ensure_full_loaded()
        full_df = st.session_state["df"]

        # User-provided rename map for display
        rename_map = {
            "PRCP": "precipitation",
            "Year": "Car Year",
            "AWND": "average wind speed",
        }

        # Identify numeric columns from dataframe
        numeric_cols = list(full_df.select_dtypes(include=[np.number]).columns)

        # Exclude columns that should not be selectable for correlation (location, ids, etc.)
        exclude_cols = {"end_location_lat", "end_location_long", "start_location_lat", "start_location_long", "charity_id", "free_credit_used"}
        numeric_cols = [c for c in numeric_cols if c not in exclude_cols]

        # Ensure any specifically mentioned columns are included if present (but respect excludes)
        for c in ["distance_travelled","driver_rating","rider_rating","surge_factor","PRCP","Tmax","Tmin","AWND","GustSpeed2","Fog","HeavyFog","Thunder","Year","Rating"]:
            if c in full_df.columns and c not in numeric_cols and c not in exclude_cols:
                numeric_cols.append(c)

        # Build labelled options and mapping from display name to original column
        display_names = [rename_map.get(col, col) for col in numeric_cols]
        display_to_col = {rename_map.get(col, col): col for col in numeric_cols}
        options = [f"{name}" for name in display_names]

        chosen = st.multiselect("Choose 2 or more numeric columns to compute correlation:", options, key="corr_cols")
        if chosen and len(chosen) >= 2:
            # Map selected display names back to original column names
            selected_cols = [display_to_col.get(c, c) for c in chosen]

            # Sanity check: ensure selected columns exist in the dataframe
            missing = [c for c in selected_cols if c not in full_df.columns]
            if missing:
                st.error(f"Selected columns not found in data: {missing}")
            else:
                # Convert selected columns to numeric (coerce errors)
                corr_df = full_df[selected_cols].apply(pd.to_numeric, errors="coerce")
            corr = corr_df.corr()

            st.write("Correlation matrix (Pearson):")
            #st.dataframe(corr, width='stretch')

            # Heatmap using matplotlib
            try:
                fig, ax = plt.subplots(figsize=(max(4, len(selected_cols)), max(4, len(selected_cols))))
                im = ax.imshow(corr.values, cmap="RdBu", vmin=-1, vmax=1)
                ax.set_xticks(range(len(selected_cols)))
                ax.set_yticks(range(len(selected_cols)))
                ax.set_xticklabels([rename_map.get(c, c) for c in selected_cols], rotation=45, ha="right")
                ax.set_yticklabels([rename_map.get(c, c) for c in selected_cols])
                # annotate
                for (i, j), val in np.ndenumerate(corr.values):
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="black", fontsize=9)
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Failed to draw heatmap: {e}")
        elif chosen:
            st.info("Please select at least two numeric columns to compute correlation.")
        else:
            st.write("Select numeric columns above to compute a correlation matrix.")
    except Exception as e:
        st.error(f"Error preparing correlation UI: {e}")
    
    # Show surge factor by time of day
    st.subheader("Surge Factor and Average rides by Time of Day")
    try:
        ensure_full_loaded()
        full_df = st.session_state["df"]
        fig, surge_by_half_hour, rides_by_half_hour, surge_pct, rides_pct = create_surge_by_hour_chart(full_df)
        
        # Display the figure
        st.pyplot(fig)
        
        # Helper: convert half-hour bin to human time
        def bin_to_time(b):
            b = int(b)
            hour = b // 2
            minute = 30 if (b % 2) else 0
            return f"{hour:02d}:{minute:02d}"
        
        # Peak / trough for surge (percent) and rides (percent)
        if surge_pct.isnull().all():
            st.info("No surge data available to compute metrics.")
        else:
            max_surge_idx = int(surge_pct.idxmax())
            min_surge_idx = int(surge_pct.idxmin())
            max_surge_pct = surge_pct.max()
            min_surge_pct = surge_pct.min()
            max_surge_raw = surge_by_half_hour.loc[max_surge_idx]
            min_surge_raw = surge_by_half_hour.loc[min_surge_idx]

            max_rides_idx = int(rides_pct.idxmax())
            min_rides_idx = int(rides_pct.idxmin())
            max_rides_pct = rides_pct.max()
            min_rides_pct = rides_pct.min()
            max_rides_count = int(rides_by_half_hour.loc[max_rides_idx])
            min_rides_count = int(rides_by_half_hour.loc[min_rides_idx])

            # Display four concise metrics
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Peak Surge Time", bin_to_time(max_surge_idx), f"{max_surge_pct:.1f}% (avg surge {max_surge_raw:.3f})")
            with c2:
                st.metric("Peak Rides Time", bin_to_time(max_rides_idx), f"{max_rides_pct:.1f}% ({max_rides_count} rides)")
            with c3:
                st.metric("Lowest Surge Time", bin_to_time(min_surge_idx), f"{min_surge_pct:.1f}% (avg surge {min_surge_raw:.3f})")
            with c4:
                st.metric("Lowest Rides Time", bin_to_time(min_rides_idx), f"{min_rides_pct:.1f}% ({min_rides_count} rides)")

            # Provide a short text summary
            st.write(
                f"Average surge across all half-hours: {float(np.nanmean(surge_by_half_hour)):.3f}. "
                f"Average rides per half-hour: {int(rides_by_half_hour.mean())} (used to compute ride % above/below avg)."
            )
            st.write(
                "Note: bar heights show absolute % above/below average; color encodes whether the value is above (red/orange) or below (blue) the overall average."
            )
    except Exception as e:
        st.error(f"Error loading surge factor visualization: {e}")


    # Car Brands — Average Driver Rating
    st.subheader("Car Brands — Average Driver Rating")
    try:
        ensure_full_loaded()
        full_df = st.session_state["df"]

        if "make" not in full_df.columns:
            st.info("No 'make' column found in the data.")
        else:
            # prefer driver_rating then rating then rider_rating
            rating_col = None
            for cand in ("driver_rating", "rating", "rider_rating"):
                if cand in full_df.columns:
                    rating_col = cand
                    break

            if not rating_col:
                st.info("No rating column found (tried driver_rating/rating/rider_rating).")
            else:
                tmp = full_df[["make", rating_col]].dropna()
                tmp[rating_col] = pd.to_numeric(tmp[rating_col], errors="coerce")
                tmp = tmp.dropna(subset=[rating_col])

                overall = float(tmp[rating_col].mean()) if not tmp.empty else None

                grp = tmp.groupby("make").agg(rides=(rating_col, "count"), avg_rating=(rating_col, "mean")).reset_index()
                grp = grp[grp["rides"] >= 5].sort_values("avg_rating", ascending=False)

                if grp.empty:
                    st.write("No brand rating data (after applying minimum count filter).")
                else:
                    # overall average rating (for percent-diff calculation)
                    overall = float(tmp[rating_col].mean()) if not tmp.empty else 0.0

                    # percent difference from overall average (signed)
                    # protect divide-by-zero by falling back to absolute difference if overall == 0
                    if overall == 0:
                        grp['pct_diff'] = (grp['avg_rating'] - overall)
                    else:
                        grp['pct_diff'] = (grp['avg_rating'] - overall) / overall * 100.0

                    # Let user choose how many top brands to show
                    max_brands = int(min(len(grp), 200))
                    top_n = st.number_input("Show top N brands", min_value=1, max_value=max_brands, value=min(20, max_brands), step=1, key='brands_top_n')

                    # Option to reverse ordering (ascending instead of descending)
                    asc = st.checkbox("Reverse order (ascending)", value=False, key='brands_asc')

                    sort_order = True if asc else False
                    grp_sorted = grp.sort_values('pct_diff', ascending=sort_order)
                    display = grp_sorted.head(int(top_n))

                    # Altair horizontal bar chart showing percent difference from average
                    try:
                        chart = alt.Chart(display).mark_bar().encode(
                            x=alt.X('pct_diff:Q', title='Pct diff from average (%)'),
                            y=alt.Y('make:N', sort=alt.EncodingSortField(field='pct_diff', order='ascending' if asc else 'descending'), title='Make'),
                            color=alt.condition(alt.datum.pct_diff > 0, alt.value('#d7191c'), alt.value('#2c7bb6'))
                        ).properties(
                            height=min(600, 25 * len(display)),
                            width=700
                        )
                        # Replace deprecated `use_container_width` with `width='stretch'`
                        st.altair_chart(chart, width='stretch')
                    except Exception:
                        # Fallback to simple streamlit series bar chart (percent diff)
                        try:
                            st.bar_chart(display.set_index('make')['pct_diff'])
                        except Exception as e:
                            st.error(f"Could not render brand rating chart: {e}")

                    # Print summary: average rating, best and worst brands (from grp with min rides filter)
                    try:
                        overall_display = overall
                        best_row = grp.loc[grp['avg_rating'].idxmax()]
                        worst_row = grp.loc[grp['avg_rating'].idxmin()]

                        st.write("\n")
                        c1, c2, c3 = st.columns(3)
                        with c1:
                            st.metric("Average driver rating", f"{overall_display:.2f}")
                        with c2:
                            st.metric("Best brand", f"{best_row['make']}", f"{best_row['avg_rating']:.2f}")
                        with c3:
                            st.metric("Worst brand", f"{worst_row['make']}", f"{worst_row['avg_rating']:.2f}")
                    except Exception:
                        # If something goes wrong computing best/worst, still continue
                        pass
    except Exception as e:
        st.error(f"Error preparing car-brand overview: {e}")

with tabs[1]:
    st.header("Data Chatbot (Data-Only Answers)")
    st.subheader("Ask anything about RideAustin trips, drivers, and fares")

    # Messages at the top
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state["messages"]:
            st.chat_message(msg["role"]).write(msg["content"])

    # Spinner BELOW messages, ABOVE input
    spinner_container = st.container()

    # Input at the bottom
    input_container = st.container()
    with input_container:
        user_input = st.chat_input("Ask a question about the data...")

    if user_input:
        # Save + show user message
        st.session_state["messages"].append({"role": "user", "content": user_input})
        with chat_container:
            st.chat_message("user").write(user_input)

        # Spinner now appears BELOW the messages + ABOVE input
        with spinner_container:
            with st.spinner("Thinking..."):
                answer = answer_from_sql(user_input)

        spinner_container.empty()

        # Save + show assistant message
        st.session_state["messages"].append({"role": "assistant", "content": answer})
        with chat_container:
            st.chat_message("assistant").write(answer)
