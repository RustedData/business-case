
import streamlit as st
import pandas as pd
import openai
import os
import sqlite3
import tempfile
from math import radians, sin, cos, sqrt, atan2
import folium
from streamlit_folium import st_folium
import matplotlib.pyplot as plt
import numpy as np

# Load API keys from .env for local development
from dotenv import load_dotenv
load_dotenv()

# Prefer Streamlit secrets (Streamlit Cloud) then environment variables
OPENAI_API_KEY = None
try:
    OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")
except Exception:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    st.error(
        "OpenAI API key not found. Set `OPENAI_API_KEY` in Streamlit Secrets (recommended) or as an environment variable.\n"
        "See README for instructions. The app cannot call OpenAI without this key."
    )
    st.stop()

openai.api_key = OPENAI_API_KEY


@st.cache_data(show_spinner=True)
def load_data_sample(nrows: int = 1000):
    CSV_URL = "https://www.dropbox.com/scl/fi/9k53mrii5tf35r4443cmv/RideAustin_Weather.csv?rlkey=bg0pl1cmn542ypxzt92y16wxt&st=koq4dks4&dl=1"
    return pd.read_csv(
        CSV_URL,
        encoding="utf-8",
        delimiter=",",
        on_bad_lines="skip",
        low_memory=False,
        nrows=nrows,
    )


@st.cache_data(show_spinner=True)
def load_data_full():
    CSV_URL = "https://www.dropbox.com/scl/fi/9k53mrii5tf35r4443cmv/RideAustin_Weather.csv?rlkey=bg0pl1cmn542ypxzt92y16wxt&st=koq4dks4&dl=1"
    return pd.read_csv(
        CSV_URL,
        encoding="utf-8",
        delimiter=",",
        on_bad_lines="skip",
        low_memory=False,
    )


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    Returns distance in kilometers
    """
    R = 6371  # Radius of earth in kilometers
    
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    distance = R * c
    
    return distance


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
    Create a circular histogram showing average surge_factor by half-hour.
    Returns the matplotlib figure.
    """
    # Create a working copy
    work_df = df.copy()
    
    # Convert started_on to datetime
    work_df['started_on'] = pd.to_datetime(work_df['started_on'])
    
    # Extract hour and minute to create half-hour bins (0-47 for 48 half-hour periods in a day)
    work_df['hour_minute'] = work_df['started_on'].dt.hour * 60 + work_df['started_on'].dt.minute
    work_df['half_hour_bin'] = (work_df['hour_minute'] // 30).astype(int)
    
    # Calculate average surge_factor for each half-hour bin
    surge_by_half_hour = work_df.groupby('half_hour_bin')['surge_factor'].mean()
    
    # Create a polar (circular) plot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='polar')
    
    # Create the angles for the polar plot (48 bins = 48 half-hour periods)
    angles = np.linspace(0, 2 * np.pi, 48, endpoint=False)
    values = surge_by_half_hour.values
    
    # Plot the bars
    bars = ax.bar(angles, values, width=2*np.pi/48, alpha=0.8, color='steelblue', edgecolor='black', linewidth=0.5)
    
    # Set the angle for 0 to be at the top (12 o'clock position)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    # Set the radial grid
    ax.set_ylim(0, values.max() * 1.1)
    
    # Add hour labels at key positions (every 2 hours = 4 half-hour bins)
    hour_angles = np.linspace(0, 2*np.pi, 24, endpoint=False)
    hour_labels = [f"{i:02d}:00" for i in range(24)]
    ax.set_xticks(hour_angles)
    ax.set_xticklabels(hour_labels, size=8)
    
    # Set title and labels
    ax.set_title('Average Surge Factor by Half-Hour of Day\n(Clock-like View)', 
                 size=14, weight='bold', pad=20)
    ax.set_ylabel('Surge Factor', labelpad=30)
    
    plt.tight_layout()
    
    return fig, surge_by_half_hour


# Load a small sample immediately so the app starts quickly for health checks
sample_df = load_data_sample()

# Session state for data + DB path (full DB stored in a temp file when loaded)
st.session_state.setdefault("df", sample_df)
st.session_state.setdefault("db_path", None)
st.session_state.setdefault("columns", list(sample_df.columns))
st.session_state.setdefault("column_types", {})
st.session_state.setdefault("loaded_full", False)


def get_column_types(df: pd.DataFrame) -> dict:
    type_map = {}
    for col in df.columns:
        dtype = str(df[col].dtype)
        if dtype.startswith("int") or dtype.startswith("float"):
            type_map[col] = "numeric"
        elif dtype.startswith("datetime"):
            type_map[col] = "date"
        else:
            type_map[col] = "text"
    return type_map


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
    st.header("Interesting insights")
    
    # Show the longest trip on a map
    st.subheader("Longest Trip Visualization")
    try:
        ensure_full_loaded()
        full_df = st.session_state["df"]
        longest_trip, distance_km = find_longest_trip(full_df)
        
        st.write(f"**Longest Trip Distance:** {distance_km:.2f} km")
        
        # Create a map centered on the midpoint between start and end
        start_lat = longest_trip['start_location_lat']
        start_lon = longest_trip['start_location_long']
        end_lat = longest_trip['end_location_lat']
        end_lon = longest_trip['end_location_long']
        
        center_lat = (start_lat + end_lat) / 2
        center_lon = (start_lon + end_lon) / 2
        
        # Create folium map
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=8,
            tiles="OpenStreetMap"
        )
        
        # Add markers for start and end locations
        folium.Marker(
            location=[start_lat, start_lon],
            popup="Start Location",
            tooltip="Start",
            icon=folium.Icon(color="green", icon="info-sign")
        ).add_to(m)
        
        folium.Marker(
            location=[end_lat, end_lon],
            popup="End Location",
            tooltip="End",
            icon=folium.Icon(color="red", icon="info-sign")
        ).add_to(m)
        
        # Add a great circle line between start and end
        folium.PolyLine(
            locations=[[start_lat, start_lon], [end_lat, end_lon]],
            color="blue",
            weight=2,
            opacity=0.8,
            popup=f"Distance: {distance_km:.2f} km"
        ).add_to(m)
        
        # Display the map
        st_folium(m, width=700, height=500)
        
    except Exception as e:
        st.error(f"Error loading longest trip visualization: {e}")
    
    # Show surge factor by time of day
    st.subheader("Surge Factor by Time of Day")
    try:
        ensure_full_loaded()
        full_df = st.session_state["df"]
        fig, surge_by_half_hour = create_surge_by_hour_chart(full_df)
        
        # Display the figure
        st.pyplot(fig)
        
        # Show summary statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Peak Surge Time", 
                     f"{surge_by_half_hour.idxmax() * 0.5:.1f} hours",
                     f"{surge_by_half_hour.max():.3f}")
        with col2:
            st.metric("Lowest Surge Time",
                     f"{surge_by_half_hour.idxmin() * 0.5:.1f} hours",
                     f"{surge_by_half_hour.min():.3f}")
        with col3:
            st.metric("Average Surge",
                     "",
                     f"{surge_by_half_hour.mean():.3f}")
        
        st.write("""
**Insight:** There is a clear relationship between time of day and surge pricing. 
Peak surges occur during typical commute hours (morning rush around 8 AM), 
while the lowest surges appear during midday hours.
        """)
        
    except Exception as e:
        st.error(f"Error loading surge factor visualization: {e}")

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
