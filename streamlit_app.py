
import streamlit as st
import pandas as pd
import openai
import os
import sqlite3
import tempfile

# Load API keys from .env for local development
from dotenv import load_dotenv
load_dotenv()

# Prefer Streamlit secrets (Streamlit Cloud) then environment variables
OPENAI_API_KEY = None
try:
    OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")
except Exception:
    # If st.secrets is unavailable for any reason, fall back to env
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    # final fallback to environment variable in case above didn't pick it up
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    st.error(
        "OpenAI API key not found. Set `OPENAI_API_KEY` in Streamlit Secrets (recommended) or as an environment variable.\n"
        "See README for instructions. The app cannot call OpenAI without this key."
    )
    # Stop further execution to avoid the low-level redacted OpenAI error
    st.stop()

openai.api_key = OPENAI_API_KEY



import streamlit as st

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
        delimiter="," ,
        on_bad_lines="skip",
        low_memory=False,
    )

# Load a small sample immediately so the app starts quickly for health checks
sample_df = load_data_sample()

# Store dataset and DB in session state; full data / DB will be created lazily
st.session_state.setdefault("df", sample_df)
st.session_state.setdefault("db_path", None)
st.session_state.setdefault("columns", list(sample_df.columns))
st.session_state.setdefault("column_types", {})
st.session_state.setdefault("loaded_full", False)


def ensure_full_loaded():
    """On first real user query, load the full dataset and create an in-memory SQLite DB."""
    if st.session_state.get("loaded_full"):
        return
    try:
        full_df = load_data_full()
    except Exception as e:
        st.error(f"Failed to load full dataset: {e}")
        return
    # Write the full dataframe to a temporary on-disk SQLite DB so each
    # request can open its own connection (avoids cross-thread sqlite errors).
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


# Step 1: Get column types for type-aware prompt
def get_column_types(df):
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

# Ensure session column types are initialized from the sample (if not already)
if not st.session_state.get("column_types"):
    st.session_state["column_types"] = get_column_types(st.session_state["df"])

# Step 2: Type-aware prompt engineering
def nl_to_sql(question, columns, column_types):
    col_desc = ", ".join([f"{col} ({column_types[col]})" for col in columns])
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
            max_tokens=4096,
        )
        sql = response.choices[0].message.content.strip()
        return sql
    except Exception as e:
        # Raise a runtime error so callers can display a friendly message
        raise RuntimeError(f"OpenAI error during SQL generation: {e}")



def answer_from_sql(question):
    # Ensure DB/data availability
    ensure_full_loaded()

    df = st.session_state.get("df")
    db_path = st.session_state.get("db_path")
    columns = st.session_state.get("columns", list(df.columns))
    column_types = st.session_state.get("column_types", get_column_types(df))
    import re
    def extract_sql(text):
        # Remove any prefix like 'Corrected SQL:' or code block markers
        import re
        lines = text.strip().splitlines()
        sql_lines = []
        for line in lines:
            # Skip lines that are not SQL
            if line.strip().lower().startswith('corrected sql:'):
                line = line.split(':', 1)[-1].strip()
                if line:
                    sql_lines.append(line)
                continue
            if line.strip().startswith('```'):
                continue
            sql_lines.append(line)
        sql = '\n'.join(sql_lines).strip()
        # Replace unsupported median SQL with SQLite-compatible version
        # Replace PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY col) with SQLite median subquery
        def median_sql(col):
            return f"(SELECT AVG(val) FROM (SELECT {col} AS val FROM rides WHERE {col} IS NOT NULL ORDER BY {col} LIMIT 2 - (SELECT COUNT(*) FROM rides WHERE {col} IS NOT NULL) % 2 OFFSET (SELECT (COUNT(*) - 1) / 2 FROM rides WHERE {col} IS NOT NULL)))"
        sql = re.sub(r"PERCENTILE_CONT\(0\.5\) WITHIN GROUP \(ORDER BY ([a-zA-Z0-9_]+)\)", lambda m: median_sql(m.group(1)), sql, flags=re.IGNORECASE)
        return sql

    try:
        raw_sql = nl_to_sql(question, columns, column_types)
    except RuntimeError as e:
        st.error(str(e))
        return "Error: failed to generate SQL from your question. Check API key and logs."
    sql = extract_sql(raw_sql)
    # Execute SQL. If no persistent conn exists (we're on sample), create a temp in-memory DB.
    temp_conn = None
    exec_conn = None
    opened_here = False
    if db_path:
        try:
            exec_conn = sqlite3.connect(db_path)
            opened_here = True
        except Exception as e:
            return f"Failed to open DB at {db_path}: {e}"
    else:
        # Use an in-memory temporary DB built from the sample df for quick queries
        try:
            temp_conn = sqlite3.connect(":memory:")
            df.to_sql("rides", temp_conn, index=False, if_exists="replace")
            exec_conn = temp_conn
            opened_here = True
        except Exception as e:
            if temp_conn:
                temp_conn.close()
            return f"Failed to prepare in-memory DB for sample execution: {e}"

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
                    response = openai.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[{"role": "user", "content": fix_prompt}],
                        max_tokens=4096,
                    )
                    sql = extract_sql(response.choices[0].message.content)
                    continue
                except Exception as oe:
                    if temp_conn:
                        temp_conn.close()
                    st.error(f"OpenAI error while attempting to fix SQL: {oe}")
                    return "Error: could not repair SQL. See logs."
            else:
                if opened_here and exec_conn:
                    exec_conn.close()
                return f"SQL Error: {e}\nSQL: {sql}"
    result_sample = result_df.head(50)
    prompt = f"""
You are a helpful assistant. Use the following data (from a SQL query) to answer the user's question in a clear, conversational way. If the answer is not in the data, say 'I'm only here to help you with the RideAustin Dataset'.'

Data:
{result_sample.to_string(index=False)}

Question: {question}

Please provide a concise, conversational answer based only on the data above.
"""
    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=4096,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"OpenAI error while generating answer: {e}")
        return "Error: failed to generate a natural language answer from the query results."
    finally:
        if opened_here and exec_conn:
            exec_conn.close()


st.title("Data Chatbot (Data-Only Answers)")
st.subheader("Ask Anything About RideAustin Trips, Drivers, and Fares")

if "messages" not in st.session_state:
    st.session_state["messages"] = []




user_input = st.chat_input("Ask a question about the data...")
if user_input:
    st.session_state["messages"].append({"role": "user", "content": user_input})
    # Rerender chat so user message appears immediately
    for msg in st.session_state["messages"]:
        st.chat_message(msg["role"]).write(msg["content"])
    with st.spinner("Thinking..."):
        answer = answer_from_sql(user_input)
    st.session_state["messages"].append({"role": "assistant", "content": answer})
    st.chat_message("assistant").write(answer)