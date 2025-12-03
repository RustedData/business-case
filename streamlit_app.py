
import streamlit as st
import pandas as pd
import openai
import os
import sqlite3

# Load API keys from .env or environment
from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

openai.api_key = OPENAI_API_KEY



import streamlit as st

@st.cache_data(show_spinner=True)
def load_data():
    CSV_URL = "https://www.dropbox.com/scl/fi/9k53mrii5tf35r4443cmv/RideAustin_Weather.csv?rlkey=bg0pl1cmn542ypxzt92y16wxt&st=koq4dks4&dl=1"  # replace with your actual link
    return pd.read_csv(CSV_URL, encoding="utf-8", delimiter=",", on_bad_lines="skip")

df = load_data()

conn = sqlite3.connect(":memory:")
df.to_sql("rides", conn, index=False, if_exists="replace")


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

column_types = get_column_types(df)

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
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=4096
    )
    sql = response.choices[0].message.content.strip()
    return sql



def answer_from_sql(question):
    columns = df.columns.tolist()
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

    sql = extract_sql(nl_to_sql(question, columns, column_types))
    for attempt in range(2):
        try:
            result_df = pd.read_sql_query(sql, conn)
            break
        except Exception as e:
            if attempt == 0:
                fix_prompt = f"""
You are a SQL expert. The following SQL query resulted in an error when run on a SQLite database. Please fix the query. Only return the corrected SQL query, nothing else.

Original question: {question}
Original SQL: {sql}
Error: {e}
"""
                response = openai.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": fix_prompt}],
                    max_tokens=4096
                )
                sql = extract_sql(response.choices[0].message.content)
                continue
            else:
                return f"SQL Error: {e}\nSQL: {sql}"
    result_sample = result_df.head(50)
    prompt = f"""
You are a helpful assistant. Use the following data (from a SQL query) to answer the user's question in a clear, conversational way. If the answer is not in the data, say 'I don't know.'

Data:
{result_sample.to_string(index=False)}

Question: {question}

Please provide a concise, conversational answer based only on the data above.
"""
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=4096
    )
    return response.choices[0].message.content.strip()


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