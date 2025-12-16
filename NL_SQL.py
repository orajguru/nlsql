import streamlit as st
import pandas as pd
from groq import Groq
from sqlalchemy import create_engine, text
import os
import re
import hashlib

# ============================
# Configuration
# ============================
st.set_page_config(
    page_title="NL → SQL Analytics Copilot",
    page_icon="📊",
    layout="wide"
)

MODEL_NAME = "llama-3.1-70b-instant"
#GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_KEY = st.secrets.get("groq", {}).get("key")

# ============================
# Initialize Clients
# ============================
client = Groq(api_key=GROQ_API_KEY)
engine = create_engine("sqlite:///sample.db")

# ============================
# Schema with FK inference
# ============================
SCHEMA_DESCRIPTION = """
Tables:

employees
- id (INTEGER, PK)
- name (TEXT)
- department_id (INTEGER, FK → departments.id)
- salary (INTEGER)
- joining_date (DATE)

 departments
- id (INTEGER, PK)
- name (TEXT)

Relationships:
- employees.department_id joins to departments.id

Guidelines:
- Always use explicit JOINs
- Prefer table aliases (e, d)
- Use departments.name for grouping and display
"""

ALLOWED_TABLES = {"employees", "departments"}
MAX_ROWS = 1000
FORBIDDEN_KEYWORDS = ["insert", "update", "delete", "drop", "alter", "truncate", "create"]

# ============================
# Semantic Cache
# ============================
if "semantic_cache" not in st.session_state:
    st.session_state.semantic_cache = {}


def semantic_key(question: str) -> str:
    return hashlib.sha256(question.lower().encode()).hexdigest()

# ============================
# Helpers
# ============================
def validate_sql(sql):
    sql_l = sql.lower()
    if not sql_l.startswith("select"):
        raise ValueError("Only SELECT queries allowed")
    for kw in FORBIDDEN_KEYWORDS:
        if kw in sql_l:
            raise ValueError(f"Forbidden keyword: {kw}")
    if "limit" not in sql_l:
        sql += f" LIMIT {MAX_ROWS}"
    return sql


def generate_sql(nl, history):
    messages = [
        {"role": "system", "content": "You are a senior analytics engineer generating SQL with joins."},
        {"role": "system", "content": "Rules: Use joins based on relationships. Output only SQL. SELECT only."},
        {"role": "system", "content": f"Schema:\n{SCHEMA_DESCRIPTION}"}
    ]

    for h in history[-4:]:
        messages.append({"role": "user", "content": h["content"]})
        #messages.append({"role": "assistant", "content": h["sql"]})

    messages.append({"role": "user", "content": nl})

    res = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.05
    )
    return res.choices[0].message.content.strip()


def explain_sql(sql):
    res = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "Explain SQL clearly for analysts."},
            {"role": "user", "content": sql}
        ],
        temperature=0.2
    )
    return res.choices[0].message.content


def explain_chart(df, chart_type, x, y):
    sample = df.head(5).to_csv(index=False)
    prompt = f"""
Explain this chart in business terms.
Chart type: {chart_type}
X-axis: {x}
Y-axis: {y}
Sample data:
{sample}
"""
    res = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    return res.choices[0].message.content


def auto_chart(df):
    num_cols = df.select_dtypes(include="number").columns
    cat_cols = df.select_dtypes(exclude="number").columns
    if len(num_cols) >= 1 and len(cat_cols) >= 1:
        return "bar", cat_cols[0], num_cols[0]
    if len(num_cols) >= 2:
        return "line", num_cols[0], num_cols[1]
    return None, None, None

# ============================
# Session State
# ============================
if "messages" not in st.session_state:
    st.session_state.messages = []

# ============================
# Chat UI
# ============================
st.title("🤖 Analytics Copilot")
st.caption("Chat with your data across multiple tables")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask a question about your data")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    try:
        cache_key = semantic_key(user_input)

        if cache_key in st.session_state.semantic_cache:
            cached = st.session_state.semantic_cache[cache_key]
            sql, df = cached["sql"], cached["df"]
            cached_hit = True
        else:
            sql = generate_sql(user_input, st.session_state.messages)
            sql = validate_sql(sql)
            df = pd.read_sql(text(sql), engine)
            st.session_state.semantic_cache[cache_key] = {"sql": sql, "df": df}
            cached_hit = False

        chart_type, x, y = auto_chart(df)

        with st.chat_message("assistant"):
            if cached_hit:
                st.info("⚡ Loaded from cache")

            st.dataframe(df, use_container_width=True)

            if chart_type == "bar":
                st.bar_chart(df, x=x, y=y)
            elif chart_type == "line":
                st.line_chart(df[[x, y]])

            with st.expander("Explain Chart"):
                if chart_type:
                    st.markdown(explain_chart(df, chart_type, x, y))
                else:
                    st.markdown("No chart explanation available")

            with st.expander("View SQL"):
                st.code(sql, language="sql")

            with st.expander("Explain SQL"):
                st.markdown(explain_sql(sql))

        st.session_state.messages.append({
            "role": "assistant",
            "content": f"Returned {len(df)} rows"
        })

    except Exception as e:
        with st.chat_message("assistant"):
            st.error(str(e))
