import streamlit as st
import pandas as pd
from groq import Groq
from sqlalchemy import create_engine, text
import hashlib
import re

# ============================
# Configuration
# ============================
st.set_page_config(
    page_title="NL → SQL Analytics Copilot (No Column Clash)",
    page_icon="📊",
    layout="wide"
)

MODEL_NAME = "llama-3.1-8b-instant"
GROQ_API_KEY = st.secrets.get("groq", {}).get("key")

# ============================
# Initialize Clients
# ============================
client = Groq(api_key=GROQ_API_KEY)
engine = create_engine("sqlite:///sample.db")

# ============================
# Initialize Database (NO CLASH schema)
# ============================
def init_db():
    with engine.begin() as conn:
        # Department table with prefixed columns
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS dim_department (
                dept_id INTEGER PRIMARY KEY,
                department_name TEXT NOT NULL
            );
        """))

        # Employee table with prefixed columns
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS fact_employee (
                emp_id INTEGER PRIMARY KEY,
                employee_name TEXT NOT NULL,
                dept_id INTEGER,
                employee_salary INTEGER,
                joining_date DATE,
                FOREIGN KEY(dept_id) REFERENCES dim_department(dept_id)
            );
        """))

        # Seed departments
        if conn.execute(text("SELECT COUNT(*) FROM dim_department")).scalar() == 0:
            conn.execute(text("""
                INSERT INTO dim_department (dept_id, department_name) VALUES
                (1, 'IT'),
                (2, 'HR'),
                (3, 'Finance'),
                (4, 'Operations');
            """))

        # Seed employees
        if conn.execute(text("SELECT COUNT(*) FROM fact_employee")).scalar() == 0:
            conn.execute(text("""
                INSERT INTO fact_employee (employee_name, dept_id, employee_salary, joining_date) VALUES
                ('Alice', 1, 120000, '2021-04-12'),
                ('Bob', 1, 95000, '2022-06-01'),
                ('Carol', 2, 60000, '2023-02-18'),
                ('Dave', 3, 80000, '2020-09-10'),
                ('Eve', 4, 70000, '2022-11-05'),
                ('Frank', 1, 110000, '2023-01-20');
            """))

init_db()

# ============================
# Schema (NO CLASH, CANONICAL)
# ============================
SCHEMA_DESCRIPTION = """
Tables:

fact_employee
- emp_id (PK)
- employee_name
- dept_id (FK → dim_department.dept_id)
- employee_salary
- joining_date

dim_department
- dept_id (PK)
- department_name

Relationships:
- fact_employee.dept_id → dim_department.dept_id

RULES:
- NEVER invent columns
- Use names exactly as defined
- Aliases are OPTIONAL (names are already unique)
"""

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
def validate_sql(sql: str) -> str:
    s = sql.strip().lower()
    if not (s.startswith("select") or s.startswith("with")):
        raise ValueError("Only SELECT queries allowed")
    for kw in FORBIDDEN_KEYWORDS:
        if re.search(rf"\b{kw}\b", s):
            raise ValueError(f"Forbidden keyword: {kw}")
    if "limit" not in s:
        sql += f" LIMIT {MAX_ROWS}"
    return sql


def sqlite_sql_fixups(sql: str) -> str:
    return re.sub(
        r"EXTRACT\s*\(\s*YEAR\s+FROM\s+([^)]+)\)",
        r"strftime('%Y', \1)",
        sql,
        flags=re.IGNORECASE
    )
def normalize_sql(sql: str) -> str:
    # Remove trailing semicolons and extra whitespace
    return sql.strip().rstrip(";")

def extract_sql(text_out: str) -> str:
    text_out = text_out.strip()
    if text_out.lower().startswith(("select", "with")):
        return text_out
    m = re.search(r"(select\s+.+|with\s+.+)", text_out, re.I | re.S)
    if not m:
        raise ValueError("No valid SQL returned by model")
    return m.group(1).strip()


def generate_sql(nl: str, history: list) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a SQLite SQL generator.\n"
                "Return ONLY valid SQL.\n"
                "Use ONLY the tables and columns exactly as defined.\n"
                "Do NOT invent names."
            )
        },
        {"role": "system", "content": f"Schema:\n{SCHEMA_DESCRIPTION}"}
    ]

    for h in history[-4:]:
        messages.append({"role": h["role"], "content": h["content"]})

    messages.append({"role": "user", "content": nl})

    res = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.05
    )
    return res.choices[0].message.content.strip()


# ============================
# UI
# ============================
if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("🤖 Analytics Copilot")
st.caption("No column clashes by design")

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

user_input = st.chat_input("Ask a question about your data")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    try:
        key = semantic_key(user_input)
        if key in st.session_state.semantic_cache:
            sql, df = st.session_state.semantic_cache[key]
        else:
            raw = generate_sql(user_input, st.session_state.messages)
            sql = validate_sql(sqlite_sql_fixups(extract_sql(raw)))
            sql = normalize_sql(sql)
            df = pd.read_sql(text(sql), engine)
            st.session_state.semantic_cache[key] = (sql, df)

        with st.chat_message("assistant"):
            st.dataframe(df, use_container_width=True)
            with st.expander("View SQL"):
                st.code(sql, language="sql")

        st.session_state.messages.append({"role": "assistant", "content": f"Returned {len(df)} rows"})

    except Exception as e:
        with st.chat_message("assistant"):
            st.error(str(e))
