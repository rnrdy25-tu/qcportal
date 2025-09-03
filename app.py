# app.py — Quality Portal - Pilot (Non-Conformities: Import + View + Export)

import os
import io
import json
import sqlite3
from pathlib import Path
from datetime import datetime, date

import pandas as pd
import streamlit as st

# ──────────────────────────────────────────────────────────────────────────────
# Storage (cloud-safe)
# ──────────────────────────────────────────────────────────────────────────────
def pick_data_dir() -> Path:
    """
    Find a writable directory both locally and on Streamlit Cloud.
    """
    for base in (Path("/mount/data"), Path("/tmp/quality_portal")):
        try:
            base.mkdir(parents=True, exist_ok=True)
            (base / ".write_probe").write_text("ok", encoding="utf-8")
            return base
        except Exception:
            pass
    raise RuntimeError("No writable directory available")

DATA_DIR = pick_data_dir()
DB_PATH  = DATA_DIR / "quality_portal.sqlite3"

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def now_iso() -> str:
    return datetime.utcnow().isoformat()

def cur_user() -> str:
    # fallback if the platform doesn't expose the user name
    return os.environ.get("USERNAME") or os.environ.get("USER") or "Admin1"

# ──────────────────────────────────────────────────────────────────────────────
# DB schema + migrations
# ──────────────────────────────────────────────────────────────────────────────
TABLE_NC = "non_conformities"

SCHEMA_NC = f"""
CREATE TABLE IF NOT EXISTS {TABLE_NC}(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at   TEXT,      -- ISO datetime or date
  model_no     TEXT,
  model_version TEXT,
  sn           TEXT,
  mo           TEXT,
  nc_type      TEXT,      -- Nonconformity/type
  description  TEXT,
  category     TEXT,      -- Minor/Major/Critical or provided
  line         TEXT,
  station      TEXT,
  reporter     TEXT,
  month        TEXT,
  week         TEXT,
  customer     TEXT,      -- 客戶/供應商
  extra        JSON
);
"""

def get_conn():
    return sqlite3.connect(DB_PATH)

def _cols(c, table: str) -> set:
    return {r[1] for r in c.execute(f"PRAGMA table_info({table})").fetchall()}

def _ensure_column(c, table: str, col: str, decl: str, default=None):
    cols = _cols(c, table)
    if col not in cols:
        c.execute(f"ALTER TABLE {table} ADD COLUMN {col} {decl}")
        if default is not None:
            c.execute(f"UPDATE {table} SET {col}=?", (default,))

def init_db_and_migrate():
    with get_conn() as c:
        # Create table if missing
        c.execute(SCHEMA_NC)

        # Add columns that might not exist in older files
        _ensure_column(c, TABLE_NC, "created_at", "TEXT")
        _ensure_column(c, TABLE_NC, "model_no", "TEXT")
        _ensure_column(c, TABLE_NC, "model_version", "TEXT", "")
        _ensure_column(c, TABLE_NC, "sn", "TEXT", "")
        _ensure_column(c, TABLE_NC, "mo", "TEXT", "")
        _ensure_column(c, TABLE_NC, "nc_type", "TEXT", "")
        _ensure_column(c, TABLE_NC, "description", "TEXT", "")
        _ensure_column(c, TABLE_NC, "category", "TEXT", "")
        _ensure_column(c, TABLE_NC, "line", "TEXT", "")
        _ensure_column(c, TABLE_NC, "station", "TEXT", "")
        _ensure_column(c, TABLE_NC, "reporter", "TEXT", "")
        _ensure_column(c, TABLE_NC, "month", "TEXT", "")
        _ensure_column(c, TABLE_NC, "week", "TEXT", "")
        _ensure_column(c, TABLE_NC, "customer", "TEXT", "")
        _ensure_column(c, TABLE_NC, "extra", "JSON", None)

        c.commit()

init_db_and_migrate()

# ──────────────────────────────────────────────────────────────────────────────
# CSV header normalizer
# (maps many possible column names to a stable internal schema)
# ──────────────────────────────────────────────────────────────────────────────
HEADER_MAP = {
    # created/date fields
    "date": "created_at",
    "application date": "created_at",
    "發出日期": "created_at",          # CHT sample
    "created_at": "created_at",

    # model identifiers
    "model": "model_no",
    "機種/料號": "model_no",
    "model/part no.": "model_no",
    "model no": "model_no",

    "model version": "model_version",
    "version": "model_version",

    "sn": "sn",
    "barcode": "sn",

    "mo": "mo",
    "工作單/採購單號": "mo",
    "mo/po": "mo",

    # nonconformity fields
    "nonconformity": "nc_type",
    "不符合分類": "nc_type",
    "defective item": "nc_type",
    "異常分類": "nc_type",

    "description": "description",
    "不符合說明": "description",

    "category": "category",
    "severity": "category",   # if you use Minor/Major/Critical here

    "line": "line",
    "線別": "line",

    "work station": "station",
    "作業站別": "station",

    "reporter": "reporter",
    "提報人員": "reporter",

    "customer": "customer",
    "客戶/供應商": "customer",

    "month": "month",
    "月份": "month",

    "week": "week",
    "週數": "week",
}

def normalize_headers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy of df with normalized (lowercased) headers mapped to
    the internal schema. Unknown columns are kept for extra JSON.
    """
    new = {}
    for col in df.columns:
        key = str(col).strip().lower()
        mapped = HEADER_MAP.get(key, key)  # keep originals if unknown
        new[col] = mapped
    return df.rename(columns=new)

def coerce_date(v):
    """Try to convert v to ISO date string (YYYY-MM-DD)."""
    if pd.isna(v):
        return None
    if isinstance(v, (datetime, date)):
        return pd.to_datetime(v).date().isoformat()
    try:
        return pd.to_datetime(v, errors="coerce").date().isoformat()
    except Exception:
        return None

# ──────────────────────────────────────────────────────────────────────────────
# Insert / Load
# ──────────────────────────────────────────────────────────────────────────────
def insert_nc_rows(rows: list[dict]) -> int:
    """
    Insert many non-conformity rows. Returns number of inserted rows.
    """
    fields = ["created_at", "model_no", "model_version", "sn", "mo",
              "nc_type", "description", "category", "line", "station",
              "reporter", "month", "week", "customer", "extra"]
    values = [[r.get(k) for k in fields] for r in rows]
    with get_conn() as c:
        c.executemany(
            f"INSERT INTO {TABLE_NC}({','.join(fields)}) VALUES({','.join(['?']*len(fields))})",
            values,
        )
        c.commit()
    return len(values)

@st.cache_data(show_spinner=False)
def load_nc_df(model=None, version=None, sn=None, mo=None, text=None,
               dt_from=None, dt_to=None, limit=2000) -> pd.DataFrame:
    q = f"""
    SELECT id, created_at, model_no, model_version, sn, mo,
           nc_type, description, category, line, station, reporter,
           month, week, customer
    FROM {TABLE_NC}
    WHERE 1=1
    """
    params = []

    if model:
        q += " AND model_no LIKE ?"
        params.append(f"%{model.strip()}%")
    if version:
        q += " AND model_version LIKE ?"
        params.append(f"%{version.strip()}%")
    if sn:
        q += " AND sn LIKE ?"
        params.append(f"%{sn.strip()}%")
    if mo:
        q += " AND mo LIKE ?"
        params.append(f"%{mo.strip()}%")
    if text:
        q += " AND (description LIKE ? OR nc_type LIKE ? OR reporter LIKE ? OR category LIKE ?)"
        params.extend([f"%{text.strip()}%"]*4)
    if dt_from:
        q += " AND created_at >= ?"
        params.append(dt_from)
    if dt_to:
        q += " AND created_at <= ?"
        params.append(dt_to)

    q += " ORDER BY created_at DESC, id DESC LIMIT ?"
    params.append(limit)

    with get_conn() as c:
        return pd.read_sql_query(q, c, params=params)

def delete_nc_row(row_id: int):
    with get_conn() as c:
        c.execute(f"DELETE FROM {TABLE_NC} WHERE id=?", (row_id,))
        c.commit()

# ──────────────────────────────────────────────────────────────────────────────
# UI
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Quality Portal - Pilot", layout="wide")
st.title("Quality Portal - Pilot")

# Sidebar ─ Upload
with st.sidebar:
    st.header("📤 Import Non-Conformities")
    st.caption("Upload a CSV (Teams export or Excel → CSV). Headers in English or 中文 are OK.")
    csv_file = st.file_uploader("CSV file", type=["csv"], key="nc_csv")

    if csv_file:
        try:
            raw = csv_file.read()
            df = pd.read_csv(io.BytesIO(raw), encoding="utf-8", keep_default_na=False)
        except UnicodeDecodeError:
            csv_file.seek(0)
            df = pd.read_csv(csv_file, encoding="latin-1", keep_default_na=False)

        # normalize headers and coerce types
        ndf = normalize_headers(df.copy())
        # build rows
        rows = []
        for _, r in ndf.iterrows():
            # split known columns from extras
            known = {
                "created_at": coerce_date(r.get("created_at")),
                "model_no": str(r.get("model_no") or "").strip(),
                "model_version": str(r.get("model_version") or "").strip(),
                "sn": str(r.get("sn") or "").strip(),
                "mo": str(r.get("mo") or "").strip(),
                "nc_type": str(r.get("nc_type") or "").strip(),
                "description": str(r.get("description") or "").strip(),
                "category": str(r.get("category") or "").strip(),
                "line": str(r.get("line") or "").strip(),
                "station": str(r.get("station") or "").strip(),
                "reporter": str(r.get("reporter") or cur_user()).strip() or cur_user(),
                "month": str(r.get("month") or "").strip(),
                "week": str(r.get("week") or "").strip(),
                "customer": str(r.get("customer") or "").strip(),
            }
            # everything else => extra JSON
            keep_cols = set(known.keys())
            extra = {k: r[k] for k in ndf.columns if k not in keep_cols}
            known["extra"] = json.dumps(extra, ensure_ascii=False) if extra else None
            # if date missing, use today
            if not known["created_at"]:
                known["created_at"] = date.today().isoformat()
            rows.append(known)

        if rows:
            inserted = insert_nc_rows(rows)
            load_nc_df.clear()   # clear cache so new rows appear
            st.success(f"Imported {inserted} record(s). Scroll down to view them.")
        else:
            st.warning("No rows recognized in the CSV. Please check headers.")

# Main ─ Search & View
st.subheader("🔎 Search & View")
with st.expander("Filters", expanded=True):
    f1, f2, f3, f4, f5 = st.columns([1, 1, 1, 1, 2])
    model_q = f1.text_input("Model contains", key="q_model", label_visibility="visible")
    version_q = f2.text_input("Version contains", key="q_version")
    sn_q = f3.text_input("SN contains", key="q_sn")
    mo_q = f4.text_input("MO contains", key="q_mo")
    text_q = f5.text_input("Text in description/type/reporter/category", key="q_text")

    d1, d2, d3 = st.columns([1, 1, 1])
    dt_from = d1.date_input("From date", value=None, format="YYYY-MM-DD", key="q_from")
    dt_to   = d2.date_input("To date", value=None, format="YYYY-MM-DD", key="q_to")
    limit   = d3.number_input("Max rows", min_value=50, max_value=10000, step=50, value=1000)

# Convert dates to string or None
dt_from_str = dt_from.isoformat() if dt_from else None
dt_to_str   = dt_to.isoformat()   if dt_to else None

df = load_nc_df(
    model=model_q, version=version_q, sn=sn_q, mo=mo_q, text=text_q,
    dt_from=dt_from_str, dt_to=dt_to_str, limit=int(limit)
)

# Results (cards)
st.markdown(f"**{len(df)} record(s)**")
if df.empty:
    st.info("No matching records.")
else:
    for _, r in df.iterrows():
        with st.container(border=True):
            top = st.columns([4, 1])
            with top[0]:
                st.markdown(
                    f"**Model:** {r['model_no'] or '-'} &nbsp; | &nbsp; "
                    f"**Version:** {r['model_version'] or '-'} &nbsp; | &nbsp; "
                    f"**SN:** {r['sn'] or '-'} &nbsp; | &nbsp; "
                    f"**MO:** {r['mo'] or '-'}"
                )
                st.caption(
                    f"🗓 {r['created_at'] or '-'}  &nbsp; | &nbsp; "
                    f"👤 Reporter: {r['reporter'] or '-'}  &nbsp; | &nbsp; "
                    f"🏷 Type: {r['nc_type'] or '-'}  &nbsp; | &nbsp; "
                    f"📦 Category: {r['category'] or '-'}"
                )
                if r.get("description"):
                    st.write(r["description"])
                meta = []
                if r.get("line"): meta.append(f"Line: {r['line']}")
                if r.get("station"): meta.append(f"Station: {r['station']}")
                if r.get("customer"): meta.append(f"Customer/Supplier: {r['customer']}")
                if r.get("week"): meta.append(f"Week: {r['week']}")
                if r.get("month"): meta.append(f"Month: {r['month']}")
                if meta:
                    st.caption(" · ".join(meta))

            with top[1]:
                if st.button("Delete", key=f"del_{r['id']}"):
                    delete_nc_row(int(r["id"]))
                    load_nc_df.clear()
                    st.rerun()

    # Table + Export
    with st.expander("Table view + Export"):
        tdf = df.copy()
        st.dataframe(tdf, use_container_width=True, hide_index=True)
        # Export CSV
        csv_bytes = tdf.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download filtered CSV",
            data=csv_bytes,
            file_name=f"non_conformities_filtered_{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}.csv",
            mime="text/csv",
        )

# Footer
st.caption(f"Data directory: `{DATA_DIR}` · DB: `{DB_PATH.name}` · User: {cur_user()}")
