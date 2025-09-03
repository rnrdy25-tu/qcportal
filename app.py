# app.py — Quality Portal - Pilot
# First Piece (manual upload) + Non-Conformities (manual + CSV) + Search/View/Export

import os
import io
import json
import sqlite3
from pathlib import Path
from datetime import datetime, date

import pandas as pd
import streamlit as st
from PIL import Image

# ──────────────────────────────────────────────────────────────────────────────
# Storage
# ──────────────────────────────────────────────────────────────────────────────
def pick_data_dir() -> Path:
    for base in (Path("/mount/data"), Path("/tmp/quality_portal")):
        try:
            base.mkdir(parents=True, exist_ok=True)
            (base / ".ok").write_text("ok", encoding="utf-8")
            return base
        except Exception:
            pass
    raise RuntimeError("No writable directory available")

DATA_DIR = pick_data_dir()
IMG_DIR  = DATA_DIR / "images"
IMG_DIR.mkdir(exist_ok=True, parents=True)
DB_PATH  = DATA_DIR / "quality_portal.sqlite3"

def cur_user() -> str:
    return os.environ.get("USERNAME") or os.environ.get("USER") or "Admin1"

def today_iso() -> str:
    return date.today().isoformat()

# Save uploaded image to images/<sub>/<timestamp>_<orig>
def save_image(file, sub: str) -> str:
    subdir = IMG_DIR / sub
    subdir.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    safe = file.name.replace(" ", "_")
    out = subdir / f"{ts}_{safe}"
    # re-encode as JPEG to be consistent
    img = Image.open(file).convert("RGB")
    img.save(out, format="JPEG", quality=90)
    return str(out.relative_to(DATA_DIR))

# ──────────────────────────────────────────────────────────────────────────────
# DB schema + migrations
# ──────────────────────────────────────────────────────────────────────────────
TABLE_FP = "first_piece"
TABLE_NC = "non_conformities"

SCHEMA_FP = f"""
CREATE TABLE IF NOT EXISTS {TABLE_FP}(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at   TEXT,
  model_no     TEXT,
  model_version TEXT,
  sn           TEXT,
  mo           TEXT,
  description  TEXT,
  reporter     TEXT,
  top_path     TEXT,
  bottom_path  TEXT
);
"""

SCHEMA_NC = f"""
CREATE TABLE IF NOT EXISTS {TABLE_NC}(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at   TEXT,
  model_no     TEXT,
  model_version TEXT,
  sn           TEXT,
  mo           TEXT,
  nc_type      TEXT,
  description  TEXT,
  category     TEXT,
  line         TEXT,
  station      TEXT,
  reporter     TEXT,
  month        TEXT,
  week         TEXT,
  customer     TEXT,
  extra        JSON
);
"""

def get_conn():
    return sqlite3.connect(DB_PATH)

def _cols(c, table: str) -> set:
    return {r[1] for r in c.execute(f"PRAGMA table_info({table})").fetchall()}

def _ensure(c, table, col, decl, default=None):
    if col not in _cols(c, table):
        c.execute(f"ALTER TABLE {table} ADD COLUMN {col} {decl}")
        if default is not None:
            c.execute(f"UPDATE {table} SET {col}=?", (default,))

def init_db():
    with get_conn() as c:
        c.execute(SCHEMA_FP)
        c.execute(SCHEMA_NC)
        # simple migrations (add missing columns safely)
        for tbl, cols in {
            TABLE_FP: {
                "created_at":"TEXT","model_no":"TEXT","model_version":"TEXT","sn":"TEXT",
                "mo":"TEXT","description":"TEXT","reporter":"TEXT","top_path":"TEXT","bottom_path":"TEXT"
            },
            TABLE_NC: {
                "created_at":"TEXT","model_no":"TEXT","model_version":"TEXT","sn":"TEXT","mo":"TEXT",
                "nc_type":"TEXT","description":"TEXT","category":"TEXT","line":"TEXT","station":"TEXT",
                "reporter":"TEXT","month":"TEXT","week":"TEXT","customer":"TEXT","extra":"JSON"
            }
        }.items():
            for col, decl in cols.items():
                _ensure(c, tbl, col, decl)
        c.commit()

init_db()

# ──────────────────────────────────────────────────────────────────────────────
# CSV header mapping for NC import
# ──────────────────────────────────────────────────────────────────────────────
HEADER_MAP = {
    "date": "created_at", "application date": "created_at", "發出日期": "created_at",
    "created_at": "created_at",

    "model": "model_no", "機種/料號": "model_no", "model/part no.": "model_no", "model no": "model_no",

    "model version": "model_version", "version": "model_version",

    "sn": "sn", "barcode": "sn",
    "mo": "mo", "mo/po": "mo", "工作單/採購單號": "mo",

    "nonconformity": "nc_type", "不符合分類": "nc_type", "defective item": "nc_type", "異常分類": "nc_type",

    "description": "description", "不符合說明": "description",
    "category": "category", "severity": "category",

    "line": "line", "線別": "line",
    "work station": "station", "作業站別": "station",

    "reporter": "reporter", "提報人員": "reporter",
    "customer": "customer", "客戶/供應商": "customer",
    "month": "month", "月份": "month",
    "week": "week", "週數": "week",
}

def normalize_headers(df: pd.DataFrame) -> pd.DataFrame:
    ren = {}
    for col in df.columns:
        key = str(col).strip().lower()
        ren[col] = HEADER_MAP.get(key, key)
    return df.rename(columns=ren)

def coerce_date(v):
    if pd.isna(v):
        return None
    if isinstance(v, (datetime, date)):
        return pd.to_datetime(v).date().isoformat()
    try:
        d = pd.to_datetime(v, errors="coerce")
        if pd.isna(d):
            return None
        return d.date().isoformat()
    except Exception:
        return None

# ──────────────────────────────────────────────────────────────────────────────
# Inserts / loads
# ──────────────────────────────────────────────────────────────────────────────
def insert_first_piece(payload: dict):
    fields = ["created_at","model_no","model_version","sn","mo","description","reporter","top_path","bottom_path"]
    vals   = [payload.get(k) for k in fields]
    with get_conn() as c:
        c.execute(f"INSERT INTO {TABLE_FP}({','.join(fields)}) VALUES({','.join(['?']*len(fields))})", vals)
        c.commit()

@st.cache_data(show_spinner=False)
def load_firstpiece_df(model=None, version=None, sn=None, mo=None, text=None, dt_from=None, dt_to=None, limit=1000):
    q = f"""SELECT id, created_at, model_no, model_version, sn, mo, description, reporter, top_path, bottom_path
            FROM {TABLE_FP}
            WHERE 1=1"""
    params = []
    if model:   q += " AND model_no LIKE ?";        params.append(f"%{model}%")
    if version: q += " AND model_version LIKE ?";   params.append(f"%{version}%")
    if sn:      q += " AND sn LIKE ?";              params.append(f"%{sn}%")
    if mo:      q += " AND mo LIKE ?";              params.append(f"%{mo}%")
    if text:    q += " AND (description LIKE ? OR reporter LIKE ?)"; params.extend([f"%{text}%", f"%{text}%"])
    if dt_from: q += " AND created_at >= ?";        params.append(dt_from)
    if dt_to:   q += " AND created_at <= ?";        params.append(dt_to)
    q += " ORDER BY created_at DESC, id DESC LIMIT ?"; params.append(limit)
    with get_conn() as c:
        return pd.read_sql_query(q, c, params=params)

def delete_first_piece(row_id: int):
    with get_conn() as c:
        c.execute(f"DELETE FROM {TABLE_FP} WHERE id=?", (row_id,))
        c.commit()

def insert_nc_rows(rows: list[dict]) -> int:
    fields = ["created_at","model_no","model_version","sn","mo","nc_type","description","category",
              "line","station","reporter","month","week","customer","extra"]
    values = [[r.get(k) for k in fields] for r in rows]
    with get_conn() as c:
        c.executemany(f"INSERT INTO {TABLE_NC}({','.join(fields)}) VALUES({','.join(['?']*len(fields))})", values)
        c.commit()
    return len(values)

@st.cache_data(show_spinner=False)
def load_nc_df(model=None, version=None, sn=None, mo=None, text=None, dt_from=None, dt_to=None, limit=2000):
    q = f"""SELECT id, created_at, model_no, model_version, sn, mo, nc_type, description, category,
                   line, station, reporter, month, week, customer
            FROM {TABLE_NC}
            WHERE 1=1"""
    params = []
    if model:   q += " AND model_no LIKE ?";        params.append(f"%{model}%")
    if version: q += " AND model_version LIKE ?";   params.append(f"%{version}%")
    if sn:      q += " AND sn LIKE ?";              params.append(f"%{sn}%")
    if mo:      q += " AND mo LIKE ?";              params.append(f"%{mo}%")
    if text:    q += " AND (description LIKE ? OR nc_type LIKE ? OR reporter LIKE ? OR category LIKE ?)"
    if text:        params.extend([f"%{text}%"]*4)
    if dt_from: q += " AND created_at >= ?";        params.append(dt_from)
    if dt_to:   q += " AND created_at <= ?";        params.append(dt_to)
    q += " ORDER BY created_at DESC, id DESC LIMIT ?"; params.append(limit)
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

# ─ Sidebar: Create / Upload ─
with st.sidebar:
    st.header("🧾 Create / Upload")

    # First Piece form
    with st.expander("📷 First Piece — New", expanded=False):
        fp_date = st.date_input("Date", value=date.today(), format="YYYY-MM-DD")
        fp_model = st.text_input("Model", key="fp_model")
        fp_ver   = st.text_input("Version", key="fp_ver")
        fp_sn    = st.text_input("SN", key="fp_sn")
        fp_mo    = st.text_input("MO / Work Order", key="fp_mo")
        fp_desc  = st.text_area("Notes / Description", key="fp_desc")
        col_up1, col_up2 = st.columns(2)
        with col_up1:
            up_top = st.file_uploader("TOP photo", type=["jpg","jpeg","png"], key="fp_top")
        with col_up2:
            up_bot = st.file_uploader("BOTTOM photo", type=["jpg","jpeg","png"], key="fp_bottom")
        if st.button("Save First Piece", type="primary", key="save_fp"):
            if not fp_model.strip():
                st.error("Model is required.")
            elif not (up_top or up_bot):
                st.error("Please upload at least one photo (TOP or BOTTOM).")
            else:
                top_rel = save_image(up_top, "firstpiece/top") if up_top else None
                bot_rel = save_image(up_bot, "firstpiece/bottom") if up_bot else None
                insert_first_piece({
                    "created_at": fp_date.isoformat(),
                    "model_no": fp_model.strip(),
                    "model_version": fp_ver.strip(),
                    "sn": fp_sn.strip(),
                    "mo": fp_mo.strip(),
                    "description": fp_desc.strip(),
                    "reporter": cur_user(),
                    "top_path": top_rel,
                    "bottom_path": bot_rel,
                })
                load_firstpiece_df.clear()
                st.success("First Piece saved.")

    # Non-Conformity manual form
    with st.expander("📝 Non-Conformity — New", expanded=False):
        nc_date = st.date_input("Date", value=date.today(), format="YYYY-MM-DD", key="nc_date")
        nc_model = st.text_input("Model", key="nc_model")
        nc_ver   = st.text_input("Version", key="nc_ver")
        nc_sn    = st.text_input("SN", key="nc_sn")
        nc_mo    = st.text_input("MO", key="nc_mo")
        nc_type  = st.text_input("Nonconformity Type", key="nc_type")
        nc_desc  = st.text_area("Description", key="nc_desc")
        nc_cat   = st.selectbox("Category", ["", "Minor", "Major", "Critical"], key="nc_cat")
        nc_line  = st.text_input("Line", key="nc_line")
        nc_sta   = st.text_input("Station", key="nc_sta")
        nc_cust  = st.text_input("Customer/Supplier", key="nc_cust")
        nc_week  = st.text_input("Week", key="nc_week")
        nc_month = st.text_input("Month", key="nc_month")
        if st.button("Save Non-Conformity", type="primary", key="nc_save"):
            if not nc_model.strip():
                st.error("Model is required.")
            else:
                row = {
                    "created_at": nc_date.isoformat(),
                    "model_no": nc_model.strip(),
                    "model_version": nc_ver.strip(),
                    "sn": nc_sn.strip(),
                    "mo": nc_mo.strip(),
                    "nc_type": nc_type.strip(),
                    "description": nc_desc.strip(),
                    "category": nc_cat.strip(),
                    "line": nc_line.strip(),
                    "station": nc_sta.strip(),
                    "reporter": cur_user(),
                    "month": nc_month.strip(),
                    "week": nc_week.strip(),
                    "customer": nc_cust.strip(),
                    "extra": None,
                }
                insert_nc_rows([row])
                load_nc_df.clear()
                st.success("Non-Conformity saved.")

    # NC CSV import
    with st.expander("📤 Import Non-Conformities (CSV)", expanded=False):
        up = st.file_uploader("CSV file", type=["csv"], key="nc_csv")
        if up:
            try:
                raw = up.read()
                df = pd.read_csv(io.BytesIO(raw), encoding="utf-8", keep_default_na=False)
            except UnicodeDecodeError:
                up.seek(0)
                df = pd.read_csv(up, encoding="latin-1", keep_default_na=False)
            ndf = normalize_headers(df)
            rows = []
            for _, r in ndf.iterrows():
                known = {
                    "created_at": coerce_date(r.get("created_at")) or today_iso(),
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
                extras = {k: r[k] for k in ndf.columns if k not in known}
                known["extra"] = json.dumps(extras, ensure_ascii=False) if extras else None
                rows.append(known)
            if rows:
                inserted = insert_nc_rows(rows)
                load_nc_df.clear()
                st.success(f"Imported {inserted} record(s).")

# ─ Search & View (Tabs) ─
st.subheader("🔎 Search & View")
tab_nc, tab_fp = st.tabs(["Non-Conformities", "First Piece"])

# Common filter controls helper
def filter_controls(prefix: str):
    c1, c2, c3, c4, c5 = st.columns([1,1,1,1,2])
    model = c1.text_input("Model contains", key=f"{prefix}_m")
    ver   = c2.text_input("Version contains", key=f"{prefix}_v")
    sn    = c3.text_input("SN contains", key=f"{prefix}_s")
    mo    = c4.text_input("MO contains", key=f"{prefix}_o")
    text  = c5.text_input("Text (desc/type/reporter/category)", key=f"{prefix}_t")
    d1, d2, d3 = st.columns([1,1,1])
    dt_from = d1.date_input("From date", value=None, format="YYYY-MM-DD", key=f"{prefix}_from")
    dt_to   = d2.date_input("To date", value=None, format="YYYY-MM-DD", key=f"{prefix}_to")
    limit   = d3.number_input("Max rows", 50, 10000, 1000, 50, key=f"{prefix}_lim")
    return model, ver, sn, mo, text, (dt_from.isoformat() if dt_from else None), (dt_to.isoformat() if dt_to else None), int(limit)

with tab_nc:
    st.caption("Search Non-Conformities")
    m,v,s,o,t,dfrom,dto,lim = filter_controls("nc")
    ndf = load_nc_df(m, v, s, o, t, dfrom, dto, lim)
    st.markdown(f"**{len(ndf)} record(s)**")
    if ndf.empty:
        st.info("No matching records.")
    else:
        for _, r in ndf.iterrows():
            with st.container(border=True):
                cols = st.columns([4,1])
                with cols[0]:
                    st.markdown(
                        f"**Model:** {r['model_no'] or '-'} | **Version:** {r['model_version'] or '-'} "
                        f"| **SN:** {r['sn'] or '-'} | **MO:** {r['mo'] or '-'}"
                    )
                    st.caption(
                        f"🗓 {r['created_at'] or '-'}  |  👤 {r['reporter'] or '-'}  |  "
                        f"🏷 {r['nc_type'] or '-'}  |  📦 {r['category'] or '-'}"
                    )
                    if r.get("description"):
                        st.write(r["description"])
                    meta=[]
                    for k,label in (("line","Line"),("station","Station"),("customer","Customer"),("week","Week"),("month","Month")):
                        if r.get(k): meta.append(f"{label}: {r[k]}")
                    if meta: st.caption(" · ".join(meta))
                with cols[1]:
                    if st.button("Delete", key=f"del_nc_{r['id']}"):
                        delete_nc_row(int(r["id"]))
                        load_nc_df.clear()
                        st.rerun()
        with st.expander("Table + Export"):
            st.dataframe(ndf, use_container_width=True, hide_index=True)
            st.download_button(
                "Download filtered CSV",
                data=ndf.to_csv(index=False).encode("utf-8"),
                file_name=f"nonconformities_{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}.csv",
                mime="text/csv"
            )

with tab_fp:
    st.caption("Search First Piece")
    m,v,s,o,t,dfrom,dto,lim = filter_controls("fp")
    fdf = load_firstpiece_df(m, v, s, o, t, dfrom, dto, lim)
    st.markdown(f"**{len(fdf)} record(s)**")
    if fdf.empty:
        st.info("No matching records.")
    else:
        for _, r in fdf.iterrows():
            with st.container(border=True):
                top = st.columns([1,1,2])
                # left: TOP
                with top[0]:
                    if r.get("top_path"):
                        p = DATA_DIR / str(r["top_path"])
                        if p.exists():
                            st.image(str(p), caption="TOP", use_container_width=True)
                # middle: BOTTOM
                with top[1]:
                    if r.get("bottom_path"):
                        p = DATA_DIR / str(r["bottom_path"])
                        if p.exists():
                            st.image(str(p), caption="BOTTOM", use_container_width=True)
                # right: meta
                with top[2]:
                    st.markdown(
                        f"**Model:** {r['model_no'] or '-'} | **Version:** {r['model_version'] or '-'} "
                        f"| **SN:** {r['sn'] or '-'} | **MO:** {r['mo'] or '-'}"
                    )
                    st.caption(f"🗓 {r['created_at'] or '-'}  |  👤 Reporter: {r['reporter'] or '-'}")
                    if r.get("description"):
                        st.write(r["description"])
                    if st.button("Delete", key=f"del_fp_{r['id']}"):
                        delete_first_piece(int(r["id"]))
                        load_firstpiece_df.clear()
                        st.rerun()

        with st.expander("Table + Export"):
            st.dataframe(fdf, use_container_width=True, hide_index=True)
            st.download_button(
                "Download filtered CSV",
                data=fdf.to_csv(index=False).encode("utf-8"),
                file_name=f"firstpiece_{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}.csv",
                mime="text/csv"
            )

st.caption(f"Data: `{DATA_DIR}` · DB: `{DB_PATH.name}` · User: {cur_user()}`")
