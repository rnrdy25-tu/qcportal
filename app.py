# Quality Portal - Pilot
# End-to-end Streamlit app with Models, First Piece, Non-Conformities,
# Search & View, CSV import, and Export (robust against nested-block errors)

from __future__ import annotations

import os
import io
import json
import sqlite3
from pathlib import Path
from datetime import datetime, date
from typing import Iterable, Dict, Any, Tuple, List

import streamlit as st
import pandas as pd
from PIL import Image

# =============================================================================
# ------------------------------ SETTINGS -------------------------------------
# =============================================================================
APP_NAME = "Quality Portal - Pilot"
DEFAULT_REPORTER = "Admin1"   # Your default login name (can wire SSO later)

# Prefer a writable folder on Streamlit Cloud; fall back locally
def pick_data_dir() -> Path:
    for base in (Path("/mount/data"), Path("/tmp/qc_portal")):
        try:
            base.mkdir(parents=True, exist_ok=True)
            (base / ".w").write_text("ok", encoding="utf-8")
            return base
        except Exception:
            continue
    raise RuntimeError("No writable directory available")

DATA_DIR = pick_data_dir()
IMG_DIR = DATA_DIR / "images"
DB_PATH = DATA_DIR / "qc_portal.sqlite3"
IMG_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# ------------------------------- UTILITIES -----------------------------------
# =============================================================================
def now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds")

def safe_user() -> str:
    # Best effort. Use DEFAULT_REPORTER if available.
    return os.environ.get("USERNAME") or os.environ.get("USER") or DEFAULT_REPORTER

def save_image(rel_subdir: str, uploaded_file) -> str:
    """
    Save uploaded image under images/<rel_subdir>/... -> return path relative to DATA_DIR.
    """
    sub = IMG_DIR / rel_subdir
    sub.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    fname = f"{ts}_{uploaded_file.name.replace(' ', '_')}"
    out_path = sub / fname
    img = Image.open(uploaded_file).convert("RGB")
    img.save(out_path, format="JPEG", quality=90)
    return str(out_path.relative_to(DATA_DIR))

def try_parse_date(s: str | None) -> date | None:
    if not s or pd.isna(s):
        return None
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(str(s), fmt).date()
        except Exception:
            continue
    try:
        return pd.to_datetime(s).date()
    except Exception:
        return None

# =============================================================================
# ------------------------------- DATABASE ------------------------------------
# =============================================================================
SCHEMA_MODELS = """
CREATE TABLE IF NOT EXISTS models(
  model_no TEXT PRIMARY KEY,
  name     TEXT
);
"""

SCHEMA_FIRSTPIECE = """
CREATE TABLE IF NOT EXISTS first_piece(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at TEXT,
  model_no TEXT,
  model_version TEXT,
  sn TEXT,
  mo TEXT,
  reporter TEXT,
  description TEXT,
  top_image TEXT,
  bottom_image TEXT,
  extra JSON
);
"""

SCHEMA_NONCONF = """
CREATE TABLE IF NOT EXISTS nonconformities(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at TEXT,           -- occurrence date
  model_no TEXT,
  model_version TEXT,
  sn TEXT,
  mo TEXT,
  reporter TEXT,
  nonconformity TEXT,        -- title / category
  description TEXT,          -- details
  type TEXT,                 -- Minor/Major/Critical etc.
  image_path TEXT,           -- optional representative
  images JSON,               -- list of extra photos
  raw JSON                   -- full row JSON for anything else
);
"""

def get_conn():
    return sqlite3.connect(DB_PATH)

def init_db():
    with get_conn() as c:
        c.execute(SCHEMA_MODELS)
        c.execute(SCHEMA_FIRSTPIECE)
        c.execute(SCHEMA_NONCONF)
        c.commit()

init_db()

# =============================================================================
# ----------------------------- CACHED QUERIES --------------------------------
# =============================================================================
@st.cache_data(show_spinner=False)
def list_models_df() -> pd.DataFrame:
    with get_conn() as c:
        return pd.read_sql_query(
            "SELECT model_no, COALESCE(name,'') AS name FROM models ORDER BY model_no", c
        )

@st.cache_data(show_spinner=False)
def query_firstpiece(
    model_kw: str = "",
    version_kw: str = "",
    sn_kw: str = "",
    mo_kw: str = "",
    text_kw: str = "",
    date_from: date | None = None,
    date_to: date | None = None,
) -> pd.DataFrame:
    q = """SELECT * FROM first_piece WHERE 1=1"""
    params: List[Any] = []

    def like_col(col: str, kw: str):
        nonlocal q, params
        if kw.strip():
            q += f" AND {col} LIKE ?"
            params.append(f"%{kw.strip()}%")

    like_col("model_no", model_kw)
    like_col("model_version", version_kw)
    like_col("sn", sn_kw)
    like_col("mo", mo_kw)
    if text_kw.strip():
        q += " AND (description LIKE ? OR reporter LIKE ?)"
        params.extend([f"%{text_kw.strip()}%", f"%{text_kw.strip()}%"])

    if date_from:
        q += " AND date(substr(created_at,1,10)) >= date(?)"
        params.append(date_from.isoformat())
    if date_to:
        q += " AND date(substr(created_at,1,10)) <= date(?)"
        params.append(date_to.isoformat())

    q += " ORDER BY id DESC"
    with get_conn() as c:
        return pd.read_sql_query(q, c, params=params)

@st.cache_data(show_spinner=False)
def query_nonconf(
    model_kw: str = "",
    version_kw: str = "",
    sn_kw: str = "",
    mo_kw: str = "",
    text_kw: str = "",
    date_from: date | None = None,
    date_to: date | None = None,
) -> pd.DataFrame:
    q = """SELECT * FROM nonconformities WHERE 1=1"""
    params: List[Any] = []

    def like_col(col: str, kw: str):
        nonlocal q, params
        if kw.strip():
            q += f" AND {col} LIKE ?"
            params.append(f"%{kw.strip()}%")

    like_col("model_no", model_kw)
    like_col("model_version", version_kw)
    like_col("sn", sn_kw)
    like_col("mo", mo_kw)
    if text_kw.strip():
        q += " AND (nonconformity LIKE ? OR description LIKE ? OR reporter LIKE ? OR type LIKE ?)"
        params.extend([f"%{text_kw.strip()}%"] * 4)

    if date_from:
        q += " AND date(substr(created_at,1,10)) >= date(?)"
        params.append(date_from.isoformat())
    if date_to:
        q += " AND date(substr(created_at,1,10)) <= date(?)"
        params.append(date_to.isoformat())

    q += " ORDER BY id DESC"
    with get_conn() as c:
        return pd.read_sql_query(q, c, params=params)

def clear_caches():
    list_models_df.clear()
    query_firstpiece.clear()
    query_nonconf.clear()

# =============================================================================
# ------------------------------ DB WRITERS -----------------------------------
# =============================================================================
def upsert_model(model_no: str, name: str = ""):
    with get_conn() as c:
        c.execute(
            """INSERT INTO models(model_no, name) VALUES(?,?)
               ON CONFLICT(model_no) DO UPDATE SET name=excluded.name""",
            (model_no.strip(), name.strip()),
        )
        c.commit()
    list_models_df.clear()

def insert_firstpiece(payload: Dict[str, Any]):
    cols = [
        "created_at","model_no","model_version","sn","mo",
        "reporter","description","top_image","bottom_image","extra"
    ]
    vals = [payload.get(k) for k in cols]
    with get_conn() as c:
        c.execute(
            f"INSERT INTO first_piece({','.join(cols)}) VALUES({','.join(['?']*len(cols))})",
            vals,
        )
        c.commit()
    query_firstpiece.clear()

def insert_nonconf(payload: Dict[str, Any]):
    cols = [
        "created_at","model_no","model_version","sn","mo","reporter",
        "nonconformity","description","type","image_path","images","raw"
    ]
    vals = [payload.get(k) for k in cols]
    with get_conn() as c:
        c.execute(
            f"INSERT INTO nonconformities({','.join(cols)}) VALUES({','.join(['?']*len(cols))})",
            vals,
        )
        c.commit()
    query_nonconf.clear()

def append_nonconf_image(nid: int, rel_path: str):
    with get_conn() as c:
        row = c.execute("SELECT images FROM nonconformities WHERE id=?", (nid,)).fetchone()
        images = []
        if row and row[0]:
            try:
                images = json.loads(row[0])
            except Exception:
                images = []
        images.append(rel_path)
        c.execute("UPDATE nonconformities SET images=? WHERE id=?", (json.dumps(images), nid))
        c.commit()
    query_nonconf.clear()

def delete_firstpiece(fid: int):
    with get_conn() as c:
        c.execute("DELETE FROM first_piece WHERE id=?", (fid,))
        c.commit()
    query_firstpiece.clear()

def delete_nonconf(nid: int):
    with get_conn() as c:
        c.execute("DELETE FROM nonconformities WHERE id=?", (nid,))
        c.commit()
    query_nonconf.clear()

# =============================================================================
# ------------------------------ CSV MAPPING ----------------------------------
# =============================================================================
# Flexible column aliases for import
ALIASES = {
    "date": ["date", "週數", "月份", "發生日期", "回覆日期", "Date", "時間"],
    "model_no": ["Model", "機種/料號", "model_no", "機種", "料號"],
    "model_version": ["Model Version", "型號版本", "版本", "型號", "Model Version (full)"],
    "sn": ["SN", "序號", "Barcode", "條碼"],
    "mo": ["MO", "MO/PO", "PO", "工單/採購單號", "工單", "MO/Working Order"],
    "nonconformity": ["不符合分類", "Nonconformity", "異常分類", "Defective Item", "異常項目"],
    "description": ["不符合說明", "Description of Nonconformity", "描述", "說明", "description"],
    "type": ["Category", "Severity", "類別", "嚴重度", "類型"],
    "reporter": ["提報人員", "Exception reporters", "Reporter", "檢查員", "檢驗人員"],
}

def resolve(df: pd.DataFrame, key: str) -> str | None:
    for cand in ALIASES.get(key, []):
        if cand in df.columns:
            return cand
    return None

def import_nonconf_csv(file) -> Tuple[int, List[str]]:
    df = pd.read_csv(file) if file.name.lower().endswith(".csv") else pd.read_excel(file)
    df.columns = [str(c).strip() for c in df.columns]

    col = {k: resolve(df, k) for k in ALIASES.keys()}
    msgs = []

    count = 0
    for _, r in df.iterrows():
        created = try_parse_date(r.get(col["date"])) or date.today()
        payload = {
            "created_at": created.isoformat(),
            "model_no": str(r.get(col["model_no"])) if col["model_no"] else "",
            "model_version": str(r.get(col["model_version"])) if col["model_version"] else "",
            "sn": str(r.get(col["sn"])) if col["sn"] else "",
            "mo": str(r.get(col["mo"])) if col["mo"] else "",
            "reporter": str(r.get(col["reporter"])) if col["reporter"] else safe_user(),
            "nonconformity": str(r.get(col["nonconformity"])) if col["nonconformity"] else "",
            "description": str(r.get(col["description"])) if col["description"] else "",
            "type": str(r.get(col["type"])) if col["type"] else "",
            "image_path": "",
            "images": json.dumps([]),
            "raw": r.to_json(force_ascii=False),
        }
        try:
            insert_nonconf(payload)
            count += 1
        except Exception as e:
            msgs.append(f"Row import failed: {e}")
    return count, msgs

# =============================================================================
# --------------------------------- UI ----------------------------------------
# =============================================================================
st.set_page_config(page_title=APP_NAME, layout="wide")
st.markdown(
    """
    <style>
      /* Smaller filter inputs & labels */
      .small * {font-size: 0.92rem !important;}
      .muted { color: #6b7280; font-size: 0.92rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title(APP_NAME)

# ----------------------------- SIDEBAR ---------------------------------------
with st.sidebar:
    st.header("Admin / Data Entry")

    # -- Add / Update Model --
    with st.expander("Add / Update Model", expanded=False):
        with st.form("f_model", clear_on_submit=True):
            m_no = st.text_input("Model short (e.g., 190-56980)")
            m_name = st.text_input("Name / Customer (optional)")
            ok = st.form_submit_button("Save model")
            if ok:
                if m_no.strip():
                    upsert_model(m_no, m_name)
                    st.success("Model saved.")
                else:
                    st.error("Model cannot be empty.")

    # -- First Piece entry --
    with st.expander("Create First Piece", expanded=False):
        with st.form("f_fp", clear_on_submit=True):
            fp_model = st.text_input("Model")
            fp_version = st.text_input("Model Version (full)")
            fp_sn = st.text_input("SN / Barcode")
            fp_mo = st.text_input("MO / Work Order")
            fp_desc = st.text_area("Description / Notes")
            colu = st.columns(2)
            with colu[0]:
                up_top = st.file_uploader("TOP photo", type=["jpg","jpeg","png"], key="up_top")
            with colu[1]:
                up_bot = st.file_uploader("BOTTOM photo", type=["jpg","jpeg","png"], key="up_bot")
            ok2 = st.form_submit_button("Save First Piece")
            if ok2:
                if not fp_model.strip():
                    st.error("Model is required.")
                else:
                    top_rel = save_image(fp_model.strip(), up_top) if up_top else ""
                    bot_rel = save_image(fp_model.strip(), up_bot) if up_bot else ""
                    insert_firstpiece({
                        "created_at": now_iso(),
                        "model_no": fp_model.strip(),
                        "model_version": fp_version.strip(),
                        "sn": fp_sn.strip(),
                        "mo": fp_mo.strip(),
                        "reporter": safe_user(),
                        "description": fp_desc.strip(),
                        "top_image": top_rel,
                        "bottom_image": bot_rel,
                        "extra": json.dumps({}),
                    })
                    st.success("First Piece saved.")

    # -- Non-Conformity entry (manual) --
    with st.expander("Create Non-Conformity", expanded=False):
        with st.form("f_nc", clear_on_submit=True):
            nc_date = st.date_input("Date", value=date.today())
            model = st.text_input("Model")
            version = st.text_input("Model Version")
            sn = st.text_input("SN")
            mo = st.text_input("MO")
            nctype = st.selectbox("Category / Severity", ["", "Minor", "Major", "Critical"])
            title = st.text_input("Nonconformity")
            desc = st.text_area("Description")
            up_nc = st.file_uploader("Photo (optional)", type=["jpg","jpeg","png"])
            ok3 = st.form_submit_button("Save Non-Conformity")
            if ok3:
                image_rel = save_image(model.strip() or "nonconformity", up_nc) if up_nc else ""
                insert_nonconf({
                    "created_at": nc_date.isoformat(),
                    "model_no": model.strip(),
                    "model_version": version.strip(),
                    "sn": sn.strip(),
                    "mo": mo.strip(),
                    "reporter": safe_user(),
                    "nonconformity": title.strip(),
                    "description": desc.strip(),
                    "type": nctype.strip(),
                    "image_path": image_rel,
                    "images": json.dumps([image_rel] if image_rel else []),
                    "raw": json.dumps({}),
                })
                st.success("Non-Conformity saved.")

    # -- Non-Conformity import CSV/Excel --
    with st.expander("Import Non-Conformities (CSV/Excel)", expanded=False):
        with st.form("f_nc_import", clear_on_submit=True):
            up_csv = st.file_uploader("Upload CSV or XLSX", type=["csv", "xlsx"])
            ok4 = st.form_submit_button("Import")
            if ok4:
                if up_csv is None:
                    st.error("Please choose a file to import.")
                else:
                    try:
                        n, msgs = import_nonconf_csv(up_csv)
                        st.success(f"Imported {n} rows.")
                        if msgs:
                            with st.expander("Import warnings"):
                                for m in msgs:
                                    st.write("•", m)
                    except Exception as e:
                        st.error(f"Import failed: {e}")

# ----------------------------- SEARCH & VIEW ---------------------------------
st.subheader("🔎 Search & View", divider="gray")

with st.container():
    st.markdown('<div class="small">', unsafe_allow_html=True)
    c1, c2, c3, c4, c5, c6 = st.columns([1.2, 1.2, 1, 1, 1.5, 1.5])
    with c1:
        f_model = st.text_input("Model contains", "")
    with c2:
        f_version = st.text_input("Version contains", "")
    with c3:
        f_sn = st.text_input("SN contains", "")
    with c4:
        f_mo = st.text_input("MO contains", "")
    with c5:
        f_text = st.text_input("Text in description/reporter/type", "")
    with c6:
        dcol1, dcol2 = st.columns(2)
        with dcol1:
            f_from = st.date_input("From", value=None, format="YYYY-MM-DD")
        with dcol2:
            f_to = st.date_input("To", value=None, format="YYYY-MM-DD")
    st.markdown("</div>", unsafe_allow_html=True)

# ---------- First Piece Results ----------
fp_df = query_firstpiece(f_model, f_version, f_sn, f_mo, f_text, f_from, f_to)
with st.expander(f"📁 First Piece (results) – {len(fp_df)} record(s)", expanded=True):
    if fp_df.empty:
        st.caption("No first-piece records.")
    else:
        for _, r in fp_df.iterrows():
            with st.container(border=True):
                top_path = DATA_DIR / str(r["top_image"]) if r.get("top_image") else None
                bot_path = DATA_DIR / str(r["bottom_image"]) if r.get("bottom_image") else None
                cimg = st.columns(2)
                with cimg[0]:
                    if top_path and top_path.exists():
                        st.image(str(top_path), use_container_width=True, caption="TOP")
                with cimg[1]:
                    if bot_path and bot_path.exists():
                        st.image(str(bot_path), use_container_width=True, caption="BOTTOM")

                st.markdown(
                    f"**Model:** {r['model_no'] or '-'} | "
                    f"**Version:** {r['model_version'] or '-'} | "
                    f"**SN:** {r['sn'] or '-'} | **MO:** {r['mo'] or '-'}"
                )
                st.caption(f"🕒 {r['created_at']}  ·  👤 Reporter: {r['reporter'] or safe_user()}")
                st.write(r["description"] or "*No description*")
                if st.button("Delete", key=f"del_fp_{r['id']}"):
                    delete_firstpiece(int(r["id"]))
                    st.rerun()

# Toggle for table & export (avoid nested-block/expander problems)
st.subheader("First Piece – Table & Export", divider="gray")
if st.toggle("Show table & export", key="fp_tbl_tog"):
    st.dataframe(fp_df, use_container_width=True, hide_index=True)
    if not fp_df.empty:
        csv = fp_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button("Download CSV", data=csv, file_name="first_piece_export.csv", mime="text/csv")

# ---------- Non-Conformities Results ----------
nc_df = query_nonconf(f_model, f_version, f_sn, f_mo, f_text, f_from, f_to)
with st.expander(f"🧭 Non-Conformities (results) – {len(nc_df)} record(s)", expanded=True):
    if nc_df.empty:
        st.caption("No non-conformities.")
    else:
        for _, r in nc_df.iterrows():
            with st.container(border=True):
                row = dict(r)

                # Head line
                st.markdown(
                    f"**Model:** {row.get('model_no') or '-'} | "
                    f"**Version:** {row.get('model_version') or '-'} | "
                    f"**SN:** {row.get('sn') or '-'} | **MO:** {row.get('mo') or '-'}"
                )

                # info line
                imgs = []
                try:
                    imgs = json.loads(row.get("images") or "[]")
                except Exception:
                    imgs = []

                ic = st.columns([1.2, 6, 1.2, 1.2])
                with ic[0]:
                    st.caption(f"🕒 {row.get('created_at') or ''}")
                with ic[1]:
                    st.caption(f"👤 {row.get('reporter') or safe_user()}  |  🏷 {row.get('type') or ''}")
                with ic[2]:
                    # add photo button/uploader
                    addf = st.file_uploader("Add photo", type=["jpg","jpeg","png"], key=f"addp_{r['id']}")
                    if addf:
                        rel = save_image(row.get("model_no") or "nonconformity", addf)
                        append_nonconf_image(int(row["id"]), rel)
                        st.success("Photo added.")
                        st.rerun()
                with ic[3]:
                    if st.button("Delete", key=f"del_nc_{r['id']}"):
                        delete_nonconf(int(r["id"]))
                        st.rerun()

                # images row
                if row.get("image_path"):
                    p0 = DATA_DIR / str(row["image_path"])
                    if p0.exists():
                        st.image(str(p0), use_container_width=True)

                if imgs:
                    gcols = st.columns(min(4, len(imgs)))
                    for i, rel in enumerate(imgs[:4]):
                        p = DATA_DIR / rel
                        with gcols[i % len(gcols)]:
                            if p.exists():
                                st.image(str(p), use_container_width=True)

                # text area
                st.markdown(f"**{row.get('nonconformity') or ''}**")
                st.write(row.get("description") or "*No description*")

# Toggle for table & export (avoid nested block)
st.subheader("Non-Conformities – Table & Export", divider="gray")
if st.toggle("Show table & export", key="nc_tbl_tog"):
    st.dataframe(nc_df, use_container_width=True, hide_index=True)
    if not nc_df.empty:
        csv = nc_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button("Download CSV", data=csv, file_name="nonconformities_export.csv", mime="text/csv")
