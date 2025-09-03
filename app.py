# Quality Portal – Pilot
# Models, First Piece (Top/Bottom), Non-Conformities, Search (with dates) & Export

import os
import io
import json
import sqlite3
from pathlib import Path
from datetime import datetime, date, time

import streamlit as st
import pandas as pd
from PIL import Image

# ========================= Storage (Cloud-safe) =========================

def pick_data_dir() -> Path:
    for base in (Path("/mount/data"), Path("/tmp/qc_portal")):
        try:
            base.mkdir(parents=True, exist_ok=True)
            (base / ".write_check").write_text("ok", encoding="utf-8")
            return base
        except Exception:
            pass
    raise RuntimeError("No writable directory found.")

DATA_DIR = pick_data_dir()
IMG_DIR  = DATA_DIR / "images"
DB_PATH  = DATA_DIR / "qc_portal.sqlite3"
IMG_DIR.mkdir(parents=True, exist_ok=True)

# ========================= Helpers =========================

ADMIN_USERS = {"Admin1"}

def now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds")

def current_user() -> str:
    return st.session_state.get("user_name") or "Admin1"

def is_admin() -> bool:
    return current_user() in ADMIN_USERS

def save_image(model_no: str, uploaded_file) -> str:
    """
    Saves an uploaded image under images/<model_no>/ts_filename.jpg.
    Returns path RELATIVE to DATA_DIR.
    """
    folder = IMG_DIR / model_no
    folder.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    safe_name = uploaded_file.name.replace(" ", "_")
    out_path = folder / f"{ts}_{safe_name}"
    img = Image.open(uploaded_file).convert("RGB")
    img.save(out_path, format="JPEG", quality=90)
    return str(out_path.relative_to(DATA_DIR))

def dt_bounds(d_from: date | None, d_to: date | None):
    """Return (iso_from, iso_to) strings or Nones to filter created_at."""
    iso_from = None
    iso_to = None
    if d_from:
        iso_from = datetime.combine(d_from, time.min).isoformat(timespec="seconds")
    if d_to:
        iso_to = datetime.combine(d_to, time.max).isoformat(timespec="seconds")
    return iso_from, iso_to

# ========================= Database =========================

SCHEMA_MODELS = """
CREATE TABLE IF NOT EXISTS models(
  model_no TEXT PRIMARY KEY,
  name     TEXT
);
"""

# first_piece keeps image_path (for thumbnail) and extra JSON
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
  image_path TEXT,               -- thumbnail / first image (we use TOP here if present)
  extra JSON                     -- {"top": "...", "bottom": "..."} or legacy {"images": [...]}
);
"""

SCHEMA_NC = """
CREATE TABLE IF NOT EXISTS nonconformities(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at TEXT,
  model_no TEXT,
  model_version TEXT,
  mo TEXT,
  line TEXT,
  station TEXT,
  reporter TEXT,
  category TEXT,
  nc_type TEXT,
  description TEXT,
  image_path TEXT,
  extra JSON
);
"""

def get_conn():
    return sqlite3.connect(DB_PATH)

def init_db():
    with get_conn() as c:
        c.execute(SCHEMA_MODELS)
        c.execute(SCHEMA_FIRSTPIECE)
        c.execute(SCHEMA_NC)
        c.commit()

# ---------- Cached loaders ----------

@st.cache_data(show_spinner=False)
def list_models_df() -> pd.DataFrame:
    with get_conn() as c:
        return pd.read_sql_query(
            "SELECT model_no, COALESCE(name,'') AS name FROM models ORDER BY model_no", c
        )

@st.cache_data(show_spinner=False)
def load_firstpiece_df(
    model=None, version=None, sn=None, mo=None, text=None,
    d_from=None, d_to=None, limit=500
) -> pd.DataFrame:
    q = """
    SELECT id, created_at, model_no, model_version, sn, mo,
           reporter, description, image_top_path, image_bottom_path
    FROM first_piece
    WHERE 1=1
    """
    params = []
    if model:
        q += " AND model_no LIKE ?"
        params.append(f"%{model}%")
    if version:
        q += " AND model_version LIKE ?"
        params.append(f"%{version}%")
    if sn:
        q += " AND sn LIKE ?"
        params.append(f"%{sn}%")
    if mo:
        q += " AND mo LIKE ?"
        params.append(f"%{mo}%")
    if text:
        q += " AND (description LIKE ? OR reporter LIKE ?)"
        params.extend([f"%{text}%", f"%{text}%"])

    if d_from and d_to:
        q += " AND created_at BETWEEN ? AND ?"
        params.extend([d_from, d_to])
    elif d_from:
        q += " AND created_at >= ?"
        params.append(d_from)
    elif d_to:
        q += " AND created_at <= ?"
        params.append(d_to)

    q += " ORDER BY id DESC LIMIT ?"
    params.append(limit)

    with get_conn() as c:
        return pd.read_sql_query(q, c, params=params)

@st.cache_data(show_spinner=False)
def load_nc_df(
    model=None, version=None, mo=None, text=None, category=None,
    d_from=None, d_to=None, limit=500
) -> pd.DataFrame:
    q = """
    SELECT id, created_at, model_no, model_version, mo, line, station, reporter,
           category, nc_type, description, image_path, extra
    FROM nonconformities
    WHERE 1=1
    """
    params = []
    if model:
        q += " AND model_no LIKE ?"
        params.append(f"%{model}%")
    if version:
        q += " AND model_version LIKE ?"
        params.append(f"%{version}%")
    if mo:
        q += " AND mo LIKE ?"
        params.append(f"%{mo}%")
    if category and category != "All":
        q += " AND category = ?"
        params.append(category)
    if text:
        q += " AND (description LIKE ? OR nc_type LIKE ? OR reporter LIKE ?)"
        params.extend([f"%{text}%", f"%{text}%", f"%{text}%"])

    if d_from and d_to:
        q += " AND created_at BETWEEN ? AND ?"
        params.extend([d_from, d_to])
    elif d_from:
        q += " AND created_at >= ?"
        params.append(d_from)
    elif d_to:
        q += " AND created_at <= ?"
        params.append(d_to)

    q += " ORDER BY id DESC LIMIT ?"
    params.append(limit)

    with get_conn() as c:
        return pd.read_sql_query(q, c, params=params)

# ---------- Mutations ----------

def upsert_model(model_no: str, name: str = ""):
    with get_conn() as c:
        c.execute(
            """
            INSERT INTO models(model_no, name) VALUES(?, ?)
            ON CONFLICT(model_no) DO UPDATE SET name=excluded.name
            """,
            (model_no.strip(), name.strip()),
        )
        c.commit()

def delete_model(model_no: str):
    with get_conn() as c:
        c.execute("DELETE FROM models WHERE model_no=?", (model_no.strip(),))
        c.commit()

def insert_firstpiece(payload: dict):
    fields = [
        "created_at", "model_no", "model_version", "sn", "mo",
        "reporter", "description", "image_path", "extra"
    ]
    values = [payload.get(k) for k in fields]
    with get_conn() as c:
        c.execute(
            f"INSERT INTO first_piece({','.join(fields)}) VALUES(?,?,?,?,?,?,?,?,?)",
            values,
        )
        c.commit()

def delete_firstpiece(record_id: int):
    with get_conn() as c:
        c.execute("DELETE FROM first_piece WHERE id=?", (record_id,))
        c.commit()

def insert_nc(payload: dict):
    fields = [
        "created_at", "model_no", "model_version", "mo",
        "line", "station", "reporter", "category", "nc_type",
        "description", "image_path", "extra"
    ]
    values = [payload.get(k) for k in fields]
    with get_conn() as c:
        c.execute(
            f"INSERT INTO nonconformities({','.join(fields)}) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
            values,
        )
        c.commit()

def delete_nc(record_id: int):
    with get_conn() as c:
        c.execute("DELETE FROM nonconformities WHERE id=?", (record_id,))
        c.commit()

# ========================= App UI =========================

init_db()
st.set_page_config(page_title="Quality Portal - Pilot", layout="wide")

# small CSS to reduce sizes for this page titles/labels/filters
st.markdown(
    """
    <style>
      h1, h2 { font-size: 1.65rem; }
      h3 { font-size: 1.2rem; }
      div[data-testid="stExpander"] div[role="button"] p { font-size: 0.95rem; }
      label, .st-emotion-cache-1wbqy5l { font-size: 0.88rem !important; }
      input, textarea, select { font-size: 0.90rem !important; }
      .small-note { font-size: 0.85rem; color: #6b7280; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🔎 Quality Portal - Pilot")

# ---------- Sidebar: Sign in ----------
with st.sidebar:
    with st.expander("👤 Sign in", expanded=True):
        st.text_input("Your name", key="user_name", value=st.session_state.get("user_name", "Admin1"))
        st.caption("This name is saved as **Reporter**. Admin features are enabled for **Admin1**.")

# ---------- Sidebar: Models ----------
with st.sidebar:
    st.header("📚 Models")
    with st.expander("Add / Update Model", expanded=False):
        m_no   = st.text_input("Model (short form, e.g., 190-56980)", key="mno_admin")
        m_name = st.text_input("Name / Customer (optional)", key="mname_admin")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Save model", use_container_width=True):
                if m_no.strip():
                    upsert_model(m_no, m_name)
                    st.success("Model saved.")
                    list_models_df.clear()
                else:
                    st.error("Model cannot be empty.")
        with c2:
            if st.button("Delete model", use_container_width=True, type="secondary"):
                if not is_admin():
                    st.error("Only admins can delete models.")
                elif not m_no.strip():
                    st.error("Specify a model to delete.")
                else:
                    delete_model(m_no)
                    list_models_df.clear()
                    st.warning("Deleted (if it existed).")

    models_df = list_models_df()
    with st.expander("Models list", expanded=False):
        if models_df.empty:
            st.caption("No models yet.")
        else:
            st.dataframe(models_df, use_container_width=True, hide_index=True, height=220)

# ---------- Sidebar: Report First Piece (Top & Bottom) ----------
with st.sidebar:
    st.header("📷 First Piece")
    with st.expander("Report new First Piece", expanded=False):
        fp_model   = st.text_input("Model", key="fp_model")
        fp_version = st.text_input("Model Version")
        fp_sn      = st.text_input("SN / Barcode")
        fp_mo      = st.text_input("MO / Work Order")
        fp_desc    = st.text_area("Description / Notes")

        up_top = st.file_uploader("Upload **TOP** photo", type=["jpg","jpeg","png"], key="fp_top")
        up_bot = st.file_uploader("Upload **BOTTOM** photo", type=["jpg","jpeg","png"], key="fp_bottom")

        if st.button("Save First Piece", use_container_width=True, key="btn_fp_save"):
            if not fp_model.strip():
                st.error("Model is required.")
            elif not (up_top or up_bot):
                st.error("Upload at least one photo (TOP or BOTTOM).")
            else:
                top_rel = save_image(fp_model.strip(), up_top) if up_top else ""
                bot_rel = save_image(fp_model.strip(), up_bot) if up_bot else ""
                payload = {
                    "created_at": now_iso(),
                    "model_no": fp_model.strip(),
                    "model_version": fp_version.strip(),
                    "sn": fp_sn.strip(),
                    "mo": fp_mo.strip(),
                    "reporter": current_user(),
                    "description": fp_desc.strip(),
                    "image_path": top_rel or bot_rel,  # thumbnail
                    "extra": json.dumps({"top": top_rel, "bottom": bot_rel}, ensure_ascii=False),
                }
                insert_firstpiece(payload)
                st.success("First Piece saved.")
                upsert_model(fp_model.strip(), models_df.set_index("model_no")["name"].get(fp_model.strip(), ""))
                load_firstpiece_df.clear()

# ---------- Sidebar: Report Non-Conformity ----------
with st.sidebar:
    st.header("🛠️ Non-Conformity")
    with st.expander("Report new Non-Conformity", expanded=False):
        nc_model   = st.text_input("Model", key="nc_model")
        nc_version = st.text_input("Model Version", key="nc_version")
        nc_mo      = st.text_input("MO / Work Order", key="nc_mo")
        nc_line    = st.text_input("Line", key="nc_line")
        nc_station = st.text_input("Work Station", key="nc_station")
        nc_category= st.selectbox("Category", ["Minor", "Major", "Critical"], key="nc_cat")
        nc_type    = st.text_input("Nonconformity Type (e.g., Polarity, Short)", key="nc_type")
        nc_desc    = st.text_area("Description of Nonconformity", key="nc_desc")
        up_nc      = st.file_uploader("Upload photo (optional)", type=["jpg","jpeg","png"], key="nc_img")

        if st.button("Save Non-Conformity", use_container_width=True, key="btn_nc_save"):
            if not nc_model.strip():
                st.error("Model is required.")
            else:
                rel = save_image(nc_model.strip(), up_nc) if up_nc else ""
                payload = {
                    "created_at": now_iso(),
                    "model_no": nc_model.strip(),
                    "model_version": nc_version.strip(),
                    "mo": nc_mo.strip(),
                    "line": nc_line.strip(),
                    "station": nc_station.strip(),
                    "reporter": current_user(),
                    "category": nc_category,
                    "nc_type": nc_type.strip(),
                    "description": nc_desc.strip(),
                    "image_path": rel,
                    "extra": json.dumps({}, ensure_ascii=False),
                }
                insert_nc(payload)
                st.success("Non-Conformity saved.")
                upsert_model(nc_model.strip(), models_df.set_index("model_no")["name"].get(nc_model.strip(), ""))
                load_nc_df.clear()

# ========================= MAIN: Search & Results =========================

st.markdown("### 🔍 Search & View  <span class='small-note'>use filters below</span>", unsafe_allow_html=True)

with st.expander("Filters", expanded=True):
    c1, c2, c3, c4, c5, c6, c7 = st.columns([1,1,1,1,2,1,1])
    with c1:
        q_model = st.text_input("Model contains", key="q_model")
    with c2:
        q_version = st.text_input("Version contains", key="q_version")
    with c3:
        q_sn = st.text_input("SN contains", key="q_sn")
    with c4:
        q_mo = st.text_input("MO contains", key="q_mo")
    with c5:
        q_text = st.text_input("Text in description/reporter/type", key="q_text")
    with c6:
        q_from = st.date_input("Date from", key="q_from", value=None)
    with c7:
        q_to = st.date_input("Date to", key="q_to", value=None)

tabs = st.tabs(["📷 First Piece (results)", "🛠️ Non-Conformities (results)"])

iso_from, iso_to = dt_bounds(q_from if isinstance(q_from, date) else None,
                             q_to   if isinstance(q_to, date)   else None)

# ---------- First Piece results ----------
with tabs[0]:
    fdf = load_firstpiece_df(
        model=q_model or None,
        version=q_version or None,
        sn=q_sn or None,
        mo=q_mo or None,
        text=q_text or None,
        d_from=iso_from,
        d_to=iso_to,
    )
    st.caption(f"{len(fdf)} record(s)")

    for _, r in fdf.iterrows():
        # Parse extra for top/bottom, stay backward-compatible
        extra = {}
        try:
            extra = json.loads(r.get("extra") or "{}")
        except Exception:
            extra = {}
        top_rel = extra.get("top")
        bot_rel = extra.get("bottom")

        # legacy fallback: extra["images"] list
        if not (top_rel or bot_rel):
            imgs = extra.get("images", [])
            if isinstance(imgs, list):
                if len(imgs) >= 1: top_rel = imgs[0]
                if len(imgs) >= 2: bot_rel = imgs[1]

        with st.container(border=True):
            cols = st.columns([1, 1, 3])   # show TOP & BOTTOM side-by-side
            with cols[0]:
                p_top = DATA_DIR / str(top_rel) if top_rel else None
                if p_top and p_top.exists():
                    st.image(str(p_top), use_column_width=True, caption="TOP")
                else:
                    st.caption("No TOP")
            with cols[1]:
                p_bot = DATA_DIR / str(bot_rel) if bot_rel else None
                if p_bot and p_bot.exists():
                    st.image(str(p_bot), use_column_width=True, caption="BOTTOM")
                else:
                    st.caption("No BOTTOM")
            with cols[2]:
                st.markdown(
                    f"**Model:** {r.get('model_no','-')}  |  "
                    f"**Version:** {r.get('model_version','-')}  |  "
                    f"**SN:** {r.get('sn','-')}  |  "
                    f"**MO:** {r.get('mo','-')}"
                )
                st.caption(f"🕒 {r.get('created_at','')} · 👤 Reporter: {r.get('reporter','-')}")
                desc = (r.get("description") or "").strip()
                st.write(desc if desc else "_No description_")

                if is_admin():
                    if st.button("Delete", key=f"del_fp_{int(r['id'])}", type="secondary"):
                        delete_firstpiece(int(r["id"]))
                        load_firstpiece_df.clear()
                        st.experimental_rerun()

    with st.expander("Table • Export", expanded=False):
        st.dataframe(fdf.drop(columns=["extra"], errors="ignore"), use_container_width=True, hide_index=True)
        csv = fdf.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", data=csv, file_name="first_piece_results.csv", mime="text/csv")

# ---------- NC results ----------
with tabs[1]:
    c_cat = st.selectbox("Category", ["All", "Minor", "Major", "Critical"], key="q_cat")
    ndf = load_nc_df(
        model=q_model or None,
        version=q_version or None,
        mo=q_mo or None,
        text=q_text or None,
        category=c_cat,
        d_from=iso_from,
        d_to=iso_to,
    )
    st.caption(f"{len(ndf)} record(s)")

    for _, r in ndf.iterrows():
        with st.container(border=True):
            cols = st.columns([1, 3])
            with cols[0]:
                p = DATA_DIR / str(r.get("image_path","")) if r.get("image_path") else None
                if p and p.exists():
                    st.image(str(p), use_column_width=True)
                else:
                    st.caption("No image")
            with cols[1]:
                st.markdown(
                    f"**Model:** {r.get('model_no','-')}  |  "
                    f"**Version:** {r.get('model_version','-')}  |  "
                    f"**MO:** {r.get('mo','-')}  |  "
                    f"**Line/Station:** {r.get('line','-')}/{r.get('station','-')}  |  "
                    f"**Category:** {r.get('category','-')}  |  "
                    f"**Type:** {r.get('nc_type','-')}"
                )
                st.caption(f"🕒 {r.get('created_at','')} · 👤 Reporter: {r.get('reporter','-')}")
                desc = (r.get("description") or "").strip()
                st.write(desc if desc else "_No description_")

                if is_admin():
                    if st.button("Delete", key=f"del_nc_{int(r['id'])}", type="secondary"):
                        delete_nc(int(r["id"]))
                        load_nc_df.clear()
                        st.experimental_rerun()

    with st.expander("Table • Export", expanded=False):
        st.dataframe(ndf.drop(columns=["extra"], errors="ignore"), use_container_width=True, hide_index=True)
        csv = ndf.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", data=csv, file_name="nonconformities_results.csv", mime="text/csv")
