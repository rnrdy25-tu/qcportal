# QC Portal v2 – Models, First Piece, Non-Conformities, Search & Export
# Storage is Cloud-safe: /mount/data if available, else /tmp/qc_portal

import os
import io
import json
import zipfile
import sqlite3
from pathlib import Path
from datetime import datetime

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
    # Streamlit doesn't provide SSO user by default; we use a simple session signer.
    return st.session_state.get("user_name") or "Admin1"

def is_admin() -> bool:
    return current_user() in ADMIN_USERS

def save_image(model_no: str, uploaded_file) -> str:
    """
    Saves an uploaded image under images/<model_no>/ts_filename.jpg.
    Returns the path RELATIVE to DATA_DIR (so it’s portable).
    """
    folder = IMG_DIR / model_no
    folder.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    safe_name = uploaded_file.name.replace(" ", "_")
    out_path = folder / f"{ts}_{safe_name}"
    img = Image.open(uploaded_file).convert("RGB")
    img.save(out_path, format="JPEG", quality=90)
    return str(out_path.relative_to(DATA_DIR))

# ========================= Database =========================

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
  image_path TEXT,               -- thumbnail / first image
  extra JSON                     -- {"images": [...]}
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
  category TEXT,                 -- Minor / Major / Critical
  nc_type TEXT,                  -- Nonconformity type (admin-curated text)
  description TEXT,              -- free text
  image_path TEXT,               -- optional main image
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
    model=None, version=None, sn=None, mo=None, text=None, limit=500
) -> pd.DataFrame:
    q = """
    SELECT id, created_at, model_no, model_version, sn, mo, reporter,
           description, image_path
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
    q += " ORDER BY id DESC LIMIT ?"
    params.append(limit)

    with get_conn() as c:
        return pd.read_sql_query(q, c, params=params)

@st.cache_data(show_spinner=False)
def load_nc_df(
    model=None, version=None, mo=None, text=None, category=None, limit=500
) -> pd.DataFrame:
    q = """
    SELECT id, created_at, model_no, model_version, mo, line, station, reporter,
           category, nc_type, description, image_path
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
st.set_page_config(page_title="QC Portal", layout="wide")
st.title("🔎 QC Portal — Models, First Piece, Non-Conformities, Search & Export")

# ---------- Sidebar: Sign in ----------
with st.sidebar:
    with st.expander("👤 Sign in", expanded=True):
        st.text_input("Your name", key="user_name", value=st.session_state.get("user_name", "Admin1"))
        st.caption(
            "This name is saved as **Reporter**. Admin features are enabled for **Admin1**."
        )

# ---------- Sidebar: Models (Admin-friendly) ----------
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

# ---------- Sidebar: Report First Piece ----------
with st.sidebar:
    st.header("📷 First Piece")
    with st.expander("Report new First Piece", expanded=False):
        fp_model   = st.text_input("Model", key="fp_model")
        fp_version = st.text_input("Model Version")
        fp_sn      = st.text_input("SN / Barcode")
        fp_mo      = st.text_input("MO / Work Order")
        fp_desc    = st.text_area("Description / Notes")

        up_imgs_fp = st.file_uploader(
            "Upload photo(s) (Top/Bottom)", type=["jpg","jpeg","png"], accept_multiple_files=True
        )
        if st.button("Save First Piece", use_container_width=True, key="btn_fp_save"):
            if not fp_model.strip():
                st.error("Model is required.")
            elif not up_imgs_fp:
                st.error("Please upload at least one photo.")
            else:
                rel_paths = []
                for uf in up_imgs_fp:
                    try:
                        rel_paths.append(save_image(fp_model.strip(), uf))
                    except Exception as e:
                        st.error(f"Failed saving {uf.name}: {e}")
                        rel_paths = []
                        break
                if rel_paths:
                    payload = {
                        "created_at": now_iso(),
                        "model_no": fp_model.strip(),
                        "model_version": fp_version.strip(),
                        "sn": fp_sn.strip(),
                        "mo": fp_mo.strip(),
                        "reporter": current_user(),
                        "description": fp_desc.strip(),
                        "image_path": rel_paths[0],
                        "extra": json.dumps({"images": rel_paths}, ensure_ascii=False),
                    }
                    insert_firstpiece(payload)
                    st.success("First Piece saved.")
                    # keep model registry fresh
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
        up_imgs_nc = st.file_uploader("Upload photo(s) (optional)", type=["jpg","jpeg","png"], accept_multiple_files=True, key="nc_imgs")

        if st.button("Save Non-Conformity", use_container_width=True, key="btn_nc_save"):
            if not nc_model.strip():
                st.error("Model is required.")
            else:
                rel_paths = []
                for uf in (up_imgs_nc or []):
                    try:
                        rel_paths.append(save_image(nc_model.strip(), uf))
                    except Exception as e:
                        st.error(f"Failed saving {uf.name}: {e}")
                        rel_paths = []
                        break
                if rel_paths or not up_imgs_nc:
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
                        "image_path": rel_paths[0] if rel_paths else "",
                        "extra": json.dumps({"images": rel_paths}, ensure_ascii=False),
                    }
                    insert_nc(payload)
                    st.success("Non-Conformity saved.")
                    upsert_model(nc_model.strip(), models_df.set_index("model_no")["name"].get(nc_model.strip(), ""))
                    load_nc_df.clear()

# ========================= MAIN: Search & Results =========================

st.subheader("🔍 Search & View")

with st.expander("Filters", expanded=True):
    c1, c2, c3, c4, c5 = st.columns([1,1,1,1,2])
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

tabs = st.tabs(["📷 First Piece (results)", "🛠️ Non-Conformities (results)"])

# ---------- First Piece results ----------
with tabs[0]:
    fdf = load_firstpiece_df(
        model=q_model or None,
        version=q_version or None,
        sn=q_sn or None,
        mo=q_mo or None,
        text=q_text or None,
    )
    st.caption(f"{len(fdf)} record(s)")

    for _, r in fdf.iterrows():
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
        st.dataframe(fdf, use_container_width=True, hide_index=True)
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
        st.dataframe(ndf, use_container_width=True, hide_index=True)
        csv = ndf.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", data=csv, file_name="nonconformities_results.csv", mime="text/csv")
