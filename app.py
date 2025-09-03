# QC Portal v1.1 — Models + First Piece + History
# Storage: /mount/data if available (Streamlit Cloud), else /tmp/qc_portal

import os
import io
import json
import sqlite3
from pathlib import Path
from datetime import datetime

import streamlit as st
import pandas as pd
from PIL import Image

# ------------------ storage (Cloud-safe) ------------------
def pick_data_dir() -> Path:
    """Return a writable base dir."""
    for base in (Path("/mount/data"), Path("/tmp/qc_portal")):
        try:
            base.mkdir(parents=True, exist_ok=True)
            (base / ".write_test").write_text("ok", encoding="utf-8")
            return base
        except Exception:
            pass
    raise RuntimeError("No writable directory found")

DATA_DIR = pick_data_dir()
IMG_DIR  = DATA_DIR / "images"
DB_PATH  = DATA_DIR / "qc_portal.sqlite3"
IMG_DIR.mkdir(parents=True, exist_ok=True)

# ------------------ utils ------------------
def now_iso():
    return datetime.utcnow().isoformat()

def cur_user():
    # best effort user label
    return os.environ.get("USERNAME") or os.environ.get("USER") or "Operator"

def save_image(model_no: str, uploaded_file) -> str:
    """Save upload under images/<model_no>/..., return path relative to DATA_DIR"""
    folder = IMG_DIR / model_no
    folder.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    out_path = folder / f"{ts}_{uploaded_file.name.replace(' ', '_')}"
    img = Image.open(uploaded_file).convert("RGB")
    img.save(out_path, format="JPEG", quality=90)
    return str(out_path.relative_to(DATA_DIR))

# ------------------ database ------------------
SCHEMA_MODELS = """
CREATE TABLE IF NOT EXISTS models(
  model_no TEXT PRIMARY KEY,
  name     TEXT
);
"""

SCHEMA_FINDINGS = """
CREATE TABLE IF NOT EXISTS findings(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at TEXT,
  model_no TEXT,
  model_version TEXT,
  sn TEXT,
  mo TEXT,
  reporter TEXT,
  description TEXT,
  image_path TEXT,   -- thumbnail/first photo
  extra JSON         -- {"images": [...]} (all saved images)
);
"""

def get_conn():
    return sqlite3.connect(DB_PATH)

def init_db():
    with get_conn() as c:
        c.execute(SCHEMA_MODELS)
        c.execute(SCHEMA_FINDINGS)
        c.commit()

@st.cache_data(show_spinner=False)
def list_models_df() -> pd.DataFrame:
    with get_conn() as c:
        return pd.read_sql_query(
            "SELECT model_no, COALESCE(name,'') AS name FROM models ORDER BY model_no",
            c,
        )

@st.cache_data(show_spinner=False)
def load_findings_df(model_no: str) -> pd.DataFrame:
    with get_conn() as c:
        return pd.read_sql_query(
            """SELECT id, created_at, model_version, sn, mo, reporter,
                      description, image_path
               FROM findings
               WHERE model_no=?
               ORDER BY id DESC""",
            c,
            params=(model_no,),
        )

def upsert_model(model_no: str, name: str = ""):
    with get_conn() as c:
        c.execute(
            """INSERT INTO models(model_no, name)
               VALUES(?, ?)
               ON CONFLICT(model_no) DO UPDATE SET name=excluded.name""",
            (model_no.strip(), name.strip()),
        )
        c.commit()

def delete_model(model_no: str):
    with get_conn() as c:
        c.execute("DELETE FROM models WHERE model_no=?", (model_no,))
        c.commit()

def insert_finding(payload: dict):
    fields = [
        "created_at", "model_no", "model_version", "sn", "mo",
        "reporter", "description", "image_path", "extra"
    ]
    values = [payload.get(k) for k in fields]
    with get_conn() as c:
        c.execute(
            f"INSERT INTO findings({','.join(fields)}) VALUES(?,?,?,?,?,?,?,?,?)",
            values,
        )
        c.commit()

# ------------------ app UI ------------------
init_db()
st.set_page_config(page_title="QC Portal", layout="wide")
st.title("🔎 QC Portal – Models, First Piece, History")

# -------- Sidebar: Models registry & first-piece form --------
with st.sidebar:
    st.header("📚 Models")

    # Add / update model
    with st.expander("Add / Update Model", expanded=True):
        m_no = st.text_input("Model (short form, e.g. 190-56980)", key="mno")
        m_name = st.text_input("Name / Customer (optional)", key="mname")
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Save model"):
                if m_no.strip():
                    upsert_model(m_no, m_name)
                    st.success("Saved.")
                    list_models_df.clear()
                else:
                    st.error("Model cannot be empty.")
        with col_b:
            if st.button("Delete model", type="secondary"):
                if m_no.strip():
                    delete_model(m_no.strip())
                    st.warning("Deleted (if it existed).")
                    list_models_df.clear()
                else:
                    st.error("Enter a model to delete.")

    # Pick a model radio
    models = list_models_df()
    picked = None
    if models.empty:
        st.info("No models yet. Add one above.")
    else:
        options = models["model_no"].tolist()
        labels = [
            f"{row.model_no}  •  {row.name}" if row.name else row.model_no
            for _, row in models.iterrows()
        ]
        label_map = dict(zip(options, labels))
        picked = st.radio(
            "Select a model",
            options=options,
            format_func=lambda m: label_map.get(m, m),
            key="picked_model",
        )

    st.divider()

    # First piece report form (minimal)
    st.header("📷 First Piece")
    with st.form("first_piece_form", clear_on_submit=True):
        fp_model = picked or st.text_input("Model (if none selected)")
        fp_version = st.text_input("Model Version (full)")
        fp_sn = st.text_input("SN / Barcode")
        fp_mo = st.text_input("MO / Work Order")
        fp_desc = st.text_area("Notes (optional)")
        up_imgs = st.file_uploader(
            "Upload photo(s) (Top / Bottom)",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
        )
        submitted = st.form_submit_button("Save first piece")
        if submitted:
            if not (fp_model or "").strip():
                st.error("Model is required.")
            elif not up_imgs:
                st.error("Please upload at least one photo.")
            else:
                # Save images
                rel_paths = []
                for uf in up_imgs:
                    try:
                        rel_paths.append(save_image(fp_model.strip(), uf))
                    except Exception as e:
                        st.error(f"Failed saving {getattr(uf, 'name', 'image')}: {e}")
                        rel_paths = []
                        break
                if rel_paths:
                    payload = {
                        "created_at": now_iso(),
                        "model_no": fp_model.strip(),
                        "model_version": fp_version.strip(),
                        "sn": fp_sn.strip(),
                        "mo": fp_mo.strip(),
                        "reporter": cur_user(),
                        "description": fp_desc.strip(),
                        "image_path": rel_paths[0],
                        "extra": json.dumps({"images": rel_paths}, ensure_ascii=False),
                    }
                    insert_finding(payload)
                    st.success("Saved first piece!")
                    load_findings_df.clear()

                    # also ensure model exists (carry existing name if present)
                    name_val = ""
                    if not models.empty and fp_model.strip() in set(models["model_no"]):
                        try:
                            name_val = models.loc[
                                models["model_no"] == fp_model.strip(), "name"
                            ].iloc[0]
                        except Exception:
                            name_val = ""
                    upsert_model(fp_model.strip(), name_val)
                    list_models_df.clear()

# -------- Main: history for selected model --------
if picked:
    st.subheader(f"🗂️ History — {picked}")
    fdf = load_findings_df(picked)
    if fdf.empty:
        st.info("No records yet for this model.")
    else:
        # card list
        for _, r in fdf.iterrows():
            with st.container(border=True):
                cols = st.columns([1, 3])
                with cols[0]:
                    p = DATA_DIR / str(r["image_path"]) if r["image_path"] else None
                    if p and p.exists():
                        # st.image supports use_column_width=True on 1.36
                        st.image(str(p), use_column_width=True)
                with cols[1]:
                    st.markdown(
                        f"**Version:** {r.get('model_version', '') or '-'}  |  "
                        f"**SN:** {r.get('sn', '') or '-'}  |  "
                        f"**MO:** {r.get('mo', '') or '-'}"
                    )
                    st.caption(f"{r.get('created_at','')}  ·  Reporter: {r.get('reporter','')}")
                    desc = r.get("description", "")
                    if isinstance(desc, str) and desc.strip():
                        st.write(desc)

        # table view under cards
        with st.expander("Show table"):
            # On Streamlit 1.36 use use_container_width (not use_column_width)
            st.dataframe(fdf, use_container_width=True, hide_index=True)
else:
    st.info("Select a model in the sidebar to view its history.")
