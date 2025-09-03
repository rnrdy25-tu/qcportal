# app.py
# QC Portal – Models, First Piece, Non-Conformities, History & Export
# Storage is cloud-safe: prefer /mount/data if present, else use /tmp/qc_portal

from __future__ import annotations

import os
import io
import json
import sqlite3
from pathlib import Path
from datetime import datetime

import streamlit as st
import pandas as pd
from PIL import Image

# ===================== Storage (cloud-safe) =====================

def pick_data_dir() -> Path:
    """
    Pick a writable directory that works on Streamlit Cloud and locally.
    """
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


# ===================== Utilities =====================

def now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds")

def cur_user() -> str:
    # Best-effort user label for audit
    return os.environ.get("USERNAME") or os.environ.get("USER") or "Operator"

def save_image(model_no: str, uploaded_file) -> str:
    """
    Save an uploaded image under images/<model_no>/... and return the
    relative path (relative to DATA_DIR) to store in DB.
    """
    folder = IMG_DIR / model_no
    folder.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    safe_name = uploaded_file.name.replace(" ", "_")
    out_path = folder / f"{ts}_{safe_name}"

    img = Image.open(uploaded_file).convert("RGB")
    # Pillow 10.x: only use supported args
    img.save(out_path, format="JPEG", quality=90)

    return str(out_path.relative_to(DATA_DIR))


# ===================== Database & Schema =====================

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
  image_path TEXT,   -- first image
  extra JSON         -- {"images":[...]}
);
"""

SCHEMA_SETTINGS = """
CREATE TABLE IF NOT EXISTS settings(
  key  TEXT PRIMARY KEY,
  json TEXT
);
"""

SCHEMA_NONCONS = """
CREATE TABLE IF NOT EXISTS noncons(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at TEXT,
  model_no TEXT,
  model_version TEXT,
  mo TEXT,
  line TEXT,
  station TEXT,
  customer TEXT,
  nc_type TEXT,        -- Nonconformity type (controlled list)
  nc_desc TEXT,        -- Free text description
  origin_source TEXT,  -- Controlled list
  def_group TEXT,      -- Controlled list
  def_item TEXT,       -- Controlled list
  outflow TEXT,        -- Controlled list
  defect_qty INTEGER,
  inspect_qty INTEGER,
  lot_qty INTEGER,
  severity TEXT,       -- Minor/Major/Critical
  reporter TEXT,
  image_path TEXT,     -- first image
  extra JSON           -- {"images":[...]}
);
"""

DEFAULT_VOCAB = {
    "nonconformity_types": ["Polarity", "Short Circuit", "Open Solder"],
    "origins": ["IQC", "IPQC", "FQC", "OQC", "Customer Return"],
    "defective_groups": ["Electrical", "Mechanical", "Cosmetic"],
    "defective_items": ["Solder bridge", "Missing part", "Scratch"],
    "outflows": ["In-process", "To customer", "Scrap"],
}

def get_conn():
    return sqlite3.connect(DB_PATH)

def init_db():
    with get_conn() as c:
        c.execute(SCHEMA_MODELS)
        c.execute(SCHEMA_FINDINGS)
        c.execute(SCHEMA_SETTINGS)
        c.execute(SCHEMA_NONCONS)
        c.commit()

    # Seed default vocab once
    with get_conn() as c:
        row = c.execute("SELECT 1 FROM settings WHERE key='vocab'").fetchone()
    if not row:
        save_vocab(DEFAULT_VOCAB)

def _merge_defaults(current: dict | None) -> dict:
    data = dict(DEFAULT_VOCAB)
    if isinstance(current, dict):
        for k in DEFAULT_VOCAB:
            v = current.get(k)
            if isinstance(v, list) and v:
                data[k] = v
    return data

def load_vocab() -> dict:
    with get_conn() as c:
        row = c.execute("SELECT json FROM settings WHERE key='vocab'").fetchone()
    if not row:
        return dict(DEFAULT_VOCAB)
    try:
        return _merge_defaults(json.loads(row[0]))
    except Exception:
        return dict(DEFAULT_VOCAB)

def save_vocab(vocab: dict):
    payload = json.dumps(_merge_defaults(vocab), ensure_ascii=False)
    with get_conn() as c:
        c.execute(
            """INSERT INTO settings(key, json) VALUES('vocab', ?)
               ON CONFLICT(key) DO UPDATE SET json=excluded.json""",
            (payload,),
        )
        c.commit()

# ---------- Model operations ----------

@st.cache_data(show_spinner=False)
def list_models_df() -> pd.DataFrame:
    with get_conn() as c:
        return pd.read_sql_query(
            "SELECT model_no, COALESCE(name,'') AS name FROM models ORDER BY model_no",
            c,
        )

def upsert_model(model_no: str, name: str = ""):
    with get_conn() as c:
        c.execute(
            """INSERT INTO models(model_no, name)
               VALUES (?, ?)
               ON CONFLICT(model_no) DO UPDATE SET name=excluded.name""",
            (model_no.strip(), name.strip()),
        )
        c.commit()

def delete_model(model_no: str):
    with get_conn() as c:
        c.execute("DELETE FROM models WHERE model_no=?", (model_no.strip(),))
        c.commit()

# ---------- Findings (First Piece) ----------

def insert_finding(payload: dict):
    fields = [
        "created_at", "model_no", "model_version", "sn", "mo",
        "reporter", "description", "image_path", "extra",
    ]
    values = [payload.get(k) for k in fields]
    with get_conn() as c:
        c.execute(
            f"INSERT INTO findings({','.join(fields)}) VALUES({','.join(['?']*len(fields))})",
            values,
        )
        c.commit()

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

# ---------- Nonconformities ----------

def insert_noncon(payload: dict):
    fields = [
        "created_at", "model_no", "model_version", "mo", "line", "station", "customer",
        "nc_type", "nc_desc", "origin_source", "def_group", "def_item", "outflow",
        "defect_qty", "inspect_qty", "lot_qty", "severity", "reporter", "image_path", "extra",
    ]
    values = [payload.get(k) for k in fields]
    with get_conn() as c:
        c.execute(
            f"INSERT INTO noncons({','.join(fields)}) VALUES({','.join(['?']*len(fields))})",
            values,
        )
        c.commit()

@st.cache_data(show_spinner=False)
def load_noncon_df(model_no: str) -> pd.DataFrame:
    with get_conn() as c:
        return pd.read_sql_query(
            """SELECT id, created_at, model_version, mo, line, station, customer,
                      nc_type, nc_desc, origin_source, def_group, def_item, outflow,
                      defect_qty, inspect_qty, lot_qty, severity, reporter, image_path
               FROM noncons
               WHERE model_no=?
               ORDER BY id DESC""",
            c,
            params=(model_no,),
        )

# ===================== App UI =====================

init_db()
st.set_page_config(page_title="QC Portal", layout="wide")
st.title("🔎 QC Portal — Models, First Piece, Non-Conformities & History")

# --------------- Sidebar ---------------

with st.sidebar:
    st.header("📚 Models")

    # Add / Update Model (collapsible)
    with st.expander("Add / Update Model", expanded=False):
        m_no = st.text_input("Model (short form, e.g. 190-56980)", key="mno")
        m_name = st.text_input("Name / Customer (optional)", key="mname")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Save model", key="btn_save_model"):
                if m_no.strip():
                    upsert_model(m_no, m_name)
                    st.success("Model saved.")
                    list_models_df.clear()
                else:
                    st.error("Model cannot be empty.")
        with c2:
            if st.button("Delete model", key="btn_del_model"):
                if m_no.strip():
                    delete_model(m_no)
                    st.warning("Deleted (if existed).")
                    list_models_df.clear()
                else:
                    st.error("Enter a model to delete.")

    # Models picker (collapsible)
    with st.expander("Select Model", expanded=True):
        models = list_models_df()
        if models.empty:
            st.info("No models yet. Add one above.")
            picked = None
        else:
            labels = [
                f"{r.model_no}  •  {r.name}" if str(r.name).strip() else r.model_no
                for _, r in models.iterrows()
            ]
            model_options = models["model_no"].tolist()
            label_map = dict(zip(model_options, labels))
            picked = st.radio(
                "Models",
                options=model_options,
                format_func=lambda m: label_map.get(m, m),
                key="picked_model",
            )

    st.divider()

    # First Piece (collapsible)
    with st.expander("📷 First Piece", expanded=False):
        with st.form("first_piece_form", clear_on_submit=True):
            fp_model = picked or st.text_input("Model (if none selected)", key="fp_model")
            fp_version = st.text_input("Model Version (full)", key="fp_version")
            fp_sn = st.text_input("SN / Barcode", key="fp_sn")
            fp_mo = st.text_input("MO / Work Order", key="fp_mo")
            fp_desc = st.text_area("Notes (optional)", key="fp_desc")
            fp_imgs = st.file_uploader(
                "Upload photo(s) (Top / Bottom)",
                type=["jpg", "jpeg", "png"],
                accept_multiple_files=True,
                key="fp_imgs",
            )

            submitted_fp = st.form_submit_button("Save first piece")
            if submitted_fp:
                if not (fp_model or "").strip():
                    st.error("Model is required.")
                elif not fp_imgs:
                    st.error("Please upload at least one photo.")
                else:
                    rel_paths = []
                    for uf in fp_imgs:
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
                            "model_version": (fp_version or "").strip(),
                            "sn": (fp_sn or "").strip(),
                            "mo": (fp_mo or "").strip(),
                            "reporter": cur_user(),
                            "description": (fp_desc or "").strip(),
                            "image_path": rel_paths[0],
                            "extra": json.dumps({"images": rel_paths}, ensure_ascii=False),
                        }
                        insert_finding(payload)
                        st.success("Saved first piece.")
                        load_findings_df.clear()

                        # Ensure model exists
                        name_val = ""
                        if not models.empty and payload["model_no"] in set(models["model_no"]):
                            try:
                                name_val = (
                                    models.loc[
                                        models["model_no"] == payload["model_no"], "name"
                                    ]
                                    .astype(str)
                                    .iloc[0]
                                )
                            except Exception:
                                name_val = ""
                        upsert_model(payload["model_no"], name_val)
                        list_models_df.clear()

    # Non-Conformity (collapsible)
    with st.expander("📝 Non-Conformity", expanded=False):
        vocab = load_vocab()
        with st.form("noncon_form", clear_on_submit=True):
            nc_model = picked or st.text_input("Model (if none selected)", key="nc_model")
            nc_version = st.text_input("Model Version (full)", key="nc_version")
            nc_mo = st.text_input("MO / Work Order", key="nc_mo")
            nc_line = st.text_input("Line", key="nc_line")
            nc_station = st.text_input("Work Station", key="nc_station")
            nc_customer = st.text_input("Customer / Supplier", key="nc_customer")

            nc_type = st.selectbox("Nonconformity", options=vocab["nonconformity_types"], key="nc_type")
            nc_desc = st.text_area("Description of Nonconformity", key="nc_desc")

            origin = st.selectbox("Origin source", options=vocab["origins"], key="nc_origin")
            def_group = st.selectbox("Defective group", options=vocab["defective_groups"], key="nc_group")
            def_item = st.selectbox("Defective item", options=vocab["defective_items"], key="nc_item")
            outflow = st.selectbox("Defective outflow", options=vocab["outflows"], key="nc_outflow")

            cqty1, cqty2, cqty3 = st.columns(3)
            with cqty1:
                defect_qty = st.number_input("Defective Qty", min_value=0, step=1, value=0, key="nc_defq")
            with cqty2:
                inspect_qty = st.number_input("Inspection Qty", min_value=0, step=1, value=0, key="nc_insq")
            with cqty3:
                lot_qty = st.number_input("Lot Qty", min_value=0, step=1, value=0, key="nc_lotq")

            severity = st.selectbox("Severity", options=["Minor", "Major", "Critical"], key="nc_sev")
            nc_imgs = st.file_uploader(
                "Upload photo(s)",
                type=["jpg", "jpeg", "png"],
                accept_multiple_files=True,
                key="nc_imgs",
            )

            submitted_nc = st.form_submit_button("Save non-conformity")
            if submitted_nc:
                if not (nc_model or "").strip():
                    st.error("Model is required.")
                else:
                    rel_paths = []
                    if nc_imgs:
                        for uf in nc_imgs:
                            try:
                                rel_paths.append(save_image(nc_model.strip(), uf))
                            except Exception as e:
                                st.error(f"Failed saving {getattr(uf, 'name', 'image')}: {e}")
                                rel_paths = []
                                break
                    payload = {
                        "created_at": now_iso(),
                        "model_no": (nc_model or "").strip(),
                        "model_version": (nc_version or "").strip(),
                        "mo": (nc_mo or "").strip(),
                        "line": (nc_line or "").strip(),
                        "station": (nc_station or "").strip(),
                        "customer": (nc_customer or "").strip(),
                        "nc_type": nc_type,
                        "nc_desc": (nc_desc or "").strip(),
                        "origin_source": origin,
                        "def_group": def_group,
                        "def_item": def_item,
                        "outflow": outflow,
                        "defect_qty": int(defect_qty or 0),
                        "inspect_qty": int(inspect_qty or 0),
                        "lot_qty": int(lot_qty or 0),
                        "severity": severity,
                        "reporter": cur_user(),
                        "image_path": rel_paths[0] if rel_paths else None,
                        "extra": json.dumps({"images": rel_paths}, ensure_ascii=False),
                    }
                    insert_noncon(payload)
                    st.success("Saved non-conformity.")
                    load_noncon_df.clear()

                    # Ensure model exists
                    name_val = ""
                    if not models.empty and payload["model_no"] in set(models["model_no"]):
                        try:
                            name_val = (
                                models.loc[
                                    models["model_no"] == payload["model_no"], "name"
                                ]
                                .astype(str)
                                .iloc[0]
                            )
                        except Exception:
                            name_val = ""
                    upsert_model(payload["model_no"], name_val)
                    list_models_df.clear()

    # Admin – controlled lists (collapsible)
    with st.expander("⚙️ Admin — Controlled lists", expanded=False):
        vocab = load_vocab()
        cA, cB = st.columns(2)
        with cA:
            nc_types_txt = st.text_area(
                "Nonconformity types (comma-separated)",
                value=", ".join(vocab["nonconformity_types"]),
                height=100,
                key="adm_nc_types",
            )
            origins_txt = st.text_area(
                "Origin sources (comma-separated)",
                value=", ".join(vocab["origins"]),
                height=100,
                key="adm_origins",
            )
            outflows_txt = st.text_area(
                "Defective outflows (comma-separated)",
                value=", ".join(vocab["outflows"]),
                height=100,
                key="adm_outflows",
            )
        with cB:
            groups_txt = st.text_area(
                "Defective groups (comma-separated)",
                value=", ".join(vocab["defective_groups"]),
                height=100,
                key="adm_groups",
            )
            items_txt = st.text_area(
                "Defective items (comma-separated)",
                value=", ".join(vocab["defective_items"]),
                height=100,
                key="adm_items",
            )

        if st.button("Save lists", key="adm_save_lists"):
            new_vocab = {
                "nonconformity_types": [s.strip() for s in nc_types_txt.split(",") if s.strip()],
                "origins": [s.strip() for s in origins_txt.split(",") if s.strip()],
                "outflows": [s.strip() for s in outflows_txt.split(",") if s.strip()],
                "defective_groups": [s.strip() for s in groups_txt.split(",") if s.strip()],
                "defective_items": [s.strip() for s in items_txt.split(",") if s.strip()],
            }
            save_vocab(new_vocab)
            st.success("Lists saved.")

    # Export (collapsible)
    with st.expander("⬇️ Export", expanded=False):
        if picked:
            st.caption(f"Selected model: **{picked}**")

            # Load both datasets
            fdf = load_findings_df(picked)
            ndf = load_noncon_df(picked)

            # CSV exports
            cexp1, cexp2 = st.columns(2)
            with cexp1:
                if not fdf.empty:
                    csv1 = fdf.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "Download First Piece (CSV)",
                        data=csv1,
                        file_name=f"{picked}_first_piece.csv",
                        mime="text/csv",
                        key="dl_fp_csv",
                    )
            with cexp2:
                if not ndf.empty:
                    csv2 = ndf.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "Download Non-Conformities (CSV)",
                        data=csv2,
                        file_name=f"{picked}_noncon.csv",
                        mime="text/csv",
                        key="dl_nc_csv",
                    )

            # Excel workbook (2 sheets)
            if not fdf.empty or not ndf.empty:
                bio = io.BytesIO()
                with pd.ExcelWriter(bio, engine="xlsxwriter") as xw:
                    if not fdf.empty:
                        fdf.to_excel(xw, sheet_name="FirstPiece", index=False)
                    if not ndf.empty:
                        ndf.to_excel(xw, sheet_name="NonCon", index=False)
                bio.seek(0)
                st.download_button(
                    "Download both (Excel)",
                    data=bio.getvalue(),
                    file_name=f"{picked}_qc_export.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="dl_both_xlsx",
                )
        else:
            st.info("Pick a model above to enable exports.")


# --------------- Main Pane (History) ---------------

if picked:
    # First Piece history
    st.subheader(f"🗂️ First Piece — {picked}")
    fdf = load_findings_df(picked)
    if fdf.empty:
        st.info("No first-piece records yet for this model.")
    else:
        for _, r in fdf.iterrows():
            with st.container(border=True):
                cols = st.columns([1, 3])
                with cols[0]:
                    p = DATA_DIR / str(r.get("image_path", "")) if r.get("image_path") else None
                    if p and p.exists():
                        st.image(str(p), use_container_width=True)
                with cols[1]:
                    top = f"**Version:** {r.get('model_version','-')}  |  **SN:** {r.get('sn','-')}  |  **MO:** {r.get('mo','-')}"
                    st.markdown(top)
                    st.caption(f"{r.get('created_at','')}  ·  Reporter: {r.get('reporter','')}")
                    desc = r.get("description", "")
                    if isinstance(desc, str) and desc.strip():
                        st.write(desc)

        with st.expander("Show table (First Piece)"):
            st.dataframe(fdf, use_container_width=True, hide_index=True)

    st.markdown("---")

    # Non-Conformities history
    st.subheader(f"🗂️ Non-Conformities — {picked}")
    ndf = load_noncon_df(picked)
    if ndf.empty:
        st.info("No non-conformities yet for this model.")
    else:
        for _, r in ndf.iterrows():
            with st.container(border=True):
                cols = st.columns([1, 3])
                with cols[0]:
                    p = DATA_DIR / str(r.get("image_path", "")) if r.get("image_path") else None
                    if p and p.exists():
                        st.image(str(p), use_container_width=True)
                with cols[1]:
                    top = (
                        f"**Type:** {r.get('nc_type','-')}  |  **Severity:** {r.get('severity','-')}  |  "
                        f"**MO:** {r.get('mo','-')}  |  **Line/Station:** {r.get('line','-')}/{r.get('station','-')}"
                    )
                    st.markdown(top)
                    st.caption(
                        f"{r.get('created_at','')} · Reporter: {r.get('reporter','')} · "
                        f"Origin: {r.get('origin_source','-')} · Outflow: {r.get('outflow','-')}"
                    )
                    desc = r.get("nc_desc", "")
                    if isinstance(desc, str) and desc.strip():
                        st.write(desc)
                    st.caption(
                        f"Defective: {r.get('defect_qty',0)}  |  Inspected: {r.get('inspect_qty',0)}  |  Lot: {r.get('lot_qty',0)}"
                    )

        with st.expander("Show table (Non-Conformities)"):
            st.dataframe(ndf, use_container_width=True, hide_index=True)

else:
    st.info("Select a model in the sidebar to view its history.")
