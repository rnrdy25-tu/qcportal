# app.py
# QC Portal — Search-first view for Non-Conformities & First Piece + Capture forms
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
    return os.environ.get("USERNAME") or os.environ.get("USER") or "Operator"

def save_image(model_no: str, uploaded_file) -> str:
    folder = IMG_DIR / model_no
    folder.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    safe_name = uploaded_file.name.replace(" ", "_")
    out_path = folder / f"{ts}_{safe_name}"
    img = Image.open(uploaded_file).convert("RGB")
    img.save(out_path, format="JPEG", quality=90)  # Pillow 10.x safe
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
  image_path TEXT,
  extra JSON
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
  nc_type TEXT,
  nc_desc TEXT,
  origin_source TEXT,
  def_group TEXT,
  def_item TEXT,
  outflow TEXT,
  defect_qty INTEGER,
  inspect_qty INTEGER,
  lot_qty INTEGER,
  severity TEXT,
  reporter TEXT,
  image_path TEXT,
  extra JSON
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

# ---------- Capture ops ----------

def upsert_model(model_no: str, name: str = ""):
    with get_conn() as c:
        c.execute(
            """INSERT INTO models(model_no, name)
               VALUES (?, ?)
               ON CONFLICT(model_no) DO UPDATE SET name=excluded.name""",
            (model_no.strip(), name.strip()),
        )
        c.commit()

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

# ---------- Search helpers (dynamic SQL with optional filters) ----------

def _build_like(value: str | None) -> tuple[str, list]:
    if value and str(value).strip():
        return "LIKE ?", [f"%{str(value).strip()}%"]
    return "", []

def search_noncons(model=None, version=None, sn=None, mo=None) -> pd.DataFrame:
    sql = """SELECT id, created_at, model_no, model_version, mo, line, station, customer,
                    nc_type, nc_desc, origin_source, def_group, def_item, outflow,
                    defect_qty, inspect_qty, lot_qty, severity, reporter, image_path
             FROM noncons WHERE 1=1 """
    params: list = []

    # Filters (case-insensitive LIKE via sqlite default)
    if model and model.strip():
        sql += " AND model_no LIKE ?"
        params.append(f"%{model.strip()}%")
    if version and version.strip():
        sql += " AND model_version LIKE ?"
        params.append(f"%{version.strip()}%")
    if sn and sn.strip():
        # SN is not in noncons schema; still accept for parity and ignore if present
        pass
    if mo and mo.strip():
        sql += " AND mo LIKE ?"
        params.append(f"%{mo.strip()}%")

    sql += " ORDER BY created_at DESC, id DESC"

    with get_conn() as c:
        return pd.read_sql_query(sql, c, params=params)

def search_findings(model=None, version=None, mo=None) -> pd.DataFrame:
    sql = """SELECT id, created_at, model_no, model_version, sn, mo, reporter,
                    description, image_path
             FROM findings WHERE 1=1 """
    params: list = []
    if model and model.strip():
        sql += " AND model_no LIKE ?"
        params.append(f"%{model.strip()}%")
    if version and version.strip():
        sql += " AND model_version LIKE ?"
        params.append(f"%{version.strip()}%")
    if mo and mo.strip():
        sql += " AND mo LIKE ?"
        params.append(f"%{mo.strip()}%")
    sql += " ORDER BY created_at DESC, id DESC"
    with get_conn() as c:
        return pd.read_sql_query(sql, c, params=params)

# ===================== App UI =====================

init_db()
st.set_page_config(page_title="QC Portal", layout="wide")
st.title("🔎 QC Portal — View / Search + Capture")

# Sidebar — View/Search + Capture forms + Admin
with st.sidebar:
    # -------- VIEW / SEARCH --------
    st.header("👀 View / Search")

    with st.expander("Search Non-Conformities", expanded=True):
        s_model = st.text_input("Model", key="sv_nc_model")
        s_version = st.text_input("Model Version", key="sv_nc_version")
        s_sn = st.text_input("SN (optional; ignored for NC)", key="sv_nc_sn")
        s_mo = st.text_input("MO / Work Order", key="sv_nc_mo")
        if st.button("Search NC", key="btn_search_nc"):
            st.session_state["NC_RESULTS"] = search_noncons(
                model=s_model, version=s_version, sn=s_sn, mo=s_mo
            )

    with st.expander("Search First Piece", expanded=False):
        s2_model = st.text_input("Model", key="sv_fp_model")
        s2_version = st.text_input("Model Version", key="sv_fp_version")
        s2_mo = st.text_input("MO / Work Order", key="sv_fp_mo")
        if st.button("Search First Piece", key="btn_search_fp"):
            st.session_state["FP_RESULTS"] = search_findings(
                model=s2_model, version=s2_version, mo=s2_mo
            )

    st.divider()

    # -------- CAPTURE FORMS --------
    st.header("✍️ Capture")

    # First Piece form
    with st.expander("📷 Record First Piece", expanded=False):
        with st.form("first_piece_form", clear_on_submit=True):
            fp_model = st.text_input("Model", key="fp_model")
            fp_version = st.text_input("Model Version (full)", key="fp_version")
            fp_sn = st.text_input("SN / Barcode", key="fp_sn")
            fp_mo = st.text_input("MO / Work Order", key="fp_mo")
            fp_desc = st.text_area("Notes (optional)", key="fp_desc")
            fp_imgs = st.file_uploader(
                "Upload photo(s) (Top/Bottom)",
                type=["jpg","jpeg","png"],
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
                            st.error(f"Failed saving {getattr(uf, 'name','image')}: {e}")
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
                        upsert_model(payload["model_no"], "")
                        st.success("Saved first piece.")

    # Non-Conformity form
    with st.expander("📝 Report Non-Conformity", expanded=False):
        vocab = load_vocab()
        with st.form("noncon_form", clear_on_submit=True):
            nc_model = st.text_input("Model", key="nc_model")
            nc_version = st.text_input("Model Version (full)", key="nc_version")
            nc_mo = st.text_input("MO / Work Order", key="nc_mo")
            nc_line = st.text_input("Line", key="nc_line")
            nc_station = st.text_input("Work Station", key="nc_station")
            nc_customer = st.text_input("Customer / Supplier", key="nc_customer")

            nc_type = st.selectbox("Nonconformity", options=vocab["nonconformity_types"], key="nc_type")
            nc_desc = st.text_area("Description of Nonconformity", key="nc_desc")

            origin = st.selectbox("Origin source", options=vocab["origins"], key="nc_origin")
            def_group = st.selectbox("Defective group", options=vocab["defective_groups"], key="nc_group")
            def_item  = st.selectbox("Defective item", options=vocab["defective_items"], key="nc_item")
            outflow   = st.selectbox("Defective outflow", options=vocab["outflows"], key="nc_outflow")

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
                type=["jpg","jpeg","png"],
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
                                st.error(f"Failed saving {getattr(uf,'name','image')}: {e}")
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
                    upsert_model(payload["model_no"], "")
                    st.success("Saved non-conformity.")

    # -------- Admin Lists --------
    st.header("⚙️ Admin")
    with st.expander("Controlled Lists", expanded=False):
        vocab = load_vocab()
        cA, cB = st.columns(2)
        with cA:
            t1 = st.text_area("Nonconformity types (comma-separated)",
                              value=", ".join(vocab["nonconformity_types"]), height=100, key="adm_nc_types")
            t2 = st.text_area("Origin sources (comma-separated)",
                              value=", ".join(vocab["origins"]), height=100, key="adm_origins")
            t3 = st.text_area("Defective outflows (comma-separated)",
                              value=", ".join(vocab["outflows"]), height=100, key="adm_outflows")
        with cB:
            t4 = st.text_area("Defective groups (comma-separated)",
                              value=", ".join(vocab["defective_groups"]), height=100, key="adm_groups")
            t5 = st.text_area("Defective items (comma-separated)",
                              value=", ".join(vocab["defective_items"]), height=100, key="adm_items")
        if st.button("Save lists", key="adm_save"):
            new_vocab = {
                "nonconformity_types": [s.strip() for s in t1.split(",") if s.strip()],
                "origins": [s.strip() for s in t2.split(",") if s.strip()],
                "outflows": [s.strip() for s in t3.split(",") if s.strip()],
                "defective_groups": [s.strip() for s in t4.split(",") if s.strip()],
                "defective_items": [s.strip() for s in t5.split(",") if s.strip()],
            }
            save_vocab(new_vocab)
            st.success("Lists saved.")

# ===================== MAIN — Results Rendering =====================

st.markdown("### 🔎 Results")

# Pull last searched results from session (if any)
nc_df: pd.DataFrame = st.session_state.get("NC_RESULTS", pd.DataFrame())
fp_df: pd.DataFrame = st.session_state.get("FP_RESULTS", pd.DataFrame())

# ---- Non-Conformities ----
with st.expander("🗂️ Non-Conformities (results)", expanded=not nc_df.empty):
    if nc_df.empty:
        st.info("Use the *Search Non-Conformities* form in the sidebar.")
    else:
        st.caption(f"{len(nc_df)} record(s)")
        for _, r in nc_df.iterrows():
            with st.container(border=True):
                cols = st.columns([1, 3])
                with cols[0]:
                    p = DATA_DIR / str(r.get("image_path","")) if r.get("image_path") else None
                    if p and p.exists():
                        - st.image(str(p), use_column_width=True)
                        + st.image(str(p), use_column_width=True)
                with cols[1]:
                    st.markdown(
                        f"**{r.get('nc_type','-')}**  |  Severity: **{r.get('severity','-')}**  "
                        f"|  MO: {r.get('mo','-')}  |  Line/Station: {r.get('line','-')}/{r.get('station','-')}"
                    )
                    st.caption(
                        f"{r.get('created_at','')} · Origin: {r.get('origin_source','-')} · "
                        f"Outflow: {r.get('outflow','-')} · Reporter: {r.get('reporter','')}"
                    )
                    desc = r.get("nc_desc","")
                    if isinstance(desc, str) and desc.strip():
                        st.write(desc)
                    st.caption(
                        f"Defective: {r.get('defect_qty',0)} | Inspected: {r.get('inspect_qty',0)} | Lot: {r.get('lot_qty',0)}"
                    )

        with st.expander("Show table"):
            st.dataframe(nc_df, use_container_width=True, hide_index=True)

        # Export results
        c1, c2 = st.columns(2)
        with c1:
            csv = nc_df.to_csv(index=False).encode("utf-8")
            st.download_button("Download CSV (NC results)", data=csv, file_name="nc_results.csv", mime="text/csv")
        with c2:
            bio = io.BytesIO()
            with pd.ExcelWriter(bio, engine="xlsxwriter") as xw:
                nc_df.to_excel(xw, index=False, sheet_name="NonCon")
            bio.seek(0)
            st.download_button(
                "Download Excel (NC results)",
                data=bio.getvalue(),
                file_name="nc_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

# ---- First Piece ----
with st.expander("🗂️ First Piece (results)", expanded=not fp_df.empty):
    if fp_df.empty:
        st.info("Use the *Search First Piece* form in the sidebar.")
    else:
        st.caption(f"{len(fp_df)} record(s)")
        for _, r in fp_df.iterrows():
            with st.container(border=True):
                cols = st.columns([1, 3])
                with cols[0]:
                    p = DATA_DIR / str(r.get("image_path","")) if r.get("image_path") else None
                    if p and p.exists():
                        - st.image(str(p), use_column_width=True)
                        + st.image(str(p), use_column_width=True)
                with cols[1]:
                    st.markdown(
                        f"**Version:** {r.get('model_version','-')}  |  **SN:** {r.get('sn','-')}  |  **MO:** {r.get('mo','-')}"
                    )
                    st.caption(f"{r.get('created_at','')} · Reporter: {r.get('reporter','')}")
                    desc = r.get("description", "")
                    if isinstance(desc, str) and desc.strip():
                        st.write(desc)

        with st.expander("Show table"):
            st.dataframe(fp_df, use_container_width=True, hide_index=True)

        # Export results
        c1, c2 = st.columns(2)
        with c1:
            csv = fp_df.to_csv(index=False).encode("utf-8")
            st.download_button("Download CSV (First Piece results)", data=csv, file_name="first_piece_results.csv", mime="text/csv")
        with c2:
            bio = io.BytesIO()
            with pd.ExcelWriter(bio, engine="xlsxwriter") as xw:
                fp_df.to_excel(xw, index=False, sheet_name="FirstPiece")
            bio.seek(0)
            st.download_button(
                "Download Excel (First Piece results)",
                data=bio.getvalue(),
                file_name="first_piece_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
