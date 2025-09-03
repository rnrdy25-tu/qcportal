# app.py  — Quality Portal - Pilot
# Streamlit app: models, first-piece, non-conformities, search, import, and export
# Storage is cloud-safe: /mount/data if present, else /tmp/qc_portal

import os
import io
import json
import sqlite3
from pathlib import Path
from datetime import datetime, date

import streamlit as st
import pandas as pd
from PIL import Image

# ──────────────────────────────────────────────────────────────────────────────
# Safe storage
# ──────────────────────────────────────────────────────────────────────────────
def _pick_data_dir() -> Path:
    for base in (Path("/mount/data"), Path("/tmp/qc_portal")):
        try:
            base.mkdir(parents=True, exist_ok=True)
            (base / ".write_ok").write_text("ok", encoding="utf-8")
            return base
        except Exception:
            pass
    raise RuntimeError("No writable directory found")

DATA_DIR = _pick_data_dir()
IMG_DIR  = DATA_DIR / "images"       # /images/<bucket>/<model>/...
DB_PATH  = DATA_DIR / "qc_portal.sqlite3"
IMG_DIR.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds")

def cur_user() -> str:
    # Best-effort Streamlit user OR default admin
    return os.environ.get("USERNAME") or os.environ.get("USER") or "Admin1"

def _save_image(bucket: str, model_no: str, uploaded_file) -> str:
    """
    Save under images/<bucket>/<model_no> and return PATH RELATIVE TO DATA_DIR.
    """
    clean_name = uploaded_file.name.replace(" ", "_")
    folder = IMG_DIR / bucket / (model_no or "unknown")
    folder.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    out_path = folder / f"{ts}_{clean_name}"
    img = Image.open(uploaded_file).convert("RGB")
    img.save(out_path, format="JPEG", quality=90)
    return str(out_path.relative_to(DATA_DIR))  # stored as relative path

def _path_if_exists(rel: str | None) -> Path | None:
    if not rel:
        return None
    p = DATA_DIR / rel
    return p if p.exists() else None

# ──────────────────────────────────────────────────────────────────────────────
# Database
# ──────────────────────────────────────────────────────────────────────────────
SCHEMA_MODELS = """
CREATE TABLE IF NOT EXISTS models(
  model_no TEXT PRIMARY KEY,
  name     TEXT
);
"""

SCHEMA_FIRSTPIECE = """
CREATE TABLE IF NOT EXISTS firstpiece(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at   TEXT,
  model_no     TEXT,
  model_version TEXT,
  sn           TEXT,
  mo           TEXT,
  reporter     TEXT,
  desc_text    TEXT,
  top_path     TEXT,
  bottom_path  TEXT,
  extra        JSON
);
"""

SCHEMA_NONCONF = """
CREATE TABLE IF NOT EXISTS nonconf(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at   TEXT,       -- date/time (from file or now)
  model_no     TEXT,       -- short model code
  model_version TEXT,      -- full model/part no
  mo           TEXT,       -- MO/PO
  sn           TEXT,       -- Serial/Barcode
  customer     TEXT,       -- customer/supplier
  line         TEXT,
  workstation  TEXT,
  unit_head    TEXT,
  nc_type      TEXT,       -- category/type
  description  TEXT,       -- description of nonconformity
  root_cause   TEXT,
  corrective_action TEXT,
  responsibility TEXT,
  defect_outflow TEXT,
  defect_item  TEXT,
  defect_qty   TEXT,
  inspection_qty TEXT,
  lot_qty      TEXT,
  reporter     TEXT,
  photo_path   TEXT,
  extra        JSON        -- JSON for unmapped columns
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
               VALUES(?, ?)
               ON CONFLICT(model_no) DO UPDATE SET name=excluded.name""",
            (model_no.strip(), name.strip()),
        )
        c.commit()

# First piece CRUD
def insert_firstpiece(payload: dict):
    fields = [
        "created_at","model_no","model_version","sn","mo",
        "reporter","desc_text","top_path","bottom_path","extra"
    ]
    values = [payload.get(k) for k in fields]
    with get_conn() as c:
        c.execute(
            f"INSERT INTO firstpiece({','.join(fields)}) VALUES(?,?,?,?,?,?,?,?,?,?)",
            values,
        )
        c.commit()

def delete_firstpiece(fid: int):
    with get_conn() as c:
        c.execute("DELETE FROM firstpiece WHERE id=?", (fid,))
        c.commit()

@st.cache_data(show_spinner=False)
def load_firstpiece_df(
    model=None, version=None, sn=None, mo=None,
    text=None, date_from=None, date_to=None
) -> pd.DataFrame:
    q = "SELECT * FROM firstpiece WHERE 1=1"
    params: list = []
    if model:
        q += " AND model_no LIKE ?"; params += [f"%{model}%"]
    if version:
        q += " AND model_version LIKE ?"; params += [f"%{version}%"]
    if sn:
        q += " AND sn LIKE ?"; params += [f"%{sn}%"]
    if mo:
        q += " AND mo LIKE ?"; params += [f"%{mo}%"]
    if text:
        q += " AND (desc_text LIKE ? OR reporter LIKE ?)"; params += [f"%{text}%", f"%{text}%"]
    if date_from:
        q += " AND date(substr(created_at,1,10)) >= date(?)"; params += [str(date_from)]
    if date_to:
        q += " AND date(substr(created_at,1,10)) <= date(?)"; params += [str(date_to)]
    q += " ORDER BY id DESC"
    with get_conn() as c:
        return pd.read_sql_query(q, c, params=params)

# Nonconformities CRUD
def insert_nonconf(payload: dict):
    fields = [
        "created_at","model_no","model_version","mo","sn","customer","line","workstation",
        "unit_head","nc_type","description","root_cause","corrective_action",
        "responsibility","defect_outflow","defect_item","defect_qty",
        "inspection_qty","lot_qty","reporter","photo_path","extra"
    ]
    values = [payload.get(k) for k in fields]
    with get_conn() as c:
        c.execute(
            f"INSERT INTO nonconf({','.join(fields)}) VALUES({','.join(['?']*len(fields))})",
            values,
        )
        c.commit()

def update_nonconf_photo(nid: int, rel_path: str):
    with get_conn() as c:
        c.execute("UPDATE nonconf SET photo_path=? WHERE id=?", (rel_path, nid))
        c.commit()

def delete_nonconf(nid: int):
    with get_conn() as c:
        c.execute("DELETE FROM nonconf WHERE id=?", (nid,))
        c.commit()

@st.cache_data(show_spinner=False)
def load_nonconf_df(
    model=None, version=None, sn=None, mo=None,
    text=None, date_from=None, date_to=None
) -> pd.DataFrame:
    q = "SELECT * FROM nonconf WHERE 1=1"
    params: list = []
    if model:
        q += " AND model_no LIKE ?"; params += [f"%{model}%"]
    if version:
        q += " AND model_version LIKE ?"; params += [f"%{version}%"]
    if sn:
        q += " AND sn LIKE ?"; params += [f"%{sn}%"]
    if mo:
        q += " AND mo LIKE ?"; params += [f"%{mo}%"]
    if text:
        q += " AND (description LIKE ? OR reporter LIKE ? OR nc_type LIKE ?)"
        params += [f"%{text}%", f"%{text}%", f"%{text}%"]
    if date_from:
        q += " AND date(substr(created_at,1,10)) >= date(?)"; params += [str(date_from)]
    if date_to:
        q += " AND date(substr(created_at,1,10)) <= date(?)"; params += [str(date_to)]
    q += " ORDER BY id DESC"
    with get_conn() as c:
        return pd.read_sql_query(q, c, params=params)

# ──────────────────────────────────────────────────────────────────────────────
# CSV/XLSX column mapping for Nonconformities
# (add more aliases as needed; matching is case-insensitive substring)
# ──────────────────────────────────────────────────────────────────────────────
ALIASES = {
    "created_at": ["date", "日期"],
    "nc_type": ["nonconformity", "不符合分類"],
    "description": ["description of nonconformity", "不符合說明"],
    "customer": ["customer", "supplier", "客戶", "供應商"],
    "model_version": ["model/part no", "機種/料號"],
    "mo": ["mo/po", "工單/採購單號"],
    "defect_qty": ["defective qty", "不良數量"],
    "inspection_qty": ["inspection qty", "檢驗數量"],
    "lot_qty": ["lot qty", "工單數量"],
    "line": ["line", "線別"],
    "workstation": ["work station", "作業站別"],
    "unit_head": ["unit head", "部門 負責人"],
    "responsibility": ["責任者", "responsibility"],
    "root_cause": ["root cause", "原因分析"],
    "corrective_action": ["corrective action", "矯正預防措施及對策"],
    "defect_outflow": ["defective outflow", "不良後流"],
    "defect_item": ["defective item", "異常項目", "異常分類"],
}

def _map_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c: c for c in df.columns}
    lower = [c.lower() for c in df.columns]

    mapped = {}
    for dest, aliases in ALIASES.items():
        idx = None
        for a in aliases:
            a_low = a.lower()
            for i, lc in enumerate(lower):
                if a_low in lc:
                    idx = i; break
            if idx is not None:
                break
        if idx is not None:
            mapped[dest] = df.columns[idx]

    # Always present key columns even if missing
    for must in [
        "created_at","nc_type","description","customer","model_version","mo",
        "defect_qty","inspection_qty","lot_qty","line","workstation","unit_head",
        "responsibility","root_cause","corrective_action","defect_outflow","defect_item"
    ]:
        if must not in mapped:
            df[must] = ""

    # Build normalized frame
    out = pd.DataFrame()
    for dest in [
        "created_at","nc_type","description","customer","model_version","mo",
        "defect_qty","inspection_qty","lot_qty","line","workstation","unit_head",
        "responsibility","root_cause","corrective_action","defect_outflow","defect_item"
    ]:
        if dest in mapped:
            out[dest] = df[mapped[dest]]
        else:
            out[dest] = df[dest]  # the blank we added

    return out

# ──────────────────────────────────────────────────────────────────────────────
# UI — App
# ──────────────────────────────────────────────────────────────────────────────
init_db()
st.set_page_config(page_title="Quality Portal - Pilot", layout="wide")
st.markdown(
    """
    <style>
      /* make Search & filters slightly smaller, tidy cards */
      section { font-size: 0.95rem; }
      .small-input input {height:2rem; font-size:0.9rem;}
      .small-select div[data-baseweb="select"] {font-size:0.9rem;}
      .card {border:1px solid #e6e6e6; padding:1rem; border-radius:.6rem; margin-bottom:1rem;}
      .muted {color:#666;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Quality Portal - Pilot")

# ─────────── Sidebar: create/update & import ───────────
with st.sidebar:
    st.header("Create / Upload")

    # Add / Update Model
    with st.expander("Add / Update Model", expanded=False):
        ms, mn = st.text_input("Model (short form)", key="mdl"), st.text_input("Name / Customer (optional)", key="mname")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Save model", key="save_model_btn"):
                if ms.strip():
                    upsert_model(ms, mn)
                    list_models_df.clear()
                    st.success("Model saved.")
                else:
                    st.error("Model cannot be empty.")
        with c2:
            if st.button("Delete model", key="del_model_btn"):
                if ms.strip():
                    with get_conn() as c:
                        c.execute("DELETE FROM models WHERE model_no=?", (ms.strip(),))
                        c.commit()
                    list_models_df.clear()
                    st.warning("Model deleted (if existed).")
                else:
                    st.error("Enter a model first.")

    # First Piece (TOP/BOTTOM)
    with st.expander("New First Piece", expanded=False):
        fp_model = st.text_input("Model", key="fp_model")
        fp_ver   = st.text_input("Model Version (full)", key="fp_ver")
        fp_sn    = st.text_input("SN / Barcode", key="fp_sn")
        fp_mo    = st.text_input("MO / Work Order", key="fp_mo")
        fp_desc  = st.text_area("Notes (optional)", key="fp_desc")

        tcol, bcol = st.columns(2)
        with tcol:
            up_top = st.file_uploader("Upload TOP photo", type=["jpg","jpeg","png"], key="fp_top")
        with bcol:
            up_bottom = st.file_uploader("Upload BOTTOM photo", type=["jpg","jpeg","png"], key="fp_bottom")

        if st.button("Save first piece", key="fp_save_btn"):
            if not fp_model.strip():
                st.error("Model is required.")
            else:
                rel_top = _save_image("firstpiece", fp_model, up_top) if up_top else None
                rel_bot = _save_image("firstpiece", fp_model, up_bottom) if up_bottom else None
                payload = dict(
                    created_at=now_iso(),
                    model_no=fp_model.strip(),
                    model_version=fp_ver.strip(),
                    sn=fp_sn.strip(),
                    mo=fp_mo.strip(),
                    reporter=cur_user(),
                    desc_text=fp_desc.strip(),
                    top_path=rel_top,
                    bottom_path=rel_bot,
                    extra=json.dumps({}, ensure_ascii=False),
                )
                insert_firstpiece(payload)
                load_firstpiece_df.clear()
                st.success("First piece saved.")

    # New Non-Conformity (manual)
    with st.expander("New Non-Conformity", expanded=False):
        nc_model = st.text_input("Model (short, optional)", key="nc_model")
        nc_ver   = st.text_input("Model Version (full)", key="nc_ver")
        nc_sn    = st.text_input("SN / Barcode", key="nc_sn")
        nc_mo    = st.text_input("MO / Work Order", key="nc_mo")
        nc_cust  = st.text_input("Customer/Supplier", key="nc_cust")
        nc_line  = st.text_input("Line", key="nc_line")
        nc_ws    = st.text_input("Work Station", key="nc_ws")
        nc_unit  = st.text_input("Unit Head", key="nc_unit")
        nc_type  = st.text_input("Nonconformity Type", key="nc_type")
        nc_desc  = st.text_area("Description of Nonconformity", key="nc_desc")
        nc_root  = st.text_area("Root Cause", key="nc_root")
        nc_ca    = st.text_area("Corrective Action", key="nc_ca")
        nc_resp  = st.text_input("Responsibility", key="nc_resp")
        nc_out   = st.text_input("Defective Outflow", key="nc_out")
        nc_item  = st.text_input("Defective Item", key="nc_item")
        n_dq, n_iq, n_lq = st.text_input("Defective Qty", key="n_dq"), st.text_input("Inspection Qty", key="n_iq"), st.text_input("Lot Qty", key="n_lq")
        up_nc_photo = st.file_uploader("Photo (optional)", type=["jpg","jpeg","png"], key="nc_photo")

        if st.button("Save non-conformity", key="nc_save_btn"):
            rel = _save_image("nonconf", nc_model or "unknown", up_nc_photo) if up_nc_photo else None
            payload = dict(
                created_at=now_iso(),
                model_no=nc_model.strip(),
                model_version=nc_ver.strip(),
                mo=nc_mo.strip(),
                sn=nc_sn.strip(),
                customer=nc_cust.strip(),
                line=nc_line.strip(),
                workstation=nc_ws.strip(),
                unit_head=nc_unit.strip(),
                nc_type=nc_type.strip(),
                description=nc_desc.strip(),
                root_cause=nc_root.strip(),
                corrective_action=nc_ca.strip(),
                responsibility=nc_resp.strip(),
                defect_outflow=nc_out.strip(),
                defect_item=nc_item.strip(),
                defect_qty=n_dq.strip(),
                inspection_qty=n_iq.strip(),
                lot_qty=n_lq.strip(),
                reporter=cur_user(),
                photo_path=rel,
                extra=json.dumps({}, ensure_ascii=False),
            )
            insert_nonconf(payload)
            load_nonconf_df.clear()
            st.success("Non-conformity saved.")

    # Import (CSV/XLSX → Non-Conformities)
    with st.expander("Import Non-Conformities (CSV/XLSX)", expanded=False):
        f = st.file_uploader("Upload CSV or Excel", type=["csv","xlsx","xls"], key="imp_file")
        imp_model_hint = st.text_input("Default model short code (optional)", key="imp_model")
        if f is not None:
            try:
                if f.name.lower().endswith((".xlsx",".xls")):
                    df_raw = pd.read_excel(f)
                else:
                    # auto encoding
                    df_raw = pd.read_csv(f, encoding="utf-8-sig")
            except Exception:
                f.seek(0)
                df_raw = pd.read_csv(f, encoding="latin1")

            df = _map_columns(df_raw.copy())
            st.caption("Column mapping preview (first 5 rows):")
            st.dataframe(df.head(), use_container_width=True, hide_index=True)

            if st.button("Import now", key="import_nc_btn"):
                count = 0
                for _, row in df.iterrows():
                    # Parse created_at if possible
                    c_at = str(row.get("created_at","")).strip()
                    if not c_at:
                        created_at = now_iso()
                    else:
                        try:
                            # accept yyyy-mm-dd or excel dates
                            created_at = pd.to_datetime(c_at).isoformat(timespec="seconds")
                        except Exception:
                            created_at = now_iso()

                    payload = dict(
                        created_at=created_at,
                        model_no=(imp_model_hint or "").strip(),
                        model_version=str(row.get("model_version","") or ""),
                        mo=str(row.get("mo","") or ""),
                        sn="",  # often absent in nonconf sheet
                        customer=str(row.get("customer","") or ""),
                        line=str(row.get("line","") or ""),
                        workstation=str(row.get("workstation","") or ""),
                        unit_head=str(row.get("unit_head","") or ""),
                        nc_type=str(row.get("nc_type","") or ""),
                        description=str(row.get("description","") or ""),
                        root_cause=str(row.get("root_cause","") or ""),
                        corrective_action=str(row.get("corrective_action","") or ""),
                        responsibility=str(row.get("responsibility","") or ""),
                        defect_outflow=str(row.get("defect_outflow","") or ""),
                        defect_item=str(row.get("defect_item","") or ""),
                        defect_qty=str(row.get("defect_qty","") or ""),
                        inspection_qty=str(row.get("inspection_qty","") or ""),
                        lot_qty=str(row.get("lot_qty","") or ""),
                        reporter=cur_user(),
                        photo_path=None,
                        extra=json.dumps({}, ensure_ascii=False),
                    )
                    insert_nonconf(payload)
                    count += 1

                load_nonconf_df.clear()
                st.success(f"Imported {count} records.")

# ─────────── Search & View (center) ───────────
st.subheader("🔎 Search & View")

with st.expander("Filters", expanded=True):
    c1,c2,c3,c4,c5 = st.columns([1,1,1,1,2])
    f_model   = c1.text_input("Model contains", key="f_model", placeholder="e.g. 190-")
    f_ver     = c2.text_input("Version contains", key="f_ver")
    f_sn      = c3.text_input("SN contains", key="f_sn")
    f_mo      = c4.text_input("MO contains", key="f_mo")
    f_text    = c5.text_input("Text in description/reporter/type", key="f_text")

    d1, d2 = st.columns(2)
    f_from  = d1.date_input("From (date)", value=None, key="f_from")
    f_to    = d2.date_input("To (date)", value=None, key="f_to")

# FIRST PIECE results
with st.expander("📁 First Piece (results)", expanded=True):
    fdf = load_firstpiece_df(
        model=f_model or None, version=f_ver or None, sn=f_sn or None, mo=f_mo or None,
        text=f_text or None, date_from=f_from or None, date_to=f_to or None
    )
    st.caption(f"{len(fdf)} record(s)")
    for _, r in fdf.iterrows():
        with st.container():
            st.markdown(
                f"**Model:** {r['model_no'] or '-'}  |  **Version:** {r['model_version'] or '-'}"
                f"  |  **SN:** {r['sn'] or '-'}  |  **MO:** {r['mo'] or '-'}"
            )
            meta = f"📅 {r['created_at']} &nbsp;&nbsp; 👤 Reporter: {r['reporter'] or '-'}"
            st.markdown(f"<span class='muted'>{meta}</span>", unsafe_allow_html=True)

            cc = st.columns(2)
            p_top = _path_if_exists(r.get("top_path"))
            p_bot = _path_if_exists(r.get("bottom_path"))
            with cc[0]:
                if p_top:
                    st.image(str(p_top), use_container_width=True, caption="TOP")
            with cc[1]:
                if p_bot:
                    st.image(str(p_bot), use_container_width=True, caption="BOTTOM")

            if r.get("desc_text"):
                st.write(r["desc_text"])

            if st.button("Delete", key=f"del_fp_{r['id']}"):
                delete_firstpiece(int(r["id"]))
                load_firstpiece_df.clear()
                st.experimental_rerun()

# NON-CONFORMITIES results
with st.expander("📁 Non-Conformities (results)", expanded=True):
    ndf = load_nonconf_df(
        model=f_model or None, version=f_ver or None, sn=f_sn or None, mo=f_mo or None,
        text=f_text or None, date_from=f_from or None, date_to=f_to or None
    )
    st.caption(f"{len(ndf)} record(s)")

    # Cards
    for _, r in ndf.iterrows():
        with st.container():
            st.markdown(
                f"**Model:** {r['model_no'] or '-'} | **Version:** {r['model_version'] or '-'}"
                f" | **SN:** {r['sn'] or '-'} | **MO:** {r['mo'] or '-'}"
            )
            meta = (
                f"📅 {r['created_at'] or '-'} &nbsp;&nbsp; 👤 {r['reporter'] or '-'} "
                f"&nbsp;&nbsp; 📦 {r['defect_item'] or '-'}"
            )
            st.markdown(f"<span class='muted'>{meta}</span>", unsafe_allow_html=True)

            cimg, ctxt = st.columns([1,2])
            with cimg:
                pp = _path_if_exists(r.get("photo_path"))
                if pp:
                    st.image(str(pp), use_container_width=True)
                else:
                    up = st.file_uploader(" + Photo", type=["jpg","jpeg","png"], key=f"addph_{r['id']}")
                    if up is not None:
                        rel = _save_image("nonconf", r.get("model_no") or "unknown", up)
                        update_nonconf_photo(int(r["id"]), rel)
                        load_nonconf_df.clear()
                        st.experimental_rerun()

            with ctxt:
                # show all important fields (and anything else in extra JSON)
                lines = [
                    ("Type", r.get("nc_type")),
                    ("Customer", r.get("customer")),
                    ("Line", r.get("line")),
                    ("Work Station", r.get("workstation")),
                    ("Unit Head", r.get("unit_head")),
                    ("Responsibility", r.get("responsibility")),
                    ("Root Cause", r.get("root_cause")),
                    ("Corrective Action", r.get("corrective_action")),
                    ("Defective Outflow", r.get("defect_outflow")),
                    ("Defective Item", r.get("defect_item")),
                    ("Defective Qty", r.get("defect_qty")),
                    ("Inspection Qty", r.get("inspection_qty")),
                    ("Lot Qty", r.get("lot_qty")),
                ]
                if r.get("description"):
                    st.write(r["description"])
                for label, val in lines:
                    if val:
                        st.caption(f"**{label}:** {val}")

            cact1, cact2 = st.columns([1,8])
            with cact1:
                if st.button("Delete", key=f"del_nc_{r['id']}"):
                    delete_nonconf(int(r["id"]))
                    load_nonconf_df.clear()
                    st.experimental_rerun()

    # Table + Export
    with st.expander("Table view & export"):
        st.dataframe(ndf, use_container_width=True, hide_index=True)
        csv = ndf.to_csv(index=False).encode("utf-8-sig")
        st.download_button("Download CSV", data=csv, file_name="nonconformities_export.csv", mime="text/csv")
