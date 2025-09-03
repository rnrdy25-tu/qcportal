# Quality Portal - Pilot
# First Piece + Non-Conformities (create, import, search, view)
# Images are stored under DATA_DIR/images/<model> and file paths are saved in DB

import os, io, json, sqlite3
from pathlib import Path
from datetime import datetime, date

import streamlit as st
import pandas as pd
from PIL import Image

# ============ Cloud-safe storage ============

def _pick_data_dir() -> Path:
    for base in (Path("/mount/data"), Path("/tmp/qc_portal")):
        try:
            base.mkdir(parents=True, exist_ok=True)
            (base / ".write_check").write_text("ok", encoding="utf-8")
            return base
        except Exception:
            pass
    raise RuntimeError("No writable directory found")

DATA_DIR = _pick_data_dir()
IMG_DIR  = DATA_DIR / "images"
DB_PATH  = DATA_DIR / "qc_portal.sqlite3"
IMG_DIR.mkdir(parents=True, exist_ok=True)

# ============ helpers ============

def now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds")

def current_user() -> str:
    # set admin name here (default: Admin1 if none)
    return os.environ.get("QC_USER", "Admin1")

def save_image(model_no: str, uploaded_file) -> str:
    """Save uploaded image; return relative path from DATA_DIR."""
    folder = IMG_DIR / (model_no or "_misc")
    folder.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    fn = f"{ts}_{uploaded_file.name.replace(' ', '_')}"
    out = folder / fn
    Image.open(uploaded_file).convert("RGB").save(out, format="JPEG", quality=90)
    return str(out.relative_to(DATA_DIR))

# ============ database ============

SCHEMA_MODELS = """
CREATE TABLE IF NOT EXISTS models(
  model_no TEXT PRIMARY KEY,
  name     TEXT
);
"""

# We keep “wide” fields in extra JSON so we can ingest more columns without altering tables.
SCHEMA_NONCONF = """
CREATE TABLE IF NOT EXISTS nonconformities(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at TEXT,
  model_no TEXT,
  model_version TEXT,
  sn TEXT,
  mo TEXT,
  reporter TEXT,
  severity TEXT,
  description TEXT,
  top_image_path TEXT,
  bottom_image_path TEXT,
  extra JSON
);
"""

SCHEMA_FIRST = """
CREATE TABLE IF NOT EXISTS first_piece(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at TEXT,
  model_no TEXT,
  model_version TEXT,
  sn TEXT,
  mo TEXT,
  reporter TEXT,
  notes TEXT,
  top_image_path TEXT,
  bottom_image_path TEXT,
  extra JSON
);
"""

def get_conn():
    return sqlite3.connect(DB_PATH)

def init_db():
    with get_conn() as c:
        c.execute(SCHEMA_MODELS)
        c.execute(SCHEMA_NONCONF)
        c.execute(SCHEMA_FIRST)
        c.commit()

@st.cache_data(show_spinner=False)
def list_models():
    with get_conn() as c:
        return pd.read_sql_query(
            "SELECT model_no, COALESCE(name,'') AS name FROM models ORDER BY model_no", c
        )

def upsert_model(model_no: str, name: str = ""):
    with get_conn() as c:
        c.execute(
            """INSERT INTO models(model_no, name) VALUES(?, ?)
               ON CONFLICT(model_no) DO UPDATE SET name=excluded.name""",
            (model_no.strip(), name.strip()),
        )
        c.commit()

def insert_first_piece(payload: dict):
    with get_conn() as c:
        c.execute(
            """INSERT INTO first_piece
               (created_at, model_no, model_version, sn, mo, reporter, notes,
                top_image_path, bottom_image_path, extra)
               VALUES(?,?,?,?,?,?,?,?,?,?)""",
            (
                payload.get("created_at"), payload.get("model_no"),
                payload.get("model_version"), payload.get("sn"),
                payload.get("mo"), payload.get("reporter"),
                payload.get("notes"), payload.get("top_image_path"),
                payload.get("bottom_image_path"),
                json.dumps(payload.get("extra") or {}, ensure_ascii=False),
            ),
        )
        c.commit()

def insert_nonconf(payload: dict):
    with get_conn() as c:
        c.execute(
            """INSERT INTO nonconformities
               (created_at, model_no, model_version, sn, mo, reporter, severity,
                description, top_image_path, bottom_image_path, extra)
               VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
            (
                payload.get("created_at"), payload.get("model_no"),
                payload.get("model_version"), payload.get("sn"),
                payload.get("mo"), payload.get("reporter"),
                payload.get("severity"), payload.get("description"),
                payload.get("top_image_path"), payload.get("bottom_image_path"),
                json.dumps(payload.get("extra") or {}, ensure_ascii=False),
            ),
        )
        c.commit()

def update_row_photo(table: str, row_id: int, where_slot: str, rel_path: str):
    col = "top_image_path" if where_slot == "top" else "bottom_image_path"
    with get_conn() as c:
        c.execute(f"UPDATE {table} SET {col}=? WHERE id=?", (rel_path, row_id))
        c.commit()

def delete_row(table: str, row_id: int):
    with get_conn() as c:
        c.execute(f"DELETE FROM {table} WHERE id=?", (row_id,))
        c.commit()

# --- loaders (cached) ---

@st.cache_data(show_spinner=False)
def load_first_piece(filters: dict):
    q = "SELECT * FROM first_piece WHERE 1=1"
    params = []
    if filters.get("model"):
        q += " AND model_no LIKE ?"; params.append(f"%{filters['model']}%")
    if filters.get("version"):
        q += " AND model_version LIKE ?"; params.append(f"%{filters['version']}%")
    if filters.get("sn"):
        q += " AND sn LIKE ?"; params.append(f"%{filters['sn']}%")
    if filters.get("mo"):
        q += " AND mo LIKE ?"; params.append(f"%{filters['mo']}%")
    if filters.get("text"):
        q += " AND (notes LIKE ? OR reporter LIKE ?)"
        params += [f"%{filters['text']}%", f"%{filters['text']}%"]
    if filters.get("date_from"): q += " AND date(created_at) >= ?"; params.append(filters["date_from"])
    if filters.get("date_to"):   q += " AND date(created_at) <= ?"; params.append(filters["date_to"])
    q += " ORDER BY id DESC"
    with get_conn() as c:
        return pd.read_sql_query(q, c, params=params)

@st.cache_data(show_spinner=False)
def load_nonconf(filters: dict):
    q = "SELECT * FROM nonconformities WHERE 1=1"
    params = []
    if filters.get("model"):
        q += " AND model_no LIKE ?"; params.append(f"%{filters['model']}%")
    if filters.get("version"):
        q += " AND model_version LIKE ?"; params.append(f"%{filters['version']}%")
    if filters.get("sn"):
        q += " AND sn LIKE ?"; params.append(f"%{filters['sn']}%")
    if filters.get("mo"):
        q += " AND mo LIKE ?"; params.append(f"%{filters['mo']}%")
    if filters.get("text"):
        q += " AND (description LIKE ? OR reporter LIKE ? OR severity LIKE ?)"
        params += [f"%{filters['text']}%", f"%{filters['text']}%", f"%{filters['text']}%"]
    if filters.get("date_from"): q += " AND date(created_at) >= ?"; params.append(filters["date_from"])
    if filters.get("date_to"):   q += " AND date(created_at) <= ?"; params.append(filters["date_to"])
    q += " ORDER BY id DESC"
    with get_conn() as c:
        return pd.read_sql_query(q, c, params=params)

# ============ UI ============

init_db()
st.set_page_config(page_title="Quality Portal - Pilot", layout="wide")

# subtle compacting CSS for filters + cards
st.markdown("""
<style>
.small * {font-size: 0.87rem !important;}
.card {padding: .6rem 1rem; border: 1px solid #e6e6e6; border-radius: 10px; margin-bottom: .8rem;}
.card h4 {margin: .2rem 0 .5rem 0;}
.card .caps {color:#57606a; font-weight:600;}
img {border-radius: 6px;}
</style>
""", unsafe_allow_html=True)

st.title("Quality Portal - Pilot")

# ---------- Sidebar: Create ----------

with st.sidebar:
    st.header("➕ Create / Upload")

    with st.expander("First Piece (TOP & BOTTOM)", expanded=False):
        f_m = st.text_input("Model")
        f_v = st.text_input("Model Version")
        f_sn = st.text_input("SN / Barcode")
        f_mo = st.text_input("MO / Work Order")
        f_notes = st.text_area("Notes")
        colu = st.columns(2)
        with colu[0]:
            f_top = st.file_uploader("TOP photo", type=["jpg","jpeg","png"], key="fp_top")
        with colu[1]:
            f_bot = st.file_uploader("BOTTOM photo", type=["jpg","jpeg","png"], key="fp_bot")
        if st.button("Save First Piece", use_container_width=True):
            if not f_m.strip():
                st.error("Model is required.")
            else:
                top_rel = save_image(f_m, f_top) if f_top else None
                bot_rel = save_image(f_m, f_bot) if f_bot else None
                insert_first_piece({
                    "created_at": now_iso(), "model_no": f_m, "model_version": f_v,
                    "sn": f_sn, "mo": f_mo, "reporter": current_user(),
                    "notes": f_notes, "top_image_path": top_rel, "bottom_image_path": bot_rel,
                    "extra": {}
                })
                st.success("Saved.")
                load_first_piece.clear()

    with st.expander("Non-Conformity (aligned to Excel)", expanded=False):
        n_model = st.text_input("Model/Part No.")
        n_version = st.text_input("Version (optional)")
        n_customer = st.text_input("Customer/Supplier")
        n_mo = st.text_input("MO/PO")
        n_sn = st.text_input("SN (optional)")
        n_line = st.text_input("Line")
        n_ws = st.text_input("Work Station")
        n_unit = st.text_input("Unit Head")
        n_resp = st.text_input("Responsibility")
        n_rc = st.text_input("Root Cause")
        n_ca = st.text_input("Corrective Action")
        n_sev = st.selectbox("Severity / Type", ["Minor","Major","Critical","Info"], index=0)
        n_desc = st.text_area("Description of Nonconformity")
        cc = st.columns(2)
        with cc[0]:
            n_top = st.file_uploader("TOP photo", type=["jpg","jpeg","png"], key="nc_top")
        with cc[1]:
            n_bot = st.file_uploader("BOTTOM photo", type=["jpg","jpeg","png"], key="nc_bot")
        if st.button("Save Non-Conformity", use_container_width=True):
            if not n_model.strip():
                st.error("Model/Part No. is required.")
            else:
                top_rel = save_image(n_model, n_top) if n_top else None
                bot_rel = save_image(n_model, n_bot) if n_bot else None
                insert_nonconf({
                    "created_at": now_iso(), "model_no": n_model, "model_version": n_version,
                    "sn": n_sn, "mo": n_mo, "reporter": current_user(), "severity": n_sev,
                    "description": n_desc, "top_image_path": top_rel, "bottom_image_path": bot_rel,
                    "extra": {
                        "Customer/Supplier": n_customer, "Line": n_line, "Work Station": n_ws,
                        "Unit Head": n_unit, "Responsibility": n_resp, "Root Cause": n_rc,
                        "Corrective Action": n_ca
                    }
                })
                st.success("Saved.")
                load_nonconf.clear()

    with st.expander("Import Non-Conformities (CSV/XLSX)", expanded=False):
        st.caption("Headers supported (any order): Nonconformity, Description of Nonconformity, Date, Customer/Supplier, Model/Part No., MO/PO, Line, Work Station, Unit Head, Responsibility, Root Cause, Corrective Action, Exception reporters, Discovery, Origil Sources, Defective Item, Defective Item (2), Defective Outflow, Defective Qty, Inspection Qty, Lot Qty")
        upf = st.file_uploader("Upload file", type=["csv","xlsx"])
        if upf is not None:
            try:
                if upf.name.lower().endswith(".xlsx"):
                    df = pd.read_excel(upf, engine="openpyxl")
                else:
                    # robust CSV decoding
                    _bytes = upf.getvalue()
                    for enc in ("utf-8-sig", "utf-8", "big5", "cp950"):
                        try:
                            df = pd.read_csv(io.BytesIO(_bytes), encoding=enc)
                            break
                        except Exception:
                            df = None
                    if df is None:
                        df = pd.read_csv(io.BytesIO(_bytes), encoding="utf-8", errors="ignore")
            except Exception as e:
                st.error(f"Import failed: {e}")
                df = None

            if df is not None and not df.empty:
                # Normalize columns (strip and lower)
                rename = {c.strip(): c.strip() for c in df.columns if isinstance(c, str)}
                df = df.rename(columns=rename)

                # mapping from your Excel headers to our fields / extra
                COLS = {
                    "Model/Part No.": "model_no",
                    "MO/PO": "mo",
                    "Customer/Supplier": ("extra", "Customer/Supplier"),
                    "Description of Nonconformity": "description",
                    "Nonconformity": ("extra","Nonconformity"),
                    "Date": "created_at",
                    "Line": ("extra","Line"),
                    "Work Station": ("extra","Work Station"),
                    "Unit Head": ("extra","Unit Head"),
                    "Responsibility": ("extra","Responsibility"),
                    "Root Cause": ("extra","Root Cause"),
                    "Corrective Action": ("extra","Corrective Action"),
                    "Exception reporters": ("extra","Exception reporters"),
                    "Discovery": ("extra","Discovery"),
                    "Origil Sources": ("extra","Origil Sources"),
                    "Defective Item": ("extra","Defective Item"),
                    "Defective Outflow": ("extra","Defective Outflow"),
                    "Defective Qty": ("extra","Defective Qty"),
                    "Inspection Qty": ("extra","Inspection Qty"),
                    "Lot Qty": ("extra","Lot Qty"),
                }

                imported = 0
                for _, row in df.iterrows():
                    payload = {
                        "created_at": None, "model_no": "", "model_version": "",
                        "sn": "", "mo": "", "reporter": current_user(), "severity": "",
                        "description": "", "top_image_path": None, "bottom_image_path": None,
                        "extra": {}
                    }
                    for col, key in COLS.items():
                        if col not in df.columns: 
                            continue
                        val = row.get(col, None)
                        if pd.isna(val): 
                            continue
                        if key == "created_at":
                            # normalize date
                            try:
                                payload["created_at"] = pd.to_datetime(val).strftime("%Y-%m-%d")
                            except Exception:
                                payload["created_at"] = str(val)
                        elif key == "model_no":
                            payload["model_no"] = str(val).strip()
                        elif key == "mo":
                            payload["mo"] = str(val).strip()
                        elif key == "description":
                            payload["description"] = str(val).strip()
                        elif isinstance(key, tuple) and key[0] == "extra":
                            payload["extra"][key[1]] = str(val)

                    if not payload["created_at"]:
                        payload["created_at"] = now_iso()
                    insert_nonconf(payload)
                    # also keep models list up-to-date
                    if payload["model_no"]:
                        upsert_model(payload["model_no"])
                    imported += 1

                st.success(f"Imported {imported} row(s).")
                load_nonconf.clear()

# ---------- Search & View ----------

st.subheader("🔎 Search & View")
with st.container():
    with st.container():
        st.markdown('<div class="small">', unsafe_allow_html=True)
        fc1, fc2, fc3, fc4, fc5, fc6, fc7 = st.columns([1,1,1,1,2,1,1])
        with fc1: f_model = st.text_input("Model contains")
        with fc2: f_version = st.text_input("Version contains")
        with fc3: f_sn = st.text_input("SN contains")
        with fc4: f_mo = st.text_input("MO contains")
        with fc5: f_text = st.text_input("Text in description/reporter/type")
        with fc6: f_from = st.date_input("From", value=None, format="YYYY-MM-DD")
        with fc7: f_to = st.date_input("To", value=None, format="YYYY-MM-DD")
        st.markdown('</div>', unsafe_allow_html=True)

filters = {
    "model": f_model.strip() if f_model else "",
    "version": f_version.strip() if f_version else "",
    "sn": f_sn.strip() if f_sn else "",
    "mo": f_mo.strip() if f_mo else "",
    "text": f_text.strip() if f_text else "",
    "date_from": str(f_from) if isinstance(f_from, date) else None,
    "date_to": str(f_to) if isinstance(f_to, date) else None,
}

# Tabs: First Piece / Nonconformities
t1, t2 = st.tabs(["First Piece", "Non-Conformities"])

def _render_photos(right_col, top_rel, bottom_rel):
    # draw both images side by side or show placeholders
    cc = right_col.columns(2, gap="small")
    if top_rel:
        p = DATA_DIR / top_rel
        if p.exists():
            cc[0].image(str(p), use_container_width=True, caption="TOP")
    if bottom_rel:
        p = DATA_DIR / bottom_rel
        if p.exists():
            cc[1].image(str(p), use_container_width=True, caption="BOTTOM")

with t1:
    fdf = load_first_piece(filters)
    st.caption(f"{len(fdf)} record(s)")
    for _, r in fdf.iterrows():
        with st.container():
            st.markdown('<div class="card">', unsafe_allow_html=True)
            # compact header
            st.markdown(
                f"**Model:** {r['model_no'] or '-'} | **Version:** {r['model_version'] or '-'} | "
                f"**SN:** {r['sn'] or '-'} | **MO:** {r['mo'] or '-'}"
            )
            st.caption(f"🕒 {r['created_at']}   👤 {r['reporter']}")
            cols = st.columns([3,2], gap="large")
            with cols[0]:
                st.write(r["notes"] or "")
            with cols[1]:
                _render_photos(cols[1], r.get("top_image_path"), r.get("bottom_image_path"))
                # Add photo later
                ap = st.file_uploader("Add photo", type=["jpg","jpeg","png"], key=f"fp_add_{r['id']}")
                if ap is not None:
                    rel = save_image(r["model_no"] or "_misc", ap)
                    # put to TOP if empty else to BOTTOM
                    slot = "top" if not r.get("top_image_path") else "bottom"
                    update_row_photo("first_piece", int(r["id"]), slot, rel)
                    st.success("Photo added.")
                    load_first_piece.clear()
            col2 = st.columns([6,1])
            with col2[1]:
                if st.button("Delete", key=f"del_fp_{r['id']}", use_container_width=True):
                    delete_row("first_piece", int(r["id"]))
                    load_first_piece.clear()
            st.markdown('</div>', unsafe_allow_html=True)

    with st.expander("Table view & export (First Piece)", expanded=False):
        if not fdf.empty:
            st.dataframe(fdf.drop(columns=["extra"]), use_container_width=True, hide_index=True)
            st.download_button("Export CSV", fdf.to_csv(index=False).encode("utf-8"),
                               "first_piece.csv", "text/csv")

with t2:
    ndf = load_nonconf(filters)
    st.caption(f"{len(ndf)} record(s)")
    for _, r in ndf.iterrows():
        with st.container():
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown(
                f"**Model:** {r['model_no'] or '-'} | **Version:** {r['model_version'] or '-'} | "
                f"**SN:** {r['sn'] or '-'} | **MO:** {r['mo'] or '-'}"
            )
            sev = r.get("severity") or ""
            st.caption(f"🕒 {r['created_at']}   👤 {r['reporter']}   ·   {sev}")
            cols = st.columns([3,2], gap="large")
            with cols[0]:
                st.write(r["description"] or "")
                # show a few important extras if present
                try:
                    ex = json.loads(r.get("extra") or "{}")
                except Exception:
                    ex = {}
                highlights = []
                for key in ("Customer/Supplier","Line","Work Station","Unit Head",
                            "Responsibility","Root Cause"):
                    if ex.get(key):
                        highlights.append(f"**{key}:** {ex[key]}")
                if highlights:
                    st.markdown("  \n".join(highlights))
            with cols[1]:
                _render_photos(cols[1], r.get("top_image_path"), r.get("bottom_image_path"))
                ap2 = st.file_uploader("Add photo", type=["jpg","jpeg","png"], key=f"nc_add_{r['id']}")
                if ap2 is not None:
                    rel = save_image(r["model_no"] or "_misc", ap2)
                    slot = "top" if not r.get("top_image_path") else "bottom"
                    update_row_photo("nonconformities", int(r["id"]), slot, rel)
                    st.success("Photo added.")
                    load_nonconf.clear()
            col2 = st.columns([6,1])
            with col2[1]:
                if st.button("Delete", key=f"del_nc_{r['id']}", use_container_width=True):
                    delete_row("nonconformities", int(r["id"]))
                    load_nonconf.clear()
            st.markdown('</div>', unsafe_allow_html=True)

    with st.expander("Table view & export (Non-Conformities)", expanded=False):
        if not ndf.empty:
            st.dataframe(ndf.drop(columns=["extra"]), use_container_width=True, hide_index=True)
            st.download_button("Export CSV", ndf.to_csv(index=False).encode("utf-8"),
                               "nonconformities.csv", "text/csv")
