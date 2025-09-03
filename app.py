# app.py — Quality Portal - Pilot (safe image rendering)
import io
import os
import json
import hashlib
import sqlite3
from pathlib import Path
from datetime import datetime, date

import pandas as pd
from PIL import Image
import streamlit as st

# --------------------------- Storage & helpers ---------------------------

def pick_data_dir() -> Path:
    for base in (Path("/mount/data"), Path("/tmp/qc_portal")):
        try:
            base.mkdir(parents=True, exist_ok=True)
            (base / ".write_ok").write_text("ok", encoding="utf-8")
            return base
        except Exception:
            pass
    raise RuntimeError("No writable directory")

DATA_DIR = pick_data_dir()
IMG_DIR  = DATA_DIR / "images"
DB_PATH  = DATA_DIR / "qc_portal.sqlite3"
IMG_DIR.mkdir(parents=True, exist_ok=True)

def now_iso():
    return datetime.utcnow().isoformat(timespec="seconds")

def current_user() -> str:
    return st.session_state.get("whoami") or os.getenv("USER") or os.getenv("USERNAME") or "appuser"

def save_image_to(model_no: str, uploaded_file) -> str:
    folder = IMG_DIR / (model_no or "_misc")
    folder.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    out = folder / f"{ts}_{uploaded_file.name.replace(' ', '_')}"
    im = Image.open(uploaded_file).convert("RGB")
    im.save(out, "JPEG", quality=90)
    return str(out.relative_to(DATA_DIR))

# SAFE image preview
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}

def safe_show_image(path: Path, caption: str | None = None):
    try:
        if path and path.is_file() and path.suffix.lower() in IMG_EXTS:
            st.image(str(path), use_container_width=True, caption=caption)
        else:
            # Optional: show a light placeholder so users know no preview is available
            if caption:
                st.caption(f"🖼️ {caption}: (no preview)")
    except Exception as e:
        st.caption(f"⚠️ Unable to preview image ({e}).")
        
# ---- SAFE UI HELPERS -------------------------------------------------

def safe_dataframe(df):
    """Render a DataFrame without crashing if some args are unsupported."""
    try:
        st.dataframe(df, use_container_width=True, hide_index=True)
    except TypeError:
        # older/newer Streamlit that doesn't support hide_index
        st.dataframe(df, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not render table: {e}")

def download_csv_button(df, label, filename):
    """Offer a CSV download and make sure we always pass bytes."""
    try:
        data_bytes = df.to_csv(index=False).encode("utf-8")
    except Exception as e:
        st.warning(f"Export failed: {e}")
        return
    try:
        st.download_button(label, data=data_bytes, file_name=filename, mime="text/csv")
    except Exception as e:
        st.warning(f"Download button failed: {e}")
        
# --------------------------- Database ---------------------------

SCHEMA = {
    "models": """
        CREATE TABLE IF NOT EXISTS models(
            model_no TEXT PRIMARY KEY,
            name     TEXT
        );
    """,
    "first_piece": """
        CREATE TABLE IF NOT EXISTS first_piece(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT,
            model_no TEXT,
            model_version TEXT,
            sn TEXT,
            mo TEXT,
            reporter TEXT,
            note TEXT,
            top_image_path TEXT,
            bottom_image_path TEXT,
            extra JSON
        );
    """,
    "nonconfs": """
        CREATE TABLE IF NOT EXISTS nonconfs(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT,
            model_no TEXT,
            model_version TEXT,
            sn TEXT,
            mo TEXT,
            reporter TEXT,
            severity TEXT,
            description TEXT,
            image_path TEXT,
            extra JSON
        );
    """
}

def conn():
    return sqlite3.connect(DB_PATH)

def init_db():
    with conn() as c:
        for ddl in SCHEMA.values():
            c.execute(ddl)
        c.commit()

def upsert_model(model_no: str, name: str = ""):
    if not model_no:
        return
    with conn() as c:
        c.execute(
            """INSERT INTO models(model_no, name)
               VALUES(?, ?)
               ON CONFLICT(model_no) DO UPDATE SET name=excluded.name""",
            (model_no.strip(), (name or "").strip()),
        )
        c.commit()

def insert_first_piece(p):
    fields = ["created_at","model_no","model_version","sn","mo","reporter","note",
              "top_image_path","bottom_image_path","extra"]
    vals = [p.get(k) for k in fields]
    with conn() as c:
        c.execute(f"INSERT INTO first_piece({','.join(fields)}) VALUES(?,?,?,?,?,?,?,?,?,?)", vals)
        c.commit()

def insert_nonconf(p):
    fields = ["created_at","model_no","model_version","sn","mo","reporter","severity",
              "description","image_path","extra"]
    vals = [p.get(k) for k in fields]
    with conn() as c:
        c.execute(f"INSERT INTO nonconfs({','.join(fields)}) VALUES(?,?,?,?,?,?,?,?,?,?)", vals)
        c.commit()

@st.cache_data(show_spinner=False)
def read_first_piece(filters: dict, limit: int) -> pd.DataFrame:
    q = "SELECT * FROM first_piece WHERE 1=1"
    params = []
    if filters.get("from"):
        q += " AND date(created_at) >= date(?)"; params.append(filters["from"])
    if filters.get("to"):
        q += " AND date(created_at) <= date(?)"; params.append(filters["to"])
    for field in ("model_no","model_version","sn","mo","reporter","note"):
        val = filters.get(field)
        if val:
            q += f" AND {field} LIKE ?"; params.append(f"%{val}%")
    q += " ORDER BY id DESC LIMIT ?"; params.append(limit)
    with conn() as c:
        return pd.read_sql_query(q, c, params=params)

@st.cache_data(show_spinner=False)
def read_nonconfs(filters: dict, limit: int) -> pd.DataFrame:
    q = "SELECT * FROM nonconfs WHERE 1=1"
    params = []
    if filters.get("from"):
        q += " AND date(created_at) >= date(?)"; params.append(filters["from"])
    if filters.get("to"):
        q += " AND date(created_at) <= date(?)"; params.append(filters["to"])
    for field in ("model_no","model_version","sn","mo","reporter","severity","description"):
        val = filters.get(field)
        if val:
            q += f" AND {field} LIKE ?"; params.append(f"%{val}%")
    q += " ORDER BY id DESC LIMIT ?"; params.append(limit)
    with conn() as c:
        return pd.read_sql_query(q, c, params=params)

def delete_row(table: str, rid: int):
    with conn() as c:
        c.execute(f"DELETE FROM {table} WHERE id=?", (rid,))
        c.commit()

# --------------------------- App config ---------------------------

st.set_page_config(page_title="Quality Portal - Pilot", layout="wide")
init_db()

# --------------------------- Sidebar ---------------------------

with st.sidebar:
    st.header("⚙️ Sidebar")

    # Profile
    with st.expander("Profile", expanded=True):
        if "whoami" not in st.session_state:
            st.session_state.whoami = "appuser"
        st.session_state.whoami = st.text_input("Display name", value=st.session_state.whoami)
        st.caption("Used as Reporter for new records (imports use this if file omits Reporter).")

    # Add / Update Model
    with st.expander("Add / Update Model", expanded=False):
        m_no = st.text_input("Model number", key="mno")
        m_name = st.text_input("Customer / Name (optional)", key="mname")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Save model", key="save_model"):
                if m_no.strip():
                    upsert_model(m_no.strip(), m_name)
                    st.success("Model saved")
        with c2:
            if st.button("Delete model", key="del_model"):
                if m_no.strip():
                    with conn() as c:
                        c.execute("DELETE FROM models WHERE model_no=?", (m_no.strip(),))
                        c.commit()
                    st.warning("Model deleted")

    # Create First Piece
    with st.expander("Create First Piece", expanded=False):
        with st.form("fp_form", clear_on_submit=True):
            fp_model = st.text_input("Model")
            fp_ver   = st.text_input("Model Version")
            fp_sn    = st.text_input("SN / Barcode")
            fp_mo    = st.text_input("MO / Work Order")
            fp_note  = st.text_area("Note / Description")
            top_up   = st.file_uploader("TOP photo", type=["jpg","jpeg","png"], key="top_up")
            bot_up   = st.file_uploader("BOTTOM photo", type=["jpg","jpeg","png"], key="bot_up")
            submitted = st.form_submit_button("Save first piece")
        if submitted:
            if not fp_model.strip():
                st.error("Model is required")
            else:
                rel_top = save_image_to(fp_model, top_up) if top_up else None
                rel_bot = save_image_to(fp_model, bot_up) if bot_up else None
                payload = {
                    "created_at": now_iso(),
                    "model_no": fp_model.strip(),
                    "model_version": fp_ver.strip(),
                    "sn": fp_sn.strip(),
                    "mo": fp_mo.strip(),
                    "reporter": current_user(),
                    "note": fp_note.strip(),
                    "top_image_path": rel_top,
                    "bottom_image_path": rel_bot,
                    "extra": json.dumps({}),
                }
                insert_first_piece(payload)
                upsert_model(fp_model.strip())
                read_first_piece.clear()
                st.success("First piece saved")

    # Create Non-Conformity
    with st.expander("Create Non-Conformity", expanded=False):
        with st.form("nc_form", clear_on_submit=True):
            nc_model = st.text_input("Model", key="nc_model")
            nc_ver   = st.text_input("Model Version", key="nc_ver")
            nc_sn    = st.text_input("SN", key="nc_sn")
            nc_mo    = st.text_input("MO/PO", key="nc_mo")
            nc_sev   = st.selectbox("Severity", ["", "Minor", "Major", "Critical"], index=0)
            nc_desc  = st.text_area("Description")
            nc_img   = st.file_uploader("Photo (optional)", type=["jpg","jpeg","png"], key="nc_img")
            ok_nc    = st.form_submit_button("Save non-conformity")
        if ok_nc:
            rel = save_image_to(nc_model, nc_img) if nc_img else None
            payload = {
                "created_at": now_iso(),
                "model_no": nc_model.strip(),
                "model_version": nc_ver.strip(),
                "sn": nc_sn.strip(),
                "mo": nc_mo.strip(),
                "reporter": current_user(),
                "severity": nc_sev or "",
                "description": nc_desc.strip(),
                "image_path": rel,
                "extra": json.dumps({})
            }
            insert_nonconf(payload)
            upsert_model(nc_model.strip())
            read_nonconfs.clear()
            st.success("Non-conformity saved")

    # Import Non-Conformities (CSV/XLSX) — requires click
    with st.expander("Import Non-Conformities (CSV/XLSX)", expanded=False):
        st.caption("If a column is missing it’s ignored; extra columns go to 'extra'.")
        st.code(
            "Nonconformity, Description of Nonconformity, Date, Customer/Supplier, "
            "Model/Part No., MO/PO, Line, Work Station, Unit Head, Responsibility, "
            "Root Cause, Corrective Action, Exception reporters, Discovery, Origil Sources, "
            "Defective Item, Defective Outflow, Defective Qty, Inspection Qty, Lot Qty",
            language="text",
        )

        if "nc_last_hash" not in st.session_state:
            st.session_state.nc_last_hash = None
        if "nc_seed" not in st.session_state:
            st.session_state.nc_seed = 0

        with st.form("nc_import_form", clear_on_submit=False):
            upf = st.file_uploader(
                "Choose file", type=["csv", "xlsx"],
                key=f"nc_uploader_{st.session_state.nc_seed}"
            )
            do_import = st.form_submit_button("Import")

        if do_import:
            if upf is None:
                st.warning("Please choose a file.")
            else:
                raw = upf.getvalue()
                fhash = hashlib.md5(raw).hexdigest()
                if fhash == st.session_state.nc_last_hash:
                    st.info("This file was already imported in this session.")
                else:
                    # Read file robustly
                    try:
                        if upf.name.lower().endswith(".xlsx"):
                            df = pd.read_excel(io.BytesIO(raw), engine="openpyxl")
                        else:
                            df = None
                            for enc in ("utf-8-sig","utf-8","big5","cp950"):
                                try:
                                    df = pd.read_csv(io.BytesIO(raw), encoding=enc)
                                    break
                                except Exception:
                                    pass
                            if df is None:
                                df = pd.read_csv(io.BytesIO(raw), encoding="utf-8", errors="ignore")
                    except Exception as e:
                        st.error(f"Import failed: {e}")
                        df = None

                    if df is None or df.empty:
                        st.warning("No rows found.")
                    else:
                        df.columns = [str(c).strip() for c in df.columns]
                        COLS = {
                            "Model/Part No.": "model_no",
                            "MO/PO": "mo",
                            "Date": "created_at",
                            "Description of Nonconformity": "description",
                            "Nonconformity": ("extra", "Nonconformity"),
                            "Customer/Supplier": ("extra", "Customer/Supplier"),
                            "Line": ("extra", "Line"),
                            "Work Station": ("extra", "Work Station"),
                            "Unit Head": ("extra", "Unit Head"),
                            "Responsibility": ("extra", "Responsibility"),
                            "Root Cause": ("extra", "Root Cause"),
                            "Corrective Action": ("extra", "Corrective Action"),
                            "Exception reporters": ("extra", "Exception reporters"),
                            "Discovery": ("extra", "Discovery"),
                            "Origil Sources": ("extra", "Origil Sources"),
                            "Defective Item": ("extra", "Defective Item"),
                            "Defective Outflow": ("extra", "Defective Outflow"),
                            "Defective Qty": ("extra", "Defective Qty"),
                            "Inspection Qty": ("extra", "Inspection Qty"),
                            "Lot Qty": ("extra", "Lot Qty"),
                            "Reporter": "reporter",
                            "Severity": "severity"
                        }

                        n = 0
                        with st.spinner("Importing..."):
                            for _, row in df.iterrows():
                                payload = {
                                    "created_at": None, "model_no": "", "model_version": "",
                                    "sn": "", "mo": "", "reporter": current_user(),
                                    "severity": "", "description": "",
                                    "image_path": None, "extra": {}
                                }
                                for col, key in COLS.items():
                                    if col not in df.columns:
                                        continue
                                    val = row.get(col)
                                    if pd.isna(val):
                                        continue
                                    if key == "created_at":
                                        try:
                                            payload["created_at"] = pd.to_datetime(val).strftime("%Y-%m-%d")
                                        except Exception:
                                            payload["created_at"] = str(val)
                                    elif key in ("model_no","mo","description","reporter","severity"):
                                        payload[key] = str(val).strip()
                                    elif isinstance(key, tuple) and key[0] == "extra":
                                        payload["extra"][key[1]] = str(val)

                                if not payload["created_at"]:
                                    payload["created_at"] = now_iso()
                                payload["extra"] = json.dumps(payload["extra"], ensure_ascii=False)

                                insert_nonconf(payload)
                                if payload["model_no"]:
                                    upsert_model(payload["model_no"])
                                n += 1

                        st.success(f"Imported {n} row(s).")
                        read_nonconfs.clear()
                        st.session_state.nc_last_hash = fhash
                        st.session_state.nc_seed += 1
                        st.experimental_rerun()

# --------------------------- Main — Search & View ---------------------------

st.title("🔎 Quality Portal — Pilot")
st.subheader("Search & View")

# compact filters
st.markdown(
    """
    <style>
    .smalltxt input, .smalltxt textarea, .smalltxt select, .smalltxt label {font-size: 0.9rem !important;}
    .smalltxt [data-baseweb="input"]{min-height:2.2rem}
    </style>
    """,
    unsafe_allow_html=True,
)

with st.container():
    with st.expander("Filters", expanded=True):
        st.markdown('<div class="smalltxt">', unsafe_allow_html=True)
        cA, cB, cC, cD, cE, cF = st.columns([1.2,1.2,1.2,1.2,1.2,1.2])
        with cA:
            from_dt = st.date_input("Date from", value=None)
        with cB:
            to_dt   = st.date_input("Date to", value=None)
        with cC:
            model_q = st.text_input("Model contains", key="q_model")
        with cD:
            ver_q   = st.text_input("Version contains", key="q_ver")
        with cE:
            sn_q    = st.text_input("SN contains", key="q_sn")
        with cF:
            mo_q    = st.text_input("MO contains", key="q_mo")
        t1, t2, t3 = st.columns([1.2,1.2,1.2])
        with t1:
            reporter_q = st.text_input("Reporter contains", key="q_rep")
        with t2:
            text_q = st.text_input("Text in description/notes", key="q_text")
        with t3:
            limit_q = st.number_input("Max rows", 50, 2000, 200, step=50)
        st.markdown('</div>', unsafe_allow_html=True)

        c1, c2 = st.columns([1,1])
        with c1:
            run_first = st.button("Search First Piece")
        with c2:
            run_nc = st.button("Search Non-Conformities")

flt_common = {
    "from": str(from_dt) if isinstance(from_dt, date) else None,
    "to":   str(to_dt) if isinstance(to_dt, date) else None,
    "model_no": (model_q or "").strip(),
    "model_version": (ver_q or "").strip(),
    "sn": (sn_q or "").strip(),
    "mo": (mo_q or "").strip(),
    "reporter": (reporter_q or "").strip(),
}

# ------------------ First Piece results ------------------
if run_first:
    fp_filters = flt_common | {"note": (text_q or "").strip()}
    df_fp = read_first_piece(fp_filters, int(limit_q))
    st.markdown(f"### 📁 First Piece — {len(df_fp)} record(s)")
    if df_fp.empty:
        st.info("No records found.")
    else:
        for _, r in df_fp.iterrows():
            with st.container(border=True):
                cimg, ctxt = st.columns([2,3])
                with cimg:
                    p_top = DATA_DIR / str(r.get("top_image_path") or "")
                    p_bot = DATA_DIR / str(r.get("bottom_image_path") or "")
                    safe_show_image(p_top, caption="TOP")
                    safe_show_image(p_bot, caption="BOTTOM")
                with ctxt:
                    st.markdown(
                        f"**Model:** {r['model_no'] or '-'} | "
                        f"**Version:** {r['model_version'] or '-'} | "
                        f"**SN:** {r['sn'] or '-'} | **MO:** {r['mo'] or '-'}"
                    )
                    st.caption(f"🗓 {r['created_at']}  ·  👤 {r['reporter'] or '-'}")
                    if r.get("note"):
                        st.write(r["note"])
                    with st.expander("Add photo"):
                        add_top = st.file_uploader("Add TOP", type=["jpg","jpeg","png"], key=f"add_top_{r['id']}")
                        add_bot = st.file_uploader("Add BOTTOM", type=["jpg","jpeg","png"], key=f"add_bot_{r['id']}")
                        if st.button("Upload", key=f"up_{r['id']}"):
                            updates = {}
                            if add_top:
                                updates["top_image_path"] = save_image_to(r["model_no"], add_top)
                            if add_bot:
                                updates["bottom_image_path"] = save_image_to(r["model_no"], add_bot)
                            if updates:
                                with conn() as c:
                                    sets = ", ".join([f"{k}=?" for k in updates])
                                    c.execute(f"UPDATE first_piece SET {sets} WHERE id=?",
                                              [*updates.values(), int(r["id"])])
                                    c.commit()
                                read_first_piece.clear()
                                st.success("Updated. Click search again.")
                    if st.button("Delete", key=f"del_fp_{r['id']}"):
                        delete_row("first_piece", int(r["id"]))
                        read_first_piece.clear()
                        st.warning("Deleted. Click search again.")

        st.markdown("#### Table view & export")
        safe_dataframe(df_fp)
        download_csv_button(df_fp, "Download CSV", "first_piece.csv")

# ------------------ Non-Conformities results ------------------
if run_nc:
    nc_filters = flt_common | {"severity": "", "description": (text_q or "").strip()}
    df_nc = read_nonconfs(nc_filters, int(limit_q))
    st.markdown(f"### 🚩 Non-Conformities — {len(df_nc)} record(s)")
    if df_nc.empty:
        st.info("No records found.")
    else:
        for _, r in df_nc.iterrows():
            with st.container(border=True):
                cimg, ctxt = st.columns([2,3])
                with cimg:
                    p = DATA_DIR / str(r.get("image_path") or "")
                    safe_show_image(p)
                with ctxt:
                    st.markdown(
                        f"**Model:** {r['model_no'] or '-'} | "
                        f"**Version:** {r['model_version'] or '-'} | "
                        f"**SN:** {r['sn'] or '-'} | **MO:** {r['mo'] or '-'}"
                    )
                    sv = r.get("severity") or "-"
                    st.caption(f"🗓 {r['created_at']}  ·  👤 {r['reporter'] or '-'}  ·  {sv}")
                    if r.get("description"):
                        st.write(r["description"])
                    with st.expander("Add photo"):
                        add = st.file_uploader("Add photo", type=["jpg","jpeg","png"], key=f"add_nc_{r['id']}")
                        if st.button("Upload", key=f"up_nc_{r['id']}"):
                            if add:
                                rel = save_image_to(r["model_no"], add)
                                with conn() as c:
                                    c.execute("UPDATE nonconfs SET image_path=? WHERE id=?", (rel, int(r["id"])))
                                    c.commit()
                                read_nonconfs.clear()
                                st.success("Updated. Click search again.")
                    if st.button("Delete", key=f"del_nc_{r['id']}"):
                        delete_row("nonconfs", int(r["id"]))
                        read_nonconfs.clear()
                        st.warning("Deleted. Click search again.")

        st.markdown("#### Table view & export")
        safe_dataframe(df_nc)
        download_csv_button(df_nc, "Download CSV", "nonconfs.csv")

