# app.py — Quality Portal - Pilot
# v2: FP has department & customer_supplier, NC thumbnails smaller,
#     NC create form reordered, Customer/Supplier filter, gated search, DB auto-migrations.

import os
import json
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, List

import streamlit as st
import pandas as pd
from PIL import Image
import hashlib

# -----------------------------------------------------------------------------
# Storage (Cloud-safe)
# -----------------------------------------------------------------------------
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
IMG_DIR = DATA_DIR / "images"
FP_IMG_DIR = IMG_DIR / "first_piece"
NC_IMG_DIR = IMG_DIR / "nonconformity"
DB_PATH = DATA_DIR / "qc_portal.sqlite3"
for p in (IMG_DIR, FP_IMG_DIR, NC_IMG_DIR):
    p.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def now_iso() -> str:
    return datetime.utcnow().isoformat()

def sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def cur_user() -> Tuple[str, str, str]:
    # returns (username, display_name, role)
    ss = st.session_state
    return ss["auth_username"], ss["auth_display_name"], ss["auth_role"]

def save_image_to(folder: Path, uploaded) -> str:
    folder.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    clean = uploaded.name.replace(" ", "_")
    out_path = folder / f"{ts}_{clean}"
    Image.open(uploaded).convert("RGB").save(out_path, format="JPEG", quality=88)
    return str(out_path.relative_to(DATA_DIR))

# -----------------------------------------------------------------------------
# DB
# -----------------------------------------------------------------------------
SCHEMA_USERS = """
CREATE TABLE IF NOT EXISTS users(
  username TEXT PRIMARY KEY,
  password_hash TEXT NOT NULL,
  role TEXT NOT NULL,
  display_name TEXT NOT NULL
);
"""

SCHEMA_MODELS = """
CREATE TABLE IF NOT EXISTS models(
  model_no TEXT PRIMARY KEY,
  name TEXT
);
"""

SCHEMA_FIRST_PIECE = """
CREATE TABLE IF NOT EXISTS first_piece(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at TEXT,                 -- UTC ISO
  model_no TEXT,
  model_version TEXT,
  sn TEXT,
  mo TEXT,
  department TEXT,                 -- NEW
  customer_supplier TEXT,          -- NEW
  reporter TEXT,
  description TEXT,
  top_image_path TEXT,
  bottom_image_path TEXT,
  extra JSON
);
"""

SCHEMA_NONCONF = """
CREATE TABLE IF NOT EXISTS nonconf(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at TEXT,                 -- UTC ISO
  model_no TEXT,
  model_version TEXT,
  sn TEXT,
  mo TEXT,
  reporter TEXT,
  severity TEXT,
  nonconformity TEXT,
  description TEXT,
  customer_supplier TEXT,
  line TEXT,
  work_station TEXT,
  unit_head TEXT,
  responsibility TEXT,
  root_cause TEXT,
  corrective_action TEXT,
  exception_reporters TEXT,
  discovery TEXT,
  origin_sources TEXT,
  defective_item TEXT,
  defective_qty TEXT,
  inspection_qty TEXT,
  lot_qty TEXT,
  image_paths JSON,
  extra JSON
);
"""

def get_conn():
    return sqlite3.connect(DB_PATH)

def init_db():
    with get_conn() as c:
        c.execute(SCHEMA_USERS)
        c.execute(SCHEMA_MODELS)
        c.execute(SCHEMA_FIRST_PIECE)
        c.execute(SCHEMA_NONCONF)
        c.commit()
    ensure_default_admin()
    apply_migrations()

def ensure_default_admin():
    with get_conn() as c:
        n = c.execute("SELECT COUNT(*) FROM users WHERE username=?", ("Admin",)).fetchone()[0]
        if n == 0:
            c.execute(
                "INSERT INTO users(username, password_hash, role, display_name) VALUES(?,?,?,?)",
                ("Admin", sha256("admin1234"), "Admin", "Admin"),
            )
            c.commit()

def apply_migrations():
    """Add missing columns safely if DB is from a previous version."""
    def col_exists(table, col):
        with get_conn() as c:
            cols = [r[1] for r in c.execute(f"PRAGMA table_info({table})")]
            return col in cols

    with get_conn() as c:
        # first_piece new columns
        if not col_exists("first_piece", "department"):
            c.execute("ALTER TABLE first_piece ADD COLUMN department TEXT")
        if not col_exists("first_piece", "customer_supplier"):
            c.execute("ALTER TABLE first_piece ADD COLUMN customer_supplier TEXT")
        # nonconf fields should already match; add if missing (future-proof)
        for addcol in [
            ("nonconf", "customer_supplier", "TEXT"),
            ("nonconf", "image_paths", "JSON"),
        ]:
            if not col_exists(addcol[0], addcol[1]):
                c.execute(f"ALTER TABLE {addcol[0]} ADD COLUMN {addcol[1]} {addcol[2]}")
        c.commit()

# Cached reads
@st.cache_data(show_spinner=False)
def list_models_df() -> pd.DataFrame:
    with get_conn() as c:
        return pd.read_sql_query("SELECT model_no, COALESCE(name,'') AS name FROM models ORDER BY model_no", c)

@st.cache_data(show_spinner=False)
def list_users_df() -> pd.DataFrame:
    with get_conn() as c:
        return pd.read_sql_query("SELECT username, role, display_name FROM users ORDER BY username", c)

@st.cache_data(show_spinner=False)
def load_fp_df(filters: dict) -> pd.DataFrame:
    q = "SELECT * FROM first_piece WHERE 1=1"
    params = []
    if filters.get("date_from"):
        q += " AND created_at >= ?"
        params.append(filters["date_from"])
    if filters.get("date_to"):
        q += " AND created_at <= ?"
        params.append(filters["date_to"])
    for col in ["model_no", "model_version", "sn", "mo", "customer_supplier", "department"]:
        v = filters.get(col, "").strip()
        if v:
            q += f" AND {col} LIKE ?"
            params.append(f"%{v}%")
    with get_conn() as c:
        return pd.read_sql_query(q + " ORDER BY id DESC LIMIT 500", c, params=params)

@st.cache_data(show_spinner=False)
def load_nc_df(filters: dict) -> pd.DataFrame:
    q = "SELECT * FROM nonconf WHERE 1=1"
    params = []
    if filters.get("date_from"):
        q += " AND created_at >= ?"
        params.append(filters["date_from"])
    if filters.get("date_to"):
        q += " AND created_at <= ?"
        params.append(filters["date_to"])
    for col in ["model_no","model_version","sn","mo","customer_supplier"]:
        v = filters.get(col, "").strip()
        if v:
            q += f" AND {col} LIKE ?"
            params.append(f"%{v}%")
    text = filters.get("text", "").strip()
    if text:
        # search in reporter/severity/description/nonconformity
        q += " AND (reporter LIKE ? OR severity LIKE ? OR description LIKE ? OR nonconformity LIKE ?)"
        params.extend([f"%{text}%"] * 4)
    with get_conn() as c:
        return pd.read_sql_query(q + " ORDER BY id DESC LIMIT 500", c, params=params)

def invalidate_caches():
    list_models_df.clear()
    list_users_df.clear()
    load_fp_df.clear()
    load_nc_df.clear()

# -----------------------------------------------------------------------------
# Auth
# -----------------------------------------------------------------------------
def do_login():
    st.title("Quality Portal - Pilot")
    st.subheader("Login")

    c1, c2 = st.columns([1,1])
    with c1:
        u = st.text_input("Username", key="login_user")
    with c2:
        p = st.text_input("Password", type="password", key="login_pass")

    if st.button("Sign in", use_container_width=True):
        with get_conn() as c:
            row = c.execute("SELECT username,password_hash,role,display_name FROM users WHERE username=?", (u,)).fetchone()
        if row and sha256(p) == row[1]:
            st.session_state["auth"] = True
            st.session_state["auth_username"]  = row[0]
            st.session_state["auth_role"]      = row[2]
            st.session_state["auth_display_name"] = row[3]
            st.success("Welcome!")
            st.experimental_rerun()
        else:
            st.error("Invalid credentials")

# -----------------------------------------------------------------------------
# UI blocks
# -----------------------------------------------------------------------------
def sidebar_admin():
    username, display, role = cur_user()
    st.caption(f"Signed in as **{display}** (*{role}*)")
    st.divider()
    st.subheader("Users (Admin)")
    df = list_users_df()
    st.dataframe(df, hide_index=True, use_container_width=True)
    with st.expander("Add / Update User"):
        new_u = st.text_input("Username")
        new_d = st.text_input("Display name")
        new_p = st.text_input("Password", type="password")
        new_r = st.selectbox("Role", ["Admin","QA","QC"], index=2)
        if st.button("Save user"):
            if not new_u or not new_p or not new_d:
                st.error("Please complete all fields.")
            else:
                with get_conn() as c:
                    c.execute(
                        """INSERT INTO users(username,password_hash,role,display_name)
                           VALUES(?,?,?,?)
                           ON CONFLICT(username) DO UPDATE SET
                             password_hash=excluded.password_hash,
                             role=excluded.role,
                             display_name=excluded.display_name
                        """,
                        (new_u, sha256(new_p), new_r, new_d),
                    )
                    c.commit()
                list_users_df.clear()
                st.success("User saved.")

def fp_form():
    st.subheader("First Piece")
    with st.form("fp_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            fp_model = st.text_input("Model (short, e.g. 190-56980)")
            fp_version = st.text_input("Model Version (full)")
        with col2:
            fp_sn = st.text_input("SN / Barcode")
            fp_mo = st.text_input("MO / Work Order")
        with col3:
            fp_dept = st.text_input("Department")
            fp_cs = st.text_input("Customer / Supplier")

        fp_desc = st.text_area("Notes / Description (optional)", height=80)
        tcol, bcol = st.columns(2)
        with tcol:
            top_u = st.file_uploader("TOP image", type=["jpg","jpeg","png"], key="fp_top")
        with bcol:
            bot_u = st.file_uploader("BOTTOM image", type=["jpg","jpeg","png"], key="fp_bot")

        submitted = st.form_submit_button("Save first piece")
        if submitted:
            if not fp_model.strip():
                st.error("Model is required.")
                return
            top_rel = save_image_to(FP_IMG_DIR, top_u) if top_u else None
            bot_rel = save_image_to(FP_IMG_DIR, bot_u) if bot_u else None
            _, disp, _role = cur_user()
            payload = (
                now_iso(), fp_model.strip(), fp_version.strip(), fp_sn.strip(), fp_mo.strip(),
                fp_dept.strip(), fp_cs.strip(), disp, fp_desc.strip(),
                top_rel, bot_rel, json.dumps({})
            )
            with get_conn() as c:
                c.execute(
                    """INSERT INTO first_piece
                       (created_at, model_no, model_version, sn, mo, department, customer_supplier,
                        reporter, description, top_image_path, bottom_image_path, extra)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                    payload,
                )
                c.commit()
            load_fp_df.clear()
            st.success("First piece saved.")

def nc_form():
    st.subheader("Create Non-Conformity")

    with st.form("nc_form"):
        # Top basics
        b1, b2, b3, b4 = st.columns(4)
        with b1:
            nc_model = st.text_input("Model (short)")
        with b2:
            nc_version = st.text_input("Model Version")
        with b3:
            nc_sn = st.text_input("SN / Barcode")
        with b4:
            nc_mo = st.text_input("MO / Work Order")

        # Your specified arrangement (upload photo last)
        # 1) Nonconformity title & Description
        nc_title = st.text_input("Nonconformity")
        nc_desc  = st.text_area("Description of Nonconformity", height=80)

        # 2) Customer/Supplier + Line + Work Station + Unit Head
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            nc_cs = st.text_input("Customer/Supplier")
        with c2:
            nc_line = st.text_input("Line")
        with c3:
            nc_ws = st.text_input("Work Station")
        with c4:
            nc_head = st.text_input("Unit Head")

        # 3) Responsibility + Root Cause + Corrective Action
        r1, r2, r3 = st.columns(3)
        with r1:
            nc_resp = st.text_input("Responsibility")
        with r2:
            nc_root = st.text_input("Root Cause")
        with r3:
            nc_corr = st.text_input("Corrective Action")

        # 4) Exception reporters + Discovery + Origin Sources
        e1, e2, e3 = st.columns(3)
        with e1:
            nc_except = st.text_input("Exception reporters")
        with e2:
            nc_disc = st.text_input("Discovery")
        with e3:
            nc_origin = st.text_input("Origil Sources")

        # 5) Defective Item + quantities
        q1, q2, q3 = st.columns(3)
        with q1:
            nc_def_item = st.text_input("Defective Item")
        with q2:
            nc_def_qty  = st.text_input("Defective Qty")
        with q3:
            nc_insp_qty = st.text_input("Inspection Qty")
        lot_qty = st.text_input("Lot Qty")

        # 6) Severity & Photos (last)
        s1, s2 = st.columns([1,3])
        with s1:
            nc_sev = st.selectbox("Severity", ["Minor","Major","Critical"], index=0)
        with s2:
            nc_imgs = st.file_uploader("Upload photo(s)", accept_multiple_files=True,
                                       type=["jpg","jpeg","png"])

        submitted = st.form_submit_button("Save non-conformity")
        if submitted:
            if not nc_title.strip():
                st.error("Nonconformity title is required.")
                return
            _, disp, _ = cur_user()
            rels: List[str] = []
            for f in nc_imgs or []:
                try:
                    rels.append(save_image_to(NC_IMG_DIR, f))
                except Exception as e:
                    st.error(f"Failed saving {f.name}: {e}")
            with get_conn() as c:
                c.execute(
                    """INSERT INTO nonconf
                       (created_at, model_no, model_version, sn, mo, reporter, severity,
                        nonconformity, description, customer_supplier, line, work_station,
                        unit_head, responsibility, root_cause, corrective_action,
                        exception_reporters, discovery, origin_sources,
                        defective_item, defective_qty, inspection_qty, lot_qty,
                        image_paths, extra)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        now_iso(), nc_model.strip(), nc_version.strip(), nc_sn.strip(), nc_mo.strip(),
                        disp, nc_sev, nc_title.strip(), nc_desc.strip(), nc_cs.strip(), nc_line.strip(),
                        nc_ws.strip(), nc_head.strip(), nc_resp.strip(), nc_root.strip(), nc_corr.strip(),
                        nc_except.strip(), nc_disc.strip(), nc_origin.strip(),
                        nc_def_item.strip(), nc_def_qty.strip(), nc_insp_qty.strip(), lot_qty.strip(),
                        json.dumps(rels, ensure_ascii=False), json.dumps({})
                    ),
                )
                c.commit()
            load_nc_df.clear()
            st.success("Non-conformity saved.")

# -----------------------------------------------------------------------------
# Search & View
# -----------------------------------------------------------------------------
def search_filters() -> dict:
    st.subheader("Search & View")
    with st.expander("Filters", expanded=True):
        d1, d2 = st.columns(2)
        with d1:
            df = st.date_input("Date from", value=None)
        with d2:
            dt = st.date_input("Date to", value=None)

        r1, r2, r3, r4 = st.columns(4)
        with r1:
            m = st.text_input("Model contains")
        with r2:
            v = st.text_input("Version contains")
        with r3:
            sn = st.text_input("SN contains")
        with r4:
            mo = st.text_input("MO contains")

        r5, r6 = st.columns(2)
        with r5:
            cs = st.text_input("Customer/Supplier contains")
        with r6:
            dept = st.text_input("Department contains (FP)")

        text = st.text_input("Text in description/reporter/type (NC only)")

        run = st.button("Search", type="primary")
    return {
        "run": run,
        "date_from": df.strftime("%Y-%m-%d") if df else None,
        "date_to": (datetime.combine(dt, datetime.min.time()).strftime("%Y-%m-%dT23:59:59") if dt else None),
        "model_no": m, "model_version": v, "sn": sn, "mo": mo,
        "customer_supplier": cs, "department": dept, "text": text
    }

def render_first_piece(df: pd.DataFrame):
    if df.empty:
        st.info("No First Piece results.")
        return
    st.caption(f"{len(df)} record(s)")
    for _, r in df.iterrows():
        with st.container(border=True):
            header = (
                f"**Model:** {r['model_no'] or '-'} | "
                f"**Version:** {r['model_version'] or '-'} | "
                f"**SN:** {r['sn'] or '-'} | **MO:** {r['mo'] or '-'}"
            )
            st.markdown(header)
            sub = (
                f"🗓 {r['created_at'][:10]}  ·  👤 {r['reporter']}  ·  "
                f"🏢 **Dept:** {r.get('department') or '-'}  ·  "
                f"🏷 **Customer/Supplier:** {r.get('customer_supplier') or '-'}"
            )
            st.caption(sub)

            c1, c2 = st.columns(2)
            p_top = (DATA_DIR / str(r.get("top_image_path"))) if r.get("top_image_path") else None
            p_bot = (DATA_DIR / str(r.get("bottom_image_path"))) if r.get("bottom_image_path") else None
            with c1:
                if p_top and p_top.exists():
                    st.image(str(p_top), caption="TOP", use_container_width=True)
            with c2:
                if p_bot and p_bot.exists():
                    st.image(str(p_bot), caption="BOTTOM", use_container_width=True)

            if r.get("description"):
                st.write(r["description"])

def render_nonconf(df: pd.DataFrame, role: str):
    if df.empty:
        st.info("No Non-Conformity results.")
        return
    st.caption(f"{len(df)} record(s)")
    for _, r in df.iterrows():
        with st.container(border=True):
            st.markdown(
                f"**Model:** {r['model_no'] or '-'} | **Version:** {r['model_version'] or '-'} "
                f"| **SN:** {r['sn'] or '-'} | **MO:** {r['mo'] or '-'}"
            )
            st.caption(
                f"🗓 {r['created_at'][:10]} · 👤 {r['reporter']} · "
                f"Severity: **{r['severity'] or '-'}** · "
                f"Customer/Supplier: **{r.get('customer_supplier') or '-'}**"
            )
            if r.get("nonconformity"):
                st.markdown(f"**{r['nonconformity']}**")
            if r.get("description"):
                st.write(r["description"])

            # thumbnails grid (smaller, no huge white)
            rels = []
            try:
                rels = json.loads(r.get("image_paths") or "[]")
            except Exception:
                rels = []
            if rels:
                # 4 per row max
                thumbs = [DATA_DIR / rel for rel in rels]
                row = st.columns(4)
                i = 0
                for p in thumbs:
                    if p and p.exists():
                        with row[i % 4]:
                            st.image(str(p), width=220)
                    i += 1

            # Extra key items (compact)
            extras = [
                ("line","Line"), ("work_station","Work Station"), ("unit_head","Unit Head"),
                ("responsibility","Responsibility"), ("root_cause","Root Cause"),
                ("corrective_action","Corrective Action"), ("exception_reporters","Exception reporters"),
                ("discovery","Discovery"), ("origin_sources","Origil Sources"),
                ("defective_item","Defective Item"), ("defective_qty","Defective Qty"),
                ("inspection_qty","Inspection Qty"), ("lot_qty","Lot Qty"),
            ]
            small = []
            for key, label in extras:
                val = r.get(key)
                if val:
                    small.append(f"**{label}:** {val}")
            if small:
                st.caption(" · ".join(small))

            # deletion allowed to Admin/QA only
            if role in ("Admin","QA"):
                if st.button("Delete", key=f"del_nc_{r['id']}"):
                    with get_conn() as c:
                        c.execute("DELETE FROM nonconf WHERE id=?", (int(r["id"]),))
                        c.commit()
                    load_nc_df.clear()
                    st.success("Deleted.")
                    st.experimental_rerun()

# -----------------------------------------------------------------------------
# Main App
# -----------------------------------------------------------------------------
init_db()

st.set_page_config(page_title="Quality Portal - Pilot", layout="wide")

if "auth" not in st.session_state:
    st.session_state["auth"] = False

if not st.session_state["auth"]:
    do_login()
    st.stop()

# Header bar
title_left, title_right = st.columns([4,1])
with title_left:
    st.markdown("## **Quality Portal - Pilot**")
with title_right:
    if st.button("Sign out"):
        for k in list(st.session_state.keys()):
            if k.startswith("auth"):
                del st.session_state[k]
        st.session_state["auth"] = False
        st.experimental_rerun()

# Sidebar
u_name, disp, role = cur_user()
with st.sidebar:
    st.markdown(f"**User:** {disp}  \n**Role:** {role}")
    st.divider()

    if role == "Admin":
        sidebar_admin()

    st.subheader("Add / Update Model")
    m_no = st.text_input("Model number")
    m_nm = st.text_input("Name / Customer (optional)")
    colA, colB = st.columns(2)
    with colA:
        if st.button("Save model"):
            if not m_no.strip():
                st.error("Model cannot be empty.")
            else:
                with get_conn() as c:
                    c.execute(
                        """INSERT INTO models(model_no,name) VALUES(?,?)
                           ON CONFLICT(model_no) DO UPDATE SET name=excluded.name""",
                        (m_no.strip(), m_nm.strip()),
                    )
                    c.commit()
                list_models_df.clear()
                st.success("Saved.")
    with colB:
        if st.button("Delete model", type="secondary"):
            if not m_no.strip():
                st.error("Enter model number to delete.")
            else:
                with get_conn() as c:
                    c.execute("DELETE FROM models WHERE model_no=?", (m_no.strip(),))
                    c.commit()
                list_models_df.clear()
                st.warning("Deleted.")

st.markdown("---")

# Create blocks (role permissions)
c1, c2 = st.columns(2)
with c1:
    fp_form()
with c2:
    nc_form()

st.markdown("---")

# Search & View
f = search_filters()
if f["run"]:
    # First Piece
    st.markdown("### First Piece (results)")
    fp_df = load_fp_df(f)
    render_first_piece(fp_df)

    # Non-Conformities
    st.markdown("### Non-Conformities (results)")
    nc_df = load_nc_df(f)
    render_nonconf(nc_df, role)

    # Export (both tabs)
    with st.expander("Table view & export"):
        tab1, tab2 = st.tabs(["First Piece", "Non-Conformities"])
        with tab1:
            if not fp_df.empty:
                st.dataframe(fp_df, use_container_width=True, hide_index=True)
                st.download_button("Export First Piece (CSV)",
                                   data=fp_df.to_csv(index=False).encode("utf-8-sig"),
                                   file_name=f"first_piece_{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}.csv",
                                   mime="text/csv")
            else:
                st.info("No First Piece rows to export.")
        with tab2:
            if not nc_df.empty:
                st.dataframe(nc_df, use_container_width=True, hide_index=True)
                st.download_button("Export Non-Conformities (CSV)",
                                   data=nc_df.to_csv(index=False).encode("utf-8-sig"),
                                   file_name=f"nonconf_{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}.csv",
                                   mime="text/csv")
            else:
                st.info("No Non-Conformity rows to export.")

else:
    st.info("Use **Filters → Search** to load results. Nothing is loaded by default to keep the app fast.")
