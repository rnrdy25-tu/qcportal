# Quality Portal - Pilot
# One-file Streamlit app with login/roles + QC features
# Default Admin:  username: Admin   password: admin1234   display name: Admin

import os
import io
import json
import hashlib
import secrets
import sqlite3
from pathlib import Path
from datetime import datetime

import pandas as pd
import streamlit as st
from PIL import Image

# ───────────────────────── Storage ─────────────────────────
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

# ───────────────────────── Helpers ─────────────────────────
APP_TITLE = "Quality Portal - Pilot"

def now_iso():
    return datetime.utcnow().isoformat()

def save_image_to_model_folder(model_no: str, uploaded_file) -> str:
    """Save upload under images/<model_no>/..., return path relative to DATA_DIR."""
    folder = IMG_DIR / (model_no or "_no_model_")
    folder.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    safe_name = uploaded_file.name.replace(" ", "_")
    out_path = folder / f"{ts}_{safe_name}"
    img = Image.open(uploaded_file).convert("RGB")
    img.save(out_path, format="JPEG", quality=90)
    return str(out_path.relative_to(DATA_DIR))

def parse_extra(s):
    if not s:
        return {}
    if isinstance(s, dict):
        return s
    try:
        return json.loads(s)
    except Exception:
        return {}

# ───────────────────── Non-conf extra field config ─────────────────────
NC_EXTRA_FIELDS = [
    ("Customer/Supplier", "customer_supplier"),
    ("Line", "line"),
    ("Work Station", "work_station"),
    ("Unit Head", "unit_head"),
    ("Responsibility", "responsibility"),
    ("Root Cause", "root_cause"),
    ("Corrective Action", "corrective_action"),
    ("Exception reporters", "exception_reporters"),
    ("Discovery", "discovery"),
    ("Origil Sources", "origin_sources"),
    ("Defective Item", "defective_item"),
    ("Defective Outflow", "defective_outflow"),
    ("Defective Qty", "defective_qty"),
    ("Inspection Qty", "inspection_qty"),
    ("Lot Qty", "lot_qty"),
]

# ───────────────────────── Database ─────────────────────────
def get_conn():
    return sqlite3.connect(DB_PATH)

SCHEMA_USERS = """
CREATE TABLE IF NOT EXISTS users(
  username TEXT PRIMARY KEY,
  pw_hash  BLOB NOT NULL,
  salt     BLOB NOT NULL,
  role     TEXT NOT NULL, -- Admin | QA | QC
  display_name TEXT NOT NULL
);
"""

SCHEMA_MODELS = """
CREATE TABLE IF NOT EXISTS models(
  model_no TEXT PRIMARY KEY,
  name     TEXT
);
"""

SCHEMA_FIRSTPIECE = """
CREATE TABLE IF NOT EXISTS firstpiece(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at TEXT,
  model_no TEXT,
  model_version TEXT,
  sn TEXT,
  mo TEXT,
  reporter TEXT,       -- display_name
  description TEXT,
  top_image TEXT,
  bottom_image TEXT
);
"""

SCHEMA_NONCONFS = """
CREATE TABLE IF NOT EXISTS nonconfs(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at TEXT,
  model_no TEXT,
  model_version TEXT,
  sn TEXT,
  mo TEXT,
  reporter TEXT,       -- display_name
  severity TEXT,       -- Minor/Major/Critical
  description TEXT,
  image_path TEXT,     -- optional main photo
  extra JSON           -- big JSON blob for all extra fields
);
"""

def init_db():
    with get_conn() as c:
        c.execute(SCHEMA_USERS)
        c.execute(SCHEMA_MODELS)
        c.execute(SCHEMA_FIRSTPIECE)
        c.execute(SCHEMA_NONCONFS)
        # seed default Admin if missing
        if c.execute("SELECT COUNT(*) FROM users WHERE username='Admin'").fetchone()[0] == 0:
            salt = secrets.token_bytes(16)
            pw_hash = hashlib.pbkdf2_hmac("sha256", b"admin1234", salt, 200_000)
            c.execute(
                "INSERT INTO users(username,pw_hash,salt,role,display_name) VALUES(?,?,?,?,?)",
                ("Admin", pw_hash, salt, "Admin", "Admin")
            )
        c.commit()

init_db()

# ───────────────────────── Auth ─────────────────────────
def hash_password(password: str, salt: bytes) -> bytes:
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 200_000)

def check_credentials(username: str, password: str):
    with get_conn() as c:
        row = c.execute("SELECT username, pw_hash, salt, role, display_name FROM users WHERE username=?",
                        (username.strip(),)).fetchone()
    if not row:
        return None
    _, pw_hash, salt, role, display_name = row
    test = hash_password(password, salt)
    if secrets.compare_digest(test, pw_hash):
        return {"username": username, "role": role, "display_name": display_name}
    return None

def create_user(username, password, role, display_name):
    salt = secrets.token_bytes(16)
    pw_hash = hash_password(password, salt)
    with get_conn() as c:
        c.execute("INSERT OR REPLACE INTO users(username,pw_hash,salt,role,display_name) VALUES(?,?,?,?,?)",
                  (username.strip(), pw_hash, salt, role, display_name.strip()))
        c.commit()

def reset_password(username, new_password):
    salt = secrets.token_bytes(16)
    pw_hash = hash_password(new_password, salt)
    with get_conn() as c:
        c.execute("UPDATE users SET pw_hash=?, salt=? WHERE username=?", (pw_hash, salt, username))
        c.commit()

def set_user_role_display(username, role, display_name):
    with get_conn() as c:
        c.execute("UPDATE users SET role=?, display_name=? WHERE username=?", (role, display_name, username))
        c.commit()

def delete_user(username):
    with get_conn() as c:
        c.execute("DELETE FROM users WHERE username=?", (username,))
        c.commit()

# ───────────────────────── Cached loaders ─────────────────────────
@st.cache_data(show_spinner=False)
def list_models_df():
    with get_conn() as c:
        return pd.read_sql_query("SELECT model_no, COALESCE(name,'') AS name FROM models ORDER BY model_no", c)

@st.cache_data(show_spinner=False)
def load_firstpiece_df(**filters):
    # Basic filter: build WHERE and params
    where, params = [], []
    for key in ("model_no", "model_version", "sn", "mo"):
        v = filters.get(key)
        if v:
            where.append(f"{key} LIKE ?")
            params.append(f"%{v}%")
    if filters.get("start_date"):
        where.append("date(created_at) >= date(?)")
        params.append(filters["start_date"])
    if filters.get("end_date"):
        where.append("date(created_at) <= date(?)")
        params.append(filters["end_date"])
    sql = """SELECT id, created_at, model_no, model_version, sn, mo, reporter,
                    description, top_image, bottom_image
             FROM firstpiece"""
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY id DESC LIMIT 500"
    with get_conn() as c:
        return pd.read_sql_query(sql, c, params=params)

@st.cache_data(show_spinner=False)
def load_nonconf_df(**filters):
    where, params = [], []
    for key in ("model_no", "model_version", "sn", "mo"):
        v = filters.get(key)
        if v:
            where.append(f"{key} LIKE ?")
            params.append(f"%{v}%")
    if filters.get("start_date"):
        where.append("date(created_at) >= date(?)")
        params.append(filters["start_date"])
    if filters.get("end_date"):
        where.append("date(created_at) <= date(?)")
        params.append(filters["end_date"])
    if filters.get("text"):
        where.append("(description LIKE ? OR reporter LIKE ? OR severity LIKE ?)")
        params.extend([f"%{filters['text']}%"]*3)
    sql = """SELECT id, created_at, model_no, model_version, sn, mo, reporter,
                    severity, description, image_path, extra
             FROM nonconfs"""
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY id DESC LIMIT 500"
    with get_conn() as c:
        return pd.read_sql_query(sql, c, params=params)

@st.cache_data(show_spinner=False)
def nonconf_distinct_customers():
    with get_conn() as c:
        try:
            raw = pd.read_sql_query("SELECT extra FROM nonconfs", c)
        except Exception:
            return []
    names = set()
    for s in raw["extra"].dropna():
        nm = parse_extra(s).get("customer_supplier")
        if nm:
            names.add(str(nm).strip())
    return sorted(names)

# ───────────────────────── Inserts / updates ─────────────────────────
def upsert_model(model_no: str, name: str = ""):
    with get_conn() as c:
        c.execute(
            """INSERT INTO models(model_no, name) VALUES(?,?)
               ON CONFLICT(model_no) DO UPDATE SET name=excluded.name""",
            (model_no.strip(), name.strip()))
        c.commit()

def delete_model(model_no: str):
    with get_conn() as c:
        c.execute("DELETE FROM models WHERE model_no=?", (model_no,))
        c.commit()

def insert_firstpiece(payload: dict):
    fields = ["created_at","model_no","model_version","sn","mo","reporter","description","top_image","bottom_image"]
    values = [payload.get(k) for k in fields]
    with get_conn() as c:
        c.execute(f"INSERT INTO firstpiece({','.join(fields)}) VALUES(?,?,?,?,?,?,?,?,?)", values)
        c.commit()

def delete_firstpiece(_id: int):
    with get_conn() as c:
        c.execute("DELETE FROM firstpiece WHERE id=?", (_id,))
        c.commit()

def insert_nonconf(payload: dict):
    fields = ["created_at","model_no","model_version","sn","mo","reporter","severity","description","image_path","extra"]
    values = [payload.get(k) for k in fields]
    with get_conn() as c:
        c.execute(f"INSERT INTO nonconfs({','.join(fields)}) VALUES(?,?,?,?,?,?,?,?,?,?)", values)
        c.commit()

def delete_nonconf(_id: int):
    with get_conn() as c:
        c.execute("DELETE FROM nonconfs WHERE id=?", (_id,))
        c.commit()

# ───────────────────────── UI pieces ─────────────────────────
def require_login():
    st.session_state.setdefault("user", None)
    if st.session_state["user"]:
        return True

    st.title(APP_TITLE)
    st.subheader("Sign in")

    with st.form("login_form", clear_on_submit=False):
        u = st.text_input("Username", value="", autocomplete="username")
        p = st.text_input("Password", value="", type="password", autocomplete="current-password")
        ok = st.form_submit_button("Login")

    if ok:
        user = check_credentials(u, p)
        if user:
            st.session_state["user"] = user
            st.success(f"Welcome, {user['display_name']} ({user['role']})")
            st.experimental_rerun()
        else:
            st.error("Invalid username or password.")
    st.stop()

def top_navbar():
    user = st.session_state["user"]
    cols = st.columns([1,1,1,1,2])
    with cols[0]:
        if st.button("Search & View", use_container_width=True):
            st.session_state["page"] = "view"
    with cols[1]:
        if st.button("Create First-Piece", use_container_width=True):
            st.session_state["page"] = "fp_create"
    with cols[2]:
        if st.button("Create Non-Conformity", use_container_width=True):
            st.session_state["page"] = "nc_create"
    with cols[3]:
        if st.button("Import / Export", use_container_width=True):
            st.session_state["page"] = "import"
    with cols[4]:
        right = st.columns([3,1])
        with right[0]:
            st.markdown(f"**User:** {user['display_name']}  ·  **Role:** {user['role']}")
        with right[1]:
            if st.button("Logout", type="secondary"):
                st.session_state["user"] = None
                st.experimental_rerun()

    st.markdown("---")

def page_user_admin():
    st.header("User Administration (Admin only)")
    with get_conn() as c:
        df = pd.read_sql_query("SELECT username, role, display_name FROM users ORDER BY username", c)
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.subheader("Create / Update user")
    with st.form("user_edit"):
        col1, col2, col3 = st.columns(3)
        with col1:
            u = st.text_input("Username")
        with col2:
            role = st.selectbox("Role", ["Admin", "QA", "QC"])
        with col3:
            disp = st.text_input("Display name")
        p1 = st.text_input("New password", type="password")
        p2 = st.text_input("Confirm password", type="password")
        ok = st.form_submit_button("Save user")
        if ok:
            if not u or not disp or not p1 or p1 != p2:
                st.error("Please fill username, display name, matching passwords.")
            else:
                create_user(u, p1, role, disp)
                st.success("User created/updated.")
                st.experimental_rerun()

    st.subheader("Reset password")
    with st.form("pw_reset"):
        u2 = st.text_input("Username to reset")
        p3 = st.text_input("New password", type="password")
        ok2 = st.form_submit_button("Reset")
        if ok2:
            reset_password(u2, p3)
            st.success("Password reset.")

    st.subheader("Change role / display name")
    with st.form("role_edit"):
        u3 = st.text_input("Username to change")
        new_role = st.selectbox("New role", ["Admin","QA","QC"], key="role_chg")
        new_disp = st.text_input("New display name", key="disp_chg")
        ok3 = st.form_submit_button("Apply")
        if ok3:
            set_user_role_display(u3, new_role, new_disp)
            st.success("Updated.")

    st.subheader("Delete user")
    with st.form("user_del"):
        u4 = st.text_input("Username to delete")
        ok4 = st.form_submit_button("Delete user", type="primary")
        if ok4:
            if u4 == "Admin":
                st.error("Cannot delete seeded Admin.")
            else:
                delete_user(u4)
                st.success("Deleted.")

def page_firstpiece_create():
    st.header("Create First-Piece")
    u = st.session_state["user"]
    with st.form("fp_form", clear_on_submit=True):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            model = st.text_input("Model (short, e.g. 190-56980)")
        with col2:
            version = st.text_input("Model version (full)")
        with col3:
            sn = st.text_input("SN / Barcode")
        with col4:
            mo = st.text_input("MO / Work Order")
        desc = st.text_area("Notes / Description (optional)")
        up_top = st.file_uploader("TOP photo (JPG/PNG)", type=["jpg","jpeg","png"])
        up_bottom = st.file_uploader("BOTTOM photo (JPG/PNG)", type=["jpg","jpeg","png"])
        ok = st.form_submit_button("Save first-piece")

    if ok:
        if not model:
            st.error("Model is required.")
            return
        top_rel = save_image_to_model_folder(model, up_top) if up_top else ""
        bot_rel = save_image_to_model_folder(model, up_bottom) if up_bottom else ""
        payload = {
            "created_at": now_iso(),
            "model_no": model.strip(),
            "model_version": version.strip(),
            "sn": sn.strip(),
            "mo": mo.strip(),
            "reporter": u["display_name"],
            "description": desc.strip(),
            "top_image": top_rel,
            "bottom_image": bot_rel
        }
        insert_firstpiece(payload)
        # keep models listed
        upsert_model(model, "")
        list_models_df.clear()
        load_firstpiece_df.clear()
        st.success("Saved First-Piece.")

def page_nonconf_create():
    st.header("Create Non-Conformity")
    u = st.session_state["user"]
    with st.form("nc_form", clear_on_submit=True):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            model = st.text_input("Model (short)")
        with col2:
            version = st.text_input("Model version (full)")
        with col3:
            sn = st.text_input("SN / Barcode")
        with col4:
            mo = st.text_input("MO / Work Order")
        severity = st.selectbox("Category", ["Minor","Major","Critical"])
        desc = st.text_area("Description of Non-Conformity")

        st.markdown("**More details (optional)**")
        extra_payload = {}
        cols = st.columns(3)
        for i, (label, keyname) in enumerate(NC_EXTRA_FIELDS):
            with cols[i % 3]:
                extra_payload[keyname] = st.text_input(label, key=f"nc_extra_{keyname}")

        photo = st.file_uploader("Main photo (optional JPG/PNG)", type=["jpg","jpeg","png"])
        ok = st.form_submit_button("Save non-conformity")

    if ok:
        photo_rel = ""
        if photo and model:
            photo_rel = save_image_to_model_folder(model, photo)
        payload = {
            "created_at": now_iso(),
            "model_no": model.strip(),
            "model_version": version.strip(),
            "sn": sn.strip(),
            "mo": mo.strip(),
            "reporter": u["display_name"],
            "severity": severity,
            "description": desc.strip(),
            "image_path": photo_rel,
            "extra": json.dumps(extra_payload, ensure_ascii=False),
        }
        insert_nonconf(payload)
        upsert_model(model, "")
        list_models_df.clear()
        load_nonconf_df.clear()
        nonconf_distinct_customers.clear()
        st.success("Saved Non-Conformity.")

def _card_extra_two_cols(extra_dict: dict):
    # Show extra fields in two columns, no nested expanders
    if not extra_dict:
        return
    st.caption("More fields")
    c1, c2 = st.columns(2)
    items = [(k, v) for k, v in extra_dict.items() if f"{v}".strip()]
    left = items[: (len(items) + 1)//2]
    right = items[(len(items) + 1)//2 :]
    with c1:
        for k, v in left:
            st.markdown(f"**{k}:** {v}")
    with c2:
        for k, v in right:
            st.markdown(f"**{k}:** {v}")

def page_search_view():
    st.header("Search & View")

    # Filters (compact)
    with st.container():
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            f_model = st.text_input("Model contains")
        with c2:
            f_version = st.text_input("Version contains")
        with c3:
            f_sn = st.text_input("SN contains")
        with c4:
            f_mo = st.text_input("MO contains")

        c5, c6, c7 = st.columns(3)
        with c5:
            start_date = st.date_input("Start date", value=None)
        with c6:
            end_date = st.date_input("End date", value=None)
        with c7:
            f_text = st.text_input("Text in description/reporter/type")

        st.markdown("**Customer / Supplier**")
        cc1, cc2 = st.columns([1,3])
        with cc1:
            cs_toggle = st.toggle("Pick from list", value=True)
        with cc2:
            if cs_toggle:
                choices = ["(any)"] + nonconf_distinct_customers()
                cs_val_pick = st.selectbox("Select customer/supplier", choices)
                cs_val = "" if cs_val_pick == "(any)" else cs_val_pick
            else:
                cs_val = st.text_input("Contains (free text)", key="nc_cust_text").strip()

        search_col = st.columns([1,4])[0]
        with search_col:
            do_search = st.button("Search", type="primary")

    if not do_search:
        st.info("Enter filters and click **Search** to load results. (Nothing is loaded automatically.)")
        return

    # Load both kinds with filters
    fpd = load_firstpiece_df(
        model_no=f_model, model_version=f_version, sn=f_sn, mo=f_mo,
        start_date=str(start_date) if start_date else None,
        end_date=str(end_date) if end_date else None
    )
    ncd = load_nonconf_df(
        model_no=f_model, model_version=f_version, sn=f_sn, mo=f_mo,
        start_date=str(start_date) if start_date else None,
        end_date=str(end_date) if end_date else None,
        text=f_text
    )

    # filter nonconfs by customer/supplier on parsed extra (client-side)
    if cs_val:
        extra_series = ncd.get("extra", pd.Series([], dtype=object)).apply(parse_extra)
        mask = extra_series.apply(lambda d: cs_val.lower() in (d.get("customer_supplier","").lower()))
        ncd = ncd[mask]

    # Render
    st.subheader(f"First-Piece results · {len(fpd)}")
    for _, r in fpd.iterrows():
        with st.container():
            st.markdown(
                f"**Model:** {r['model_no'] or '-'} | **Version:** {r['model_version'] or '-'} | "
                f"**SN:** {r['sn'] or '-'} | **MO:** {r['mo'] or '-'}"
            )
            st.caption(f"{r['created_at']}  ·  Reporter: {r['reporter'] or '-'}")
            if r["description"]:
                st.write(r["description"])
            # photos in two columns
            colA, colB = st.columns(2)
            if r["top_image"]:
                p = DATA_DIR / str(r["top_image"])
                if p.exists():
                    with colA:
                        st.image(str(p), caption="TOP")
            if r["bottom_image"]:
                p2 = DATA_DIR / str(r["bottom_image"])
                if p2.exists():
                    with colB:
                        st.image(str(p2), caption="BOTTOM")

            # controls
            role = st.session_state["user"]["role"]
            if role in ("Admin", "QA"):
                if st.button("Delete First-Piece", key=f"fp_del_{r['id']}"):
                    delete_firstpiece(int(r["id"]))
                    load_firstpiece_df.clear()
                    st.experimental_rerun()

            st.markdown("---")

    st.subheader(f"Non-Conformity results · {len(ncd)}")
    for _, r in ncd.iterrows():
        with st.container():
            st.markdown(
                f"**Model:** {r['model_no'] or '-'} | **Version:** {r['model_version'] or '-'} | "
                f"**SN:** {r['sn'] or '-'} | **MO:** {r['mo'] or '-'}"
            )
            st.caption(
                f"{r['created_at']}  ·  Reporter: {r['reporter'] or '-'}  ·  Category: {r['severity'] or '-'}"
            )
            if r["description"]:
                st.write(r["description"])
            if r["image_path"]:
                p = DATA_DIR / str(r["image_path"])
                if p.exists():
                    st.image(str(p))

            _card_extra_two_cols(parse_extra(r.get("extra")))

            # Add photo later
            newphoto = st.file_uploader("Add photo (optional)", key=f"addph_{r['id']}", type=["jpg","jpeg","png"])
            if newphoto and r["model_no"]:
                rel = save_image_to_model_folder(r["model_no"], newphoto)
                # if no image_path, set it as main; otherwise ignore (kept simple)
                if not r["image_path"]:
                    with get_conn() as c:
                        c.execute("UPDATE nonconfs SET image_path=? WHERE id=?", (rel, int(r["id"])))
                        c.commit()
                    load_nonconf_df.clear()
                    st.experimental_rerun()
                else:
                    st.info("Photo saved in model folder (not set as main since one exists).")

            role = st.session_state["user"]["role"]
            if role in ("Admin", "QA"):
                if st.button("Delete Non-Conformity", key=f"nc_del_{r['id']}"):
                    delete_nonconf(int(r["id"]))
                    load_nonconf_df.clear()
                    st.experimental_rerun()

            st.markdown("---")

    # Simple export of nonconfs shown
    st.subheader("Export current Non-Conformity results")
    if not ncd.empty:
        buf = io.StringIO()
        ncd.to_csv(buf, index=False)
        st.download_button("Download CSV", buf.getvalue(), file_name="nonconfs_filtered.csv", mime="text/csv")
    else:
        st.caption("Nothing to export.")

def page_import_export():
    st.header("Import / Export")

    # Import Non-Conformities CSV (button-based)
    st.subheader("Import Non-Conformities (CSV)")
    st.write("Columns supported (case-insensitive):")
    st.code("Nonconformity, Description of Nonconformity, Date, Customer/Supplier, "
            "Model/Part No., MO/PO, Line, Work Station, Unit Head, Responsibility, "
            "Root Cause, Corrective Action, Exception reporters, Discovery, Origil Sources, "
            "Defective Item, Defective Outflow, Defective Qty, Inspection Qty, Lot Qty")

    up = st.file_uploader("Choose CSV file…", type=["csv"], key="csv_nc")
    if up:
        if st.button("Load CSV"):
            # safe reading with encoding fallbacks
            df = None
            for enc in ["utf-8-sig", "utf-8", "cp950", "big5", "latin-1"]:
                try:
                    df = pd.read_csv(up, encoding=enc)
                    break
                except Exception:
                    up.seek(0)
            if df is None:
                st.error("Failed to read CSV. Try saving as UTF-8 or Big5/CP950, then retry.")
                return

            # normalize columns
            cols_map = {c.strip().lower(): c for c in df.columns}
            def getcol(name):
                return df[cols_map.get(name.lower())] if cols_map.get(name.lower()) in df.columns else pd.Series([""]*len(df))

            # Build rows
            imported = 0
            u = st.session_state["user"]
            for i in range(len(df)):
                model = str(getcol("Model/Part No.").iloc[i]).strip()
                extra_payload = {
                    "customer_supplier":  str(getcol("Customer/Supplier").iloc[i]).strip(),
                    "line":               str(getcol("Line").iloc[i]).strip(),
                    "work_station":       str(getcol("Work Station").iloc[i]).strip(),
                    "unit_head":          str(getcol("Unit Head").iloc[i]).strip(),
                    "responsibility":     str(getcol("Responsibility").iloc[i]).strip(),
                    "root_cause":         str(getcol("Root Cause").iloc[i]).strip(),
                    "corrective_action":  str(getcol("Corrective Action").iloc[i]).strip(),
                    "exception_reporters":str(getcol("Exception reporters").iloc[i]).strip(),
                    "discovery":          str(getcol("Discovery").iloc[i]).strip(),
                    "origin_sources":     str(getcol("Origil Sources").iloc[i]).strip(),
                    "defective_item":     str(getcol("Defective Item").iloc[i]).strip(),
                    "defective_outflow":  str(getcol("Defective Outflow").iloc[i]).strip(),
                    "defective_qty":      str(getcol("Defective Qty").iloc[i]).strip(),
                    "inspection_qty":     str(getcol("Inspection Qty").iloc[i]).strip(),
                    "lot_qty":            str(getcol("Lot Qty").iloc[i]).strip(),
                }
                payload = {
                    "created_at": str(getcol("Date").iloc[i]).strip() or now_iso(),
                    "model_no": model,
                    "model_version": "",   # not in CSV explicitly
                    "sn": "", "mo": str(getcol("MO/PO").iloc[i]).strip(),
                    "reporter": u["display_name"],
                    "severity": "Minor",
                    "description": str(getcol("Description of Nonconformity").iloc[i]).strip(),
                    "image_path": "",
                    "extra": json.dumps(extra_payload, ensure_ascii=False),
                }
                insert_nonconf(payload)
                if model:
                    upsert_model(model, "")
                imported += 1

            load_nonconf_df.clear()
            nonconf_distinct_customers.clear()
            list_models_df.clear()
            st.success(f"Imported {imported} records.")

    st.markdown("---")
    # Simple full export (admin only)
    if st.session_state["user"]["role"] == "Admin":
        st.subheader("Export ALL Non-Conformities (Admin)")
        with get_conn() as c:
            all_nc = pd.read_sql_query(
                "SELECT id, created_at, model_no, model_version, sn, mo, reporter, severity, description, image_path, extra FROM nonconfs ORDER BY id DESC",
                c
            )
        if not all_nc.empty:
            buf = io.StringIO()
            all_nc.to_csv(buf, index=False)
            st.download_button("Download ALL nonconfs", buf.getvalue(), file_name="nonconfs_all.csv", mime="text/csv")
        else:
            st.caption("No data yet.")

# ───────────────────────── Main ─────────────────────────
st.set_page_config(page_title=APP_TITLE, layout="wide")

require_login()
user = st.session_state["user"]
st.title(APP_TITLE)

# Admin shortcut: user admin page link
bar1, bar2 = st.columns([5,1])
with bar1:
    st.caption("Use the buttons below to navigate.")
with bar2:
    if user["role"] == "Admin":
        if st.button("User setup"):
            st.session_state["page"] = "users"

top_navbar()
st.session_state.setdefault("page", "view")

if st.session_state["page"] == "users":
    if user["role"] != "Admin":
        st.error("Only Admin can access user setup.")
    else:
        page_user_admin()

elif st.session_state["page"] == "fp_create":
    page_firstpiece_create()

elif st.session_state["page"] == "nc_create":
    page_nonconf_create()

elif st.session_state["page"] == "import":
    page_import_export()

else:
    # default
    page_search_view()
