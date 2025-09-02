import os
import sqlite3
from pathlib import Path
from datetime import datetime

import streamlit as st
from PIL import Image

# --- Writable paths on Streamlit Cloud (works locally too) ---
DATA_DIR = Path("/mount/data")
IMG_DIR  = DATA_DIR / "images"
DB_PATH  = DATA_DIR / "qc_portal.sqlite3"

for p in [DATA_DIR, IMG_DIR]:
    p.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="QC Portal • Smoke Test", layout="centered")
st.title("✅ QC Portal — Smoke Test")

# Show where we are writing
st.write("**DATA_DIR:**", str(DATA_DIR))
st.write("**IMG_DIR:**", str(IMG_DIR))
st.write("**DB_PATH:**", str(DB_PATH))

# --- 1) Try writing a simple file ---
try:
    mark = DATA_DIR / "write_check.txt"
    mark.write_text(f"Write OK at {datetime.utcnow().isoformat()}\n", encoding="utf-8")
    st.success(f"Write test: created {mark.name}")
except Exception as e:
    st.error(f"Write test FAILED: {e}")

# --- 2) Try SQLite read/write ---
try:
    with sqlite3.connect(DB_PATH) as c:
        c.execute("""CREATE TABLE IF NOT EXISTS ping (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        ts TEXT
                     )""")
        c.execute("INSERT INTO ping(ts) VALUES(?)", (datetime.utcnow().isoformat(),))
        count = c.execute("SELECT COUNT(*) FROM ping").fetchone()[0]
    st.success(f"SQLite test: table 'ping' exists, rows = {count}")
except Exception as e:
    st.error(f"SQLite test FAILED: {e}")

# --- 3) Image upload/save test ---
st.subheader("📷 Upload a test image (optional)")
u = st.file_uploader("Choose an image", type=["jpg","jpeg","png"])
if u is not None:
    try:
        img = Image.open(u).convert("RGB")
        out = IMG_DIR / f"smoketest_{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}.jpg"
        img.save(out, format="JPEG", quality=90)
        st.success(f"Saved image to: {out}")
        st.image(str(out), caption="Saved image preview", use_container_width=True)
    except Exception as e:
        st.error(f"Image save FAILED: {e}")

st.info("If all sections above show green ✅, the environment is ready. Next step: add real features.")
