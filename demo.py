import random
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image
import time
st.set_page_config(page_title="QPI Dataset Viewer", layout="wide")

# ==========================
# Dataset Paths
# ==========================

X_DIR = Path("/sda/usama/lightweight_qpi_segmentation/dataset/X_train")
Y_DIR = Path("/sda/usama/lightweight_qpi_segmentation/dataset/Y_train")

files = sorted(
    list(X_DIR.glob("*.tif"))
    + list(X_DIR.glob("*.tiff"))
    + list(X_DIR.glob("*.TIF"))
    + list(X_DIR.glob("*.TIFF"))
)

if len(files) == 0:
    st.error("No training images found.")
    st.stop()

# =====================================
# Keep the same image for 10 minutes
# =====================================

CHANGE_INTERVAL = 600  # 10 minutes (600 seconds)

if "current_image" not in st.session_state:
    st.session_state.current_image = random.choice(files)
    st.session_state.last_change = time.time()

# Manual button still works
if st.button("🎲 Next Random Image"):
    st.session_state.current_image = random.choice(files)
    st.session_state.last_change = time.time()

# Automatically change every 10 minutes
if time.time() - st.session_state.last_change > CHANGE_INTERVAL:
    st.session_state.current_image = random.choice(files)
    st.session_state.last_change = time.time()
    st.rerun()

img_path = st.session_state.current_image

mask_path = None
for ext in [".tif", ".tiff", ".TIF", ".TIFF"]:
    p = Y_DIR / (img_path.stem + ext)
    if p.exists():
        mask_path = p
        break

if mask_path is None:
    st.error(f"No mask found for {img_path.name}")
    st.stop()

image = np.array(Image.open(img_path))
mask = np.array(Image.open(mask_path))

st.title("QPI Segmentation Dataset Viewer")
st.write(f"### {img_path.name}")

# Debug information
st.sidebar.header("Mask Information")
st.sidebar.write(f"Shape: {mask.shape}")
st.sidebar.write(f"Data type: {mask.dtype}")
st.sidebar.write(f"Min: {mask.min()}")
st.sidebar.write(f"Max: {mask.max()}")

unique = np.unique(mask)
if len(unique) <= 20:
    st.sidebar.write("Unique values:", unique)
else:
    st.sidebar.write(f"{len(unique)} unique values")

# Prepare mask for display
display_mask = mask.copy()

# Binary mask (0/1)
if display_mask.ndim == 2 and display_mask.max() <= 1:
    display_mask = display_mask.astype(np.uint8) * 255

# Other grayscale masks
elif display_mask.ndim == 2 and display_mask.max() > 0:
    display_mask = (
        display_mask.astype(np.float32)
        / display_mask.max()
        * 255
    ).astype(np.uint8)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Input Image")
    st.image(image, width="stretch")

with col2:
    st.subheader("Ground Truth Mask")
    st.image(display_mask, width="stretch")