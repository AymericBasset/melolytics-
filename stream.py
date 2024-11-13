import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import threshold_multiotsu
import pandas as pd

# Title of the app
st.title('Virus and Astrocyte Analysis Tool')

# File upload section for both images
st.sidebar.header('Upload Images')
virus_file = st.sidebar.file_uploader(
    "Upload Virus Image (JPG)", type=["jpg", "jpeg", "png"])
astrocyte_file = st.sidebar.file_uploader(
    "Upload Astrocyte Image (JPG)", type=["jpg", "jpeg", "png"])

# Function to apply Otsu thresholding


def apply_otsu_threshold(image):
    thresholds = threshold_multiotsu(image, classes=3)
    regions = np.digitize(image, bins=thresholds)
    return regions


# If files are uploaded, proceed
if virus_file and astrocyte_file:
    # Load images in grayscale
    marker_img = cv2.imdecode(np.frombuffer(
        virus_file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    cell_img = cv2.imdecode(np.frombuffer(
        astrocyte_file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)

    # Apply Otsu thresholding
    marker_otsu = apply_otsu_threshold(marker_img)
    cell_otsu = apply_otsu_threshold(cell_img)

    # Sidebar options for preprocessing parameters
    st.sidebar.header('Preprocessing Parameters')
    kernel_close_size_virus = st.sidebar.slider(
        'Virus - Closing Kernel Size', 1, 200, 90, step=1)
    kernel_erode_size_virus = st.sidebar.slider(
        'Virus - Erosion Kernel Size', 1, 50, 10, step=1)
    area_threshold_virus = st.sidebar.slider(
        'Virus - Area Threshold', 5000, 50000, 20000, step=1000)

    kernel_close_size_astro = st.sidebar.slider(
        'Astrocytes - Closing Kernel Size', 1, 200, 80, step=1)
    kernel_erode_size_astro = st.sidebar.slider(
        'Astrocytes - Erosion Kernel Size', 1, 50, 8, step=1)
    area_threshold_astro = st.sidebar.slider(
        'Astrocytes - Area Threshold', 5000, 50000, 15000, step=1000)

    # Preprocessing function
    def preprocess_image(binary_img, kernel_close_size, kernel_erode_size, area_threshold):
        kernel_close = np.ones(
            (kernel_close_size, kernel_close_size), np.uint8)
        closed_img = cv2.morphologyEx(
            binary_img, cv2.MORPH_CLOSE, kernel_close)
        kernel_erode = np.ones(
            (kernel_erode_size, kernel_erode_size), np.uint8)
        eroded_img = cv2.erode(closed_img, kernel_erode)
        contours, _ = cv2.findContours(
            eroded_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros_like(binary_img, dtype=np.uint8)
        for contour in contours:
            if cv2.contourArea(contour) > area_threshold:
                cv2.drawContours(mask, [contour], -1,
                                 color=255, thickness=cv2.FILLED)
        return mask, contours

    # Preprocess virus and astrocyte images
    virus_mask, contours_virus = preprocess_image(binary_img=np.where(marker_otsu > 0, 255, 0).astype(np.uint8),
                                                  kernel_close_size=kernel_close_size_virus,
                                                  kernel_erode_size=kernel_erode_size_virus,
                                                  area_threshold=area_threshold_virus)

    astrocyte_mask, contours_astrocytes = preprocess_image(binary_img=np.where(cell_otsu > 0, 255, 0).astype(np.uint8),
                                                           kernel_close_size=kernel_close_size_astro,
                                                           kernel_erode_size=kernel_erode_size_astro,
                                                           area_threshold=area_threshold_astro)

    # Combine masks for visualization
    combined_mask = np.zeros_like(marker_img, dtype=np.uint8)
    combined_mask[virus_mask == 255] = 50  # Light blue for virus
    combined_mask[astrocyte_mask == 255] = 100  # Light green for astrocyte
    combined_mask[(virus_mask == 255) & (astrocyte_mask == 255)
                  ] = 150  # Yellow for overlap

    # Plot results
    st.header("Preprocessing Results")
    col1, col2 = st.columns(2)

    with col1:
        st.image(marker_img, caption='Virus - Original', use_column_width=True)
        st.image(cell_img, caption='Astrocytes - Original',
                 use_column_width=True)

    with col2:
        st.image(virus_mask, caption='Virus - Mask', use_column_width=True)
        st.image(astrocyte_mask, caption='Astrocytes - Mask',
                 use_column_width=True)
        st.image(combined_mask, caption='Combined Mask', use_column_width=True)

    # Calculate metrics
    perimeter_virus = sum(cv2.arcLength(contour, True)
                          for contour in contours_virus if cv2.contourArea(contour) > area_threshold_virus)
    area_virus = sum(cv2.contourArea(contour)
                     for contour in contours_virus if cv2.contourArea(contour) > area_threshold_virus)

    perimeter_astro = sum(cv2.arcLength(contour, True)
                          for contour in contours_astrocytes if cv2.contourArea(contour) > area_threshold_astro)
    area_astro = sum(cv2.contourArea(contour)
                     for contour in contours_astrocytes if cv2.contourArea(contour) > area_threshold_astro)

    area_intersection = np.sum((virus_mask == 255) & (astrocyte_mask == 255))

    virus_mean_all = (marker_otsu > 0).mean().round(4)*100
    astro_mean_all = (cell_otsu > 0).mean().round(4)*100
    virus_mean_all_to_astro_mean_all = (virus_mean_all/astro_mean_all).round(2)

    # Masked pixel coverage stats
    masked_marker_otsu = cv2.bitwise_and(
        marker_otsu, marker_otsu, mask=virus_mask)
    masked_cell_otsu = cv2.bitwise_and(cell_otsu, cell_otsu, mask=virus_mask)

    virus_mean_masked = (masked_marker_otsu > 0).mean().round(4) * 100
    astro_mean_masked = (masked_cell_otsu > 0).mean().round(4) * 100
    virus_to_astro_ratio_masked = (
        virus_mean_masked / astro_mean_masked).round(2)

    # Summary Table
    st.header("Summary of Analysis")
    summary_data = {
        "Metric": ["Virus Perimeter", "Virus Area", "Astrocytes Perimeter", "Astrocytes Area", "Intersection Area",
                   "Virus Coverage (%)", "Astrocyte Coverage (%)", "Virus/Astrocyte Coverage Ratio",
                   "Masked Virus Coverage (%)", "Masked Astrocyte Coverage (%)", "Masked Virus/Astrocyte Coverage Ratio"],
        "Value": [perimeter_virus, area_virus, perimeter_astro, area_astro, area_intersection,
                  virus_mean_all, astro_mean_all, virus_mean_all_to_astro_mean_all,
                  virus_mean_masked, astro_mean_masked, virus_to_astro_ratio_masked]
    }
    summary_df = pd.DataFrame(summary_data)
    st.table(summary_df)
