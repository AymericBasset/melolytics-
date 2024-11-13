import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import threshold_multiotsu
import pandas as pd
from skimage import measure, morphology
from skimage.segmentation import watershed
from scipy import ndimage as ndi
from scipy.spatial import distance
import plotly.graph_objects as go
import plotly.subplots as sp

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

    def preprocess_image(binary_img, kernel_close_size, kernel_erode_size, area_threshold):
        # Apply morphological operations
        kernel_close = np.ones(
            (kernel_close_size, kernel_close_size), np.uint8)
        closed_img = cv2.morphologyEx(
            binary_img, cv2.MORPH_CLOSE, kernel_close)

        # Apply erosion to separate less dense regions
        kernel_erode = np.ones(
            (kernel_erode_size, kernel_erode_size), np.uint8)
        eroded_img = cv2.erode(closed_img, kernel_erode)

        # Find contours in the eroded image using RETR_TREE
        contours, hierarchy = cv2.findContours(
            eroded_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Create a mask from the contours
        mask = np.zeros_like(binary_img, dtype=np.uint8)

        # Iterate over contours and use the hierarchy to determine the relationship between contours
        for idx, contour in enumerate(contours):
            # Check if the area is greater than the threshold
            if cv2.contourArea(contour) > area_threshold:
                # Draw outer contour
                if hierarchy[0][idx][3] == -1:  # Only draw if it's an external contour
                    cv2.drawContours(
                        mask, [contour], -1, color=255, thickness=cv2.FILLED)
                else:
                    # If the contour is a child (hole), draw it in black to "carve out" the hole
                    cv2.drawContours(
                        mask, [contour], -1, color=0, thickness=cv2.FILLED)

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

    # Button to download all pictures
    from PIL import Image
    from io import BytesIO
    import zipfile

    def create_zip(images):
        # Create a zip file in memory
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zf:
            for filename, image in images.items():
                img_byte_arr = BytesIO()
                image.save(img_byte_arr, format='PNG')
                zf.writestr(filename, img_byte_arr.getvalue())
        zip_buffer.seek(0)
        return zip_buffer

    if st.button('Download All Images'):
        # Create images for download
        images = {
            "virus_original.png": Image.fromarray(marker_img),
            "astrocyte_original.png": Image.fromarray(cell_img),
            "virus_mask.png": Image.fromarray(virus_mask),
            "astrocyte_mask.png": Image.fromarray(astrocyte_mask),
            "combined_mask.png": Image.fromarray(combined_mask)
        }
        zip_buffer = create_zip(images)
        st.download_button(
            label="Download All Images as ZIP",
            data=zip_buffer,
            file_name="processed_images.zip",
            mime="application/zip"
        )

    # Function to save density plots
    def save_density_plots(fig, filename):
        img_buffer = BytesIO()
        fig.savefig(img_buffer, format='png')
        img_buffer.seek(0)
        return Image.open(img_buffer)

    # Example plots for density analysis
    if st.sidebar.button('Estimate Density'):
        # Placeholder density plot (replace with actual plots as needed)
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        st.pyplot(fig)

        # Save the plot for download
        density_plot_image = save_density_plots(fig, "density_plot.png")

        # Button to download density plot
        if st.button('Download Density Plot'):
            images = {
                "density_plot.png": density_plot_image
            }
            zip_buffer = create_zip(images)
            st.download_button(
                label="Download Density Plot as ZIP",
                data=zip_buffer,
                file_name="density_plot.zip",
                mime="application/zip"
            )
