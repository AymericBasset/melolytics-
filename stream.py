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

    if st.button('Generate Zip Images for Maks'):
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
        # Button to estimate density and segmentation
    if st.sidebar.button('Estimate Density'):
        # Step 1: Segment the astrocyte bodies using the given function
        def detect_and_segment_astrocyte_bodies(binary_image, original_image):
            # NAME : astrocytes_of_interest.png
            fig, axes = plt.subplots(1, 3, figsize=(36, 20))

            axes[0].imshow(original_image, cmap='gray')
            axes[0].set_title('Original Image')
            axes[0].axis('off')

            axes[1].imshow(overlay_image)
            axes[1].set_title('Overlay of Segmented Astrocyte Bodies')
            axes[1].axis('off')

            axes[2].imshow(segmented_mask, cmap='gray')
            axes[2].set_title('Segmented Astrocyte Bodies')
            axes[2].axis('off')

            plt.tight_layout()

            # Save figure to a BytesIO object
            buffer = BytesIO()
            plt.savefig(buffer, format='PNG')
            buffer.seek(0)
            fig_image = Image.open(buffer)

            st.pyplot(fig)

            return segmented_mask, fig_image

        # Step 2: Segment astrocyte bodies
        segmented_astrocytes, astrocytes_of_interest_img = detect_and_segment_astrocyte_bodies(
            (cell_otsu > 0).astype(int), cell_img)

        # Step 3: Plot the segmented image and centroids for visualization
        def analyze_astrocyte_aggregation_knn(segmented_mask):
            # NAME : segmented_astrocytes.png
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(segmented_mask, cmap='gray')
            ax.scatter(centroids[:, 1], centroids[:, 0],
                       c='red', s=10, label='Centroids')
            ax.set_title('Segmented Astrocyte Mask with Centroids')
            ax.axis('off')
            plt.legend()
            plt.tight_layout()

            # Save figure to a BytesIO object
            buffer = BytesIO()
            plt.savefig(buffer, format='PNG')
            buffer.seek(0)
            fig_image = Image.open(buffer)

            st.pyplot(fig)

            return features_df, fig_image

        # Step 4: Analyze and plot astrocyte aggregation with centroids highlighted
        features_df_knn, segmented_astrocytes_img = analyze_astrocyte_aggregation_knn(
            segmented_astrocytes)

        # Step 5: Analyze k-nearest neighbors density and display using Plotly
        def analyze_astrocyte_aggregation_knn_density(segmented_mask):
            # Step 4: Create subplots with Plotly to show both graphs side by side
            # NAME : kde_and_distance_plot.png
            fig = sp.make_subplots(rows=1, cols=2, subplot_titles=(
                "Density Plot of Astrocytes", "Mean Distance vs. K-Value"))

            # Step 5: Create the density plot for the astrocytes
            if len(centroids) > 0:
                x_coords = centroids[:, 1]
                y_coords = centroids[:, 0]

                # Add the density plot as a contour to the first subplot
                density_trace = go.Histogram2dContour(
                    x=x_coords,
                    y=y_coords,
                    colorscale=[[0, 'white'], [1, 'green']],
                    reversescale=False,
                    ncontours=20,
                    showscale=False
                )
                fig.add_trace(density_trace, row=1, col=1)

                # Fix axis limits to match original image size
                fig.update_xaxes(
                    range=[0, segmented_mask.shape[1]], row=1, col=1)
                fig.update_yaxes(
                    range=[segmented_mask.shape[0], 0], row=1, col=1, autorange=False)

            # Step 6: Create the mean distance vs. k-value plot for the second subplot
            mean_distance_trace = go.Scatter(
                x=list(k_values),
                y=knn_distances_mean,
                mode='lines',
                name='Mean Distance',
                line=dict(color='blue')
            )

            std_upper = np.array(knn_distances_mean) + \
                np.array(knn_distances_std)
            std_lower = np.array(knn_distances_mean) - \
                np.array(knn_distances_std)

            # Add shaded area for standard deviation
            std_shading = go.Scatter(
                x=list(k_values) + list(k_values)[::-1],
                y=list(std_upper) + list(std_lower)[::-1],
                fill='toself',
                fillcolor='rgba(0, 0, 255, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='Standard Deviation'
            )

            fig.add_trace(std_shading, row=1, col=2)
            fig.add_trace(mean_distance_trace, row=1, col=2)

            # Step 7: Update layout and axes for better visualization
            fig.update_layout(
                title_text='Astrocyte Aggregation Analysis',
                width=1000,
                height=500,
                plot_bgcolor='white'
            )

            fig.update_xaxes(title_text="X", row=1, col=1, showgrid=False)
            fig.update_yaxes(title_text="Y", row=1, col=1, showgrid=False)

            fig.update_xaxes(title_text="K-Value", row=1,
                             col=2, showgrid=False)
            fig.update_yaxes(
                title_text="Mean Distance to K-Nearest Neighbors", row=1, col=2, showgrid=False)

            st.plotly_chart(fig)

            # Save Plotly figure as image
            buffer = BytesIO()
            fig.write_image(buffer, format='png')
            buffer.seek(0)
            fig_image = Image.open(buffer)

            return features_df, fig_image

        # Step 6: Analyze density and display the results
        features_df_density, kde_and_distance_img = analyze_astrocyte_aggregation_knn_density(
            segmented_astrocytes)
        st.write(features_df_density)

        # Button to generate a ZIP with all images
        if st.button('Generate Zip Images for Density'):
            images = {
                "astrocytes_of_interest.png": astrocytes_of_interest_img,
                "segmented_astrocytes.png": segmented_astrocytes_img,
                "kde_and_distance_plot.png": kde_and_distance_img,
            }
            zip_buffer = create_zip(images)
            st.download_button(
                label="Download All Images as ZIP",
                data=zip_buffer,
                file_name="processed_images.zip",
                mime="application/zip"
            )
