# How Do News Media Visually Frame AI?

## 1. About this notebook

This notebook analyzes a set of roughly 200 images sampled from online news articles about artificial intelligence, drawn from four outlets: **NOS, NU.nl, The New York Times, and Reuters**.

To explore this, the notebook builds a small image-analysis pipeline that:
1. Extracts low-level visual features (brightness, edges, texture and color) and runs object detection on each image.
2. Groups the images into clusters based on those features, using two different clustering techniques.
3. Attempts to interpret and label the resulting clusters, and reflects on what such an unsupervised approach can and cannot tell us about visual framing.

There is no single "correct" clustering solution here. The notebook is an exploratory, critical walkthrough of the strengths and limitations of using unsupervised machine learning on image data.

## 2. Step-by-step walkthrough

### Step 1: Load and prepare the image data
- Installs the required image-processing packages (`pillow-heif`, `tqdm`, `pandas`, `opencv-python`).
- Unzips the provided `IMAGES_TMA.zip` file, which contains the raw images organized by outlet folder (NOS, NU.NL, NYT, Reuters).
- Converts every image to a standard JPEG format (some images may originally be `.png`, `.heic`, `.webp`, etc.) and saves them into a new `processed_jpg` folder.
- While doing this, it catalogues each image in a table (`df_2`) with information like filename, dimensions, file size, and which outlet it came from. Each image also gets a unique label, e.g. `NOS_1`, `Reuters_12`.

### Step 2: Enrich the dataset with image features
This step turns each image into a set of numbers a computer can compare. For every image, the notebook computes:
- **Gray brightness** — the average pixel intensity, i.e. how light or dark the image is overall.
- **Canny edge density** — the proportion of pixels detected as "edges," giving a sense of how visually busy or detailed the image is (e.g. text overlays, sharp contours).
- **Sobel texture** — a measure of how much the image's gradients (light-to-dark transitions) vary, capturing smoothness vs. detail.
- **Color histograms** — the average intensity of the red, green, and blue color channels, summarizing the dominant color palette.
- **Object detection (YOLOv8)** — a pretrained YOLO model is run on a sample image to detect and label objects within it (e.g. people, screens, logos), adding a layer of semantic (content-based) information on top of the low-level visual features.

The step ends with a discussion of why these particular features were chosen and what they capture.

### Step 2: Cluster the images
Now that each image has a set of numeric features, the notebook groups similar images together:
- The six numeric features (brightness, edge density, texture, and the three color means) are **standardized** (rescaled so they're on comparable scales) using `StandardScaler`.
- **K-Means clustering** is applied. The **elbow method** (plotting how clustering error decreases as the number of clusters increases) is used to help decide on a reasonable number of clusters — the notebook settles on **k = 3**.
- **Agglomerative (hierarchical) clustering** is applied as a second, different technique, also producing 3 clusters. A **dendrogram** is plotted to visualize how images merge together at different similarity thresholds.
- The two clustering solutions are evaluated using **silhouette score** and **Davies-Bouldin index**, two standard metrics for judging how well-separated clusters are.
- Example images from each cluster are displayed so the results can be inspected visually, alongside a table of average feature values per cluster.

### Step 3: Label and interpret the clusters
The final step tries to make sense of what each cluster represents:
- Cluster sizes are compared between the K-Means and hierarchical solutions.
- Representative sample images are shown for each cluster.
- A bar chart compares the average feature values (brightness, edge density, texture, color) across clusters, to help characterize what distinguishes one cluster from another.
- The notebook discusses candidate interpretations for each cluster (e.g. one cluster tending toward screenshots/text overlays with high edge density and low brightness; another brighter and smoother; another with elevated blue tones and more formal/institutional imagery), while also being explicit about the limitations of labeling clusters based only on low-level visual features, and where semantic tools like object detection or embeddings (e.g. CLIP) could help in future work.

## 3. How to run the notebook

This notebook was written to run in **Google Colab**, since it relies on `/content/` paths and installs packages via `!pip`.

### Requirements
- Python 3 environment (Google Colab recommended)
- Packages installed within the notebook: `pillow-heif`, `tqdm`, `pandas`, `opencv-python`, `numpy`, `matplotlib`, `scikit-learn`, `scipy`, `ultralytics` (for YOLOv8)
- The image dataset: `IMAGES_TMA.zip`

### Steps to run
1. **Get the data.** Obtain `IMAGES_TMA.zip` (the image dataset for this exercise) and upload it to your Colab environment so that it is available at `/content/IMAGES_TMA.zip`.
2. **Open the notebook** in Google Colab (or Jupyter, if you adjust the file paths).
3. **Run the cells in order, from top to bottom.** The notebook:
   - installs dependencies,
   - unzips and processes the images into `df_2`,
   - extracts image features and runs object detection,
   - performs clustering,
   - and visualizes/labels the clusters.
4. **Object detection cell:** the first time YOLOv8 is used, `ultralytics` will automatically download the pretrained `yolov8n.pt` model weights — make sure you have an internet connection.
5. Plots and sample images will render inline as each cell is executed; no additional configuration is needed.

**Note:** Because clustering here (K-Means, hierarchical) partly depends on `random_state` settings and on which image is randomly sampled for display, exact cluster compositions and displayed example images may vary slightly between runs unless a fixed seed is set for all random operations.
