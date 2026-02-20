# Assignment-Week-4
<!-- Back to top link -->
<a name="readme-top"></a>

<br />
<div align="center">
  <a href="https://github.com/[YourGitHubUsername]/[YourRepositoryName]">
    <img src="assets/Logo.png" alt="Project Logo" width="800" height="400">
  </a>

  <h3 align="center">GEOL0069 Week 4: Unsupervised Sea Ice vs. Lead Classification</h3>

  <p align="center">
    This repository contains the completed work for the Week 4 assignment of the UCL module <strong>GEOL0069: Artificial Intelligence for Earth Observation</strong>. The project applies unsupervised learning techniques, specifically Gaussian Mixture Models (GMM), to discriminate between sea ice and leads (open water channels) using Sentinel-3 altimetry data.
    <br />
    <a href="https://github.com/[YourGitHubUsername]/[YourRepositoryName]"><strong>Explore the full repository »</strong></a>
    <br />
    <br />
    <a href="https://github.com/[YourGitHubUsername]/[YourRepositoryName]/issues">Report Bug</a>
    ·
    <a href="https://github.com/[YourGitHubUsername]/[YourRepositoryName]/issues">Request Feature</a>
  </p>

  <!-- Badges -->
  <p>
    <img src="https://img.shields.io/badge/Python-3.10-blue?logo=python" alt="Python">
    <img src="https://img.shields.io/badge/Google%20Colab-F9AB00?logo=googlecolab&logoColor=white" alt="Colab">
    <img src="https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white" alt="sklearn">
    <img src="https://img.shields.io/badge/License-MIT-green" alt="License">
  </p>
</div>

---

<!-- TABLE OF CONTENTS -->
<details>
  <summary><strong>📑 Table of Contents</strong></summary>
  <ol>
    <li><a href="#-introduction-to-unsupervised-learning">Introduction to Unsupervised Learning</a></li>
    <ul>
      <li><a href="#-k-means-clustering">K-Means Clustering</a></li>
      <li><a href="#-gaussian-mixture-models-gmm">Gaussian Mixture Models (GMM)</a></li>
    </ul>
    <li><a href="#-methods--context">Methods & Context</a></li>
    <ul>
      <li><a href="#-project-background">Project Background</a></li>
      <li><a href="#-prerequisites--installation">Prerequisites & Installation</a></li>
    </ul>
    <li><a href="#-detailed-results-with-figures">Detailed Results with Figures</a></li>
    <ul>
      <li><a href="#1-gaussian-mixture-model-example">1. Gaussian Mixture Model Example</a></li>
      <li><a href="#2-mean--standard-deviation-of-echoes">2. Mean & Standard Deviation of Echoes</a></li>
      <li><a href="#3-lead-echo-samples">3. Lead Echo Samples</a></li>
      <li><a href="#4-sea-ice-echo-samples">4. Sea Ice Echo Samples</a></li>
      <li><a href="#5-echo-alignment-examples">5. Echo Alignment Examples</a></li>
      <li><a href="#6-confusion-matrix-vs-esa-classification">6. Confusion Matrix vs. ESA Classification</a></li>
    </ul>
    <li><a href="#-repository-structure">Repository Structure</a></li>
    <li><a href="#-contact">Contact</a></li>
    <li><a href="#-acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

---

## 📖 Introduction to Unsupervised Learning

Unsupervised learning is a branch of machine learning where algorithms identify hidden patterns or structures in data without relying on pre-existing labels. This approach is particularly valuable in Earth Observation, where manually labeling vast datasets is often impractical. The core tasks in this project revolve around classification—grouping radar echoes into meaningful categories based solely on their inherent characteristics. Two fundamental methods are explored here.

### 🧩 K-Means Clustering

K-Means clustering is a centroid-based algorithm that partitions a dataset into a pre-defined number of groups, or clusters (k). It operates by iteratively assigning each data point to the nearest cluster centroid and then recalculating the centroids based on the current assignments. The goal is to minimize the within-cluster variance, often measured by the squared Euclidean distance.

**Why K-Means for this task?**
*   **Simplicity & Scalability:** It is computationally efficient and can handle the large volumes of data typical of satellite altimetry.
*   **Exploratory Power:** It requires no prior knowledge of data distribution, making it ideal for initial investigations into the structure of echo waveforms.

**Key Components:**
*   **Choosing K:** The number of clusters must be specified beforehand. Here, we set K=2 (sea ice and leads).
*   **Centroid Initialization:** The starting points for centroids can influence the final result.
*   **Assignment & Update Steps:** Points are assigned to the nearest centroid, and centroids are updated as the mean of their assigned points. This process repeats until convergence.

### 🧬 Gaussian Mixture Models (GMM)

Gaussian Mixture Models offer a probabilistic approach to clustering. They assume that the data is generated from a mixture of several Gaussian distributions, each with its own mean, variance, and mixing probability. Unlike K-Means, GMM provides a "soft classification," giving the probability that a data point belongs to each cluster.

**Why GMM for this task?**
*   **Soft Clustering:** This is crucial for understanding ambiguity, such as echoes from mixed surfaces or transitional ice types. The probability scores offer insight into classification confidence.
*   **Flexibility in Cluster Shape:** GMM can accommodate clusters with different shapes and orientations (via the covariance matrix), which is more realistic for complex geophysical data.

**Key Components:**
*   **Number of Components (Gaussians):** Analogous to K in K-Means. We set `n_components=2`.
*   **Expectation-Maximization (EM) Algorithm:** The EM algorithm iteratively estimates the parameters of the Gaussians to maximize the likelihood of the observed data.
*   **Covariance Type:** This parameter defines the shape (e.g., spherical, full) each cluster can take.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## 🧠 Methods & Context

### 🛰️ Project Background

The Sentinel-3 mission is a cornerstone of the Copernicus programme, designed to measure Earth's oceans, land, ice, and atmosphere with high accuracy. One of its primary objectives is to measure sea surface topography, which is critical for understanding sea-level rise, ocean circulation, and sea-ice dynamics.

An altimetry satellite works by transmitting radar pulses towards the Earth's surface and measuring the time it takes for the signal to be reflected back. This reflected signal is called an **echo**. The shape, power, and travel time of this echo are directly influenced by the physical properties of the surface it interacted with.

*   **Smooth surfaces** (like calm open water or a refrozen lead) act like mirrors, producing a strong, sharp, and specular echo.
*   **Rough surfaces** (like sea ice with ridges, snow, or deformed ice) scatter the radar pulse in many directions, resulting in a weaker, more diffuse, and often broader echo.

By analyzing these differences in echo shape, we can infer the surface type. This project leverages these physical principles to automatically distinguish between **sea ice** and **leads** (fractures or channels of open water within the ice pack) using unsupervised learning.

### ⚙️ Prerequisites & Installation

This project is designed to be run in **Google Colab**. The following steps will set up the environment.

**1. Mount Google Drive (for Colab)**
All data and the main notebook should be accessible from your Google Drive.
```python
from google.colab import drive
drive.mount('/content/drive')
