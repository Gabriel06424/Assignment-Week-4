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
    An exploration of unsupervised learning techniques to discriminate between sea ice and leads using Sentinel-3 altimetry data.
    <br />
    <a href="https://github.com/[YourGitHubUsername]/[YourRepositoryName]"><strong>Explore the repository »</strong></a>
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
    <img src="https://img.shields.io/badge/Last%20Commit-Active-brightgreen" alt="Last Commit">
  </p>
</div>

---

## 📑 Table of Contents

<details open>
  <summary><strong>View Contents</strong></summary>
  <ol>
    <li><a href="#-introduction-to-unsupervised-learning">Introduction to Unsupervised Learning</a>
      <ul>
        <li><a href="#-k-means-clustering">K-means Clustering</a></li>
        <li><a href="#-gaussian-mixture-models-gmm">Gaussian Mixture Models (GMM)</a></li>
      </ul>
    </li>
    <li><a href="#-methods--workflow">Methods & Workflow</a>
      <ul>
        <li><a href="#-data-preprocessing">Data Preprocessing</a></li>
        <li><a href="#-gmm-implementation">GMM Implementation</a></li>
      </ul>
    </li>
    <li><a href="#-prerequisites--installation">Prerequisites & Installation</a>
      <ul>
        <li><a href="#-google-colab-setup">Google Colab Setup</a></li>
        <li><a href="#-required-packages">Required Packages</a></li>
      </ul>
    </li>
    <li><a href="#-context-sentinel-3-mission">Context: Sentinel-3 Mission</a></li>
    <li><a href="#-detailed-results-with-code-and-figures">Detailed Results with Code and Figures</a>
      <ul>
        <li><a href="#1-gaussian-mixture-model-demonstration">1. Gaussian Mixture Model Demonstration</a></li>
        <li><a href="#2-mean-and-standard-deviation-of-echoes">2. Mean and Standard Deviation of Echoes</a></li>
        <li><a href="#3-lead-echo-samples-cluster-1">3. Lead Echo Samples (Cluster 1)</a></li>
        <li><a href="#4-sea-ice-echo-samples-cluster-0">4. Sea Ice Echo Samples (Cluster 0)</a></li>
        <li><a href="#5-echo-alignment-examples">5. Echo Alignment Examples</a></li>
        <li><a href="#6-confusion-matrix-vs-esa-classification">6. Confusion Matrix vs. ESA Classification</a></li>
      </ul>
    </li>
    <li><a href="#-repository-structure">Repository Structure</a></li>
    <li><a href="#-contact">Contact</a></li>
    <li><a href="#-acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

---

## 🧠 Introduction to Unsupervised Learning

Unsupervised learning is a branch of machine learning where algorithms identify patterns and structures in data without relying on pre-existing labels. This makes it particularly valuable for Earth Observation applications, where ground truth data may be limited or unavailable. For this project, we focus on classification tasks—specifically, distinguishing between sea ice and leads based solely on the intrinsic properties of the radar returns.

### 🔹 K-means Clustering

K-means is a centroid-based algorithm that partitions data into a predetermined number of clusters (k). It works by iteratively assigning each data point to the nearest cluster centroid and then recalculating the centroids based on these assignments. The algorithm minimizes within-cluster variance, making it efficient for datasets where clusters are roughly spherical and evenly sized. Key components include choosing the optimal k, centroid initialization, and the iterative assignment-update process.

### 🔹 Gaussian Mixture Models (GMM)

GMM takes a probabilistic approach, assuming that the data is generated from a mixture of multiple Gaussian distributions. Unlike K-means, GMM provides a "soft classification" by estimating the probability that each data point belongs to each cluster. This is particularly useful for understanding uncertainty in classifications. The model uses the Expectation-Maximization (EM) algorithm to iteratively refine the parameters (means, covariances, and mixing coefficients) of the component distributions. GMM's flexibility in accommodating different cluster shapes and sizes makes it well-suited for the natural variability observed in sea ice and lead backscatter.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## ⚙️ Methods & Workflow

### 📊 Data Preprocessing

Before applying unsupervised learning algorithms, the raw Sentinel-3 altimetry data must be transformed into meaningful features. The key variables derived for this analysis are:

*   **Peakiness**: A measure of how sharp or peaked an echo waveform is. Leads, acting as mirror-like surfaces, typically produce very sharp, high-peak echoes.
*   **Stack Standard Deviation (SSD)**: Represents the variability within a stack of echoes. Rough surfaces like sea ice tend to cause more variability in the return signal.

Data cleaning is also essential, involving the removal of any invalid or NaN values that could skew the clustering results.

### 🤖 GMM Implementation

The GMM model is implemented using the `GaussianMixture` class from the `sklearn.mixture` library. The key steps are:

1.  **Initialization**: The model is initialized with `n_components=2` (for two classes: sea ice and lead) and a fixed `random_state` to ensure reproducibility.
2.  **Fitting**: The model is fitted to the preprocessed feature matrix (X) using the `fit()` method, which runs the EM algorithm to estimate the optimal Gaussian distributions.
3.  **Prediction**: Cluster labels for each data point are obtained using the `predict()` method. These labels are then mapped to their physical interpretations (sea ice or lead) based on the characteristics of the cluster centers (e.g., higher peakiness indicates leads).

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## 🚀 Prerequisites & Installation

### ☁️ Google Colab Setup

This project is designed to be run in Google Colab. The first step is to mount your Google Drive to access the data.

```python
from google.colab import drive
drive.mount('/content/drive')
