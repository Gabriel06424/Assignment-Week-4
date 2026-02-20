# Assignment-Week-4
# GEOL0069 – Week 4  
**Unsupervised Separation of Sea Ice and Leads in Sentinel-3 Altimetry Waveforms**

<div align="center">
  <br>
  <img src="https://sentinels.copernicus.eu/documents/247904/1877131/S3_SRAL_Product_Overview.png" alt="Sentinel-3 SRAL" width="420">
  <h4>Gaussian Mixture Model classification of radar echoes</h4>
</div>

<br>

This repository contains the Week 4 assignment submission for the module **GEOL0069 Artificial Intelligence for Earth Observation** (UCL Earth Sciences).  
The notebook extends the provided material (`Chapter1_Unsupervised_Learning_Methods_2.ipynb`) by applying Gaussian Mixture Modelling to discriminate sea ice from leads, computing class-representative echo shapes (mean ± std), aligning waveforms to reduce bin-shift artefacts, and quantitatively comparing the unsupervised result against the official ESA surface-type classification using a confusion matrix.

<p align="right">(<a href="#top">back to top</a>)</p>

## Table of Contents

- [Introduction](#introduction)
  - [K-means Clustering – Brief Overview](#k-means-clustering--brief-overview)
  - [Gaussian Mixture Models (GMM) – Core Method](#gaussian-mixture-models-gmm--core-method)
- [Methods & Prerequisites](#methods--prerequisites)
- [Scientific Context](#scientific-context)
- [Results & Interpretation](#results--interpretation)
  - [1. GMM Cluster Assignment in Feature Space](#1-gmm-cluster-assignment-in-feature-space)
  - [2. Mean Waveform and Standard Deviation per Class](#2-mean-waveform-and-standard-deviation-per-class)
  - [3. Waveform Distribution – Lead Class](#3-waveform-distribution--lead-class)
  - [4. Waveform Distribution – Sea Ice Class](#4-waveform-distribution--sea-ice-class)
  - [5. Waveform Alignment Examples (Original vs Aligned)](#5-waveform-alignment-examples-original-vs-aligned)
  - [6. Confusion Matrix – GMM vs ESA Reference Classification](#6-confusion-matrix--gmm-vs-esa-reference-classification)
- [Repository Files](#repository-files)
- [Acknowledgments](#acknowledgments)
- [Contact](#contact)

<a id="introduction"></a>
## Introduction

The primary goal of this assignment is to apply unsupervised learning techniques to Sentinel-3 SRAL (Ku-band) radar waveforms to distinguish between **sea ice** and **leads** (open water fractures within the ice pack).  

Two main clustering approaches are relevant in this context:

### K-means Clustering – Brief Overview

K-means is a simple, distance-based partitioning method that minimises within-cluster variance using Euclidean distance to centroids. While fast and scalable, it assumes spherical clusters of similar size and density, which is often not ideal for altimetry-derived features (peakiness, σ⁰, SSD) that can show elongated or overlapping distributions.

### Gaussian Mixture Models (GMM) – Core Method

GMM assumes that the data are generated from a mixture of several Gaussian distributions. It uses the Expectation-Maximization (EM) algorithm to estimate means, covariances, and mixing coefficients.  

**Key advantages in this application**:
- soft (probabilistic) assignments instead of hard labels
- ability to model elliptical / anisotropic clusters
- better capture of uncertainty in echo feature distributions

Due to these properties, GMM was selected as the main method for the analysis.

<p align="right">(<a href="#top">back to top</a>)</p>

<a id="methods--prerequisites"></a>
## Methods & Prerequisites

### Prerequisites (Google Colab environment)

```bash
!pip install netCDF4 cartopy
