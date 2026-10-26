# BuildAhead — Satellite-Based Construction Change Detection

BuildAhead is a full-stack geospatial web application for **multi-temporal satellite-image change detection and construction-related change classification**, built using genuine Planet Dove optical satellite mosaics and tracked building-footprint annotations from the official **SpaceNet 7 Multi-Temporal Urban Development Dataset**.

> **Scope Statement:** BuildAhead detects and classifies historical surface and building-related changes between two satellite observation dates. Detecting historical change is **not** equivalent to predicting future construction, and ~4.8 m/px optical pixel differences do not replace high-resolution or municipal permit verification.

---

## 1. Quick Start & Setup Instructions

### Prerequisites
- Node.js `>= 22.x` and `npm`

### Installation & Development Server
```bash
npm install
npm run dev
```
Open `http://localhost:3000`. All 8 held-out SpaceNet 7 evaluation tile pairs and ground-truth building footprint annotations are preloaded in `public/data/spacenet7/`, so the application works immediately with zero external downloads or API keys.

### Run Automated Verification Tests
```bash
npm test
```

### Production Build & Start
```bash
npm run build
NODE_ENV=production npm start
```

### Optional: Rebuild Dataset & Re-Train Classifier from Official SpaceNet 7 S3 Bucket
```bash
npx tsx scripts/build_spacenet7_dataset.ts
```

---

## 2. Dataset Provenance & Preloaded Manifest

- **Official Source:** [SpaceNet 7 Multi-Temporal Urban Development Challenge](https://spacenet.ai/sn7-challenge/)
- **AWS Open Data Registry Bucket:** `s3://spacenet-dataset/spacenet/SN7_buildings/train/`
- **Attribution:** SpaceNet Partners, CosmiQ Works, Planet Labs Inc. (Global Monthly Basemaps), Capella Space, Maxar Technologies, IEEE GRSS, Amazon Web Services.
- **License:** Creative Commons Attribution 4.0 International (`CC-BY-4.0`).
- **Preloaded Pack (`public/data/spacenet7/manifest.json`):**
  - 8 genuine native-resolution `512 × 512` pixel sub-tiles (`~2.44 km × 2.44 km` at `~3.5–4.7 m/px` latitude-corrected Ground Sample Distance) extracted directly from official SpaceNet 7 GeoTIFFs (`images_masked/`) and paired with official `labels_match_pix/` building footprints.
  - **4 Positive Held-Out Samples** (substantial multi-building construction verified in ground-truth footprints):
    - `sn7-test-01`: Meridian / Boise Metro, Idaho, USA (`2018-01` → `2020-01`, `+409` new buildings)
    - `sn7-test-02`: West Abdullah Al-Mubarak, Kuwait (`2018-01` → `2020-01`, `+2,315` new buildings)
    - `sn7-test-03`: Northern Growth Corridor, Melbourne, Australia (`2018-01` → `2020-01`, `+706` new buildings)
    - `sn7-test-04`: Port St. Lucie Corridor, Florida, USA (`2018-01` → `2020-01`, `+68` new buildings)
  - **4 Negative Held-Out Samples** (zero or minimal building-related change despite seasonal/radiometric variation):
    - `sn7-test-05`: Rotterdam Polder & Urban Fringe, Netherlands (`2018-04` → `2018-08`, `+0` new buildings)
    - `sn7-test-06`: Lehigh Valley Agricultural Fringe, Pennsylvania, USA (`2018-04` → `2018-08`, `+1` new building)
    - `sn7-test-07`: Permian Basin Semi-Arid Fringe, New Mexico, USA (`2018-04` → `2018-08`, `+0` new buildings)
    - `sn7-test-08`: Bukhara Oasis Fringe, Uzbekistan (`2018-05` → `2018-09`, `+0` new buildings)

---

## 3. Preprocessing, Change Detection & Classification Methods

1. **Spatial & Raster Validation (`src/lib/imageProcessing.ts`):** Verifies matching raster dimensions, RGBA buffer integrity, EPSG:3857 bounding box validity, and `>= 50%` cloud-free overlap (`Band 4` UDM alpha mask).
2. **Co-Registration (`alignLaterImage`):** Computes `3 × 3` Sobel gradient magnitude maps on earlier and later images and maximizes Normalized Cross-Correlation (NCC) across integer translations `[-4..+4]` px.
3. **Denoising & Contrast Enhancement:** Supports `3 × 3` Median filtering, edge-preserving Bilateral filtering (`σ_s = 1.4, σ_r = 24`), `2%–98%` linear percentile contrast stretch, and `8 × 8` tile CLAHE while preserving raw original images for side-by-side comparison.
4. **Multi-Cue Change Detection (`src/lib/changeDetection.ts`):**
   - Matches channel mean and standard deviation of the later image to the earlier clear-sky baseline.
   - Calculates normalized Euclidean RGB Change Vector Magnitude (`CVA`).
   - Suppresses seasonal vegetation phenology shifts via Excess Green (`ExG = 2G - R - B`) and Visible Atmospherically Resistant Index (`VARI = (G - R) / (G + R - B + 10)`).
   - Suppresses low-luminance cast shadow darkening.
   - Applies `3 × 3` morphological opening and closing followed by 8-connected component labeling to isolate discrete candidate regions.
5. **Calibrated Classification (`src/lib/classifier.ts`):**
   - L2-regularized (`λ = 1.0`) Logistic Regression trained strictly on **26 geographically disjoint SpaceNet 7 training AOIs** and evaluated on **8 held-out test AOIs** (minimum Train–Test distance `425.4 km`, max 1 sub-tile per AOI — zero spatial leakage).
   - Uses 5 image-derived features: `structuralChangePct`, `highAlbedoGainPct`, `meanClusterEdgeGain`, `compactClusterDensityKm2`, and `seasonalVegShiftFraction`.
   - Held-out test performance: `87.5%` accuracy (`7/8`), `0.889` F1, `0.750` ROC-AUC, `0.213` Brier score (`TP=4, FP=1, TN=3, FN=0`). Also includes a transparent Rule-Based Screening mode when uncalibrated screening is selected.
