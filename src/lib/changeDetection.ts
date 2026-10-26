import type {
  CandidateRegion,
  ChangeDetectionResult,
  ExtractedFeatures,
  ModelArtifact,
  PreprocessingConfig,
  RawImageBuffer,
  SpatialMetadata,
} from '../types/satellite.ts';
import { classifyExtractedFeatures } from './classifier.ts';
import {
  alignLaterImage,
  applyContrastEnhancement,
  applyDenoising,
  computeSobelMagnitude,
  normalizeRadiometryToBaseline,
  validateImagePair,
} from './imageProcessing.ts';

export const DEFAULT_PREPROCESSING_CONFIG: PreprocessingConfig = {
  enableAlignment: true,
  denoiseMethod: 'bilateral',
  contrastMethod: 'linear_percentile',
  radiometricMatch: true,
  suppressVegetation: true,
  suppressShadows: true,
  changeThreshold: 28,
  minClusterAreaPx: 10,
};

function binaryDilate3x3(src: Uint8Array, w: number, h: number): Uint8Array {
  const out = new Uint8Array(w * h);
  for (let y = 1; y < h - 1; y++) {
    const r0 = (y - 1) * w;
    const r1 = y * w;
    const r2 = (y + 1) * w;
    for (let x = 1; x < w - 1; x++) {
      if (
        src[r0 + x - 1] || src[r0 + x] || src[r0 + x + 1] ||
        src[r1 + x - 1] || src[r1 + x] || src[r1 + x + 1] ||
        src[r2 + x - 1] || src[r2 + x] || src[r2 + x + 1]
      ) {
        out[r1 + x] = 1;
      }
    }
  }
  return out;
}

function binaryErode3x3(src: Uint8Array, w: number, h: number): Uint8Array {
  const out = new Uint8Array(w * h);
  for (let y = 1; y < h - 1; y++) {
    const r0 = (y - 1) * w;
    const r1 = y * w;
    const r2 = (y + 1) * w;
    for (let x = 1; x < w - 1; x++) {
      if (
        src[r0 + x - 1] && src[r0 + x] && src[r0 + x + 1] &&
        src[r1 + x - 1] && src[r1 + x] && src[r1 + x + 1] &&
        src[r2 + x - 1] && src[r2 + x] && src[r2 + x + 1]
      ) {
        out[r1 + x] = 1;
      }
    }
  }
  return out;
}

/**
 * Core spectral differencing, vegetation/shadow suppression, morphological filtering,
 * connected-component region extraction, and feature extraction pass.
 */
export function computeFeaturesAndMasks(
  earlierOrig: RawImageBuffer,
  laterAligned: RawImageBuffer,
  config: PreprocessingConfig,
  gsdMetersPerPx = 4.777
): {
  candidateMask: Uint8Array;
  suppressedEnvMask: Uint8Array;
  spectralDeltaMap: Float32Array;
  candidatePixelCount: number;
  candidateChangedPct: number;
  candidateAreaM2: number;
  rawDiffPixelCount: number;
  rawDiffPct: number;
  suppressedEnvPixelCount: number;
  suppressedEnvPct: number;
  validPixelCount: number;
  regions: CandidateRegion[];
  features: ExtractedFeatures;
} {
  const { width: w, height: h } = earlierOrig;
  const n = w * h;

  // Denoise before radiometric differencing so isolated sensor noise is reduced
  const eDenoised = applyDenoising(earlierOrig, config.denoiseMethod);
  const lDenoised = applyDenoising(laterAligned, config.denoiseMethod);

  // Relative radiometric normalization (matches channel mean/std of later to earlier)
  const lNormRgb = normalizeRadiometryToBaseline(
    eDenoised,
    lDenoised,
    config.radiometricMatch
  );

  const lNormBuf: RawImageBuffer = {
    width: w,
    height: h,
    data: new Uint8ClampedArray(n * 4),
  };
  for (let i = 0; i < n; i++) {
    const idx = i * 4;
    lNormBuf.data[idx] = Math.round(lNormRgb[i * 3]);
    lNormBuf.data[idx + 1] = Math.round(lNormRgb[i * 3 + 1]);
    lNormBuf.data[idx + 2] = Math.round(lNormRgb[i * 3 + 2]);
    lNormBuf.data[idx + 3] = lDenoised.data[idx + 3];
  }

  const edge1 = computeSobelMagnitude(eDenoised);
  const edge2 = computeSobelMagnitude(lNormBuf);

  const rawCandMask = new Uint8Array(n);
  const suppressedEnvMask = new Uint8Array(n);
  const spectralDeltaMap = new Float32Array(n);
  const albedoGainArr = new Float32Array(n);

  let validPixelCount = 0;
  let rawDiffPixelCount = 0;
  let vegDrivenRawDiffCount = 0;
  let highAlbedoGainPixels = 0;

  const thresh = config.changeThreshold;

  for (let i = 0; i < n; i++) {
    const idx = i * 4;
    if (eDenoised.data[idx + 3] < 200 || lDenoised.data[idx + 3] < 200) continue;
    const r1 = eDenoised.data[idx];
    const g1 = eDenoised.data[idx + 1];
    const b1 = eDenoised.data[idx + 2];

    const r2Raw = lDenoised.data[idx];
    const g2Raw = lDenoised.data[idx + 1];
    const b2Raw = lDenoised.data[idx + 2];

    if ((r1 === 0 && g1 === 0 && b1 === 0) || (r2Raw === 0 && g2Raw === 0 && b2Raw === 0)) {
      continue;
    }

    validPixelCount++;

    const r2 = lNormRgb[i * 3];
    const g2 = lNormRgb[i * 3 + 1];
    const b2 = lNormRgb[i * 3 + 2];

    const dR = r2 - r1;
    const dG = g2 - g1;
    const dB = b2 - b1;
    const cva = Math.sqrt((dR * dR + dG * dG + dB * dB) / 3);
    spectralDeltaMap[i] = cva;

    const lum1 = 0.299 * r1 + 0.587 * g1 + 0.114 * b1;
    const lum2 = 0.299 * r2 + 0.587 * g2 + 0.114 * b2;
    const dLum = lum2 - lum1;
    albedoGainArr[i] = dLum;

    // Excess Green (ExG = 2G - R - B) and Visible Atmospherically Resistant Index (VARI)
    const exg1 = 2 * g1 - r1 - b1;
    const exg2 = 2 * g2Raw - r2Raw - b2Raw;
    const vari1 = (g1 - r1) / (g1 + r1 - b1 + 10);
    const vari2 = (g2Raw - r2Raw) / (g2Raw + r2Raw - b2Raw + 10);

    // Track non-vegetated bright surface emergence (roofs, concrete, cleared grading pads)
    if (dLum > 18 && lum2 > 112 && exg2 < 12 && vari2 < 0.05) {
      highAlbedoGainPixels++;
    }

    if (cva < thresh) continue;
    rawDiffPixelCount++;

    // Determine whether pixel difference is dominated by vegetation greenness / phenology
    const isVegGain = exg2 > 16 || vari2 > 0.065;
    const isVegToVegOrSoilPhenology =
      Math.abs(exg2 - exg1) > 18 && (exg2 > 12 || (exg1 > 18 && lum2 < 118));
    if (isVegGain || isVegToVegOrSoilPhenology) {
      vegDrivenRawDiffCount++;
    }

    if (config.suppressVegetation && (isVegGain || (exg1 > 20 && lum2 < 108 && edge2[i] < 12))) {
      suppressedEnvMask[i] = 1;
      continue;
    }

    // Shadow suppression: darkening into low luminance without new bright structure edges
    const isShadowShift =
      (lum2 < 56 && dLum < -14) || (lum1 < 52 && lum2 < 78 && edge2[i] < 14);
    if (config.suppressShadows && isShadowShift) {
      suppressedEnvMask[i] = 1;
      continue;
    }

    rawCandMask[i] = 1;
  }

  // Morphological 3x3 opening (erode -> dilate) then closing to suppress 1-2px co-registration noise
  const opened = binaryDilate3x3(binaryErode3x3(rawCandMask, w, h), w, h);
  const morphed = binaryErode3x3(binaryDilate3x3(opened, w, h), w, h);

  // Connected-component labeling (8-connectivity)
  const visited = new Uint8Array(n);
  const candidateMask = new Uint8Array(n);
  const regions: CandidateRegion[] = [];
  const queue = new Int32Array(n);
  const componentPixels: number[] = [];

  const pxAreaM2 = gsdMetersPerPx * gsdMetersPerPx;
  let regionId = 1;
  let candidatePixelCount = 0;

  for (let i = 0; i < n; i++) {
    if (!morphed[i] || visited[i]) continue;
    visited[i] = 1;
    let head = 0;
    let tail = 0;
    queue[tail++] = i;
    componentPixels.length = 0;

    let minX = w;
    let minY = h;
    let maxX = 0;
    let maxY = 0;
    let sumX = 0;
    let sumY = 0;
    let sumSpec = 0;
    let sumEdgeGain = 0;
    let sumAlbedo = 0;

    while (head < tail) {
      const curr = queue[head++];
      componentPixels.push(curr);
      const cx = curr % w;
      const cy = (curr - cx) / w;

      if (cx < minX) minX = cx;
      if (cx > maxX) maxX = cx;
      if (cy < minY) minY = cy;
      if (cy > maxY) maxY = cy;
      sumX += cx;
      sumY += cy;
      sumSpec += spectralDeltaMap[curr];
      sumEdgeGain += Math.max(0, edge2[curr] - edge1[curr]);
      sumAlbedo += albedoGainArr[curr];

      for (let ny = Math.max(0, cy - 1); ny <= Math.min(h - 1, cy + 1); ny++) {
        const rOff = ny * w;
        for (let nx = Math.max(0, cx - 1); nx <= Math.min(w - 1, cx + 1); nx++) {
          const nb = rOff + nx;
          if (morphed[nb] && !visited[nb]) {
            visited[nb] = 1;
            queue[tail++] = nb;
          }
        }
      }
    }

    const areaPx = componentPixels.length;
    if (areaPx >= config.minClusterAreaPx) {
      for (const p of componentPixels) candidateMask[p] = 1;
      candidatePixelCount += areaPx;
      regions.push({
        id: regionId++,
        areaPx,
        areaM2: Number((areaPx * pxAreaM2).toFixed(1)),
        bbox: [minX, minY, maxX, maxY],
        centroid: [
          Number((sumX / areaPx).toFixed(1)),
          Number((sumY / areaPx).toFixed(1)),
        ],
        meanSpectralDelta: Number((sumSpec / areaPx).toFixed(2)),
        meanEdgeGain: Number((sumEdgeGain / areaPx).toFixed(2)),
        albedoGain: Number((sumAlbedo / areaPx).toFixed(2)),
      });
    }
  }

  regions.sort((a, b) => b.areaPx - a.areaPx);

  let suppressedEnvPixelCount = 0;
  for (let i = 0; i < n; i++) {
    if (suppressedEnvMask[i]) suppressedEnvPixelCount++;
  }

  const safeValid = Math.max(1, validPixelCount);
  const tileAreaKm2 = Math.max(0.01, (safeValid * pxAreaM2) / 1_000_000);

  let weightedEdgeGainSum = 0;
  let compactClusters = 0;
  for (const reg of regions) {
    weightedEdgeGainSum += reg.meanEdgeGain * reg.areaPx;
    const bw = reg.bbox[2] - reg.bbox[0] + 1;
    const bh = reg.bbox[3] - reg.bbox[1] + 1;
    const fillRatio = reg.areaPx / Math.max(1, bw * bh);
    if (reg.areaPx >= 10 && reg.areaPx <= 1400 && fillRatio >= 0.25) {
      compactClusters++;
    }
  }

  const meanClusterEdgeGain =
    candidatePixelCount > 0 ? weightedEdgeGainSum / candidatePixelCount : 0;

  const features: ExtractedFeatures = {
    structuralChangePct: Number(((candidatePixelCount / safeValid) * 100).toFixed(4)),
    highAlbedoGainPct: Number(((highAlbedoGainPixels / safeValid) * 100).toFixed(4)),
    meanClusterEdgeGain: Number(meanClusterEdgeGain.toFixed(4)),
    compactClusterDensityKm2: Number((compactClusters / tileAreaKm2).toFixed(4)),
    seasonalVegShiftFraction: Number(
      (rawDiffPixelCount > 0 ? vegDrivenRawDiffCount / rawDiffPixelCount : 0).toFixed(4)
    ),
  };

  return {
    candidateMask,
    suppressedEnvMask,
    spectralDeltaMap,
    candidatePixelCount,
    candidateChangedPct: Number(((candidatePixelCount / safeValid) * 100).toFixed(2)),
    candidateAreaM2: Number((candidatePixelCount * pxAreaM2).toFixed(0)),
    rawDiffPixelCount,
    rawDiffPct: Number(((rawDiffPixelCount / safeValid) * 100).toFixed(2)),
    suppressedEnvPixelCount,
    suppressedEnvPct: Number(((suppressedEnvPixelCount / safeValid) * 100).toFixed(2)),
    validPixelCount,
    regions,
    features,
  };
}

/**
 * Runs the complete end-to-end satellite change detection and construction classification pipeline.
 */
export function runChangeDetectionPipeline(params: {
  earlier: RawImageBuffer;
  later: RawImageBuffer;
  spatial?: SpatialMetadata;
  config?: PreprocessingConfig;
  model?: ModelArtifact | null;
}): ChangeDetectionResult {
  const t0 = performance.now();
  const config = params.config ?? DEFAULT_PREPROCESSING_CONFIG;
  const gsd = params.spatial?.gsdMetersPerPx ?? 4.777;

  const validation = validateImagePair(params.earlier, params.later, params.spatial);
  if (!validation.valid) {
    const emptyBuf: RawImageBuffer = {
      width: Math.max(1, params.earlier?.width || 1),
      height: Math.max(1, params.earlier?.height || 1),
      data: new Uint8ClampedArray(
        Math.max(1, params.earlier?.width || 1) *
          Math.max(1, params.earlier?.height || 1) *
          4
      ),
    };
    const zeroFeatures: ExtractedFeatures = {
      structuralChangePct: 0,
      highAlbedoGainPct: 0,
      meanClusterEdgeGain: 0,
      compactClusterDensityKm2: 0,
      seasonalVegShiftFraction: 0,
    };
    return {
      validation,
      alignment: { applied: false, dx: 0, dy: 0, nccScoreBefore: 0, nccScoreAfter: 0 },
      earlierCleaned: emptyBuf,
      laterCleaned: emptyBuf,
      changeMapOverlay: emptyBuf,
      differenceHeatmap: emptyBuf,
      candidateMask: new Uint8Array(emptyBuf.width * emptyBuf.height),
      suppressedEnvMask: new Uint8Array(emptyBuf.width * emptyBuf.height),
      candidatePixelCount: 0,
      candidateChangedPct: 0,
      candidateAreaM2: 0,
      rawDiffPixelCount: 0,
      rawDiffPct: 0,
      suppressedEnvPixelCount: 0,
      suppressedEnvPct: 0,
      regions: [],
      extractedFeatures: zeroFeatures,
      classification: classifyExtractedFeatures(zeroFeatures, null),
      processingDurationMs: Number((performance.now() - t0).toFixed(1)),
    };
  }

  // 1. Co-registration / alignment
  const { alignedLater, alignment } = alignLaterImage(
    params.earlier,
    params.later,
    config.enableAlignment
  );

  // 2. Produce cleaned visual buffers (denoising + contrast enhancement) while preserving originals
  const earlierCleaned = applyContrastEnhancement(
    applyDenoising(params.earlier, config.denoiseMethod),
    config.contrastMethod
  );
  const laterCleaned = applyContrastEnhancement(
    applyDenoising(alignedLater, config.denoiseMethod),
    config.contrastMethod
  );

  // 3. Compute active change map under user's chosen settings
  const activePass = computeFeaturesAndMasks(
    params.earlier,
    alignedLater,
    config,
    gsd
  );

  // 4. For calibrated model inference, evaluate on standardized calibration parameters if user customized sliders,
  //    ensuring calibrated probabilities match the model's exact training feature distribution.
  const isDefaultFeatureConfig =
    config.denoiseMethod === DEFAULT_PREPROCESSING_CONFIG.denoiseMethod &&
    config.radiometricMatch === DEFAULT_PREPROCESSING_CONFIG.radiometricMatch &&
    config.suppressVegetation === DEFAULT_PREPROCESSING_CONFIG.suppressVegetation &&
    config.suppressShadows === DEFAULT_PREPROCESSING_CONFIG.suppressShadows &&
    config.changeThreshold === DEFAULT_PREPROCESSING_CONFIG.changeThreshold &&
    config.minClusterAreaPx === DEFAULT_PREPROCESSING_CONFIG.minClusterAreaPx;

  const modelFeatures = isDefaultFeatureConfig
    ? activePass.features
    : computeFeaturesAndMasks(
        params.earlier,
        alignedLater,
        DEFAULT_PREPROCESSING_CONFIG,
        gsd
      ).features;

  const classification = classifyExtractedFeatures(modelFeatures, params.model);

  // 5. Render Change Map Overlay (later image dimmed desaturated backdrop + amber/coral candidate regions + cyan perimeter + subtle slate-teal suppressed seasonal areas)
  const { width: w, height: h } = params.earlier;
  const n = w * h;
  const overlayData = new Uint8ClampedArray(n * 4);
  const heatmapData = new Uint8ClampedArray(n * 4);

  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      const idx = i * 4;
      const r = laterCleaned.data[idx];
      const g = laterCleaned.data[idx + 1];
      const b = laterCleaned.data[idx + 2];
      const a = laterCleaned.data[idx + 3];

      if (a < 200) {
        overlayData[idx] = 15;
        overlayData[idx + 1] = 23;
        overlayData[idx + 2] = 42;
        overlayData[idx + 3] = 255;
        heatmapData[idx] = 15;
        heatmapData[idx + 1] = 23;
        heatmapData[idx + 2] = 42;
        heatmapData[idx + 3] = 255;
        continue;
      }

      // Muted grayscale/tinted satellite context for crisp visual contrast
      const lum = Math.round(0.299 * r + 0.587 * g + 0.114 * b);
      const baseR = Math.round(lum * 0.55 + r * 0.18);
      const baseG = Math.round(lum * 0.58 + g * 0.18);
      const baseB = Math.round(lum * 0.64 + b * 0.18);

      if (activePass.candidateMask[i]) {
        const isBoundary =
          x === 0 ||
          x === w - 1 ||
          y === 0 ||
          y === h - 1 ||
          !activePass.candidateMask[i - 1] ||
          !activePass.candidateMask[i + 1] ||
          !activePass.candidateMask[i - w] ||
          !activePass.candidateMask[i + w];

        if (isBoundary) {
          // Crisp high-contrast amber-gold boundary outline
          overlayData[idx] = 251;
          overlayData[idx + 1] = 191;
          overlayData[idx + 2] = 36;
          overlayData[idx + 3] = 255;
        } else {
          // Translucent warm coral-amber fill preserving underlying roof texture
          overlayData[idx] = Math.min(255, Math.round(baseR * 0.35 + 245 * 0.65));
          overlayData[idx + 1] = Math.min(255, Math.round(baseG * 0.35 + 105 * 0.65));
          overlayData[idx + 2] = Math.min(255, Math.round(baseB * 0.35 + 32 * 0.65));
          overlayData[idx + 3] = 255;
        }
      } else if (activePass.suppressedEnvMask[i]) {
        // Subtle cool cyan-teal tint showing suppressed vegetation/shadow difference
        overlayData[idx] = Math.round(baseR * 0.65 + 20 * 0.35);
        overlayData[idx + 1] = Math.round(baseG * 0.65 + 148 * 0.35);
        overlayData[idx + 2] = Math.round(baseB * 0.65 + 168 * 0.35);
        overlayData[idx + 3] = 255;
      } else {
        overlayData[idx] = baseR;
        overlayData[idx + 1] = baseG;
        overlayData[idx + 2] = baseB;
        overlayData[idx + 3] = 255;
      }

      // Scientific Viridis/Inferno-inspired continuous spectral difference ramp [0..90]
      const t = Math.min(1, activePass.spectralDeltaMap[i] / 85);
      if (t < 0.33) {
        const u = t / 0.33;
        heatmapData[idx] = Math.round(15 + u * 45);
        heatmapData[idx + 1] = Math.round(23 + u * 55);
        heatmapData[idx + 2] = Math.round(52 + u * 95);
      } else if (t < 0.66) {
        const u = (t - 0.33) / 0.33;
        heatmapData[idx] = Math.round(60 + u * 165);
        heatmapData[idx + 1] = Math.round(78 + u * 42);
        heatmapData[idx + 2] = Math.round(147 - u * 95);
      } else {
        const u = (t - 0.66) / 0.34;
        heatmapData[idx] = Math.round(225 + u * 30);
        heatmapData[idx + 1] = Math.round(120 + u * 115);
        heatmapData[idx + 2] = Math.round(52 + u * 25);
      }
      heatmapData[idx + 3] = 255;
    }
  }

  return {
    validation,
    alignment,
    earlierCleaned,
    laterCleaned,
    changeMapOverlay: { width: w, height: h, data: overlayData },
    differenceHeatmap: { width: w, height: h, data: heatmapData },
    candidateMask: activePass.candidateMask,
    suppressedEnvMask: activePass.suppressedEnvMask,
    candidatePixelCount: activePass.candidatePixelCount,
    candidateChangedPct: activePass.candidateChangedPct,
    candidateAreaM2: activePass.candidateAreaM2,
    rawDiffPixelCount: activePass.rawDiffPixelCount,
    rawDiffPct: activePass.rawDiffPct,
    suppressedEnvPixelCount: activePass.suppressedEnvPixelCount,
    suppressedEnvPct: activePass.suppressedEnvPct,
    regions: activePass.regions,
    extractedFeatures: activePass.features,
    classification,
    processingDurationMs: Number((performance.now() - t0).toFixed(1)),
  };
}
