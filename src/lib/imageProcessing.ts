import type {
  AlignmentResult,
  RawImageBuffer,
  SpatialMetadata,
  ValidationResult,
} from '../types/satellite.ts';

export function cloneBuffer(src: RawImageBuffer): RawImageBuffer {
  return {
    width: src.width,
    height: src.height,
    data: new Uint8ClampedArray(src.data),
  };
}

/**
 * Validates spatial metadata, dimensions, buffer integrity, and clear-sky overlap
 * between an earlier and later satellite image pair.
 */
export function validateImagePair(
  earlier: RawImageBuffer | null | undefined,
  later: RawImageBuffer | null | undefined,
  spatial?: SpatialMetadata
): ValidationResult {
  const errors: string[] = [];
  const warnings: string[] = [];

  if (!earlier || !later || !earlier.data || !later.data) {
    return {
      valid: false,
      errors: ['Missing earlier or later image raster buffer.'],
      warnings: [],
      dimensionsMatch: false,
      spatialBoundsMatch: false,
      validClearPixelRatio: 0,
      validPixelCount: 0,
      totalPixelCount: 0,
    };
  }

  if (
    !Number.isInteger(earlier.width) ||
    !Number.isInteger(earlier.height) ||
    earlier.width < 32 ||
    earlier.height < 32
  ) {
    errors.push(
      `Invalid earlier image dimensions (${earlier.width}x${earlier.height}); minimum 32x32 px required.`
    );
  }

  if (
    !Number.isInteger(later.width) ||
    !Number.isInteger(later.height) ||
    later.width < 32 ||
    later.height < 32
  ) {
    errors.push(
      `Invalid later image dimensions (${later.width}x${later.height}); minimum 32x32 px required.`
    );
  }

  const dimensionsMatch =
    earlier.width === later.width && earlier.height === later.height;
  if (!dimensionsMatch) {
    errors.push(
      `Spatial dimension mismatch: earlier is ${earlier.width}x${earlier.height} px, later is ${later.width}x${later.height} px.`
    );
  }

  const expectedEarlierBytes = earlier.width * earlier.height * 4;
  const expectedLaterBytes = later.width * later.height * 4;
  if (earlier.data.length !== expectedEarlierBytes) {
    errors.push(
      `Corrupted earlier RGBA buffer: expected ${expectedEarlierBytes} bytes, got ${earlier.data.length}.`
    );
  }
  if (later.data.length !== expectedLaterBytes) {
    errors.push(
      `Corrupted later RGBA buffer: expected ${expectedLaterBytes} bytes, got ${later.data.length}.`
    );
  }

  let spatialBoundsMatch = true;
  if (spatial) {
    const [minX, minY, maxX, maxY] = spatial.bboxMercator;
    if (
      !Number.isFinite(minX) ||
      !Number.isFinite(minY) ||
      !Number.isFinite(maxX) ||
      !Number.isFinite(maxY) ||
      minX >= maxX ||
      minY >= maxY
    ) {
      spatialBoundsMatch = false;
      errors.push('Invalid EPSG:3857 bounding box coordinates in spatial metadata.');
    }
    if (!Number.isFinite(spatial.gsdMetersPerPx) || spatial.gsdMetersPerPx <= 0) {
      spatialBoundsMatch = false;
      errors.push('Invalid ground sample distance (GSD) in spatial metadata.');
    }
  }

  if (errors.length > 0) {
    return {
      valid: false,
      errors,
      warnings,
      dimensionsMatch,
      spatialBoundsMatch,
      validClearPixelRatio: 0,
      validPixelCount: 0,
      totalPixelCount: Math.max(0, earlier.width * earlier.height),
    };
  }

  const totalPixels = earlier.width * earlier.height;
  let validPixels = 0;
  for (let i = 0; i < totalPixels; i++) {
    const idx = i * 4;
    const a1 = earlier.data[idx + 3];
    const a2 = later.data[idx + 3];
    const nonZero1 =
      earlier.data[idx] > 0 || earlier.data[idx + 1] > 0 || earlier.data[idx + 2] > 0;
    const nonZero2 =
      later.data[idx] > 0 || later.data[idx + 1] > 0 || later.data[idx + 2] > 0;
    if (a1 >= 200 && a2 >= 200 && nonZero1 && nonZero2) {
      validPixels++;
    }
  }

  const validClearPixelRatio = totalPixels > 0 ? validPixels / totalPixels : 0;
  if (validClearPixelRatio < 0.5) {
    errors.push(
      `Insufficient cloud-free overlap (${(validClearPixelRatio * 100).toFixed(1)}% valid pixels; minimum 50% required).`
    );
  } else if (validClearPixelRatio < 0.92) {
    warnings.push(
      `Partial UDM cloud/nodata masking present (${((1 - validClearPixelRatio) * 100).toFixed(1)}% masked pixels excluded from analysis).`
    );
  }

  return {
    valid: errors.length === 0,
    errors,
    warnings,
    dimensionsMatch,
    spatialBoundsMatch,
    validClearPixelRatio,
    validPixelCount: validPixels,
    totalPixelCount: totalPixels,
  };
}

/**
 * Computes perceptual luminance [0..255] per pixel.
 */
export function computeLuminance(img: RawImageBuffer): Float32Array {
  const n = img.width * img.height;
  const lum = new Float32Array(n);
  const d = img.data;
  for (let i = 0; i < n; i++) {
    const idx = i * 4;
    lum[i] = 0.299 * d[idx] + 0.587 * d[idx + 1] + 0.114 * d[idx + 2];
  }
  return lum;
}

/**
 * Computes 3x3 Sobel edge gradient magnitude normalized by 4.0.
 */
export function computeSobelMagnitude(img: RawImageBuffer): Float32Array {
  const { width: w, height: h } = img;
  const lum = computeLuminance(img);
  const mag = new Float32Array(w * h);

  for (let y = 1; y < h - 1; y++) {
    const row = y * w;
    for (let x = 1; x < w - 1; x++) {
      const idx = row + x;
      if (img.data[idx * 4 + 3] < 200) continue;

      const tl = lum[(y - 1) * w + (x - 1)];
      const tc = lum[(y - 1) * w + x];
      const tr = lum[(y - 1) * w + (x + 1)];
      const ml = lum[y * w + (x - 1)];
      const mr = lum[y * w + (x + 1)];
      const bl = lum[(y + 1) * w + (x - 1)];
      const bc = lum[(y + 1) * w + x];
      const br = lum[(y + 1) * w + (x + 1)];

      const gx = -tl + tr - 2 * ml + 2 * mr - bl + br;
      const gy = -tl - 2 * tc - tr + bl + 2 * bc + br;
      mag[idx] = Math.hypot(gx, gy) * 0.25;
    }
  }
  return mag;
}

/**
 * Estimates integer pixel translation (dx, dy) between earlier and later images
 * via Normalized Cross-Correlation on Sobel gradient magnitude maps, then shifts
 * the later image into alignment with the earlier image.
 */
export function alignLaterImage(
  earlier: RawImageBuffer,
  later: RawImageBuffer,
  enableAlignment: boolean,
  maxShift = 4
): { alignedLater: RawImageBuffer; alignment: AlignmentResult } {
  const edge1 = computeSobelMagnitude(earlier);
  const edge2 = computeSobelMagnitude(later);
  const { width: w, height: h } = earlier;

  const evalNcc = (dx: number, dy: number): number => {
    const margin = maxShift + 2;
    let sum1 = 0;
    let sum2 = 0;
    let count = 0;
    // Stride 2 for fast, accurate correlation across interior window
    for (let y = margin; y < h - margin; y += 2) {
      const y2 = y - dy;
      for (let x = margin; x < w - margin; x += 2) {
        const x2 = x - dx;
        const i1 = y * w + x;
        const i2 = y2 * w + x2;
        if (earlier.data[i1 * 4 + 3] < 200 || later.data[i2 * 4 + 3] < 200) continue;
        sum1 += edge1[i1];
        sum2 += edge2[i2];
        count++;
      }
    }
    if (count < 100) return 0;
    const mean1 = sum1 / count;
    const mean2 = sum2 / count;

    let num = 0;
    let den1 = 0;
    let den2 = 0;
    for (let y = margin; y < h - margin; y += 2) {
      const y2 = y - dy;
      for (let x = margin; x < w - margin; x += 2) {
        const x2 = x - dx;
        const i1 = y * w + x;
        const i2 = y2 * w + x2;
        if (earlier.data[i1 * 4 + 3] < 200 || later.data[i2 * 4 + 3] < 200) continue;
        const d1 = edge1[i1] - mean1;
        const d2 = edge2[i2] - mean2;
        num += d1 * d2;
        den1 += d1 * d1;
        den2 += d2 * d2;
      }
    }
    const denom = Math.sqrt(den1 * den2);
    return denom > 1e-6 ? num / denom : 0;
  };

  const nccZero = evalNcc(0, 0);
  if (!enableAlignment) {
    return {
      alignedLater: cloneBuffer(later),
      alignment: {
        applied: false,
        dx: 0,
        dy: 0,
        nccScoreBefore: Number(nccZero.toFixed(4)),
        nccScoreAfter: Number(nccZero.toFixed(4)),
      },
    };
  }

  let bestDx = 0;
  let bestDy = 0;
  let bestNcc = nccZero;

  for (let dy = -maxShift; dy <= maxShift; dy++) {
    for (let dx = -maxShift; dx <= maxShift; dx++) {
      if (dx === 0 && dy === 0) continue;
      const score = evalNcc(dx, dy);
      if (score > bestNcc + 0.008) {
        bestNcc = score;
        bestDx = dx;
        bestDy = dy;
      }
    }
  }

  if (bestDx === 0 && bestDy === 0) {
    return {
      alignedLater: cloneBuffer(later),
      alignment: {
        applied: true,
        dx: 0,
        dy: 0,
        nccScoreBefore: Number(nccZero.toFixed(4)),
        nccScoreAfter: Number(nccZero.toFixed(4)),
      },
    };
  }

  const shifted = new Uint8ClampedArray(w * h * 4);
  for (let y = 0; y < h; y++) {
    const sy = y - bestDy;
    if (sy < 0 || sy >= h) continue;
    for (let x = 0; x < w; x++) {
      const sx = x - bestDx;
      if (sx < 0 || sx >= w) continue;
      const dstIdx = (y * w + x) * 4;
      const srcIdx = (sy * w + sx) * 4;
      shifted[dstIdx] = later.data[srcIdx];
      shifted[dstIdx + 1] = later.data[srcIdx + 1];
      shifted[dstIdx + 2] = later.data[srcIdx + 2];
      shifted[dstIdx + 3] = later.data[srcIdx + 3];
    }
  }

  return {
    alignedLater: { width: w, height: h, data: shifted },
    alignment: {
      applied: true,
      dx: bestDx,
      dy: bestDy,
      nccScoreBefore: Number(nccZero.toFixed(4)),
      nccScoreAfter: Number(bestNcc.toFixed(4)),
    },
  };
}

/**
 * Applies genuine spatial denoising while preserving alpha mask.
 */
export function applyDenoising(
  img: RawImageBuffer,
  method: 'none' | 'median3x3' | 'bilateral'
): RawImageBuffer {
  if (method === 'none') return cloneBuffer(img);

  const { width: w, height: h, data: src } = img;
  const out = new Uint8ClampedArray(src);

  if (method === 'median3x3') {
    const bufR = new Int32Array(9);
    const bufG = new Int32Array(9);
    const bufB = new Int32Array(9);

    for (let y = 1; y < h - 1; y++) {
      for (let x = 1; x < w - 1; x++) {
        const centerIdx = (y * w + x) * 4;
        if (src[centerIdx + 3] < 200) continue;

        let k = 0;
        for (let ky = -1; ky <= 1; ky++) {
          const rOff = (y + ky) * w;
          for (let kx = -1; kx <= 1; kx++) {
            const p = (rOff + (x + kx)) * 4;
            bufR[k] = src[p];
            bufG[k] = src[p + 1];
            bufB[k] = src[p + 2];
            k++;
          }
        }
        bufR.sort();
        bufG.sort();
        bufB.sort();
        out[centerIdx] = bufR[4];
        out[centerIdx + 1] = bufG[4];
        out[centerIdx + 2] = bufB[4];
      }
    }
    return { width: w, height: h, data: out };
  }

  // Edge-preserving 3x3 bilateral filter (sigma_s = 1.4, sigma_r = 24)
  const spatialWeights = [
    0.3679, 0.6065, 0.3679,
    0.6065, 1.0,    0.6065,
    0.3679, 0.6065, 0.3679,
  ];
  const invTwoSigmaR2 = 1 / (2 * 24 * 24);

  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const cIdx = (y * w + x) * 4;
      if (src[cIdx + 3] < 200) continue;

      const cR = src[cIdx];
      const cG = src[cIdx + 1];
      const cB = src[cIdx + 2];

      let sumR = 0;
      let sumG = 0;
      let sumB = 0;
      let wSum = 0;
      let k = 0;

      for (let ky = -1; ky <= 1; ky++) {
        const rOff = (y + ky) * w;
        for (let kx = -1; kx <= 1; kx++) {
          const p = (rOff + (x + kx)) * 4;
          if (src[p + 3] >= 200) {
            const dR = src[p] - cR;
            const dG = src[p + 1] - cG;
            const dB = src[p + 2] - cB;
            const dist2 = (dR * dR + dG * dG + dB * dB) * 0.3333;
            const weight = spatialWeights[k] * Math.exp(-dist2 * invTwoSigmaR2);
            sumR += src[p] * weight;
            sumG += src[p + 1] * weight;
            sumB += src[p + 2] * weight;
            wSum += weight;
          }
          k++;
        }
      }

      if (wSum > 0) {
        out[cIdx] = Math.round(sumR / wSum);
        out[cIdx + 1] = Math.round(sumG / wSum);
        out[cIdx + 2] = Math.round(sumB / wSum);
      }
    }
  }

  return { width: w, height: h, data: out };
}

/**
 * Applies contrast enhancement for improved visual clarity of building roofs and roads.
 */
export function applyContrastEnhancement(
  img: RawImageBuffer,
  method: 'none' | 'linear_percentile' | 'clahe'
): RawImageBuffer {
  if (method === 'none') return cloneBuffer(img);

  const { width: w, height: h, data: src } = img;
  const out = new Uint8ClampedArray(src);
  const n = w * h;

  if (method === 'linear_percentile') {
    for (let c = 0; c < 3; c++) {
      const hist = new Uint32Array(256);
      let validCount = 0;
      for (let i = 0; i < n; i++) {
        const idx = i * 4;
        if (src[idx + 3] < 200) continue;
        hist[src[idx + c]]++;
        validCount++;
      }
      if (validCount < 50) continue;

      const lowTarget = validCount * 0.02;
      const highTarget = validCount * 0.98;
      let cum = 0;
      let lowVal = 0;
      let highVal = 255;

      for (let v = 0; v < 256; v++) {
        cum += hist[v];
        if (cum >= lowTarget) {
          lowVal = v;
          break;
        }
      }
      cum = 0;
      for (let v = 0; v < 256; v++) {
        cum += hist[v];
        if (cum >= highTarget) {
          highVal = v;
          break;
        }
      }

      const span = Math.max(1, highVal - lowVal);
      for (let i = 0; i < n; i++) {
        const idx = i * 4;
        if (src[idx + 3] < 200) continue;
        out[idx + c] = Math.max(0, Math.min(255, Math.round(((src[idx + c] - lowVal) * 255) / span)));
      }
    }
    return { width: w, height: h, data: out };
  }

  // Tile-based CLAHE on perceptual luminance (8x8 grid with bilinear interpolation)
  const gridX = 8;
  const gridY = 8;
  const tileW = Math.ceil(w / gridX);
  const tileH = Math.ceil(h / gridY);
  const lum = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    const idx = i * 4;
    lum[i] = Math.round(0.299 * src[idx] + 0.587 * src[idx + 1] + 0.114 * src[idx + 2]);
  }

  const luts = new Float32Array(gridX * gridY * 256);
  const clipFactor = 2.2;

  for (let ty = 0; ty < gridY; ty++) {
    for (let tx = 0; tx < gridX; tx++) {
      const x0 = tx * tileW;
      const y0 = ty * tileH;
      const x1 = Math.min(w, x0 + tileW);
      const y1 = Math.min(h, y0 + tileH);

      const hist = new Float32Array(256);
      let count = 0;
      for (let y = y0; y < y1; y++) {
        for (let x = x0; x < x1; x++) {
          const i = y * w + x;
          if (src[i * 4 + 3] < 200) continue;
          hist[lum[i]]++;
          count++;
        }
      }

      const lutBase = (ty * gridX + tx) * 256;
      if (count < 16) {
        for (let v = 0; v < 256; v++) luts[lutBase + v] = v;
        continue;
      }

      const clipLimit = Math.max(1, (clipFactor * count) / 256);
      let excess = 0;
      for (let v = 0; v < 256; v++) {
        if (hist[v] > clipLimit) {
          excess += hist[v] - clipLimit;
          hist[v] = clipLimit;
        }
      }
      const redist = excess / 256;
      let cdf = 0;
      for (let v = 0; v < 256; v++) {
        hist[v] += redist;
        cdf += hist[v];
        luts[lutBase + v] = Math.min(255, (cdf / count) * 255);
      }
    }
  }

  for (let y = 0; y < h; y++) {
    const gy = (y + 0.5) / tileH - 0.5;
    const ty0 = Math.max(0, Math.min(gridY - 1, Math.floor(gy)));
    const ty1 = Math.max(0, Math.min(gridY - 1, ty0 + 1));
    const wy = Math.max(0, Math.min(1, gy - ty0));

    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      const idx = i * 4;
      if (src[idx + 3] < 200) continue;

      const gx = (x + 0.5) / tileW - 0.5;
      const tx0 = Math.max(0, Math.min(gridX - 1, Math.floor(gx)));
      const tx1 = Math.max(0, Math.min(gridX - 1, tx0 + 1));
      const wx = Math.max(0, Math.min(1, gx - tx0));

      const v = lum[i];
      const v00 = luts[(ty0 * gridX + tx0) * 256 + v];
      const v10 = luts[(ty0 * gridX + tx1) * 256 + v];
      const v01 = luts[(ty1 * gridX + tx0) * 256 + v];
      const v11 = luts[(ty1 * gridX + tx1) * 256 + v];

      const top = v00 * (1 - wx) + v10 * wx;
      const bot = v01 * (1 - wx) + v11 * wx;
      const newLum = top * (1 - wy) + bot * wy;

      const scale = v > 2 ? newLum / v : 1;
      out[idx] = Math.min(255, Math.round(src[idx] * scale));
      out[idx + 1] = Math.min(255, Math.round(src[idx + 1] * scale));
      out[idx + 2] = Math.min(255, Math.round(src[idx + 2] * scale));
    }
  }

  return { width: w, height: h, data: out };
}

/**
 * Matches channel mean and standard deviation of `later` to `earlier` across mutually valid pixels
 * to suppress global illumination, haze, and sensor exposure drift prior to spectral differencing.
 */
export function normalizeRadiometryToBaseline(
  earlier: RawImageBuffer,
  later: RawImageBuffer,
  enableMatch: boolean
): Float32Array {
  const n = earlier.width * earlier.height;
  const normRgb = new Float32Array(n * 3);

  if (!enableMatch) {
    for (let i = 0; i < n; i++) {
      const idx = i * 4;
      normRgb[i * 3] = later.data[idx];
      normRgb[i * 3 + 1] = later.data[idx + 1];
      normRgb[i * 3 + 2] = later.data[idx + 2];
    }
    return normRgb;
  }

  for (let c = 0; c < 3; c++) {
    let sum1 = 0;
    let sum2 = 0;
    let count = 0;
    for (let i = 0; i < n; i++) {
      const idx = i * 4;
      if (earlier.data[idx + 3] < 200 || later.data[idx + 3] < 200) continue;
      sum1 += earlier.data[idx + c];
      sum2 += later.data[idx + c];
      count++;
    }
    if (count < 50) continue;
    const mean1 = sum1 / count;
    const mean2 = sum2 / count;

    let var1 = 0;
    let var2 = 0;
    for (let i = 0; i < n; i++) {
      const idx = i * 4;
      if (earlier.data[idx + 3] < 200 || later.data[idx + 3] < 200) continue;
      const d1 = earlier.data[idx + c] - mean1;
      const d2 = later.data[idx + c] - mean2;
      var1 += d1 * d1;
      var2 += d2 * d2;
    }
    const std1 = Math.sqrt(var1 / count);
    const std2 = Math.sqrt(var2 / count);
    // Clamp gain to [0.65, 1.50] to avoid amplifying extreme outliers
    const gain = std2 > 1e-3 ? Math.max(0.65, Math.min(1.5, std1 / std2)) : 1;

    for (let i = 0; i < n; i++) {
      const v = mean1 + (later.data[i * 4 + c] - mean2) * gain;
      normRgb[i * 3 + c] = Math.max(0, Math.min(255, v));
    }
  }

  return normRgb;
}
