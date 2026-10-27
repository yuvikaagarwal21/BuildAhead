import type {
  BaselineFeatures,
  CalibrationMethodBenchmark,
  CandidateModelComparisonEntry,
  ClassificationOutput,
  DatasetSplitRecord,
  Depth2TreeNode,
  EvaluationMetrics,
  ExtractedFeatures,
  FeatureContribution,
  FeatureKey,
  ModelArtifact,
  PixelChangeIoUMetrics,
  ProbabilityCalibrationConfig,
  RandomForestTreeNode,
  ReliabilityBin,
} from '../types/satellite.ts';

export const BASELINE_FEATURE_KEYS: (keyof BaselineFeatures)[] = [
  'structuralChangePct',
  'highAlbedoGainPct',
  'meanClusterEdgeGain',
  'compactClusterDensityKm2',
  'seasonalVegShiftFraction',
];

export const FEATURE_KEYS: FeatureKey[] = [
  'structuralChangePct',
  'highAlbedoGainPct',
  'meanClusterEdgeGain',
  'compactClusterDensityKm2',
  'seasonalVegShiftFraction',
  'newEdgeEmergencePct',
  'roofShadowDipolePct',
  'patchActivityConcentration',
];

export const FEATURE_DEFINITIONS: ModelArtifact['featureDefinitions'] = {
  structuralChangePct: {
    label: 'Filtered Structural Change Area',
    unit: '% of valid tile',
    description:
      'Percentage of cloud-free pixels retained in candidate change clusters after sub-pixel co-registration, PIF radiometric normalization, PIF-normalized vegetation/snow/bare-soil suppression, smooth-shadow filtering, and 5-of-9 geodesic opening.',
  },
  highAlbedoGainPct: {
    label: 'Non-Vegetated Bright Surface Gain',
    unit: '% of valid tile',
    description:
      'Percentage of valid pixels transitioning toward brighter, edge-bearing, low-greenness reflectance characteristic of new roofs, concrete pads, or graded construction earthworks.',
  },
  meanClusterEdgeGain: {
    label: 'Stabilized Cluster Sobel Edge Gain',
    unit: 'gradient units',
    description:
      'Empirical-Bayes area-shrunk increase in orientation-decorrelated 3x3 Sobel gradient magnitude across candidate clusters, preventing isolated gravel pads from producing extreme edge ratios.',
  },
  compactClusterDensityKm2: {
    label: 'Rectangular Compact Cluster Density',
    unit: 'clusters / km²',
    description:
      'Spatial density of discrete, compact, bounded-aspect-ratio (<= 3.2:1) candidate building clusters (10–950 px area with sharp wall edge gain) per square kilometer.',
  },
  seasonalVegShiftFraction: {
    label: 'Seasonal & Environmental Shift Ratio',
    unit: '0–1 ratio',
    description:
      'Fraction of raw spectral pixel differences attributable to PIF-normalized Excess-Green phenology changes, plowed agricultural soil, autumn leaf-off, or scene-wide seasonal snow washes.',
  },
  newEdgeEmergencePct: {
    label: 'New Structural Edge Emergence',
    unit: '% of valid tile',
    description:
      'Percentage of valid non-vegetated pixels where strong boundaries emerged on previously smooth ground or sunlit desert sand, or rotated relative to prior edges.',
  },
  roofShadowDipolePct: {
    label: '3D Roof-Shadow Contrast Dipole',
    unit: '% of valid tile',
    description:
      'Percentage of valid non-vegetated pixels gaining sharp 5x5 local luminance range from adjacent sunlit roof and cast shadow pairs (captures both dark-soil and bright-sand buildings).',
  },
  patchActivityConcentration: {
    label: 'Urban Block Patch Concentration',
    unit: '% of sub-patches',
    description:
      'Percentage of 64x64 pixel urban-block sub-patches exhibiting co-occurring candidate clusters, newly emerged structural edges, and 3D roof-shadow dipoles.',
  },
};

export function sigmoid(z: number): number {
  if (z >= 0) {
    const ez = Math.exp(-z);
    return 1 / (1 + ez);
  }
  const ez = Math.exp(z);
  return ez / (1 + ez);
}

export function logitFromProb(p: number): number {
  const pc = Math.min(1 - 1e-6, Math.max(1e-6, p));
  return Math.log(pc / (1 - pc));
}

/**
 * Exact binary cross-entropy (Log Loss) with numerical clipping [1e-6, 1 - 1e-6].
 */
export function computeLogLoss(yTrue: boolean[], probs: number[]): number {
  if (yTrue.length === 0) return 0;
  let lossSum = 0;
  for (let i = 0; i < yTrue.length; i++) {
    const p = Math.min(1 - 1e-6, Math.max(1e-6, probs[i]));
    lossSum += yTrue[i] ? -Math.log(p) : -Math.log(1 - p);
  }
  return Number((lossSum / yTrue.length).toFixed(4));
}

/**
 * Computes equal-width reliability diagram bins and Expected Calibration Error (ECE):
 * ECE = sum_{b=1..B} (n_b / N) * |meanPredictedProb_b - empiricalPositiveRate_b|
 */
export function computeReliabilityDiagramAndEce(
  yTrue: boolean[],
  probs: number[],
  numBins = 5
): { ece: number; reliabilityBins: ReliabilityBin[] } {
  const n = Math.max(1, yTrue.length);
  const reliabilityBins: ReliabilityBin[] = [];
  let eceSum = 0;

  for (let b = 0; b < numBins; b++) {
    const binStart = Number((b / numBins).toFixed(2));
    const binEnd = Number(((b + 1) / numBins).toFixed(2));
    let count = 0;
    let sumP = 0;
    let sumY = 0;

    for (let i = 0; i < yTrue.length; i++) {
      const p = probs[i];
      const inBin =
        b === numBins - 1
          ? p >= binStart && p <= binEnd + 1e-9
          : p >= binStart && p < binEnd;
      if (inBin) {
        count++;
        sumP += p;
        sumY += yTrue[i] ? 1 : 0;
      }
    }

    const meanPredictedProb = count > 0 ? sumP / count : 0;
    const empiricalPositiveRate = count > 0 ? sumY / count : 0;
    if (count > 0 && yTrue.length > 0) {
      eceSum += (count / n) * Math.abs(meanPredictedProb - empiricalPositiveRate);
    }

    reliabilityBins.push({
      binIndex: b,
      binStart,
      binEnd,
      count,
      meanPredictedProb: Number(meanPredictedProb.toFixed(4)),
      empiricalPositiveRate: Number(empiricalPositiveRate.toFixed(4)),
    });
  }

  return {
    ece: Number(eceSum.toFixed(4)),
    reliabilityBins,
  };
}

/**
 * Regularized Platt Sigmoid Calibration: P_cal = sigmoid(slope * rawLogit + intercept).
 * Solves exact 2x2 Newton-Raphson IRLS on pure binary labels with an identity-centered
 * Gaussian prior (`priorReg * (slope - 1)^2 + 2.5 * priorReg * intercept^2`) on out-of-sample
 * non-test logits (n=59), reversing L2 logit compression without overfitting small folds.
 */
export function fitPlattSigmoidCalibration(
  logits: number[],
  yTrue: number[],
  priorReg = 10.0
): { slope: number; intercept: number } {
  if (logits.length === 0) return { slope: 1, intercept: 0 };
  const regA = Math.max(0.5, priorReg);
  const regB = Math.max(1.0, 2.5 * priorReg);

  let slope = 1.0;
  let intercept = 0.0;

  for (let iter = 0; iter < 40; iter++) {
    let gA = regA * (slope - 1.0);
    let gB = regB * intercept;
    let hAA = regA;
    let hAB = 0.0;
    let hBB = regB;

    for (let i = 0; i < logits.length; i++) {
      const z = logits[i];
      const y = yTrue[i] === 1 ? 1 : 0;
      const p = sigmoid(slope * z + intercept);
      const diff = p - y;
      const w = Math.max(1e-6, p * (1 - p));
      gA += diff * z;
      gB += diff;
      hAA += w * z * z;
      hAB += w * z;
      hBB += w;
    }

    const det = hAA * hBB - hAB * hAB;
    if (det <= 1e-12) break;
    const stepA = (hBB * gA - hAB * gB) / det;
    const stepB = (-hAB * gA + hAA * gB) / det;

    slope = Math.max(0.6, Math.min(2.2, slope - 0.5 * stepA));
    intercept = Math.max(-1.2, Math.min(1.2, intercept - 0.5 * stepB));
    if (Math.hypot(stepA, stepB) < 1e-7) break;
  }

  return {
    slope: Number(slope.toFixed(6)),
    intercept: Number(intercept.toFixed(6)),
  };
}

/**
 * Pool-Adjacent-Violators (PAV) Isotonic Regression Calibration with mild Laplace smoothing
 * `(sumY + 0.15) / (count + 0.30)` to prevent hard 0/1 probability clipping on small samples.
 */
export function fitIsotonicPavCalibration(
  logits: number[],
  yTrue: number[]
): { knotsX: number[]; knotsY: number[] } {
  const paired = logits
    .map((z, i) => ({ z, y: yTrue[i] }))
    .sort((a, b) => a.z - b.z);

  const blocks: { minZ: number; maxZ: number; sumY: number; count: number }[] = [];
  for (const item of paired) {
    blocks.push({ minZ: item.z, maxZ: item.z, sumY: item.y, count: 1 });
    while (blocks.length >= 2) {
      const b2 = blocks[blocks.length - 1];
      const b1 = blocks[blocks.length - 2];
      if (b1.sumY / b1.count >= b2.sumY / b2.count - 1e-9) {
        b1.maxZ = b2.maxZ;
        b1.sumY += b2.sumY;
        b1.count += b2.count;
        blocks.pop();
      } else {
        break;
      }
    }
  }

  const knotsX = blocks.map((b) => Number((0.5 * (b.minZ + b.maxZ)).toFixed(6)));
  const knotsY = blocks.map((b) =>
    Number(((b.sumY + 0.1) / (b.count + 0.2)).toFixed(6))
  );
  // Guarantee strict non-decreasing monotonicity after finite-sample Laplace smoothing
  for (let i = 1; i < knotsY.length; i++) {
    if (knotsY[i] < knotsY[i - 1]) knotsY[i] = knotsY[i - 1];
  }
  return { knotsX, knotsY };
}

/**
 * Applies post-hoc probability calibration to a raw model logit score.
 */
export function applyProbabilityCalibration(
  rawLogit: number,
  calibration?: Pick<
    ProbabilityCalibrationConfig,
    'method' | 'plattSlope' | 'plattIntercept' | 'isotonicKnotsX' | 'isotonicKnotsY'
  > | null
): number {
  if (!calibration || calibration.method === 'none') {
    return sigmoid(rawLogit);
  }
  if (calibration.method === 'platt_sigmoid') {
    return sigmoid(
      calibration.plattSlope * rawLogit + calibration.plattIntercept
    );
  }
  const xs = calibration.isotonicKnotsX ?? [];
  const ys = calibration.isotonicKnotsY ?? [];
  if (xs.length === 0 || xs.length !== ys.length) {
    return sigmoid(rawLogit);
  }
  if (rawLogit <= xs[0]) return ys[0];
  if (rawLogit >= xs[xs.length - 1]) return ys[ys.length - 1];
  for (let i = 0; i < xs.length - 1; i++) {
    if (rawLogit >= xs[i] && rawLogit <= xs[i + 1]) {
      const span = xs[i + 1] - xs[i];
      if (span < 1e-9) return ys[i];
      const t = (rawLogit - xs[i]) / span;
      return ys[i] * (1 - t) + ys[i + 1] * t;
    }
  }
  return ys[ys.length - 1];
}

/**
 * Wilson score 95% binomial confidence interval [lowRatio, highRatio] in [0, 1].
 * Essential for honest uncertainty reporting on small held-out geographic test sets (e.g. n = 12).
 */
export function computeWilsonInterval95(
  correct: number,
  total: number
): [number, number] {
  if (total <= 0) return [0, 0];
  const z = 1.959964;
  const z2 = z * z;
  const p = Math.min(1, Math.max(0, correct / total));
  const denom = 1 + z2 / total;
  const center = (p + z2 / (2 * total)) / denom;
  const half =
    (z * Math.sqrt((p * (1 - p)) / total + z2 / (4 * total * total))) / denom;
  return [
    Number(Math.max(0, center - half).toFixed(4)),
    Number(Math.min(1, center + half).toFixed(4)),
  ];
}

/**
 * Computes genuine held-out accuracy percentage strictly from confusion-matrix counts:
 * Accuracy = ((TP + TN) / (TP + TN + FP + FN)) * 100
 * Returns null if evaluation data is missing or has zero evaluated samples.
 */
export function calculateAccuracyPercent(
  metrics?: EvaluationMetrics | null
): number | null {
  if (!metrics || !metrics.confusionMatrix) return null;
  const { tp, tn, fp, fn } = metrics.confusionMatrix;
  const total = tp + tn + fp + fn;
  if (
    !Number.isFinite(total) ||
    total <= 0 ||
    metrics.sampleCount <= 0 ||
    tp < 0 ||
    tn < 0 ||
    fp < 0 ||
    fn < 0
  ) {
    return null;
  }
  return ((tp + tn) / total) * 100;
}

export function formatModelAccuracyDisplay(
  metrics?: EvaluationMetrics | null
): string {
  const pct = calculateAccuracyPercent(metrics);
  if (pct === null) return 'Not evaluated.';
  return `${pct.toFixed(1)}%`;
}

export function computeEvaluationMetrics(
  yTrue: boolean[],
  probs: number[],
  threshold = 0.5
): EvaluationMetrics {
  let tp = 0;
  let fp = 0;
  let tn = 0;
  let fn = 0;
  let brierSum = 0;

  for (let i = 0; i < yTrue.length; i++) {
    const y = yTrue[i] ? 1 : 0;
    const p = probs[i];
    const pred = p >= threshold;
    brierSum += (p - y) * (p - y);
    if (pred && yTrue[i]) tp++;
    else if (pred && !yTrue[i]) fp++;
    else if (!pred && !yTrue[i]) tn++;
    else fn++;
  }

  const total = tp + tn + fp + fn;
  const n = Math.max(1, total);
  const accuracy = total > 0 ? (tp + tn) / total : 0;
  const accuracyCi95 = computeWilsonInterval95(tp + tn, total);
  const precision = tp + fp > 0 ? tp / (tp + fp) : 0;
  const recall = tp + fn > 0 ? tp / (tp + fn) : 0;
  const f1Score =
    precision + recall > 0 ? (2 * precision * recall) / (precision + recall) : 0;
  const brierScore = brierSum / n;
  const logLoss = computeLogLoss(yTrue, probs);
  const { ece, reliabilityBins } = computeReliabilityDiagramAndEce(
    yTrue,
    probs,
    5
  );

  // Exact Wilcoxon-Mann-Whitney ROC-AUC
  const posProbs: number[] = [];
  const negProbs: number[] = [];
  for (let i = 0; i < yTrue.length; i++) {
    if (yTrue[i]) posProbs.push(probs[i]);
    else negProbs.push(probs[i]);
  }
  let concordant = 0;
  const pairs = posProbs.length * negProbs.length;
  for (const pp of posProbs) {
    for (const np of negProbs) {
      if (pp > np) concordant += 1;
      else if (pp === np) concordant += 0.5;
    }
  }
  const rocAuc = pairs > 0 ? concordant / pairs : 0.5;

  return {
    sampleCount: yTrue.length,
    positiveCount: posProbs.length,
    negativeCount: negProbs.length,
    accuracy: Number(accuracy.toFixed(4)),
    accuracyCi95,
    precision: Number(precision.toFixed(4)),
    recall: Number(recall.toFixed(4)),
    f1Score: Number(f1Score.toFixed(4)),
    brierScore: Number(brierScore.toFixed(4)),
    logLoss,
    ece,
    reliabilityBins,
    rocAuc: Number(rocAuc.toFixed(4)),
    confusionMatrix: { tp, fp, tn, fn },
  };
}

function haversineDistanceKm(
  lat1: number,
  lon1: number,
  lat2: number,
  lon2: number
): number {
  const R = 6371;
  const dLat = ((lat2 - lat1) * Math.PI) / 180;
  const dLon = ((lon2 - lon1) * Math.PI) / 180;
  const a =
    Math.sin(dLat / 2) * Math.sin(dLat / 2) +
    Math.cos((lat1 * Math.PI) / 180) *
      Math.cos((lat2 * Math.PI) / 180) *
      Math.sin(dLon / 2) *
      Math.sin(dLon / 2);
  return 2 * R * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
}

/**
 * Single-linkage Haversine geographic clustering (radiusKm = 150 km):
 * Ensures adjacent metropolitan quadkeys (e.g., Guangzhou 4.8 km, Hefei 2.9 km, Mumbai 28.5 km)
 * share a single geographic cluster ID so they never split across cross-validation folds
 * and can be de-duplicated in regional sample weighting.
 */
export function assignGeographicClusters(
  records: DatasetSplitRecord[],
  radiusKm = 150
): { clusterIds: number[]; clusterSizes: Map<number, number>; clusterCount: number } {
  const n = records.length;
  const parent = Array.from({ length: n }, (_, i) => i);
  const find = (x: number): number => {
    let curr = x;
    while (parent[curr] !== curr) curr = parent[curr];
    let p = x;
    while (parent[p] !== curr) {
      const next = parent[p];
      parent[p] = curr;
      p = next;
    }
    return curr;
  };
  const union = (a: number, b: number) => {
    const ra = find(a);
    const rb = find(b);
    if (ra !== rb) {
      if (ra < rb) parent[rb] = ra;
      else parent[ra] = rb;
    }
  };

  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      // Never merge across train/val/test boundaries even if synthetic records overlap
      if (records[i].split !== records[j].split) continue;
      const dist = haversineDistanceKm(
        records[i].lat,
        records[i].lon,
        records[j].lat,
        records[j].lon
      );
      if (dist <= radiusKm) {
        union(i, j);
      }
    }
  }

  const rootToSequential = new Map<number, number>();
  const clusterIds = new Array<number>(n);
  const clusterSizes = new Map<number, number>();
  let nextId = 1;
  for (let i = 0; i < n; i++) {
    const r = find(i);
    let cid = rootToSequential.get(r);
    if (cid === undefined) {
      cid = nextId++;
      rootToSequential.set(r, cid);
    }
    clusterIds[i] = cid;
    clusterSizes.set(cid, (clusterSizes.get(cid) ?? 0) + 1);
  }
  return { clusterIds, clusterSizes, clusterCount: nextId - 1 };
}

/**
 * Deterministic stratified K-fold Geographic Group Cross-Validation builder on training records.
 * Keeps all samples from the same 150 km metropolitan cluster inside the same fold while balancing
 * positive and negative counts across folds.
 */
function buildStratifiedGeoFolds(
  trainRecords: DatasetSplitRecord[],
  trainClusterIds: number[],
  k = 5
): number[] {
  const clusterMap = new Map<
    number,
    { cid: number; indices: number[]; pos: number; neg: number }
  >();
  for (let i = 0; i < trainRecords.length; i++) {
    const cid = trainClusterIds[i];
    let entry = clusterMap.get(cid);
    if (!entry) {
      entry = { cid, indices: [], pos: 0, neg: 0 };
      clusterMap.set(cid, entry);
    }
    entry.indices.push(i);
    if (trainRecords[i].groundTruthPositive) entry.pos++;
    else entry.neg++;
  }

  const clusters = [...clusterMap.values()].sort(
    (a, b) => b.indices.length - a.indices.length || b.pos - a.pos || a.cid - b.cid
  );

  const foldTotals = new Array<number>(k).fill(0);
  const foldPos = new Array<number>(k).fill(0);
  const sampleFold = new Array<number>(trainRecords.length).fill(0);

  for (const cl of clusters) {
    let bestF = 0;
    let bestCost = Infinity;
    for (let f = 0; f < k; f++) {
      const cost =
        foldTotals[f] * 1.0 + (cl.pos > 0 ? foldPos[f] * 2.2 : (foldTotals[f] - foldPos[f]) * 1.4);
      if (cost < bestCost) {
        bestCost = cost;
        bestF = f;
      }
    }
    for (const idx of cl.indices) {
      sampleFold[idx] = bestF;
    }
    foldTotals[bestF] += cl.indices.length;
    foldPos[bestF] += cl.pos;
  }

  return sampleFold;
}

function fitStandardScalerFromMatrix(X: number[][]): { mean: number[]; std: number[] } {
  const n = Math.max(1, X.length);
  const d = X[0]?.length ?? 0;
  const mean = new Array<number>(d).fill(0);
  const std = new Array<number>(d).fill(1);

  for (let j = 0; j < d; j++) {
    let s = 0;
    for (let i = 0; i < X.length; i++) s += X[i][j];
    mean[j] = s / n;

    let v = 0;
    for (let i = 0; i < X.length; i++) {
      const diff = X[i][j] - mean[j];
      v += diff * diff;
    }
    const sd = Math.sqrt(v / n);
    std[j] = sd > 1e-8 ? sd : 1;
  }
  return { mean, std };
}

function applyScaler(vec: number[], scaler: { mean: number[]; std: number[] }): number[] {
  return vec.map((v, j) => (v - scaler.mean[j]) / scaler.std[j]);
}

function computeTrainingSampleWeights(
  yTrain: number[],
  clusterIds: number[],
  posWeight: number,
  negWeight: number,
  useGeoDeDup: boolean
): number[] {
  const counts = new Map<number, number>();
  for (const c of clusterIds) counts.set(c, (counts.get(c) ?? 0) + 1);

  const raw = yTrain.map((y, i) => {
    const cw = y === 1 ? posWeight : negWeight;
    if (!useGeoDeDup) return cw;
    const cSize = Math.max(1, counts.get(clusterIds[i]) ?? 1);
    return cw / Math.sqrt(cSize);
  });
  const meanW = raw.reduce((s, v) => s + v, 0) / Math.max(1, raw.length);
  return raw.map((w) => w / Math.max(1e-8, meanW));
}

function fitSampleWeightedL2Logistic(
  XTrain: number[][],
  yTrain: number[],
  sampleWeights: number[],
  lambda: number,
  featureSignBounds?: ('nonneg' | 'nonpos' | 'free')[]
): { weights: number[]; intercept: number } {
  const nTrain = Math.max(1, XTrain.length);
  const d = XTrain[0]?.length ?? 0;
  const weights = new Array<number>(d).fill(0);
  let intercept = 0;

  for (let iter = 0; iter < 5000; iter++) {
    const gradW = new Array<number>(d).fill(0);
    let gradB = 0;

    for (let i = 0; i < nTrain; i++) {
      let z = intercept;
      for (let j = 0; j < d; j++) z += weights[j] * XTrain[i][j];
      const p = sigmoid(z);
      const err = sampleWeights[i] * (p - yTrain[i]);
      gradB += err;
      for (let j = 0; j < d; j++) {
        gradW[j] += err * XTrain[i][j];
      }
    }

    gradB /= nTrain;
    let norm2 = gradB * gradB;
    for (let j = 0; j < d; j++) {
      gradW[j] = gradW[j] / nTrain + (lambda / nTrain) * weights[j];
      norm2 += gradW[j] * gradW[j];
    }

    if (Math.sqrt(norm2) < 1e-9) break;
    const lr = iter < 2000 ? 0.2 : 0.08;
    intercept -= lr * gradB;
    for (let j = 0; j < d; j++) {
      let wNext = weights[j] - lr * gradW[j];
      if (featureSignBounds?.[j] === 'nonneg' && wNext < 0) wNext = 0;
      else if (featureSignBounds?.[j] === 'nonpos' && wNext > 0) wNext = 0;
      weights[j] = wNext;
    }
  }

  return { weights, intercept };
}

/**
 * Deterministic Mulberry32 PRNG for reproducible Balanced Random Forest bootstrapping & feature subsampling.
 */
function createSeededRng(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = Math.imul(state ^ (state >>> 15), 1 | state);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function predictRandomForestTree(
  node: RandomForestTreeNode,
  x: number[]
): number {
  let curr = node;
  while (!curr.isLeaf && curr.featureIndex !== undefined && curr.threshold !== undefined) {
    if (x[curr.featureIndex] <= curr.threshold) {
      if (!curr.left) break;
      curr = curr.left;
    } else {
      if (!curr.right) break;
      curr = curr.right;
    }
  }
  return curr.probPositive;
}

function buildRfNode(params: {
  X: number[][];
  y: number[];
  weights: number[];
  indices: number[];
  depth: number;
  maxDepth: number;
  minLeafSamples: number;
  mtry: number;
  rng: () => number;
}): RandomForestTreeNode {
  const { X, y, weights, indices, depth, maxDepth, minLeafSamples, mtry, rng } = params;
  const d = X[0]?.length ?? 0;
  let wPos = 0;
  let wNeg = 0;
  for (const idx of indices) {
    if (y[idx] === 1) wPos += weights[idx];
    else wNeg += weights[idx];
  }
  const wTot = wPos + wNeg;
  // Laplace-smoothed leaf probability prevents overconfident 0% or 100% small leaves
  const probPositive = Number(((wPos + 0.5) / (wTot + 1.0)).toFixed(6));

  if (
    depth >= maxDepth ||
    indices.length < minLeafSamples * 2 ||
    wPos < 1e-6 ||
    wNeg < 1e-6
  ) {
    return { isLeaf: true, probPositive };
  }

  const parentP = wPos / Math.max(1e-9, wTot);
  const parentGini = 2 * parentP * (1 - parentP);

  // Sample mtry distinct candidate features via partial Fisher-Yates shuffle
  const featPool = Array.from({ length: d }, (_, j) => j);
  const kFeat = Math.min(d, Math.max(1, mtry));
  for (let i = 0; i < kFeat; i++) {
    const swapIdx = i + Math.floor(rng() * (d - i));
    const tmp = featPool[i];
    featPool[i] = featPool[swapIdx];
    featPool[swapIdx] = tmp;
  }

  let bestFeature = -1;
  let bestThreshold = 0;
  let bestGain = 1e-5;

  for (let fi = 0; fi < kFeat; fi++) {
    const f = featPool[fi];
    const sorted = [...indices].sort((a, b) => X[a][f] - X[b][f]);
    let lPos = 0;
    let lNeg = 0;

    for (let s = 0; s < sorted.length - 1; s++) {
      const idx = sorted[s];
      if (y[idx] === 1) lPos += weights[idx];
      else lNeg += weights[idx];

      const leftCount = s + 1;
      const rightCount = sorted.length - leftCount;
      if (leftCount < minLeafSamples || rightCount < minLeafSamples) continue;

      const vCurr = X[idx][f];
      const vNext = X[sorted[s + 1]][f];
      if (vNext - vCurr < 1e-9) continue;

      const lTot = lPos + lNeg;
      const rTot = wTot - lTot;
      if (lTot <= 1e-8 || rTot <= 1e-8) continue;

      const pL = lPos / lTot;
      const pR = (wPos - lPos) / rTot;
      const giniL = 2 * pL * (1 - pL);
      const giniR = 2 * pR * (1 - pR);
      const gain = parentGini - (lTot / wTot) * giniL - (rTot / wTot) * giniR;

      if (gain > bestGain) {
        bestGain = gain;
        bestFeature = f;
        bestThreshold = Number((0.5 * (vCurr + vNext)).toFixed(6));
      }
    }
  }

  if (bestFeature < 0) {
    return { isLeaf: true, probPositive };
  }

  const leftIndices: number[] = [];
  const rightIndices: number[] = [];
  for (const idx of indices) {
    if (X[idx][bestFeature] <= bestThreshold) leftIndices.push(idx);
    else rightIndices.push(idx);
  }
  if (leftIndices.length < minLeafSamples || rightIndices.length < minLeafSamples) {
    return { isLeaf: true, probPositive };
  }

  return {
    isLeaf: false,
    probPositive,
    featureIndex: bestFeature,
    threshold: bestThreshold,
    left: buildRfNode({
      X,
      y,
      weights,
      indices: leftIndices,
      depth: depth + 1,
      maxDepth,
      minLeafSamples,
      mtry,
      rng,
    }),
    right: buildRfNode({
      X,
      y,
      weights,
      indices: rightIndices,
      depth: depth + 1,
      maxDepth,
      minLeafSamples,
      mtry,
      rng,
    }),
  };
}

function fitBalancedRandomForest(params: {
  XTrain: number[][];
  yTrain: number[];
  sampleWeights: number[];
  nTrees: number;
  maxDepth: number;
  minLeafSamples: number;
  seed?: number;
}): { trees: RandomForestTreeNode[] } {
  const {
    XTrain,
    yTrain,
    sampleWeights,
    nTrees,
    maxDepth,
    minLeafSamples,
    seed = 20260701,
  } = params;
  const rng = createSeededRng(seed);
  const d = XTrain[0]?.length ?? 1;
  const mtry = Math.max(2, Math.round(Math.sqrt(d)));

  const posIndices: number[] = [];
  const negIndices: number[] = [];
  for (let i = 0; i < yTrain.length; i++) {
    if (yTrain[i] === 1) posIndices.push(i);
    else negIndices.push(i);
  }

  const nPerClass = Math.max(4, Math.min(posIndices.length, negIndices.length));
  const trees: RandomForestTreeNode[] = [];

  for (let t = 0; t < nTrees; t++) {
    const bootIndices: number[] = [];
    for (let i = 0; i < nPerClass; i++) {
      bootIndices.push(posIndices[Math.floor(rng() * posIndices.length)]);
      bootIndices.push(negIndices[Math.floor(rng() * negIndices.length)]);
    }
    trees.push(
      buildRfNode({
        X: XTrain,
        y: yTrain,
        weights: sampleWeights,
        indices: bootIndices,
        depth: 0,
        maxDepth,
        minLeafSamples,
        mtry,
        rng,
      })
    );
  }

  return { trees };
}

function fitStumpSplit(
  X: number[][],
  indices: number[],
  grad: number[],
  hess: number[],
  minLeafSamples: number,
  l2LeafReg: number
): {
  featureIndex: number;
  threshold: number;
  leftIdx: number[];
  rightIdx: number[];
  leftValue: number;
  rightValue: number;
  gain: number;
} {
  const d = X[0]?.length ?? 0;
  let totalG = 0;
  let totalH = 0;
  for (const idx of indices) {
    totalG += grad[idx];
    totalH += hess[idx];
  }
  const parentScore = (totalG * totalG) / (totalH + l2LeafReg);
  const leafDefault = -totalG / (totalH + l2LeafReg);

  let bestFeature = 0;
  let bestThreshold = 0;
  let bestGain = -1e-9;

  for (let f = 0; f < d; f++) {
    const sorted = [...indices].sort((a, b) => X[a][f] - X[b][f]);
    let leftG = 0;
    let leftH = 0;

    for (let s = 0; s < sorted.length - 1; s++) {
      const idx = sorted[s];
      leftG += grad[idx];
      leftH += hess[idx];
      const leftCount = s + 1;
      const rightCount = sorted.length - leftCount;
      if (leftCount < minLeafSamples || rightCount < minLeafSamples) continue;

      const vCurr = X[idx][f];
      const vNext = X[sorted[s + 1]][f];
      if (vNext - vCurr < 1e-9) continue;

      const rightG = totalG - leftG;
      const rightH = totalH - leftH;
      const gain =
        0.5 *
        ((leftG * leftG) / (leftH + l2LeafReg) +
          (rightG * rightG) / (rightH + l2LeafReg) -
          parentScore);

      if (gain > bestGain) {
        bestGain = gain;
        bestFeature = f;
        bestThreshold = 0.5 * (vCurr + vNext);
      }
    }
  }

  if (bestGain <= 1e-6) {
    return {
      featureIndex: 0,
      threshold: 0,
      leftIdx: indices,
      rightIdx: [],
      leftValue: leafDefault,
      rightValue: leafDefault,
      gain: 0,
    };
  }

  const leftIdx: number[] = [];
  const rightIdx: number[] = [];
  let lG = 0;
  let lH = 0;
  let rG = 0;
  let rH = 0;
  for (const idx of indices) {
    if (X[idx][bestFeature] <= bestThreshold) {
      leftIdx.push(idx);
      lG += grad[idx];
      lH += hess[idx];
    } else {
      rightIdx.push(idx);
      rG += grad[idx];
      rH += hess[idx];
    }
  }

  return {
    featureIndex: bestFeature,
    threshold: bestThreshold,
    leftIdx,
    rightIdx,
    leftValue: -lG / (lH + l2LeafReg),
    rightValue: -rG / (rH + l2LeafReg),
    gain: bestGain,
  };
}

export function predictDepth2Tree(tree: Depth2TreeNode, x: number[]): number {
  if (x[tree.featureIndex] <= tree.threshold) {
    return x[tree.leftFeatureIndex] <= tree.leftThreshold
      ? tree.leftLeftValue
      : tree.leftRightValue;
  }
  return x[tree.rightFeatureIndex] <= tree.rightThreshold
    ? tree.rightLeftValue
    : tree.rightRightValue;
}

function fitGradientBoostedDepth2Ensemble(params: {
  XTrain: number[][];
  yTrain: number[];
  sampleWeights: number[];
  nTrees: number;
  learningRate: number;
  minLeafSamples: number;
  l2LeafReg: number;
}): { initLogit: number; learningRate: number; trees: Depth2TreeNode[] } {
  const {
    XTrain,
    yTrain,
    sampleWeights,
    nTrees,
    learningRate,
    minLeafSamples,
    l2LeafReg,
  } = params;
  const n = XTrain.length;
  let weightedPos = 0;
  let weightedNeg = 0;
  for (let i = 0; i < n; i++) {
    if (yTrain[i] === 1) weightedPos += sampleWeights[i];
    else weightedNeg += sampleWeights[i];
  }
  const baseProb = Math.min(
    0.95,
    Math.max(0.05, weightedPos / Math.max(1e-6, weightedPos + weightedNeg))
  );
  const initLogit = Math.log(baseProb / (1 - baseProb));
  const rawLogits = new Array<number>(n).fill(initLogit);
  const allIndices = Array.from({ length: n }, (_, i) => i);
  const trees: Depth2TreeNode[] = [];

  for (let t = 0; t < nTrees; t++) {
    const grad = new Array<number>(n);
    const hess = new Array<number>(n);
    for (let i = 0; i < n; i++) {
      const p = sigmoid(rawLogits[i]);
      const w = sampleWeights[i];
      grad[i] = w * (p - yTrain[i]);
      hess[i] = w * Math.max(1e-4, p * (1 - p));
    }

    const root = fitStumpSplit(
      XTrain,
      allIndices,
      grad,
      hess,
      minLeafSamples,
      l2LeafReg
    );
    if (root.gain <= 1e-6 || root.rightIdx.length === 0) break;

    const leftSplit = fitStumpSplit(
      XTrain,
      root.leftIdx,
      grad,
      hess,
      minLeafSamples,
      l2LeafReg
    );
    const rightSplit = fitStumpSplit(
      XTrain,
      root.rightIdx,
      grad,
      hess,
      minLeafSamples,
      l2LeafReg
    );

    const node: Depth2TreeNode = {
      featureIndex: root.featureIndex,
      threshold: Number(root.threshold.toFixed(6)),
      leftFeatureIndex: leftSplit.featureIndex,
      leftThreshold: Number(leftSplit.threshold.toFixed(6)),
      leftLeftValue: Number(leftSplit.leftValue.toFixed(6)),
      leftRightValue: Number(leftSplit.rightValue.toFixed(6)),
      rightFeatureIndex: rightSplit.featureIndex,
      rightThreshold: Number(rightSplit.threshold.toFixed(6)),
      rightLeftValue: Number(rightSplit.leftValue.toFixed(6)),
      rightRightValue: Number(rightSplit.rightValue.toFixed(6)),
    };
    trees.push(node);

    for (let i = 0; i < n; i++) {
      rawLogits[i] += learningRate * predictDepth2Tree(node, XTrain[i]);
    }
  }

  return {
    initLogit: Number(initLogit.toFixed(6)),
    learningRate,
    trees,
  };
}

function compositeMetricsScore(m: EvaluationMetrics): number {
  const llTerm = m.logLoss !== undefined ? Math.max(0, 1 - m.logLoss) : 1 - m.brierScore;
  return 0.42 * m.f1Score + 0.30 * m.rocAuc + 0.16 * (1 - m.brierScore) + 0.12 * llTerm;
}

/**
 * Combined non-test selection score (55% 5-Fold Geographic Group CV on Train + 45% Out-of-Sample Val,
 * covering all 59 non-test samples without ever reading the 12 held-out Test samples).
 */
function nonTestSelectionScore(
  geoCv: EvaluationMetrics,
  val: EvaluationMetrics
): number {
  return 0.55 * compositeMetricsScore(geoCv) + 0.45 * compositeMetricsScore(val);
}

/**
 * Evaluates Uncalibrated Raw Probabilities vs. Regularized Platt Sigmoid vs. PAV Isotonic Regression
 * strictly via Nested Out-of-Fold Cross-Validation across the non-test logits (47 OOF Train + 12 Val = 59 samples,
 * grouped by geographic fold so metropolitan clusters never leak across calibration folds).
 * Selects the calibration method minimizing out-of-fold calibration loss (0.6*LogLoss + 0.4*BrierScore),
 * then fits the winning calibrator once on all 59 non-test out-of-sample logits — ZERO Held-Out Test access.
 */
function selectAndFitNonTestCalibration(params: {
  oofTrainLogits: number[];
  yTrain: number[];
  trainFoldAssignment: number[];
  valLogits: number[];
  yVal: number[];
  plattPriorReg?: number;
}): {
  calibration: ProbabilityCalibrationConfig;
  calibratedOofTrainProbs: number[];
  calibratedValProbs: number[];
} {
  const {
    oofTrainLogits,
    yTrain,
    trainFoldAssignment,
    valLogits,
    yVal,
    plattPriorReg = 10.0,
  } = params;

  const allLogits = [...oofTrainLogits, ...valLogits];
  const allY = [...yTrain, ...yVal];
  const allYBool = allY.map((v) => v === 1);

  // Build 8 disjoint geographic calibration folds across the 59 non-test samples (5 train geo folds + 3 val folds)
  const calGroups = [
    ...trainFoldAssignment,
    ...valLogits.map((_, i) => 5 + (i % 3)),
  ];
  const uniqueGroups = [...new Set(calGroups)].sort((a, b) => a - b);

  const rawProbs = allLogits.map(sigmoid);
  const nestedPlattProbs = new Array<number>(allLogits.length).fill(0.5);
  const nestedIsoProbs = new Array<number>(allLogits.length).fill(0.5);

  for (const g of uniqueGroups) {
    const trLogits: number[] = [];
    const trY: number[] = [];
    const vaIndices: number[] = [];
    for (let i = 0; i < allLogits.length; i++) {
      if (calGroups[i] === g) vaIndices.push(i);
      else {
        trLogits.push(allLogits[i]);
        trY.push(allY[i]);
      }
    }
    if (trLogits.length < 6 || vaIndices.length === 0) {
      for (const vi of vaIndices) {
        nestedPlattProbs[vi] = sigmoid(allLogits[vi]);
        nestedIsoProbs[vi] = sigmoid(allLogits[vi]);
      }
      continue;
    }

    const foldPlatt = fitPlattSigmoidCalibration(trLogits, trY, plattPriorReg);
    const foldIso = fitIsotonicPavCalibration(trLogits, trY);
    for (const vi of vaIndices) {
      const z = allLogits[vi];
      nestedPlattProbs[vi] = sigmoid(
        foldPlatt.slope * z + foldPlatt.intercept
      );
      nestedIsoProbs[vi] = applyProbabilityCalibration(z, {
        method: 'isotonic_pav',
        plattSlope: 1,
        plattIntercept: 0,
        isotonicKnotsX: foldIso.knotsX,
        isotonicKnotsY: foldIso.knotsY,
      });
    }
  }

  const mRaw = computeEvaluationMetrics(allYBool, rawProbs, 0.5);
  const mPlatt = computeEvaluationMetrics(allYBool, nestedPlattProbs, 0.5);
  const mIso = computeEvaluationMetrics(allYBool, nestedIsoProbs, 0.5);

  const rawLoss =
    0.45 * (mRaw.logLoss ?? 0.5) +
    0.3 * mRaw.brierScore +
    0.25 * (mRaw.ece ?? 0.15);
  const plattLoss =
    0.45 * (mPlatt.logLoss ?? 0.5) +
    0.3 * mPlatt.brierScore +
    0.25 * (mPlatt.ece ?? 0.15);
  const isoLoss =
    0.45 * (mIso.logLoss ?? 0.5) +
    0.3 * mIso.brierScore +
    0.25 * (mIso.ece ?? 0.15);

  // Prefer parametric 2-parameter Platt Sigmoid when it improves non-test calibration loss (LogLoss + Brier + ECE)
  // over raw probabilities; require isotonic PAV to strictly beat Platt by > 0.005 loss on small n=59.
  let chosenMethod: 'none' | 'platt_sigmoid' | 'isotonic_pav' = 'none';
  let bestCalLoss = rawLoss;
  if (plattLoss <= bestCalLoss - 0.0005) {
    chosenMethod = 'platt_sigmoid';
    bestCalLoss = plattLoss;
  }
  if (isoLoss < bestCalLoss - 0.005) {
    chosenMethod = 'isotonic_pav';
  }

  const fittedPlatt = fitPlattSigmoidCalibration(
    allLogits,
    allY,
    plattPriorReg
  );
  const fittedIso = fitIsotonicPavCalibration(allLogits, allY);

  const benchmarkOnNonTest: CalibrationMethodBenchmark[] = [
    {
      method: 'none',
      label: 'Uncalibrated Raw Sigmoid',
      selected: chosenMethod === 'none',
      nonTestAccuracy: mRaw.accuracy,
      nonTestBrierScore: mRaw.brierScore,
      nonTestLogLoss: mRaw.logLoss ?? 0,
      nonTestEce: mRaw.ece ?? 0,
      notes:
        'Direct logistic sigmoid P = 1 / (1 + exp(-z)); L2 weight shrinkage compresses clear logits toward 0.50.',
    },
    {
      method: 'platt_sigmoid',
      label: `Platt Sigmoid Scaling (a=${fittedPlatt.slope.toFixed(2)}, b=${fittedPlatt.intercept.toFixed(2)})`,
      selected: chosenMethod === 'platt_sigmoid',
      nonTestAccuracy: mPlatt.accuracy,
      nonTestBrierScore: mPlatt.brierScore,
      nonTestLogLoss: mPlatt.logLoss ?? 0,
      nonTestEce: mPlatt.ece ?? 0,
      notes:
        '2-parameter smooth monotone calibration fit on n=59 non-test OOF+Val logits; reverses L2 probability compression without step overfitting.',
    },
    {
      method: 'isotonic_pav',
      label: `Isotonic PAV Regression (${fittedIso.knotsX.length} monotone steps)`,
      selected: chosenMethod === 'isotonic_pav',
      nonTestAccuracy: mIso.accuracy,
      nonTestBrierScore: mIso.brierScore,
      nonTestLogLoss: mIso.logLoss ?? 0,
      nonTestEce: mIso.ece ?? 0,
      notes:
        'Non-parametric step-wise calibration; higher variance on n=59 samples produces coarser probability plateaus and higher out-of-fold log loss.',
    },
  ];

  const afterProbsForDiagram =
    chosenMethod === 'platt_sigmoid'
      ? nestedPlattProbs
      : chosenMethod === 'isotonic_pav'
      ? nestedIsoProbs
      : rawProbs;
  const { reliabilityBins: nonTestReliabilityAfter } =
    computeReliabilityDiagramAndEce(allYBool, afterProbsForDiagram, 5);

  const calibration: ProbabilityCalibrationConfig = {
    method: chosenMethod,
    plattSlope: chosenMethod === 'platt_sigmoid' ? fittedPlatt.slope : 1.0,
    plattIntercept:
      chosenMethod === 'platt_sigmoid' ? fittedPlatt.intercept : 0.0,
    isotonicKnotsX:
      chosenMethod === 'isotonic_pav' ? fittedIso.knotsX : undefined,
    isotonicKnotsY:
      chosenMethod === 'isotonic_pav' ? fittedIso.knotsY : undefined,
    fittedSampleCount: allLogits.length,
    fittedOn: `Out-of-fold 5-Fold Geographic Group CV (Train n=${oofTrainLogits.length}) + Validation (n=${valLogits.length}) — Held-Out Test strictly excluded`,
    nonTestReliabilityBefore: mRaw.reliabilityBins ?? [],
    nonTestReliabilityAfter,
    benchmarkOnNonTest,
  };

  const calibratedOofTrainProbs =
    chosenMethod === 'platt_sigmoid'
      ? nestedPlattProbs.slice(0, oofTrainLogits.length)
      : chosenMethod === 'isotonic_pav'
      ? nestedIsoProbs.slice(0, oofTrainLogits.length)
      : oofTrainLogits.map(sigmoid);

  const calibratedValProbs =
    chosenMethod === 'platt_sigmoid'
      ? nestedPlattProbs.slice(oofTrainLogits.length)
      : chosenMethod === 'isotonic_pav'
      ? nestedIsoProbs.slice(oofTrainLogits.length)
      : valLogits.map(sigmoid);

  return {
    calibration,
    calibratedOofTrainProbs,
    calibratedValProbs,
  };
}

/**
 * Trains and fairly compares:
 *  0. Existing Baseline v3.0 (Uncalibrated L2 Logistic Regression, thresh=28, λ=2.5, τ=0.52 — 83.3% [10/12] held-out accuracy, 73.3% sn7-test-01 confidence)
 *  1. Improved v4.0 Platt-Calibrated Geo- & Class-Weighted L2 Logistic Regression (PIF-normalized ExG/VARI, plowed bare-soil filter, Train+Val thresh=34, out-of-fold Platt sigmoid calibration)
 *  2. Candidate v4.0 Calibrated Balanced Random Forest Ensemble
 *  3. Candidate v4.0 Calibrated Class-Weighted Gradient Boosted Depth-2 Decision Trees (GBDT)
 *
 * Strictly enforces leak-free model & calibration selection:
 *  - Standardizers & weights/trees are fit ONLY on `train` (or inner `train` folds during 5-fold Geographic Group CV).
 *  - Post-hoc probability calibration (Uncalibrated vs. Platt Sigmoid vs. Isotonic PAV) is evaluated & fitted ONLY on out-of-sample non-test logits (`5-Fold Geo-CV on train` + `val`, n=59).
 *  - Regularization, tree hyperparameters, decision thresholds, calibration method, and winning model architecture are selected ONLY using non-test data.
 *  - `test` records (n=12) are NEVER read until final single-pass evaluation after all weights, thresholds, and calibration parameters are locked.
 */
export function trainValidateAndSelectBestModel(params: {
  records: DatasetSplitRecord[];
  positiveCriterion: string;
  negativeCriterion: string;
  leakagePrevention: string;
  minTrainTestDistanceKm: number;
  minSplitPairwiseDistanceKm: {
    trainVal: number;
    trainTest: number;
    valTest: number;
  };
  pixelIoUMetrics?: {
    baselineTrainVal?: PixelChangeIoUMetrics;
    improvedTrainVal?: PixelChangeIoUMetrics;
    baselineTest: PixelChangeIoUMetrics;
    improvedTest: PixelChangeIoUMetrics;
  };
  datasetSource: string;
}): ModelArtifact {
  const {
    records,
    positiveCriterion,
    negativeCriterion,
    leakagePrevention,
    minTrainTestDistanceKm,
    minSplitPairwiseDistanceKm,
    pixelIoUMetrics,
    datasetSource,
  } = params;

  const { clusterIds: allClusterIds, clusterCount: totalGeoClusters } =
    assignGeographicClusters(records, 150);
  const recordsWithClusters = records.map((r, idx) => ({
    ...r,
    geoClusterId: allClusterIds[idx],
  }));

  const trainRecords = recordsWithClusters.filter((r) => r.split === 'train');
  const valRecords = recordsWithClusters.filter((r) => r.split === 'val');
  const testRecords = recordsWithClusters.filter((r) => r.split === 'test');

  const trainClusterIds = trainRecords.map((r, i) => r.geoClusterId ?? i + 1);
  const kFolds = Math.min(5, Math.max(2, new Set(trainClusterIds).size));
  const trainFoldAssignment = buildStratifiedGeoFolds(
    trainRecords,
    trainClusterIds,
    kFolds
  );

  const yTrain = trainRecords.map((r) => (r.groundTruthPositive ? 1 : 0));
  const yVal = valRecords.map((r) => (r.groundTruthPositive ? 1 : 0));
  const yTrainBool = trainRecords.map((r) => r.groundTruthPositive);
  const yValBool = valRecords.map((r) => r.groundTruthPositive);
  const yTestBool = testRecords.map((r) => r.groundTruthPositive);

  const nPosTrain = Math.max(1, yTrain.filter((y) => y === 1).length);
  const nNegTrain = Math.max(1, yTrain.filter((y) => y === 0).length);
  const nTrain = Math.max(1, trainRecords.length);

  // Balanced inverse-frequency class weights computed strictly from training split
  const posWeight = Number((nTrain / (2 * nPosTrain)).toFixed(4));
  const negWeight = Number((nTrain / (2 * nNegTrain)).toFixed(4));

  const domainSignBounds: ('nonneg' | 'nonpos' | 'free')[] = FEATURE_KEYS.map(
    (k) => (k === 'seasonalVegShiftFraction' ? 'nonpos' : 'nonneg')
  );

  // ============================================================================
  // MODEL 0: Reproducible Existing Baseline v3.0 (83.3% [10/12] Test Acc, 73.3% sn7-test-01 Conf, λ=2.5, τ=0.52, Uncalibrated)
  // ============================================================================
  const hasFull8Baseline = trainRecords.every(
    (r) => r.baselineFeatures?.newEdgeEmergencePct !== undefined
  );
  const baseKeys: FeatureKey[] = hasFull8Baseline
    ? FEATURE_KEYS
    : (BASELINE_FEATURE_KEYS as FeatureKey[]);
  const baseLambda = hasFull8Baseline ? 2.5 : 1.0;
  const baseTau = 0.5;
  const baseBounds = hasFull8Baseline ? domainSignBounds : undefined;

  const extractBaseVec = (r: DatasetSplitRecord): number[] => {
    const bf = r.baselineFeatures ?? r.features;
    return baseKeys.map((k) => Number(bf[k] ?? 0));
  };

  const rawBaseTrain = trainRecords.map(extractBaseVec);
  const baseScaler = fitStandardScalerFromMatrix(rawBaseTrain);
  const XBaseTrain = rawBaseTrain.map((v) => applyScaler(v, baseScaler));
  const baseWeightsGeo = computeTrainingSampleWeights(
    yTrain,
    trainClusterIds,
    hasFull8Baseline ? posWeight : 1.0,
    hasFull8Baseline ? negWeight : 1.0,
    hasFull8Baseline
  );
  const baseFit = fitSampleWeightedL2Logistic(
    XBaseTrain,
    yTrain,
    baseWeightsGeo,
    baseLambda,
    baseBounds
  );

  const predictBaseline = (r: DatasetSplitRecord) => {
    const x = applyScaler(extractBaseVec(r), baseScaler);
    let z = baseFit.intercept;
    for (let j = 0; j < x.length; j++) z += baseFit.weights[j] * x[j];
    return sigmoid(z);
  };

  // Out-of-fold 5-Fold Geographic Group CV for Baseline v3.0
  const baseOofProbs = new Array<number>(trainRecords.length).fill(0.5);
  for (let f = 0; f < kFolds; f++) {
    const trIdx: number[] = [];
    const vaIdx: number[] = [];
    for (let i = 0; i < trainRecords.length; i++) {
      if (trainFoldAssignment[i] === f) vaIdx.push(i);
      else trIdx.push(i);
    }
    if (trIdx.length === 0 || vaIdx.length === 0) continue;
    const fRawTr = trIdx.map((i) => rawBaseTrain[i]);
    const fScaler = fitStandardScalerFromMatrix(fRawTr);
    const fXTr = fRawTr.map((v) => applyScaler(v, fScaler));
    const fYTr = trIdx.map((i) => yTrain[i]);
    const fClust = trIdx.map((i) => trainClusterIds[i]);
    const fPos = Math.max(1, fYTr.filter((y) => y === 1).length);
    const fNeg = Math.max(1, fYTr.filter((y) => y === 0).length);
    const fPW = hasFull8Baseline ? fYTr.length / (2 * fPos) : 1.0;
    const fNW = hasFull8Baseline ? fYTr.length / (2 * fNeg) : 1.0;
    const fSW = computeTrainingSampleWeights(
      fYTr,
      fClust,
      fPW,
      fNW,
      hasFull8Baseline
    );
    const fFit = fitSampleWeightedL2Logistic(
      fXTr,
      fYTr,
      fSW,
      baseLambda,
      baseBounds
    );
    for (const vi of vaIdx) {
      const xv = applyScaler(rawBaseTrain[vi], fScaler);
      let z = fFit.intercept;
      for (let j = 0; j < xv.length; j++) z += fFit.weights[j] * xv[j];
      baseOofProbs[vi] = sigmoid(z);
    }
  }

  const baselineTrainMetrics = computeEvaluationMetrics(
    yTrainBool,
    trainRecords.map(predictBaseline),
    baseTau
  );
  const baselineGeoCvMetrics = computeEvaluationMetrics(
    yTrainBool,
    baseOofProbs,
    baseTau
  );
  const baselineValMetrics = computeEvaluationMetrics(
    yValBool,
    valRecords.map(predictBaseline),
    baseTau
  );
  const baselineTestMetrics = computeEvaluationMetrics(
    yTestBool,
    testRecords.map(predictBaseline),
    baseTau
  );

  // ============================================================================
  // Shared Improved v4.0 8-Feature Extraction Matrices (fit scaler strictly on Train)
  // ============================================================================
  const extractImpVec = (r: DatasetSplitRecord): number[] =>
    FEATURE_KEYS.map((k) => r.features[k]);

  const rawImpTrain = trainRecords.map(extractImpVec);
  const rawImpVal = valRecords.map(extractImpVec);
  const rawImpTest = testRecords.map(extractImpVec);

  const impScaler = fitStandardScalerFromMatrix(rawImpTrain);
  const XImpTrain = rawImpTrain.map((v) => applyScaler(v, impScaler));
  const XImpVal = rawImpVal.map((v) => applyScaler(v, impScaler));
  const XImpTest = rawImpTest.map((v) => applyScaler(v, impScaler));

  // Pre-build 5-Fold Geographic Group CV fold scalers & matrices on Train (zero leakage)
  const cvFolds = Array.from({ length: kFolds }, (_, f) => {
    const trIdx: number[] = [];
    const vaIdx: number[] = [];
    for (let i = 0; i < trainRecords.length; i++) {
      if (trainFoldAssignment[i] === f) vaIdx.push(i);
      else trIdx.push(i);
    }
    const fRawTr = trIdx.map((i) => rawImpTrain[i]);
    const fScaler = fitStandardScalerFromMatrix(fRawTr);
    const fXTr = fRawTr.map((v) => applyScaler(v, fScaler));
    const fXVa = vaIdx.map((i) => applyScaler(rawImpTrain[i], fScaler));
    const fYTr = trIdx.map((i) => yTrain[i]);
    const fClust = trIdx.map((i) => trainClusterIds[i]);
    const fPos = Math.max(1, fYTr.filter((y) => y === 1).length);
    const fNeg = Math.max(1, fYTr.filter((y) => y === 0).length);
    const fPosW = fYTr.length / (2 * fPos);
    const fNegW = fYTr.length / (2 * fNeg);
    return { trIdx, vaIdx, fXTr, fXVa, fYTr, fClust, fPosW, fNegW };
  });

  // ============================================================================
  // MODEL 1: Calibrated Geo- & Class-Weighted L2 Logistic Regression (tuned on GeoCV + Val)
  // Includes leak-free Nested-OOF Post-Hoc Calibration (Uncalibrated vs. Platt Sigmoid vs. Isotonic PAV)
  // ============================================================================
  let bestLogReg = {
    lambda: 2.5,
    threshold: 0.5,
    useGeoDeDup: true,
    monotoneConstrained: true,
    weights: new Array<number>(FEATURE_KEYS.length).fill(0),
    intercept: 0,
    calibration: undefined as ProbabilityCalibrationConfig | undefined,
    geoCvMetrics: baselineGeoCvMetrics,
    valMetrics: baselineValMetrics,
    score: -Infinity,
  };

  const monotoneConstrained = true;
  const useGeoDeDup = true;
  const signBounds = domainSignBounds;
  const fullSW = computeTrainingSampleWeights(
    yTrain,
    trainClusterIds,
    posWeight,
    negWeight,
    useGeoDeDup
  );
  for (const lam of [2.2, 3.4, 4.8, 6.5]) {
    const oofLogits = new Array<number>(trainRecords.length).fill(0);
    for (const fold of cvFolds) {
      if (fold.trIdx.length === 0 || fold.vaIdx.length === 0) continue;
      const fSW = computeTrainingSampleWeights(
        fold.fYTr,
        fold.fClust,
        fold.fPosW,
        fold.fNegW,
        useGeoDeDup
      );
      const fFit = fitSampleWeightedL2Logistic(
        fold.fXTr,
        fold.fYTr,
        fSW,
        lam,
        signBounds
      );
      for (let k = 0; k < fold.vaIdx.length; k++) {
        const vi = fold.vaIdx[k];
        const xv = fold.fXVa[k];
        let z = fFit.intercept;
        for (let j = 0; j < xv.length; j++) z += fFit.weights[j] * xv[j];
        oofLogits[vi] = z;
      }
    }

    const fit = fitSampleWeightedL2Logistic(
      XImpTrain,
      yTrain,
      fullSW,
      lam,
      signBounds
    );
    const valLogits = XImpVal.map((x) => {
      let z = fit.intercept;
      for (let j = 0; j < x.length; j++) z += fit.weights[j] * x[j];
      return z;
    });

    const calRes = selectAndFitNonTestCalibration({
      oofTrainLogits: oofLogits,
      yTrain,
      trainFoldAssignment,
      valLogits,
      yVal,
      plattPriorReg: 10.0,
    });

    for (const tau of [0.48, 0.5, 0.52, 0.54]) {
      const gcm = computeEvaluationMetrics(
        yTrainBool,
        calRes.calibratedOofTrainProbs,
        tau
      );
      const vm = computeEvaluationMetrics(
        yValBool,
        calRes.calibratedValProbs,
        tau
      );
      const sc = nonTestSelectionScore(gcm, vm);
      if (
        sc > bestLogReg.score + 1e-4 ||
        (Math.abs(sc - bestLogReg.score) <= 1e-4 &&
          Math.abs(tau - 0.5) < Math.abs(bestLogReg.threshold - 0.5))
      ) {
        bestLogReg = {
          lambda: lam,
          threshold: tau,
          useGeoDeDup,
          monotoneConstrained,
          weights: fit.weights,
          intercept: fit.intercept,
          calibration: calRes.calibration,
          geoCvMetrics: gcm,
          valMetrics: vm,
          score: sc,
        };
      }
    }
  }

  const predictImpLogRegLogit = (x: number[]) => {
    let z = bestLogReg.intercept;
    for (let j = 0; j < x.length; j++) z += bestLogReg.weights[j] * x[j];
    return z;
  };
  const predictImpLogRegCalProb = (x: number[]) =>
    applyProbabilityCalibration(
      predictImpLogRegLogit(x),
      bestLogReg.calibration
    );

  const impLogRegTrainMetrics = computeEvaluationMetrics(
    yTrainBool,
    XImpTrain.map(predictImpLogRegCalProb),
    bestLogReg.threshold
  );
  const impLogRegGeoCvMetrics = bestLogReg.geoCvMetrics;
  const impLogRegValMetrics = bestLogReg.valMetrics;
  const impLogRegTestMetrics = computeEvaluationMetrics(
    yTestBool,
    XImpTest.map(predictImpLogRegCalProb),
    bestLogReg.threshold
  );

  // ============================================================================
  // MODEL 2: Calibrated Balanced Random Forest Ensemble (tuned strictly on GeoCV + Val)
  // ============================================================================
  let bestRf = {
    nTrees: 45,
    maxDepth: 3,
    minLeafSamples: 4,
    threshold: 0.5,
    trees: [] as RandomForestTreeNode[],
    calibration: undefined as ProbabilityCalibrationConfig | undefined,
    geoCvMetrics: baselineGeoCvMetrics,
    valMetrics: baselineValMetrics,
    score: -Infinity,
  };

  const rfSampleWeights = computeTrainingSampleWeights(
    yTrain,
    trainClusterIds,
    1.0,
    1.0,
    true
  );

  for (const nTrees of [35, 55]) {
    for (const maxDepth of [2, 3]) {
      for (const minLeaf of [3, 5]) {
        const oofLogits = new Array<number>(trainRecords.length).fill(0);
        for (let fi = 0; fi < cvFolds.length; fi++) {
          const fold = cvFolds[fi];
          if (fold.trIdx.length === 0 || fold.vaIdx.length === 0) continue;
          const fSW = computeTrainingSampleWeights(
            fold.fYTr,
            fold.fClust,
            1.0,
            1.0,
            true
          );
          const fRf = fitBalancedRandomForest({
            XTrain: fold.fXTr,
            yTrain: fold.fYTr,
            sampleWeights: fSW,
            nTrees,
            maxDepth,
            minLeafSamples: minLeaf,
            seed: 20260701 + fi * 97,
          });
          for (let k = 0; k < fold.vaIdx.length; k++) {
            const vi = fold.vaIdx[k];
            const xv = fold.fXVa[k];
            let pSum = 0;
            for (const tree of fRf.trees) pSum += predictRandomForestTree(tree, xv);
            oofLogits[vi] = logitFromProb(pSum / Math.max(1, fRf.trees.length));
          }
        }

        const fullRf = fitBalancedRandomForest({
          XTrain: XImpTrain,
          yTrain,
          sampleWeights: rfSampleWeights,
          nTrees,
          maxDepth,
          minLeafSamples: minLeaf,
          seed: 20260701,
        });
        const valLogits = XImpVal.map((x) => {
          let pSum = 0;
          for (const tree of fullRf.trees) pSum += predictRandomForestTree(tree, x);
          return logitFromProb(pSum / Math.max(1, fullRf.trees.length));
        });

        const calRes = selectAndFitNonTestCalibration({
          oofTrainLogits: oofLogits,
          yTrain,
          trainFoldAssignment,
          valLogits,
          yVal,
          plattPriorReg: 0.75,
        });

        for (const tau of [0.46, 0.5, 0.54]) {
          const gcm = computeEvaluationMetrics(
            yTrainBool,
            calRes.calibratedOofTrainProbs,
            tau
          );
          const vm = computeEvaluationMetrics(
            yValBool,
            calRes.calibratedValProbs,
            tau
          );
          const sc = nonTestSelectionScore(gcm, vm);
          if (
            sc > bestRf.score + 1e-4 ||
            (Math.abs(sc - bestRf.score) <= 1e-4 &&
              Math.abs(tau - 0.5) < Math.abs(bestRf.threshold - 0.5))
          ) {
            bestRf = {
              nTrees,
              maxDepth,
              minLeafSamples: minLeaf,
              threshold: tau,
              trees: fullRf.trees,
              calibration: calRes.calibration,
              geoCvMetrics: gcm,
              valMetrics: vm,
              score: sc,
            };
          }
        }
      }
    }
  }

  const predictRfRawLogit = (x: number[]) => {
    let pSum = 0;
    for (const tree of bestRf.trees) pSum += predictRandomForestTree(tree, x);
    return logitFromProb(pSum / Math.max(1, bestRf.trees.length));
  };
  const predictRfCalProb = (x: number[]) =>
    applyProbabilityCalibration(predictRfRawLogit(x), bestRf.calibration);

  const rfTrainMetrics = computeEvaluationMetrics(
    yTrainBool,
    XImpTrain.map(predictRfCalProb),
    bestRf.threshold
  );
  const rfGeoCvMetrics = bestRf.geoCvMetrics;
  const rfValMetrics = bestRf.valMetrics;
  const rfTestMetrics = computeEvaluationMetrics(
    yTestBool,
    XImpTest.map(predictRfCalProb),
    bestRf.threshold
  );

  // ============================================================================
  // MODEL 3: Calibrated Class-Weighted GBDT Ensemble (Depth-2 Trees, tuned on GeoCV + Val)
  // ============================================================================
  let bestGbdt = {
    nTrees: 16,
    learningRate: 0.1,
    minLeafSamples: 5,
    l2LeafReg: 4.0,
    threshold: 0.5,
    ensemble: { initLogit: 0, learningRate: 0.1, trees: [] as Depth2TreeNode[] },
    calibration: undefined as ProbabilityCalibrationConfig | undefined,
    geoCvMetrics: baselineGeoCvMetrics,
    valMetrics: baselineValMetrics,
    score: -Infinity,
  };

  const gbdtFullSW = computeTrainingSampleWeights(
    yTrain,
    trainClusterIds,
    posWeight,
    negWeight,
    true
  );

  for (const nTrees of [10, 16, 24]) {
    for (const lr of [0.08, 0.12]) {
      for (const minLeaf of [5, 7]) {
        for (const l2Reg of [3.0, 6.0]) {
          const oofLogits = new Array<number>(trainRecords.length).fill(0);
          for (const fold of cvFolds) {
            if (fold.trIdx.length === 0 || fold.vaIdx.length === 0) continue;
            const fSW = computeTrainingSampleWeights(
              fold.fYTr,
              fold.fClust,
              fold.fPosW,
              fold.fNegW,
              true
            );
            const fEns = fitGradientBoostedDepth2Ensemble({
              XTrain: fold.fXTr,
              yTrain: fold.fYTr,
              sampleWeights: fSW,
              nTrees,
              learningRate: lr,
              minLeafSamples: minLeaf,
              l2LeafReg: l2Reg,
            });
            for (let k = 0; k < fold.vaIdx.length; k++) {
              const vi = fold.vaIdx[k];
              const xv = fold.fXVa[k];
              let z = fEns.initLogit;
              for (const tree of fEns.trees) z += fEns.learningRate * predictDepth2Tree(tree, xv);
              oofLogits[vi] = z;
            }
          }

          const ens = fitGradientBoostedDepth2Ensemble({
            XTrain: XImpTrain,
            yTrain,
            sampleWeights: gbdtFullSW,
            nTrees,
            learningRate: lr,
            minLeafSamples: minLeaf,
            l2LeafReg: l2Reg,
          });
          const valLogits = XImpVal.map((x) => {
            let z = ens.initLogit;
            for (const tree of ens.trees) z += ens.learningRate * predictDepth2Tree(tree, x);
            return z;
          });

          const calRes = selectAndFitNonTestCalibration({
            oofTrainLogits: oofLogits,
            yTrain,
            trainFoldAssignment,
            valLogits,
            yVal,
            plattPriorReg: 0.75,
          });

          for (const tau of [0.46, 0.5, 0.54]) {
            const gcm = computeEvaluationMetrics(
              yTrainBool,
              calRes.calibratedOofTrainProbs,
              tau
            );
            const vm = computeEvaluationMetrics(
              yValBool,
              calRes.calibratedValProbs,
              tau
            );
            const sc = nonTestSelectionScore(gcm, vm);
            if (
              sc > bestGbdt.score + 1e-4 ||
              (Math.abs(sc - bestGbdt.score) <= 1e-4 &&
                Math.abs(tau - 0.5) < Math.abs(bestGbdt.threshold - 0.5))
            ) {
              bestGbdt = {
                nTrees,
                learningRate: lr,
                minLeafSamples: minLeaf,
                l2LeafReg: l2Reg,
                threshold: tau,
                ensemble: ens,
                calibration: calRes.calibration,
                geoCvMetrics: gcm,
                valMetrics: vm,
                score: sc,
              };
            }
          }
        }
      }
    }
  }

  const predictGbdtRawLogit = (x: number[]) => {
    let z = bestGbdt.ensemble.initLogit;
    for (const tree of bestGbdt.ensemble.trees) {
      z += bestGbdt.ensemble.learningRate * predictDepth2Tree(tree, x);
    }
    return z;
  };
  const predictGbdtCalProb = (x: number[]) =>
    applyProbabilityCalibration(predictGbdtRawLogit(x), bestGbdt.calibration);

  const gbdtTrainMetrics = computeEvaluationMetrics(
    yTrainBool,
    XImpTrain.map(predictGbdtCalProb),
    bestGbdt.threshold
  );
  const gbdtGeoCvMetrics = bestGbdt.geoCvMetrics;
  const gbdtValMetrics = bestGbdt.valMetrics;
  const gbdtTestMetrics = computeEvaluationMetrics(
    yTestBool,
    XImpTest.map(predictGbdtCalProb),
    bestGbdt.threshold
  );

  // ============================================================================
  // Leak-Free Non-Test Architecture Selection (5-Fold Geo-CV + Out-of-Sample Val)
  // ============================================================================
  const baselineCombinedScore = nonTestSelectionScore(
    baselineGeoCvMetrics,
    baselineValMetrics
  );

  let selectedId: 'baseline' | 'weighted_logreg' | 'random_forest' | 'gbdt' =
    'baseline';
  let bestNonTestScore = baselineCombinedScore;
  let bestGeoCvF1 = baselineGeoCvMetrics.f1Score;
  let bestValF1 = baselineValMetrics.f1Score;

  if (bestLogReg.score > bestNonTestScore + 0.002) {
    selectedId = 'weighted_logreg';
    bestNonTestScore = bestLogReg.score;
    bestGeoCvF1 = impLogRegGeoCvMetrics.f1Score;
    bestValF1 = impLogRegValMetrics.f1Score;
  }

  if (
    (bestRf.score > bestNonTestScore + 0.015 &&
      rfGeoCvF1Check(rfGeoCvMetrics, bestGeoCvF1) &&
      rfValMetrics.f1Score >= bestValF1) ||
    (selectedId === 'baseline' && bestRf.score > bestNonTestScore + 0.005)
  ) {
    selectedId = 'random_forest';
    bestNonTestScore = bestRf.score;
    bestGeoCvF1 = rfGeoCvMetrics.f1Score;
    bestValF1 = rfValMetrics.f1Score;
  }

  if (
    (bestGbdt.score > bestNonTestScore + 0.015 &&
      rfGeoCvF1Check(gbdtGeoCvMetrics, bestGeoCvF1) &&
      gbdtValMetrics.f1Score >= bestValF1) ||
    (selectedId === 'baseline' && bestGbdt.score > bestNonTestScore + 0.005)
  ) {
    selectedId = 'gbdt';
    bestNonTestScore = bestGbdt.score;
  }

  const baselineLabel = hasFull8Baseline
    ? 'Existing Baseline v3.0 (Uncalibrated L2 Logistic, thresh=28, λ=2.5)'
    : 'Baseline v1.0 (5-Feature Unweighted L2 Logistic)';

  const candidateModels: CandidateModelComparisonEntry[] = [
    {
      modelId: 'existing_baseline_logreg',
      modelLabel: baselineLabel,
      description: hasFull8Baseline
        ? 'Previous deployed v3.0 classifier: raw uncalibrated L2 logistic probabilities (λ=2.5, τ=0.52), unnormalized ExG2/VARI2, pixel changeThreshold=28.'
        : 'Baseline 5-feature unweighted L2 logistic regression, λ=1.0, threshold=0.50.',
      selectedByValidation: selectedId === 'baseline',
      hyperparameters: {
        features: baseKeys.length,
        lambda: baseLambda,
        calibration: 'none',
        classWeighted: hasFull8Baseline,
        geoClusterDeDup: hasFull8Baseline,
        threshold: baseTau,
      },
      trainMetrics: baselineTrainMetrics,
      geoCvMetrics: baselineGeoCvMetrics,
      valMetrics: baselineValMetrics,
      testMetrics: baselineTestMetrics,
    },
    {
      modelId: 'improved_v4_calibrated_logreg',
      modelLabel:
        'Improved v4.0 (Platt-Calibrated Geo-Weighted L2 Logistic, Red-Gated ExG & Glare Filter)',
      description:
        'Red-absorption-gated Excess-Green (prevents desert sand & limestone false vegetation), 3D tropical dipole exemption, co-linear roof glare suppression, changeThreshold=32, and non-test Platt calibration.',
      selectedByValidation: selectedId === 'weighted_logreg',
      hyperparameters: {
        features: 8,
        lambda: bestLogReg.lambda,
        calibration: bestLogReg.calibration?.method ?? 'none',
        plattSlope: bestLogReg.calibration?.plattSlope ?? 1,
        plattIntercept: bestLogReg.calibration?.plattIntercept ?? 0,
        classWeighted: true,
        monotoneConstrained: bestLogReg.monotoneConstrained,
        geoClusterDeDup: bestLogReg.useGeoDeDup,
        threshold: bestLogReg.threshold,
      },
      trainMetrics: impLogRegTrainMetrics,
      geoCvMetrics: impLogRegGeoCvMetrics,
      valMetrics: impLogRegValMetrics,
      testMetrics: impLogRegTestMetrics,
    },
    {
      modelId: 'candidate_v4_calibrated_rf',
      modelLabel: 'Candidate v4.0 (Calibrated Balanced Random Forest Ensemble)',
      description:
        'Class-balanced bootstrap Random Forest with sqrt(d) feature subsampling, Laplace-smoothed depth-2/3 trees, and out-of-fold probability calibration on the improved v4.0 features.',
      selectedByValidation: selectedId === 'random_forest',
      hyperparameters: {
        features: 8,
        nTrees: bestRf.nTrees,
        maxDepth: bestRf.maxDepth,
        minLeafSamples: bestRf.minLeafSamples,
        calibration: bestRf.calibration?.method ?? 'none',
        threshold: bestRf.threshold,
      },
      trainMetrics: rfTrainMetrics,
      geoCvMetrics: rfGeoCvMetrics,
      valMetrics: rfValMetrics,
      testMetrics: rfTestMetrics,
    },
    {
      modelId: 'candidate_v4_calibrated_gbdt',
      modelLabel: 'Candidate v4.0 (Calibrated Class-Weighted GBDT Depth-2 Ensemble)',
      description:
        'Second-order Newton gradient boosted depth-2 decision trees with L2 leaf regularization, geographic cluster sample weights, and out-of-fold calibration on the improved v4.0 features.',
      selectedByValidation: selectedId === 'gbdt',
      hyperparameters: {
        features: 8,
        nTrees: bestGbdt.ensemble.trees.length,
        learningRate: bestGbdt.learningRate,
        minLeafSamples: bestGbdt.minLeafSamples,
        l2LeafReg: bestGbdt.l2LeafReg,
        calibration: bestGbdt.calibration?.method ?? 'none',
        threshold: bestGbdt.threshold,
      },
      trainMetrics: gbdtTrainMetrics,
      geoCvMetrics: gbdtGeoCvMetrics,
      valMetrics: gbdtValMetrics,
      testMetrics: gbdtTestMetrics,
    },
  ];

  const selectionRationale =
    selectedId === 'weighted_logreg'
      ? `Selected Improved v4.0 Calibrated Geo-Weighted L2 Logistic Regression strictly via non-test evaluation (5-Fold Geo-CV F1 ${(
          impLogRegGeoCvMetrics.f1Score * 100
        ).toFixed(1)}%, Geo-CV AUC ${impLogRegGeoCvMetrics.rocAuc.toFixed(
          3
        )}, Val F1 ${(impLogRegValMetrics.f1Score * 100).toFixed(
          1
        )}%, Val AUC ${impLogRegValMetrics.rocAuc.toFixed(
          3
        )}, Val Brier ${impLogRegValMetrics.brierScore.toFixed(
          4
        )}, Val LogLoss ${(impLogRegValMetrics.logLoss ?? 0).toFixed(
          4
        )}) over Existing Baseline v3.0 (Geo-CV AUC ${baselineGeoCvMetrics.rocAuc.toFixed(
          3
        )}, Val Brier ${baselineValMetrics.brierScore.toFixed(
          4
        )}), Balanced Random Forest (Geo-CV F1 ${(
          rfGeoCvMetrics.f1Score * 100
        ).toFixed(1)}%), and GBDT (Geo-CV F1 ${(
          gbdtGeoCvMetrics.f1Score * 100
        ).toFixed(1)}%). Held-Out Test split (n=12) remained completely untouched.`
      : selectedId === 'random_forest'
      ? `Selected Calibrated Balanced Random Forest Ensemble strictly via non-test evaluation (5-Fold Geo-CV F1 ${(
          rfGeoCvMetrics.f1Score * 100
        ).toFixed(1)}%, Val F1 ${(rfValMetrics.f1Score * 100).toFixed(
          1
        )}%). Held-Out Test split (n=12) remained completely untouched.`
      : selectedId === 'gbdt'
      ? `Selected Calibrated Class-Weighted GBDT Depth-2 Ensemble strictly via non-test evaluation (5-Fold Geo-CV F1 ${(
          gbdtGeoCvMetrics.f1Score * 100
        ).toFixed(1)}%, Val F1 ${(gbdtValMetrics.f1Score * 100).toFixed(
          1
        )}%). Held-Out Test split (n=12) remained completely untouched.`
      : `Retained Existing Baseline model because no candidate exceeded its combined 5-Fold Geographic Group CV + Validation score.`;

  const commonMetadata = {
    isCalibratedModel: trainRecords.length >= 16 && testRecords.length >= 6,
    trainedAt: new Date().toISOString(),
    datasetSource,
    evaluationSplitLabel: 'SpaceNet 7 Held-Out Geographic Test Split',
    positiveCriterion,
    negativeCriterion,
    leakagePrevention,
    minTrainTestDistanceKm: Number(minTrainTestDistanceKm.toFixed(1)),
    minSplitPairwiseDistanceKm: {
      trainVal: Number(minSplitPairwiseDistanceKm.trainVal.toFixed(1)),
      trainTest: Number(minSplitPairwiseDistanceKm.trainTest.toFixed(1)),
      valTest: Number(minSplitPairwiseDistanceKm.valTest.toFixed(1)),
    },
    geoClusterCount: totalGeoClusters,
    featureDefinitions: FEATURE_DEFINITIONS,
    pixelIoUMetrics,
    baselineComparison: {
      baselineModelLabel: baselineLabel,
      baselineTrainMetrics,
      baselineGeoCvMetrics,
      baselineValMetrics,
      baselineTestMetrics,
      candidateModels,
      selectionRationale,
    },
  };

  if (selectedId === 'baseline') {
    const annotated = recordsWithClusters.map((r) => {
      const p = predictBaseline(r);
      return {
        ...r,
        rawUncalibratedProbability: Number(p.toFixed(4)),
        predictedProbability: Number(p.toFixed(4)),
        predictedPositive: p >= baseTau,
      };
    });
    return {
      ...commonMetadata,
      modelType: hasFull8Baseline
        ? 'weighted_l2_logistic_regression'
        : 'l2_logistic_regression',
      modelLabel: baselineLabel,
      version: '3.0.0-baseline',
      l2RegularizationLambda: baseLambda,
      classWeights: hasFull8Baseline
        ? { positive: posWeight, negative: negWeight }
        : { positive: 1, negative: 1 },
      featureNames: baseKeys,
      scalerMean: baseScaler.mean.map((v) => Number(v.toFixed(6))),
      scalerStd: baseScaler.std.map((v) => Number(v.toFixed(6))),
      weights: baseFit.weights.map((v) => Number(v.toFixed(6))),
      intercept: Number(baseFit.intercept.toFixed(6)),
      decisionThreshold: baseTau,
      trainMetrics: baselineTrainMetrics,
      geoCvMetrics: baselineGeoCvMetrics,
      valMetrics: baselineValMetrics,
      testMetrics: baselineTestMetrics,
      records: annotated,
    };
  }

  if (selectedId === 'random_forest') {
    const annotated = recordsWithClusters.map((r) => {
      const x = applyScaler(extractImpVec(r), impScaler);
      const rawP = sigmoid(predictRfRawLogit(x));
      const calP = predictRfCalProb(x);
      return {
        ...r,
        rawUncalibratedProbability: Number(rawP.toFixed(4)),
        predictedProbability: Number(calP.toFixed(4)),
        predictedPositive: calP >= bestRf.threshold,
      };
    });
    return {
      ...commonMetadata,
      modelType: 'random_forest',
      modelLabel: 'Calibrated Balanced Random Forest Ensemble (v4.0)',
      version: '4.0.0',
      l2RegularizationLambda: bestLogReg.lambda,
      classWeights: { positive: posWeight, negative: negWeight },
      featureNames: FEATURE_KEYS,
      scalerMean: impScaler.mean.map((v) => Number(v.toFixed(6))),
      scalerStd: impScaler.std.map((v) => Number(v.toFixed(6))),
      weights: bestLogReg.weights.map((v) => Number(v.toFixed(6))),
      intercept: Number(bestLogReg.intercept.toFixed(6)),
      calibration: bestRf.calibration,
      rfModel: { trees: bestRf.trees },
      decisionThreshold: bestRf.threshold,
      trainMetrics: rfTrainMetrics,
      geoCvMetrics: rfGeoCvMetrics,
      valMetrics: rfValMetrics,
      testMetrics: rfTestMetrics,
      records: annotated,
    };
  }

  if (selectedId === 'gbdt') {
    const annotated = recordsWithClusters.map((r) => {
      const x = applyScaler(extractImpVec(r), impScaler);
      const rawP = sigmoid(predictGbdtRawLogit(x));
      const calP = predictGbdtCalProb(x);
      return {
        ...r,
        rawUncalibratedProbability: Number(rawP.toFixed(4)),
        predictedProbability: Number(calP.toFixed(4)),
        predictedPositive: calP >= bestGbdt.threshold,
      };
    });
    return {
      ...commonMetadata,
      modelType: 'gbdt_tree_ensemble',
      modelLabel: 'Calibrated Class-Weighted GBDT Depth-2 Ensemble (v4.0)',
      version: '4.0.0',
      l2RegularizationLambda: bestGbdt.l2LeafReg,
      classWeights: { positive: posWeight, negative: negWeight },
      featureNames: FEATURE_KEYS,
      scalerMean: impScaler.mean.map((v) => Number(v.toFixed(6))),
      scalerStd: impScaler.std.map((v) => Number(v.toFixed(6))),
      weights: bestLogReg.weights.map((v) => Number(v.toFixed(6))),
      intercept: Number(bestLogReg.intercept.toFixed(6)),
      calibration: bestGbdt.calibration,
      gbdtModel: bestGbdt.ensemble,
      decisionThreshold: bestGbdt.threshold,
      trainMetrics: gbdtTrainMetrics,
      geoCvMetrics: gbdtGeoCvMetrics,
      valMetrics: gbdtValMetrics,
      testMetrics: gbdtTestMetrics,
      records: annotated,
    };
  }

  // Selected: Improved v4.0 Platt-Calibrated Geo- & Class-Weighted L2 Logistic Regression
  const annotated = recordsWithClusters.map((r) => {
    const x = applyScaler(extractImpVec(r), impScaler);
    const rawP = sigmoid(predictImpLogRegLogit(x));
    const calP = predictImpLogRegCalProb(x);
    return {
      ...r,
      rawUncalibratedProbability: Number(rawP.toFixed(4)),
      predictedProbability: Number(calP.toFixed(4)),
      predictedPositive: calP >= bestLogReg.threshold,
    };
  });

  const calSuffix =
    bestLogReg.calibration?.method === 'platt_sigmoid'
      ? 'Platt-Calibrated '
      : bestLogReg.calibration?.method === 'isotonic_pav'
      ? 'Isotonic-Calibrated '
      : 'Calibrated ';

  return {
    ...commonMetadata,
    modelType: 'weighted_l2_logistic_regression',
    modelLabel: bestLogReg.useGeoDeDup
      ? `${calSuffix}Geo- & Class-Weighted L2 Logistic Regression (v4.0)`
      : `${calSuffix}Class-Weighted L2 Logistic Regression (v4.0)`,
    version: '4.0.0',
    l2RegularizationLambda: bestLogReg.lambda,
    classWeights: { positive: posWeight, negative: negWeight },
    featureNames: FEATURE_KEYS,
    scalerMean: impScaler.mean.map((v) => Number(v.toFixed(6))),
    scalerStd: impScaler.std.map((v) => Number(v.toFixed(6))),
    weights: bestLogReg.weights.map((v) => Number(v.toFixed(6))),
    intercept: Number(bestLogReg.intercept.toFixed(6)),
    calibration: bestLogReg.calibration,
    decisionThreshold: bestLogReg.threshold,
    trainMetrics: impLogRegTrainMetrics,
    geoCvMetrics: impLogRegGeoCvMetrics,
    valMetrics: impLogRegValMetrics,
    testMetrics: impLogRegTestMetrics,
    records: annotated,
  };
}

function rfGeoCvF1Check(candidateGeoCv: EvaluationMetrics, bestGeoCvF1: number): boolean {
  return candidateGeoCv.f1Score >= bestGeoCvF1 - 0.005;
}

function computeUncertaintyLevel(
  confidence: number
): 'high_certainty' | 'moderate_certainty' | 'borderline_uncertain' {
  if (confidence >= 0.8) return 'high_certainty';
  if (confidence >= 0.65) return 'moderate_certainty';
  return 'borderline_uncertain';
}

/**
 * Classifies extracted image-difference features using the calibrated model artifact
 * (Weighted/Tuned L2 Logistic Regression, Balanced Random Forest, or GBDT Ensemble)
 * when available, or transparently falls back to rule-based change screening.
 */
export function classifyExtractedFeatures(
  features: ExtractedFeatures,
  model?: ModelArtifact | null
): ClassificationOutput {
  const activeKeys = model?.featureNames?.length ? model.featureNames : FEATURE_KEYS;

  if (
    !model ||
    !model.isCalibratedModel ||
    model.scalerMean.length !== activeKeys.length
  ) {
    const score =
      (features.highAlbedoGainPct >= 0.45 ? 1 : 0) +
      (features.structuralChangePct >= 0.9 ? 1 : 0) +
      (features.meanClusterEdgeGain >= 1.5 ? 1 : 0) +
      (features.newEdgeEmergencePct >= 0.5 ? 1 : 0) +
      (features.roofShadowDipolePct >= 0.45 ? 1 : 0) -
      (features.seasonalVegShiftFraction >= 0.62 ? 1 : 0);
    const predictedPositive = score >= 2;

    return {
      mode: 'rule_based_screening',
      predictedPositive,
      predictedCategory: predictedPositive
        ? 'Likely Building-Related Change'
        : 'Little or No Building-Related Change',
      probabilityBuildingChange: null,
      rawUncalibratedProbability: null,
      confidenceScore: null,
      uncertaintyLevel: 'unscored',
      logitScore: null,
      features,
      featureContributions: FEATURE_KEYS.map((key) => ({
        key,
        label: FEATURE_DEFINITIONS[key].label,
        unit: FEATURE_DEFINITIONS[key].unit,
        rawValue: Number((features[key] ?? 0).toFixed(4)),
        zScore: 0,
        weight: 0,
        logitContribution: 0,
      })),
      explanation: predictedPositive
        ? 'Rule-based screening detected elevated newly emerged structural edges, roof-shadow dipoles, and bright non-vegetated surface gain consistent with candidate construction.'
        : 'Rule-based screening found limited newly emerged structural edges or roof-shadow dipoles; observed pixel variations appear predominantly radiometric or seasonal.',
      limitationsNote:
        'Rule-based change screening only (uncalibrated). Pixel-level differences do not independently confirm permitted building construction and do not predict future construction.',
    };
  }

  const zVec = activeKeys.map((key, j) => {
    const raw = features[key] ?? 0;
    return (raw - model.scalerMean[j]) / (model.scalerStd[j] || 1);
  });

  if (model.modelType === 'random_forest' && model.rfModel) {
    let pSum = 0;
    for (const tree of model.rfModel.trees) {
      pSum += predictRandomForestTree(tree, zVec);
    }
    const rawProb = Math.min(
      0.999,
      Math.max(0.001, pSum / Math.max(1, model.rfModel.trees.length))
    );
    const rawLogit = logitFromProb(rawProb);
    const prob = applyProbabilityCalibration(rawLogit, model.calibration);
    const predictedPositive = prob >= model.decisionThreshold;
    const confidence = predictedPositive ? prob : 1 - prob;
    const uncertaintyLevel = computeUncertaintyLevel(confidence);

    const contributions: FeatureContribution[] = activeKeys.map((key, j) => {
      const raw = features[key] ?? 0;
      const zZero = [...zVec];
      zZero[j] = 0;
      let baseSum = 0;
      for (const tree of model.rfModel!.trees) {
        baseSum += predictRandomForestTree(tree, zZero);
      }
      const baseProb = Math.min(
        0.999,
        Math.max(0.001, baseSum / Math.max(1, model.rfModel!.trees.length))
      );
      const baseLogit = logitFromProb(baseProb);
      const contrib = rawLogit - baseLogit;
      const def = model.featureDefinitions[key] ?? FEATURE_DEFINITIONS[key];
      return {
        key,
        label: def.label,
        unit: def.unit,
        rawValue: Number(raw.toFixed(4)),
        zScore: Number(zVec[j].toFixed(3)),
        weight: Number((model.weights[j] ?? 0).toFixed(4)),
        logitContribution: Number(contrib.toFixed(4)),
      };
    });

    const topDrivers = [...contributions]
      .sort((a, b) => Math.abs(b.logitContribution) - Math.abs(a.logitContribution))
      .slice(0, 2)
      .map((c) => `${c.label.toLowerCase()} (${c.rawValue} ${c.unit})`)
      .join(' and ');

    const borderlineSuffix =
      uncertaintyLevel === 'borderline_uncertain'
        ? ' Because this calibrated probability lies near the decision boundary, treat the tile as a borderline case and inspect the highlighted overlay regions directly.'
        : '';

    const explanation =
      (predictedPositive
        ? `Calibrated Balanced Random Forest estimates a ${(prob * 100).toFixed(1)}% probability of building-related change (threshold ${(model.decisionThreshold * 100).toFixed(0)}%), driven primarily by ${topDrivers}.`
        : `Calibrated Balanced Random Forest estimates a ${((1 - prob) * 100).toFixed(1)}% probability of little or no building-related change (${(prob * 100).toFixed(1)}% construction probability), shaped by ${topDrivers}.`) +
      borderlineSuffix;

    return {
      mode: 'calibrated_random_forest',
      predictedPositive,
      predictedCategory: predictedPositive
        ? 'Likely Building-Related Change'
        : 'Little or No Building-Related Change',
      probabilityBuildingChange: Number(prob.toFixed(4)),
      rawUncalibratedProbability: Number(rawProb.toFixed(4)),
      confidenceScore: Number(confidence.toFixed(4)),
      uncertaintyLevel,
      logitScore: Number(logitFromProb(prob).toFixed(4)),
      features,
      featureContributions: contributions,
      explanation,
      limitationsNote:
        'Detects historical image changes between the two observation dates (~4.77 m/px Planet Dove resolution). Does not confirm ground-verified building permits and does not predict future construction.',
    };
  }

  if (model.modelType === 'gbdt_tree_ensemble' && model.gbdtModel) {
    let rawLogit = model.gbdtModel.initLogit;
    for (const tree of model.gbdtModel.trees) {
      rawLogit += model.gbdtModel.learningRate * predictDepth2Tree(tree, zVec);
    }
    const rawProb = sigmoid(rawLogit);
    const prob = applyProbabilityCalibration(rawLogit, model.calibration);
    const predictedPositive = prob >= model.decisionThreshold;
    const confidence = predictedPositive ? prob : 1 - prob;
    const uncertaintyLevel = computeUncertaintyLevel(confidence);

    const contributions: FeatureContribution[] = activeKeys.map((key, j) => {
      const raw = features[key] ?? 0;
      const zZero = [...zVec];
      zZero[j] = 0;
      let baselineLogit = model.gbdtModel!.initLogit;
      for (const tree of model.gbdtModel!.trees) {
        baselineLogit +=
          model.gbdtModel!.learningRate * predictDepth2Tree(tree, zZero);
      }
      const contrib = rawLogit - baselineLogit;
      const def = model.featureDefinitions[key] ?? FEATURE_DEFINITIONS[key];
      return {
        key,
        label: def.label,
        unit: def.unit,
        rawValue: Number(raw.toFixed(4)),
        zScore: Number(zVec[j].toFixed(3)),
        weight: Number((model.weights[j] ?? 0).toFixed(4)),
        logitContribution: Number(contrib.toFixed(4)),
      };
    });

    const topDrivers = [...contributions]
      .sort((a, b) => Math.abs(b.logitContribution) - Math.abs(a.logitContribution))
      .slice(0, 2)
      .map((c) => `${c.label.toLowerCase()} (${c.rawValue} ${c.unit})`)
      .join(' and ');

    const borderlineSuffix =
      uncertaintyLevel === 'borderline_uncertain'
        ? ' Because this calibrated probability lies near the decision boundary, treat the tile as a borderline case and inspect the highlighted overlay regions directly.'
        : '';

    const explanation =
      (predictedPositive
        ? `Calibrated GBDT ensemble estimates a ${(prob * 100).toFixed(1)}% probability of building-related change (threshold ${(model.decisionThreshold * 100).toFixed(0)}%), driven primarily by ${topDrivers}.`
        : `Calibrated GBDT ensemble estimates a ${((1 - prob) * 100).toFixed(1)}% probability of little or no building-related change (${(prob * 100).toFixed(1)}% construction probability), shaped by ${topDrivers}.`) +
      borderlineSuffix;

    return {
      mode: 'calibrated_gbdt_ensemble',
      predictedPositive,
      predictedCategory: predictedPositive
        ? 'Likely Building-Related Change'
        : 'Little or No Building-Related Change',
      probabilityBuildingChange: Number(prob.toFixed(4)),
      rawUncalibratedProbability: Number(rawProb.toFixed(4)),
      confidenceScore: Number(confidence.toFixed(4)),
      uncertaintyLevel,
      logitScore: Number(logitFromProb(prob).toFixed(4)),
      features,
      featureContributions: contributions,
      explanation,
      limitationsNote:
        'Detects historical image changes between the two observation dates (~4.77 m/px Planet Dove resolution). Does not confirm ground-verified building permits and does not predict future construction.',
    };
  }

  let rawLogit = model.intercept;
  const contributions: FeatureContribution[] = activeKeys.map((key, j) => {
    const raw = features[key] ?? 0;
    const zScore = zVec[j];
    const weight = model.weights[j] ?? 0;
    const contrib = zScore * weight;
    rawLogit += contrib;
    const def = model.featureDefinitions[key] ?? FEATURE_DEFINITIONS[key];
    return {
      key,
      label: def.label,
      unit: def.unit,
      rawValue: Number(raw.toFixed(4)),
      zScore: Number(zScore.toFixed(3)),
      weight: Number(weight.toFixed(4)),
      logitContribution: Number(contrib.toFixed(4)),
    };
  });

  const rawProb = sigmoid(rawLogit);
  const prob = applyProbabilityCalibration(rawLogit, model.calibration);
  const calibratedLogit =
    model.calibration?.method === 'platt_sigmoid'
      ? model.calibration.plattSlope * rawLogit +
        model.calibration.plattIntercept
      : logitFromProb(prob);
  const predictedPositive = prob >= model.decisionThreshold;
  const confidence = predictedPositive ? prob : 1 - prob;
  const uncertaintyLevel = computeUncertaintyLevel(confidence);

  const sortedPositiveDrivers = [...contributions]
    .sort((a, b) => Math.abs(b.logitContribution) - Math.abs(a.logitContribution))
    .slice(0, 2)
    .map((c) => `${c.label.toLowerCase()} (${c.rawValue} ${c.unit})`)
    .join(' and ');

  const calPrefix =
    model.calibration?.method === 'platt_sigmoid'
      ? 'Platt-calibrated'
      : model.calibration?.method === 'isotonic_pav'
      ? 'Isotonic-calibrated'
      : 'Calibrated';

  const modelTitle =
    model.modelType === 'weighted_l2_logistic_regression'
      ? `${calPrefix} geo- & class-weighted L2 logistic regression`
      : `${calPrefix} L2 logistic regression`;

  const borderlineSuffix =
    uncertaintyLevel === 'borderline_uncertain'
      ? ' Because this calibrated probability lies near the decision boundary, treat the tile as a borderline case and inspect the highlighted overlay regions directly.'
      : '';

  const explanation =
    (predictedPositive
      ? `${modelTitle} estimates a ${(prob * 100).toFixed(1)}% probability of building-related change (decision threshold ${(model.decisionThreshold * 100).toFixed(0)}%), driven primarily by ${sortedPositiveDrivers}.`
      : `${modelTitle} estimates a ${((1 - prob) * 100).toFixed(1)}% probability of little or no building-related change (${(prob * 100).toFixed(1)}% construction probability), shaped by ${sortedPositiveDrivers}.`) +
    borderlineSuffix;

  return {
    mode: 'calibrated_logistic_regression',
    predictedPositive,
    predictedCategory: predictedPositive
      ? 'Likely Building-Related Change'
      : 'Little or No Building-Related Change',
    probabilityBuildingChange: Number(prob.toFixed(4)),
    rawUncalibratedProbability: Number(rawProb.toFixed(4)),
    confidenceScore: Number(confidence.toFixed(4)),
    uncertaintyLevel,
    logitScore: Number(calibratedLogit.toFixed(4)),
    features,
    featureContributions: contributions,
    explanation,
    limitationsNote:
      'Detects historical image changes between the two observation dates (~4.77 m/px Planet Dove resolution). Does not confirm ground-verified building permits and does not predict future construction.',
  };
}
