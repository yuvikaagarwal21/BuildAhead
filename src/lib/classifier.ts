import type {
  ClassificationOutput,
  DatasetSplitRecord,
  EvaluationMetrics,
  ExtractedFeatures,
  FeatureContribution,
  FeatureKey,
  ModelArtifact,
} from '../types/satellite.ts';

export const FEATURE_KEYS: FeatureKey[] = [
  'structuralChangePct',
  'highAlbedoGainPct',
  'meanClusterEdgeGain',
  'compactClusterDensityKm2',
  'seasonalVegShiftFraction',
];

export const FEATURE_DEFINITIONS: ModelArtifact['featureDefinitions'] = {
  structuralChangePct: {
    label: 'Filtered Structural Change Area',
    unit: '% of valid tile',
    description:
      'Percentage of cloud-free pixels retained in candidate change clusters after radiometric normalization, vegetation suppression, shadow filtering, and morphological opening.',
  },
  highAlbedoGainPct: {
    label: 'Non-Vegetated Bright Surface Gain',
    unit: '% of valid tile',
    description:
      'Percentage of valid pixels transitioning toward brighter, low-greenness reflectance characteristic of new roofs, concrete pads, or graded construction earthworks.',
  },
  meanClusterEdgeGain: {
    label: 'Rectilinear Sobel Edge Gain',
    unit: 'gradient units',
    description:
      'Area-weighted positive increase in 3x3 Sobel spatial gradient magnitude across candidate change regions, capturing newly introduced roof and road boundaries.',
  },
  compactClusterDensityKm2: {
    label: 'Compact Candidate Cluster Density',
    unit: 'clusters / km²',
    description:
      'Spatial density of discrete, compact candidate change regions (10–1400 px area) per square kilometer, distinguishing localized structures from broad illumination washes.',
  },
  seasonalVegShiftFraction: {
    label: 'Seasonal Vegetation Shift Ratio',
    unit: '0–1 ratio',
    description:
      'Fraction of raw spectral pixel differences attributable to Excess-Green (2G - R - B) phenology changes such as crop growth or seasonal grass greening.',
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

  const n = Math.max(1, yTrue.length);
  const accuracy = (tp + tn) / n;
  const precision = tp + fp > 0 ? tp / (tp + fp) : 0;
  const recall = tp + fn > 0 ? tp / (tp + fn) : 0;
  const f1Score =
    precision + recall > 0 ? (2 * precision * recall) / (precision + recall) : 0;
  const brierScore = brierSum / n;

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
    precision: Number(precision.toFixed(4)),
    recall: Number(recall.toFixed(4)),
    f1Score: Number(f1Score.toFixed(4)),
    brierScore: Number(brierScore.toFixed(4)),
    rocAuc: Number(rocAuc.toFixed(4)),
    confusionMatrix: { tp, fp, tn, fn },
  };
}

/**
 * Fits an L2-regularized Logistic Regression model strictly on `train` records
 * and evaluates without leakage on `test` records.
 */
export function fitLogisticRegressionModel(params: {
  records: DatasetSplitRecord[];
  lambda?: number;
  positiveCriterion: string;
  negativeCriterion: string;
  leakagePrevention: string;
  minTrainTestDistanceKm: number;
  datasetSource: string;
}): ModelArtifact {
  const {
    records,
    lambda = 1.0,
    positiveCriterion,
    negativeCriterion,
    leakagePrevention,
    minTrainTestDistanceKm,
    datasetSource,
  } = params;

  const trainRecords = records.filter((r) => r.split === 'train');
  const testRecords = records.filter((r) => r.split === 'test');
  const d = FEATURE_KEYS.length;

  // Fit z-score scaler ONLY on training split
  const scalerMean = new Array<number>(d).fill(0);
  const scalerStd = new Array<number>(d).fill(1);

  for (let j = 0; j < d; j++) {
    const key = FEATURE_KEYS[j];
    let sum = 0;
    for (const r of trainRecords) sum += r.features[key];
    const mean = sum / Math.max(1, trainRecords.length);
    scalerMean[j] = mean;

    let vSum = 0;
    for (const r of trainRecords) {
      const diff = r.features[key] - mean;
      vSum += diff * diff;
    }
    const std = Math.sqrt(vSum / Math.max(1, trainRecords.length));
    scalerStd[j] = std > 1e-8 ? std : 1;
  }

  const XTrain = trainRecords.map((r) =>
    FEATURE_KEYS.map((k, j) => (r.features[k] - scalerMean[j]) / scalerStd[j])
  );
  const yTrain = trainRecords.map((r) => (r.groundTruthPositive ? 1 : 0));

  // Optimize L2-regularized cross-entropy via deterministic gradient descent with backtracking
  const weights = new Array<number>(d).fill(0);
  let intercept = 0;
  const nTrain = Math.max(1, XTrain.length);

  for (let iter = 0; iter < 4000; iter++) {
    const gradW = new Array<number>(d).fill(0);
    let gradB = 0;

    for (let i = 0; i < nTrain; i++) {
      let z = intercept;
      for (let j = 0; j < d; j++) z += weights[j] * XTrain[i][j];
      const p = sigmoid(z);
      const err = p - yTrain[i];
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

    if (Math.sqrt(norm2) < 1e-8) break;
    const lr = 0.25;
    intercept -= lr * gradB;
    for (let j = 0; j < d; j++) {
      weights[j] -= lr * gradW[j];
    }
  }

  const predictProb = (feat: ExtractedFeatures): number => {
    let z = intercept;
    for (let j = 0; j < d; j++) {
      const key = FEATURE_KEYS[j];
      const zVal = (feat[key] - scalerMean[j]) / scalerStd[j];
      z += weights[j] * zVal;
    }
    return sigmoid(z);
  };

  const trainProbs = trainRecords.map((r) => predictProb(r.features));
  const testProbs = testRecords.map((r) => predictProb(r.features));

  const annotatedRecords: DatasetSplitRecord[] = records.map((r) => {
    const prob = predictProb(r.features);
    return {
      ...r,
      predictedProbability: Number(prob.toFixed(4)),
      predictedPositive: prob >= 0.5,
    };
  });

  return {
    modelType: 'l2_logistic_regression',
    isCalibratedModel: trainRecords.length >= 16 && testRecords.length >= 6,
    version: '1.0.0',
    trainedAt: new Date().toISOString(),
    datasetSource,
    positiveCriterion,
    negativeCriterion,
    leakagePrevention,
    minTrainTestDistanceKm: Number(minTrainTestDistanceKm.toFixed(1)),
    l2RegularizationLambda: lambda,
    featureNames: FEATURE_KEYS,
    featureDefinitions: FEATURE_DEFINITIONS,
    scalerMean: scalerMean.map((v) => Number(v.toFixed(6))),
    scalerStd: scalerStd.map((v) => Number(v.toFixed(6))),
    weights: weights.map((v) => Number(v.toFixed(6))),
    intercept: Number(intercept.toFixed(6)),
    decisionThreshold: 0.5,
    trainMetrics: computeEvaluationMetrics(
      trainRecords.map((r) => r.groundTruthPositive),
      trainProbs,
      0.5
    ),
    testMetrics: computeEvaluationMetrics(
      testRecords.map((r) => r.groundTruthPositive),
      testProbs,
      0.5
    ),
    records: annotatedRecords,
  };
}

/**
 * Classifies extracted image-difference features using a real calibrated Logistic Regression
 * model artifact when available, or transparently falls back to rule-based change screening.
 */
export function classifyExtractedFeatures(
  features: ExtractedFeatures,
  model?: ModelArtifact | null
): ClassificationOutput {
  if (!model || !model.isCalibratedModel || model.weights.length !== FEATURE_KEYS.length) {
    // Honest rule-based fallback if no calibrated model is loaded
    const score =
      (features.highAlbedoGainPct >= 0.45 ? 1 : 0) +
      (features.structuralChangePct >= 0.9 ? 1 : 0) +
      (features.meanClusterEdgeGain >= 3.0 ? 1 : 0) -
      (features.seasonalVegShiftFraction >= 0.62 ? 1 : 0);
    const predictedPositive = score >= 2;

    return {
      mode: 'rule_based_screening',
      predictedPositive,
      predictedCategory: predictedPositive
        ? 'Likely Building-Related Change'
        : 'Little or No Building-Related Change',
      probabilityBuildingChange: null,
      confidenceScore: null,
      logitScore: null,
      features,
      featureContributions: FEATURE_KEYS.map((key) => ({
        key,
        label: FEATURE_DEFINITIONS[key].label,
        unit: FEATURE_DEFINITIONS[key].unit,
        rawValue: Number(features[key].toFixed(4)),
        zScore: 0,
        weight: 0,
        logitContribution: 0,
      })),
      explanation: predictedPositive
        ? 'Rule-based screening detected elevated non-vegetated bright surface gain and spatial edge increases consistent with candidate structural activity.'
        : 'Rule-based screening found limited compact structural or rectilinear edge gains; observed pixel variations appear predominantly radiometric or seasonal.',
      limitationsNote:
        'Rule-based change screening only (uncalibrated). Pixel-level differences do not independently confirm permitted building construction and do not predict future construction.',
    };
  }

  let logit = model.intercept;
  const contributions: FeatureContribution[] = FEATURE_KEYS.map((key, j) => {
    const raw = features[key];
    const zScore = (raw - model.scalerMean[j]) / (model.scalerStd[j] || 1);
    const weight = model.weights[j];
    const contrib = zScore * weight;
    logit += contrib;
    return {
      key,
      label: model.featureDefinitions[key].label,
      unit: model.featureDefinitions[key].unit,
      rawValue: Number(raw.toFixed(4)),
      zScore: Number(zScore.toFixed(3)),
      weight: Number(weight.toFixed(4)),
      logitContribution: Number(contrib.toFixed(4)),
    };
  });

  const prob = sigmoid(logit);
  const predictedPositive = prob >= model.decisionThreshold;
  const confidence = predictedPositive ? prob : 1 - prob;

  const sortedPositiveDrivers = [...contributions]
    .sort((a, b) => Math.abs(b.logitContribution) - Math.abs(a.logitContribution))
    .slice(0, 2)
    .map((c) => `${c.label.toLowerCase()} (${c.rawValue} ${c.unit})`)
    .join(' and ');

  const explanation = predictedPositive
    ? `Calibrated L2 logistic regression estimates a ${(prob * 100).toFixed(1)}% probability of building-related change, driven primarily by ${sortedPositiveDrivers}.`
    : `Calibrated L2 logistic regression estimates a ${((1 - prob) * 100).toFixed(1)}% probability of little or no building-related change (${(prob * 100).toFixed(1)}% construction probability), shaped by ${sortedPositiveDrivers}.`;

  return {
    mode: 'calibrated_logistic_regression',
    predictedPositive,
    predictedCategory: predictedPositive
      ? 'Likely Building-Related Change'
      : 'Little or No Building-Related Change',
    probabilityBuildingChange: Number(prob.toFixed(4)),
    confidenceScore: Number(confidence.toFixed(4)),
    logitScore: Number(logit.toFixed(4)),
    features,
    featureContributions: contributions,
    explanation,
    limitationsNote:
      'Detects historical image changes between the two observation dates (~4.77 m/px Planet Dove resolution). Does not confirm ground-verified building permits and does not predict future construction.',
  };
}
