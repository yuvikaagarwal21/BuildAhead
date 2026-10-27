export interface RawImageBuffer {
  width: number;
  height: number;
  /** RGBA byte array of length width * height * 4 */
  data: Uint8ClampedArray;
}

export interface SpatialMetadata {
  crs: string;
  /** [minX, minY, maxX, maxY] in EPSG:3857 meters */
  bboxMercator: [number, number, number, number];
  centerWgs84: {
    lat: number;
    lon: number;
  };
  gsdMetersPerPx: number;
  tileAreaKm2: number;
  cropWindow: {
    x0: number;
    y0: number;
    width: number;
    height: number;
    sourceWidth: number;
    sourceHeight: number;
  };
}

export interface BuildingFootprintPolygon {
  id: number;
  status: 'persistent' | 'new_construction' | 'removed';
  areaPx2: number;
  areaM2: number;
  /** Closed polygon ring in local sub-tile pixel coordinates [x, y] */
  ring: [number, number][];
}

export interface GroundTruthSummary {
  available: boolean;
  earlierCount: number;
  laterCount: number;
  newConstructionCount: number;
  removedCount: number;
  persistentCount: number;
  earlierCoveragePct: number;
  laterCoveragePct: number;
  netCoverageDeltaPct: number;
  newConstructionAreaM2: number;
  groundTruthPositive: boolean;
  groundTruthClassLabel:
    | 'Building-Related Change Present'
    | 'Little or No Building-Related Change';
  footprintsUrl?: string;
}

export interface SampleManifestEntry {
  id: string;
  aoiId: string;
  title: string;
  regionLabel: string;
  split: 'test' | 'val' | 'train';
  earlierDate: string;
  laterDate: string;
  intervalMonths: number;
  earlierImageUrl: string;
  laterImageUrl: string;
  earlierSourceS3Key: string;
  laterSourceS3Key: string;
  earlierLabelS3Key: string;
  laterLabelS3Key: string;
  sourceDataset: string;
  sourceUrl: string;
  sourceAttribution: string;
  license: string;
  isSyntheticFallback: boolean;
  spatial: SpatialMetadata;
  groundTruth: GroundTruthSummary;
  knownLimitations: string[];
}

export interface DatasetManifest {
  datasetName: string;
  officialSourceUrl: string;
  awsBucketUri: string;
  license: string;
  attribution: string;
  generatedAt: string;
  isGenuineSpaceNet7: boolean;
  methodologyNote: string;
  samples: SampleManifestEntry[];
}

export interface PreprocessingConfig {
  enableAlignment: boolean;
  denoiseMethod: 'none' | 'median3x3' | 'bilateral';
  contrastMethod: 'none' | 'linear_percentile' | 'clahe';
  radiometricMatch: boolean;
  suppressVegetation: boolean;
  suppressShadows: boolean;
  changeThreshold: number;
  minClusterAreaPx: number;
}

export interface ValidationResult {
  valid: boolean;
  errors: string[];
  warnings: string[];
  dimensionsMatch: boolean;
  spatialBoundsMatch: boolean;
  validClearPixelRatio: number;
  validPixelCount: number;
  totalPixelCount: number;
}

export interface AlignmentResult {
  applied: boolean;
  dx: number;
  dy: number;
  nccScoreBefore: number;
  nccScoreAfter: number;
}

export interface CandidateRegion {
  id: number;
  areaPx: number;
  areaM2: number;
  bbox: [number, number, number, number];
  centroid: [number, number];
  meanSpectralDelta: number;
  meanEdgeGain: number;
  albedoGain: number;
  localStructureCorrelation?: number;
}

export interface BaselineFeatures {
  structuralChangePct: number;
  highAlbedoGainPct: number;
  meanClusterEdgeGain: number;
  compactClusterDensityKm2: number;
  seasonalVegShiftFraction: number;
  newEdgeEmergencePct?: number;
  roofShadowDipolePct?: number;
  patchActivityConcentration?: number;
}

export interface ExtractedFeatures extends BaselineFeatures {
  /** % of valid pixels where new non-vegetated edges emerged on previously low-gradient or orientation-decorrelated ground */
  newEdgeEmergencePct: number;
  /** % of valid pixels exhibiting newly formed 5x5 local roof-to-shadow luminance dipoles */
  roofShadowDipolePct: number;
  /** % of 64x64 spatial grid patches exhibiting co-occurring new structural edges and dipole contrast */
  patchActivityConcentration: number;
}

export type FeatureKey = keyof ExtractedFeatures;

export interface ReliabilityBin {
  binIndex: number;
  binStart: number;
  binEnd: number;
  count: number;
  meanPredictedProb: number;
  empiricalPositiveRate: number;
}

export interface CalibrationMethodBenchmark {
  method: 'none' | 'platt_sigmoid' | 'isotonic_pav';
  label: string;
  selected: boolean;
  nonTestAccuracy: number;
  nonTestBrierScore: number;
  nonTestLogLoss: number;
  nonTestEce: number;
  notes: string;
}

export interface ProbabilityCalibrationConfig {
  method: 'none' | 'platt_sigmoid' | 'isotonic_pav';
  plattSlope: number;
  plattIntercept: number;
  isotonicKnotsX?: number[];
  isotonicKnotsY?: number[];
  fittedSampleCount: number;
  fittedOn: string;
  nonTestReliabilityBefore: ReliabilityBin[];
  nonTestReliabilityAfter: ReliabilityBin[];
  benchmarkOnNonTest: CalibrationMethodBenchmark[];
}

export interface EvaluationMetrics {
  sampleCount: number;
  positiveCount: number;
  negativeCount: number;
  /** Ratio 0..1; display percentage is ((tp + tn) / (tp + tn + fp + fn)) * 100 */
  accuracy: number;
  /** Wilson score 95% binomial confidence interval [lowRatio, highRatio] in 0..1 */
  accuracyCi95?: [number, number];
  precision: number;
  recall: number;
  f1Score: number;
  brierScore: number;
  logLoss?: number;
  ece?: number;
  reliabilityBins?: ReliabilityBin[];
  rocAuc: number;
  confusionMatrix: {
    tp: number;
    fp: number;
    tn: number;
    fn: number;
  };
}

export interface PixelChangeIoUMetrics {
  evaluatedSampleCount: number;
  changedSampleCount: number;
  /** Mean pixel-level IoU across positive (building-change) test samples */
  meanChangedSampleIoU: number;
  /** Aggregate micro-IoU = total intersection pixels / total union pixels across evaluated samples */
  microPixelIoU: number;
}

export interface Depth2TreeNode {
  featureIndex: number;
  threshold: number;
  leftFeatureIndex: number;
  leftThreshold: number;
  leftLeftValue: number;
  leftRightValue: number;
  rightFeatureIndex: number;
  rightThreshold: number;
  rightLeftValue: number;
  rightRightValue: number;
}

export interface RandomForestTreeNode {
  isLeaf: boolean;
  probPositive: number;
  featureIndex?: number;
  threshold?: number;
  left?: RandomForestTreeNode;
  right?: RandomForestTreeNode;
}

export interface CandidateModelComparisonEntry {
  modelId: string;
  modelLabel: string;
  description: string;
  selectedByValidation: boolean;
  hyperparameters: Record<string, string | number | boolean>;
  trainMetrics: EvaluationMetrics;
  geoCvMetrics?: EvaluationMetrics;
  valMetrics: EvaluationMetrics;
  testMetrics: EvaluationMetrics;
}

export interface DatasetSplitRecord {
  pairId: string;
  aoiId: string;
  geoClusterId?: number;
  split: 'train' | 'val' | 'test';
  lat: number;
  lon: number;
  earlierDate: string;
  laterDate: string;
  newBuildings: number;
  newBuildingAreaM2: number;
  groundTruthPositive: boolean;
  features: ExtractedFeatures;
  baselineFeatures?: BaselineFeatures;
  pixelIoUBaseline?: number;
  pixelIoUImproved?: number;
  rawUncalibratedProbability?: number;
  predictedProbability?: number;
  predictedPositive?: boolean;
}

export interface ModelArtifact {
  modelType:
    | 'l2_logistic_regression'
    | 'weighted_l2_logistic_regression'
    | 'random_forest'
    | 'gbdt_tree_ensemble'
    | 'rule_based_screening';
  modelLabel?: string;
  isCalibratedModel: boolean;
  version: string;
  trainedAt: string;
  datasetSource: string;
  evaluationSplitLabel?: string;
  positiveCriterion: string;
  negativeCriterion: string;
  leakagePrevention: string;
  minTrainTestDistanceKm: number;
  minSplitPairwiseDistanceKm?: {
    trainVal: number;
    trainTest: number;
    valTest: number;
  };
  l2RegularizationLambda: number;
  classWeights?: {
    positive: number;
    negative: number;
  };
  featureNames: FeatureKey[];
  featureDefinitions: Record<
    FeatureKey,
    {
      label: string;
      unit: string;
      description: string;
    }
  >;
  scalerMean: number[];
  scalerStd: number[];
  weights: number[];
  intercept: number;
  calibration?: ProbabilityCalibrationConfig;
  gbdtModel?: {
    initLogit: number;
    learningRate: number;
    trees: Depth2TreeNode[];
  };
  rfModel?: {
    trees: RandomForestTreeNode[];
  };
  decisionThreshold: number;
  geoClusterCount?: number;
  trainMetrics: EvaluationMetrics;
  geoCvMetrics?: EvaluationMetrics;
  valMetrics?: EvaluationMetrics;
  testMetrics: EvaluationMetrics;
  pixelIoUMetrics?: {
    baselineTrainVal?: PixelChangeIoUMetrics;
    improvedTrainVal?: PixelChangeIoUMetrics;
    baselineTest: PixelChangeIoUMetrics;
    improvedTest: PixelChangeIoUMetrics;
  };
  baselineComparison?: {
    baselineModelLabel: string;
    baselineTrainMetrics: EvaluationMetrics;
    baselineGeoCvMetrics?: EvaluationMetrics;
    baselineValMetrics: EvaluationMetrics;
    baselineTestMetrics: EvaluationMetrics;
    candidateModels: CandidateModelComparisonEntry[];
    selectionRationale: string;
  };
  records: DatasetSplitRecord[];
}

export interface FeatureContribution {
  key: FeatureKey;
  label: string;
  unit: string;
  rawValue: number;
  zScore: number;
  weight: number;
  logitContribution: number;
}

export interface ClassificationOutput {
  mode:
    | 'calibrated_logistic_regression'
    | 'calibrated_random_forest'
    | 'calibrated_gbdt_ensemble'
    | 'rule_based_screening';
  predictedPositive: boolean;
  predictedCategory:
    | 'Likely Building-Related Change'
    | 'Little or No Building-Related Change';
  probabilityBuildingChange: number | null;
  rawUncalibratedProbability?: number | null;
  confidenceScore: number | null;
  uncertaintyLevel?: 'high_certainty' | 'moderate_certainty' | 'borderline_uncertain' | 'unscored';
  logitScore: number | null;
  features: ExtractedFeatures;
  featureContributions: FeatureContribution[];
  explanation: string;
  limitationsNote: string;
}

export interface ChangeDetectionResult {
  validation: ValidationResult;
  alignment: AlignmentResult;
  earlierCleaned: RawImageBuffer;
  laterCleaned: RawImageBuffer;
  /** Rendered RGBA overlay map highlighting structural candidate changes & suppressed seasonal shifts */
  changeMapOverlay: RawImageBuffer;
  /** Pure difference heatmap RGBA buffer */
  differenceHeatmap: RawImageBuffer;
  /** Binary mask (1 = candidate structural change pixel, 0 = unchanged/suppressed) */
  candidateMask: Uint8Array;
  /** Binary mask (1 = suppressed seasonal/vegetation or shadow change) */
  suppressedEnvMask: Uint8Array;
  candidatePixelCount: number;
  candidateChangedPct: number;
  candidateAreaM2: number;
  rawDiffPixelCount: number;
  rawDiffPct: number;
  suppressedEnvPixelCount: number;
  suppressedEnvPct: number;
  /** Optional pixel-level IoU against SpaceNet 7 ground-truth changed building polygons when available */
  pixelChangeIoU?: number | null;
  regions: CandidateRegion[];
  extractedFeatures: ExtractedFeatures;
  classification: ClassificationOutput;
  processingDurationMs: number;
}
