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
  split: 'test' | 'train';
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
}

export interface ExtractedFeatures {
  structuralChangePct: number;
  highAlbedoGainPct: number;
  meanClusterEdgeGain: number;
  compactClusterDensityKm2: number;
  seasonalVegShiftFraction: number;
}

export type FeatureKey = keyof ExtractedFeatures;

export interface EvaluationMetrics {
  sampleCount: number;
  positiveCount: number;
  negativeCount: number;
  accuracy: number;
  precision: number;
  recall: number;
  f1Score: number;
  brierScore: number;
  rocAuc: number;
  confusionMatrix: {
    tp: number;
    fp: number;
    tn: number;
    fn: number;
  };
}

export interface DatasetSplitRecord {
  pairId: string;
  aoiId: string;
  split: 'train' | 'test';
  lat: number;
  lon: number;
  earlierDate: string;
  laterDate: string;
  newBuildings: number;
  newBuildingAreaM2: number;
  groundTruthPositive: boolean;
  features: ExtractedFeatures;
  predictedProbability?: number;
  predictedPositive?: boolean;
}

export interface ModelArtifact {
  modelType: 'l2_logistic_regression' | 'rule_based_screening';
  isCalibratedModel: boolean;
  version: string;
  trainedAt: string;
  datasetSource: string;
  positiveCriterion: string;
  negativeCriterion: string;
  leakagePrevention: string;
  minTrainTestDistanceKm: number;
  l2RegularizationLambda: number;
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
  decisionThreshold: number;
  trainMetrics: EvaluationMetrics;
  testMetrics: EvaluationMetrics;
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
  mode: 'calibrated_logistic_regression' | 'rule_based_screening';
  predictedPositive: boolean;
  predictedCategory:
    | 'Likely Building-Related Change'
    | 'Little or No Building-Related Change';
  probabilityBuildingChange: number | null;
  confidenceScore: number | null;
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
  regions: CandidateRegion[];
  extractedFeatures: ExtractedFeatures;
  classification: ClassificationOutput;
  processingDurationMs: number;
}
