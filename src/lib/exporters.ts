import type {
  ChangeDetectionResult,
  ModelArtifact,
  PreprocessingConfig,
  RawImageBuffer,
  SampleManifestEntry,
} from '../types/satellite.ts';
import {
  calculateAccuracyPercent,
  formatModelAccuracyDisplay,
} from './classifier.ts';

function escapeCsvCell(value: string | number | boolean | null | undefined): string {
  if (value === null || value === undefined) return '';
  const str = String(value);
  if (str.includes(',') || str.includes('"') || str.includes('\n')) {
    return `"${str.replace(/"/g, '""')}"`;
  }
  return str;
}

/**
 * Pure builder for the CSV comparison summary so automated tests can verify exact parity
 * between displayed metrics and downloaded exports.
 */
export function buildComparisonCsvString(
  sample: SampleManifestEntry,
  config: PreprocessingConfig,
  result: ChangeDetectionResult,
  model?: ModelArtifact | null
): string {
  const gt = sample.groundTruth;
  const cls = result.classification;
  const testMetrics = model?.isCalibratedModel ? model.testMetrics : null;
  const accuracyPct = calculateAccuracyPercent(testMetrics);

  const rows: [string, string | number | boolean | null][] = [
    ['sample_id', sample.id],
    ['spacenet7_aoi_id', sample.aoiId],
    ['region_label', sample.regionLabel],
    ['dataset_split', sample.split],
    ['source_dataset', sample.sourceDataset],
    ['source_url', sample.sourceUrl],
    ['source_attribution', sample.sourceAttribution],
    ['license', sample.license],
    ['is_synthetic_fallback', sample.isSyntheticFallback],
    ['earlier_observation_date', sample.earlierDate],
    ['later_observation_date', sample.laterDate],
    ['observation_interval_months', sample.intervalMonths],
    ['center_latitude_wgs84', sample.spatial.centerWgs84.lat],
    ['center_longitude_wgs84', sample.spatial.centerWgs84.lon],
    ['crs', sample.spatial.crs],
    ['ground_sample_distance_m_per_px', sample.spatial.gsdMetersPerPx],
    ['tile_dimensions_px', `${result.earlierCleaned.width}x${result.earlierCleaned.height}`],
    ['valid_clear_pixel_ratio', Number(result.validation.validClearPixelRatio.toFixed(4))],
    ['alignment_enabled', config.enableAlignment],
    ['alignment_shift_dx_px', result.alignment.dx],
    ['alignment_shift_dy_px', result.alignment.dy],
    ['alignment_ncc_before', result.alignment.nccScoreBefore],
    ['alignment_ncc_after', result.alignment.nccScoreAfter],
    ['denoise_method', config.denoiseMethod],
    ['contrast_enhancement', config.contrastMethod],
    ['radiometric_normalization', config.radiometricMatch],
    ['vegetation_suppression_enabled', config.suppressVegetation],
    ['shadow_suppression_enabled', config.suppressShadows],
    ['spectral_change_threshold', config.changeThreshold],
    ['min_cluster_area_px', config.minClusterAreaPx],
    ['candidate_changed_pixels', result.candidatePixelCount],
    ['candidate_changed_proportion_pct', result.candidateChangedPct],
    ['candidate_changed_area_m2', result.candidateAreaM2],
    ['candidate_region_count', result.regions.length],
    [
      'pixel_change_iou_vs_ground_truth',
      result.pixelChangeIoU !== null && result.pixelChangeIoU !== undefined
        ? result.pixelChangeIoU
        : 'N/A (no changed footprints)',
    ],
    ['raw_unfiltered_diff_pixels', result.rawDiffPixelCount],
    ['raw_unfiltered_diff_pct', result.rawDiffPct],
    ['suppressed_seasonal_shadow_pixels', result.suppressedEnvPixelCount],
    ['suppressed_seasonal_shadow_pct', result.suppressedEnvPct],
    ['feature_structural_change_pct', cls.features.structuralChangePct],
    ['feature_high_albedo_gain_pct', cls.features.highAlbedoGainPct],
    ['feature_mean_cluster_edge_gain', cls.features.meanClusterEdgeGain],
    ['feature_compact_cluster_density_km2', cls.features.compactClusterDensityKm2],
    ['feature_seasonal_veg_shift_fraction', cls.features.seasonalVegShiftFraction],
    ['feature_new_edge_emergence_pct', cls.features.newEdgeEmergencePct],
    ['feature_roof_shadow_dipole_pct', cls.features.roofShadowDipolePct],
    ['feature_patch_activity_concentration', cls.features.patchActivityConcentration],
    ['classification_mode', cls.mode],
    ['predicted_category', cls.predictedCategory],
    [
      'predicted_building_change_probability',
      cls.probabilityBuildingChange !== null ? cls.probabilityBuildingChange : 'N/A (rule screening)',
    ],
    [
      'raw_uncalibrated_probability',
      cls.rawUncalibratedProbability !== null && cls.rawUncalibratedProbability !== undefined
        ? cls.rawUncalibratedProbability
        : 'N/A',
    ],
    [
      'model_confidence_single_sample',
      cls.confidenceScore !== null ? cls.confidenceScore : 'N/A (rule screening)',
    ],
    [
      'confidence_uncertainty_level',
      cls.uncertaintyLevel ?? 'unscored',
    ],
    [
      'probability_calibration_method',
      model?.calibration?.method ?? 'none',
    ],
    [
      'model_accuracy_held_out_test_pct',
      accuracyPct !== null ? Number(accuracyPct.toFixed(2)) : 'Not evaluated.',
    ],
    [
      'model_accuracy_held_out_ci95_pct',
      testMetrics?.accuracyCi95
        ? `${(testMetrics.accuracyCi95[0] * 100).toFixed(1)}-${(
            testMetrics.accuracyCi95[1] * 100
          ).toFixed(1)}`
        : 'Not evaluated.',
    ],
    [
      'model_accuracy_evaluated_samples',
      testMetrics ? testMetrics.sampleCount : 0,
    ],
    [
      'model_accuracy_evaluation_split',
      model?.evaluationSplitLabel ?? (testMetrics ? 'Held-out test split' : 'Not evaluated.'),
    ],
    [
      'model_geo_cv_5fold_accuracy_pct',
      model?.geoCvMetrics
        ? Number((model.geoCvMetrics.accuracy * 100).toFixed(2))
        : 'Not evaluated.',
    ],
    [
      'model_test_brier_score',
      testMetrics ? testMetrics.brierScore : 'Not evaluated.',
    ],
    [
      'model_test_log_loss',
      testMetrics?.logLoss !== undefined ? testMetrics.logLoss : 'Not evaluated.',
    ],
    [
      'model_test_ece',
      testMetrics?.ece !== undefined ? testMetrics.ece : 'Not evaluated.',
    ],
    ['ground_truth_available', gt.available],
    ['ground_truth_earlier_buildings', gt.earlierCount],
    ['ground_truth_later_buildings', gt.laterCount],
    ['ground_truth_new_buildings_added', gt.newConstructionCount],
    ['ground_truth_removed_buildings', gt.removedCount],
    ['ground_truth_earlier_coverage_pct', gt.earlierCoveragePct],
    ['ground_truth_later_coverage_pct', gt.laterCoveragePct],
    ['ground_truth_net_coverage_delta_pct', gt.netCoverageDeltaPct],
    ['ground_truth_new_building_area_m2', gt.newConstructionAreaM2],
    ['ground_truth_reference_category', gt.groundTruthClassLabel],
    ['interpretation_summary', cls.explanation],
    [
      'methodological_limitations',
      [cls.limitationsNote, ...sample.knownLimitations].join(' | '),
    ],
  ];

  const lines = ['metric,value'];
  for (const [k, v] of rows) {
    lines.push(`${escapeCsvCell(k)},${escapeCsvCell(v)}`);
  }
  return lines.join('\n');
}

/**
 * Pure builder for the complete JSON analysis & model results export.
 */
export function buildResultsJsonPayload(
  sample: SampleManifestEntry,
  config: PreprocessingConfig,
  result: ChangeDetectionResult,
  model?: ModelArtifact | null
) {
  const testMetrics = model?.isCalibratedModel ? model.testMetrics : null;
  const accuracyPct = calculateAccuracyPercent(testMetrics);

  return {
    application: 'BuildAhead — Satellite-Based Construction Change Detection',
    scopeNotice:
      'Historical multi-temporal satellite-image change detection and construction-related change classification. Does not predict future construction.',
    exportedAt: new Date().toISOString(),
    sample: {
      id: sample.id,
      aoiId: sample.aoiId,
      regionLabel: sample.regionLabel,
      split: sample.split,
      earlierDate: sample.earlierDate,
      laterDate: sample.laterDate,
      intervalMonths: sample.intervalMonths,
      sourceDataset: sample.sourceDataset,
      sourceUrl: sample.sourceUrl,
      sourceAttribution: sample.sourceAttribution,
      license: sample.license,
      earlierSourceS3Key: sample.earlierSourceS3Key,
      laterSourceS3Key: sample.laterSourceS3Key,
      isSyntheticFallback: sample.isSyntheticFallback,
      spatial: sample.spatial,
    },
    processingSettings: {
      ...config,
      preprocessingNote:
        'Image cleaning (denoising and contrast enhancement) improves visual legibility and reduces sensor noise, but does not by itself establish whether physical construction occurred.',
    },
    validation: result.validation,
    spatialAlignment: result.alignment,
    changeDetectionMetrics: {
      candidateChangedPixels: result.candidatePixelCount,
      candidateChangedPct: result.candidateChangedPct,
      candidateAreaM2: result.candidateAreaM2,
      candidateRegionCount: result.regions.length,
      pixelChangeIoU: result.pixelChangeIoU ?? null,
      rawUnfilteredDiffPixels: result.rawDiffPixelCount,
      rawUnfilteredDiffPct: result.rawDiffPct,
      suppressedEnvironmentalPixels: result.suppressedEnvPixelCount,
      suppressedEnvironmentalPct: result.suppressedEnvPct,
      topCandidateRegions: result.regions.slice(0, 20),
    },
    classificationOutput: result.classification,
    groundTruthAnnotations: sample.groundTruth,
    calibratedModelMetadata: model
      ? {
          modelType: model.modelType,
          modelLabel: model.modelLabel ?? model.modelType,
          version: model.version,
          datasetSource: model.datasetSource,
          evaluationSplitLabel:
            model.evaluationSplitLabel ?? 'SpaceNet 7 Held-Out Geographic Test Split',
          heldOutModelAccuracyDisplay: formatModelAccuracyDisplay(testMetrics),
          heldOutModelAccuracyPct:
            accuracyPct !== null ? Number(accuracyPct.toFixed(2)) : null,
          heldOutModelAccuracyCi95Pct: testMetrics?.accuracyCi95
            ? [
                Number((testMetrics.accuracyCi95[0] * 100).toFixed(2)),
                Number((testMetrics.accuracyCi95[1] * 100).toFixed(2)),
              ]
            : null,
          leakagePrevention: model.leakagePrevention,
          minTrainTestDistanceKm: model.minTrainTestDistanceKm,
          minSplitPairwiseDistanceKm: model.minSplitPairwiseDistanceKm ?? null,
          decisionThreshold: model.decisionThreshold,
          classWeights: model.classWeights ?? null,
          calibration: model.calibration ?? null,
          trainMetrics: model.trainMetrics,
          geoCvMetrics: model.geoCvMetrics ?? null,
          validationMetrics: model.valMetrics ?? null,
          heldOutTestMetrics: model.testMetrics,
          pixelIoUMetrics: model.pixelIoUMetrics ?? null,
          baselineComparison: model.baselineComparison ?? null,
        }
      : {
          heldOutModelAccuracyDisplay: 'Not evaluated.',
          heldOutModelAccuracyPct: null,
        },
    knownLimitations: [
      result.classification.limitationsNote,
      ...sample.knownLimitations,
    ],
  };
}

function triggerBlobDownload(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = filename;
  anchor.style.display = 'none';
  document.body.appendChild(anchor);
  anchor.click();
  document.body.removeChild(anchor);
  setTimeout(() => URL.revokeObjectURL(url), 1500);
}

export function downloadChangeMapPng(
  buffer: RawImageBuffer,
  sample: SampleManifestEntry
): void {
  const canvas = document.createElement('canvas');
  canvas.width = buffer.width;
  canvas.height = buffer.height;
  const ctx = canvas.getContext('2d');
  if (!ctx) return;

  const imgData = new ImageData(
    new Uint8ClampedArray(buffer.data),
    buffer.width,
    buffer.height
  );
  ctx.putImageData(imgData, 0, 0);

  canvas.toBlob((blob) => {
    if (!blob) return;
    const fname = `buildahead_${sample.id}_${sample.earlierDate}_to_${sample.laterDate}_changemap.png`;
    triggerBlobDownload(blob, fname);
  }, 'image/png');
}

export function downloadComparisonCsv(
  sample: SampleManifestEntry,
  config: PreprocessingConfig,
  result: ChangeDetectionResult,
  model?: ModelArtifact | null
): void {
  const csv = buildComparisonCsvString(sample, config, result, model);
  const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
  const fname = `buildahead_${sample.id}_${sample.earlierDate}_to_${sample.laterDate}_summary.csv`;
  triggerBlobDownload(blob, fname);
}

export function downloadResultsJson(
  sample: SampleManifestEntry,
  config: PreprocessingConfig,
  result: ChangeDetectionResult,
  model?: ModelArtifact | null
): void {
  const payload = buildResultsJsonPayload(sample, config, result, model);
  const blob = new Blob([JSON.stringify(payload, null, 2)], {
    type: 'application/json;charset=utf-8;',
  });
  const fname = `buildahead_${sample.id}_${sample.earlierDate}_to_${sample.laterDate}_results.json`;
  triggerBlobDownload(blob, fname);
}
