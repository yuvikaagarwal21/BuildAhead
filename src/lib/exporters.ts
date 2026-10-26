import type {
  ChangeDetectionResult,
  ModelArtifact,
  PreprocessingConfig,
  RawImageBuffer,
  SampleManifestEntry,
} from '../types/satellite';

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
  result: ChangeDetectionResult
): string {
  const gt = sample.groundTruth;
  const cls = result.classification;

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
    ['raw_unfiltered_diff_pixels', result.rawDiffPixelCount],
    ['raw_unfiltered_diff_pct', result.rawDiffPct],
    ['suppressed_seasonal_shadow_pixels', result.suppressedEnvPixelCount],
    ['suppressed_seasonal_shadow_pct', result.suppressedEnvPct],
    ['feature_structural_change_pct', cls.features.structuralChangePct],
    ['feature_high_albedo_gain_pct', cls.features.highAlbedoGainPct],
    ['feature_mean_cluster_edge_gain', cls.features.meanClusterEdgeGain],
    ['feature_compact_cluster_density_km2', cls.features.compactClusterDensityKm2],
    ['feature_seasonal_veg_shift_fraction', cls.features.seasonalVegShiftFraction],
    ['classification_mode', cls.mode],
    ['predicted_category', cls.predictedCategory],
    [
      'predicted_building_change_probability',
      cls.probabilityBuildingChange !== null ? cls.probabilityBuildingChange : 'N/A (rule screening)',
    ],
    [
      'predicted_class_confidence',
      cls.confidenceScore !== null ? cls.confidenceScore : 'N/A (rule screening)',
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
          version: model.version,
          datasetSource: model.datasetSource,
          leakagePrevention: model.leakagePrevention,
          minTrainTestDistanceKm: model.minTrainTestDistanceKm,
          trainMetrics: model.trainMetrics,
          heldOutTestMetrics: model.testMetrics,
        }
      : null,
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
  result: ChangeDetectionResult
): void {
  const csv = buildComparisonCsvString(sample, config, result);
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
