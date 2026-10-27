import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { PNG } from 'pngjs';
import {
  DEFAULT_PREPROCESSING_CONFIG,
  runChangeDetectionPipeline,
} from '../src/lib/changeDetection.ts';
import {
  calculateAccuracyPercent,
  computeLogLoss,
  computeReliabilityDiagramAndEce,
  formatModelAccuracyDisplay,
  trainValidateAndSelectBestModel,
} from '../src/lib/classifier.ts';
import {
  buildComparisonCsvString,
  buildResultsJsonPayload,
} from '../src/lib/exporters.ts';
import {
  alignLaterImage,
  applyContrastEnhancement,
  applyDenoising,
  validateImagePair,
} from '../src/lib/imageProcessing.ts';
import type {
  BuildingFootprintPolygon,
  DatasetManifest,
  ModelArtifact,
  RawImageBuffer,
} from '../src/types/satellite.ts';

const ROOT = process.cwd();

function haversineKm(lat1: number, lon1: number, lat2: number, lon2: number): number {
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

function readPng(relPublicUrl: string): RawImageBuffer {
  const full = path.join(ROOT, 'public', relPublicUrl.replace(/^\//, ''));
  assert.ok(fs.existsSync(full), `Missing PNG file: ${full}`);
  const decoded = PNG.sync.read(fs.readFileSync(full));
  return {
    width: decoded.width,
    height: decoded.height,
    data: new Uint8ClampedArray(decoded.data),
  };
}

function bufferMeanAbsDiff(a: RawImageBuffer, b: RawImageBuffer): number {
  let sum = 0;
  const n = a.width * a.height;
  for (let i = 0; i < n; i++) {
    const idx = i * 4;
    sum +=
      Math.abs(a.data[idx] - b.data[idx]) +
      Math.abs(a.data[idx + 1] - b.data[idx + 1]) +
      Math.abs(a.data[idx + 2] - b.data[idx + 2]);
  }
  return sum / (n * 3);
}

async function runTests() {
  let passed = 0;
  const logPass = (name: string, detail: string) => {
    passed++;
    console.log(`PASS [${passed}] ${name} — ${detail}`);
  };

  // 1. Preloaded Dataset & Default Sample Load Check
  const manifestPath = path.join(ROOT, 'public/data/spacenet7/manifest.json');
  const modelPath = path.join(ROOT, 'src/data/modelArtifact.json');
  assert.ok(fs.existsSync(manifestPath), 'manifest.json must exist');
  assert.ok(fs.existsSync(modelPath), 'modelArtifact.json must exist');

  const manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf-8')) as DatasetManifest;
  const model = JSON.parse(fs.readFileSync(modelPath, 'utf-8')) as ModelArtifact;

  assert.ok(manifest.samples.length >= 8, 'At least 8 preloaded samples required');
  const posSamples = manifest.samples.filter((s) => s.groundTruth.groundTruthPositive);
  const negSamples = manifest.samples.filter((s) => !s.groundTruth.groundTruthPositive);
  assert.ok(posSamples.length >= 4, 'Must include >= 4 positive construction pairs');
  assert.ok(negSamples.length >= 4, 'Must include >= 4 little/no construction pairs');
  logPass(
    'Preloaded SpaceNet 7 pack integrity',
    `${manifest.samples.length} samples (${posSamples.length} positive, ${negSamples.length} negative), default=${manifest.samples[0].id}`
  );

  // 2. Model Accuracy Formula & "Not evaluated." Fallback Check
  assert.equal(formatModelAccuracyDisplay(null), 'Not evaluated.');
  assert.equal(
    formatModelAccuracyDisplay({
      sampleCount: 0,
      positiveCount: 0,
      negativeCount: 0,
      accuracy: 0,
      precision: 0,
      recall: 0,
      f1Score: 0,
      brierScore: 0,
      rocAuc: 0.5,
      confusionMatrix: { tp: 0, fp: 0, tn: 0, fn: 0 },
    }),
    'Not evaluated.'
  );

  const cm = model.testMetrics.confusionMatrix;
  const manualFormulaPct =
    ((cm.tp + cm.tn) / (cm.tp + cm.tn + cm.fp + cm.fn)) * 100;
  const calcPct = calculateAccuracyPercent(model.testMetrics);
  assert.ok(calcPct !== null, 'Calibrated model test accuracy must be non-null');
  assert.ok(
    Math.abs(calcPct - manualFormulaPct) < 1e-9,
    'calculateAccuracyPercent must equal ((TP+TN)/(TP+TN+FP+FN))*100'
  );
  assert.equal(
    formatModelAccuracyDisplay(model.testMetrics),
    `${manualFormulaPct.toFixed(1)}%`
  );
  logPass(
    'Model Accuracy formula & fallback check',
    `((TP ${cm.tp} + TN ${cm.tn}) / ${cm.tp + cm.tn + cm.fp + cm.fn}) * 100 = ${manualFormulaPct.toFixed(
      2
    )}% (display="${formatModelAccuracyDisplay(model.testMetrics)}", null fallback="Not evaluated.")`
  );

  // 3. Leak-Free 3-Way Geographic Split & Zero Test-Leakage Proof
  const trainRecs = model.records.filter((r) => r.split === 'train');
  const valRecs = model.records.filter((r) => r.split === 'val');
  const testRecs = model.records.filter((r) => r.split === 'test');
  assert.ok(trainRecs.length >= 20, 'Train split must contain >= 20 records');
  assert.ok(valRecs.length >= 10, 'Val split must contain >= 10 records');
  assert.equal(
    testRecs.length,
    manifest.samples.length,
    'Held-out test split count must match preloaded manifest samples'
  );

  const trainAois = new Set(trainRecs.map((r) => r.aoiId));
  const valAois = new Set(valRecs.map((r) => r.aoiId));
  const testAois = new Set(testRecs.map((r) => r.aoiId));
  for (const aoi of testAois) {
    assert.ok(!trainAois.has(aoi), `Test AOI ${aoi} leaked into Train`);
    assert.ok(!valAois.has(aoi), `Test AOI ${aoi} leaked into Val`);
  }
  for (const aoi of valAois) {
    assert.ok(!trainAois.has(aoi), `Val AOI ${aoi} leaked into Train`);
  }

  for (const te of testRecs) {
    for (const tr of [...trainRecs, ...valRecs]) {
      const dist = haversineKm(te.lat, te.lon, tr.lat, tr.lon);
      assert.ok(
        dist >= 200,
        `Cross-split pair ${te.pairId} vs ${tr.pairId} too close (${dist.toFixed(1)} km)`
      );
    }
  }

  // Prove mathematically that flipping ALL test labels does NOT change model selection, weights, scaler, or threshold
  const flippedTestRecords = model.records.map((r) =>
    r.split === 'test' ? { ...r, groundTruthPositive: !r.groundTruthPositive } : r
  );
  const refitWithFlippedTest = trainValidateAndSelectBestModel({
    records: flippedTestRecords,
    positiveCriterion: model.positiveCriterion,
    negativeCriterion: model.negativeCriterion,
    leakagePrevention: model.leakagePrevention,
    minTrainTestDistanceKm: model.minTrainTestDistanceKm,
    minSplitPairwiseDistanceKm: model.minSplitPairwiseDistanceKm ?? {
      trainVal: 300,
      trainTest: 300,
      valTest: 300,
    },
    datasetSource: model.datasetSource,
  });
  assert.equal(refitWithFlippedTest.modelType, model.modelType);
  assert.equal(refitWithFlippedTest.decisionThreshold, model.decisionThreshold);
  assert.deepEqual(refitWithFlippedTest.weights, model.weights);
  assert.deepEqual(refitWithFlippedTest.scalerMean, model.scalerMean);
  assert.equal(refitWithFlippedTest.calibration?.method, model.calibration?.method);
  assert.equal(refitWithFlippedTest.calibration?.plattSlope, model.calibration?.plattSlope);
  assert.equal(refitWithFlippedTest.calibration?.plattIntercept, model.calibration?.plattIntercept);
  logPass(
    'Zero test-set leakage (classifier + Platt probability calibration) & 3-way geographic separation proof',
    `Train=${trainRecs.length}, Val=${valRecs.length}, Test=${testRecs.length}; 0 shared AOIs; flipping all test labels leaves modelType, weights, scaler, threshold=${model.decisionThreshold}, and calibration (${model.calibration?.method}, a=${model.calibration?.plattSlope}, b=${model.calibration?.plattIntercept}) 100% identical`
  );

  // 3b. Proper Scoring Rules (Log Loss, Brier, ECE) & Reliability Diagram Check
  assert.ok(model.calibration, 'Model artifact must include leak-free calibration metadata');
  assert.equal(
    model.calibration.benchmarkOnNonTest.length,
    3,
    'Must benchmark Uncalibrated, Platt Sigmoid, and Isotonic PAV'
  );
  const nonTestBinSum = model.calibration.nonTestReliabilityAfter.reduce(
    (s, b) => s + b.count,
    0
  );
  assert.equal(
    nonTestBinSum,
    trainRecs.length + valRecs.length,
    'Non-test reliability diagram bins must sum to Train + Val sample count (59)'
  );
  const perfectLogLoss = computeLogLoss([true, false], [0.999, 0.001]);
  const poorLogLoss = computeLogLoss([true, false], [0.4, 0.6]);
  assert.ok(perfectLogLoss < 0.01 && poorLogLoss > 0.9);
  const relCheck = computeReliabilityDiagramAndEce(
    [true, true, false, false],
    [0.9, 0.85, 0.1, 0.15],
    5
  );
  assert.ok(relCheck.ece >= 0 && relCheck.ece < 0.2);
  logPass(
    'Probability calibration benchmark & Reliability Diagram math',
    `Selected=${model.calibration.method}, Test Brier=${model.testMetrics.brierScore}, Test LogLoss=${model.testMetrics.logLoss}, Test ECE=${model.testMetrics.ece}`
  );

  // 4. Observation Dates & Spatial Metadata Check across all samples
  for (const s of manifest.samples) {
    assert.match(s.earlierDate, /^\d{4}-\d{2}$/);
    assert.match(s.laterDate, /^\d{4}-\d{2}$/);
    assert.ok(
      s.earlierDate < s.laterDate,
      `Earlier date ${s.earlierDate} must precede later date ${s.laterDate}`
    );
    assert.ok(s.intervalMonths > 0);
    const eImg = readPng(s.earlierImageUrl);
    const lImg = readPng(s.laterImageUrl);
    assert.equal(eImg.width, 512);
    assert.equal(eImg.height, 512);
    assert.equal(lImg.width, 512);
    assert.equal(lImg.height, 512);
  }
  logPass(
    'Observation dates & raster dimensions',
    `All ${manifest.samples.length} pairs have chronological YYYY-MM dates and 512x512 RGBA rasters`
  );

  // 5. Genuine Preprocessing Calculations Check (Denoising, Contrast, Co-Registration)
  const defaultSample = manifest.samples[0];
  const eDefault = readPng(defaultSample.earlierImageUrl);
  const lDefault = readPng(defaultSample.laterImageUrl);

  const denoiseMed = applyDenoising(eDefault, 'median3x3');
  const denoiseBil = applyDenoising(eDefault, 'bilateral');
  const diffMed = bufferMeanAbsDiff(eDefault, denoiseMed);
  const diffBil = bufferMeanAbsDiff(eDefault, denoiseBil);
  assert.ok(diffMed > 0.1 && diffMed < 35, `Median filter must alter pixels (${diffMed})`);
  assert.ok(diffBil > 0.1 && diffBil < 35, `Bilateral filter must alter pixels (${diffBil})`);

  const contrastLin = applyContrastEnhancement(eDefault, 'linear_percentile');
  const contrastClahe = applyContrastEnhancement(eDefault, 'clahe');
  const diffLin = bufferMeanAbsDiff(eDefault, contrastLin);
  const diffClahe = bufferMeanAbsDiff(eDefault, contrastClahe);
  assert.ok(diffLin > 1.0, `Linear percentile stretch must alter contrast (${diffLin})`);
  assert.ok(diffClahe > 1.0, `CLAHE must alter local contrast (${diffClahe})`);

  const shiftedTest: RawImageBuffer = {
    width: eDefault.width,
    height: eDefault.height,
    data: new Uint8ClampedArray(eDefault.width * eDefault.height * 4),
  };
  const injectedDx = 3;
  const injectedDy = -2;
  for (let y = 0; y < eDefault.height; y++) {
    const sy = y + injectedDy;
    if (sy < 0 || sy >= eDefault.height) continue;
    for (let x = 0; x < eDefault.width; x++) {
      const sx = x + injectedDx;
      if (sx < 0 || sx >= eDefault.width) continue;
      const dst = (y * eDefault.width + x) * 4;
      const src = (sy * eDefault.width + sx) * 4;
      shiftedTest.data[dst] = eDefault.data[src];
      shiftedTest.data[dst + 1] = eDefault.data[src + 1];
      shiftedTest.data[dst + 2] = eDefault.data[src + 2];
      shiftedTest.data[dst + 3] = 255;
    }
  }
  const regCheck = alignLaterImage(eDefault, shiftedTest, true, 4);
  assert.equal(regCheck.alignment.dx, injectedDx, 'Recovered dx must equal +3');
  assert.equal(regCheck.alignment.dy, injectedDy, 'Recovered dy must equal -2');
  assert.ok(regCheck.alignment.nccScoreAfter > regCheck.alignment.nccScoreBefore);
  logPass(
    'Genuine image-processing & co-registration math',
    `median Δ=${diffMed.toFixed(2)}, bilateral Δ=${diffBil.toFixed(2)}, CLAHE Δ=${diffClahe.toFixed(
      2
    )}, recovered shift=(${regCheck.alignment.dx}, ${regCheck.alignment.dy})`
  );

  // 6. End-to-End Change Detection, Pixel IoU & Calibrated vs Rule-Based Classification Check
  let correctHeldOut = 0;
  for (const s of manifest.samples) {
    const eBuf = readPng(s.earlierImageUrl);
    const lBuf = readPng(s.laterImageUrl);
    const fpFull = s.groundTruth.footprintsUrl
      ? (JSON.parse(
          fs.readFileSync(
            path.join(ROOT, 'public', s.groundTruth.footprintsUrl.replace(/^\//, '')),
            'utf-8'
          )
        ) as BuildingFootprintPolygon[])
      : [];
    const out = runChangeDetectionPipeline({
      earlier: eBuf,
      later: lBuf,
      spatial: s.spatial,
      config: DEFAULT_PREPROCESSING_CONFIG,
      model,
      groundTruthFootprints: fpFull,
    });
    assert.ok(out.validation.valid);
    assert.equal(out.changeMapOverlay.width, 512);
    assert.equal(out.differenceHeatmap.width, 512);
    assert.ok(
      out.classification.mode === 'calibrated_logistic_regression' ||
        out.classification.mode === 'calibrated_random_forest' ||
        out.classification.mode === 'calibrated_gbdt_ensemble'
    );
    assert.ok(
      out.classification.probabilityBuildingChange !== null &&
        out.classification.probabilityBuildingChange >= 0 &&
        out.classification.probabilityBuildingChange <= 1
    );
    if (s.groundTruth.groundTruthPositive) {
      assert.ok(
        out.pixelChangeIoU !== null && out.pixelChangeIoU !== undefined && out.pixelChangeIoU >= 0,
        `Positive sample ${s.id} must report non-null pixelChangeIoU`
      );
    }
    if (out.classification.predictedPositive === s.groundTruth.groundTruthPositive) {
      correctHeldOut++;
    }
  }
  assert.equal(
    correctHeldOut,
    model.testMetrics.confusionMatrix.tp + model.testMetrics.confusionMatrix.tn,
    'Live pipeline predictions on preloaded samples must match recorded held-out test metrics'
  );
  assert.ok(
    Array.isArray(model.testMetrics.accuracyCi95) &&
      model.testMetrics.accuracyCi95[0] <= model.testMetrics.accuracy &&
      model.testMetrics.accuracy <= model.testMetrics.accuracyCi95[1],
    'Held-out test metrics must include valid 95% Wilson confidence interval'
  );
  assert.ok(
    model.geoCvMetrics && model.geoCvMetrics.sampleCount === trainRecs.length,
    'Model artifact must include out-of-fold 5-Fold Geographic Group CV metrics on training split'
  );

  const ruleFallbackOut = runChangeDetectionPipeline({
    earlier: eDefault,
    later: lDefault,
    spatial: defaultSample.spatial,
    config: DEFAULT_PREPROCESSING_CONFIG,
    model: null,
  });
  assert.equal(ruleFallbackOut.classification.mode, 'rule_based_screening');
  assert.equal(ruleFallbackOut.classification.probabilityBuildingChange, null);
  logPass(
    'Change-detection, Pixel IoU & calibrated/rule classification parity',
    `Verified ${correctHeldOut}/${manifest.samples.length} held-out test matches (${formatModelAccuracyDisplay(
      model.testMetrics
    )}) + rule-screening fallback`
  );

  // 7. Invalid Sample Data Safe Handling Check
  const nullVal = validateImagePair(null, null);
  assert.equal(nullVal.valid, false);

  const mismatchBuf: RawImageBuffer = {
    width: 128,
    height: 128,
    data: new Uint8ClampedArray(128 * 128 * 4),
  };
  const mismatchOut = runChangeDetectionPipeline({
    earlier: eDefault,
    later: mismatchBuf,
    model,
  });
  assert.equal(mismatchOut.validation.valid, false);
  assert.equal(mismatchOut.validation.dimensionsMatch, false);

  const cloudyBuf: RawImageBuffer = {
    width: 512,
    height: 512,
    data: new Uint8ClampedArray(512 * 512 * 4),
  };
  const cloudyOut = runChangeDetectionPipeline({
    earlier: eDefault,
    later: cloudyBuf,
    model,
  });
  assert.equal(cloudyOut.validation.valid, false);
  logPass(
    'Safe handling of invalid/corrupted/cloudy sample data',
    'Cleanly rejected null buffers, dimension mismatches, and 0% clear-sky tiles without throwing'
  );

  // 8. Export Parity Check (CSV and JSON outputs keep Model Confidence and Model Accuracy separate & accurate)
  const defaultFp = defaultSample.groundTruth.footprintsUrl
    ? (JSON.parse(
        fs.readFileSync(
          path.join(
            ROOT,
            'public',
            defaultSample.groundTruth.footprintsUrl.replace(/^\//, '')
          ),
          'utf-8'
        )
      ) as BuildingFootprintPolygon[])
    : [];
  const sampleOut = runChangeDetectionPipeline({
    earlier: eDefault,
    later: lDefault,
    spatial: defaultSample.spatial,
    config: DEFAULT_PREPROCESSING_CONFIG,
    model,
    groundTruthFootprints: defaultFp,
  });
  const csvStr = buildComparisonCsvString(
    defaultSample,
    DEFAULT_PREPROCESSING_CONFIG,
    sampleOut,
    model
  );
  const jsonPayload = buildResultsJsonPayload(
    defaultSample,
    DEFAULT_PREPROCESSING_CONFIG,
    sampleOut,
    model
  );

  assert.ok(csvStr.includes(`sample_id,${defaultSample.id}`));
  assert.ok(
    csvStr.includes(
      `model_confidence_single_sample,${sampleOut.classification.confidenceScore}`
    )
  );
  assert.ok(
    csvStr.includes(
      `model_accuracy_held_out_test_pct,${Number(manualFormulaPct.toFixed(2))}`
    )
  );
  assert.notEqual(
    sampleOut.classification.confidenceScore! * 100,
    manualFormulaPct,
    'Single-pair Model Confidence and Held-Out Model Accuracy must remain distinct quantities'
  );
  assert.equal(
    jsonPayload.calibratedModelMetadata.heldOutModelAccuracyDisplay,
    formatModelAccuracyDisplay(model.testMetrics)
  );
  assert.ok(
    csvStr.includes(`model_test_brier_score,${model.testMetrics.brierScore}`),
    'CSV export must include held-out test Brier score'
  );
  assert.ok(
    csvStr.includes(`model_test_log_loss,${model.testMetrics.logLoss}`),
    'CSV export must include held-out test Log Loss'
  );
  logPass(
    'Exported CSV & JSON parity (Model Confidence, Calibration Metrics & Model Accuracy separated)',
    `Sample=${defaultSample.id}, Confidence=${(
      sampleOut.classification.confidenceScore! * 100
    ).toFixed(1)}% (raw=${(
      (sampleOut.classification.rawUncalibratedProbability ?? 0) * 100
    ).toFixed(1)}%), Held-Out Accuracy=${formatModelAccuracyDisplay(model.testMetrics)}`
  );

  console.log(`\nALL ${passed} AUTOMATED VERIFICATION SUITES PASSED.`);
}

runTests().catch((err) => {
  console.error('VERIFICATION FAILED:', err);
  process.exit(1);
});
