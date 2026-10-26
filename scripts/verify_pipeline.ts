import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { PNG } from 'pngjs';
import {
  DEFAULT_PREPROCESSING_CONFIG,
  runChangeDetectionPipeline,
} from '../src/lib/changeDetection.ts';
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
  DatasetManifest,
  ModelArtifact,
  RawImageBuffer,
} from '../src/types/satellite.ts';

const ROOT = process.cwd();

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

  assert.ok(manifest.samples.length >= 6, 'At least 6 preloaded samples required');
  const posSamples = manifest.samples.filter((s) => s.groundTruth.groundTruthPositive);
  const negSamples = manifest.samples.filter((s) => !s.groundTruth.groundTruthPositive);
  assert.ok(posSamples.length >= 1, 'Must include >= 1 positive construction pair');
  assert.ok(negSamples.length >= 1, 'Must include >= 1 little/no construction pair');
  logPass(
    'Preloaded SpaceNet 7 pack integrity',
    `${manifest.samples.length} samples (${posSamples.length} positive, ${negSamples.length} negative), default=${manifest.samples[0].id}`
  );

  // 2. Observation Dates & Spatial Metadata Check across all samples
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

  // 3. Genuine Preprocessing Calculations Check (Denoising, Contrast, Co-Registration)
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

  // Inject known (+3, -2) pixel shift into earlier image and verify Sobel-edge NCC recovers it
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
    )}, recovered shift=(${regCheck.alignment.dx}, ${regCheck.alignment.dy}) NCC ${regCheck.alignment.nccScoreBefore.toFixed(
      3
    )}->${regCheck.alignment.nccScoreAfter.toFixed(3)}`
  );

  // 4. End-to-End Change Detection & Calibrated vs Rule-Based Classification Check
  let correctHeldOut = 0;
  for (const s of manifest.samples) {
    const eBuf = readPng(s.earlierImageUrl);
    const lBuf = readPng(s.laterImageUrl);
    const out = runChangeDetectionPipeline({
      earlier: eBuf,
      later: lBuf,
      spatial: s.spatial,
      config: DEFAULT_PREPROCESSING_CONFIG,
      model,
    });
    assert.ok(out.validation.valid);
    assert.equal(out.changeMapOverlay.width, 512);
    assert.equal(out.differenceHeatmap.width, 512);
    assert.equal(out.classification.mode, 'calibrated_logistic_regression');
    assert.ok(
      out.classification.probabilityBuildingChange !== null &&
        out.classification.probabilityBuildingChange >= 0 &&
        out.classification.probabilityBuildingChange <= 1
    );
    if (out.classification.predictedPositive === s.groundTruth.groundTruthPositive) {
      correctHeldOut++;
    }
  }
  assert.equal(
    correctHeldOut,
    model.testMetrics.confusionMatrix.tp + model.testMetrics.confusionMatrix.tn,
    'Live pipeline predictions on preloaded samples must match recorded held-out test metrics'
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
    'Change-detection & calibrated/rule classification parity',
    `Verified ${correctHeldOut}/${manifest.samples.length} held-out test matches (${(
      model.testMetrics.accuracy * 100
    ).toFixed(1)}%) + rule-screening fallback`
  );

  // 5. Invalid Sample Data Safe Handling Check
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
  assert.ok(mismatchOut.validation.errors[0].includes('mismatch'));

  const cloudyBuf: RawImageBuffer = {
    width: 512,
    height: 512,
    data: new Uint8ClampedArray(512 * 512 * 4), // alpha = 0 everywhere
  };
  const cloudyOut = runChangeDetectionPipeline({
    earlier: eDefault,
    later: cloudyBuf,
    model,
  });
  assert.equal(cloudyOut.validation.valid, false);
  assert.ok(cloudyOut.validation.errors[0].includes('cloud-free overlap'));
  logPass(
    'Safe handling of invalid/corrupted/cloudy sample data',
    'Cleanly rejected null buffers, dimension mismatches (512x512 vs 128x128), and 0% clear-sky masked tiles without throwing'
  );

  // 6. Export Parity Check (CSV and JSON outputs match displayed pipeline results)
  const sampleOut = runChangeDetectionPipeline({
    earlier: eDefault,
    later: lDefault,
    spatial: defaultSample.spatial,
    config: DEFAULT_PREPROCESSING_CONFIG,
    model,
  });
  const csvStr = buildComparisonCsvString(
    defaultSample,
    DEFAULT_PREPROCESSING_CONFIG,
    sampleOut
  );
  const jsonPayload = buildResultsJsonPayload(
    defaultSample,
    DEFAULT_PREPROCESSING_CONFIG,
    sampleOut,
    model
  );

  assert.ok(csvStr.includes(`sample_id,${defaultSample.id}`));
  assert.ok(csvStr.includes(`earlier_observation_date,${defaultSample.earlierDate}`));
  assert.ok(csvStr.includes(`later_observation_date,${defaultSample.laterDate}`));
  assert.ok(
    csvStr.includes(`candidate_changed_pixels,${sampleOut.candidatePixelCount}`)
  );
  assert.ok(
    csvStr.includes(`predicted_category,${sampleOut.classification.predictedCategory}`)
  );

  assert.equal(jsonPayload.sample.id, defaultSample.id);
  assert.equal(
    jsonPayload.changeDetectionMetrics.candidateChangedPixels,
    sampleOut.candidatePixelCount
  );
  assert.equal(
    jsonPayload.classificationOutput.probabilityBuildingChange,
    sampleOut.classification.probabilityBuildingChange
  );
  assert.ok(jsonPayload.knownLimitations.length >= 2);
  logPass(
    'Exported CSV & JSON parity with displayed outputs',
    `Verified exact match for sample=${defaultSample.id}, candidatePixels=${sampleOut.candidatePixelCount}, P(bldg)=${sampleOut.classification.probabilityBuildingChange}`
  );

  console.log(`\nALL ${passed} AUTOMATED VERIFICATION SUITES PASSED.`);
}

runTests().catch((err) => {
  console.error('VERIFICATION FAILED:', err);
  process.exit(1);
});
