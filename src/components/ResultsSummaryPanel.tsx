import React from 'react';
import { FileJson, FileSpreadsheet, ImageDown } from 'lucide-react';
import type {
  ChangeDetectionResult,
  ModelArtifact,
  PreprocessingConfig,
  SampleManifestEntry,
} from '../types/satellite.ts';
import {
  calculateAccuracyPercent,
  formatModelAccuracyDisplay,
} from '../lib/classifier.ts';
import {
  downloadChangeMapPng,
  downloadComparisonCsv,
  downloadResultsJson,
} from '../lib/exporters.ts';

interface ResultsSummaryPanelProps {
  sample: SampleManifestEntry;
  config: PreprocessingConfig;
  result: ChangeDetectionResult;
  model: ModelArtifact | null;
}

export const ResultsSummaryPanel: React.FC<ResultsSummaryPanelProps> = ({
  sample,
  config,
  result,
  model,
}) => {
  const { classification: cls } = result;
  const gt = sample.groundTruth;

  // Genuine held-out test set accuracy: ((TP + TN) / (TP + TN + FP + FN)) * 100
  const heldOutTestMetrics = model?.isCalibratedModel ? model.testMetrics : null;
  const accuracyPct = calculateAccuracyPercent(heldOutTestMetrics);
  const accuracyDisplay = formatModelAccuracyDisplay(heldOutTestMetrics);
  const cm = heldOutTestMetrics?.confusionMatrix;
  const evaluatedCount = cm ? cm.tp + cm.tn + cm.fp + cm.fn : 0;
  const splitLabel =
    model?.evaluationSplitLabel ?? 'SpaceNet 7 Held-Out Test Split';

  return (
    <section
      id="results"
      className="bg-white border border-slate-200 rounded-xl p-6 flex flex-col gap-6"
    >
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-4 border-b border-slate-200">
        <div>
          <h2 className="text-lg font-semibold text-slate-900">
            Analysis Results
          </h2>
          <p className="text-xs text-slate-500 mt-0.5">
            {sample.regionLabel} · {sample.earlierDate} to {sample.laterDate}
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <button
            type="button"
            onClick={() => downloadChangeMapPng(result.changeMapOverlay, sample)}
            className="inline-flex items-center gap-1.5 px-3 py-2 text-xs font-medium text-slate-800 bg-slate-100 hover:bg-slate-200 rounded-lg transition-colors whitespace-nowrap shrink-0 cursor-pointer"
          >
            <ImageDown className="w-4 h-4" />
            Change Map (PNG)
          </button>
          <button
            type="button"
            onClick={() => downloadComparisonCsv(sample, config, result, model)}
            className="inline-flex items-center gap-1.5 px-3 py-2 text-xs font-medium text-slate-800 bg-slate-100 hover:bg-slate-200 rounded-lg transition-colors whitespace-nowrap shrink-0 cursor-pointer"
          >
            <FileSpreadsheet className="w-4 h-4" />
            Summary (CSV)
          </button>
          <button
            type="button"
            onClick={() => downloadResultsJson(sample, config, result, model)}
            className="inline-flex items-center gap-1.5 px-3 py-2 text-xs font-medium text-white bg-blue-600 hover:bg-blue-700 rounded-lg transition-colors whitespace-nowrap shrink-0 cursor-pointer"
          >
            <FileJson className="w-4 h-4" />
            Full Results (JSON)
          </button>
        </div>
      </div>

      {/* Headline Metrics — Keeping Model Confidence (single pair) and Model Accuracy (held-out test set) strictly separate */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-6 pb-5 border-b border-slate-200">
        <div>
          <div className="text-xs text-slate-500">Predicted Category</div>
          <div
            className={`text-base font-semibold mt-1 leading-snug ${
              cls.predictedPositive ? 'text-amber-700' : 'text-slate-900'
            }`}
          >
            {cls.predictedCategory}
          </div>
          <div className="text-xs text-slate-500 mt-1">
            {gt.groundTruthPositive === cls.predictedPositive
              ? 'Matches SpaceNet 7 ground truth'
              : 'Differs from SpaceNet 7 ground truth'}
          </div>
        </div>

        <div>
          <div className="text-xs text-slate-500">Model Confidence</div>
          <div
            className={`text-2xl font-mono font-semibold tabular-nums mt-1 ${
              cls.uncertaintyLevel === 'borderline_uncertain'
                ? 'text-amber-700'
                : 'text-slate-900'
            }`}
          >
            {cls.confidenceScore !== null
              ? `${(cls.confidenceScore * 100).toFixed(1)}%`
              : 'Unscored'}
          </div>
          <div className="text-xs font-mono tabular-nums text-slate-600 mt-1">
            {cls.probabilityBuildingChange !== null
              ? `This pair · P(bldg) ${(cls.probabilityBuildingChange * 100).toFixed(1)}%`
              : 'Rule-based screening'}
          </div>
          {cls.confidenceScore !== null && (
            <div
              className={`text-[11px] mt-0.5 ${
                cls.uncertaintyLevel === 'borderline_uncertain'
                  ? 'text-amber-700 font-medium'
                  : 'text-slate-500'
              }`}
            >
              {cls.uncertaintyLevel === 'borderline_uncertain'
                ? 'Borderline / uncertain — inspect overlay'
                : cls.uncertaintyLevel === 'moderate_certainty'
                ? 'Moderate calibrated certainty'
                : 'Platt-calibrated (Non-Test OOF)'}
            </div>
          )}
        </div>

        <div>
          <div className="text-xs text-slate-500">Model Accuracy</div>
          <div className="text-2xl font-mono font-semibold tabular-nums text-blue-700 mt-1">
            {accuracyDisplay}
          </div>
          <div className="text-xs font-mono tabular-nums text-slate-600 mt-1">
            {accuracyPct !== null && cm
              ? `${cm.tp + cm.tn}/${evaluatedCount} correct (n=${evaluatedCount}) · ${splitLabel.replace(
                  'SpaceNet 7 ',
                  ''
                )}`
              : 'No held-out evaluation split loaded'}
          </div>
          {accuracyPct !== null && heldOutTestMetrics?.accuracyCi95 && (
            <div className="text-[11px] font-mono tabular-nums text-slate-500 mt-0.5">
              95% Wilson CI:{' '}
              {(heldOutTestMetrics.accuracyCi95[0] * 100).toFixed(1)}%–
              {(heldOutTestMetrics.accuracyCi95[1] * 100).toFixed(1)}%
              {model?.geoCvMetrics
                ? ` · 5-Fold Geo-CV ${(model.geoCvMetrics.accuracy * 100).toFixed(1)}% (n=${model.geoCvMetrics.sampleCount})`
                : ''}
            </div>
          )}
        </div>

        <div>
          <div className="text-xs text-slate-500">Candidate Changed Pixels</div>
          <div className="text-2xl font-mono font-semibold tabular-nums text-slate-900 mt-1">
            {result.candidateChangedPct.toFixed(2)}%
          </div>
          <div className="text-xs font-mono tabular-nums text-slate-600 mt-1">
            {result.candidatePixelCount.toLocaleString()} px ·{' '}
            {result.regions.length} regions
            {result.pixelChangeIoU !== null &&
            result.pixelChangeIoU !== undefined
              ? ` · IoU ${result.pixelChangeIoU.toFixed(2)}`
              : ''}
          </div>
        </div>

        <div>
          <div className="text-xs text-slate-500">
            Ground-Truth Building Change
          </div>
          <div className="text-2xl font-mono font-semibold tabular-nums text-emerald-700 mt-1">
            +{gt.newConstructionCount.toLocaleString()}
          </div>
          <div className="text-xs font-mono tabular-nums text-slate-600 mt-1">
            {gt.earlierCount.toLocaleString()} → {gt.laterCount.toLocaleString()}{' '}
            bldgs · +{gt.newConstructionAreaM2.toLocaleString()} m²
          </div>
        </div>
      </div>

      {/* Plain-English Summary & Limitations */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 text-xs text-slate-600 leading-relaxed">
        <div>
          <h3 className="text-xs font-semibold text-slate-900 mb-1">
            Summary
          </h3>
          <p>{cls.explanation}</p>
          <p className="mt-1.5">
            Seasonal, snow, and smooth-shadow filtering suppressed{' '}
            <span className="font-mono tabular-nums font-medium text-slate-800">
              {result.suppressedEnvPct.toFixed(2)}%
            </span>{' '}
            of non-structural pixel differences across the tile.
          </p>
        </div>

        <div>
          <h3 className="text-xs font-semibold text-slate-900 mb-1">
            Important Limitations &amp; Small-Sample Uncertainty
          </h3>
          <p>
            <strong>Model Confidence</strong> reflects Platt-calibrated
            probability on this single tile (fitted strictly on n = 59 non-test
            out-of-fold logits; borderline pairs near the decision boundary are
            reported honestly rather than inflated), whereas{' '}
            <strong>Model Accuracy</strong> measures overall correctness on the
            untouched {evaluatedCount > 0 ? `n = ${evaluatedCount}` : 'held-out'}{' '}
            geographic test split (±8.3% per sample) and{' '}
            <strong>Candidate Changed Pixels</strong> measures pixel-level mask
            coverage. Historical change detection does not predict future
            construction.
          </p>
        </div>
      </div>
    </section>
  );
};
