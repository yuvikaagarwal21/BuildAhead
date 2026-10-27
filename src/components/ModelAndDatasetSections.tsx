import React from 'react';
import { ChevronDown, Download, ExternalLink } from 'lucide-react';
import type {
  DatasetManifest,
  EvaluationMetrics,
  ModelArtifact,
} from '../types/satellite.ts';
import { formatModelAccuracyDisplay } from '../lib/classifier.ts';

interface ModelAndDatasetSectionsProps {
  manifest: DatasetManifest | null;
  model: ModelArtifact | null;
  selectedSampleId: string;
  onSelectSample: (id: string) => void;
}

function formatCm(m: EvaluationMetrics): string {
  const { tp, fp, tn, fn } = m.confusionMatrix;
  return `TP ${tp} · TN ${tn} · FP ${fp} · FN ${fn}`;
}

function formatCi95(m?: EvaluationMetrics | null): string {
  if (!m?.accuracyCi95) return '—';
  return `[${(m.accuracyCi95[0] * 100).toFixed(1)}%, ${(
    m.accuracyCi95[1] * 100
  ).toFixed(1)}%]`;
}

export const ModelAndDatasetSections: React.FC<ModelAndDatasetSectionsProps> = ({
  manifest,
  model,
  selectedSampleId,
  onSelectSample,
}) => {
  const candidates = model?.baselineComparison?.candidateModels ?? [];
  const improvedTest = model?.testMetrics ?? null;

  return (
    <section id="about-dataset" className="flex flex-col gap-3">
      {/* Collapsible Dataset Manifest */}
      <details className="group bg-white border border-slate-200 rounded-xl overflow-hidden">
        <summary className="px-6 py-4 flex items-center justify-between gap-4 cursor-pointer select-none list-none hover:bg-slate-50/70">
          <div>
            <h2 className="text-sm font-semibold text-slate-900">
              Dataset Manifest &amp; Attribution (SpaceNet 7)
            </h2>
            <p className="text-xs text-slate-500 mt-0.5">
              View all {manifest?.samples.length ?? 0} preloaded held-out test
              pairs, observation dates, ground-truth counts, and CC-BY-4.0 source
              attribution
            </p>
          </div>
          <ChevronDown className="w-4 h-4 text-slate-500 transition-transform group-open:rotate-180 shrink-0" />
        </summary>

        {manifest && (
          <div className="px-6 pb-6 pt-2 border-t border-slate-200 flex flex-col gap-4">
            <div className="overflow-x-auto">
              <table className="w-full text-left border-collapse text-xs">
                <thead>
                  <tr className="border-b border-slate-200 text-slate-500">
                    <th className="py-2 pr-3 font-medium">Sample</th>
                    <th className="py-2 px-3 font-medium">Location</th>
                    <th className="py-2 px-3 font-medium">Dates</th>
                    <th className="py-2 px-3 font-medium text-right">
                      Footprints (Earlier → Later)
                    </th>
                    <th className="py-2 pl-3 font-medium">Ground-Truth Label</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100">
                  {manifest.samples.map((s) => (
                    <tr
                      key={s.id}
                      onClick={() => {
                        onSelectSample(s.id);
                        window.scrollTo({ top: 0, behavior: 'smooth' });
                      }}
                      className={`cursor-pointer ${
                        s.id === selectedSampleId
                          ? 'bg-blue-50/50'
                          : 'hover:bg-slate-50'
                      }`}
                    >
                      <td className="py-2 pr-3 font-mono tabular-nums font-medium text-blue-700">
                        {s.id}
                      </td>
                      <td className="py-2 px-3 text-slate-900">{s.regionLabel}</td>
                      <td className="py-2 px-3 font-mono tabular-nums text-slate-600">
                        {s.earlierDate} → {s.laterDate}
                      </td>
                      <td className="py-2 px-3 text-right font-mono tabular-nums text-slate-800">
                        {s.groundTruth.earlierCount} → {s.groundTruth.laterCount}{' '}
                        (+{s.groundTruth.newConstructionCount})
                      </td>
                      <td className="py-2 pl-3 text-slate-600">
                        {s.groundTruth.groundTruthClassLabel}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <div className="flex flex-wrap items-center justify-between gap-3 pt-2 border-t border-slate-100 text-xs text-slate-500">
              <span>
                Source: {manifest.attribution} ({manifest.license})
              </span>
              <a
                href={manifest.officialSourceUrl}
                target="_blank"
                rel="noreferrer"
                className="inline-flex items-center gap-1 text-blue-700 font-medium hover:underline"
              >
                SpaceNet 7 Official Page
                <ExternalLink className="w-3.5 h-3.5" />
              </a>
            </div>
          </div>
        )}
      </details>

      {/* Collapsible Classification Model, Leak-Free Splits & Baseline Comparison */}
      <details className="group bg-white border border-slate-200 rounded-xl overflow-hidden">
        <summary className="px-6 py-4 flex items-center justify-between gap-4 cursor-pointer select-none list-none hover:bg-slate-50/70">
          <div>
            <h2 className="text-sm font-semibold text-slate-900">
              Classification Model, Leak-Free Geographic Splits &amp; Baseline Comparison
            </h2>
            <p className="text-xs text-slate-500 mt-0.5">
              Selected classifier ({model?.modelLabel ?? 'Calibrated L2 Logistic Regression'}),
              5-fold geographic group cross-validation, candidate model benchmark, and small-sample uncertainty
            </p>
          </div>
          <ChevronDown className="w-4 h-4 text-slate-500 transition-transform group-open:rotate-180 shrink-0" />
        </summary>

        {model ? (
          <div className="px-6 pb-6 pt-3 border-t border-slate-200 flex flex-col gap-5 text-xs text-slate-600 leading-relaxed">
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-4 font-mono tabular-nums text-slate-800">
              <div>
                <div className="font-sans text-slate-500">
                  3-Way Geographic Split
                </div>
                <div className="text-sm font-semibold mt-0.5">
                  {model.trainMetrics.sampleCount} train ·{' '}
                  {model.valMetrics?.sampleCount ?? 0} val ·{' '}
                  {model.testMetrics.sampleCount} test
                </div>
                <div className="font-sans text-[11px] text-slate-500 mt-0.5">
                  {model.geoClusterCount ?? 50} disjoint 150 km clusters
                </div>
              </div>

              <div>
                <div className="font-sans text-slate-500">
                  5-Fold Geo-CV (Train)
                </div>
                <div className="text-sm font-semibold text-slate-900 mt-0.5">
                  {model.geoCvMetrics
                    ? `${formatModelAccuracyDisplay(model.geoCvMetrics)} (F1 ${(
                        model.geoCvMetrics.f1Score * 100
                      ).toFixed(1)}%)`
                    : '—'}
                </div>
                <div className="font-sans text-[11px] text-slate-500 mt-0.5">
                  {model.geoCvMetrics
                    ? `AUC ${model.geoCvMetrics.rocAuc.toFixed(3)} · 95% CI ${formatCi95(model.geoCvMetrics)}`
                    : 'Out-of-fold metropolitan groups'}
                </div>
              </div>

              <div>
                <div className="font-sans text-slate-500">
                  Held-Out Test Accuracy
                </div>
                <div className="text-sm font-semibold text-blue-700 mt-0.5">
                  {formatModelAccuracyDisplay(model.testMetrics)} (
                  {formatCm(model.testMetrics)})
                </div>
                <div className="font-sans text-[11px] text-slate-500 mt-0.5">
                  95% Wilson CI {formatCi95(model.testMetrics)}
                </div>
              </div>

              <div>
                <div className="font-sans text-slate-500">
                  Test Precision / Recall / F1
                </div>
                <div className="text-sm font-semibold mt-0.5">
                  {(model.testMetrics.precision * 100).toFixed(1)}% /{' '}
                  {(model.testMetrics.recall * 100).toFixed(1)}% /{' '}
                  {(model.testMetrics.f1Score * 100).toFixed(1)}%
                </div>
                <div className="font-sans text-[11px] text-slate-500 mt-0.5">
                  AUC {model.testMetrics.rocAuc.toFixed(3)} · Brier{' '}
                  {model.testMetrics.brierScore.toFixed(3)}
                  {model.testMetrics.logLoss !== undefined
                    ? ` · LogLoss ${model.testMetrics.logLoss.toFixed(3)}`
                    : ''}
                </div>
              </div>

              <div>
                <div className="font-sans text-slate-500">
                  Cross-Split Separation &amp; IoU
                </div>
                <div className="text-sm font-semibold mt-0.5">
                  ≥ {model.minTrainTestDistanceKm.toFixed(1)} km (0 shared AOIs)
                </div>
                {model.pixelIoUMetrics && (
                  <div className="font-sans text-[11px] text-slate-500 mt-0.5">
                    Test IoU:{' '}
                    {model.pixelIoUMetrics.baselineTest.meanChangedSampleIoU.toFixed(
                      3
                    )}{' '}
                    →{' '}
                    {model.pixelIoUMetrics.improvedTest.meanChangedSampleIoU.toFixed(
                      3
                    )}
                    {model.pixelIoUMetrics.improvedTrainVal
                      ? ` (Non-Test ${model.pixelIoUMetrics.improvedTrainVal.meanChangedSampleIoU.toFixed(3)})`
                      : ''}
                  </div>
                )}
              </div>
            </div>

            {candidates.length > 0 && improvedTest && (
              <div className="overflow-x-auto pt-3 border-t border-slate-100">
                <div className="text-xs font-semibold text-slate-900 mb-2">
                  Baseline (v3.0) vs. Improved (v4.0) Candidate Models Across Identical Leak-Free Protocol (Untouched Held-Out Test n ={' '}
                  {improvedTest.sampleCount})
                </div>
                <table className="w-full text-left border-collapse text-xs">
                  <thead>
                    <tr className="border-b border-slate-200 text-slate-500">
                      <th className="py-2 pr-3 font-medium">Model Architecture</th>
                      <th className="py-2 px-3 font-medium text-right">
                        5-Fold Geo-CV Acc / F1
                      </th>
                      <th className="py-2 px-3 font-medium text-right">
                        Val F1 (n=12)
                      </th>
                      <th className="py-2 px-3 font-medium text-right">
                        Test Accuracy (95% CI)
                      </th>
                      <th className="py-2 px-3 font-medium text-right">
                        Precision / Recall
                      </th>
                      <th className="py-2 px-3 font-medium text-right">
                        Test F1 / AUC
                      </th>
                      <th className="py-2 px-3 font-medium text-right">
                        Brier / LogLoss
                      </th>
                      <th className="py-2 pl-3 font-medium">
                        Test Confusion Matrix
                      </th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100 font-mono tabular-nums">
                    {candidates.map((c) => {
                      const isSelected = c.selectedByValidation;
                      return (
                        <tr
                          key={c.modelId}
                          className={
                            isSelected
                              ? 'bg-blue-50/40 font-medium text-slate-900'
                              : 'text-slate-700'
                          }
                        >
                          <td className="py-2 pr-3 font-sans">
                            <div>
                              {c.modelLabel}
                              {isSelected ? ' — Selected' : ''}
                            </div>
                          </td>
                          <td className="py-2 px-3 text-right">
                            {c.geoCvMetrics
                              ? `${formatModelAccuracyDisplay(
                                  c.geoCvMetrics
                                )} / ${(c.geoCvMetrics.f1Score * 100).toFixed(1)}%`
                              : '—'}
                          </td>
                          <td
                            className={`py-2 px-3 text-right ${
                              isSelected ? 'text-blue-700' : ''
                            }`}
                          >
                            {(c.valMetrics.f1Score * 100).toFixed(1)}%
                          </td>
                          <td
                            className={`py-2 px-3 text-right ${
                              isSelected ? 'text-blue-700 font-semibold' : ''
                            }`}
                          >
                            {formatModelAccuracyDisplay(c.testMetrics)}{' '}
                            <span className="text-[11px] font-normal text-slate-500">
                              {formatCi95(c.testMetrics)}
                            </span>
                          </td>
                          <td className="py-2 px-3 text-right">
                            {(c.testMetrics.precision * 100).toFixed(1)}% /{' '}
                            {(c.testMetrics.recall * 100).toFixed(1)}%
                          </td>
                          <td
                            className={`py-2 px-3 text-right ${
                              isSelected ? 'font-semibold' : ''
                            }`}
                          >
                            {(c.testMetrics.f1Score * 100).toFixed(1)}% /{' '}
                            {c.testMetrics.rocAuc.toFixed(3)}
                          </td>
                          <td className="py-2 px-3 text-right text-slate-700">
                            {c.testMetrics.brierScore.toFixed(3)} /{' '}
                            {c.testMetrics.logLoss !== undefined
                              ? c.testMetrics.logLoss.toFixed(3)
                              : '—'}
                          </td>
                          <td className="py-2 pl-3 text-slate-600">
                            {formatCm(c.testMetrics)}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}

            {/* Probability Calibration & Reliability Diagram */}
            {model.calibration && (
              <div className="pt-3 border-t border-slate-100 flex flex-col gap-4">
                <div>
                  <div className="text-xs font-semibold text-slate-900">
                    Probability Calibration &amp; Reliability Evaluation (Leak-Free Non-Test Fitting)
                  </div>
                  <p className="text-xs text-slate-500 mt-0.5">
                    {model.calibration.fittedOn} — Held-Out Test (n ={' '}
                    {model.testMetrics.sampleCount}) was strictly excluded from calibration fitting.
                  </p>
                </div>

                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 items-start">
                  {/* Calibration Method Benchmark Table */}
                  <div className="overflow-x-auto">
                    <div className="text-[11px] font-semibold text-slate-700 mb-1.5">
                      Nested Out-of-Fold Calibration Benchmark (Non-Test n = 59)
                    </div>
                    <table className="w-full text-left border-collapse text-xs">
                      <thead>
                        <tr className="border-b border-slate-200 text-slate-500">
                          <th className="py-1.5 pr-2 font-medium">Method</th>
                          <th className="py-1.5 px-2 font-medium text-right">OOF Acc</th>
                          <th className="py-1.5 px-2 font-medium text-right">Brier ↓</th>
                          <th className="py-1.5 px-2 font-medium text-right">Log Loss ↓</th>
                          <th className="py-1.5 pl-2 font-medium text-right">ECE ↓</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-100 font-mono tabular-nums">
                        {model.calibration.benchmarkOnNonTest.map((b) => (
                          <tr
                            key={b.method}
                            className={
                              b.selected
                                ? 'bg-blue-50/40 font-medium text-slate-900'
                                : 'text-slate-700'
                            }
                          >
                            <td className="py-1.5 pr-2 font-sans">
                              {b.label}
                              {b.selected ? ' — Selected' : ''}
                            </td>
                            <td className="py-1.5 px-2 text-right">
                              {(b.nonTestAccuracy * 100).toFixed(1)}%
                            </td>
                            <td className="py-1.5 px-2 text-right">
                              {b.nonTestBrierScore.toFixed(4)}
                            </td>
                            <td
                              className={`py-1.5 px-2 text-right ${
                                b.selected ? 'text-blue-700 font-semibold' : ''
                              }`}
                            >
                              {b.nonTestLogLoss.toFixed(4)}
                            </td>
                            <td className="py-1.5 pl-2 text-right">
                              {b.nonTestEce.toFixed(4)}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                    <div className="text-[11px] font-mono tabular-nums text-slate-500 mt-1.5">
                      {model.calibration.method === 'platt_sigmoid'
                        ? `Fitted Platt mapping: P_cal = σ(${model.calibration.plattSlope.toFixed(3)} · z ${
                            model.calibration.plattIntercept >= 0 ? '+' : '−'
                          } ${Math.abs(model.calibration.plattIntercept).toFixed(3)})`
                        : `Selected calibration: ${model.calibration.method}`}{' '}
                      · Test ECE:{' '}
                      {model.testMetrics.ece !== undefined
                        ? model.testMetrics.ece.toFixed(3)
                        : '—'}
                    </div>
                  </div>

                  {/* Reliability Diagram (5 Probability Bins) */}
                  <div>
                    <div className="flex items-center justify-between text-[11px] mb-1.5">
                      <span className="font-semibold text-slate-700">
                        Reliability Diagram (Non-Test OOF, n = {model.calibration.fittedSampleCount})
                      </span>
                      <span className="text-slate-500 font-mono tabular-nums">
                        Blue: Mean Predicted P · Emerald: Empirical Pos %
                      </span>
                    </div>
                    <div className="flex flex-col gap-2 border border-slate-200 rounded-lg p-3 bg-slate-50/50">
                      {model.calibration.nonTestReliabilityAfter.map((bin) => {
                        const predPct = bin.count > 0 ? bin.meanPredictedProb * 100 : 0;
                        const empPct = bin.count > 0 ? bin.empiricalPositiveRate * 100 : 0;
                        return (
                          <div
                            key={bin.binIndex}
                            className="grid grid-cols-12 items-center gap-2 text-[11px] font-mono tabular-nums"
                          >
                            <div className="col-span-3 text-slate-600">
                              {Math.round(bin.binStart * 100)}–
                              {Math.round(bin.binEnd * 100)}% (n={bin.count})
                            </div>
                            <div className="col-span-6 flex flex-col gap-1">
                              <div className="h-1.5 w-full bg-slate-200 rounded-xs overflow-hidden">
                                <div
                                  className="h-full bg-blue-600"
                                  style={{ width: `${Math.min(100, Math.max(0, predPct))}%` }}
                                />
                              </div>
                              <div className="h-1.5 w-full bg-slate-200 rounded-xs overflow-hidden">
                                <div
                                  className="h-full bg-emerald-600"
                                  style={{ width: `${Math.min(100, Math.max(0, empPct))}%` }}
                                />
                              </div>
                            </div>
                            <div className="col-span-3 text-right text-slate-700">
                              {bin.count > 0
                                ? `${predPct.toFixed(0)}% vs ${empPct.toFixed(0)}%`
                                : 'empty'}
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                </div>
              </div>
            )}

            <div className="grid grid-cols-1 md:grid-cols-3 gap-5 pt-3 border-t border-slate-100">
              <div>
                <div className="font-semibold text-slate-900 mb-1">
                  Why Calibrated L2 Logistic Regression Beat Tree Ensembles
                </div>
                <p>
                  {model.baselineComparison?.selectionRationale ??
                    model.leakagePrevention}{' '}
                  With 8 physically monotone spectral-structural features across 59
                  non-test tiles, regularized linear logits generalize smoothly
                  across unseen arid and temperate cities, whereas axis-aligned tree
                  splits (RF / GBDT) suffer step-function variance. A deep Siamese
                  CNN/U-Net was intentionally not trained from scratch because 59
                  CPU-processed tiles are orders of magnitude too few for
                  convolutional generalization without overfitting.
                </p>
              </div>

              <div>
                <div className="font-semibold text-slate-900 mb-1">
                  Small-Sample Uncertainty on Held-Out Test (n ={' '}
                  {model.testMetrics.sampleCount})
                </div>
                <p>
                  Because all 12 preloaded demonstration cities are strictly
                  isolated in the untouched Held-Out Geographic Test Split (≥{' '}
                  {model.minTrainTestDistanceKm.toFixed(1)} km from every
                  training and validation AOI), each test pair represents{' '}
                  <span className="font-mono tabular-nums">8.33</span>{' '}
                  percentage points, yielding a 95% Wilson confidence interval
                  of <span className="font-mono tabular-nums">{formatCi95(model.testMetrics)}</span>{' '}
                  at <span className="font-mono tabular-nums">{formatModelAccuracyDisplay(model.testMetrics)}</span>.
                  Out-of-fold 5-Fold Geographic Group CV across all{' '}
                  <span className="font-mono tabular-nums">{model.trainMetrics.sampleCount}</span>{' '}
                  training tiles confirms consistent out-of-sample gain (
                  <span className="font-mono tabular-nums">
                    {model.geoCvMetrics ? formatModelAccuracyDisplay(model.geoCvMetrics) : '—'}
                  </span>
                  ).
                </p>
              </div>

              <div>
                <div className="font-semibold text-slate-900 mb-1">
                  Expanding &amp; Validating Additional SpaceNet 7 Samples
                </div>
                <p>
                  To add more labelled pairs via{' '}
                  <span className="font-mono">scripts/build_spacenet7_dataset.ts</span>,
                  stream additional AOIs or non-overlapping 512×512 quadrants from{' '}
                  <span className="font-mono">s3://spacenet-dataset/spacenet/SN7_buildings/train/</span>{' '}
                  (public AWS Open Data; note that{' '}
                  <span className="font-mono">test_public/</span> omits ground-truth
                  footprints). Validate every candidate pair for{' '}
                  <span className="font-mono">UDM</span> cloud-free coverage ≥ 98%,
                  verify <span className="font-mono">labels_match_pix</span> polygon
                  areas against physical thresholds, and assign all tiles within 150 km
                  to the same metropolitan split group so geographic leakage remains
                  impossible.
                </p>
              </div>
            </div>

            <div>
              <a
                href="/api/model/dataset.csv"
                download="spacenet7_feature_dataset.csv"
                className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-slate-800 bg-slate-100 hover:bg-slate-200 rounded-lg transition-colors"
              >
                <Download className="w-3.5 h-3.5" />
                Download Train / Val / Test Features (CSV)
              </a>
            </div>
          </div>
        ) : (
          <div className="px-6 pb-6 pt-3 border-t border-slate-200 text-xs text-slate-600">
            Model Accuracy: Not evaluated (no calibrated model artifact loaded).
          </div>
        )}
      </details>
    </section>
  );
};
