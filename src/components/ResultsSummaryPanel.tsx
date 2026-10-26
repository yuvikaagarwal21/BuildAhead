import React from 'react';
import { FileJson, FileSpreadsheet, ImageDown } from 'lucide-react';
import type {
  ChangeDetectionResult,
  ModelArtifact,
  PreprocessingConfig,
  SampleManifestEntry,
} from '../types/satellite.ts';
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
            onClick={() => downloadComparisonCsv(sample, config, result)}
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

      {/* 4 Simple Key Metrics */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6 pb-5 border-b border-slate-200">
        <div>
          <div className="text-xs text-slate-500">Predicted Category</div>
          <div
            className={`text-base font-semibold mt-1 ${
              cls.predictedPositive ? 'text-amber-700' : 'text-slate-900'
            }`}
          >
            {cls.predictedCategory}
          </div>
          <div className="text-xs font-mono tabular-nums text-slate-600 mt-1">
            {cls.confidenceScore !== null
              ? `Model confidence: ${(cls.confidenceScore * 100).toFixed(1)}%`
              : 'Rule-based screening'}
          </div>
        </div>

        <div>
          <div className="text-xs text-slate-500">Candidate Changed Pixels</div>
          <div className="text-2xl font-mono font-semibold tabular-nums text-slate-900 mt-1">
            {result.candidateChangedPct.toFixed(2)}%
          </div>
          <div className="text-xs font-mono tabular-nums text-slate-600 mt-1">
            {result.candidatePixelCount.toLocaleString()} px ·{' '}
            {result.regions.length} highlighted regions
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
            labelled footprints
          </div>
        </div>

        <div>
          <div className="text-xs text-slate-500">Building Coverage</div>
          <div className="text-2xl font-mono font-semibold tabular-nums text-slate-900 mt-1">
            {gt.earlierCoveragePct.toFixed(1)}% → {gt.laterCoveragePct.toFixed(1)}%
          </div>
          <div className="text-xs font-mono tabular-nums text-slate-600 mt-1">
            +{gt.newConstructionAreaM2.toLocaleString()} m² new footprint area
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
            Seasonal and shadow filtering suppressed{' '}
            <span className="font-mono tabular-nums font-medium text-slate-800">
              {result.suppressedEnvPct.toFixed(2)}%
            </span>{' '}
            of non-structural pixel differences across the tile.
          </p>
        </div>

        <div>
          <h3 className="text-xs font-semibold text-slate-900 mb-1">
            Important Limitations
          </h3>
          <p>
            Not every pixel difference indicates construction: seasonal crops, bare
            soil grading, shadows, or roof glare can cause visible changes at ~4.8 m/px
            resolution. In addition, detecting historical changes between two dates
            does not predict future construction.
          </p>
        </div>
      </div>
    </section>
  );
};
