import React from 'react';
import { ChevronDown, Download, ExternalLink } from 'lucide-react';
import type { DatasetManifest, ModelArtifact } from '../types/satellite.ts';

interface ModelAndDatasetSectionsProps {
  manifest: DatasetManifest | null;
  model: ModelArtifact | null;
  selectedSampleId: string;
  onSelectSample: (id: string) => void;
}

export const ModelAndDatasetSections: React.FC<ModelAndDatasetSectionsProps> = ({
  manifest,
  model,
  selectedSampleId,
  onSelectSample,
}) => {
  return (
    <section
      id="about-dataset"
      className="flex flex-col gap-3"
    >
      {/* Collapsible Dataset Manifest */}
      <details className="group bg-white border border-slate-200 rounded-xl overflow-hidden">
        <summary className="px-6 py-4 flex items-center justify-between gap-4 cursor-pointer select-none list-none hover:bg-slate-50/70">
          <div>
            <h2 className="text-sm font-semibold text-slate-900">
              Dataset Manifest &amp; Attribution (SpaceNet 7)
            </h2>
            <p className="text-xs text-slate-500 mt-0.5">
              View all {manifest?.samples.length ?? 0} preloaded satellite image pairs, observation dates, ground-truth counts, and CC-BY-4.0 source attribution
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
                        {s.groundTruth.earlierCount} → {s.groundTruth.laterCount} (+
                        {s.groundTruth.newConstructionCount})
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

      {/* Collapsible Classification Model Details */}
      <details className="group bg-white border border-slate-200 rounded-xl overflow-hidden">
        <summary className="px-6 py-4 flex items-center justify-between gap-4 cursor-pointer select-none list-none hover:bg-slate-50/70">
          <div>
            <h2 className="text-sm font-semibold text-slate-900">
              Classification Model &amp; Validation Details
            </h2>
            <p className="text-xs text-slate-500 mt-0.5">
              How the logistic regression classifier was trained and tested on separate SpaceNet 7 areas
            </p>
          </div>
          <ChevronDown className="w-4 h-4 text-slate-500 transition-transform group-open:rotate-180 shrink-0" />
        </summary>

        {model && (
          <div className="px-6 pb-6 pt-3 border-t border-slate-200 flex flex-col gap-4 text-xs text-slate-600 leading-relaxed">
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 font-mono tabular-nums text-slate-800">
              <div>
                <div className="font-sans text-slate-500">Dataset Split</div>
                <div className="text-sm font-semibold mt-0.5">
                  {model.trainMetrics.sampleCount} train · {model.testMetrics.sampleCount}{' '}
                  held-out test
                </div>
              </div>
              <div>
                <div className="font-sans text-slate-500">Held-Out Test Accuracy</div>
                <div className="text-sm font-semibold mt-0.5">
                  {(model.testMetrics.accuracy * 100).toFixed(1)}% (TP{' '}
                  {model.testMetrics.confusionMatrix.tp}, TN{' '}
                  {model.testMetrics.confusionMatrix.tn}, FP{' '}
                  {model.testMetrics.confusionMatrix.fp}, FN{' '}
                  {model.testMetrics.confusionMatrix.fn})
                </div>
              </div>
              <div>
                <div className="font-sans text-slate-500">Minimum Tile Separation</div>
                <div className="text-sm font-semibold mt-0.5">
                  {model.minTrainTestDistanceKm.toFixed(1)} km (zero tile overlap)
                </div>
              </div>
            </div>

            <p>
              The classifier uses L2-regularized logistic regression on five
              image-derived features: filtered structural change percentage, bright
              non-vegetated surface gain, Sobel edge gain inside changed clusters,
              compact cluster density per km², and seasonal vegetation shift ratio.
            </p>

            <div>
              <a
                href="/api/model/dataset.csv"
                download="spacenet7_feature_dataset.csv"
                className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-slate-800 bg-slate-100 hover:bg-slate-200 rounded-lg transition-colors"
              >
                <Download className="w-3.5 h-3.5" />
                Download Training &amp; Test Features (CSV)
              </a>
            </div>
          </div>
        )}
      </details>
    </section>
  );
};
