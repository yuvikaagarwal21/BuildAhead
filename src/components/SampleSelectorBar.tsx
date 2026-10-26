import React from 'react';
import type { SampleManifestEntry } from '../types/satellite.ts';

interface SampleSelectorBarProps {
  samples: SampleManifestEntry[];
  selectedId: string;
  onSelect: (id: string) => void;
}

export const SampleSelectorBar: React.FC<SampleSelectorBarProps> = ({
  samples,
  selectedId,
  onSelect,
}) => {
  const current = samples.find((s) => s.id === selectedId) ?? samples[0];

  return (
    <section className="bg-white border border-slate-200 rounded-xl p-5">
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-5 items-center">
        <div className="lg:col-span-6 flex flex-col gap-1.5">
          <label
            htmlFor="sample-select"
            className="text-xs font-medium text-slate-600"
          >
            Preloaded Satellite Sample (SpaceNet 7)
          </label>
          <select
            id="sample-select"
            value={selectedId}
            onChange={(e) => onSelect(e.target.value)}
            className="w-full px-3.5 py-2.5 text-sm font-medium text-slate-900 bg-slate-50 border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-600 cursor-pointer"
          >
            {samples.map((s) => (
              <option key={s.id} value={s.id}>
                {s.regionLabel} ({s.earlierDate} to {s.laterDate}) —{' '}
                {s.groundTruth.groundTruthPositive
                  ? `+${s.groundTruth.newConstructionCount} new buildings`
                  : 'Minimal / no building change'}
              </option>
            ))}
          </select>
        </div>

        {current && (
          <div className="lg:col-span-6 grid grid-cols-3 gap-4 pt-3 lg:pt-0 border-t lg:border-t-0 lg:border-l border-slate-200 lg:pl-6">
            <div>
              <div className="text-xs text-slate-500">Earlier Date</div>
              <div className="text-base font-mono font-semibold tabular-nums text-slate-900 mt-0.5">
                {current.earlierDate}
              </div>
            </div>
            <div>
              <div className="text-xs text-slate-500">Later Date</div>
              <div className="text-base font-mono font-semibold tabular-nums text-slate-900 mt-0.5">
                {current.laterDate}
              </div>
            </div>
            <div>
              <div className="text-xs text-slate-500">Sample ID</div>
              <div className="text-base font-mono font-semibold tabular-nums text-slate-900 mt-0.5">
                {current.id}
              </div>
            </div>
          </div>
        )}
      </div>
    </section>
  );
};
