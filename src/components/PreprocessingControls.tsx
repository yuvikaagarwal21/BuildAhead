import React from 'react';
import { RotateCcw } from 'lucide-react';
import { DEFAULT_PREPROCESSING_CONFIG } from '../lib/changeDetection.ts';
import type { AlignmentResult, PreprocessingConfig } from '../types/satellite.ts';

interface PreprocessingControlsProps {
  config: PreprocessingConfig;
  onChange: (next: PreprocessingConfig) => void;
  showOriginalImages: boolean;
  onToggleOriginalImages: (showOrig: boolean) => void;
  alignment?: AlignmentResult;
}

export const PreprocessingControls: React.FC<PreprocessingControlsProps> = ({
  config,
  onChange,
  showOriginalImages,
  onToggleOriginalImages,
  alignment,
}) => {
  const update = <K extends keyof PreprocessingConfig>(
    key: K,
    value: PreprocessingConfig[K]
  ) => {
    onChange({ ...config, [key]: value });
  };

  const filterEnvNoise = config.suppressVegetation && config.suppressShadows;

  return (
    <section className="bg-white border border-slate-200 rounded-xl p-5 flex flex-col gap-4">
      <div className="flex flex-wrap items-center justify-between gap-3 pb-3 border-b border-slate-200">
        <div>
          <h2 className="text-sm font-semibold text-slate-900">
            Image Cleaning &amp; Detection Settings
          </h2>
          <p className="text-xs text-slate-500 mt-0.5">
            Image cleaning improves visual clarity and reduces noise, but does not prove construction occurred.
          </p>
        </div>

        <div className="flex items-center gap-2">
          <div className="flex items-center gap-1 p-1 bg-slate-100 rounded-lg">
            <button
              type="button"
              onClick={() => onToggleOriginalImages(false)}
              className={`px-3 py-1 text-xs font-medium rounded-md transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
                !showOriginalImages
                  ? 'bg-white text-slate-900 shadow-xs'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              Cleaned View
            </button>
            <button
              type="button"
              onClick={() => onToggleOriginalImages(true)}
              className={`px-3 py-1 text-xs font-medium rounded-md transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
                showOriginalImages
                  ? 'bg-white text-slate-900 shadow-xs'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              Original Images
            </button>
          </div>

          <button
            type="button"
            onClick={() => {
              onChange(DEFAULT_PREPROCESSING_CONFIG);
              onToggleOriginalImages(false);
            }}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-slate-700 bg-slate-100 hover:bg-slate-200 rounded-lg transition-colors whitespace-nowrap shrink-0 cursor-pointer"
          >
            <RotateCcw className="w-3.5 h-3.5" />
            Reset
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-5 items-end">
        {/* Denoising */}
        <div className="flex flex-col gap-1.5">
          <label htmlFor="select-denoise" className="text-xs font-medium text-slate-700">
            Denoising
          </label>
          <select
            id="select-denoise"
            value={config.denoiseMethod}
            onChange={(e) =>
              update(
                'denoiseMethod',
                e.target.value as PreprocessingConfig['denoiseMethod']
              )
            }
            className="w-full px-3 py-2 text-xs text-slate-800 bg-slate-50 border border-slate-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-600"
          >
            <option value="bilateral">Edge-Preserving Smooth</option>
            <option value="median3x3">3×3 Median Denoise</option>
            <option value="none">Off</option>
          </select>
        </div>

        {/* Contrast */}
        <div className="flex flex-col gap-1.5">
          <label htmlFor="select-contrast" className="text-xs font-medium text-slate-700">
            Contrast Enhancement
          </label>
          <select
            id="select-contrast"
            value={config.contrastMethod}
            onChange={(e) =>
              update(
                'contrastMethod',
                e.target.value as PreprocessingConfig['contrastMethod']
              )
            }
            className="w-full px-3 py-2 text-xs text-slate-800 bg-slate-50 border border-slate-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-600"
          >
            <option value="linear_percentile">Auto Contrast Stretch</option>
            <option value="clahe">Adaptive Contrast (CLAHE)</option>
            <option value="none">Off</option>
          </select>
        </div>

        {/* Threshold */}
        <div className="flex flex-col gap-1.5">
          <div className="flex items-center justify-between text-xs">
            <label htmlFor="slider-thresh" className="font-medium text-slate-700">
              Change Threshold
            </label>
            <span className="font-mono tabular-nums text-slate-500">
              {config.changeThreshold}
            </span>
          </div>
          <input
            id="slider-thresh"
            type="range"
            min={16}
            max={54}
            step={2}
            value={config.changeThreshold}
            onChange={(e) => update('changeThreshold', Number(e.target.value))}
            className="w-full accent-blue-600 cursor-pointer py-1.5"
          />
        </div>

        {/* Alignment & Seasonal Filtering */}
        <div className="flex flex-col gap-2">
          <label className="flex items-center justify-between gap-2 text-xs text-slate-700 cursor-pointer">
            <span>
              Auto-Align Images
              {alignment ? ` (${alignment.dx}, ${alignment.dy} px)` : ''}
            </span>
            <input
              type="checkbox"
              checked={config.enableAlignment}
              onChange={(e) => update('enableAlignment', e.target.checked)}
              className="w-4 h-4 accent-blue-600 rounded cursor-pointer"
            />
          </label>

          <label className="flex items-center justify-between gap-2 text-xs text-slate-700 cursor-pointer">
            <span>Filter Seasonal &amp; Shadow Noise</span>
            <input
              type="checkbox"
              checked={filterEnvNoise}
              onChange={(e) => {
                onChange({
                  ...config,
                  suppressVegetation: e.target.checked,
                  suppressShadows: e.target.checked,
                  radiometricMatch: e.target.checked,
                });
              }}
              className="w-4 h-4 accent-blue-600 rounded cursor-pointer"
            />
          </label>
        </div>
      </div>
    </section>
  );
};
