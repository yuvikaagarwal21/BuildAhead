import React, { useEffect, useMemo, useState } from 'react';
import { Download } from 'lucide-react';
import {
  DEFAULT_PREPROCESSING_CONFIG,
  runChangeDetectionPipeline,
} from './lib/changeDetection.ts';
import { downloadResultsJson } from './lib/exporters.ts';
import { ModelAndDatasetSections } from './components/ModelAndDatasetSections.tsx';
import { PreprocessingControls } from './components/PreprocessingControls.tsx';
import { ResultsSummaryPanel } from './components/ResultsSummaryPanel.tsx';
import { SampleSelectorBar } from './components/SampleSelectorBar.tsx';
import { SatelliteCanvas } from './components/SatelliteCanvas.tsx';
import type {
  BuildingFootprintPolygon,
  DatasetManifest,
  ModelArtifact,
  PreprocessingConfig,
  RawImageBuffer,
} from './types/satellite.ts';

function loadImageAsRawBuffer(url: string): Promise<RawImageBuffer> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => {
      const canvas = document.createElement('canvas');
      canvas.width = img.naturalWidth;
      canvas.height = img.naturalHeight;
      const ctx = canvas.getContext('2d');
      if (!ctx) {
        reject(new Error('2D canvas context unavailable'));
        return;
      }
      ctx.drawImage(img, 0, 0);
      const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
      resolve({
        width: canvas.width,
        height: canvas.height,
        data: imageData.data,
      });
    };
    img.onerror = () => reject(new Error(`Failed to load satellite image: ${url}`));
    img.src = url;
  });
}

export default function App() {
  const [manifest, setManifest] = useState<DatasetManifest | null>(null);
  const [model, setModel] = useState<ModelArtifact | null>(null);
  const [selectedId, setSelectedId] = useState<string>('');
  const [config, setConfig] = useState<PreprocessingConfig>(
    DEFAULT_PREPROCESSING_CONFIG
  );

  const [earlierRaw, setEarlierRaw] = useState<RawImageBuffer | null>(null);
  const [laterRaw, setLaterRaw] = useState<RawImageBuffer | null>(null);
  const [footprints, setFootprints] = useState<BuildingFootprintPolygon[]>([]);
  const [loadingSample, setLoadingSample] = useState<boolean>(true);
  const [loadError, setLoadError] = useState<string | null>(null);

  const [showOriginalImages, setShowOriginalImages] = useState<boolean>(false);
  const [showFootprints, setShowFootprints] = useState<boolean>(false);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const [manifestRes, modelRes] = await Promise.all([
          fetch('/data/spacenet7/manifest.json'),
          fetch('/api/model/artifact').catch(() => null),
        ]);

        if (!manifestRes.ok) {
          throw new Error(`Dataset manifest HTTP ${manifestRes.status}`);
        }
        const manifestData = (await manifestRes.json()) as DatasetManifest;
        let modelData: ModelArtifact | null = null;
        if (modelRes && modelRes.ok) {
          modelData = (await modelRes.json()) as ModelArtifact;
        }

        if (!cancelled) {
          setManifest(manifestData);
          setModel(modelData);
          if (manifestData.samples.length > 0) {
            setSelectedId(manifestData.samples[0].id);
          }
        }
      } catch (err: any) {
        if (!cancelled) {
          setLoadError(err?.message ?? 'Unable to load preloaded dataset manifest.');
          setLoadingSample(false);
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const selectedSample = useMemo(
    () => manifest?.samples.find((s) => s.id === selectedId) ?? null,
    [manifest, selectedId]
  );

  useEffect(() => {
    if (!selectedSample) return;
    let cancelled = false;
    setLoadingSample(true);
    setLoadError(null);

    (async () => {
      try {
        const [eBuf, lBuf, fpData] = await Promise.all([
          loadImageAsRawBuffer(selectedSample.earlierImageUrl),
          loadImageAsRawBuffer(selectedSample.laterImageUrl),
          selectedSample.groundTruth.footprintsUrl
            ? fetch(selectedSample.groundTruth.footprintsUrl).then((r) =>
                r.ok ? r.json() : []
              )
            : Promise.resolve([]),
        ]);

        if (!cancelled) {
          setEarlierRaw(eBuf);
          setLaterRaw(lBuf);
          setFootprints(fpData as BuildingFootprintPolygon[]);
          setLoadingSample(false);
        }
      } catch (err: any) {
        if (!cancelled) {
          setLoadError(err?.message ?? 'Failed to decode satellite image pair.');
          setLoadingSample(false);
        }
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [selectedSample]);

  const detectionResult = useMemo(() => {
    if (!earlierRaw || !laterRaw || !selectedSample) return null;
    return runChangeDetectionPipeline({
      earlier: earlierRaw,
      later: laterRaw,
      spatial: selectedSample.spatial,
      config,
      model,
      groundTruthFootprints: footprints,
    });
  }, [earlierRaw, laterRaw, selectedSample, config, model, footprints]);

  return (
    <div className="min-h-screen flex flex-col bg-slate-50 text-slate-900">
      {/* Clean 3-Zone Header */}
      <header className="sticky top-0 z-30 bg-white/95 backdrop-blur-xs border-b border-slate-200 px-6 py-3.5">
        <div className="max-w-[1200px] mx-auto flex items-center justify-between gap-8">
          <a
            href="#compare"
            className="text-lg font-bold tracking-tight text-slate-900 font-display whitespace-nowrap shrink-0"
          >
            BuildAhead
          </a>

          <nav className="hidden sm:flex items-center gap-6 text-sm font-medium text-slate-600">
            <a
              href="#compare"
              className="hover:text-slate-900 transition-colors whitespace-nowrap shrink-0"
            >
              Compare
            </a>
            <a
              href="#results"
              className="hover:text-slate-900 transition-colors whitespace-nowrap shrink-0"
            >
              Results
            </a>
            <a
              href="#about-dataset"
              className="hover:text-slate-900 transition-colors whitespace-nowrap shrink-0"
            >
              Dataset &amp; Model
            </a>
          </nav>

          <div className="flex items-center shrink-0">
            <button
              type="button"
              disabled={!selectedSample || !detectionResult}
              onClick={() => {
                if (selectedSample && detectionResult) {
                  downloadResultsJson(selectedSample, config, detectionResult, model);
                }
              }}
              className="inline-flex items-center gap-2 px-4 py-2 text-xs font-medium text-white bg-slate-900 hover:bg-slate-800 disabled:opacity-40 rounded-lg transition-colors whitespace-nowrap shrink-0 cursor-pointer"
            >
              <Download className="w-3.5 h-3.5" />
              Export JSON
            </button>
          </div>
        </div>
      </header>

      <main
        id="compare"
        className="flex-1 w-full max-w-[1200px] mx-auto px-4 sm:px-6 py-7 flex flex-col gap-6"
      >
        {/* Simple Title & Intro */}
        <section className="max-w-3xl">
          <h1 className="text-2xl sm:text-3xl font-bold tracking-tight text-slate-900 text-balance">
            Satellite-Based Construction Change Detection
          </h1>
          <p className="text-sm text-slate-600 leading-relaxed mt-1.5">
            Compare earlier and later SpaceNet 7 satellite images of the same urban area
            to highlight visible changes and classify likely building-related
            construction. Detects historical image changes only—does not predict future
            construction.
          </p>
        </section>

        {/* 1. Simple Dropdown Sample Selector */}
        {manifest && (
          <SampleSelectorBar
            samples={manifest.samples}
            selectedId={selectedId}
            onSelect={setSelectedId}
          />
        )}

        {loadError && (
          <div className="p-4 bg-rose-50 border border-rose-200 rounded-xl text-sm text-rose-800">
            {loadError}
          </div>
        )}

        {/* 2. Side-by-Side Earlier, Later, and Change Map */}
        <section className="flex flex-col gap-3">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <h2 className="text-sm font-semibold text-slate-800">
              Satellite Image Comparison
            </h2>
            <label className="inline-flex items-center gap-2 text-xs font-medium text-slate-700 cursor-pointer">
              <input
                type="checkbox"
                checked={showFootprints}
                onChange={(e) => setShowFootprints(e.target.checked)}
                className="w-4 h-4 accent-emerald-600 rounded cursor-pointer"
              />
              <span>
                Overlay Ground-Truth New Buildings (+
                {selectedSample?.groundTruth.newConstructionCount ?? 0})
              </span>
            </label>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <SatelliteCanvas
              title="Earlier Satellite Image"
              subtitle={selectedSample?.earlierDate ?? '—'}
              buffer={
                loadingSample
                  ? null
                  : showOriginalImages
                  ? earlierRaw
                  : detectionResult?.earlierCleaned ?? null
              }
              caption={
                showOriginalImages ? 'Original raw image' : 'Cleaned earlier image'
              }
            />

            <SatelliteCanvas
              title="Later Satellite Image"
              subtitle={selectedSample?.laterDate ?? '—'}
              buffer={
                loadingSample
                  ? null
                  : showOriginalImages
                  ? laterRaw
                  : detectionResult?.laterCleaned ?? null
              }
              footprints={footprints}
              showFootprints={showFootprints}
              footprintFilter="new_only"
              caption={
                showOriginalImages
                  ? 'Original raw image'
                  : 'Aligned & cleaned later image'
              }
            />

            <SatelliteCanvas
              title="Change-Detection Map"
              subtitle={
                detectionResult
                  ? `${detectionResult.candidateChangedPct.toFixed(2)}% changed`
                  : '—'
              }
              buffer={loadingSample ? null : detectionResult?.changeMapOverlay ?? null}
              footprints={footprints}
              showFootprints={showFootprints}
              footprintFilter="new_only"
              caption="Amber: candidate structural changes · Teal: filtered seasonal/shadow shifts"
            />
          </div>
        </section>

        {/* 3. Compact Image-Cleaning & Threshold Controls */}
        <PreprocessingControls
          config={config}
          onChange={setConfig}
          showOriginalImages={showOriginalImages}
          onToggleOriginalImages={setShowOriginalImages}
          alignment={detectionResult?.alignment}
        />

        {/* 4. Concise Results & Downloads */}
        {selectedSample && detectionResult && detectionResult.validation.valid && (
          <ResultsSummaryPanel
            sample={selectedSample}
            config={config}
            result={detectionResult}
            model={model}
          />
        )}

        {/* 5. Expandable Dataset Manifest & Model Details */}
        <ModelAndDatasetSections
          manifest={manifest}
          model={model}
          selectedSampleId={selectedId}
          onSelectSample={setSelectedId}
        />
      </main>

      <footer className="border-t border-slate-200 bg-white py-4 px-6 mt-6">
        <div className="max-w-[1200px] mx-auto flex flex-col sm:flex-row items-center justify-between gap-2 text-xs text-slate-500">
          <span>BuildAhead — Satellite-Based Construction Change Detection</span>
          <span>Data: Official SpaceNet 7 Dataset (Planet Labs, CC-BY-4.0)</span>
        </div>
      </footer>
    </div>
  );
}
