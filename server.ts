import fs from 'node:fs';
import path from 'node:path';
import express from 'express';
import { PNG } from 'pngjs';
import {
  DEFAULT_PREPROCESSING_CONFIG,
  runChangeDetectionPipeline,
} from './src/lib/changeDetection.ts';
import type {
  DatasetManifest,
  ModelArtifact,
  PreprocessingConfig,
  RawImageBuffer,
} from './src/types/satellite.ts';

const PORT = 3000;
const ROOT_DIR = process.cwd();

function loadJsonFile<T>(relPath: string): T | null {
  try {
    const full = path.join(ROOT_DIR, relPath);
    if (!fs.existsSync(full)) return null;
    return JSON.parse(fs.readFileSync(full, 'utf-8')) as T;
  } catch {
    return null;
  }
}

function readPngBuffer(relPublicPath: string): RawImageBuffer {
  const cleaned = relPublicPath.replace(/^\//, '');
  const full = path.join(ROOT_DIR, 'public', cleaned);
  const raw = fs.readFileSync(full);
  const decoded = PNG.sync.read(raw);
  return {
    width: decoded.width,
    height: decoded.height,
    data: new Uint8ClampedArray(decoded.data),
  };
}

async function startServer() {
  const app = express();
  app.use(express.json({ limit: '2mb' }));

  app.get('/api/health', (_req, res) => {
    const manifest = loadJsonFile<DatasetManifest>('public/data/spacenet7/manifest.json');
    const model = loadJsonFile<ModelArtifact>('src/data/modelArtifact.json');
    res.json({
      ok: true,
      datasetLoaded: Boolean(manifest && manifest.samples.length > 0),
      sampleCount: manifest?.samples.length ?? 0,
      calibratedModelLoaded: Boolean(model?.isCalibratedModel),
    });
  });

  app.get('/api/dataset/manifest', (_req, res) => {
    const manifest = loadJsonFile<DatasetManifest>('public/data/spacenet7/manifest.json');
    if (!manifest) {
      res.status(404).json({ error: 'Dataset manifest not found.' });
      return;
    }
    res.json(manifest);
  });

  app.get('/api/model/artifact', (_req, res) => {
    const model = loadJsonFile<ModelArtifact>('src/data/modelArtifact.json');
    if (!model) {
      res.status(404).json({ error: 'Model artifact not found.' });
      return;
    }
    res.json(model);
  });

  app.get('/api/model/dataset.csv', (_req, res) => {
    const csvPath = path.join(ROOT_DIR, 'src/data/spacenet7_feature_dataset.csv');
    if (!fs.existsSync(csvPath)) {
      res.status(404).send('Feature dataset CSV not found.');
      return;
    }
    res.setHeader('Content-Type', 'text/csv; charset=utf-8');
    res.setHeader(
      'Content-Disposition',
      'attachment; filename="spacenet7_feature_dataset.csv"'
    );
    res.send(fs.readFileSync(csvPath, 'utf-8'));
  });

  app.post('/api/analyze/:sampleId', (req, res) => {
    const manifest = loadJsonFile<DatasetManifest>('public/data/spacenet7/manifest.json');
    const model = loadJsonFile<ModelArtifact>('src/data/modelArtifact.json');
    if (!manifest) {
      res.status(500).json({ error: 'Dataset manifest unavailable.' });
      return;
    }

    const sample = manifest.samples.find((s) => s.id === req.params.sampleId);
    if (!sample) {
      res.status(404).json({ error: `Unknown sample ID: ${req.params.sampleId}` });
      return;
    }

    try {
      const earlier = readPngBuffer(sample.earlierImageUrl);
      const later = readPngBuffer(sample.laterImageUrl);
      const userConfig = (req.body?.config ?? {}) as Partial<PreprocessingConfig>;
      const config: PreprocessingConfig = {
        ...DEFAULT_PREPROCESSING_CONFIG,
        ...userConfig,
      };

      const result = runChangeDetectionPipeline({
        earlier,
        later,
        spatial: sample.spatial,
        config,
        model,
      });

      res.json({
        sampleId: sample.id,
        validation: result.validation,
        alignment: result.alignment,
        candidatePixelCount: result.candidatePixelCount,
        candidateChangedPct: result.candidateChangedPct,
        candidateAreaM2: result.candidateAreaM2,
        rawDiffPixelCount: result.rawDiffPixelCount,
        rawDiffPct: result.rawDiffPct,
        suppressedEnvPixelCount: result.suppressedEnvPixelCount,
        suppressedEnvPct: result.suppressedEnvPct,
        regionCount: result.regions.length,
        topRegions: result.regions.slice(0, 15),
        extractedFeatures: result.extractedFeatures,
        classification: result.classification,
        processingDurationMs: result.processingDurationMs,
      });
    } catch (err: any) {
      res.status(500).json({ error: err?.message ?? 'Failed to process sample pair.' });
    }
  });

  const isProd = process.env.NODE_ENV === 'production' && fs.existsSync(path.join(ROOT_DIR, 'dist'));

  if (!isProd) {
    const { createServer: createViteServer } = await import('vite');
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distDir = path.join(ROOT_DIR, 'dist');
    app.use(express.static(distDir));
    app.get('*', (_req, res) => {
      res.sendFile(path.join(distDir, 'index.html'));
    });
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`BuildAhead server listening on http://0.0.0.0:${PORT}`);
  });
}

startServer();
