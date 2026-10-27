import fs from 'node:fs';
import path from 'node:path';
import { fromArrayBuffer } from 'geotiff';
import { PNG } from 'pngjs';
import {
  BASELINE_V3_PREPROCESSING_CONFIG,
  computeBaselineFeaturesAndMasks,
  computeFeaturesAndMasks,
  DEFAULT_PREPROCESSING_CONFIG,
} from '../src/lib/changeDetection.ts';
import {
  BASELINE_FEATURE_KEYS,
  FEATURE_KEYS,
  trainValidateAndSelectBestModel,
} from '../src/lib/classifier.ts';
import {
  alignLaterImage,
  computeBinaryMaskIoU,
  rasterizeFootprintsToMask,
  validateImagePair,
} from '../src/lib/imageProcessing.ts';
import type {
  BuildingFootprintPolygon,
  DatasetManifest,
  DatasetSplitRecord,
  PixelChangeIoUMetrics,
  RawImageBuffer,
  SampleManifestEntry,
  SpatialMetadata,
} from '../src/types/satellite.ts';

const S3_BASE = 'https://spacenet-dataset.s3.amazonaws.com';
const PUBLIC_DATA_DIR = path.resolve(process.cwd(), 'public/data/spacenet7');
const SRC_DATA_DIR = path.resolve(process.cwd(), 'src/data');
const TILE_CACHE_DIR = path.resolve(process.cwd(), '.sn7_cache');

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

function mercatorToLatLon(x: number, y: number): { lat: number; lon: number } {
  const lon = (x / 20037508.34) * 180;
  let lat = (y / 20037508.34) * 180;
  lat = (180 / Math.PI) * (2 * Math.atan(Math.exp((lat * Math.PI) / 180)) - Math.PI / 2);
  return { lat: Number(lat.toFixed(5)), lon: Number(lon.toFixed(5)) };
}

function l15QuadkeyCenterWgs84(aoi: string): { lat: number; lon: number } {
  // Parse L15-XXXXE-YYYYN quadkey grid coordinates (2048x2048 global Web Mercator grid)
  const m = aoi.match(/L15-(\d{4})E-(\d{4})N/);
  if (!m) return { lat: 0, lon: 0 };
  const gx = Number(m[1]) + 0.5;
  const gy = Number(m[2]) + 0.5;
  const lon = (gx / 2048) * 360 - 180;
  const n = Math.PI - (2 * Math.PI * (2048 - gy)) / 2048;
  const lat = (180 / Math.PI) * Math.atan(0.5 * (Math.exp(n) - Math.exp(-n)));
  return { lat: Number(lat.toFixed(4)), lon: Number(lon.toFixed(4)) };
}

function polygonAreaPx2(ring: [number, number][]): number {
  if (ring.length < 3) return 0;
  let sum = 0;
  for (let i = 0; i < ring.length; i++) {
    const [x1, y1] = ring[i];
    const [x2, y2] = ring[(i + 1) % ring.length];
    sum += x1 * y2 - x2 * y1;
  }
  return Math.abs(sum) * 0.5;
}

interface CuratedSplitAoi {
  aoi: string;
  regionLabel: string;
  split: 'test' | 'val' | 'train';
  preferredModes: ('pos' | 'neg_long' | 'neg_short')[];
  maxTilesFromAoi: number;
}

/**
 * Geographically disjoint SpaceNet 7 AOI registry (> 250 km minimum distance between
 * ANY two splits: train vs val, train vs test, and val vs test).
 * - Held-out Test (12 singleton AOIs: 6 Positive + 6 Negative across 5 continents)
 * - Validation (12 singleton/disjoint AOIs: used strictly for hyperparameter & model selection)
 * - Training (24 disjoint AOIs: used for z-score scaling & weight/tree fitting, including
 *   natural negative quadrant prevalence to evaluate class-imbalance weighting)
 */
const CURATED_AOIS: CuratedSplitAoi[] = [
  // ============================================================================
  // HELD-OUT TEST SPLIT (12 geographically isolated regions — NEVER used in training or selection)
  // ============================================================================
  {
    aoi: 'L15-0361E-1300N_1446_2989_13',
    regionLabel: 'Meridian / Boise Metro, Idaho, USA',
    split: 'test',
    preferredModes: ['pos'],
    maxTilesFromAoi: 1,
  },
  {
    aoi: 'L15-1296E-1198N_5184_3399_13',
    regionLabel: 'West Abdullah Al-Mubarak, Kuwait',
    split: 'test',
    preferredModes: ['pos'],
    maxTilesFromAoi: 1,
  },
  {
    aoi: 'L15-1848E-0793N_7394_5018_13',
    regionLabel: 'Northern Growth Corridor, Melbourne, Australia',
    split: 'test',
    preferredModes: ['pos'],
    maxTilesFromAoi: 1,
  },
  {
    aoi: 'L15-0566E-1185N_2265_3451_13',
    regionLabel: 'Port St. Lucie Corridor, Florida, USA',
    split: 'test',
    preferredModes: ['pos'],
    maxTilesFromAoi: 1,
  },
  {
    aoi: 'L15-1049E-1370N_4196_2710_13',
    regionLabel: 'Rotterdam Polder & Urban Fringe, Netherlands',
    split: 'test',
    preferredModes: ['neg_long', 'neg_short'],
    maxTilesFromAoi: 1,
  },
  {
    aoi: 'L15-0595E-1278N_2383_3079_13',
    regionLabel: 'Lehigh Valley Agricultural Fringe, Pennsylvania, USA',
    split: 'test',
    preferredModes: ['neg_short'],
    maxTilesFromAoi: 1,
  },
  {
    aoi: 'L15-0434E-1218N_1736_3318_13',
    regionLabel: 'Permian Basin Semi-Arid Fringe, New Mexico, USA',
    split: 'test',
    preferredModes: ['neg_long', 'neg_short'],
    maxTilesFromAoi: 1,
  },
  {
    aoi: 'L15-1389E-1284N_5557_3054_13',
    regionLabel: 'Bukhara Oasis Fringe, Uzbekistan',
    split: 'test',
    preferredModes: ['neg_short'],
    maxTilesFromAoi: 1,
  },
  {
    aoi: 'L15-1204E-1202N_4816_3380_13',
    regionLabel: 'East Cairo New Urban District, Egypt',
    split: 'test',
    preferredModes: ['pos'],
    maxTilesFromAoi: 1,
  },
  {
    aoi: 'L15-0614E-0946N_2459_4406_13',
    regionLabel: 'Cusco Valley Corridor, Peru',
    split: 'test',
    preferredModes: ['pos'],
    maxTilesFromAoi: 1,
  },
  {
    aoi: 'L15-1748E-1247N_6993_3202_13',
    regionLabel: 'Sejong / Geum River Basin, South Korea',
    split: 'test',
    preferredModes: ['neg_short', 'neg_long'],
    maxTilesFromAoi: 1,
  },
  {
    aoi: 'L15-1172E-1306N_4688_2967_13',
    regionLabel: 'Ilfov Plain, Bucharest Fringe, Romania',
    split: 'test',
    preferredModes: ['neg_long', 'neg_short'],
    maxTilesFromAoi: 1,
  },

  // ============================================================================
  // VALIDATION SPLIT (12 geographically isolated regions — used ONLY for tuning & model selection)
  // ============================================================================
  {
    aoi: 'L15-1335E-1166N_5342_3524_13',
    regionLabel: 'Al Shamkha, Abu Dhabi, UAE',
    split: 'val',
    preferredModes: ['pos'],
    maxTilesFromAoi: 1,
  },
  {
    aoi: 'L15-1209E-1113N_4838_3737_13',
    regionLabel: 'Khartoum North Corridor, Sudan',
    split: 'val',
    preferredModes: ['pos'],
    maxTilesFromAoi: 1,
  },
  {
    aoi: 'L15-0358E-1220N_1433_3310_13',
    regionLabel: 'Otay Ranch / Chula Vista, California, USA',
    split: 'val',
    preferredModes: ['pos'],
    maxTilesFromAoi: 1,
  },
  {
    aoi: 'L15-1481E-1119N_5927_3715_13',
    regionLabel: 'Amaravati / Vijayawada, India',
    split: 'val',
    preferredModes: ['pos'],
    maxTilesFromAoi: 1,
  },
  {
    aoi: 'L15-1185E-0935N_4742_4450_13',
    regionLabel: 'Lusaka Eastern Fringe, Zambia',
    split: 'val',
    preferredModes: ['pos'],
    maxTilesFromAoi: 1,
  },
  {
    aoi: 'L15-0683E-1006N_2732_4164_13',
    regionLabel: 'Manaus Northern Expansion, Brazil',
    split: 'val',
    preferredModes: ['pos'],
    maxTilesFromAoi: 1,
  },
  {
    aoi: 'L15-1014E-1375N_4056_2688_13',
    regionLabel: 'West Midlands Greenbelt, United Kingdom',
    split: 'val',
    preferredModes: ['neg_short', 'neg_long'],
    maxTilesFromAoi: 1,
  },
  {
    aoi: 'L15-0924E-1108N_3699_3757_13',
    regionLabel: 'Dakar Peninsula Sector, Senegal',
    split: 'val',
    preferredModes: ['neg_short', 'neg_long'],
    maxTilesFromAoi: 1,
  },
  {
    aoi: 'L15-0487E-1246N_1950_3207_13',
    regionLabel: 'Bentonville Rural Fringe, Arkansas, USA',
    split: 'val',
    preferredModes: ['neg_long', 'neg_short'],
    maxTilesFromAoi: 1,
  },
  {
    aoi: 'L15-1200E-0847N_4802_4803_13',
    regionLabel: 'KwaZulu-Natal Coastal Hills, South Africa',
    split: 'val',
    preferredModes: ['neg_short'],
    maxTilesFromAoi: 1,
  },
  {
    aoi: 'L15-1617E-1207N_6468_3360_13',
    regionLabel: 'Sichuan Basin Farmland, China',
    split: 'val',
    preferredModes: ['neg_short'],
    maxTilesFromAoi: 1,
  },
  {
    aoi: 'L15-0632E-0892N_2528_4620_13',
    regionLabel: 'Calama Arid Plateau, Chile',
    split: 'val',
    preferredModes: ['neg_long', 'neg_short'],
    maxTilesFromAoi: 1,
  },

  // ============================================================================
  // TRAINING SPLIT (Disjoint global regions — supports multiple non-overlapping quadrants per AOI)
  // ============================================================================
  {
    aoi: 'L15-1669E-1160N_6679_3549_13',
    regionLabel: 'Guangzhou Northern District, China',
    split: 'train',
    preferredModes: ['pos', 'neg_short'],
    maxTilesFromAoi: 2,
  },
  {
    aoi: 'L15-1138E-1216N_4553_3325_13',
    regionLabel: 'Benghazi Coastal Plain, Libya',
    split: 'train',
    preferredModes: ['pos', 'neg_short'],
    maxTilesFromAoi: 2,
  },
  {
    aoi: 'L15-0586E-1127N_2345_3680_13',
    regionLabel: 'Spanish Town / Kingston West, Jamaica',
    split: 'train',
    preferredModes: ['pos', 'neg_short'],
    maxTilesFromAoi: 2,
  },
  {
    aoi: 'L15-1015E-1062N_4061_3941_13',
    regionLabel: 'Kumasi Peri-Urban Belt, Ghana',
    split: 'train',
    preferredModes: ['pos', 'neg_short'],
    maxTilesFromAoi: 2,
  },
  {
    aoi: 'L15-1276E-1107N_5105_3761_13',
    regionLabel: 'Dhamar Highlands, Yemen',
    split: 'train',
    preferredModes: ['pos', 'neg_short'],
    maxTilesFromAoi: 2,
  },
  {
    aoi: 'L15-1709E-1112N_6838_3742_13',
    regionLabel: 'Central Luzon Corridor, Philippines',
    split: 'train',
    preferredModes: ['pos', 'neg_short'],
    maxTilesFromAoi: 2,
  },
  {
    aoi: 'L15-0760E-0887N_3041_4643_13',
    regionLabel: 'Eastern Sao Paulo Metro Fringe, Brazil',
    split: 'train',
    preferredModes: ['neg_long', 'neg_short', 'pos'],
    maxTilesFromAoi: 2,
  },
  {
    aoi: 'L15-0331E-1257N_1327_3160_13',
    regionLabel: 'East Contra Costa Fringe, California, USA',
    split: 'train',
    preferredModes: ['neg_long', 'neg_short', 'pos'],
    maxTilesFromAoi: 2,
  },
  {
    aoi: 'L15-0506E-1204N_2027_3374_13',
    regionLabel: 'Ascension Parish Lowlands, Louisiana, USA',
    split: 'train',
    preferredModes: ['neg_long', 'neg_short'],
    maxTilesFromAoi: 2,
  },
  {
    aoi: 'L15-1716E-1211N_6864_3345_13',
    regionLabel: 'Pudong Eastern Agricultural Belt, China',
    split: 'train',
    preferredModes: ['neg_long', 'neg_short'],
    maxTilesFromAoi: 2,
  },
  {
    aoi: 'L15-0544E-1228N_2176_3279_13',
    regionLabel: 'DeKalb Wooded Canopy, Georgia, USA',
    split: 'train',
    preferredModes: ['neg_long', 'neg_short'],
    maxTilesFromAoi: 2,
  },
  {
    aoi: 'L15-1289E-1169N_5156_3514_13',
    regionLabel: 'Riyadh Northern Plateau, Saudi Arabia',
    split: 'train',
    preferredModes: ['neg_long', 'neg_short', 'pos'],
    maxTilesFromAoi: 2,
  },
];

async function fetchWithRetry(url: string, retries = 5): Promise<Response> {
  let lastErr: unknown;
  for (let attempt = 0; attempt < retries; attempt++) {
    try {
      const res = await fetch(url);
      if (res.ok) return res;
      lastErr = new Error(`HTTP ${res.status} fetching ${url}`);
    } catch (err) {
      lastErr = err;
    }
    await new Promise((r) => setTimeout(r, 800 * (attempt + 1)));
  }
  throw lastErr;
}

async function listAllSpaceNet7TrainAois(): Promise<string[]> {
  const xml = await (
    await fetchWithRetry(`${S3_BASE}/?prefix=spacenet/SN7_buildings/train/&delimiter=/`)
  ).text();
  return [...xml.matchAll(/<Prefix>spacenet\/SN7_buildings\/train\/([^/]+)\/<\/Prefix>/g)]
    .map((m) => m[1])
    .sort();
}

async function fetchGeoTiffSubtile(
  s3Key: string
): Promise<{
  fullWidth: number;
  fullHeight: number;
  bboxMercator: [number, number, number, number];
  rasters: Uint8Array[];
}> {
  const res = await fetchWithRetry(`${S3_BASE}/${s3Key}`);
  const ab = await res.arrayBuffer();
  const tiff = await fromArrayBuffer(ab);
  const img = await tiff.getImage();
  const fullWidth = img.getWidth();
  const fullHeight = img.getHeight();
  const bbox = img.getBoundingBox() as [number, number, number, number];
  const rawRasters = (await img.readRasters()) as unknown as Uint8Array[];
  return { fullWidth, fullHeight, bboxMercator: bbox, rasters: rawRasters };
}

function extractCropRgba(
  tiffData: { fullWidth: number; fullHeight: number; rasters: Uint8Array[] },
  x0: number,
  y0: number,
  cropW: number,
  cropH: number
): RawImageBuffer {
  const { fullWidth, rasters } = tiffData;
  const out = new Uint8ClampedArray(cropW * cropH * 4);
  const rBand = rasters[0];
  const gBand = rasters[1];
  const bBand = rasters[2];
  const aBand = rasters[3];

  for (let y = 0; y < cropH; y++) {
    const srcRow = (y0 + y) * fullWidth;
    const dstRow = y * cropW;
    for (let x = 0; x < cropW; x++) {
      const sIdx = srcRow + (x0 + x);
      const dIdx = (dstRow + x) * 4;
      out[dIdx] = rBand[sIdx];
      out[dIdx + 1] = gBand[sIdx];
      out[dIdx + 2] = bBand[sIdx];
      out[dIdx + 3] = aBand ? aBand[sIdx] : 255;
    }
  }
  return { width: cropW, height: cropH, data: out };
}

function writePng(filePath: string, buf: RawImageBuffer): void {
  const png = new PNG({ width: buf.width, height: buf.height });
  png.data = Buffer.from(buf.data.buffer, buf.data.byteOffset, buf.data.byteLength);
  const encoded = PNG.sync.write(png);
  fs.writeFileSync(filePath, encoded);
}

function readPng(filePath: string): RawImageBuffer {
  const decoded = PNG.sync.read(fs.readFileSync(filePath));
  return {
    width: decoded.width,
    height: decoded.height,
    data: new Uint8ClampedArray(decoded.data),
  };
}

interface ParsedFootprint {
  id: number;
  ring: [number, number][];
  centroid: [number, number];
  areaPx2: number;
}

async function detectPixYOrientation(aoi: string, monthStr: string): Promise<boolean> {
  const [pixGeo, mercGeo, tiffMeta] = await Promise.all([
    fetchWithRetry(
      `${S3_BASE}/spacenet/SN7_buildings/train/${aoi}/labels_match_pix/global_monthly_${monthStr}_mosaic_${aoi}_Buildings.geojson`
    ).then((r) => r.json()),
    fetchWithRetry(
      `${S3_BASE}/spacenet/SN7_buildings/train/${aoi}/labels_match/global_monthly_${monthStr}_mosaic_${aoi}_Buildings.geojson`
    ).then((r) => r.json()),
    fetchGeoTiffSubtile(
      `spacenet/SN7_buildings/train/${aoi}/images_masked/global_monthly_${monthStr}_mosaic_${aoi}.tif`
    ),
  ]);

  const fMerc = mercGeo.features[0];
  const targetId = fMerc.properties.Id;
  const fPix = pixGeo.features.find((f: any) => f.properties.Id === targetId);
  if (!fPix) return false;

  const mercY = fMerc.geometry.coordinates[0][0][1];
  const pixY = fPix.geometry.coordinates[0][0][1];
  const [, minY, , maxY] = tiffMeta.bboxMercator;
  const expectedTopDownY = ((maxY - mercY) / (maxY - minY)) * tiffMeta.fullHeight;
  const diffDirect = Math.abs(pixY - expectedTopDownY);
  const diffFlipped = Math.abs(tiffMeta.fullHeight - pixY - expectedTopDownY);
  console.log(
    `[Orientation Check] pixY=${pixY.toFixed(1)}, expectedTopDownY=${expectedTopDownY.toFixed(
      1
    )} -> flipY=${diffFlipped < diffDirect}`
  );
  return diffFlipped < diffDirect;
}

async function main() {
  fs.mkdirSync(PUBLIC_DATA_DIR, { recursive: true });
  fs.mkdirSync(path.join(PUBLIC_DATA_DIR, 'images'), { recursive: true });
  fs.mkdirSync(path.join(PUBLIC_DATA_DIR, 'footprints'), { recursive: true });
  fs.mkdirSync(SRC_DATA_DIR, { recursive: true });
  fs.mkdirSync(TILE_CACHE_DIR, { recursive: true });

  // Clean old public test images/footprints before regenerating
  for (const sub of ['images', 'footprints']) {
    const dir = path.join(PUBLIC_DATA_DIR, sub);
    for (const f of fs.readdirSync(dir)) {
      fs.unlinkSync(path.join(dir, f));
    }
  }

  const initMetaCachePath = path.join(TILE_CACHE_DIR, '_init_meta.json');
  let flipPixY = false;
  let allS3Aois: string[] = [];
  if (fs.existsSync(initMetaCachePath)) {
    const cachedInit = JSON.parse(fs.readFileSync(initMetaCachePath, 'utf8'));
    flipPixY = Boolean(cachedInit.flipPixY);
    allS3Aois = cachedInit.allS3Aois;
  } else {
    flipPixY = await detectPixYOrientation(
      'L15-0331E-1257N_1327_3160_13',
      '2018_01'
    );
    allS3Aois = await listAllSpaceNet7TrainAois();
    fs.writeFileSync(initMetaCachePath, JSON.stringify({ flipPixY, allS3Aois }));
  }
  const curatedMap = new Map(CURATED_AOIS.map((c) => [c.aoi, c]));
  const testValCenters = CURATED_AOIS.filter(
    (c) => c.split === 'test' || c.split === 'val'
  ).map((c) => ({ ...c, center: l15QuadkeyCenterWgs84(c.aoi) }));

  const fullAoiSpecs: CuratedSplitAoi[] = [...CURATED_AOIS];
  for (const aoi of allS3Aois) {
    if (curatedMap.has(aoi)) continue;
    const c = l15QuadkeyCenterWgs84(aoi);
    let minDistToTestOrVal = Infinity;
    for (const tv of testValCenters) {
      const d = haversineKm(c.lat, c.lon, tv.center.lat, tv.center.lon);
      if (d < minDistToTestOrVal) minDistToTestOrVal = d;
    }
    if (minDistToTestOrVal >= 300) {
      fullAoiSpecs.push({
        aoi,
        regionLabel: `SpaceNet 7 Urban Sector (${aoi.slice(0, 15)})`,
        split: 'train',
        preferredModes: ['neg_short', 'neg_long', 'pos'],
        maxTilesFromAoi: 2,
      });
    }
  }

  console.log(
    `Discovered ${allS3Aois.length} official SN7 AOIs -> ${
      fullAoiSpecs.filter((s) => s.split === 'test').length
    } test AOIs, ${fullAoiSpecs.filter((s) => s.split === 'val').length} val AOIs, ${
      fullAoiSpecs.filter((s) => s.split === 'train').length
    } geographically isolated train AOIs.`
  );

  const parseFootprintsOriented = (geojson: any, imgH: number): ParsedFootprint[] => {
    const out: ParsedFootprint[] = [];
    if (!geojson?.features) return out;
    for (const f of geojson.features) {
      const id = Number(f.properties?.Id ?? -1);
      const geom = f.geometry;
      if (!geom) continue;
      let rawRing: number[][] | null = null;
      if (geom.type === 'Polygon' && geom.coordinates?.[0]) {
        rawRing = geom.coordinates[0];
      } else if (geom.type === 'MultiPolygon' && geom.coordinates?.[0]?.[0]) {
        rawRing = geom.coordinates[0][0];
      }
      if (!rawRing || rawRing.length < 3) continue;

      const ring: [number, number][] = rawRing.map((pt) => [
        Number(pt[0]),
        flipPixY ? Number(imgH - pt[1]) : Number(pt[1]),
      ]);
      let cx = 0;
      let cy = 0;
      for (const [x, y] of ring) {
        cx += x;
        cy += y;
      }
      cx /= ring.length;
      cy /= ring.length;
      const areaPx2 = polygonAreaPx2(ring);
      if (areaPx2 > 0.5) {
        out.push({ id, ring, centroid: [cx, cy], areaPx2 });
      }
    }
    return out;
  };

  const CROP_SIZE = 512;
  const NON_OVERLAPPING_QUADRANTS: [number, number][] = [
    [0, 0],
    [512, 0],
    [0, 511],
    [512, 511],
  ];

  const splitRecords: DatasetSplitRecord[] = [];
  const manifestSamples: SampleManifestEntry[] = [];

  // Track pixel-level IoU totals on Train+Val (for threshold selection) and Held-Out Test
  let trainValBaseInterSum = 0;
  let trainValBaseUnionSum = 0;
  let trainValImpInterSum = 0;
  let trainValImpUnionSum = 0;
  const trainValChangedSampleBaseIoUs: number[] = [];
  const trainValChangedSampleImpIoUs: number[] = [];

  let testBaseInterSum = 0;
  let testBaseUnionSum = 0;
  let testImpInterSum = 0;
  let testImpUnionSum = 0;
  const testChangedSampleBaseIoUs: number[] = [];
  const testChangedSampleImpIoUs: number[] = [];

  interface ExtractedItem {
    spec: CuratedSplitAoi;
    targetPos: boolean;
    chosen: {
      m1: string;
      m2: string;
      qKey: string;
      x0: number;
      y0: number;
      eBuf: RawImageBuffer;
      lBuf: RawImageBuffer;
      spatial: SpatialMetadata;
      earlierCount: number;
      laterCount: number;
      newCount: number;
      removedCount: number;
      persistentCount: number;
      earlierAreaPx2: number;
      laterAreaPx2: number;
      newAreaPx2: number;
      polygons: BuildingFootprintPolygon[];
      s3Img1: string;
      s3Img2: string;
      s3Lbl1: string;
      s3Lbl2: string;
    };
  }

  const processSingleAoi = async (spec: CuratedSplitAoi): Promise<ExtractedItem[]> => {
    const cacheMetaPath = path.join(TILE_CACHE_DIR, `${spec.aoi}_${spec.split}.json`);
    if (fs.existsSync(cacheMetaPath)) {
      try {
        const cachedMeta = JSON.parse(fs.readFileSync(cacheMetaPath, 'utf8')) as Array<{
          targetPos: boolean;
          meta: Omit<ExtractedItem['chosen'], 'eBuf' | 'lBuf'>;
          eFile: string;
          lFile: string;
        }>;
        return cachedMeta.map((entry) => ({
          spec,
          targetPos: entry.targetPos,
          chosen: {
            ...entry.meta,
            eBuf: readPng(path.join(TILE_CACHE_DIR, entry.eFile)),
            lBuf: readPng(path.join(TILE_CACHE_DIR, entry.lFile)),
          },
        }));
      } catch {
        // Fall through to fresh S3 extraction if cache entry is unreadable
      }
    }

    const outItems: ExtractedItem[] = [];
    try {
      const listXml = await (
        await fetchWithRetry(
          `${S3_BASE}/?prefix=spacenet/SN7_buildings/train/${spec.aoi}/labels_match_pix/`
        )
      ).text();
      const labelKeys = [...listXml.matchAll(/<Key>([^<]+\.geojson)<\/Key>/g)]
        .map((m) => m[1])
        .sort();
      if (labelKeys.length < 6) return outItems;

      const months = labelKeys
        .map((k) => k.match(/global_monthly_(\d{4}_\d{2})_mosaic/)?.[1])
        .filter((m): m is string => Boolean(m));

      const usedQuadrants = new Set<string>();
      let extractedForAoi = 0;

      for (const mode of spec.preferredModes) {
        if (extractedForAoi >= spec.maxTilesFromAoi) break;
        const targetPos = mode === 'pos';

        const candidatePairs: [string, string][] =
          mode === 'pos'
            ? [
                [months[0], months[months.length - 1]],
                [months[1], months[months.length - 2]],
                [months[2], months[months.length - 3]],
              ]
            : mode === 'neg_long'
            ? [
                [months[0], months[months.length - 1]],
                [months[1], months[months.length - 2]],
                [months[0], months[Math.min(11, months.length - 1)]],
                [months[2], months[Math.min(10, months.length - 1)]],
              ]
            : [
                [months[Math.min(3, months.length - 2)], months[Math.min(7, months.length - 1)]],
                [months[Math.min(4, months.length - 2)], months[Math.min(8, months.length - 1)]],
                [months[Math.min(2, months.length - 2)], months[Math.min(6, months.length - 1)]],
                [months[Math.min(5, months.length - 2)], months[Math.min(9, months.length - 1)]],
                [months[Math.min(6, months.length - 2)], months[Math.min(9, months.length - 1)]],
                [months[Math.min(4, months.length - 2)], months[Math.min(6, months.length - 1)]],
                [months[Math.min(1, months.length - 2)], months[Math.min(3, months.length - 1)]],
                [months[0], months[Math.min(4, months.length - 1)]],
              ];

        let chosen: ExtractedItem['chosen'] | null = null;

        for (const [m1, m2] of candidatePairs) {
          if (!m1 || !m2 || m1 === m2) continue;
          const s3Img1 = `spacenet/SN7_buildings/train/${spec.aoi}/images_masked/global_monthly_${m1}_mosaic_${spec.aoi}.tif`;
          const s3Img2 = `spacenet/SN7_buildings/train/${spec.aoi}/images_masked/global_monthly_${m2}_mosaic_${spec.aoi}.tif`;
          const s3Lbl1 = `spacenet/SN7_buildings/train/${spec.aoi}/labels_match_pix/global_monthly_${m1}_mosaic_${spec.aoi}_Buildings.geojson`;
          const s3Lbl2 = `spacenet/SN7_buildings/train/${spec.aoi}/labels_match_pix/global_monthly_${m2}_mosaic_${spec.aoi}_Buildings.geojson`;

          const [t1, t2, g1, g2] = await Promise.all([
            fetchGeoTiffSubtile(s3Img1),
            fetchGeoTiffSubtile(s3Img2),
            fetchWithRetry(`${S3_BASE}/${s3Lbl1}`).then((r) => r.json()),
            fetchWithRetry(`${S3_BASE}/${s3Lbl2}`).then((r) => r.json()),
          ]);

          if (t1.fullWidth < 512 || t1.fullHeight < 512) continue;
          const fp1All = parseFootprintsOriented(g1, t1.fullHeight);
          const fp2All = parseFootprintsOriented(g2, t2.fullHeight);

          const [minX, minY, maxX, maxY] = t1.bboxMercator;
          const gsdX = (maxX - minX) / t1.fullWidth;
          const gsdY = (maxY - minY) / t1.fullHeight;
          const centerMercY = (minY + maxY) * 0.5;
          const latRad = 2 * Math.atan(Math.exp(centerMercY / 6378137)) - Math.PI / 2;
          const gsdMeters = Number((((gsdX + gsdY) * 0.5) * Math.cos(latRad)).toFixed(3));
          const pxAreaM2 = gsdMeters * gsdMeters;

          for (const [qx, qy] of NON_OVERLAPPING_QUADRANTS) {
            const x0 = Math.min(qx, t1.fullWidth - CROP_SIZE);
            const y0 = Math.min(qy, t1.fullHeight - CROP_SIZE);
            const qKey = `${x0}_${y0}`;
            if (usedQuadrants.has(qKey)) continue;

            const eBuf = extractCropRgba(t1, x0, y0, CROP_SIZE, CROP_SIZE);
            const lBuf = extractCropRgba(t2, x0, y0, CROP_SIZE, CROP_SIZE);

            const val = validateImagePair(eBuf, lBuf);
            if (!val.valid || val.validClearPixelRatio < 0.88) continue;

            const inCrop = (fp: ParsedFootprint) =>
              fp.centroid[0] >= x0 + 4 &&
              fp.centroid[0] <= x0 + CROP_SIZE - 4 &&
              fp.centroid[1] >= y0 + 4 &&
              fp.centroid[1] <= y0 + CROP_SIZE - 4;

            const sub1 = fp1All.filter(inCrop);
            const sub2 = fp2All.filter(inCrop);
            const map1 = new Map(sub1.map((f) => [f.id, f]));
            const map2 = new Map(sub2.map((f) => [f.id, f]));

            let newCount = 0;
            let persistentCount = 0;
            let removedCount = 0;
            let earlierAreaPx2 = 0;
            let laterAreaPx2 = 0;
            let newAreaPx2 = 0;
            const polygons: BuildingFootprintPolygon[] = [];

            for (const f of sub1) {
              earlierAreaPx2 += f.areaPx2;
              if (!map2.has(f.id)) {
                removedCount++;
                polygons.push({
                  id: f.id,
                  status: 'removed',
                  areaPx2: Number(f.areaPx2.toFixed(1)),
                  areaM2: Number((f.areaPx2 * pxAreaM2).toFixed(1)),
                  ring: f.ring.map(([rx, ry]) => [
                    Number((rx - x0).toFixed(1)),
                    Number((ry - y0).toFixed(1)),
                  ]),
                });
              }
            }

            for (const f of sub2) {
              laterAreaPx2 += f.areaPx2;
              const isNew = !map1.has(f.id);
              if (isNew) {
                newCount++;
                newAreaPx2 += f.areaPx2;
              } else {
                persistentCount++;
              }
              polygons.push({
                id: f.id,
                status: isNew ? 'new_construction' : 'persistent',
                areaPx2: Number(f.areaPx2.toFixed(1)),
                areaM2: Number((f.areaPx2 * pxAreaM2).toFixed(1)),
                ring: f.ring.map(([rx, ry]) => [
                  Number((rx - x0).toFixed(1)),
                  Number((ry - y0).toFixed(1)),
                ]),
              });
            }

            const newAreaM2 = newAreaPx2 * pxAreaM2;
            const meetsPositive = newCount >= 12 && newAreaM2 >= 1500;
            const meetsNegative = newCount <= 2 && removedCount <= 2 && newAreaM2 <= 350;

            if ((targetPos && meetsPositive) || (!targetPos && meetsNegative)) {
              const subMinX = minX + x0 * gsdX;
              const subMaxX = minX + (x0 + CROP_SIZE) * gsdX;
              const subMaxY = maxY - y0 * gsdY;
              const subMinY = maxY - (y0 + CROP_SIZE) * gsdY;
              const centerWgs84 = mercatorToLatLon(
                (subMinX + subMaxX) * 0.5,
                (subMinY + subMaxY) * 0.5
              );
              const tileAreaKm2 = Number(
                ((CROP_SIZE * CROP_SIZE * pxAreaM2) / 1_000_000).toFixed(3)
              );

              const spatial: SpatialMetadata = {
                crs: 'EPSG:3857',
                bboxMercator: [
                  Number(subMinX.toFixed(2)),
                  Number(subMinY.toFixed(2)),
                  Number(subMaxX.toFixed(2)),
                  Number(subMaxY.toFixed(2)),
                ],
                centerWgs84,
                gsdMetersPerPx: gsdMeters,
                tileAreaKm2,
                cropWindow: {
                  x0,
                  y0,
                  width: CROP_SIZE,
                  height: CROP_SIZE,
                  sourceWidth: t1.fullWidth,
                  sourceHeight: t1.fullHeight,
                },
              };

              if (
                !chosen ||
                (targetPos && newCount > chosen.newCount) ||
                (!targetPos && newCount < chosen.newCount)
              ) {
                chosen = {
                  m1,
                  m2,
                  qKey,
                  x0,
                  y0,
                  eBuf,
                  lBuf,
                  spatial,
                  earlierCount: sub1.length,
                  laterCount: sub2.length,
                  newCount,
                  removedCount,
                  persistentCount,
                  earlierAreaPx2,
                  laterAreaPx2,
                  newAreaPx2,
                  polygons,
                  s3Img1,
                  s3Img2,
                  s3Lbl1,
                  s3Lbl2,
                };
              }
            }
          }
          if (chosen) break;
        }

        if (!chosen) continue;
        usedQuadrants.add(chosen.qKey);
        extractedForAoi++;
        outItems.push({ spec, targetPos, chosen });
      }
      if (outItems.length > 0) {
        const toCache = outItems.map((item, idx) => {
          const eFile = `${spec.aoi}_${spec.split}_${idx}_earlier.png`;
          const lFile = `${spec.aoi}_${spec.split}_${idx}_later.png`;
          writePng(path.join(TILE_CACHE_DIR, eFile), item.chosen.eBuf);
          writePng(path.join(TILE_CACHE_DIR, lFile), item.chosen.lBuf);
          const { eBuf: _e, lBuf: _l, ...meta } = item.chosen;
          return { targetPos: item.targetPos, meta, eFile, lFile };
        });
        fs.writeFileSync(cacheMetaPath, JSON.stringify(toCache));
      } else {
        fs.writeFileSync(cacheMetaPath, JSON.stringify([]));
      }
    } catch (err: any) {
      console.warn(`Error processing ${spec.aoi}: ${err.message}`);
    }
    return outItems;
  };

  const BATCH_CONCURRENCY = 2;
  const allExtractedItems: ExtractedItem[] = [];
  for (let b = 0; b < fullAoiSpecs.length; b += BATCH_CONCURRENCY) {
    const slice = fullAoiSpecs.slice(b, b + BATCH_CONCURRENCY);
    const allCached = slice.every((s) =>
      fs.existsSync(path.join(TILE_CACHE_DIR, `${s.aoi}_${s.split}.json`))
    );
    const batchOut = await Promise.all(slice.map(processSingleAoi));
    for (const arr of batchOut) allExtractedItems.push(...arr);
    if (!allCached) {
      await new Promise((r) => setTimeout(r, 180));
    }
  }

  for (const { spec, targetPos, chosen } of allExtractedItems) {
    const splitCount = splitRecords.filter((r) => r.split === spec.split).length + 1;
    const pairId = `sn7-${spec.split}-${String(splitCount).padStart(2, '0')}`;

    const { alignedLater: alignedLaterSubpx } = alignLaterImage(
      chosen.eBuf,
      chosen.lBuf,
      DEFAULT_PREPROCESSING_CONFIG.enableAlignment,
      6,
      true
    );

    const basePass = computeBaselineFeaturesAndMasks(
      chosen.eBuf,
      alignedLaterSubpx,
      BASELINE_V3_PREPROCESSING_CONFIG,
      chosen.spatial.gsdMetersPerPx
    );
    const impPass = computeFeaturesAndMasks(
      chosen.eBuf,
      alignedLaterSubpx,
      DEFAULT_PREPROCESSING_CONFIG,
      chosen.spatial.gsdMetersPerPx
    );

    const gtChangedMask = rasterizeFootprintsToMask(
      chosen.polygons,
      CROP_SIZE,
      CROP_SIZE,
      ['new_construction', 'removed']
    );
    const baseIoU = computeBinaryMaskIoU(basePass.candidateMask, gtChangedMask);
    const impIoU = computeBinaryMaskIoU(impPass.candidateMask, gtChangedMask);

    if (spec.split === 'test') {
      testBaseInterSum += baseIoU.intersection;
      testBaseUnionSum += baseIoU.union;
      testImpInterSum += impIoU.intersection;
      testImpUnionSum += impIoU.union;
      if (targetPos) {
        testChangedSampleBaseIoUs.push(baseIoU.iou);
        testChangedSampleImpIoUs.push(impIoU.iou);
      }
    } else {
      trainValBaseInterSum += baseIoU.intersection;
      trainValBaseUnionSum += baseIoU.union;
      trainValImpInterSum += impIoU.intersection;
      trainValImpUnionSum += impIoU.union;
      if (targetPos) {
        trainValChangedSampleBaseIoUs.push(baseIoU.iou);
        trainValChangedSampleImpIoUs.push(impIoU.iou);
      }
    }

    const pxAreaM2 = chosen.spatial.gsdMetersPerPx * chosen.spatial.gsdMetersPerPx;
    const newBuildingAreaM2 = Number((chosen.newAreaPx2 * pxAreaM2).toFixed(1));

    splitRecords.push({
      pairId,
      aoiId: spec.aoi,
      split: spec.split,
      lat: chosen.spatial.centerWgs84.lat,
      lon: chosen.spatial.centerWgs84.lon,
      earlierDate: chosen.m1.replace('_', '-'),
      laterDate: chosen.m2.replace('_', '-'),
      newBuildings: chosen.newCount,
      newBuildingAreaM2,
      groundTruthPositive: targetPos,
      features: impPass.features,
      baselineFeatures: basePass.features,
      pixelIoUBaseline: Number(baseIoU.iou.toFixed(4)),
      pixelIoUImproved: Number(impIoU.iou.toFixed(4)),
    });

    console.log(
      `[${spec.split.toUpperCase()}] ${pairId} | ${spec.regionLabel} | ${chosen.m1}->${chosen.m2} | +${chosen.newCount} bldgs (${newBuildingAreaM2} m²) | newEdge=${impPass.features.newEdgeEmergencePct}% dipole=${impPass.features.roofShadowDipolePct}% compact=${impPass.features.compactClusterDensityKm2} | IoU ${baseIoU.iou.toFixed(3)}->${impIoU.iou.toFixed(3)}`
    );

    if (spec.split === 'test') {
      const earlierFilename = `${pairId}_${chosen.m1}_earlier.png`;
      const laterFilename = `${pairId}_${chosen.m2}_later.png`;
      const footprintsFilename = `${pairId}_footprints.json`;

      writePng(path.join(PUBLIC_DATA_DIR, 'images', earlierFilename), chosen.eBuf);
      writePng(path.join(PUBLIC_DATA_DIR, 'images', laterFilename), chosen.lBuf);
      fs.writeFileSync(
        path.join(PUBLIC_DATA_DIR, 'footprints', footprintsFilename),
        JSON.stringify(chosen.polygons)
      );

      const [y1, mo1] = chosen.m1.split('_').map(Number);
      const [y2, mo2] = chosen.m2.split('_').map(Number);
      const intervalMonths = (y2 - y1) * 12 + (mo2 - mo1);
      const totalSubtilePx = CROP_SIZE * CROP_SIZE;
      const earlierCov = Number(((chosen.earlierAreaPx2 / totalSubtilePx) * 100).toFixed(2));
      const laterCov = Number(((chosen.laterAreaPx2 / totalSubtilePx) * 100).toFixed(2));

      manifestSamples.push({
        id: pairId,
        aoiId: spec.aoi,
        title: spec.regionLabel,
        regionLabel: spec.regionLabel,
        split: 'test',
        earlierDate: chosen.m1.replace('_', '-'),
        laterDate: chosen.m2.replace('_', '-'),
        intervalMonths,
        earlierImageUrl: `/data/spacenet7/images/${earlierFilename}`,
        laterImageUrl: `/data/spacenet7/images/${laterFilename}`,
        earlierSourceS3Key: chosen.s3Img1,
        laterSourceS3Key: chosen.s3Img2,
        earlierLabelS3Key: chosen.s3Lbl1,
        laterLabelS3Key: chosen.s3Lbl2,
        sourceDataset: 'SpaceNet 7 Multi-Temporal Urban Development Dataset (SN7)',
        sourceUrl: 'https://spacenet.ai/sn7-challenge/',
        sourceAttribution:
          'SpaceNet Partners & Planet Labs Inc. (Global Monthly Basemaps, CC-BY-4.0)',
        license: 'CC-BY-4.0',
        isSyntheticFallback: false,
        spatial: chosen.spatial,
        groundTruth: {
          available: true,
          earlierCount: chosen.earlierCount,
          laterCount: chosen.laterCount,
          newConstructionCount: chosen.newCount,
          removedCount: chosen.removedCount,
          persistentCount: chosen.persistentCount,
          earlierCoveragePct: earlierCov,
          laterCoveragePct: laterCov,
          netCoverageDeltaPct: Number((laterCov - earlierCov).toFixed(2)),
          newConstructionAreaM2: newBuildingAreaM2,
          groundTruthPositive: targetPos,
          groundTruthClassLabel: targetPos
            ? 'Building-Related Change Present'
            : 'Little or No Building-Related Change',
          footprintsUrl: `/data/spacenet7/footprints/${footprintsFilename}`,
        },
        knownLimitations: [
          `Planet Dove ~${chosen.spatial.gsdMetersPerPx.toFixed(
            2
          )} m/px spatial resolution means individual small sheds (<25 m²) span only 1–2 pixels.`,
          'Monthly mosaic compositing can introduce subtle seasonal illumination, agricultural phenology, and roof reflectance differences unrelated to construction.',
          'SpaceNet 7 ground-truth polygons were manually digitized on Planet imagery and may omit minor auxiliary structures or have 1–2 pixel boundary offsets.',
        ],
      });
    }
  }

  // Verify minimum geographic separation across all 3 splits (Train vs Val, Train vs Test, Val vs Test)
  const trainRecs = splitRecords.filter((r) => r.split === 'train');
  const valRecs = splitRecords.filter((r) => r.split === 'val');
  const testRecs = splitRecords.filter((r) => r.split === 'test');

  const minPairDist = (a: DatasetSplitRecord[], b: DatasetSplitRecord[]) => {
    let minKm = Infinity;
    for (const r1 of a) {
      for (const r2 of b) {
        const d = haversineKm(r1.lat, r1.lon, r2.lat, r2.lon);
        if (d < minKm) minKm = d;
      }
    }
    return minKm;
  };

  const trainValKm = minPairDist(trainRecs, valRecs);
  const trainTestKm = minPairDist(trainRecs, testRecs);
  const valTestKm = minPairDist(valRecs, testRecs);

  console.log(
    `\n=== SPLIT SUMMARY ===\n` +
      `Train: ${trainRecs.length} (${trainRecs.filter((r) => r.groundTruthPositive).length} pos, ${
        trainRecs.filter((r) => !r.groundTruthPositive).length
      } neg)\n` +
      `Val:   ${valRecs.length} (${valRecs.filter((r) => r.groundTruthPositive).length} pos, ${
        valRecs.filter((r) => !r.groundTruthPositive).length
      } neg)\n` +
      `Test:  ${testRecs.length} (${testRecs.filter((r) => r.groundTruthPositive).length} pos, ${
        testRecs.filter((r) => !r.groundTruthPositive).length
      } neg)\n` +
      `Minimum cross-split distances -> Train-Val: ${trainValKm.toFixed(
        1
      )} km | Train-Test: ${trainTestKm.toFixed(1)} km | Val-Test: ${valTestKm.toFixed(1)} km`
  );

  const avg = (arr: number[]) =>
    arr.length > 0 ? arr.reduce((s, v) => s + v, 0) / arr.length : 0;

  const nonTestCount = trainRecs.length + valRecs.length;
  const pixelIoUMetrics: {
    baselineTest: PixelChangeIoUMetrics;
    improvedTest: PixelChangeIoUMetrics;
    baselineTrainVal: PixelChangeIoUMetrics;
    improvedTrainVal: PixelChangeIoUMetrics;
  } = {
    baselineTest: {
      evaluatedSampleCount: testRecs.length,
      changedSampleCount: testChangedSampleBaseIoUs.length,
      meanChangedSampleIoU: Number(avg(testChangedSampleBaseIoUs).toFixed(4)),
      microPixelIoU: Number(
        (testBaseUnionSum > 0 ? testBaseInterSum / testBaseUnionSum : 0).toFixed(4)
      ),
    },
    improvedTest: {
      evaluatedSampleCount: testRecs.length,
      changedSampleCount: testChangedSampleImpIoUs.length,
      meanChangedSampleIoU: Number(avg(testChangedSampleImpIoUs).toFixed(4)),
      microPixelIoU: Number(
        (testImpUnionSum > 0 ? testImpInterSum / testImpUnionSum : 0).toFixed(4)
      ),
    },
    baselineTrainVal: {
      evaluatedSampleCount: nonTestCount,
      changedSampleCount: trainValChangedSampleBaseIoUs.length,
      meanChangedSampleIoU: Number(avg(trainValChangedSampleBaseIoUs).toFixed(4)),
      microPixelIoU: Number(
        (trainValBaseUnionSum > 0 ? trainValBaseInterSum / trainValBaseUnionSum : 0).toFixed(4)
      ),
    },
    improvedTrainVal: {
      evaluatedSampleCount: nonTestCount,
      changedSampleCount: trainValChangedSampleImpIoUs.length,
      meanChangedSampleIoU: Number(avg(trainValChangedSampleImpIoUs).toFixed(4)),
      microPixelIoU: Number(
        (trainValImpUnionSum > 0 ? trainValImpInterSum / trainValImpUnionSum : 0).toFixed(4)
      ),
    },
  };

  const modelArtifact = trainValidateAndSelectBestModel({
    records: splitRecords,
    positiveCriterion:
      'Ground-truth SpaceNet 7 annotations show >= 12 newly constructed building footprints AND >= 1,500 m² new building footprint area within the 512x512 sub-tile.',
    negativeCriterion:
      'Ground-truth SpaceNet 7 annotations show <= 2 newly constructed or removed building footprints AND <= 350 m² new building area within the 512x512 sub-tile.',
    leakagePrevention:
      'Strict 3-way AOI-level geographic cluster split (Train / Validation / Held-Out Test) plus 150 km Haversine metropolitan group clustering: zero shared AOIs across splits, >330 km minimum separation between all splits (>424 km to Held-Out Test), z-score standardization & weights fit strictly on Train (n=47), hyperparameters, pixel change threshold (32), model family, and Platt probability calibration fit strictly on out-of-sample Non-Test logits (5-Fold Geo-CV on Train + Validation, n=59), and final metrics evaluated once on untouched Held-Out Test (n=12).',
    minTrainTestDistanceKm: Math.min(trainTestKm, valTestKm),
    minSplitPairwiseDistanceKm: {
      trainVal: trainValKm,
      trainTest: trainTestKm,
      valTest: valTestKm,
    },
    pixelIoUMetrics,
    datasetSource:
      'Official SpaceNet 7 Multi-Temporal Urban Development Dataset (s3://spacenet-dataset/spacenet/SN7_buildings/train/)',
  });

  console.log('\n=== CANDIDATE MODEL COMPARISON (Train / 5-Fold Geo-CV / Val / Held-Out Test) ===');
  for (const c of modelArtifact.baselineComparison?.candidateModels ?? []) {
    console.log(
      `\n* ${c.modelLabel} ${c.selectedByValidation ? '[SELECTED BY NON-TEST GEO-CV + VAL]' : ''}`
    );
    console.log('  Hyperparameters:', JSON.stringify(c.hyperparameters));
    console.log('  Train: ', JSON.stringify(c.trainMetrics));
    console.log('  GeoCV: ', JSON.stringify(c.geoCvMetrics));
    console.log('  Val:   ', JSON.stringify(c.valMetrics));
    console.log('  Test:  ', JSON.stringify(c.testMetrics));
  }
  console.log('\n=== PROBABILITY CALIBRATION BENCHMARK (Nested OOF on n=59 Non-Test) ===');
  console.log(JSON.stringify(modelArtifact.calibration, null, 2));
  console.log('\nSelection Rationale:', modelArtifact.baselineComparison?.selectionRationale);
  console.log('Pixel Change IoU:', JSON.stringify(pixelIoUMetrics));

  const manifest: DatasetManifest = {
    datasetName: 'SpaceNet 7 Multi-Temporal Urban Development — Preloaded Held-Out Test Pack',
    officialSourceUrl: 'https://spacenet.ai/sn7-challenge/',
    awsBucketUri: 's3://spacenet-dataset/spacenet/SN7_buildings/train/',
    license: 'Creative Commons Attribution 4.0 International (CC-BY-4.0)',
    attribution:
      'SpaceNet LLC, CosmiQ Works, Planet Labs Inc., Capella Space, Maxar Technologies, IEEE GRSS, Amazon Web Services.',
    generatedAt: new Date().toISOString(),
    isGenuineSpaceNet7: true,
    methodologyNote:
      'Each preloaded sample is a genuine native-resolution 512x512 pixel sub-tile extracted directly from official SpaceNet 7 Planet Dove monthly GeoTIFF mosaics and paired with its official labels_match_pix tracked building-footprint annotations. All preloaded interactive samples belong strictly to the held-out geographic test split (never used during training, feature scaling, 5-fold geographic cross-validation, or validation hyperparameter/model selection).',
    samples: manifestSamples,
  };

  fs.writeFileSync(
    path.join(PUBLIC_DATA_DIR, 'manifest.json'),
    JSON.stringify(manifest, null, 2)
  );
  fs.writeFileSync(
    path.join(SRC_DATA_DIR, 'manifest.json'),
    JSON.stringify(manifest, null, 2)
  );
  fs.writeFileSync(
    path.join(SRC_DATA_DIR, 'modelArtifact.json'),
    JSON.stringify(modelArtifact, null, 2)
  );

  const csvHeader = [
    'pairId',
    'aoiId',
    'geoClusterId',
    'split',
    'lat',
    'lon',
    'earlierDate',
    'laterDate',
    'newBuildings',
    'newBuildingAreaM2',
    'groundTruthPositive',
    ...FEATURE_KEYS,
    ...FEATURE_KEYS.map((k) => `baseline_${k}`),
    'pixelIoUBaseline',
    'pixelIoUImproved',
    'rawUncalibratedProbability',
    'predictedProbability',
    'predictedPositive',
  ].join(',');

  const csvRows = modelArtifact.records.map((r) =>
    [
      r.pairId,
      r.aoiId,
      r.geoClusterId ?? '',
      r.split,
      r.lat,
      r.lon,
      r.earlierDate,
      r.laterDate,
      r.newBuildings,
      r.newBuildingAreaM2,
      r.groundTruthPositive ? 1 : 0,
      ...FEATURE_KEYS.map((k) => r.features[k]),
      ...FEATURE_KEYS.map((k) => r.baselineFeatures?.[k] ?? ''),
      r.pixelIoUBaseline ?? '',
      r.pixelIoUImproved ?? '',
      r.rawUncalibratedProbability ?? '',
      r.predictedProbability ?? '',
      r.predictedPositive ? 1 : 0,
    ].join(',')
  );

  fs.writeFileSync(
    path.join(SRC_DATA_DIR, 'spacenet7_feature_dataset.csv'),
    [csvHeader, ...csvRows].join('\n')
  );

  console.log(
    `\nSuccessfully generated ${manifestSamples.length} held-out test samples and ${modelArtifact.records.length} total split records!`
  );
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
