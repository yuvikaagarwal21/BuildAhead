import fs from 'node:fs';
import path from 'node:path';
import { fromArrayBuffer } from 'geotiff';
import { PNG } from 'pngjs';
import {
  computeFeaturesAndMasks,
  DEFAULT_PREPROCESSING_CONFIG,
} from '../src/lib/changeDetection';
import { FEATURE_KEYS, fitLogisticRegressionModel } from '../src/lib/classifier';
import { alignLaterImage, validateImagePair } from '../src/lib/imageProcessing';
import {
  BuildingFootprintPolygon,
  DatasetManifest,
  DatasetSplitRecord,
  RawImageBuffer,
  SampleManifestEntry,
  SpatialMetadata,
} from '../src/types/satellite';

const S3_BASE = 'https://spacenet-dataset.s3.amazonaws.com';
const PUBLIC_DATA_DIR = path.resolve(process.cwd(), 'public/data/spacenet7');
const SRC_DATA_DIR = path.resolve(process.cwd(), 'src/data');

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

interface CandidateAoiSpec {
  aoi: string;
  regionLabel: string;
  targetPositive: boolean;
  preferredTest?: boolean;
  shortSeasonalWindow?: boolean;
}

/**
 * Curated list of geographically distinct official SpaceNet 7 `train/` AOIs (>250 km pairwise separation).
 * Each AOI appears AT MOST ONCE in the entire dataset to guarantee zero spatial or tile leakage.
 */
const DISTINCT_AOI_SPECS: CandidateAoiSpec[] = [
  // --- HELD-OUT TEST CANDIDATES (8 distinct global regions: 4 Positive, 4 Negative) ---
  {
    aoi: 'L15-0361E-1300N_1446_2989_13',
    regionLabel: 'Meridian / Boise Metro, Idaho, USA',
    targetPositive: true,
    preferredTest: true,
  },
  {
    aoi: 'L15-1296E-1198N_5184_3399_13',
    regionLabel: 'West Abdullah Al-Mubarak, Kuwait',
    targetPositive: true,
    preferredTest: true,
  },
  {
    aoi: 'L15-1848E-0793N_7394_5018_13',
    regionLabel: 'Northern Growth Corridor, Melbourne, Australia',
    targetPositive: true,
    preferredTest: true,
  },
  {
    aoi: 'L15-0566E-1185N_2265_3451_13',
    regionLabel: 'Port St. Lucie Corridor, Florida, USA',
    targetPositive: true,
    preferredTest: true,
  },
  {
    aoi: 'L15-1049E-1370N_4196_2710_13',
    regionLabel: 'Rotterdam Polder & Urban Fringe, Netherlands',
    targetPositive: false,
    preferredTest: true,
    shortSeasonalWindow: false,
  },
  {
    aoi: 'L15-0595E-1278N_2383_3079_13',
    regionLabel: 'Lehigh Valley Agricultural Fringe, Pennsylvania, USA',
    targetPositive: false,
    preferredTest: true,
    shortSeasonalWindow: true,
  },
  {
    aoi: 'L15-0434E-1218N_1736_3318_13',
    regionLabel: 'Permian Basin Semi-Arid Fringe, New Mexico, USA',
    targetPositive: false,
    preferredTest: true,
    shortSeasonalWindow: false,
  },
  {
    aoi: 'L15-1389E-1284N_5557_3054_13',
    regionLabel: 'Bukhara Oasis Fringe, Uzbekistan',
    targetPositive: false,
    preferredTest: true,
    shortSeasonalWindow: true,
  },

  // --- TRAINING CANDIDATES (Geographically separate from all Test AOIs and each other) ---
  // Positive Training AOIs (Active Multi-Building Construction)
  {
    aoi: 'L15-0683E-1006N_2732_4164_13',
    regionLabel: 'Manaus Northern Expansion, Brazil',
    targetPositive: true,
  },
  {
    aoi: 'L15-0614E-0946N_2459_4406_13',
    regionLabel: 'Cusco Valley Corridor, Peru',
    targetPositive: true,
  },
  {
    aoi: 'L15-1669E-1160N_6679_3549_13',
    regionLabel: 'Guangzhou Northern District, China',
    targetPositive: true,
  },
  {
    aoi: 'L15-1204E-1202N_4816_3380_13',
    regionLabel: 'East Cairo New Urban District, Egypt',
    targetPositive: true,
  },
  {
    aoi: 'L15-1138E-1216N_4553_3325_13',
    regionLabel: 'Benghazi Coastal Plain, Libya',
    targetPositive: true,
  },
  {
    aoi: 'L15-0586E-1127N_2345_3680_13',
    regionLabel: 'Spanish Town / Kingston West, Jamaica',
    targetPositive: true,
  },
  {
    aoi: 'L15-1209E-1113N_4838_3737_13',
    regionLabel: 'Khartoum North Corridor, Sudan',
    targetPositive: true,
  },
  {
    aoi: 'L15-0358E-1220N_1433_3310_13',
    regionLabel: 'Otay Ranch / Chula Vista, California, USA',
    targetPositive: true,
  },
  {
    aoi: 'L15-1015E-1062N_4061_3941_13',
    regionLabel: 'Kumasi Peri-Urban Belt, Ghana',
    targetPositive: true,
  },
  {
    aoi: 'L15-1276E-1107N_5105_3761_13',
    regionLabel: 'Dhamar Highlands, Yemen',
    targetPositive: true,
  },
  {
    aoi: 'L15-1335E-1166N_5342_3524_13',
    regionLabel: 'Al Shamkha, Abu Dhabi, UAE',
    targetPositive: true,
  },
  {
    aoi: 'L15-1185E-0935N_4742_4450_13',
    regionLabel: 'Lusaka Eastern Fringe, Zambia',
    targetPositive: true,
  },
  {
    aoi: 'L15-1709E-1112N_6838_3742_13',
    regionLabel: 'Central Luzon Corridor, Philippines',
    targetPositive: true,
  },
  {
    aoi: 'L15-1481E-1119N_5927_3715_13',
    regionLabel: 'Amaravati / Vijayawada, India',
    targetPositive: true,
  },

  // Negative Training AOIs (Zero or Minimal Building Change with Seasonal/Radiometric Variation)
  {
    aoi: 'L15-1014E-1375N_4056_2688_13',
    regionLabel: 'West Midlands Greenbelt, United Kingdom',
    targetPositive: false,
    shortSeasonalWindow: true,
  },
  {
    aoi: 'L15-0632E-0892N_2528_4620_13',
    regionLabel: 'Calama Arid Plateau, Chile',
    targetPositive: false,
    shortSeasonalWindow: true,
  },
  {
    aoi: 'L15-0760E-0887N_3041_4643_13',
    regionLabel: 'Eastern Sao Paulo Metro Fringe, Brazil',
    targetPositive: false,
    shortSeasonalWindow: true,
  },
  {
    aoi: 'L15-0487E-1246N_1950_3207_13',
    regionLabel: 'Bentonville Rural Fringe, Arkansas, USA',
    targetPositive: false,
    shortSeasonalWindow: true,
  },
  {
    aoi: 'L15-1200E-0847N_4802_4803_13',
    regionLabel: 'KwaZulu-Natal Coastal Hills, South Africa',
    targetPositive: false,
    shortSeasonalWindow: true,
  },
  {
    aoi: 'L15-0924E-1108N_3699_3757_13',
    regionLabel: 'Dakar Peninsula Sector, Senegal',
    targetPositive: false,
    shortSeasonalWindow: true,
  },
  {
    aoi: 'L15-0331E-1257N_1327_3160_13',
    regionLabel: 'East Contra Costa Fringe, California, USA',
    targetPositive: false,
    shortSeasonalWindow: true,
  },
  {
    aoi: 'L15-0506E-1204N_2027_3374_13',
    regionLabel: 'Ascension Parish Lowlands, Louisiana, USA',
    targetPositive: false,
    shortSeasonalWindow: true,
  },
  {
    aoi: 'L15-1748E-1247N_6993_3202_13',
    regionLabel: 'Sejong / Geum River Basin, South Korea',
    targetPositive: false,
    shortSeasonalWindow: true,
  },
  {
    aoi: 'L15-1172E-1306N_4688_2967_13',
    regionLabel: 'Ilfov Plain, Bucharest Fringe, Romania',
    targetPositive: false,
    shortSeasonalWindow: true,
  },
  {
    aoi: 'L15-1716E-1211N_6864_3345_13',
    regionLabel: 'Pudong Eastern Agricultural Belt, China',
    targetPositive: false,
    shortSeasonalWindow: true,
  },
  {
    aoi: 'L15-0544E-1228N_2176_3279_13',
    regionLabel: 'DeKalb Wooded Canopy, Georgia, USA',
    targetPositive: false,
    shortSeasonalWindow: true,
  },
  {
    aoi: 'L15-1289E-1169N_5156_3514_13',
    regionLabel: 'Riyadh Northern Plateau, Saudi Arabia',
    targetPositive: false,
    shortSeasonalWindow: true,
  },
  {
    aoi: 'L15-1617E-1207N_6468_3360_13',
    regionLabel: 'Sichuan Basin Farmland, China',
    targetPositive: false,
    shortSeasonalWindow: true,
  },
];

async function fetchGeoTiffSubtile(
  s3Key: string
): Promise<{
  fullWidth: number;
  fullHeight: number;
  bboxMercator: [number, number, number, number];
  rasters: Uint8Array[];
}> {
  const res = await fetch(`${S3_BASE}/${s3Key}`);
  if (!res.ok) throw new Error(`HTTP ${res.status} fetching ${s3Key}`);
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

interface ParsedFootprint {
  id: number;
  ring: [number, number][];
  centroid: [number, number];
  areaPx2: number;
}

function parsePixGeojson(geojson: any, imgHeight: number): ParsedFootprint[] {
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

    // Note: SpaceNet 7 `labels_match_pix` uses origin at bottom-left or top-left?
    // Let's preserve exact [x, y] and verify orientation below!
    const ring: [number, number][] = rawRing.map((pt) => [
      Number(pt[0]),
      Number(imgHeight - pt[1]),
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
}

/**
 * Checks whether SpaceNet 7 labels_match_pix y-coordinates match top-down image rows directly
 * or bottom-up rows by comparing against geographic labels_match on the first AOI.
 */
async function detectPixYOrientation(aoi: string, monthStr: string): Promise<boolean> {
  const [pixGeo, mercGeo, tiffMeta] = await Promise.all([
    fetch(
      `${S3_BASE}/spacenet/SN7_buildings/train/${aoi}/labels_match_pix/global_monthly_${monthStr}_mosaic_${aoi}_Buildings.geojson`
    ).then((r) => r.json()),
    fetch(
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
  // In top-down raster coordinates, row 0 is maxY (North) and row H is minY (South)
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

  const flipPixY = await detectPixYOrientation(
    'L15-0331E-1257N_1327_3160_13',
    '2018_01'
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
  const QUADRANTS: [number, number][] = [
    [0, 0],
    [512, 0],
    [0, 511],
    [512, 511],
    [256, 255],
  ];

  const splitRecords: DatasetSplitRecord[] = [];
  const manifestSamples: SampleManifestEntry[] = [];

  console.log(`Processing ${DISTINCT_AOI_SPECS.length} distinct SpaceNet 7 AOIs...`);

  for (let idx = 0; idx < DISTINCT_AOI_SPECS.length; idx++) {
    const spec = DISTINCT_AOI_SPECS[idx];
    try {
      const listXml = await (
        await fetch(
          `${S3_BASE}/?prefix=spacenet/SN7_buildings/train/${spec.aoi}/labels_match_pix/`
        )
      ).text();
      const labelKeys = [...listXml.matchAll(/<Key>([^<]+\.geojson)<\/Key>/g)]
        .map((m) => m[1])
        .sort();
      if (labelKeys.length < 6) {
        console.warn(`Skipping ${spec.aoi}: fewer than 6 months`);
        continue;
      }

      const months = labelKeys
        .map((k) => k.match(/global_monthly_(\d{4}_\d{2})_mosaic/)?.[1])
        .filter((m): m is string => Boolean(m));

      // Pick clean cloud-free earlier and later candidate months
      const candidatePairs: [string, string][] = !spec.targetPositive
        ? [
            [months[Math.min(3, months.length - 2)], months[Math.min(7, months.length - 1)]],
            [months[Math.min(4, months.length - 2)], months[Math.min(8, months.length - 1)]],
            [months[Math.min(2, months.length - 2)], months[Math.min(5, months.length - 1)]],
            [months[0], months[Math.min(3, months.length - 1)]],
            [months[0], months[months.length - 1]],
          ]
        : [
            [months[0], months[months.length - 1]],
            [months[1], months[months.length - 2]],
            [months[2], months[months.length - 3]],
          ];

      let chosen: {
        m1: string;
        m2: string;
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
      } | null = null;

      for (const [m1, m2] of candidatePairs) {
        if (!m1 || !m2 || m1 === m2) continue;
        const s3Img1 = `spacenet/SN7_buildings/train/${spec.aoi}/images_masked/global_monthly_${m1}_mosaic_${spec.aoi}.tif`;
        const s3Img2 = `spacenet/SN7_buildings/train/${spec.aoi}/images_masked/global_monthly_${m2}_mosaic_${spec.aoi}.tif`;
        const s3Lbl1 = `spacenet/SN7_buildings/train/${spec.aoi}/labels_match_pix/global_monthly_${m1}_mosaic_${spec.aoi}_Buildings.geojson`;
        const s3Lbl2 = `spacenet/SN7_buildings/train/${spec.aoi}/labels_match_pix/global_monthly_${m2}_mosaic_${spec.aoi}_Buildings.geojson`;

        const [t1, t2, g1, g2] = await Promise.all([
          fetchGeoTiffSubtile(s3Img1),
          fetchGeoTiffSubtile(s3Img2),
          fetch(`${S3_BASE}/${s3Lbl1}`).then((r) => r.json()),
          fetch(`${S3_BASE}/${s3Lbl2}`).then((r) => r.json()),
        ]);

        if (t1.fullWidth < 512 || t1.fullHeight < 512) continue;
        const fp1All = parseFootprintsOriented(g1, t1.fullHeight);
        const fp2All = parseFootprintsOriented(g2, t2.fullHeight);

        // Score quadrants to find cloud-free sub-tile matching targetPositive criterion
        for (const [qx, qy] of QUADRANTS) {
          const x0 = Math.min(qx, t1.fullWidth - CROP_SIZE);
          const y0 = Math.min(qy, t1.fullHeight - CROP_SIZE);
          const eBuf = extractCropRgba(t1, x0, y0, CROP_SIZE, CROP_SIZE);
          const lBuf = extractCropRgba(t2, x0, y0, CROP_SIZE, CROP_SIZE);

          const val = validateImagePair(eBuf, lBuf);
          if (!val.valid || val.validClearPixelRatio < 0.97) continue;

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

          const [minX, minY, maxX, maxY] = t1.bboxMercator;
          const gsdX = (maxX - minX) / t1.fullWidth;
          const gsdY = (maxY - minY) / t1.fullHeight;
          const centerMercY = (minY + maxY) * 0.5;
          const latRad =
            2 * Math.atan(Math.exp(centerMercY / 6378137)) - Math.PI / 2;
          // True ground sample distance corrected for Web Mercator scale factor cos(lat)
          const gsdMeters = Number((((gsdX + gsdY) * 0.5) * Math.cos(latRad)).toFixed(3));
          const pxAreaM2 = gsdMeters * gsdMeters;

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

          if ((spec.targetPositive && meetsPositive) || (!spec.targetPositive && meetsNegative)) {
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
              (spec.targetPositive && newCount > chosen.newCount) ||
              (!spec.targetPositive && newCount < chosen.newCount)
            ) {
              chosen = {
                m1,
                m2,
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

      if (!chosen) {
        console.warn(`No qualifying clean sub-tile found for ${spec.aoi} (targetPos=${spec.targetPositive})`);
        continue;
      }

      const split: 'test' | 'train' = spec.preferredTest ? 'test' : 'train';
      const pairId =
        split === 'test'
          ? `sn7-test-${String(manifestSamples.length + 1).padStart(2, '0')}`
          : `sn7-train-${String(splitRecords.filter((r) => r.split === 'train').length + 1).padStart(2, '0')}`;

      // Run exact shared pipeline feature extraction
      const { alignedLater } = alignLaterImage(
        chosen.eBuf,
        chosen.lBuf,
        DEFAULT_PREPROCESSING_CONFIG.enableAlignment
      );
      const pass = computeFeaturesAndMasks(
        chosen.eBuf,
        alignedLater,
        DEFAULT_PREPROCESSING_CONFIG,
        chosen.spatial.gsdMetersPerPx
      );

      const pxAreaM2 = chosen.spatial.gsdMetersPerPx * chosen.spatial.gsdMetersPerPx;
      const newBuildingAreaM2 = Number((chosen.newAreaPx2 * pxAreaM2).toFixed(1));

      splitRecords.push({
        pairId,
        aoiId: spec.aoi,
        split,
        lat: chosen.spatial.centerWgs84.lat,
        lon: chosen.spatial.centerWgs84.lon,
        earlierDate: chosen.m1.replace('_', '-'),
        laterDate: chosen.m2.replace('_', '-'),
        newBuildings: chosen.newCount,
        newBuildingAreaM2,
        groundTruthPositive: spec.targetPositive,
        features: pass.features,
      });

      console.log(
        `[${split.toUpperCase()}] ${pairId} | ${spec.regionLabel} | ${chosen.m1}->${chosen.m2} | newBldgs=${chosen.newCount} (${newBuildingAreaM2} m²) | structPct=${pass.features.structuralChangePct}% albedoPct=${pass.features.highAlbedoGainPct}% edge=${pass.features.meanClusterEdgeGain} veg=${pass.features.seasonalVegShiftFraction}`
      );

      // Save all held-out TEST samples into public/data/spacenet7 for immediate preloaded demo!
      if (split === 'test') {
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
            groundTruthPositive: spec.targetPositive,
            groundTruthClassLabel: spec.targetPositive
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
    } catch (err: any) {
      console.warn(`Error processing ${spec.aoi}: ${err.message}`);
    }
  }

  // Verify minimum geographic distance between ANY train record and ANY test record
  const trainRecs = splitRecords.filter((r) => r.split === 'train');
  const testRecs = splitRecords.filter((r) => r.split === 'test');
  let minTrainTestKm = Infinity;
  for (const tr of trainRecs) {
    for (const te of testRecs) {
      const dist = haversineKm(tr.lat, tr.lon, te.lat, te.lon);
      if (dist < minTrainTestKm) {
        minTrainTestKm = dist;
      }
    }
  }
  console.log(
    `\nVerified Train (${trainRecs.length}) vs Test (${testRecs.length}) split | Minimum Train-Test geographic separation: ${minTrainTestKm.toFixed(
      1
    )} km`
  );

  // Train calibrated L2-regularized Logistic Regression strictly on Train records
  const modelArtifact = fitLogisticRegressionModel({
    records: splitRecords,
    lambda: 1.0,
    positiveCriterion:
      ' Ground-truth SpaceNet 7 annotations show >= 12 newly constructed building footprints AND >= 1,500 m² new building footprint area within the 512x512 sub-tile.',
    negativeCriterion:
      'Ground-truth SpaceNet 7 annotations show <= 2 newly constructed or removed building footprints AND <= 350 m² new building area within the 512x512 sub-tile.',
    leakagePrevention:
      'Strict AOI-level geographic split: at most one 512x512 sub-tile extracted per SpaceNet 7 AOI, with all training AOIs geographically disjoint from all held-out test AOIs.',
    minTrainTestDistanceKm: minTrainTestKm,
    datasetSource:
      'Official SpaceNet 7 Multi-Temporal Urban Development Dataset (s3://spacenet-dataset/spacenet/SN7_buildings/train/)',
  });

  console.log('\n--- MODEL TRAINING & HELD-OUT TEST EVALUATION ---');
  console.log('Train Metrics:', JSON.stringify(modelArtifact.trainMetrics));
  console.log('Held-Out Test Metrics:', JSON.stringify(modelArtifact.testMetrics));
  console.log('Weights:', modelArtifact.weights, 'Intercept:', modelArtifact.intercept);

  const manifest: DatasetManifest = {
    datasetName: 'SpaceNet 7 Multi-Temporal Urban Development — Preloaded Evaluation Pack',
    officialSourceUrl: 'https://spacenet.ai/sn7-challenge/',
    awsBucketUri: 's3://spacenet-dataset/spacenet/SN7_buildings/train/',
    license: 'Creative Commons Attribution 4.0 International (CC-BY-4.0)',
    attribution:
      'SpaceNet LLC, CosmiQ Works, Planet Labs Inc., Capella Space, Maxar Technologies, IEEE GRSS, Amazon Web Services.',
    generatedAt: new Date().toISOString(),
    isGenuineSpaceNet7: true,
    methodologyNote:
      'Each preloaded sample is a genuine native-resolution 512x512 pixel sub-tile extracted directly from official SpaceNet 7 Planet Dove monthly GeoTIFF mosaics and paired with its official labels_match_pix tracked building-footprint annotations. All preloaded interactive samples belong strictly to the held-out test split (never seen during classifier parameter fitting).',
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

  // Also export full empirical feature table as CSV for transparency & reproducibility
  const csvHeader = [
    'pairId',
    'aoiId',
    'split',
    'lat',
    'lon',
    'earlierDate',
    'laterDate',
    'newBuildings',
    'newBuildingAreaM2',
    'groundTruthPositive',
    ...FEATURE_KEYS,
    'predictedProbability',
    'predictedPositive',
  ].join(',');

  const csvRows = modelArtifact.records.map((r) =>
    [
      r.pairId,
      r.aoiId,
      r.split,
      r.lat,
      r.lon,
      r.earlierDate,
      r.laterDate,
      r.newBuildings,
      r.newBuildingAreaM2,
      r.groundTruthPositive ? 1 : 0,
      ...FEATURE_KEYS.map((k) => r.features[k]),
      r.predictedProbability ?? '',
      r.predictedPositive ? 1 : 0,
    ].join(',')
  );
  fs.writeFileSync(
    path.join(SRC_DATA_DIR, 'spacenet7_feature_dataset.csv'),
    [csvHeader, ...csvRows].join('\n')
  );

  console.log(
    `\nSuccessfully wrote ${manifestSamples.length} preloaded held-out test pairs to public/data/spacenet7/ and modelArtifact.json!`
  );
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
