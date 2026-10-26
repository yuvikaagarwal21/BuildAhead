import React, { useEffect, useRef } from 'react';
import type {
  BuildingFootprintPolygon,
  RawImageBuffer,
} from '../types/satellite.ts';

interface SatelliteCanvasProps {
  title: string;
  subtitle: string;
  buffer: RawImageBuffer | null;
  footprints?: BuildingFootprintPolygon[];
  showFootprints?: boolean;
  footprintFilter?: 'earlier_only' | 'all_tracked' | 'new_only';
  caption?: string;
}

export const SatelliteCanvas: React.FC<SatelliteCanvasProps> = ({
  title,
  subtitle,
  buffer,
  footprints,
  showFootprints = false,
  footprintFilter = 'new_only',
  caption,
}) => {
  const baseCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const overlayCanvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = baseCanvasRef.current;
    if (!canvas || !buffer) return;
    if (canvas.width !== buffer.width) canvas.width = buffer.width;
    if (canvas.height !== buffer.height) canvas.height = buffer.height;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const imgData = new ImageData(
      new Uint8ClampedArray(buffer.data),
      buffer.width,
      buffer.height
    );
    ctx.putImageData(imgData, 0, 0);
  }, [buffer]);

  useEffect(() => {
    const canvas = overlayCanvasRef.current;
    if (!canvas || !buffer) return;
    if (canvas.width !== buffer.width) canvas.width = buffer.width;
    if (canvas.height !== buffer.height) canvas.height = buffer.height;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (showFootprints && footprints && footprints.length > 0) {
      for (const fp of footprints) {
        if (footprintFilter === 'earlier_only' && fp.status === 'new_construction') {
          continue;
        }
        if (footprintFilter === 'new_only' && fp.status !== 'new_construction') {
          continue;
        }
        if (fp.ring.length < 3) continue;

        ctx.beginPath();
        ctx.moveTo(fp.ring[0][0], fp.ring[0][1]);
        for (let i = 1; i < fp.ring.length; i++) {
          ctx.lineTo(fp.ring[i][0], fp.ring[i][1]);
        }
        ctx.closePath();

        if (fp.status === 'new_construction') {
          ctx.strokeStyle = 'rgba(16, 185, 129, 0.95)';
          ctx.fillStyle = 'rgba(16, 185, 129, 0.22)';
          ctx.lineWidth = 1.6;
          ctx.fill();
          ctx.stroke();
        } else {
          ctx.strokeStyle = 'rgba(148, 163, 184, 0.65)';
          ctx.lineWidth = 1;
          ctx.stroke();
        }
      }
    }
  }, [buffer, footprints, showFootprints, footprintFilter]);

  return (
    <div className="flex flex-col bg-white border border-slate-200 rounded-xl overflow-hidden">
      <div className="px-4 py-3 border-b border-slate-200 flex items-baseline justify-between gap-2">
        <h3 className="text-sm font-semibold text-slate-900 truncate">{title}</h3>
        <span className="text-xs font-mono tabular-nums text-slate-500 whitespace-nowrap shrink-0">
          {subtitle}
        </span>
      </div>

      <div className="relative aspect-square w-full bg-slate-900 overflow-hidden">
        {buffer ? (
          <>
            <canvas
              ref={baseCanvasRef}
              className="w-full h-full block object-contain"
            />
            <canvas
              ref={overlayCanvasRef}
              className="absolute inset-0 w-full h-full block object-contain pointer-events-none"
            />
          </>
        ) : (
          <div className="w-full h-full flex items-center justify-center text-xs text-slate-400">
            Loading satellite image...
          </div>
        )}
      </div>

      {caption && (
        <div className="px-4 py-2.5 border-t border-slate-200 text-xs text-slate-600">
          {caption}
        </div>
      )}
    </div>
  );
};
