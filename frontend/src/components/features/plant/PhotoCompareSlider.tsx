'use client';

import { useRef, useState, PointerEvent as ReactPointerEvent } from 'react';
import Image from 'next/image';

interface PhotoCompareSliderProps {
  beforeUrl: string;
  beforeLabel: string;
  afterUrl: string;
  afterLabel: string;
}

export const PhotoCompareSlider: React.FC<PhotoCompareSliderProps> = ({
  beforeUrl,
  beforeLabel,
  afterUrl,
  afterLabel,
}) => {
  const [pct, setPct] = useState(50);
  const containerRef = useRef<HTMLDivElement>(null);
  const draggingRef = useRef(false);

  const updateFromClientX = (clientX: number) => {
    const el = containerRef.current;
    if (!el) return;
    const rect = el.getBoundingClientRect();
    const x = clientX - rect.left;
    const next = Math.max(0, Math.min(100, (x / rect.width) * 100));
    setPct(next);
  };

  const handlePointerDown = (e: ReactPointerEvent<HTMLDivElement>) => {
    draggingRef.current = true;
    (e.target as Element).setPointerCapture?.(e.pointerId);
    updateFromClientX(e.clientX);
  };
  const handlePointerMove = (e: ReactPointerEvent<HTMLDivElement>) => {
    if (!draggingRef.current) return;
    updateFromClientX(e.clientX);
  };
  const handlePointerUp = () => {
    draggingRef.current = false;
  };

  return (
    <div
      ref={containerRef}
      onPointerDown={handlePointerDown}
      onPointerMove={handlePointerMove}
      onPointerUp={handlePointerUp}
      onPointerLeave={handlePointerUp}
      className="relative w-full aspect-square sm:aspect-video rounded-2xl overflow-hidden bg-neutral-900 select-none touch-none cursor-ew-resize"
    >
      {/* object-cover ensures both images fill the same frame regardless of orientation. */}
      <Image
        src={afterUrl}
        alt={afterLabel}
        fill
        className="object-cover pointer-events-none"
        unoptimized
      />

      {/* "Before" image is clipped to the slider percentage on the left half. */}
      <div
        className="absolute inset-0 overflow-hidden pointer-events-none"
        style={{ clipPath: `inset(0 ${100 - pct}% 0 0)` }}
      >
        <Image
          src={beforeUrl}
          alt={beforeLabel}
          fill
          className="object-cover"
          unoptimized
        />
      </div>

      <span className="absolute top-3 left-3 px-2 py-1 rounded-md bg-black/60 text-white text-xs font-medium">
        {beforeLabel}
      </span>
      <span className="absolute top-3 right-3 px-2 py-1 rounded-md bg-black/60 text-white text-xs font-medium">
        {afterLabel}
      </span>

      <div
        className="absolute top-0 bottom-0 w-1 bg-white shadow-[0_0_8px_rgba(0,0,0,0.6)] pointer-events-none"
        style={{ left: `calc(${pct}% - 2px)` }}
      />
      <div
        className="absolute top-1/2 -translate-y-1/2 w-10 h-10 rounded-full bg-white shadow-lg flex items-center justify-center text-neutral-700 text-lg font-bold pointer-events-none"
        style={{ left: `calc(${pct}% - 20px)` }}
        aria-hidden="true"
      >
        ‹›
      </div>
    </div>
  );
};
