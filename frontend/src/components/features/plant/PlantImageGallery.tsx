'use client';

import { useState, useRef, ChangeEvent } from 'react';
import Image from 'next/image';
import { Plus, X, Trash2, ChevronLeft } from 'lucide-react';
import {
  usePlantImagesQuery,
  useUploadPlantImageMutation,
  useDeletePlantImageMutation,
} from '@/hooks/usePlants';
import type { UserPlantImage } from '@/types';

interface PlantImageGalleryProps {
  plantId: number;
  plantName: string;
  plantSpecies?: string;
  healthPercent: number;
  onBack?: () => void;
  darkMode?: boolean;
}

function imageUrlFor(image: UserPlantImage): string {
  // SeaweedFS fid is "<volumeId>,<fileKey>". The dev/prod proxy at /blob/
  // forwards directly to the blob store with no auth (parity with the existing
  // single-image flow at GET /my-plants/{id}/image which also uses X-Accel-Redirect).
  return `/blob/${image.fid}`;
}

/**
 * Plant detail hero AND photo gallery in one component.
 *  ┌─────────────────────┬─────────┐
 *  │ ← back   70% HEALTH │ prev-1  │   (overlays on latest)
 *  │                     ├─────────┤
 *  │   latest photo      │ prev-2  │
 *  │  name / species ↓   │ +N more │
 *  └─────────────────────┴─────────┘
 * Empty state: three grey tiles, latest has a big `+` icon and acts as the
 * upload trigger. Right-column tiles are passive placeholders.
 */
export function PlantImageGallery({
  plantId,
  plantName,
  plantSpecies,
  healthPercent,
  onBack,
  darkMode = false,
}: PlantImageGalleryProps) {
  const { data: images = [] } = usePlantImagesQuery(plantId);
  const uploadMutation = useUploadPlantImageMutation(plantId);
  const deleteMutation = useDeletePlantImageMutation(plantId);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [lightboxIndex, setLightboxIndex] = useState<number | null>(null);

  const handleFileSelect = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) uploadMutation.mutate(file);
    e.target.value = '';
  };
  const openUpload = () => fileInputRef.current?.click();

  const latest = images[0];
  const prev1 = images[1];
  const prev2 = images[2];
  const extraCount = Math.max(0, images.length - 3);
  const isEmpty = images.length === 0;

  const greyTileClass = darkMode
    ? 'bg-neutral-800 border border-neutral-700'
    : 'bg-neutral-200 border border-neutral-300';

  return (
    <div className="rounded-3xl overflow-hidden">
      <div className="grid grid-cols-2 gap-2 aspect-[16/9] sm:aspect-[3/1] lg:aspect-[4/1]">
        {/* LEFT: latest (or empty + upload trigger) — full hero with overlays */}
        <button
          type="button"
          onClick={isEmpty ? openUpload : () => setLightboxIndex(0)}
          disabled={uploadMutation.isPending}
          className={`relative row-span-2 overflow-hidden rounded-2xl ${
            isEmpty ? `${greyTileClass} cursor-pointer` : 'bg-neutral-200 dark:bg-neutral-900 cursor-pointer'
          }`}
          aria-label={isEmpty ? 'Add first photo' : 'Open latest photo'}
        >
          {latest ? (
            <Image
              src={imageUrlFor(latest)}
              alt={plantName}
              fill
              className="object-cover"
              sizes="50vw"
              unoptimized
              priority
            />
          ) : (
            <div className="absolute inset-0 flex items-center justify-center">
              <div
                className={`w-24 h-24 rounded-full flex items-center justify-center ${
                  darkMode ? 'bg-neutral-700/60' : 'bg-white/70'
                }`}
              >
                <Plus
                  size={56}
                  className={darkMode ? 'text-neutral-400' : 'text-neutral-500'}
                />
              </div>
            </div>
          )}

          {/* Dark gradient + overlays for text readability */}
          <div className="absolute inset-0 bg-gradient-to-b from-black/40 via-transparent to-black/60 pointer-events-none" />

          {/* Top-left back button */}
          {onBack && (
            <div
              onClick={(e) => {
                e.stopPropagation();
                onBack();
              }}
              className="absolute top-3 left-3 w-10 h-10 rounded-full bg-white/90 hover:bg-white shadow-md flex items-center justify-center cursor-pointer"
              role="button"
              aria-label="Back to collection"
            >
              <ChevronLeft size={20} className="text-neutral-700" />
            </div>
          )}

          {/* Top-right health badge */}
          <div className="absolute top-3 right-3 px-3 py-1 rounded-full bg-nature text-white text-xs font-bold shadow-md">
            {healthPercent}% HEALTH
          </div>

          {/* Bottom-left name + species */}
          <div className="absolute bottom-4 left-4 right-4 text-white">
            <h1 className="text-xl sm:text-2xl font-bold drop-shadow-md">{plantName}</h1>
            {plantSpecies && (
              <p className="italic text-sm opacity-90 drop-shadow-md">{plantSpecies}</p>
            )}
          </div>

          {/* Bottom-right add-photo pill (only when gallery is non-empty;
              empty-state click already triggers upload via the tile itself). */}
          {!isEmpty && (
            <div
              onClick={(e) => {
                e.stopPropagation();
                openUpload();
              }}
              className="absolute bottom-4 right-4 flex items-center gap-1 px-3 py-1.5 rounded-full bg-white/90 hover:bg-white text-secondary text-xs font-medium cursor-pointer shadow-md"
              role="button"
              aria-label="Add photo"
            >
              <Plus size={14} />
              {uploadMutation.isPending ? 'Uploading…' : 'Add photo'}
            </div>
          )}
        </button>

        {/* RIGHT TOP: prev-1 */}
        {prev1 ? (
          <button
            type="button"
            onClick={() => setLightboxIndex(1)}
            className="relative rounded-2xl overflow-hidden bg-neutral-200 dark:bg-neutral-900 cursor-pointer"
          >
            <Image
              src={imageUrlFor(prev1)}
              alt="Previous photo"
              fill
              className="object-cover"
              sizes="25vw"
              unoptimized
            />
          </button>
        ) : (
          <div className={`rounded-2xl ${greyTileClass}`} />
        )}

        {/* RIGHT BOTTOM: prev-2 (with +N more overlay if there are extras) */}
        {prev2 ? (
          <button
            type="button"
            onClick={() => setLightboxIndex(2)}
            className="relative rounded-2xl overflow-hidden bg-neutral-200 dark:bg-neutral-900 cursor-pointer"
          >
            <Image
              src={imageUrlFor(prev2)}
              alt="Older photo"
              fill
              className="object-cover"
              sizes="25vw"
              unoptimized
            />
            {extraCount > 0 && (
              <div className="absolute inset-0 bg-black/55 flex items-center justify-center text-white font-semibold text-sm">
                +{extraCount} more
              </div>
            )}
          </button>
        ) : (
          <div className={`rounded-2xl ${greyTileClass}`} />
        )}
      </div>

      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        className="hidden"
        onChange={handleFileSelect}
      />

      {/* Lightbox */}
      {lightboxIndex !== null && images[lightboxIndex] && (
        <div
          className="fixed inset-0 z-50 bg-black/90 flex items-center justify-center p-4"
          onClick={() => setLightboxIndex(null)}
        >
          <button
            type="button"
            onClick={(e) => {
              e.stopPropagation();
              setLightboxIndex(null);
            }}
            className="absolute top-4 right-4 text-white p-2 hover:bg-white/10 rounded-full"
            aria-label="Close"
          >
            <X size={24} />
          </button>
          <button
            type="button"
            onClick={(e) => {
              e.stopPropagation();
              const id = images[lightboxIndex].id;
              if (window.confirm('Delete this photo?')) {
                deleteMutation.mutate(id);
                setLightboxIndex(null);
              }
            }}
            className="absolute top-4 left-4 text-red-400 p-2 hover:bg-white/10 rounded-full"
            aria-label="Delete photo"
          >
            <Trash2 size={20} />
          </button>
          <div
            className="relative max-w-4xl max-h-[80vh] w-full h-full"
            onClick={(e) => e.stopPropagation()}
          >
            <Image
              src={imageUrlFor(images[lightboxIndex])}
              alt="Plant photo"
              fill
              className="object-contain"
              unoptimized
            />
          </div>
          {images.length > 1 && (
            <div className="absolute bottom-4 left-1/2 -translate-x-1/2 flex gap-2">
              {images.map((_, i) => (
                <button
                  key={i}
                  type="button"
                  onClick={(e) => {
                    e.stopPropagation();
                    setLightboxIndex(i);
                  }}
                  className={`w-2 h-2 rounded-full ${
                    i === lightboxIndex ? 'bg-white' : 'bg-white/40'
                  }`}
                  aria-label={`Photo ${i + 1}`}
                />
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
