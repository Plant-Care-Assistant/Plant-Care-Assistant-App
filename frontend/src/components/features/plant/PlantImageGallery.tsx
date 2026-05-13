'use client';

import { useState, useRef, ChangeEvent } from 'react';
import Image from 'next/image';
import { Plus, X, Trash2 } from 'lucide-react';
import type { UserPlantImage } from '@/types';
import {
  usePlantImagesQuery,
  useUploadPlantImageMutation,
  useDeletePlantImageMutation,
} from '@/hooks/usePlants';

interface PlantImageGalleryProps {
  plantId: number;
  fallbackImageUrl?: string;
  darkMode?: boolean;
}

function imageUrlFor(plantId: number, imageId: number): string {
  return `/api/my-plants/${plantId}/images/${imageId}`;
}

/**
 * Gallery layout:
 *  ┌─────────────┬──────────┐
 *  │             │  prev-1  │
 *  │  latest     ├──────────┤
 *  │             │  prev-2  │
 *  └─────────────┴──────────┘
 *  Below: small "+N more" pill if there are more than 3 photos; click opens
 *  the full gallery in a lightbox.
 */
export function PlantImageGallery({
  plantId,
  fallbackImageUrl,
  darkMode = false,
}: PlantImageGalleryProps) {
  const { data: images = [], isLoading } = usePlantImagesQuery(plantId);
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

  // Empty state: no gallery images yet — show fallback (legacy single image) +
  // a big "Add photo" CTA.
  if (!isLoading && images.length === 0) {
    return (
      <div className={`rounded-2xl p-4 ${darkMode ? 'bg-neutral-800' : 'bg-white'} shadow-sm`}>
        <div className="flex items-center justify-between mb-3">
          <h3 className={`text-sm font-semibold ${darkMode ? 'text-neutral-300' : 'text-neutral-700'}`}>
            Photos
          </h3>
          <button
            type="button"
            onClick={openUpload}
            disabled={uploadMutation.isPending}
            className="flex items-center gap-1 text-xs font-medium text-secondary hover:text-secondary/80 disabled:opacity-50"
          >
            <Plus size={14} />
            {uploadMutation.isPending ? 'Uploading…' : 'Add photo'}
          </button>
        </div>
        <div
          onClick={openUpload}
          className={`relative aspect-video w-full rounded-xl overflow-hidden cursor-pointer ${
            darkMode ? 'bg-neutral-900' : 'bg-neutral-100'
          } flex items-center justify-center`}
        >
          {fallbackImageUrl ? (
            <Image src={fallbackImageUrl} alt="Plant" fill className="object-cover" />
          ) : (
            <div className="flex flex-col items-center gap-2 text-neutral-400">
              <Plus size={32} />
              <span className="text-sm">Add the first photo</span>
            </div>
          )}
        </div>
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          className="hidden"
          onChange={handleFileSelect}
        />
      </div>
    );
  }

  const latest = images[0];
  const prev1 = images[1];
  const prev2 = images[2];
  const extraCount = Math.max(0, images.length - 3);

  return (
    <div className={`rounded-2xl p-4 ${darkMode ? 'bg-neutral-800' : 'bg-white'} shadow-sm`}>
      <div className="flex items-center justify-between mb-3">
        <h3 className={`text-sm font-semibold ${darkMode ? 'text-neutral-300' : 'text-neutral-700'}`}>
          Photos ({images.length})
        </h3>
        <button
          type="button"
          onClick={openUpload}
          disabled={uploadMutation.isPending}
          className="flex items-center gap-1 text-xs font-medium text-secondary hover:text-secondary/80 disabled:opacity-50"
        >
          <Plus size={14} />
          {uploadMutation.isPending ? 'Uploading…' : 'Add photo'}
        </button>
      </div>

      <div className="grid grid-cols-2 gap-2 aspect-video">
        {/* Latest — left half, full height */}
        {latest && (
          <button
            type="button"
            onClick={() => setLightboxIndex(0)}
            className="relative row-span-2 rounded-xl overflow-hidden bg-neutral-200 dark:bg-neutral-900 cursor-pointer"
          >
            <Image
              src={imageUrlFor(plantId, latest.id)}
              alt="Latest photo"
              fill
              className="object-cover"
              sizes="50vw"
              unoptimized
            />
          </button>
        )}

        {/* Right side: previous 1 (top) + previous 2 (bottom) */}
        <div className="grid grid-rows-2 gap-2">
          {prev1 ? (
            <button
              type="button"
              onClick={() => setLightboxIndex(1)}
              className="relative rounded-xl overflow-hidden bg-neutral-200 dark:bg-neutral-900 cursor-pointer"
            >
              <Image
                src={imageUrlFor(plantId, prev1.id)}
                alt="Previous photo"
                fill
                className="object-cover"
                sizes="25vw"
                unoptimized
              />
            </button>
          ) : (
            <div className="rounded-xl bg-neutral-100 dark:bg-neutral-900" />
          )}
          {prev2 ? (
            <button
              type="button"
              onClick={() => setLightboxIndex(2)}
              className="relative rounded-xl overflow-hidden bg-neutral-200 dark:bg-neutral-900 cursor-pointer"
            >
              <Image
                src={imageUrlFor(plantId, prev2.id)}
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
            <div className="rounded-xl bg-neutral-100 dark:bg-neutral-900" />
          )}
        </div>
      </div>

      {/* Three dots indicator below the grid if there's anything beyond the 3 tiles */}
      {extraCount > 0 && (
        <button
          type="button"
          onClick={() => setLightboxIndex(0)}
          className={`mt-3 w-full flex items-center justify-center gap-1 text-xs ${
            darkMode ? 'text-neutral-400' : 'text-neutral-500'
          } hover:text-secondary`}
        >
          <span className="text-base leading-none">•</span>
          <span className="text-base leading-none">•</span>
          <span className="text-base leading-none">•</span>
          <span className="ml-1">see all {images.length} photos</span>
        </button>
      )}

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
          <div className="relative max-w-4xl max-h-[80vh] w-full h-full" onClick={(e) => e.stopPropagation()}>
            <Image
              src={imageUrlFor(plantId, images[lightboxIndex].id)}
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
