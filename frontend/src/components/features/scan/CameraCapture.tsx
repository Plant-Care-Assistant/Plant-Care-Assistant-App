'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import Webcam from 'react-webcam';
import { Check, Loader2, RefreshCw, X, Camera as CameraIcon } from 'lucide-react';

export interface CameraCaptureProps {
  isOpen: boolean;
  onClose: () => void;
  onCapture: (image: Blob) => void;
}

/**
 * Full-screen camera overlay for capturing plant photos.
 * - Uses device camera with a bottom capture control similar to mobile camera apps.
 * - Provides preview + confirm/retake flow before emitting the captured blob.
 */
export function CameraCapture({ isOpen, onClose, onCapture }: CameraCaptureProps) {
  const webcamRef = useRef<Webcam>(null);
  const [capturedDataUrl, setCapturedDataUrl] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [permissionError, setPermissionError] = useState<string | null>(null);
  const [hasPermission, setHasPermission] = useState<boolean | null>(null);

  const videoConstraints = {
    facingMode: { ideal: 'environment' },
    width: { ideal: 1920 },
    height: { ideal: 1080 },
  } satisfies MediaTrackConstraints;

  const resetState = () => {
    setCapturedDataUrl(null);
    setPermissionError(null);
    setHasPermission(null);
    setIsLoading(false);
  };

  useEffect(() => {
    if (isOpen) {
      // Just set initial loading state - let react-webcam handle the actual stream
      setIsLoading(true);
      
      // Warn developers about HTTPS requirement
      if (typeof window !== 'undefined' && window.location.protocol === 'http:' && window.location.hostname !== 'localhost') {
        console.warn(
          '⚠️ Camera access requires HTTPS on mobile devices.\n' +
          'Run: npm run dev:https\n' +
          'See: CAMERA_HTTPS_GUIDE.md for setup instructions'
        );
      }
    } else {
      resetState();
    }

    // Cleanup function - stop webcam stream when closing
    return () => {
      if (webcamRef.current) {
        const stream = webcamRef.current.video?.srcObject as MediaStream;
        if (stream) {
          stream.getTracks().forEach((track) => track.stop());
        }
      }
      resetState();
    };
  }, [isOpen]);

  // Add timeout for camera access - prevent infinite loading state
  useEffect(() => {
    if (isOpen && isLoading) {
      const timeoutId = setTimeout(() => {
        setPermissionError('Camera access timed out. Please try again.');
        setHasPermission(false);
        setIsLoading(false);
      }, 5000); // 5 second timeout

      return () => clearTimeout(timeoutId);
    }
  }, [isLoading, isOpen]);

  const dataUrlToBlob = (dataUrl: string): Blob | null => {
    try {
      if (!dataUrl || !dataUrl.includes(',')) {
        throw new Error('Invalid data URL format');
      }

      const [meta, content] = dataUrl.split(',');
      const mimeMatch = /data:(.*?);base64/.exec(meta);
      const mime = mimeMatch ? mimeMatch[1] : 'image/jpeg';
      const binary = atob(content);
      const len = binary.length;
      const buffer = new Uint8Array(len);
      for (let i = 0; i < len; i += 1) {
        buffer[i] = binary.charCodeAt(i);
      }
      return new Blob([buffer], { type: mime });
    } catch (error) {
      console.error('❌ Failed to convert data URL to blob:', error);
      return null;
    }
  };

  const handleCapture = () => {
    const shot = webcamRef.current?.getScreenshot();
    if (shot) {
      setCapturedDataUrl(shot);
    }
  };

  const handleConfirm = () => {
    if (!capturedDataUrl) return;
    const blob = dataUrlToBlob(capturedDataUrl);
    if (!blob) {
      console.error('Failed to process captured image');
      setPermissionError('Failed to process image. Please try again.');
      return;
    }
    onCapture(blob);
    onClose();
    setCapturedDataUrl(null);
  };

  const handleRetake = () => {
    setCapturedDataUrl(null);
  };

  const renderErrorState = () => {
    const isHttpsRequired = permissionError?.includes('getUserMedia is not implemented');
    const isHttpConnection = typeof window !== 'undefined' && window.location.protocol === 'http:';
    
    return (
      <div className="absolute inset-0 flex flex-col items-center justify-center gap-4 bg-black/70 px-6 text-center text-white">
        <div className="rounded-full bg-red-500/20 p-4 mb-2">
          <svg className="h-12 w-12 text-red-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
          </svg>
        </div>
        
        <p className="text-lg font-semibold">
          {isHttpsRequired && isHttpConnection ? 'HTTPS Required for Camera' : 'Camera Access Required'}
        </p>
        
        {isHttpsRequired && isHttpConnection ? (
          <div className="space-y-3 max-w-md">
            <p className="text-sm text-neutral-300">
              Camera access requires a secure connection (HTTPS) on mobile devices.
            </p>
            <div className="bg-black/40 rounded-lg p-4 text-left space-y-2">
              <p className="text-xs font-semibold text-secondary">Quick Fix Options:</p>
              <ol className="text-xs text-neutral-300 space-y-1 list-decimal list-inside">
                <li>Use localhost: <code className="bg-white/10 px-1 rounded">npm run dev</code> on device</li>
                <li>Enable HTTPS in Next.js (see console for command)</li>
                <li>Use ngrok: <code className="bg-white/10 px-1 rounded">npx ngrok http 3000</code></li>
              </ol>
            </div>
            <p className="text-xs text-neutral-400">
              For now, you can use "Choose from Gallery" instead.
            </p>
          </div>
        ) : (
          <p className="text-sm text-neutral-300">{permissionError ?? 'Please grant permission to use the camera.'}</p>
        )}
        
        <div className="flex gap-3">
          {!isHttpsRequired && (
            <button
              type="button"
              onClick={() => {
                setHasPermission(null);
                setPermissionError(null);
                setIsLoading(true);
              }}
              className="rounded-full bg-secondary px-6 py-3 text-sm font-semibold text-white shadow-lg transition hover:bg-secondary/90 focus:ring-2 focus:ring-secondary focus:outline-none"
              aria-label="Retry camera access"
              autoFocus
            >
              Try Again
            </button>
          )}
          <button
            type="button"
            onClick={onClose}
            className="rounded-full bg-white/10 px-6 py-3 text-sm font-semibold text-white hover:bg-white/20 focus:ring-2 focus:ring-secondary focus:outline-none"
            aria-label={isHttpsRequired && isHttpConnection ? 'Use gallery instead' : 'Close error dialog'}
          >
            {isHttpsRequired && isHttpConnection ? 'Use Gallery Instead' : 'Close'}
          </button>
        </div>
      </div>
    );
  };

  if (!isOpen) return null;

  return (
    <AnimatePresence>
      {isOpen && (
        <motion.div
          key="camera-capture"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="fixed inset-0 z-[100] bg-black"
        >
          <div className="relative h-full w-full flex flex-col">
            {/* Camera view container - expanded to fill more space */}
            <div className="absolute inset-0">
              {capturedDataUrl ? (
                <img src={capturedDataUrl} alt="Captured" className="h-full w-full object-cover" />
              ) : hasPermission === false ? (
                renderErrorState()
              ) : (
                <Webcam
                  ref={webcamRef}
                  audio={false}
                  screenshotFormat="image/jpeg"
                  videoConstraints={videoConstraints}
                  className="h-full w-full object-cover"
                  onUserMedia={() => {
                    setHasPermission(true);
                    setIsLoading(false);
                  }}
                  onUserMediaError={(err) => {
                    setHasPermission(false);
                    setIsLoading(false);
                    const errorMessage = typeof err === 'string' ? err : err?.message ?? 'Unable to access camera';
                    setPermissionError(errorMessage);
                  }}
                />
              )}

              {/* Gradient overlays for better contrast on controls */}
              <div className="pointer-events-none absolute inset-0 bg-gradient-to-t from-black/70 via-transparent to-black/70" />
            </div>

            {/* Top controls */}
            <div className="absolute left-0 right-0 top-0 z-40 px-4 pt-2 safe-area-inset-top">
              <div className="flex items-center justify-start">
                <button
                  type="button"
                  onClick={onClose}
                  className="rounded-full bg-white/10 p-3 text-white backdrop-blur focus:outline-none focus:ring-2 focus:ring-secondary"
                  aria-label="Close camera"
                >
                  <X className="h-5 w-5" />
                </button>
              </div>
            </div>

            {/* Loading state overlay - only shows message, doesn't block controls */}
            {isLoading && (
              <div className="absolute inset-x-0 top-1/2 -translate-y-1/2 flex items-center justify-center text-white pointer-events-none z-30">
                <div className="flex items-center gap-3 rounded-full bg-black/60 backdrop-blur px-4 py-2 text-sm font-medium">
                  <Loader2 className="h-5 w-5 animate-spin" />
                  Accessing camera...
                </div>
              </div>
            )}

            {/* Bottom controls - highest z-index */}
            <div className="absolute inset-x-0 bottom-0 z-50 pb-8 safe-area-inset-bottom">
              <div className="flex flex-col items-center gap-4 px-6">
                {!capturedDataUrl ? (
                  <motion.button
                    type="button"
                    onClick={handleCapture}
                    whileTap={{ scale: 0.94 }}
                    className="relative h-20 w-20 rounded-full border-[6px] border-white bg-gradient-to-br from-white/30 to-white/10 backdrop-blur-lg shadow-2xl disabled:opacity-30 disabled:cursor-not-allowed transition-all"
                    aria-label="Capture photo"
                    disabled={isLoading || hasPermission !== true}
                  >
                    <span className="absolute inset-2 rounded-full bg-secondary shadow-lg" />
                    <CameraIcon className="absolute left-1/2 top-1/2 h-7 w-7 -translate-x-1/2 -translate-y-1/2 text-white z-10" />
                  </motion.button>
                ) : (
                  <div className="flex w-full items-center justify-center gap-4">
                  <button
                    type="button"
                    onClick={handleRetake}
                    className="flex flex-1 items-center justify-center gap-2 rounded-full bg-white/10 px-4 py-3 text-sm font-semibold text-white backdrop-blur transition hover:bg-white/20 focus:outline-none focus:ring-2 focus:ring-secondary"
                    aria-label="Retake photo"
                  >
                    <RefreshCw className="h-4 w-4" />
                    Retake
                  </button>
                  <button
                    type="button"
                    onClick={handleConfirm}
                    className="flex flex-1 items-center justify-center gap-2 rounded-full bg-secondary px-4 py-3 text-sm font-semibold text-white shadow-lg transition hover:bg-secondary/90 focus:outline-none focus:ring-2 focus:ring-secondary"
                    aria-label="Use this photo"
                  >
                    <Check className="h-4 w-4" />
                    Use Photo
                  </button>
                </div>
              )}
              </div>
            </div>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
