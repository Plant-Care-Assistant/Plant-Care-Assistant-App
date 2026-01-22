"use client";

import { Layout } from "@/components/layout";
import { Button, Input } from "@/components/ui";
import { Upload } from "lucide-react";
import { useState } from "react";
import { useIdentifyPlantMutation } from "@/hooks/usePlants";

export default function ScanPage() {
  const [file, setFile] = useState<File | null>(null);
  const identifyMutation = useIdentifyPlantMutation();

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selected = e.target.files?.[0];
    if (selected) setFile(selected);
  };

  const handleIdentify = () => {
    if (file) identifyMutation.mutate(file);
  };

  return (
    <Layout
      header={{ title: "Scan", subtitle: "Upload or snap a plant photo" }}
      showBottomNav
    >
      <div className="space-y-4">
        <p className="text-neutral-600 dark:text-neutral-300">
          Capture or upload a plant photo to identify it and get care tips.
        </p>
        <div className="rounded-2xl border border-dashed border-neutral-300 bg-white p-6 text-center shadow-sm dark:border-neutral-700 dark:bg-neutral-900">
          <div className="mx-auto flex h-24 w-24 items-center justify-center rounded-full bg-primary/10 text-primary">
            <Upload className="h-10 w-10" aria-hidden="true" />
          </div>
          <p className="mt-4 text-lg font-medium">Drop a photo or use your camera</p>
          <p className="text-sm text-neutral-600 dark:text-neutral-400">JPG, PNG up to 10MB</p>
          <div className="mt-5 flex flex-col gap-3 sm:flex-row sm:justify-center">
            <Button variant="primary">Use Camera</Button>
            <div className="flex flex-col items-center gap-2 sm:flex-row">
              <Input type="file" accept="image/*" onChange={handleFileChange} aria-label="Upload image" />
              <Button variant="outline" onClick={handleIdentify} disabled={!file || identifyMutation.isPending}>
                {identifyMutation.isPending ? "Identifying..." : "Identify"}
              </Button>
            </div>
          </div>
        </div>
        <div className="space-y-2">
          <label className="text-sm font-medium text-neutral-700 dark:text-neutral-200">Paste image URL</label>
          <div className="flex gap-2">
            <Input placeholder="https://..." aria-label="Image URL" />
            <Button variant="secondary">Fetch</Button>
          </div>
        </div>

        {identifyMutation.isError && (
          <p className="text-sm text-red-600 dark:text-red-400">Identification failed. Try another image.</p>
        )}

        {identifyMutation.data && (
          <div className="rounded-xl border border-neutral-200 bg-white p-4 shadow-sm dark:border-neutral-800 dark:bg-neutral-900">
            <h2 className="text-lg font-semibold">Result</h2>
            <p className="text-sm text-neutral-700 dark:text-neutral-300">
              {identifyMutation.data.name} ({identifyMutation.data.scientificName})
            </p>
            <p className="text-sm text-neutral-700 dark:text-neutral-300">
              Confidence: {(identifyMutation.data.confidence * 100).toFixed(1)}%
            </p>
          </div>
        )}
      </div>
    </Layout>
  );
}
