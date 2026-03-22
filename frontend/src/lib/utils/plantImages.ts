const STORAGE_KEY = "plant_images";

function getImageMap(): Record<string, string> {
  try {
    return JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}");
  } catch {
    return {};
  }
}

export function savePlantImage(plantId: number, dataUrl: string) {
  const map = getImageMap();
  map[String(plantId)] = dataUrl;
  localStorage.setItem(STORAGE_KEY, JSON.stringify(map));
}

export function getPlantImage(plantId: number): string | undefined {
  return getImageMap()[String(plantId)];
}

export function removePlantImage(plantId: number) {
  const map = getImageMap();
  delete map[String(plantId)];
  localStorage.setItem(STORAGE_KEY, JSON.stringify(map));
}
