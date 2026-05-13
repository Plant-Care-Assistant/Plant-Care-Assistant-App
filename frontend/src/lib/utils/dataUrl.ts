/**
 * Convert a `data:` URL (e.g. from a `<canvas>` snapshot or `FileReader`) into
 * a File suitable for `multipart/form-data` upload. Throws on malformed input.
 */
export function dataUrlToFile(dataUrl: string, filename: string): File {
  const [header, data] = dataUrl.split(",");
  const mimeType = header.match(/:(.*?);/)?.[1] || "image/jpeg";
  let bytes: Uint8Array;
  try {
    bytes = new Uint8Array(Array.from(atob(data), (c) => c.charCodeAt(0)));
  } catch {
    throw new Error("Invalid image data: base64 decoding failed");
  }
  return new File([bytes.buffer as ArrayBuffer], filename, { type: mimeType });
}
