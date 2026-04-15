type UploadContext = {
  source: string;
  content: string;
  uploadedAt: number;
};

type UploadImageContext = {
  source: string;
  mimeType: string;
  dataUrl: string;
  uploadedAt: number;
};

let latestUploadContext: UploadContext | null = null;
let latestUploadImageContext: UploadImageContext | null = null;

export function setLatestUploadContext(source: string, content: string) {
  latestUploadContext = {
    source,
    content,
    uploadedAt: Date.now(),
  };
}

export function getLatestUploadContext(maxAgeMs = 30 * 60 * 1000): UploadContext | null {
  if (!latestUploadContext) return null;
  if (Date.now() - latestUploadContext.uploadedAt > maxAgeMs) return null;
  return latestUploadContext;
}

export function setLatestUploadImageContext(source: string, mimeType: string, dataUrl: string) {
  latestUploadImageContext = {
    source,
    mimeType,
    dataUrl,
    uploadedAt: Date.now(),
  };
}

export function getLatestUploadImageContext(maxAgeMs = 30 * 60 * 1000): UploadImageContext | null {
  if (!latestUploadImageContext) return null;
  if (Date.now() - latestUploadImageContext.uploadedAt > maxAgeMs) return null;
  return latestUploadImageContext;
}

export default {
  setLatestUploadContext,
  getLatestUploadContext,
  setLatestUploadImageContext,
  getLatestUploadImageContext,
};
