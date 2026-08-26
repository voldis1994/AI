const STORAGE_KEYS = {
  profile: "personal-ai-profile",
  memories: "personal-ai-memories",
  sessions: "personal-ai-sessions",
  currentSession: "personal-ai-current-session",
} as const;

export function loadFromStorage<T>(key: string, fallback: T): T {
  if (typeof window === "undefined") return fallback;
  try {
    const raw = localStorage.getItem(key);
    if (!raw) return fallback;
    return JSON.parse(raw) as T;
  } catch {
    return fallback;
  }
}

export function saveToStorage<T>(key: string, value: T): void {
  if (typeof window === "undefined") return;
  localStorage.setItem(key, JSON.stringify(value));
}

export { STORAGE_KEYS };
