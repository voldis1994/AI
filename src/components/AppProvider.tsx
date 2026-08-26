"use client";

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useState,
  type ReactNode,
} from "react";
import {
  DEFAULT_PROFILE,
  generateId,
  type ChatMessage,
  type ChatSession,
  type Memory,
  type UserProfile,
} from "@/lib/types";
import { loadFromStorage, saveToStorage, STORAGE_KEYS } from "@/lib/storage";

interface AppContextValue {
  profile: UserProfile;
  setProfile: (profile: UserProfile) => void;
  memories: Memory[];
  addMemory: (content: string) => void;
  removeMemory: (id: string) => void;
  sessions: ChatSession[];
  currentSession: ChatSession | null;
  createSession: () => ChatSession;
  selectSession: (id: string) => void;
  deleteSession: (id: string) => void;
  addMessage: (role: "user" | "assistant", content: string) => void;
  updateLastAssistantMessage: (content: string) => void;
  clearCurrentSession: () => void;
  isSettingsOpen: boolean;
  setIsSettingsOpen: (open: boolean) => void;
  isSidebarOpen: boolean;
  setIsSidebarOpen: (open: boolean) => void;
}

const AppContext = createContext<AppContextValue | null>(null);

function createEmptySession(): ChatSession {
  const now = new Date().toISOString();
  return {
    id: generateId(),
    title: "Jauna saruna",
    messages: [],
    createdAt: now,
    updatedAt: now,
  };
}

export function AppProvider({ children }: { children: ReactNode }) {
  const [profile, setProfileState] = useState<UserProfile>(DEFAULT_PROFILE);
  const [memories, setMemories] = useState<Memory[]>([]);
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(null);
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [hydrated, setHydrated] = useState(false);

  useEffect(() => {
    const savedProfile = loadFromStorage(STORAGE_KEYS.profile, DEFAULT_PROFILE);
    const savedMemories = loadFromStorage<Memory[]>(STORAGE_KEYS.memories, []);
    const savedSessions = loadFromStorage<ChatSession[]>(STORAGE_KEYS.sessions, []);
    const savedCurrentId = loadFromStorage<string | null>(
      STORAGE_KEYS.currentSession,
      null
    );

    setProfileState(savedProfile);
    setMemories(savedMemories);
    setSessions(savedSessions.length > 0 ? savedSessions : [createEmptySession()]);
    setCurrentSessionId(
      savedCurrentId || (savedSessions.length > 0 ? savedSessions[0].id : null)
    );
    setHydrated(true);
  }, []);

  useEffect(() => {
    if (!hydrated) return;
    saveToStorage(STORAGE_KEYS.profile, profile);
  }, [profile, hydrated]);

  useEffect(() => {
    if (!hydrated) return;
    saveToStorage(STORAGE_KEYS.memories, memories);
  }, [memories, hydrated]);

  useEffect(() => {
    if (!hydrated) return;
    saveToStorage(STORAGE_KEYS.sessions, sessions);
  }, [sessions, hydrated]);

  useEffect(() => {
    if (!hydrated || !currentSessionId) return;
    saveToStorage(STORAGE_KEYS.currentSession, currentSessionId);
  }, [currentSessionId, hydrated]);

  const currentSession =
    sessions.find((s) => s.id === currentSessionId) ?? sessions[0] ?? null;

  const setProfile = useCallback((p: UserProfile) => setProfileState(p), []);

  const addMemory = useCallback((content: string) => {
    const memory: Memory = {
      id: generateId(),
      content: content.trim(),
      createdAt: new Date().toISOString(),
    };
    setMemories((prev) => [...prev, memory]);
  }, []);

  const removeMemory = useCallback((id: string) => {
    setMemories((prev) => prev.filter((m) => m.id !== id));
  }, []);

  const createSession = useCallback(() => {
    const session = createEmptySession();
    setSessions((prev) => [session, ...prev]);
    setCurrentSessionId(session.id);
    return session;
  }, []);

  const selectSession = useCallback((id: string) => {
    setCurrentSessionId(id);
  }, []);

  const deleteSession = useCallback(
    (id: string) => {
      setSessions((prev) => {
        const filtered = prev.filter((s) => s.id !== id);
        if (filtered.length === 0) {
          const newSession = createEmptySession();
          setCurrentSessionId(newSession.id);
          return [newSession];
        }
        if (currentSessionId === id) {
          setCurrentSessionId(filtered[0].id);
        }
        return filtered;
      });
    },
    [currentSessionId]
  );

  const addMessage = useCallback(
    (role: "user" | "assistant", content: string) => {
      if (!currentSessionId) return;

      const message: ChatMessage = {
        id: generateId(),
        role,
        content,
        createdAt: new Date().toISOString(),
      };

      setSessions((prev) =>
        prev.map((s) => {
          if (s.id !== currentSessionId) return s;
          const updated: ChatSession = {
            ...s,
            messages: [...s.messages, message],
            updatedAt: new Date().toISOString(),
            title:
              s.messages.length === 0 && role === "user"
                ? content.slice(0, 40) + (content.length > 40 ? "…" : "")
                : s.title,
          };
          return updated;
        })
      );
    },
    [currentSessionId]
  );

  const updateLastAssistantMessage = useCallback(
    (content: string) => {
      if (!currentSessionId) return;

      setSessions((prev) =>
        prev.map((s) => {
          if (s.id !== currentSessionId) return s;
          const messages = [...s.messages];
          const lastIdx = messages.length - 1;
          if (lastIdx >= 0 && messages[lastIdx].role === "assistant") {
            messages[lastIdx] = { ...messages[lastIdx], content };
          }
          return { ...s, messages, updatedAt: new Date().toISOString() };
        })
      );
    },
    [currentSessionId]
  );

  const clearCurrentSession = useCallback(() => {
    if (!currentSessionId) return;
    setSessions((prev) =>
      prev.map((s) =>
        s.id === currentSessionId
          ? { ...s, messages: [], title: "Jauna saruna", updatedAt: new Date().toISOString() }
          : s
      )
    );
  }, [currentSessionId]);

  if (!hydrated) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="w-8 h-8 border-2 border-accent border-t-transparent rounded-full animate-spin" />
      </div>
    );
  }

  return (
    <AppContext.Provider
      value={{
        profile,
        setProfile,
        memories,
        addMemory,
        removeMemory,
        sessions,
        currentSession,
        createSession,
        selectSession,
        deleteSession,
        addMessage,
        updateLastAssistantMessage,
        clearCurrentSession,
        isSettingsOpen,
        setIsSettingsOpen,
        isSidebarOpen,
        setIsSidebarOpen,
      }}
    >
      {children}
    </AppContext.Provider>
  );
}

export function useApp() {
  const ctx = useContext(AppContext);
  if (!ctx) throw new Error("useApp must be used within AppProvider");
  return ctx;
}
