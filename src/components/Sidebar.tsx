"use client";

import {
  Brain,
  Menu,
  MessageSquarePlus,
  Settings,
  Sparkles,
  Trash2,
} from "lucide-react";
import { useApp } from "@/components/AppProvider";

export function Sidebar() {
  const {
    profile,
    sessions,
    currentSession,
    createSession,
    selectSession,
    deleteSession,
    setIsSettingsOpen,
    setIsSidebarOpen,
  } = useApp();

  const t = profile.language === "lv";

  return (
    <aside className="w-72 glass border-r flex flex-col shrink-0 animate-fade-in">
      <div className="p-4 border-b border-white/10">
        <div className="flex items-center gap-3 mb-4">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center">
            <Sparkles className="w-5 h-5 text-white" />
          </div>
          <div>
            <h1 className="font-semibold text-sm gradient-text">{profile.assistantName}</h1>
            <p className="text-xs text-white/50">
              {t ? "Personīgais AI" : "Personal AI"}
            </p>
          </div>
        </div>

        <button
          onClick={() => createSession()}
          className="w-full flex items-center gap-2 px-3 py-2.5 rounded-lg bg-accent/20 hover:bg-accent/30 text-accent-hover text-sm font-medium transition-colors"
        >
          <MessageSquarePlus className="w-4 h-4" />
          {t ? "Jauna saruna" : "New chat"}
        </button>
      </div>

      <div className="flex-1 overflow-y-auto p-2 space-y-1">
        {sessions.map((session) => (
          <div
            key={session.id}
            className={`group flex items-center gap-2 px-3 py-2 rounded-lg cursor-pointer transition-colors ${
              currentSession?.id === session.id
                ? "bg-white/10"
                : "hover:bg-white/5"
            }`}
            onClick={() => selectSession(session.id)}
          >
            <Brain className="w-4 h-4 shrink-0 text-white/40" />
            <span className="flex-1 text-sm truncate text-white/80">
              {session.title}
            </span>
            <button
              onClick={(e) => {
                e.stopPropagation();
                deleteSession(session.id);
              }}
              className="opacity-0 group-hover:opacity-100 p-1 hover:bg-white/10 rounded transition-all"
              aria-label={t ? "Dzēst sarunu" : "Delete chat"}
            >
              <Trash2 className="w-3.5 h-3.5 text-white/40" />
            </button>
          </div>
        ))}
      </div>

      <div className="p-3 border-t border-white/10 flex gap-2">
        <button
          onClick={() => setIsSettingsOpen(true)}
          className="flex-1 flex items-center justify-center gap-2 px-3 py-2 rounded-lg hover:bg-white/5 text-sm text-white/70 transition-colors"
        >
          <Settings className="w-4 h-4" />
          {t ? "Iestatījumi" : "Settings"}
        </button>
        <button
          onClick={() => setIsSidebarOpen(false)}
          className="p-2 rounded-lg hover:bg-white/5 text-white/50 lg:hidden"
          aria-label={t ? "Aizvērt sānjoslu" : "Close sidebar"}
        >
          <Menu className="w-4 h-4" />
        </button>
      </div>
    </aside>
  );
}
