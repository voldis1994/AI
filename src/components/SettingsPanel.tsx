"use client";

import { useEffect, useState } from "react";
import { Brain, Plus, X } from "lucide-react";
import { useApp } from "@/components/AppProvider";
import {
  PERSONALITY_LABELS,
  type Personality,
  type UserProfile,
} from "@/lib/types";

export function SettingsPanel() {
  const {
    profile,
    setProfile,
    memories,
    addMemory,
    removeMemory,
    setIsSettingsOpen,
  } = useApp();

  const [draft, setDraft] = useState<UserProfile>(profile);
  const [newMemory, setNewMemory] = useState("");
  const [newInterest, setNewInterest] = useState("");

  useEffect(() => {
    setDraft(profile);
  }, [profile]);

  const t = draft.language === "lv";

  const save = () => {
    setProfile(draft);
    setIsSettingsOpen(false);
  };

  const addInterest = () => {
    const trimmed = newInterest.trim();
    if (!trimmed || draft.interests.includes(trimmed)) return;
    setDraft({ ...draft, interests: [...draft.interests, trimmed] });
    setNewInterest("");
  };

  const removeInterest = (interest: string) => {
    setDraft({
      ...draft,
      interests: draft.interests.filter((i) => i !== interest),
    });
  };

  return (
    <div className="fixed inset-0 z-50 flex">
      <div
        className="absolute inset-0 bg-black/60 backdrop-blur-sm"
        onClick={() => setIsSettingsOpen(false)}
      />
      <aside className="relative ml-auto w-full max-w-md glass border-l flex flex-col animate-slide-up h-full overflow-hidden">
        <div className="p-5 border-b border-white/10 flex items-center justify-between">
          <h2 className="text-lg font-semibold">
            {t ? "Iestatījumi" : "Settings"}
          </h2>
          <button
            onClick={() => setIsSettingsOpen(false)}
            className="p-2 rounded-lg hover:bg-white/5 text-white/50"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        <div className="flex-1 overflow-y-auto p-5 space-y-6">
          {/* Language */}
          <section>
            <label className="block text-xs font-medium text-white/50 uppercase tracking-wider mb-2">
              {t ? "Valoda" : "Language"}
            </label>
            <div className="flex gap-2">
              {(["lv", "en"] as const).map((lang) => (
                <button
                  key={lang}
                  onClick={() => setDraft({ ...draft, language: lang })}
                  className={`flex-1 py-2 rounded-lg text-sm font-medium transition-colors ${
                    draft.language === lang
                      ? "bg-accent text-white"
                      : "bg-white/5 text-white/60 hover:bg-white/10"
                  }`}
                >
                  {lang === "lv" ? "Latviešu" : "English"}
                </button>
              ))}
            </div>
          </section>

          {/* Profile */}
          <section className="space-y-3">
            <label className="block text-xs font-medium text-white/50 uppercase tracking-wider">
              {t ? "Profils" : "Profile"}
            </label>
            <input
              type="text"
              value={draft.name}
              onChange={(e) => setDraft({ ...draft, name: e.target.value })}
              placeholder={t ? "Tavs vārds" : "Your name"}
              className="w-full px-3 py-2.5 rounded-lg bg-white/5 border border-white/10 text-sm focus:outline-none focus:border-accent/50"
            />
            <input
              type="text"
              value={draft.assistantName}
              onChange={(e) =>
                setDraft({ ...draft, assistantName: e.target.value })
              }
              placeholder={t ? "AI vārds" : "AI name"}
              className="w-full px-3 py-2.5 rounded-lg bg-white/5 border border-white/10 text-sm focus:outline-none focus:border-accent/50"
            />
            <textarea
              value={draft.about}
              onChange={(e) => setDraft({ ...draft, about: e.target.value })}
              placeholder={
                t
                  ? "Par sevi (AI izmantos šo informāciju)"
                  : "About you (AI will use this info)"
              }
              rows={3}
              className="w-full px-3 py-2.5 rounded-lg bg-white/5 border border-white/10 text-sm focus:outline-none focus:border-accent/50 resize-none"
            />
          </section>

          {/* Personality */}
          <section>
            <label className="block text-xs font-medium text-white/50 uppercase tracking-wider mb-2">
              {t ? "Personība" : "Personality"}
            </label>
            <div className="grid grid-cols-2 gap-2">
              {(Object.keys(PERSONALITY_LABELS) as Personality[]).map((key) => {
                const p = PERSONALITY_LABELS[key];
                return (
                  <button
                    key={key}
                    onClick={() => setDraft({ ...draft, personality: key })}
                    className={`p-3 rounded-lg text-left transition-colors ${
                      draft.personality === key
                        ? "bg-accent/30 border border-accent/50"
                        : "bg-white/5 border border-transparent hover:bg-white/10"
                    }`}
                  >
                    <span className="text-sm font-medium block">
                      {t ? p.lv : p.en}
                    </span>
                    <span className="text-xs text-white/40 mt-0.5 block">
                      {t ? p.description.lv : p.description.en}
                    </span>
                  </button>
                );
              })}
            </div>
          </section>

          {/* Interests */}
          <section>
            <label className="block text-xs font-medium text-white/50 uppercase tracking-wider mb-2">
              {t ? "Intereses" : "Interests"}
            </label>
            <div className="flex gap-2 mb-2">
              <input
                type="text"
                value={newInterest}
                onChange={(e) => setNewInterest(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && addInterest()}
                placeholder={t ? "Pievienot interesi" : "Add interest"}
                className="flex-1 px-3 py-2 rounded-lg bg-white/5 border border-white/10 text-sm focus:outline-none focus:border-accent/50"
              />
              <button
                onClick={addInterest}
                className="px-3 py-2 rounded-lg bg-white/5 hover:bg-white/10 transition-colors"
              >
                <Plus className="w-4 h-4" />
              </button>
            </div>
            <div className="flex flex-wrap gap-2">
              {draft.interests.map((interest) => (
                <span
                  key={interest}
                  className="inline-flex items-center gap-1 px-2.5 py-1 rounded-full bg-accent/20 text-accent-hover text-xs"
                >
                  {interest}
                  <button onClick={() => removeInterest(interest)}>
                    <X className="w-3 h-3" />
                  </button>
                </span>
              ))}
            </div>
          </section>

          {/* Memories */}
          <section>
            <label className="block text-xs font-medium text-white/50 uppercase tracking-wider mb-2 flex items-center gap-1.5">
              <Brain className="w-3.5 h-3.5" />
              {t ? "Atmiņa" : "Memory"}
            </label>
            <p className="text-xs text-white/40 mb-3">
              {t
                ? "Fakti, ko AI atcerēsies par tevi visās sarunās."
                : "Facts the AI will remember about you across all chats."}
            </p>
            <div className="flex gap-2 mb-3">
              <input
                type="text"
                value={newMemory}
                onChange={(e) => setNewMemory(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && newMemory.trim()) {
                    addMemory(newMemory);
                    setNewMemory("");
                  }
                }}
                placeholder={
                  t ? "piem., Man patīk programmēšana" : "e.g., I love coding"
                }
                className="flex-1 px-3 py-2 rounded-lg bg-white/5 border border-white/10 text-sm focus:outline-none focus:border-accent/50"
              />
              <button
                onClick={() => {
                  if (newMemory.trim()) {
                    addMemory(newMemory);
                    setNewMemory("");
                  }
                }}
                className="px-3 py-2 rounded-lg bg-white/5 hover:bg-white/10 transition-colors"
              >
                <Plus className="w-4 h-4" />
              </button>
            </div>
            <div className="space-y-2">
              {memories.map((m) => (
                <div
                  key={m.id}
                  className="flex items-start gap-2 p-2.5 rounded-lg bg-white/5 text-sm"
                >
                  <span className="flex-1 text-white/80">{m.content}</span>
                  <button
                    onClick={() => removeMemory(m.id)}
                    className="text-white/30 hover:text-white/60 shrink-0"
                  >
                    <X className="w-3.5 h-3.5" />
                  </button>
                </div>
              ))}
              {memories.length === 0 && (
                <p className="text-xs text-white/30 text-center py-4">
                  {t ? "Vēl nav atmiņu" : "No memories yet"}
                </p>
              )}
            </div>
          </section>

          {/* API info */}
          <section className="p-4 rounded-xl bg-white/5 border border-white/10">
            <p className="text-xs text-white/50 leading-relaxed">
              {t ? (
                <>
                  Lai sāktu lietot, pievieno savu OpenAI API atslēgu{" "}
                  <code className="text-accent-hover">.env.local</code> failā:
                  <br />
                  <code className="text-white/70 mt-1 block">
                    OPENAI_API_KEY=sk-...
                  </code>
                </>
              ) : (
                <>
                  To get started, add your OpenAI API key to{" "}
                  <code className="text-accent-hover">.env.local</code>:
                  <br />
                  <code className="text-white/70 mt-1 block">
                    OPENAI_API_KEY=sk-...
                  </code>
                </>
              )}
            </p>
          </section>
        </div>

        <div className="p-5 border-t border-white/10 flex gap-3">
          <button
            onClick={() => setIsSettingsOpen(false)}
            className="flex-1 py-2.5 rounded-lg bg-white/5 hover:bg-white/10 text-sm transition-colors"
          >
            {t ? "Atcelt" : "Cancel"}
          </button>
          <button
            onClick={save}
            className="flex-1 py-2.5 rounded-lg bg-accent hover:bg-accent-hover text-sm font-medium transition-colors"
          >
            {t ? "Saglabāt" : "Save"}
          </button>
        </div>
      </aside>
    </div>
  );
}
