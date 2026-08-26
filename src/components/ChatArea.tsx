"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  Menu,
  RotateCcw,
  Send,
  Settings,
  Sparkles,
} from "lucide-react";
import { useApp } from "@/components/AppProvider";
import { MessageBubble, TypingIndicator } from "@/components/MessageBubble";

const SUGGESTIONS_LV = [
  "Palīdzi man plānot dienu",
  "Izskaidro kā darbojas mākslīgais intelekts",
  "Uzraksti radošu stāstu par Latviju",
  "Ko tu atceries par mani?",
];

const SUGGESTIONS_EN = [
  "Help me plan my day",
  "Explain how AI works",
  "Write a creative story",
  "What do you remember about me?",
];

export function ChatArea() {
  const {
    profile,
    memories,
    currentSession,
    addMessage,
    updateLastAssistantMessage,
    clearCurrentSession,
    setIsSettingsOpen,
    setIsSidebarOpen,
    isSidebarOpen,
  } = useApp();

  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const t = profile.language === "lv";
  const suggestions = t ? SUGGESTIONS_LV : SUGGESTIONS_EN;
  const messages = currentSession?.messages ?? [];

  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages, isLoading, scrollToBottom]);

  const sendMessage = useCallback(
    async (text: string) => {
      const trimmed = text.trim();
      if (!trimmed || isLoading) return;

      setError(null);
      setInput("");
      addMessage("user", trimmed);
      addMessage("assistant", "");
      setIsLoading(true);

      try {
        const history = [
          ...(currentSession?.messages ?? []).map((m) => ({
            role: m.role,
            content: m.content,
          })),
          { role: "user" as const, content: trimmed },
        ];

        const response = await fetch("/api/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            messages: history,
            profile,
            memories,
          }),
        });

        if (!response.ok) {
          const data = await response.json().catch(() => ({}));
          throw new Error(
            data.error ||
              (t ? "Neizdevās saņemt atbildi" : "Failed to get response")
          );
        }

        const reader = response.body?.getReader();
        if (!reader) throw new Error("No response stream");

        const decoder = new TextDecoder();
        let fullText = "";

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          fullText += decoder.decode(value, { stream: true });
          updateLastAssistantMessage(fullText);
        }
      } catch (err) {
        const msg = err instanceof Error ? err.message : "Unknown error";
        setError(msg);
        updateLastAssistantMessage(
          t
            ? `⚠️ Kļūda: ${msg}\n\nPārliecinies, ka .env.local failā ir iestatīta OPENAI_API_KEY.`
            : `⚠️ Error: ${msg}\n\nMake sure OPENAI_API_KEY is set in .env.local.`
        );
      } finally {
        setIsLoading(false);
        textareaRef.current?.focus();
      }
    },
    [
      isLoading,
      addMessage,
      currentSession,
      profile,
      memories,
      updateLastAssistantMessage,
      t,
    ]
  );

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    sendMessage(input);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage(input);
    }
  };

  const isEmpty = messages.length === 0;

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <header className="glass border-b px-4 py-3 flex items-center gap-3 shrink-0">
        {!isSidebarOpen && (
          <button
            onClick={() => setIsSidebarOpen(true)}
            className="p-2 rounded-lg hover:bg-white/5 text-white/60"
            aria-label={t ? "Atvērt sānjoslu" : "Open sidebar"}
          >
            <Menu className="w-5 h-5" />
          </button>
        )}
        <div className="flex-1">
          <h2 className="font-medium text-sm">{profile.assistantName}</h2>
          <p className="text-xs text-white/40">
            {t ? "Tavs personīgais asistents" : "Your personal assistant"}
            {profile.name ? ` · ${profile.name}` : ""}
          </p>
        </div>
        {messages.length > 0 && (
          <button
            onClick={clearCurrentSession}
            className="p-2 rounded-lg hover:bg-white/5 text-white/50 hover:text-white/80 transition-colors"
            title={t ? "Notīrīt sarunu" : "Clear chat"}
          >
            <RotateCcw className="w-4 h-4" />
          </button>
        )}
        <button
          onClick={() => setIsSettingsOpen(true)}
          className="p-2 rounded-lg hover:bg-white/5 text-white/50 hover:text-white/80 transition-colors"
          title={t ? "Iestatījumi" : "Settings"}
        >
          <Settings className="w-4 h-4" />
        </button>
      </header>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-4 py-6">
        {isEmpty ? (
          <div className="h-full flex flex-col items-center justify-center text-center max-w-lg mx-auto animate-fade-in">
            <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center mb-6 shadow-lg shadow-indigo-500/20">
              <Sparkles className="w-8 h-8 text-white" />
            </div>
            <h2 className="text-2xl font-semibold mb-2">
              {t ? `Sveiki${profile.name ? `, ${profile.name}` : ""}!` : `Hello${profile.name ? `, ${profile.name}` : ""}!`}
            </h2>
            <p className="text-white/50 mb-8 text-sm leading-relaxed">
              {t
                ? `Es esmu ${profile.assistantName} — tavs personīgais AI. Jautā jebko, un es palīdzēšu!`
                : `I'm ${profile.assistantName} — your personal AI. Ask me anything!`}
            </p>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 w-full">
              {suggestions.map((s) => (
                <button
                  key={s}
                  onClick={() => sendMessage(s)}
                  className="text-left px-4 py-3 rounded-xl glass hover:bg-white/10 text-sm text-white/70 transition-colors"
                >
                  {s}
                </button>
              ))}
            </div>
          </div>
        ) : (
          <div className="max-w-3xl mx-auto space-y-6">
            {messages.map((msg) =>
              msg.role === "assistant" &&
              msg.content === "" &&
              isLoading ? null : (
                <MessageBubble key={msg.id} message={msg} />
              )
            )}
            {isLoading && <TypingIndicator />}
            <div ref={messagesEndRef} />
          </div>
        )}
      </div>

      {/* Input */}
      <div className="shrink-0 p-4 border-t border-white/10">
        {error && (
          <p className="text-red-400 text-xs mb-2 text-center">{error}</p>
        )}
        <form
          onSubmit={handleSubmit}
          className="max-w-3xl mx-auto flex gap-3 items-end glass rounded-2xl p-2"
        >
          <textarea
            ref={textareaRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={t ? "Raksti savu jautājumu…" : "Type your message…"}
            rows={1}
            disabled={isLoading}
            className="flex-1 bg-transparent resize-none px-3 py-2 text-sm text-white placeholder:text-white/30 focus:outline-none max-h-32 disabled:opacity-50"
            style={{ minHeight: "40px" }}
            onInput={(e) => {
              const target = e.target as HTMLTextAreaElement;
              target.style.height = "auto";
              target.style.height = `${Math.min(target.scrollHeight, 128)}px`;
            }}
          />
          <button
            type="submit"
            disabled={!input.trim() || isLoading}
            className="p-2.5 rounded-xl bg-accent hover:bg-accent-hover disabled:opacity-40 disabled:cursor-not-allowed transition-colors shrink-0"
            aria-label={t ? "Sūtīt" : "Send"}
          >
            <Send className="w-4 h-4 text-white" />
          </button>
        </form>
        <p className="text-center text-xs text-white/20 mt-2">
          {t
            ? "Mans AI var kļūdīties. Pārbaudi svarīgu informāciju."
            : "AI can make mistakes. Verify important information."}
        </p>
      </div>
    </div>
  );
}
