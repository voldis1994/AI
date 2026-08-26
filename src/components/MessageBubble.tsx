"use client";

import { Bot, User } from "lucide-react";
import { useApp } from "@/components/AppProvider";
import type { ChatMessage } from "@/lib/types";

function formatTime(iso: string): string {
  return new Date(iso).toLocaleTimeString("lv-LV", {
    hour: "2-digit",
    minute: "2-digit",
  });
}

export function MessageBubble({ message }: { message: ChatMessage }) {
  const { profile } = useApp();
  const isUser = message.role === "user";

  return (
    <div
      className={`flex gap-3 animate-slide-up ${isUser ? "flex-row-reverse" : ""}`}
    >
      <div
        className={`w-8 h-8 rounded-lg flex items-center justify-center shrink-0 ${
          isUser
            ? "bg-accent/30"
            : "bg-gradient-to-br from-indigo-500/30 to-purple-500/30"
        }`}
      >
        {isUser ? (
          <User className="w-4 h-4 text-accent-hover" />
        ) : (
          <Bot className="w-4 h-4 text-purple-300" />
        )}
      </div>

      <div className={`flex-1 max-w-[80%] ${isUser ? "text-right" : ""}`}>
        <div
          className={`inline-block px-4 py-3 rounded-2xl text-sm leading-relaxed ${
            isUser
              ? "bg-accent/20 text-white rounded-tr-sm"
              : "glass text-white/90 rounded-tl-sm"
          }`}
        >
          <p className="whitespace-pre-wrap">{message.content}</p>
        </div>
        <p className="text-xs text-white/30 mt-1 px-1">
          {formatTime(message.createdAt)}
        </p>
      </div>
    </div>
  );
}

export function TypingIndicator() {
  const { profile } = useApp();

  return (
    <div className="flex gap-3 animate-fade-in">
      <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-indigo-500/30 to-purple-500/30 flex items-center justify-center">
        <Bot className="w-4 h-4 text-purple-300" />
      </div>
      <div className="glass px-4 py-3 rounded-2xl rounded-tl-sm">
        <div className="flex gap-1.5">
          <span className="w-2 h-2 bg-white/40 rounded-full animate-pulse-soft" />
          <span
            className="w-2 h-2 bg-white/40 rounded-full animate-pulse-soft"
            style={{ animationDelay: "0.2s" }}
          />
          <span
            className="w-2 h-2 bg-white/40 rounded-full animate-pulse-soft"
            style={{ animationDelay: "0.4s" }}
          />
        </div>
        <p className="text-xs text-white/30 mt-2">
          {profile.assistantName}{" "}
          {profile.language === "lv" ? "domā…" : "is thinking…"}
        </p>
      </div>
    </div>
  );
}
