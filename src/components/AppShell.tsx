"use client";

import { Sidebar } from "@/components/Sidebar";
import { ChatArea } from "@/components/ChatArea";
import { SettingsPanel } from "@/components/SettingsPanel";
import { useApp } from "@/components/AppProvider";

export function AppShell() {
  const { isSettingsOpen, isSidebarOpen } = useApp();

  return (
    <div className="flex h-screen overflow-hidden">
      {isSidebarOpen && <Sidebar />}
      <main className="flex-1 flex flex-col min-w-0">
        <ChatArea />
      </main>
      {isSettingsOpen && <SettingsPanel />}
    </div>
  );
}
