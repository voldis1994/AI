import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import { AppProvider } from "@/components/AppProvider";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Mans AI — Personīgais asistents",
  description: "Tavs personīgais AI asistents ar atmiņu un personalizāciju",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="lv">
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased bg-surface text-white min-h-screen`}
      >
        <AppProvider>{children}</AppProvider>
      </body>
    </html>
  );
}
