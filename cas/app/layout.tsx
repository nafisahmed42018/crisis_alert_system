import type { Metadata } from "next";
import { Geist_Mono, Inter } from "next/font/google";
import "./globals.css";
import Link from "next/link";
import { AlertTriangle } from "lucide-react";
import { ThemeProvider } from "@/components/theme-provider";
import { cn } from "@/lib/utils";

const inter = Inter({ subsets: ["latin"], variable: "--font-sans" });
const fontMono = Geist_Mono({ subsets: ["latin"], variable: "--font-mono" });

export const metadata: Metadata = {
  title: "Crisis Alert System",
  description: "Real-time social media crisis detection — BERT + LSTM + LDA",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html
      lang="en"
      suppressHydrationWarning
      className={cn("antialiased", fontMono.variable, inter.variable)}
    >
      <body>
        <ThemeProvider>
          <header className="sticky top-0 z-50 border-b bg-background/80 backdrop-blur">
            <div className="mx-auto flex h-14 max-w-6xl items-center justify-between px-4">
              <Link href="/" className="flex items-center gap-2 font-bold text-foreground">
                <AlertTriangle size={20} className="text-destructive" />
                Crisis Alert System
              </Link>
              <nav className="flex items-center gap-6 text-sm font-medium text-muted-foreground">
                <Link href="/" className="hover:text-foreground transition-colors">Dashboard</Link>
                <Link href="/analyzer" className="hover:text-foreground transition-colors">Analyzer</Link>
                <Link href="/fetch" className="hover:text-foreground transition-colors">Fetch Tweets</Link>
                <Link href="/alerts" className="hover:text-foreground transition-colors">Alerts</Link>
              </nav>
            </div>
          </header>
          <main className="mx-auto max-w-6xl px-4 py-8">{children}</main>
        </ThemeProvider>
      </body>
    </html>
  );
}
