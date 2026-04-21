import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import Link from "next/link";
import { AlertTriangle } from "lucide-react";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "Crisis Alert System",
  description: "Real-time social media crisis detection — BERT + LSTM + LDA",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className={`${inter.className} bg-gray-50 min-h-screen`}>
        <header className="bg-white border-b border-gray-200 sticky top-0 z-50">
          <div className="max-w-6xl mx-auto px-4 h-14 flex items-center justify-between">
            <Link href="/" className="flex items-center gap-2 font-bold text-gray-900">
              <AlertTriangle size={20} className="text-red-600" />
              Crisis Alert System
            </Link>
            <nav className="flex items-center gap-6 text-sm font-medium text-gray-600">
              <Link href="/"         className="hover:text-gray-900 transition-colors">Dashboard</Link>
              <Link href="/analyzer" className="hover:text-gray-900 transition-colors">Analyzer</Link>
              <Link href="/fetch"    className="hover:text-gray-900 transition-colors">Fetch Tweets</Link>
              <Link href="/alerts"   className="hover:text-gray-900 transition-colors">Alerts</Link>
            </nav>
          </div>
        </header>
        <main className="max-w-6xl mx-auto px-4 py-8">
          {children}
        </main>
      </body>
    </html>
  );
}
