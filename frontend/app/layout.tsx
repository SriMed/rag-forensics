import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "RAG Forensics",
  description: "Inspect evidence, diagnostic hypotheses, and follow-up tests for RAG answers.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="h-full antialiased">
      <body className="min-h-full flex flex-col">{children}</body>
    </html>
  );
}
