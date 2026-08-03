import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import { WorkbenchShell } from "@/components/layout/WorkbenchShell";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
});

export const metadata: Metadata = {
  title: "VisionForge | Computer Vision Workbench",
  description:
    "A modern Computer Vision Workbench for integrating, benchmarking, visualizing, and experimenting with foundation models.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body
        className={`${inter.variable} font-sans bg-[#0a0a0a] text-neutral-100 antialiased min-h-screen selection:bg-blue-600/30 selection:text-blue-200`}
      >
        <WorkbenchShell>{children}</WorkbenchShell>
      </body>
    </html>
  );
}
