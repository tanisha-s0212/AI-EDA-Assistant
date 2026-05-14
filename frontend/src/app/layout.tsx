import type { Metadata } from "next";
import Script from "next/script";
import "./globals.css";
import { Toaster } from "@/components/ui/toaster";
import { ThemeProvider } from "@/lib/theme-context";

export const metadata: Metadata = {
  title: "Intelligent Data Assistant",
  description: "Universal data science platform for EDA, data cleaning, ML training, and predictions. Works with any dataset.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className="antialiased bg-background text-foreground min-h-screen">
        <ThemeProvider>
          {children}
          <Toaster />
        </ThemeProvider>
        <Script src="http://192.168.1.87:5055/launcher.js" strategy="afterInteractive" />
      </body>
    </html>
  );
}
