import type { Metadata } from "next";
import "./globals.css";
import { Toaster } from "@/components/ui/toaster";
import { ThemeProvider } from "@/lib/theme-context";

export const metadata: Metadata = {
  title: "Intelligent Data Assistant",
  description: "Universal data science platform for EDA, cleaning, ML training, and predictions. Works with any dataset.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body
        className="antialiased bg-background text-foreground min-h-screen"
        style={{
          ['--font-geist-sans' as string]: 'ui-sans-serif, system-ui, sans-serif',
          ['--font-geist-mono' as string]: 'ui-monospace, SFMono-Regular, monospace',
        }}
      >
        <ThemeProvider>
          {children}
          <Toaster />
        </ThemeProvider>
      </body>
    </html>
  );
}
