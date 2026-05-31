import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "BrailleVision",
  description: "Real-time Braille to English camera translator",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
