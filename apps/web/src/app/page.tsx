import { CameraVision } from "@/components/CameraVision";

export default function Home() {
  return (
    <main className="min-h-screen px-4 py-8 sm:px-6 lg:px-10">
      <div className="mx-auto flex max-w-7xl flex-col gap-6">
        <header className="rounded-[2rem] border-4 border-ink bg-accent px-6 py-8 shadow-panel">
          <p className="text-sm font-bold uppercase tracking-[0.3em] text-ink/80">
            Real-Time Braille Computer Vision
          </p>
          <h1 className="mt-3 font-display text-4xl font-bold tracking-tight text-ink sm:text-6xl">
            BrailleVision
          </h1>
          <p className="mt-4 max-w-3xl text-lg leading-8 text-ink/80 sm:text-xl">
            Point the rear camera at embossed Braille and receive instant written and spoken English with live tactile dot detection.
          </p>
        </header>
        <CameraVision />
      </div>
    </main>
  );
}
