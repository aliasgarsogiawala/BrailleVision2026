"use client";

import { useEffect, useRef, useState } from "react";

type VisionResponse = {
  text: string;
  confidence: number;
  boxes: number[][];
};

const API_URL = "http://localhost:8000/api/v1/process-frame";
const CAPTURE_INTERVAL_MS = 300;
const PROCESS_WIDTH = 640;
const PROCESS_HEIGHT = 480;
const ROI_TOP_PERCENT = 20;
const ROI_BOTTOM_PERCENT = 80;
const ROI_LEFT_PERCENT = 15;
const ROI_RIGHT_PERCENT = 85;

export function CameraVision() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const overlayRef = useRef<HTMLCanvasElement | null>(null);
  const frameLoopRef = useRef<number | null>(null);
  const lastCaptureRef = useRef(0);
  const streamRef = useRef<MediaStream | null>(null);
  const speakingTextRef = useRef("");
  const processingRef = useRef(false);

  const [isScanning, setIsScanning] = useState(true);
  const [isCameraReady, setIsCameraReady] = useState(false);
  const [text, setText] = useState("Waiting for Braille...");
  const [confidence, setConfidence] = useState(0);
  const [status, setStatus] = useState("Initializing camera");
  const [error, setError] = useState("");

  useEffect(() => {
    let isMounted = true;

    const startCamera = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: {
            facingMode: { ideal: "environment" },
            width: { ideal: 1280 },
            height: { ideal: 960 },
            aspectRatio: { ideal: PROCESS_WIDTH / PROCESS_HEIGHT },
          },
          audio: false,
        });

        if (!isMounted || !videoRef.current) {
          stream.getTracks().forEach((track) => track.stop());
          return;
        }

        streamRef.current = stream;
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
        setIsCameraReady(true);
        setStatus("Camera ready. Keep Braille inside the center guide.");
      } catch (cameraError) {
        const message =
          cameraError instanceof Error
            ? cameraError.message
            : "Camera access was denied.";
        setError(message);
        setStatus("Camera unavailable");
      }
    };

    void startCamera();

    return () => {
      isMounted = false;
      if (frameLoopRef.current) {
        cancelAnimationFrame(frameLoopRef.current);
      }
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
      }
      window.speechSynthesis.cancel();
    };
  }, []);

  useEffect(() => {
    if (!isCameraReady) {
      return;
    }

    const loop = (timestamp: number) => {
      frameLoopRef.current = window.requestAnimationFrame(loop);
      if (!isScanning || processingRef.current) {
        return;
      }
      if (timestamp - lastCaptureRef.current < CAPTURE_INTERVAL_MS) {
        return;
      }
      lastCaptureRef.current = timestamp;
      void captureAndProcessFrame();
    };

    frameLoopRef.current = window.requestAnimationFrame(loop);
    return () => {
      if (frameLoopRef.current) {
        cancelAnimationFrame(frameLoopRef.current);
      }
    };
  }, [isCameraReady, isScanning]);

  useEffect(() => {
    if (!text || text === "Waiting for Braille..." || text === speakingTextRef.current) {
      return;
    }

    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = 0.95;
    utterance.pitch = 1;
    utterance.lang = "en-US";
    speakingTextRef.current = text;
    window.speechSynthesis.speak(utterance);
  }, [text]);

  const drawBoxes = (boxes: number[][]) => {
    const overlay = overlayRef.current;
    if (!overlay) {
      return;
    }

    overlay.width = PROCESS_WIDTH;
    overlay.height = PROCESS_HEIGHT;
    const context = overlay.getContext("2d");
    if (!context) {
      return;
    }

    context.clearRect(0, 0, PROCESS_WIDTH, PROCESS_HEIGHT);
    context.lineWidth = 3;
    context.strokeStyle = "#E4FF4F";
    context.fillStyle = "rgba(228, 255, 79, 0.18)";
    context.font = "bold 18px Trebuchet MS";

    boxes.forEach(([x, y, w, h], index) => {
      context.fillRect(x, y, w, h);
      context.strokeRect(x, y, w, h);
      context.fillStyle = "#111111";
      context.fillRect(x, Math.max(0, y - 26), 74, 24);
      context.fillStyle = "#E4FF4F";
      context.fillText(`Cell ${index + 1}`, x + 6, Math.max(17, y - 8));
      context.fillStyle = "rgba(228, 255, 79, 0.18)";
    });
  };

  const captureAndProcessFrame = async () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || video.videoWidth === 0 || video.videoHeight === 0) {
      return;
    }

    processingRef.current = true;
    setStatus("Analyzing ROI for Braille cells");

    try {
      canvas.width = PROCESS_WIDTH;
      canvas.height = PROCESS_HEIGHT;
      const context = canvas.getContext("2d");
      if (!context) {
        throw new Error("Canvas context unavailable.");
      }

      context.drawImage(video, 0, 0, PROCESS_WIDTH, PROCESS_HEIGHT);

      const blob = await new Promise<Blob | null>((resolve) => {
        canvas.toBlob(resolve, "image/jpeg", 0.7);
      });

      if (!blob) {
        throw new Error("Unable to encode video frame.");
      }

      const formData = new FormData();
      formData.append("file", blob, "frame.jpg");

      const response = await fetch(API_URL, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Vision API error: ${response.status}`);
      }

      const payload = (await response.json()) as VisionResponse;
      setText(payload.text || "No Braille detected");
      setConfidence(payload.confidence || 0);
      setStatus(payload.text ? "Translation updated" : "Keep the paper inside the scan window");
      setError("");
      drawBoxes(payload.boxes || []);
    } catch (processingError) {
      const message =
        processingError instanceof Error
          ? processingError.message
          : "Unable to process frame.";
      setError(message);
      setStatus("Processing interrupted");
    } finally {
      processingRef.current = false;
    }
  };

  return (
    <section className="grid gap-6 lg:grid-cols-[1.2fr_0.8fr]">
      <div className="overflow-hidden rounded-[2rem] border-4 border-ink bg-ink p-3 shadow-panel">
        <div className="relative overflow-hidden rounded-[1.4rem] border-4 border-panel bg-black">
          <video
            ref={videoRef}
            className="aspect-[4/3] w-full object-cover"
            autoPlay
            muted
            playsInline
            aria-label="Live camera feed for Braille scanning"
          />
          <canvas
            ref={overlayRef}
            className="pointer-events-none absolute inset-0 h-full w-full"
          />
          <div className="pointer-events-none absolute inset-x-6 top-6 rounded-full border-4 border-accent bg-ink/85 px-4 py-2 text-sm font-bold uppercase tracking-[0.3em] text-accent">
            Live-Scan-View
          </div>
          <div
            className="pointer-events-none absolute border-4 border-dashed border-accent/90 rounded-[1.4rem]"
            style={{
              top: `${ROI_TOP_PERCENT}%`,
              bottom: `${100 - ROI_BOTTOM_PERCENT}%`,
              left: `${ROI_LEFT_PERCENT}%`,
              right: `${100 - ROI_RIGHT_PERCENT}%`,
            }}
          />
          <div className="pointer-events-none absolute bottom-5 left-5 rounded-xl border-4 border-accent bg-ink/85 px-4 py-3 text-sm font-bold uppercase tracking-[0.18em] text-panel">
            Align Braille paper inside the dashed ROI only
          </div>
        </div>
      </div>

      <div className="flex flex-col gap-5 rounded-[2rem] border-4 border-ink bg-panel p-6 shadow-panel">
        <div className="flex items-start justify-between gap-4">
          <div>
            <p className="text-sm font-bold uppercase tracking-[0.25em] text-ink/70">
              Recognition Console
            </p>
            <h2 className="mt-2 font-display text-3xl font-bold text-ink">
              English Output
            </h2>
          </div>
          <button
            type="button"
            onClick={() => setIsScanning((current) => !current)}
            className={`rounded-full border-4 px-5 py-3 text-base font-bold transition ${
              isScanning
                ? "border-ink bg-accent text-ink"
                : "border-ink bg-white text-ink"
            }`}
            aria-pressed={isScanning}
          >
            {isScanning ? "Pause Scanner" : "Resume Scanner"}
          </button>
        </div>

        <div className="rounded-[1.5rem] border-4 border-ink bg-white p-5">
          <p className="text-sm font-bold uppercase tracking-[0.3em] text-ink/60">
            Parsed Text
          </p>
          <p className="mt-4 min-h-28 text-3xl font-bold leading-tight text-ink sm:text-4xl">
            {text}
          </p>
        </div>

        <div className="grid gap-4 sm:grid-cols-2">
          <div className="rounded-[1.5rem] border-4 border-ink bg-signal px-5 py-4 text-ink">
            <p className="text-sm font-bold uppercase tracking-[0.25em]">Confidence</p>
            <p className="mt-3 text-4xl font-bold">{Math.round(confidence * 100)}%</p>
          </div>
          <div className="rounded-[1.5rem] border-4 border-ink bg-white px-5 py-4 text-ink">
            <p className="text-sm font-bold uppercase tracking-[0.25em]">Status</p>
            <p className="mt-3 text-xl font-bold">{status}</p>
          </div>
        </div>

        <div className="rounded-[1.5rem] border-4 border-ink bg-ink px-5 py-4 text-panel">
          <p className="text-sm font-bold uppercase tracking-[0.25em] text-accent">
            Accessibility
          </p>
          <p className="mt-3 text-lg leading-7">
            Spoken feedback only fires when the translated text changes to a fully new string, preventing stutter loops.
          </p>
        </div>

        {error ? (
          <div
            className="rounded-[1.5rem] border-4 border-ink bg-alert px-5 py-4 font-bold text-ink"
            role="alert"
          >
            {error}
          </div>
        ) : null}
      </div>

      <canvas ref={canvasRef} className="hidden" />
    </section>
  );
}
