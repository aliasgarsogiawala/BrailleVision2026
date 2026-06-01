"use client";

import type { ChangeEvent } from "react";
import { useEffect, useRef, useState } from "react";

type VisionResponse = {
  text: string;
  confidence: number;
  boxes: number[][];
  debug_image?: string;
};

const BACKEND_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
const PROCESS_WIDTH = 1280;
const PROCESS_HEIGHT = 720;
const ROI_TOP_PERCENT = 20;
const ROI_BOTTOM_PERCENT = 80;
const ROI_LEFT_PERCENT = 20;
const ROI_RIGHT_PERCENT = 80;

export function CameraVision() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const overlayRef = useRef<HTMLCanvasElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const speakingTextRef = useRef("");

  const [isCameraReady, setIsCameraReady] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [text, setText] = useState("Capture a clear Braille photo to translate.");
  const [confidence, setConfidence] = useState(0);
  const [status, setStatus] = useState("Initializing camera");
  const [error, setError] = useState("");
  const [debugImage, setDebugImage] = useState("");

  useEffect(() => {
    let isMounted = true;

    const startCamera = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: {
            facingMode: { ideal: "environment" },
            width: { ideal: PROCESS_WIDTH },
            height: { ideal: PROCESS_HEIGHT },
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
        setStatus("Camera ready. Hold the paper inside the central guide.");
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
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
      }
      window.speechSynthesis.cancel();
    };
  }, []);

  useEffect(() => {
    if (
      !text ||
      text === "Capture a clear Braille photo to translate." ||
      text === "No Braille detected" ||
      text === speakingTextRef.current
    ) {
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
    context.lineWidth = 4;
    context.strokeStyle = "#E4FF4F";
    context.fillStyle = "rgba(228, 255, 79, 0.2)";
    context.font = "bold 22px Trebuchet MS";

    boxes.forEach(([x, y, w, h], index) => {
      context.fillRect(x, y, w, h);
      context.strokeRect(x, y, w, h);
      context.fillStyle = "#111111";
      context.fillRect(x, Math.max(0, y - 32), 88, 28);
      context.fillStyle = "#E4FF4F";
      context.fillText(`Cell ${index + 1}`, x + 8, Math.max(22, y - 10));
      context.fillStyle = "rgba(228, 255, 79, 0.2)";
    });
  };

  const roiBounds = () => {
    const left = Math.round(PROCESS_WIDTH * (ROI_LEFT_PERCENT / 100));
    const top = Math.round(PROCESS_HEIGHT * (ROI_TOP_PERCENT / 100));
    const right = Math.round(PROCESS_WIDTH * (ROI_RIGHT_PERCENT / 100));
    const bottom = Math.round(PROCESS_HEIGHT * (ROI_BOTTOM_PERCENT / 100));
    return {
      left,
      top,
      width: right - left,
      height: bottom - top,
    };
  };

  const processFormData = async (formData: FormData) => {
    const response = await fetch(`${BACKEND_URL}/api/process-braille/upload`, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`Vision API error: ${response.status}`);
    }

    const data = (await response.json()) as VisionResponse;
    setText(data.text || "No Braille detected");
    setConfidence(data.confidence || 0);
    setDebugImage(data.debug_image || "");
    setStatus(data.text ? "Translation updated" : "No Braille found. Adjust lighting and try again.");
    drawBoxes(data.boxes || []);
  };

  const capturePhoto = async () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || video.videoWidth === 0 || video.videoHeight === 0) {
      return;
    }

    setIsProcessing(true);
    setStatus("Capturing still image");
    setError("");

    try {
      canvas.width = PROCESS_WIDTH;
      canvas.height = PROCESS_HEIGHT;
      const context = canvas.getContext("2d");
      if (!context) {
        throw new Error("Canvas context unavailable.");
      }

      context.drawImage(video, 0, 0, PROCESS_WIDTH, PROCESS_HEIGHT);
      const { left, top, width, height } = roiBounds();
      const cropped = context.getImageData(left, top, width, height);

      canvas.width = width;
      canvas.height = height;
      const croppedContext = canvas.getContext("2d");
      if (!croppedContext) {
        throw new Error("Canvas context unavailable after crop.");
      }
      croppedContext.putImageData(cropped, 0, 0);

      const blob = await new Promise<Blob | null>((resolve) => {
        canvas.toBlob(resolve, "image/jpeg", 0.95);
      });

      if (!blob) {
        throw new Error("Unable to encode captured photo.");
      }

      setStatus("Sending photo to vision engine");
      const formData = new FormData();
      formData.append("file", blob, "capture.jpg");
      await processFormData(formData);
    } catch (processingError) {
      const message =
        processingError instanceof Error
          ? processingError.message
          : "Unable to process captured photo.";
      setError(message);
      setStatus("Capture failed");
    } finally {
      setIsProcessing(false);
    }
  };

  const handleUploadClick = () => {
    fileInputRef.current?.click();
  };

  const handleFileChange = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) {
      return;
    }

    setIsProcessing(true);
    setStatus("Uploading selected photo");
    setError("");

    try {
      const formData = new FormData();
      formData.append("file", file);
      await processFormData(formData);
    } catch (processingError) {
      const message =
        processingError instanceof Error
          ? processingError.message
          : "Unable to process uploaded photo.";
      setError(message);
      setStatus("Upload failed");
    } finally {
      event.target.value = "";
      setIsProcessing(false);
    }
  };

  return (
    <section className="grid gap-6 lg:grid-cols-[1.15fr_0.85fr]">
      <div className="overflow-hidden rounded-[2rem] border-4 border-ink bg-ink p-3 shadow-panel">
        <div className="relative overflow-hidden rounded-[1.4rem] border-4 border-panel bg-black">
          <video
            ref={videoRef}
            className="aspect-video w-full object-cover"
            autoPlay
            muted
            playsInline
            aria-label="Live camera preview for Braille capture"
          />
          <canvas
            ref={overlayRef}
            className="pointer-events-none absolute inset-0 h-full w-full"
          />
          <div className="pointer-events-none absolute inset-x-4 top-4 rounded-full border-4 border-accent bg-ink/85 px-4 py-2 text-center text-xs font-bold uppercase tracking-[0.3em] text-accent sm:inset-x-6 sm:top-6 sm:text-sm">
            Live-Scan-View
          </div>
          <div
            className="pointer-events-none absolute rounded-[1.2rem] border-4 border-dashed border-accent/90 sm:rounded-[1.4rem]"
            style={{
              top: `${ROI_TOP_PERCENT}%`,
              bottom: `${100 - ROI_BOTTOM_PERCENT}%`,
              left: `${ROI_LEFT_PERCENT}%`,
              right: `${100 - ROI_RIGHT_PERCENT}%`,
            }}
          />
          <div className="pointer-events-none absolute bottom-3 left-3 right-3 rounded-xl border-4 border-accent bg-ink/85 px-4 py-3 text-center text-[11px] font-bold uppercase tracking-[0.16em] text-panel sm:bottom-5 sm:left-5 sm:right-auto sm:text-sm sm:tracking-[0.18em]">
            Keep Braille centered and evenly lit before capture
          </div>
        </div>
      </div>

      <div className="flex flex-col gap-5 rounded-[2rem] border-4 border-ink bg-panel p-5 shadow-panel sm:p-6">
        <div className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
          <div>
            <p className="text-sm font-bold uppercase tracking-[0.25em] text-ink/70">
              Snapshot Console
            </p>
            <h2 className="mt-2 font-display text-3xl font-bold text-ink">
              Translated Text
            </h2>
          </div>
          <div className="flex w-full flex-col gap-3 sm:w-auto">
            <button
              type="button"
              onClick={() => void capturePhoto()}
              disabled={!isCameraReady || isProcessing}
              className="min-h-16 rounded-[1.4rem] border-4 border-ink bg-accent px-6 py-4 text-lg font-bold text-ink transition disabled:cursor-not-allowed disabled:bg-white disabled:text-ink/50"
            >
              {isProcessing ? "Processing..." : "Capture & Translate"}
            </button>
            <button
              type="button"
              onClick={handleUploadClick}
              disabled={isProcessing}
              className="min-h-14 rounded-[1.2rem] border-4 border-ink bg-white px-6 py-3 text-base font-bold text-ink transition disabled:cursor-not-allowed disabled:text-ink/50"
            >
              Upload Photo
            </button>
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              className="hidden"
              onChange={(event) => void handleFileChange(event)}
            />
          </div>
        </div>

        <div className="rounded-[1.5rem] border-4 border-ink bg-white p-5">
          <p className="text-sm font-bold uppercase tracking-[0.3em] text-ink/60">
            Parsed Text
          </p>
          <p className="mt-4 min-h-28 text-3xl font-bold leading-tight text-ink sm:text-4xl">
            {text}
          </p>
        </div>

        {debugImage ? (
          <div className="rounded-[1.5rem] border-4 border-ink bg-white p-5">
            <p className="text-sm font-bold uppercase tracking-[0.3em] text-ink/60">
              OpenCV Mask View
            </p>
            <img
              src={debugImage}
              alt="Thresholded OpenCV mask showing Braille detection"
              className="mt-4 w-full rounded-xl border-4 border-ink bg-panel object-contain"
            />
          </div>
        ) : null}

        <div className="grid gap-4 sm:grid-cols-2">
          <div className="rounded-[1.5rem] border-4 border-ink bg-signal px-5 py-4 text-ink">
            <p className="text-sm font-bold uppercase tracking-[0.25em]">Confidence</p>
            <p className="mt-3 text-4xl font-bold">{Math.round(confidence * 100)}%</p>
          </div>
          <div className="rounded-[1.5rem] border-4 border-ink bg-white px-5 py-4 text-ink">
            <p className="text-sm font-bold uppercase tracking-[0.25em]">Status</p>
            <p className="mt-3 text-lg font-bold sm:text-xl">{status}</p>
          </div>
        </div>

        <div className="rounded-[1.5rem] border-4 border-ink bg-ink px-5 py-4 text-panel">
          <p className="text-sm font-bold uppercase tracking-[0.25em] text-accent">
            Accessibility
          </p>
          <p className="mt-3 text-base leading-7 sm:text-lg">
            Spoken feedback only fires when the translated text changes to a fully new string, keeping repeated captures calm on mobile.
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
