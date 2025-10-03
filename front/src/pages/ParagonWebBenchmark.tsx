import React, {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";

/**
 * ParagonWebBenchmark
 * ------------------------------------------------------------
 * Front‑end analogue of the backend "Evaluate / Telemetry" flow.
 *
 * What it does
 *  - Discovers models from `/models/manifest.json` (same shape as backend)
 *  - For each model:
 *      * Loads network JSON (expects Paragon JS/wasm to be available)
 *      * Tries WebGPU init; if that fails, marks GPU as unavailable and runs CPU only
 *      * Runs 10 samples (digits 0..9) from `/mnist_png/all/<d>/img_*.png`
 *      * Measures per‑sample latency, records predictions & top1 score
 *      * If GPU path is available, also runs CPU path and computes drift (MAE, max abs)
 *  - Streams progress to the UI as it goes (per‑model live log + table)
 *  - Produces a JSON report compatible with the backend uploader
 *  - Exposes buttons to Download (JSON/HTML) and Submit (POST /upload)
 *
 * Assumptions
 *  - A global `Paragon` runtime is available (from your Portal/WASM bootstrap)
 *  - Model JSON files live in `/models/*.json` and `manifest.json` is `{ id, filename }[]`
 *  - One example PNG per digit lives under `/mnist_png/all/<digit>/`.
 *    If `img_00000.png` is missing, we probe subsequent indices up to MAX_PROBE.
 *
 * You can adapt the `ParagonAdapter` to match your Portal/Paracast JS surface.
 */

// ===== Types matching the backend telemetry schema (subset) =====

type TelemetrySource = "wasm-ionic" | "wasm-bun" | "native" | "web";

interface SystemInfo {
  architecture?: string;
  os?: string;
  os_version?: string;
  cpu_model?: string;
  gpu_model?: string;
  device_model?: string;
  ram_bytes?: number;
  gpus?: Array<Record<string, string>>;
}

interface ADHDBuckets {
  CPUCorrect: number;
  CPUWrong: number;
  GPUCorrect: number;
  GPUWrong: number;
  CPUOffBy1: number;
  GPUOffBy1: number;
  Agree: number;
  Disagree: number;
}

interface ADHDSample {
  Digit: number;
  Idx: number;
  CPUPred: number;
  GPUPred: number;
  CPUBucket: "correct" | "off_by_1" | "wrong";
  GPUBucket: "correct" | "off_by_1" | "wrong";
  Agreement: "agree" | "disagree";
}

interface ADHDScore {
  Top1AccuracyCPU: number;
  Top1AccuracyGPU: number;
  CPUvsGPUAgreeCount: number;
  AvgDriftMAE: number;
  MaxDriftMaxAbs: number;
  Buckets: ADHDBuckets;
  PerSample: ADHDSample[];
}

interface SampleTiming {
  Digit: number;
  Idx: number;
  ElapsedMS: number;
  Pred: number;
  Top1Score: number;
  Output: number[];
}

interface DriftMetrics {
  Digit: number;
  Idx: number;
  MaxAbs: number;
  MAE: number;
}

interface ModelRun {
  ModelFile: string;
  WebGPUInitOK: boolean;
  WebGPUInitTimeMS: number;
  CPU: SampleTiming[];
  GPU: SampleTiming[];
  Drift: DriftMetrics[];
  ADHD10: ADHDScore;
  Summary?: Record<string, any>;
  Meta?: Record<string, string>;
}

interface TelemetryReport {
  Version: string;
  Source: TelemetrySource;
  MachineID: string;
  System: SystemInfo;
  FromHost: string;
  ModelsUsed: string[];
  Samples: number[];
  StartedAt: string; // ISO
  EndedAt: string; // ISO
  Notes?: string;
  PerModel: ModelRun[];
}

// ===== Minimal Paragon JS surface (adapt as needed) =====

type ParagonNetwork = {
  WebGPUNative: boolean;
  InitializeOptimizedGPU: () => Promise<void> | void;
  CleanupOptimizedGPU: () => Promise<void> | void;
  Forward: (img: number[][]) => void; // expects 28x28 for MNIST
  ExtractOutput: () => number[]; // softmax logits
};

type ParagonModule = {
  LoadNamedNetworkFromJSON: (
    jsonText: string
  ) => Promise<ParagonNetwork | any> | ParagonNetwork | any;
  NewNetworkLike?: (
    jsonText: string
  ) => Promise<ParagonNetwork> | ParagonNetwork; // optional helper if you have it
};

// Hook your real Portal/Paragon surface here
declare global {
  interface Window {
    Paragon?: ParagonModule;
  }
}

// A small adapter that builds a runnable network from a model JSON string.
async function buildParagonFromJSON(jsonText: string): Promise<ParagonNetwork> {
  if (!window.Paragon)
    throw new Error(
      "Paragon JS runtime not found. Make sure Portal/WASM is initialized."
    );
  const loaded = await window.Paragon.LoadNamedNetworkFromJSON(jsonText);
  // Some runtimes return a typed variant; ensure it exposes the methods we need
  const nn = loaded as ParagonNetwork;
  if (
    typeof nn.Forward !== "function" ||
    typeof nn.ExtractOutput !== "function"
  ) {
    throw new Error(
      "Loaded network missing required methods (Forward/ExtractOutput)"
    );
  }
  return nn;
}

// ===== Utilities =====

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

function argmax(v: number[]): number {
  let best = 0;
  for (let i = 1; i < v.length; i++) if (v[i] > v[best]) best = i;
  return best;
}
function top1(p: number[]): number {
  return Math.max(...p);
}
function driftPair(a: number[], b: number[]) {
  let maxAbs = 0,
    sum = 0;
  for (let i = 0; i < Math.min(a.length, b.length); i++) {
    const d = Math.abs(a[i] - b[i]);
    sum += d;
    if (d > maxAbs) maxAbs = d;
  }
  return { maxAbs, mae: sum / Math.min(a.length, b.length || 1) };
}

function hashString(s: string) {
  // Poor-man's hash for a stable MachineID in browser context
  let h = 2166136261 >>> 0;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 16777619) >>> 0;
  }
  return ("00000000" + h.toString(16)).slice(-8);
}

async function getSystemInfo(): Promise<SystemInfo> {
  const nav = (navigator || {}) as any;
  const gpu = (navigator as any).gpu ? "WebGPU present" : "WebGPU not found";
  return {
    architecture: nav?.userAgentData?.architecture || undefined,
    os: nav?.userAgentData?.platform || undefined,
    os_version: undefined,
    cpu_model: nav?.hardwareConcurrency
      ? `${nav.hardwareConcurrency} threads`
      : undefined,
    gpu_model: gpu,
    device_model: nav?.userAgentData?.model || undefined,
    ram_bytes: (nav as any).deviceMemory
      ? (nav as any).deviceMemory * 1024 * 1024 * 1024
      : undefined,
  };
}

async function computeMachineId(sys: SystemInfo) {
  const ua = navigator.userAgent;
  const lang = navigator.language;
  const scr = `${window.screen.width}x${window.screen.height}@${window.devicePixelRatio}`;
  return hashString(
    [ua, lang, scr, sys.cpu_model, sys.gpu_model].filter(Boolean).join("|")
  );
}

async function fetchJSON(url: string) {
  const r = await fetch(url, { cache: "no-cache" });
  if (!r.ok) throw new Error(`${url}: ${r.status}`);
  return r.json();
}

async function fetchText(url: string) {
  const r = await fetch(url, { cache: "no-cache" });
  if (!r.ok) throw new Error(`${url}: ${r.status}`);
  return r.text();
}

// Load first usable PNG for each digit 0..9 and convert to 28x28 grayscale [0..1]
const MAX_PROBE = 200; // how deep to look for a first sample

// We don't have directory listing in browsers, so we probe common filename patterns per digit.
const DIGIT_FILENAME_PATTERNS = (d: number) => [
  ...Array.from(
    { length: MAX_PROBE },
    (_, i) => `img_${String(i).padStart(5, "0")}.png`
  ),
  `${d}.png`,
  `sample_${d}.png`,
  ...Array.from({ length: 5 }, (_, i) => `${d}_${i}.png`),
  ...Array.from({ length: 5 }, (_, i) => `idx_${i}.png`),
];
async function loadDigitPNG(
  d: number,
  base = "/mnist_png/all"
): Promise<number[][]> {
  const dir = `${base}/${d}`;
  const candidates = DIGIT_FILENAME_PATTERNS(d);
  for (const name of candidates) {
    const url = `${dir}/${name}`;
    try {
      const img = await loadImage(url);
      const data = rasterizeToGray(img, 28, 28);
      return data;
    } catch {
      /* try next candidate */
    }
  }
  // Last‑resort guesses when files are flat under /all
  const flat = [
    `${base}/${d}.png`,
    `${base}/img_${String(d).padStart(5, "0")}.png`,
  ];
  for (const url of flat) {
    try {
      const img = await loadImage(url);
      const data = rasterizeToGray(img, 28, 28);
      return data;
    } catch {
      /* keep going */
    }
  }
  throw new Error(`No PNG sample found for digit ${d} under ${base}`);
}

function loadImage(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error("image load failed"));
    img.src = src;
  });
}

function rasterizeToGray(
  img: HTMLImageElement,
  w: number,
  h: number
): number[][] {
  const canvas = document.createElement("canvas");
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("2D context not available");
  ctx.drawImage(img, 0, 0, w, h);
  const id = ctx.getImageData(0, 0, w, h);
  const out: number[][] = new Array(h).fill(0).map(() => new Array(w).fill(0));
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const idx = (y * w + x) * 4;
      const r = id.data[idx],
        g = id.data[idx + 1],
        b = id.data[idx + 2];
      // grayscale luma, normalized
      const v = (0.299 * r + 0.587 * g + 0.114 * b) / 255;
      out[y][x] = v;
    }
  }
  return out;
}

// ===== Benchmark core (OOP-ish controller + React shell) =====

class BenchmarkController {
  private cancel = false;
  public readonly logs: string[] = [];
  public readonly perModel: ModelRun[] = [];
  public readonly onProgress: () => void;
  private modelsBase: string;
  private samplesBase: string;

  constructor(
    onProgress: () => void,
    opts?: { modelsBase?: string; samplesBase?: string }
  ) {
    this.onProgress = onProgress;
    this.modelsBase = opts?.modelsBase ?? "/models";
    this.samplesBase = opts?.samplesBase ?? "/mnist_png/all";
  }
  stop() {
    this.cancel = true;
  }

  private log(s: string) {
    this.logs.push(s);
    this.onProgress();
  }

  async runAll(hostBase = window.location.origin): Promise<TelemetryReport> {
    this.cancel = false;

    const sys = await getSystemInfo();
    const machineId = await computeMachineId(sys);

    // Try modelsBase/manifest.json then a common alternate at /models_remote
    let manifest: Array<{ id: string; filename: string }> = [];
    try {
      manifest = await fetchJSON(`${this.modelsBase}/manifest.json`);
    } catch {
      try {
        const alt = this.modelsBase.replace(/\/models$/, "/models_remote");
        manifest = await fetchJSON(`${alt}/manifest.json`);
        this.log(`Found manifest at ${alt}/manifest.json`);
        this.modelsBase = alt; // switch base
      } catch {
        manifest = [];
      }
    }

    if (!manifest.length)
      this.log(
        "No manifest found under /models. You can still Add Model JSON manually."
      );

    const models: string[] = manifest.map((m) => m.filename);
    const samples = Array.from({ length: 10 }, (_, i) => i);
    const startedAt = new Date();

    // Preload samples
    const digitInputs: Record<number, number[][]> = {};
    for (const d of samples) {
      try {
        digitInputs[d] = await loadDigitPNG(d, this.samplesBase);
        this.log(`Loaded digit ${d} PNG sample`);
      } catch (e: any) {
        this.log(`⚠️  ${e?.message || e}`);
      }
      if (this.cancel) throw new Error("cancelled");
    }

    for (let i = 0; i < models.length; i++) {
      const mf = models[i];
      try {
        const mr = await this.runSingleModel(
          `${this.modelsBase}/${mf}`,
          digitInputs
        );
        this.perModel.push(mr);
        this.onProgress();
      } catch (e: any) {
        this.log(`❌ Model ${mf} failed: ${e?.message || e}`);
        // still push a stub so the UI shows it tried
        this.perModel.push({
          ModelFile: mf,
          WebGPUInitOK: false,
          WebGPUInitTimeMS: 0,
          CPU: [],
          GPU: [],
          Drift: [],
          ADHD10: emptyADHD(),
        });
        this.onProgress();
      }
      if (this.cancel) break;
      await sleep(10); // yield to paint
    }

    const endedAt = new Date();

    const report: TelemetryReport = {
      Version: "1.2.0-web",
      Source: "web",
      MachineID: machineId,
      System: sys,
      FromHost: hostBase,
      ModelsUsed: models,
      Samples: samples,
      StartedAt: startedAt.toISOString(),
      EndedAt: endedAt.toISOString(),
      PerModel: this.perModel,
    };

    return report;
  }

  private async runSingleModel(
    modelURL: string,
    digitInputs: Record<number, number[][]>
  ): Promise<ModelRun> {
    this.log(`\n📦 Model: ${modelURL.split("/").pop()}`);
    const jsonText = await fetchText(modelURL);
    const nnCPU = await buildParagonFromJSON(jsonText);
    (nnCPU as any).WebGPUNative = false;

    const nnGPU = await buildParagonFromJSON(jsonText);
    (nnGPU as any).WebGPUNative = true;

    let webgpuOK = false;
    let initMS = 0;
    const t0 = performance.now();
    try {
      await nnGPU.InitializeOptimizedGPU?.();
      webgpuOK = true;
      initMS = performance.now() - t0;
      this.log(`✅ WebGPU initialized in ${initMS.toFixed(2)} ms`);
      // warmup
      const warm = digitInputs[0];
      nnGPU.Forward(warm);
      nnGPU.ExtractOutput();
    } catch (e: any) {
      this.log(
        `⚠️ WebGPU init failed: ${e?.message || e}. Falling back to CPU.`
      );
      (nnGPU as any).WebGPUNative = false;
      initMS = performance.now() - t0;
    }

    const CPU: SampleTiming[] = [];
    const GPU: SampleTiming[] = [];
    const Drift: DriftMetrics[] = [];

    for (let d = 0; d <= 9; d++) {
      const sample = digitInputs[d];
      if (!sample) {
        this.log(`⚠️  Missing sample for digit ${d}`);
        continue;
      }

      // CPU pass
      let t1 = performance.now();
      nnCPU.Forward(sample);
      const outCPU = nnCPU.ExtractOutput();
      let t2 = performance.now();
      const predCPU = argmax(outCPU);
      CPU.push({
        Digit: d,
        Idx: d,
        ElapsedMS: t2 - t1,
        Pred: predCPU,
        Top1Score: top1(outCPU),
        Output: round4(outCPU),
      });

      // GPU (or CPU fallback if WebGPU failed, because nnGPU.WebGPUNative may be false)
      t1 = performance.now();
      nnGPU.Forward(sample);
      const outGPU = nnGPU.ExtractOutput();
      t2 = performance.now();
      const predGPU = argmax(outGPU);
      GPU.push({
        Digit: d,
        Idx: d,
        ElapsedMS: t2 - t1,
        Pred: predGPU,
        Top1Score: top1(outGPU),
        Output: round4(outGPU),
      });

      // drift if both ran distinctly.
      const { maxAbs, mae } = driftPair(outCPU, outGPU);
      Drift.push({ Digit: d, Idx: d, MaxAbs: maxAbs, MAE: mae });

      this.log(
        `Digit ${d} → CPU=${predCPU} (${(CPU.at(-1)?.ElapsedMS || 0).toFixed(
          2
        )}ms)  GPU=${predGPU} (${(GPU.at(-1)?.ElapsedMS || 0).toFixed(
          2
        )}ms)  drift_max=${maxAbs.toFixed(6)} mae=${mae.toFixed(6)}`
      );
      await sleep(0);
    }

    try {
      await nnGPU.CleanupOptimizedGPU?.();
    } catch {}

    const adhd = computeADHD(CPU, GPU);

    return {
      ModelFile: modelURL.split("/").pop() || modelURL,
      WebGPUInitOK: webgpuOK,
      WebGPUInitTimeMS: initMS,
      CPU,
      GPU,
      Drift,
      ADHD10: adhd,
    };
  }
}

function round4(v: number[]) {
  return v.map((x) => Math.round(x * 1e4) / 1e4);
}

function emptyADHD(): ADHDScore {
  return {
    Top1AccuracyCPU: 0,
    Top1AccuracyGPU: 0,
    CPUvsGPUAgreeCount: 0,
    AvgDriftMAE: 0,
    MaxDriftMaxAbs: 0,
    Buckets: {
      CPUCorrect: 0,
      CPUWrong: 0,
      GPUCorrect: 0,
      GPUWrong: 0,
      CPUOffBy1: 0,
      GPUOffBy1: 0,
      Agree: 0,
      Disagree: 0,
    },
    PerSample: [],
  };
}

function computeADHD(CPU: SampleTiming[], GPU: SampleTiming[]): ADHDScore {
  const per: ADHDSample[] = [];
  const B: ADHDBuckets = {
    CPUCorrect: 0,
    CPUWrong: 0,
    GPUCorrect: 0,
    GPUWrong: 0,
    CPUOffBy1: 0,
    GPUOffBy1: 0,
    Agree: 0,
    Disagree: 0,
  };
  let driftSum = 0,
    driftMax = 0,
    agree = 0;

  for (let i = 0; i < Math.min(CPU.length, GPU.length); i++) {
    const c = CPU[i],
      g = GPU[i];
    const truth = c.Digit; // using the known digit target for these fixed samples

    const cpuBucket = bucket(truth, c.Pred);
    const gpuBucket = bucket(truth, g.Pred);
    if (cpuBucket === "correct") B.CPUCorrect++;
    else if (cpuBucket === "off_by_1") B.CPUOffBy1++;
    else B.CPUWrong++;
    if (gpuBucket === "correct") B.GPUCorrect++;
    else if (gpuBucket === "off_by_1") B.GPUOffBy1++;
    else B.GPUWrong++;

    const agreeNow = c.Pred === g.Pred;
    if (agreeNow) {
      B.Agree++;
      agree++;
    } else B.Disagree++;

    // approximate drift using |CPU top1 - GPU top1|
    const dd = Math.abs(c.Top1Score - g.Top1Score);
    driftSum += dd;
    if (dd > driftMax) driftMax = dd;

    per.push({
      Digit: truth,
      Idx: c.Idx,
      CPUPred: c.Pred,
      GPUPred: g.Pred,
      CPUBucket: cpuBucket,
      GPUBucket: gpuBucket,
      Agreement: agreeNow ? "agree" : "disagree",
    });
  }

  const n = Math.max(1, Math.min(CPU.length, GPU.length));
  return {
    Top1AccuracyCPU: (B.CPUCorrect / n) * 100,
    Top1AccuracyGPU: (B.GPUCorrect / n) * 100,
    CPUvsGPUAgreeCount: agree,
    AvgDriftMAE: driftSum / n,
    MaxDriftMaxAbs: driftMax,
    Buckets: B,
    PerSample: per,
  };
}

function bucket(truth: number, pred: number): "correct" | "off_by_1" | "wrong" {
  if (pred === truth) return "correct";
  if (Math.abs(pred - truth) === 1) return "off_by_1";
  return "wrong";
}

// ===== React component =====

export default function ParagonWebBenchmark() {
  const [running, setRunning] = useState(false);
  const [controller, setController] = useState<BenchmarkController | null>(
    null
  );
  const [logsVersion, force] = useState(0);
  const [report, setReport] = useState<TelemetryReport | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [modelsBase, setModelsBase] = useState<string>("/models");
  const [samplesBase, setSamplesBase] = useState<string>("/mnist_png/all");

  // progressive derived state
  const perModel = controller?.perModel ?? [];
  const logs = controller?.logs ?? [];

  const start = useCallback(async () => {
    setError(null);
    const c = new BenchmarkController(() => force((v) => v + 1), {
      modelsBase,
      samplesBase,
    });
    setController(c);
    setRunning(true);
    try {
      console.log(window.location.origin);
      //const r = await c.runAll(window.location.origin);
      const r = await c.runAll("http://localhost:8080");
      setReport(r);
    } catch (e: any) {
      setError(e?.message || String(e));
    } finally {
      setRunning(false);
    }
  }, []);

  const stop = useCallback(() => controller?.stop(), [controller]);

  const canSubmit = !!report || (perModel.length > 0 && !running);

  // ---- Actions: download & submit ----

  const downloadJSON = () => {
    const blob = new Blob(
      [JSON.stringify(report ?? buildPartialReport(perModel), null, 2)],
      { type: "application/json" }
    );
    const a = document.createElement("a");
    const name = `web_telemetry_${Date.now()}.json`;
    a.href = URL.createObjectURL(blob);
    a.download = name;
    a.click();
    URL.revokeObjectURL(a.href);
  };

  const downloadHTML = () => {
    const data = report ?? buildPartialReport(perModel);
    const html = renderHTMLReport(data);
    const blob = new Blob([html], { type: "text/html" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `web_telemetry_${Date.now()}.html`;
    a.click();
    URL.revokeObjectURL(a.href);
  };

  const submitToBackend = async () => {
    try {
      const data = report ?? buildPartialReport(perModel);
      const fname = `telemetry_web_${
        data.MachineID || "anon"
      }_${Date.now()}.json`;
      const form = new FormData();
      form.append(
        "file",
        new Blob([JSON.stringify(data, null, 2)], { type: "application/json" }),
        fname
      );
      const res = await fetch(`/upload`, { method: "POST", body: form });
      if (!res.ok) throw new Error(`Upload failed: ${res.status}`);
      alert("✅ Report uploaded to backend /reports");
    } catch (e: any) {
      alert(`❌ Upload error: ${e?.message || e}`);
    }
  };

  return (
    <div className="mx-auto max-w-6xl p-4 space-y-6">
      <header className="flex items-center justify-between">
        <h1 className="text-2xl font-bold">Paragon Web Benchmark</h1>
        <div className="flex gap-2">
          {!running && (
            <button
              onClick={start}
              className="px-3 py-2 rounded-xl bg-blue-600 text-white shadow"
            >
              Start
            </button>
          )}
          {running && (
            <button
              onClick={stop}
              className="px-3 py-2 rounded-xl bg-amber-600 text-white shadow"
            >
              Stop
            </button>
          )}
          <button
            onClick={downloadJSON}
            disabled={!canSubmit}
            className={`px-3 py-2 rounded-xl ${
              canSubmit
                ? "bg-slate-700 text-white"
                : "bg-slate-300 text-slate-500 cursor-not-allowed"
            }`}
          >
            Download JSON
          </button>
          <button
            onClick={downloadHTML}
            disabled={!canSubmit}
            className={`px-3 py-2 rounded-xl ${
              canSubmit
                ? "bg-slate-700 text-white"
                : "bg-slate-300 text-slate-500 cursor-not-allowed"
            }`}
          >
            Download HTML
          </button>
          <button
            onClick={submitToBackend}
            disabled={!canSubmit}
            className={`px-3 py-2 rounded-xl ${
              canSubmit
                ? "bg-green-600 text-white"
                : "bg-green-200 text-green-600 cursor-not-allowed"
            }`}
          >
            Submit to Backend
          </button>
        </div>
      </header>

      {/* Paths config */}
      <section className="grid grid-cols-1 md:grid-cols-2 gap-3">
        <label className="flex flex-col gap-1">
          <span className="text-sm text-slate-600">Models base URL</span>
          <input
            value={modelsBase}
            onChange={(e) => setModelsBase(e.target.value)}
            className="px-3 py-2 rounded-lg border"
            placeholder="/models"
          />
        </label>
        <label className="flex flex-col gap-1">
          <span className="text-sm text-slate-600">MNIST PNG base</span>
          <input
            value={samplesBase}
            onChange={(e) => setSamplesBase(e.target.value)}
            className="px-3 py-2 rounded-lg border"
            placeholder="/mnist_png/all"
          />
        </label>
      </section>

      {error && (
        <div className="p-3 rounded-md bg-red-50 text-red-700 border border-red-200">
          {error}
        </div>
      )}

      {/* Live log */}
      <section>
        <h2 className="text-lg font-semibold mb-2">Live Log</h2>
        <div className="h-40 overflow-auto rounded-lg border bg-black text-green-300 p-2 text-sm font-mono whitespace-pre-wrap">
          {logs.length ? logs.join("\n") : "Idle."}
        </div>
      </section>

      {/* Progressive table */}
      <section>
        <h2 className="text-lg font-semibold mb-2">Per‑Model Summary</h2>
        <div className="overflow-x-auto">
          <table className="w-full text-sm border-separate border-spacing-y-1">
            <thead>
              <tr className="text-left">
                <th className="px-2">Model</th>
                <th className="px-2">WebGPU</th>
                <th className="px-2">Init</th>
                <th className="px-2">CPU Acc%</th>
                <th className="px-2">GPU Acc%</th>
                <th className="px-2">Agree</th>
                <th className="px-2">Avg Drift</th>
                <th className="px-2">Max Drift</th>
              </tr>
            </thead>
            <tbody>
              {perModel.map((m, i) => (
                <tr key={i} className="bg-white shadow-sm">
                  <td className="px-2 py-1 font-medium">{m.ModelFile}</td>
                  <td className="px-2 py-1">{m.WebGPUInitOK ? "✅" : "—"}</td>
                  <td className="px-2 py-1">
                    {m.WebGPUInitTimeMS?.toFixed?.(2)} ms
                  </td>
                  <td className="px-2 py-1">
                    {m.ADHD10?.Top1AccuracyCPU?.toFixed?.(2)}
                  </td>
                  <td className="px-2 py-1">
                    {m.ADHD10?.Top1AccuracyGPU?.toFixed?.(2)}
                  </td>
                  <td className="px-2 py-1">{m.ADHD10?.CPUvsGPUAgreeCount}</td>
                  <td className="px-2 py-1">
                    {m.ADHD10?.AvgDriftMAE?.toFixed?.(5)}
                  </td>
                  <td className="px-2 py-1">
                    {m.ADHD10?.MaxDriftMaxAbs?.toFixed?.(5)}
                  </td>
                </tr>
              ))}
              {!perModel.length && (
                <tr>
                  <td className="px-2 py-1 text-slate-500" colSpan={8}>
                    No results yet.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </section>

      {/* Detail accordion */}
      <section>
        <h2 className="text-lg font-semibold mb-2">Details</h2>
        <div className="space-y-2">
          {perModel.map((m, idx) => (
            <details key={idx} className="group border rounded-lg bg-white">
              <summary className="cursor-pointer select-none px-3 py-2 flex items-center justify-between">
                <span className="font-medium">{m.ModelFile}</span>
                <span className="text-xs text-slate-500">
                  {m.WebGPUInitOK
                    ? `WebGPU ${(m.WebGPUInitTimeMS || 0).toFixed(2)} ms`
                    : "GPU n/a"}
                </span>
              </summary>
              <div className="p-3 space-y-3">
                <DigitTable CPU={m.CPU} GPU={m.GPU} Drift={m.Drift} />
                <pre className="bg-slate-50 p-2 rounded-md text-xs overflow-auto">
                  {JSON.stringify(m.ADHD10, null, 2)}
                </pre>
              </div>
            </details>
          ))}
          {!perModel.length && (
            <div className="text-slate-500">Nothing to show yet.</div>
          )}
        </div>
      </section>
    </div>
  );
}

function DigitTable({
  CPU,
  GPU,
  Drift,
}: {
  CPU: SampleTiming[];
  GPU: SampleTiming[];
  Drift: DriftMetrics[];
}) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-xs">
        <thead>
          <tr className="text-left">
            <th className="px-2">Digit</th>
            <th className="px-2">CPU pred</th>
            <th className="px-2">CPU ms</th>
            <th className="px-2">GPU pred</th>
            <th className="px-2">GPU ms</th>
            <th className="px-2">drift_max</th>
            <th className="px-2">mae</th>
          </tr>
        </thead>
        <tbody>
          {CPU.map((c, i) => {
            const g = GPU[i];
            const d = Drift[i];
            return (
              <tr key={i} className="border-b border-slate-100">
                <td className="px-2 py-1 font-medium">{c.Digit}</td>
                <td className="px-2 py-1">{c.Pred}</td>
                <td className="px-2 py-1">{c.ElapsedMS.toFixed(2)}</td>
                <td className="px-2 py-1">{g?.Pred ?? "—"}</td>
                <td className="px-2 py-1">
                  {g ? g.ElapsedMS.toFixed(2) : "—"}
                </td>
                <td className="px-2 py-1">{d ? d.MaxAbs.toFixed(6) : "—"}</td>
                <td className="px-2 py-1">{d ? d.MAE.toFixed(6) : "—"}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

// Build a partial report from progressive state when user downloads before completion
function buildPartialReport(perModel: ModelRun[]): TelemetryReport {
  const base: TelemetryReport = {
    Version: "1.2.0-web",
    Source: "web",
    MachineID: "adhoc",
    System: {},
    FromHost: window.location.origin,
    ModelsUsed: perModel.map((m) => m.ModelFile),
    Samples: Array.from({ length: 10 }, (_, i) => i),
    StartedAt: new Date().toISOString(),
    EndedAt: new Date().toISOString(),
    PerModel: perModel,
  };
  return base;
}

// Very small, styleless HTML export so you can "Save as PDF" in browser
function renderHTMLReport(r: TelemetryReport): string {
  const rows = r.PerModel.map(
    (m) => `
    <tr>
      <td>${escapeHTML(m.ModelFile)}</td>
      <td>${m.WebGPUInitOK ? "yes" : "no"}</td>
      <td>${m.WebGPUInitTimeMS?.toFixed?.(2) ?? ""}</td>
      <td>${m.ADHD10?.Top1AccuracyCPU?.toFixed?.(2) ?? ""}</td>
      <td>${m.ADHD10?.Top1AccuracyGPU?.toFixed?.(2) ?? ""}</td>
      <td>${m.ADHD10?.CPUvsGPUAgreeCount ?? ""}</td>
      <td>${m.ADHD10?.AvgDriftMAE?.toFixed?.(5) ?? ""}</td>
      <td>${m.ADHD10?.MaxDriftMaxAbs?.toFixed?.(5) ?? ""}</td>
    </tr>`
  ).join("");

  return `<!doctype html><meta charset="utf-8"><title>Paragon Web Telemetry</title>
  <style>body{font-family:ui-sans-serif,system-ui,Segoe UI,Roboto,Helvetica,Arial;margin:24px} table{border-collapse:collapse;width:100%} th,td{border:1px solid #ddd;padding:6px;text-align:left} th{background:#f6f6f6}</style>
  <h1>Paragon Web Telemetry</h1>
  <p><b>Source:</b> ${r.Source} &nbsp; <b>MachineID:</b> ${escapeHTML(
    r.MachineID
  )} &nbsp; <b>FromHost:</b> ${escapeHTML(r.FromHost)}</p>
  <p><b>Started:</b> ${r.StartedAt} &nbsp; <b>Ended:</b> ${r.EndedAt}</p>
  <table><thead><tr>
    <th>Model</th><th>WebGPU</th><th>Init (ms)</th><th>CPU Acc%</th><th>GPU Acc%</th><th>Agree</th><th>Avg Drift</th><th>Max Drift</th>
  </tr></thead><tbody>${rows}</tbody></table>
  <h2>Raw JSON</h2>
  <pre>${escapeHTML(JSON.stringify(r, null, 2))}</pre>
  `;
}

function escapeHTML(s: string) {
  return s.replace(
    /[&<>"']/g,
    (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[
        c
      ] as string)
  );
}
