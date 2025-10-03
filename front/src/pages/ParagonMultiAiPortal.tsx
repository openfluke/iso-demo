// src/pages/ParagonMultiAIPortal.tsx
import React from "react";
import { initPortal } from "@openfluke/portal";

type OneOutput = {
  pulse: number;
  output: number[];
  maxIndex: number;
  executionTime: string; // ms
  timestamp: number;
  inputData: number;
};

type ModelConfig = {
  id: string;
  name: string;
  layers: { Width: number; Height: number }[];
  activations: string[];
  color: string; // Bulma color class for the table headers
};

type ModelState = ModelConfig & {
  network: any;
  outputs: OneOutput[];
  status: "ready" | "error";
};

type LogEntry = { msg: string; timestamp: string; colorClass: string };

type State = {
  wasmReady: boolean;
  status: string;
  statusClass: string; // Bulma class
  aiModels: ModelState[];
  isRunning: boolean;
  pulseCount: number;
  currentData: number[] | null;
  logs: LogEntry[];
};

class ParagonMultiAIPortal extends React.Component<unknown, State> {
  private logContainerRef = React.createRef<HTMLDivElement>();
  private portalAPI: { NewNetworkFloat32: Function } | null = null;

  private readonly modelConfigs: ModelConfig[] = [
    {
      id: "model1",
      name: "Alpha Network",
      layers: [{ Width: 1, Height: 1 }, { Width: 2, Height: 1 }, { Width: 3, Height: 1 }],
      activations: ["linear", "relu", "softmax"],
      color: "has-text-primary",
    },
    {
      id: "model2",
      name: "Beta Network",
      layers: [{ Width: 1, Height: 1 }, { Width: 4, Height: 1 }, { Width: 3, Height: 1 }],
      activations: ["linear", "tanh", "softmax"],
      color: "has-text-success",
    },
    {
      id: "model3",
      name: "Gamma Network",
      layers: [{ Width: 1, Height: 1 }, { Width: 3, Height: 1 }, { Width: 3, Height: 1 }],
      activations: ["linear", "sigmoid", "softmax"],
      color: "has-text-warning",
    },
  ];

  constructor(props: unknown) {
    super(props);
    this.state = {
      wasmReady: false,
      status: "Loading Paragon (via portal)...",
      statusClass: "is-info",
      aiModels: [],
      isRunning: false,
      pulseCount: 0,
      currentData: null,
      logs: [],
    };
  }

  componentDidMount() {
    this.boot();
  }

  componentDidUpdate(_: unknown, prevState: State) {
    if (this.logContainerRef.current && this.state.logs !== prevState.logs) {
      this.logContainerRef.current.scrollTop = this.logContainerRef.current.scrollHeight;
    }
  }

  addLog = (msg: string, type: "info" | "success" | "warning" | "error" = "info") => {
    const timestamp = new Date().toLocaleTimeString();
    const colorClass =
      type === "error"
        ? "has-text-danger"
        : type === "success"
        ? "has-text-success"
        : type === "warning"
        ? "has-text-warning"
        : "has-text-grey";
    this.setState((prev) => ({ logs: [...prev.logs, { msg, timestamp, colorClass }] }));
    console.log(`[${timestamp}] ${msg}`);
  };

  boot = async () => {
    try {
      this.addLog("🚀 initPortal() starting...", "info");
      this.setState({ status: "Initializing portal...", statusClass: "is-info" });

      // this auto-loads wasm_exec + paragon.wasm packaged inside `portal`
      const api = await initPortal();
      this.portalAPI = api;

      this.setState({ wasmReady: true, status: "✅ Portal ready. Initializing models...", statusClass: "is-success" });
      this.addLog("✅ portal initialized; WASM is running.", "success");

      this.initializeModels();
    } catch (e: any) {
      const message = e?.message || String(e);
      this.addLog(`❌ portal init failed: ${message}`, "error");
      this.setState({ status: `❌ Portal init failed: ${message}`, statusClass: "is-danger" });
    }
  };

  initializeModels = () => {
    if (!this.portalAPI) return;
    const makeNN = (cfg: ModelConfig) => {
      const nn = this.portalAPI!.NewNetworkFloat32(
        JSON.stringify(cfg.layers),
        JSON.stringify(cfg.activations),
        JSON.stringify([true, true, true])
      );
      nn.Debug = false;
      nn.PerturbWeights(JSON.stringify([0.1, Date.now() % 1000]));
      return nn;
    };

    const models: ModelState[] = this.modelConfigs.map((cfg) => {
      try {
        const nn = makeNN(cfg);
        return { ...cfg, network: nn, outputs: [], status: "ready" as const };
      } catch (err: any) {
        this.addLog(`Failed to create ${cfg.name}: ${err?.message || String(err)}`, "error");
        return { ...cfg, // keep it visible but flagged
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          network: null as any,
          outputs: [],
          status: "error" as const,
        };
      }
    });

    const okCount = models.filter((m) => m.status === "ready").length;
    this.setState({ aiModels: models });
    this.addLog(`✅ Initialized ${okCount} AI models`, okCount === 3 ? "success" : "warning");
  };

  generateNewData = () => {
    const data = [Math.random()];
    this.addLog(`📊 New input: [${data[0].toFixed(6)}]`, "info");
    return data;
  };

  runModelOnce = async (model: ModelState, inputData: number[]) => {
    if (model.status !== "ready" || !model.network) return null;

    try {
      const input = JSON.stringify([[inputData]]);
      const t0 = performance.now();
      model.network.Forward(input);
      const raw = model.network.ExtractOutput();
      const t1 = performance.now();

      const parsed = JSON.parse(raw);
      const flat: number[] = Array.isArray(parsed[0]) ? parsed[0] : parsed;
      const maxIndex = flat.indexOf(Math.max(...flat));

      return {
        output: flat,
        maxIndex,
        executionTime: (t1 - t0).toFixed(2),
        timestamp: Date.now(),
      };
    } catch (e: any) {
      this.addLog(`❌ ${model.name} error: ${e?.message || String(e)}`, "error");
      return null;
    }
  };

  runSixPulses = async () => {
    const { wasmReady, aiModels } = this.state;
    if (!wasmReady || aiModels.length === 0) return;

    this.setState({ isRunning: true });

    let data = this.state.currentData ?? this.generateNewData();
    this.setState({ currentData: data });

    for (let pulse = 1; pulse <= 6; pulse++) {
      this.setState({ pulseCount: pulse });

      // change data on pulses 1,3,5,6 (like your original)
      if ([1, 3, 5, 6].includes(pulse)) {
        data = this.generateNewData();
        this.setState({ currentData: data });
        this.addLog(`🔄 Pulse ${pulse}: new data`, "warning");
      } else {
        this.addLog(`🔁 Pulse ${pulse}: reuse previous data`, "info");
      }

      this.addLog(`⚡ Pulse ${pulse}/6 for ${aiModels.length} models...`, "info");
      const results = await Promise.all(aiModels.map((m) => this.runModelOnce(m, data)));

      // update outputs
      this.setState((prev) => ({
        aiModels: prev.aiModels.map((m, i) => {
          const r = results[i];
          if (!r) return m;
          const newOut: OneOutput = {
            pulse,
            ...r,
            inputData: data[0],
          };
          return { ...m, outputs: [...m.outputs, newOut].slice(-10) };
        }),
      }));

      if (pulse < 6) {
        await new Promise((r) => setTimeout(r, 500));
      }
    }

    this.addLog("🎉 All pulses completed!", "success");
    this.setState({ status: "✅ Cycle complete: 6 pulses", statusClass: "is-success", isRunning: false });
  };

  resetModels = () => {
    this.setState((prev) => ({
      aiModels: prev.aiModels.map((m) => ({ ...m, outputs: [] })),
      pulseCount: 0,
      currentData: null,
      logs: [],
    }));
    this.addLog("🔄 Models reset", "info");
  };

  render() {
    const { wasmReady, status, statusClass, aiModels, isRunning, pulseCount, currentData, logs } = this.state;

    // Build table rows (aligned by pulse index)
    const rows: any[] = [];
    if (aiModels.length > 0 && aiModels.every((m) => m.outputs.length > 0)) {
      const numPulses = Math.max(...aiModels.map((m) => m.outputs.length));
      for (let i = 0; i < numPulses; i++) {
        const pulse = aiModels[0].outputs[i]?.pulse ?? i + 1;
        const input = aiModels[0].outputs[i]?.inputData ?? currentData?.[0] ?? 0;
        const row: any = { pulse, input: input.toFixed(4) };
        aiModels.forEach((m) => {
          const out = m.outputs[i];
          if (out) {
            row[`${m.name}_output`] = `[${out.output.map((v) => v.toFixed(3)).join(", ")}]`;
            row[`${m.name}_class`] = out.maxIndex;
            row[`${m.name}_time`] = out.executionTime;
          } else {
            row[`${m.name}_output`] = "-";
            row[`${m.name}_class`] = "-";
            row[`${m.name}_time`] = "-";
          }
        });
        rows.push(row);
      }
    }

    return (
      <div className="box full-page-box">
        <section className="section">
          <div className="container">
            <div className="box">
              <h1 className="title is-2">Paragon Multi-AI (via portal)</h1>
              <p className="subtitle is-5">3 networks • 6 pulses • inputs change on 1, 3, 5, 6</p>

              <div className="field is-grouped mb-4">
                <p className="control">
                  <button
                    onClick={this.runSixPulses}
                    disabled={!wasmReady || isRunning}
                    className={`button is-primary ${!wasmReady || isRunning ? "is-static" : ""}`}
                  >
                    {isRunning ? `Running Pulse ${pulseCount}/6...` : "Run 6 Pulses"}
                  </button>
                </p>
                <p className="control">
                  <button onClick={this.resetModels} disabled={isRunning} className={`button is-dark ${isRunning ? "is-static" : ""}`}>
                    Reset
                  </button>
                </p>
              </div>

              <div className={`notification ${statusClass}`}>{status}</div>
            </div>

            <div className="box">
              <h3 className="title is-4">Pulse Outputs</h3>
              {rows.length === 0 ? (
                <p className="has-text-grey-light">No outputs yet...</p>
              ) : (
                <div className="table-container">
                  <table className="table is-bordered is-striped is-narrow is-hoverable is-fullwidth">
                    <thead>
                      <tr>
                        <th>Pulse</th>
                        <th>Input</th>
                        {aiModels.map((m) => (
                          <React.Fragment key={m.id}>
                            <th className={m.color}>{m.name} Output</th>
                            <th className={m.color}>{m.name} Class</th>
                            <th className={m.color}>{m.name} Time (ms)</th>
                          </React.Fragment>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {rows.map((r, idx) => (
                        <tr key={idx}>
                          <td>{r.pulse}</td>
                          <td>{r.input}</td>
                          {aiModels.map((m) => (
                            <React.Fragment key={m.id}>
                              <td className="is-family-monospace">{r[`${m.name}_output`]}</td>
                              <td>{r[`${m.name}_class`]}</td>
                              <td>{r[`${m.name}_time`]}</td>
                            </React.Fragment>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>

            <div className="box">
              <h3 className="title is-4">Console Logs</h3>
              <div
                ref={this.logContainerRef}
                className="content has-background-dark p-3 has-text-light is-family-monospace is-size-7"
                style={{ maxHeight: 240, overflowY: "auto" }}
              >
                {logs.length === 0 ? (
                  <p className="has-text-grey">Waiting for logs...</p>
                ) : (
                  logs.map((log, idx) => (
                    <div key={idx} className="mb-1">
                      <span className="has-text-grey">[{log.timestamp}]</span>{" "}
                      <span className={log.colorClass}>{log.msg}</span>
                    </div>
                  ))
                )}
              </div>
            </div>

            {currentData && (
              <div className="notification is-info mt-4">
                <h4 className="title is-5">Current Input Data:</h4>
                <code>[{currentData[0]?.toFixed(6)}]</code>
              </div>
            )}
          </div>
        </section>
      </div>
    );
  }
}

export default ParagonMultiAIPortal;
