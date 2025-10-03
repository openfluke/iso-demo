import React from "react";
import { initPortal } from "@openfluke/portal";

// ----- Types you can refine once @openfluke/portal ships types -----
type Activation = "linear" | "relu" | "softmax";

interface PortalAPI {
  NewNetworkFloat32(
    layersJSON: string,
    activationsJSON: string,
    trainablesJSON: string
  ): NetworkAPI;
}

interface NetworkAPI {
  PerturbWeights(argsJSON: string): void;
  Forward(inputJSON: string): void;
  ExtractOutput(): string;
  Dispose?: () => void; // optional, in case your API supports cleanup
}

// ----- Props / State -----
interface PortalDemoProps {
  /** e.g. [{ Width: 1, Height: 1 }, { Width: 2, Height: 1 }, { Width: 3, Height: 1 }] */
  layers?: Array<{ Width: number; Height: number }>;
  /** e.g. ["linear","relu","softmax"] */
  activations?: Activation[];
  /** e.g. [true,true,true] */
  trainables?: boolean[];
  /** Run a random forward pass on mount */
  autoRun?: boolean;
}

interface PortalDemoState {
  output: string;
  loading: boolean;
  error?: string;
}

// ----- Class Component -----
export default class PortalDemo extends React.PureComponent<
  PortalDemoProps,
  PortalDemoState
> {
  static defaultProps: Partial<PortalDemoProps> = {
    layers: [
      { Width: 1, Height: 1 },
      { Width: 2, Height: 1 },
      { Width: 3, Height: 1 },
    ],
    activations: ["linear", "relu", "softmax"],
    trainables: [true, true, true],
    autoRun: true,
  };

  private portal: PortalAPI | null = null;
  private nn: NetworkAPI | null = null;
  private mounted = false;

  constructor(props: PortalDemoProps) {
    super(props);
    this.state = {
      output: "Loading…",
      loading: true,
    };
  }

  async componentDidMount() {
    this.mounted = true;
    try {
      await this.init();
      if (this.props.autoRun) {
        await this.buildAndRunOnce();
      } else {
        this.setState({ loading: false, output: "Ready." });
      }
    } catch (err: any) {
      this.safeSetState({
        loading: false,
        error: err?.message ?? String(err),
        output: "Initialization failed.",
      });
    }
  }

  componentWillUnmount() {
    this.mounted = false;
    try {
      this.nn?.Dispose?.();
    } catch {
      /* optional */
    }
    this.nn = null;
    this.portal = null;
  }

  // ----- Helpers -----
  private safeSetState(next: Partial<PortalDemoState>) {
    if (this.mounted) this.setState(next as PortalDemoState);
  }

  private async init() {
    this.portal = (await initPortal()) as unknown as PortalAPI;
  }

  private ensurePortal(): asserts this is { portal: PortalAPI } {
    if (!this.portal) {
      throw new Error("Portal API not initialized");
    }
  }

  private buildNetwork() {
    this.ensurePortal();
    const { layers, activations, trainables } = this.props;
    const nn = this.portal.NewNetworkFloat32(
      JSON.stringify(layers ?? []),
      JSON.stringify(activations ?? []),
      JSON.stringify(trainables ?? [])
    );
    this.nn = nn;
  }

  private forwardRandomOnce() {
    if (!this.nn) throw new Error("Network not built");
    // single 1×1 input with a random scalar (matches default first layer)
    const input = [[[Math.random()]]];
    this.nn.Forward(JSON.stringify(input));
    const out = this.nn.ExtractOutput();
    this.safeSetState({ output: out });
  }

  private perturb(seedScale = 0.1) {
    if (!this.nn) throw new Error("Network not built");
    // Example: [scale, seed]
    const args = [seedScale, Date.now() % 1000];
    this.nn.PerturbWeights(JSON.stringify(args));
  }

  private async buildAndRunOnce() {
    this.safeSetState({ loading: true, error: undefined });
    this.buildNetwork();
    this.perturb(0.1);
    this.forwardRandomOnce();
    this.safeSetState({ loading: false });
  }

  // ----- UI callbacks -----
  private handleRebuildAndRun = async () => {
    try {
      await this.buildAndRunOnce();
    } catch (err: any) {
      this.safeSetState({
        error: err?.message ?? String(err),
        loading: false,
      });
    }
  };

  private handlePerturbAndForward = () => {
    try {
      this.perturb(0.1);
      this.forwardRandomOnce();
    } catch (err: any) {
      this.safeSetState({
        error: err?.message ?? String(err),
      });
    }
  };

  private handleForwardOnly = () => {
    try {
      this.forwardRandomOnce();
    } catch (err: any) {
      this.safeSetState({
        error: err?.message ?? String(err),
      });
    }
  };

  // ----- Render -----
  render() {
    const { output, loading, error } = this.state;

    return (
      <div className="p-4">
        <h1 className="text-xl font-semibold mb-2">Portal Demo</h1>

        <div className="flex gap-2 mb-3">
          <button
            onClick={this.handleRebuildAndRun}
            disabled={loading}
            className="px-3 py-1 rounded border"
          >
            Rebuild & Run
          </button>
          <button
            onClick={this.handlePerturbAndForward}
            disabled={loading}
            className="px-3 py-1 rounded border"
          >
            Perturb & Forward
          </button>
          <button
            onClick={this.handleForwardOnly}
            disabled={loading}
            className="px-3 py-1 rounded border"
          >
            Forward (Random Input)
          </button>
        </div>

        {loading && <div>Loading…</div>}
        {error && (
          <div style={{ color: "crimson", marginBottom: 8 }}>
            Error: {error}
          </div>
        )}

        <pre
          style={{
            padding: 12,
            border: "1px solid #ddd",
            borderRadius: 8,
            overflowX: "auto",
          }}
        >
          {output}
        </pre>
      </div>
    );
  }
}
