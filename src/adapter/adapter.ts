import type {
    BreakpointResolvedEvent,
    CallFrame,
    CallFrameId,
    PausedEvent,
    RemoteObject,
    ScriptParsedEvent,
    SetBreakpointByUrlParams,
} from '../inspector/jsc-types';
import { formatRemoteObject } from '../inspector/preview';
import { WebSocketInspector } from '../inspector/websocket-inspector';
import { type AnySourceMap, NoopSourceMap, SourceMap } from '../sourcemap';
import { isLoopbackHost } from '../util/loopback';
import { getAvailablePort } from '../util/port';
import { Handles } from '@vscode/debugadapter';
import type { DebugProtocol } from '@vscode/debugprotocol';
import { type ChildProcess, spawn } from 'node:child_process';
import { existsSync, realpathSync } from 'node:fs';
import { basename, isAbsolute, resolve } from 'node:path';

/**
 * Collections with at least this many properties advertise
 * `indexedVariables` so a compliant DAP client pages with start/count. Below
 * it the field is omitted to keep responses clean. The start/count slicing honors
 * any start/count a client sends regardless, but without advertised counts a
 * well-behaved client never sends them — so this threshold makes the paging
 * reachable rather than dead code.
 */
const PAGING_THRESHOLD = 100;

// ---------------------------------------------------------------------------
// Host interface — the session supplies these by constructing DAP events.
// ---------------------------------------------------------------------------

export interface AdapterHost {
    sendInitialized(): void;
    sendOutput(category: 'stdout' | 'stderr' | 'console', text: string): void;
    sendStopped(reason: string, threadId: number, exceptionText?: string, hitBreakpointIds?: number[]): void;
    sendBreakpointChanged(bp: DebugProtocol.Breakpoint): void;
    sendTerminated(): void;
    sendExited(code: number): void;
    sendThread(reason: 'started' | 'exited', id: number): void;
    sendProcessStarted(pid: number): void;
}

// ---------------------------------------------------------------------------
// Launch/attach argument shapes
// ---------------------------------------------------------------------------

export interface LaunchArgs {
    program: string;
    args?: string[];
    cwd?: string;
    env?: Record<string, string>;
    strictEnv?: boolean;
    runtime?: string;
    runtimeArgs?: string[];
    stopOnEntry?: boolean;
    noDebug?: boolean;
}

export interface AttachArgs {
    url: string;
}

// ---------------------------------------------------------------------------
// State store entry shapes
// ---------------------------------------------------------------------------

interface SourceEntry {
    scriptId: string;
    /** The path from Debugger.scriptParsed.url (a TS absolute path for file scripts). */
    path: string;
    sourceMap: AnySourceMap | null;
}

interface LineBreakpoint {
    id: number; // DAP breakpoint id
    path: string;
    line: number; // 1-based TS line
    condition?: string;
    hitCondition?: string;
    logMessage?: string;
    jscBreakpointId?: string;
    verified: boolean;
    hitCount: number;
    message?: string;
}

interface FunctionBreakpoint {
    id: number;
    symbol: string;
    condition?: string;
    hitCondition?: string;
    jscBreakpointId: string;
    verified: boolean;
    hitCount: number;
}

/**
 * A line-0 placeholder breakpoint for one not-yet-parsed source (upstream bun
 * design, `#setBreakpointsByUrl`): JSC resolves it to that source's first
 * pausable statement and pauses execution there — a module-scoped engine brake
 * that holds the module body until the mapped arms bind. Kept by jsc id until
 * the source parses; ids linger in placeholderJscIds after teardown so
 * in-flight placeholder pauses still get swallowed.
 */
interface PlaceholderRecord {
    /** JSC id of the armed placeholder breakpoint. */
    jscId: string;
    /** Canonical (realpath'd) source path the placeholder was armed against. */
    path: string;
    /** DAP bp keys (lineBreakpoints) waiting on this source, in request order. */
    bpKeys: string[];
}

const URL_RE = /ws:\/\/[\w.-]+:\d+\/[^\s]+/;

/**
 * BunDebugAdapter — a lean DAP⇄JSC engine. Owns the inspector connection, the
 * breakpoint/state stores, and the translation between DAP requests and the
 * WebKit inspector protocol. Emissions go out through the `host` callbacks;
 * the session (src/session.ts) turns them into DAP events.
 */
export class BunDebugAdapter {
    host: AdapterHost | null = null;

    private inspector: WebSocketInspector | null = null;
    private launchedChild: ChildProcess | null = null;
    private mode: 'launch' | 'attach' | null = null;
    private noDebug = false;
    private stopOnEntry = false;
    private configDoneSent = false;
    private initializedSent = false;
    private firstPausePending = false;
    /**
     * True between a next/stepIn/stepOut request and the pause that completes
     * it. JSC reports step completion as reason PauseOnNextStatement (or
     * 'other' when stepping from a bp pause) — indistinguishable from a
     * Debugger.pause brake on the wire — so mapStopReason consults this flag
     * to report 'step' (upstream tracks the pending step the same way).
     * Cleared by every resume path (client continue, brake/placeholder/
     * logpoint/condition swallows): a bp-family hit mid-step cancels JSC's
     * step mode, so no completion pause follows and the flag must not strand.
     * A bp hit reported to the client (reason 'breakpoint') leaves the flag
     * for the subsequent continue, which clears it in resume().
     */
    private stepPending = false;
    /** Dedupes exited/terminated emission across child-exit and inspector-disconnect. */
    private debuggeeGoneSent = false;
    /** Exit code from the child (launch); attach reports 0. */
    private exitCode: number | null = null;
    private paused = false;
    private attachHost: string | null = null;
    private entryBreakpointUrl: string | null = null;
    private entryBreakpointId: string | undefined = undefined;

    // Sources: TS path -> entry; plus scriptId -> entry for frames.
    private sources = new Map<string, SourceEntry>();
    private sourcesByScriptId = new Map<string, SourceEntry>();

    private lineBreakpoints = new Map<string, LineBreakpoint>();
    private functionBreakpoints = new Map<string, FunctionBreakpoint>();
    /** jscBreakpointId -> dap bp key (line and function). */
    private jscToDapKey = new Map<string, string>();
    /**
     * Deferred line breakpoints per canonical (realpath'd) source path. A bp
     * lands here when setBreakpoints arrives before its script's
     * Debugger.scriptParsed — the normal launch flow, since --inspect-wait holds
     * parsing until configurationDone. The bp arms at its source-mapped
     * generated coordinates the moment the script (and its synchronously
     * decoded map) shows up in onScriptParsed.
     */
    private pendingByPath = new Map<string, string[]>();
    /**
     * Per-source line-0 placeholder breakpoints for deferred sources — the
     * upstream bun design (`#setBreakpointsByUrl`): JSC resolves a placeholder
     * to its source's first pausable statement and pauses execution THERE,
     * giving the module's own deferred arms time to bind. The global brake
     * covers import-free entries (first executed statement == entry statement);
     * the placeholder covers entries with imports, where the first executed
     * statement lives in a builtin initializer and the brake is consumed too
     * early. Keyed by jsc breakpoint id.
     */
    private placeholdersById = new Map<string, PlaceholderRecord>();
    /** Every placeholder id ever armed; permanent, so late pauses are swallowed. */
    private placeholderJscIds = new Set<string>();
    /**
     * Placeholder id -> DAP bp key when the placeholder's resolved location IS
     * that bp's mapped target (the pause stands in for the hit). One-shot.
     */
    private placeholderAlias = new Map<string, string>();
    /**
     * True between configurationDone arming the one-shot entry brake (a
     * pre-gate `Debugger.pause`, legal and deterministic per probe) and its
     * pause firing at the module body's first executed statement. Without it,
     * deferred bps whose target executes within microseconds of parse end would
     * bind too late. See onPaused for the swallow/overlap handling.
     */
    private brakePending = false;
    private nextDapBpId = 1;
    private nextBpKey = 1;

    private currentCallFrames: CallFrame[] = [];
    private currentFrameIds: number[] = [];
    private frameIdToCallFrame = new Map<number, CallFrame>();
    private nextFrameId = 1;
    /**
     * Every variablesReference issued to the client, from ONE counter shared
     * by scopes and nested objects (upstream uses a single counter too).
     * Separate per-kind counters starting at the same base collide: a nested
     * object's ref numerically equal to a scope ref resolves to the scope's
     * objectId and the client sees the wrong variables (nvim log: expanding
     * `tc` showed the local scope's vars). Reset at the START of each pause —
     * NOT on Debugger.resumed: JSC does not reliably send resumed after
     * stepNext (probe-verified), so objectIds from earlier pauses stayed
     * reachable and later fetches errored (-32000 Internal error) or returned
     * stale data.
     */
    private objectRefs = new Handles<{ objectId: string; callFrameId?: CallFrameId } | undefined>();

    // Buffered stopped emissions (waiting for `initialized` to be sent).
    private stoppedBuffer: (() => void)[] = [];

    // -------------------------------------------------------------------------
    // Launch / attach
    // -------------------------------------------------------------------------

    async launch(args: LaunchArgs): Promise<void> {
        this.mode = 'launch';
        this.noDebug = !!args.noDebug;
        this.stopOnEntry = !!args.stopOnEntry;
        this.debuggeeGoneSent = false;
        this.exitCode = null;
        const cwd = args.cwd ?? process.cwd();
        const program = resolve(cwd, args.program);
        if (!existsSync(program)) {
            throw new Error(`launch: program does not exist: ${program}`);
        }

        const runtime = args.runtime ?? 'bun';
        const runtimeArgs = args.runtimeArgs ?? [];
        const env = args.strictEnv ? (args.env ?? {}) : { ...process.env, ...(args.env ?? {}) };
        if (this.noDebug) {
            this.launchedChild = spawn(runtime, [...runtimeArgs, program, ...(args.args ?? [])], {
                cwd,
                env,
                stdio: ['ignore', 'pipe', 'pipe'],
            });
            this.forwardOutput(this.launchedChild);
            this.launchedChild.on('exit', (code) => {
                this.onChildExit(code);
            });
            return;
        }
        const port = await getAvailablePort();
        // Always --inspect-wait: --inspect-brk adds no pause event on top of the
        // gate (verified by probe), and stopOnEntry is implemented as a deferred
        // line-0 breakpoint armed at configurationDone.
        const inspectFlag = `--inspect-wait=127.0.0.1:${port}`;
        // Canonicalize like setBreakpoints does: Bun reports script URLs with
        // symlinks resolved, so the entry bp URL must match (and the brake's
        // inEntryFile guard compares against it).
        this.entryBreakpointUrl = this.stopOnEntry ? (existsSync(program) ? realpathSync(program) : program) : null;
        const child = spawn(runtime, [...runtimeArgs, inspectFlag, program, ...(args.args ?? [])], {
            cwd,
            env,
            stdio: ['ignore', 'pipe', 'pipe'],
        });
        this.launchedChild = child;

        try {
            // Discover the inspector URL on stderr: scrapeInspectorUrl forwards
            // pre-URL bytes as output, drops the banner line carrying the URL,
            // and resolves with the URL.
            // stdio pipes stderr, so Bun guarantees the stream exists.
            const inspectorUrl = await this.scrapeInspectorUrl(child.stderr);
            // forwardOutput re-attaches stderr so post-URL bytes (and any
            // remainder buffered during the scrape) are forwarded from here on.
            this.forwardOutput(child);
            this.inspector = new WebSocketInspector(inspectorUrl);
            this.inspector.on('error', (err: Error) => {
                this.diagnostic(`inspector: ${err.message}`);
            });
            await this.awaitConnected(this.inspector);
            await this.attachInspector();
        } catch (err) {
            // The debuggee was spawned but launch failed (no inspector URL, connect
            // refused, attach error): kill the orphaned child so it does not outlive
            // the failed launch and leak. Clear launchedChild before kill so the
            // exit handler does not fire a spurious onChildExit.
            this.launchedChild = null;
            try {
                child.kill('SIGKILL');
            } catch {
                /* already gone */
            }
            throw err;
        }
        child.on('exit', (code) => {
            this.onChildExit(code);
        });
    }

    async attach(args: AttachArgs): Promise<void> {
        this.mode = 'attach';
        this.debuggeeGoneSent = false;
        this.exitCode = null;
        let url = args.url.trim();
        if (!/^ws:\/\/.+/.test(url)) throw new Error(`attach: not a ws:// url: ${url}`);
        if (!/^ws:\/\/[^/]+\/.+/.test(url)) url += '/';
        try {
            this.attachHost = new URL(url).hostname;
        } catch {
            throw new Error(`attach: invalid inspector url: ${url}`);
        }
        const maxAttempts = 3;
        for (let attempt = 1; ; attempt++) {
            this.inspector = new WebSocketInspector(url);
            this.inspector.on('error', (err: Error) => {
                this.diagnostic(`inspector: ${err.message}`);
            });
            try {
                await this.awaitConnected(this.inspector);
                break;
            } catch {
                if (attempt >= maxAttempts)
                    throw new Error(`attach: could not connect to ${url} after ${maxAttempts} attempts`);
                await new Promise((r) => setTimeout(r, 100));
            }
        }
        this.noDebug = false;
        await this.attachInspector();
    }

    /**
     * Resolve once the inspector socket is open, or reject on the first connect
     * failure. A plain `once('connected')` hangs forever when the WebSocket
     * errors before opening ('error'/'disconnected' fire, 'connected' never
     * does); race all three with listener cleanup so both launch and attach
     * either connect or fail loudly.
     */
    private awaitConnected(inspector: WebSocketInspector): Promise<void> {
        return new Promise((resolve, reject) => {
            if (inspector.isConnected) {
                resolve();
                return;
            }
            const onConnected = (): void => {
                cleanup();
                resolve();
            };
            const onError = (err: Error): void => {
                cleanup();
                reject(err);
            };
            const onClosed = (): void => {
                cleanup();
                reject(new Error('inspector connection closed before open'));
            };
            const cleanup = (): void => {
                inspector.off('connected', onConnected);
                inspector.off('error', onError);
                inspector.off('disconnected', onClosed);
            };
            inspector.on('connected', onConnected);
            inspector.on('error', onError);
            inspector.on('disconnected', onClosed);
        });
    }

    /** Shared JSC enable sequence + event wiring. */
    private async attachInspector(): Promise<void> {
        const inspector = this.requireInspector();
        await inspector.send('Runtime.enable');
        await inspector.send('Console.enable');
        await inspector.send('Debugger.enable');
        await inspector.send('Debugger.setAsyncStackTraceDepth', { depth: 0 });

        inspector.on('Debugger.scriptParsed', (params) => {
            this.onScriptParsed(params as ScriptParsedEvent);
        });
        inspector.on(
            'Debugger.paused',
            (params) =>
                void this.onPaused(params as PausedEvent).catch((err: unknown) => {
                    this.diagnostic(`onPaused: ${(err as Error).message}`);
                })
        );
        inspector.on('Debugger.breakpointResolved', (params) => {
            this.onBreakpointResolved(params as BreakpointResolvedEvent);
        });
        inspector.on('Debugger.resumed', () => {
            this.onResumed();
        });
        // The debuggee exiting closes the inspector socket. In attach mode that
        // is the ONLY exit signal (no child handle) — without this the client
        // never sees exited/terminated and the session stays open after the
        // program ran to completion (nvim log: final continue, then nothing).
        // In launch mode onChildExit usually fires first; the flag dedupes.
        inspector.on('disconnected', () => {
            this.onDebuggeeGone();
        });

        if (this.mode === 'launch' && this.stopOnEntry) this.firstPausePending = true;

        this.initializedSent = true;
        this.host?.sendInitialized();
        const buffered = this.stoppedBuffer;
        this.stoppedBuffer = [];
        for (const emit of buffered) emit();
    }

    /** Sentinel for launch gates. Returns the live inspector or throws. */
    private requireInspector(): WebSocketInspector {
        if (!this.inspector) throw new Error('no inspector connected');
        return this.inspector;
    }

    private forwardOutput(child: ChildProcess): void {
        child.stdout?.on('data', (chunk: Buffer) => {
            this.emitOutput('stdout', chunk.toString('utf8'));
        });
        child.stderr?.on('data', (chunk: Buffer) => {
            this.emitOutput('stderr', chunk.toString('utf8'));
        });
    }

    /**
     * Scrape the Bun inspector URL off a child's stderr stream. Pre-URL bytes
     * are forwarded as `stderr` output; the banner line carrying the URL is
     * dropped; the URL is resolved. Post-URL bytes buffered during the scrape
     * (the remainder after the URL on the same chunk) are forwarded too, so
     * nothing is lost — `forwardOutput` re-attaches to carry subsequent chunks.
     * Tracks emitted length so pre-URL text split across chunks is forwarded
     * exactly once (not re-emitted from the accumulated buffer when the URL
     * arrives).
     */
    private scrapeInspectorUrl(stderr: NodeJS.ReadableStream): Promise<string> {
        return new Promise<string>((resolve, reject) => {
            const chunks: Buffer[] = [];
            let emitted = 0; // bytes already forwarded as output
            const deadline = setTimeout(() => {
                reject(
                    new Error(
                        `could not find Bun inspector URL on stderr; got: ${Buffer.concat(chunks).toString('utf8').slice(-2048)}`
                    )
                );
            }, 10_000);
            const onExit = () => {
                clearTimeout(deadline);
                reject(
                    new Error(
                        `debuggee exited before inspector connected; stderr: ${Buffer.concat(chunks).toString('utf8').slice(-2048)}`
                    )
                );
            };
            stderr.once('close', onExit);
            const onData = (chunk: Buffer): void => {
                chunks.push(chunk);
                const text = Buffer.concat(chunks).toString('utf8');
                const m = URL_RE.exec(text);
                if (m) {
                    // Forward only the NOT-yet-emitted prefix (before the URL).
                    const prefix = text.slice(emitted, m.index);
                    if (prefix) this.emitOutput('stderr', prefix);
                    emitted = m.index + m[0].length;
                    // Forward the post-URL remainder on this chunk; forwardOutput
                    // takes over for later chunks.
                    const remainder = text.slice(emitted);
                    if (remainder) this.emitOutput('stderr', remainder);
                    clearTimeout(deadline);
                    stderr.removeListener('close', onExit);
                    stderr.removeListener('data', onData);
                    resolve(m[0]);
                    return;
                }
                // URL not found yet: forward only the newly-arrived bytes so the
                // prefix isn't re-emitted when the URL-bearing chunk lands.
                if (chunk.length > 0) {
                    this.emitOutput('stderr', chunk.toString('utf8'));
                    emitted += chunk.length;
                }
                // Bounded buffer: keep the last 1024 bytes for the error message.
                if (Buffer.concat(chunks).length > 4096) {
                    const tail = Buffer.concat(chunks).subarray(-1024);
                    chunks.length = 0;
                    chunks.push(tail);
                }
            };
            stderr.on('data', onData);
        });
    }
    /** Debugger control — valid while paused (resume/step) or running (pause). */
    async resume(): Promise<void> {
        // Client continue abandons any in-flight step (bp hits mid-step cancel
        // JSC's step mode — probe-verified), so the flag must not strand and
        // mislabel a later manual pause as 'step'. resumeInternal clears it for
        // the swallow paths for the same reason.
        this.stepPending = false;
        this.releaseObjectGroups();
        await this.inspector?.send('Debugger.resume');
        // JSC may omit Debugger.resumed (see clearPauseState); clear eagerly.
        this.clearPauseState();
    }

    async next(): Promise<void> {
        // Mark synchronously (upstream sets #stopped = "step" the same way): JSC
        // reports the step-completion pause as PauseOnNextStatement — the same
        // reason as a Debugger.pause brake — so mapStopReason needs this flag to
        // tell them apart. nvim-dap only re-renders the cursor on reason 'step'.
        this.stepPending = true;
        this.releaseObjectGroups();
        try {
            await this.inspector?.send('Debugger.stepNext');
        } catch (err) {
            // No completion pause is coming — clear the flag so a later manual
            // pause isn't mislabeled 'step'.
            this.stepPending = false;
            throw err;
        }
        // JSC does not reliably emit Debugger.resumed after stepNext
        // (probe-verified); clear pause state eagerly so a stackTrace before
        // the completion pause does not return stale frames.
        this.clearPauseState();
    }

    async stepIn(): Promise<void> {
        this.stepPending = true;
        this.releaseObjectGroups();
        try {
            await this.inspector?.send('Debugger.stepInto');
        } catch (err) {
            this.stepPending = false;
            throw err;
        }
        this.clearPauseState();
    }

    async stepOut(): Promise<void> {
        this.stepPending = true;
        this.releaseObjectGroups();
        try {
            await this.inspector?.send('Debugger.stepOut');
        } catch (err) {
            this.stepPending = false;
            throw err;
        }
        this.clearPauseState();
    }

    async pause(): Promise<void> {
        await this.inspector?.send('Debugger.pause');
    }

    private emitOutput(category: 'stdout' | 'stderr' | 'console', text: string): void {
        if (!text) return;
        this.host?.sendOutput(category, text);
    }

    private onChildExit(code: number | null): void {
        this.exitCode = code ?? 0;
        this.inspector?.close();
        this.inspector = null;
        this.onDebuggeeGone();
    }

    /** Emit exited+terminated exactly once, however the debuggee went away. */
    private onDebuggeeGone(): void {
        if (this.debuggeeGoneSent) return;
        this.debuggeeGoneSent = true;
        this.inspector = null;
        this.host?.sendExited(this.exitCode ?? 0);
        this.host?.sendTerminated();
    }

    private diagnostic(message: string): void {
        try {
            process.stderr.write(`[bun-dap] ${message}\n`);
        } catch {
            // stderr is best-effort
        }
    }

    // ------------------------------------------------------------------
    // configurationDone — releases the --inspect-wait / --inspect-brk gate
    // ------------------------------------------------------------------

    async configurationDone(): Promise<void> {
        if (this.configDoneSent) return;
        const inspector = this.inspector;
        if (inspector) {
            // stopOnEntry: Bun's --inspect-brk never delivers an entry pause over
            // the protocol (verified: resume pre-initialized errors "Must be paused
            // or waiting to pause", and no paused event arrives after the gate
            // opens). Arm a deferred line-0 breakpoint on the entry file instead —
            // JSC resolves it to the first executable statement once the module
            // parses (the same mechanism the original placeholder design used).
            if (this.stopOnEntry && this.entryBreakpointUrl) {
                // A deferred-bp placeholder for the entry file is already armed at
                // its line 0 (Path B ran before configurationDone — the common
                // stopOnEntry+breakpoint flow): JSC rejects a second bp at the same
                // location ("Breakpoint for given location already exists"), so the
                // placeholder doubles as the entry bp. Its pause still owes the
                // 'entry' stop — onPaused's placeholder branch falls through for it.
                let existing: string | undefined;
                for (const [pid, record] of this.placeholdersById) {
                    if (record.path === this.entryBreakpointUrl) {
                        existing = pid;
                        break;
                    }
                }
                if (existing !== undefined) {
                    this.entryBreakpointId = existing;
                } else {
                    try {
                        const res = await inspector.send('Debugger.setBreakpointByUrl', {
                            url: this.entryBreakpointUrl,
                            lineNumber: 0,
                        });
                        this.entryBreakpointId = res.breakpointId;
                    } catch (err) {
                        this.diagnostic(`entry breakpoint failed: ${(err as Error).message}`);
                    }
                }
            }
            // Deferred-breakpoint brake: one pre-gate Debugger.pause so the VM
            // freezes at the module body's FIRST executed statement (probe-verified:
            // legal while gated; arrives as reason PauseOnNextStatement with no
            // data). Deferred bps arm at their mapped coordinates while frozen
            // (see onScriptParsed); without the brake a bp whose target runs within
            // microseconds of parse end would bind too late — the attach-mode
            // failure the nvim log showed (bp verified:true, no stop, program ran
            // to completion). The pause is consumed — swallowed or surfaced as a
            // bp's stop — in onPaused. Skipped when nothing is deferred. Applies to
            // attach too: pendingByPath non-empty at configurationDone implies the
            // --inspect-wait gate still holds, so the same pre-gate pause is safe.
            if (this.pendingByPath.size > 0) {
                // Set brakePending BEFORE the await: the brake's pause can arrive
                // the instant Debugger.pause resolves, and onPaused reads this flag
                // to decide swallow-vs-stop. Setting it after the await would miss a
                // pause that lands between resolve and the assignment. Clear it only
                // if the pause call itself rejects (then no brake pause will come).
                this.brakePending = true;
                try {
                    await inspector.send('Debugger.pause');
                } catch (err) {
                    this.brakePending = false;
                    this.diagnostic(`entry brake failed: ${(err as Error).message}`);
                }
            }
            await inspector.send('Debugger.setBreakpointsActive', { active: !this.noDebug });
            await inspector.send('Inspector.initialized');
        }
        this.configDoneSent = true;
        this.host?.sendThread('started', 1);
        // A `process` event is only meaningful for launch, where the child's pid
        // is known; in attach mode there is no launched child, so reporting it
        // would fabricate pid 0.
        if (this.launchedChild?.pid !== undefined) this.host?.sendProcessStarted(this.launchedChild.pid);
    }

    // ------------------------------------------------------------------
    // Script parsing + source map loading
    // ------------------------------------------------------------------

    /**
     * Registers the script and decodes its source map SYNCHRONOUSLY from
     * `scriptParsed.sourceMapURL` (a base64 data URI in Bun), mirroring bun's own
     * debug adapter (`SourceMap(sourceMapURL)`; no `Debugger.getScriptSource` RPC).
     * Synchronous decode is load-bearing for deferred breakpoints: the pending
     * arms below must issue their sends the moment the map exists — for the
     * launch entry, while the VM is still frozen by configurationDone's brake.
     */
    private onScriptParsed(params: ScriptParsedEvent): void {
        const { scriptId, url } = params;
        let entry = this.sources.get(url);
        if (!entry) {
            entry = { scriptId, path: url, sourceMap: null };
            this.sources.set(url, entry);
        }
        this.sourcesByScriptId.set(scriptId, entry);
        let map: AnySourceMap | null = null;
        if (params.sourceMapURL) {
            try {
                map = SourceMap.fromSourceMapURL(params.sourceMapURL, url);
            } catch (err) {
                this.diagnostic(`source map decode failed for ${url}: ${(err as Error).message}`);
            }
        }
        entry.sourceMap = map ?? new NoopSourceMap();

        // Deferred breakpoints for this source: arm them now, at map-derived
        // generated coordinates. When this script is the launch entry, the VM is
        // guaranteed frozen at (or before) its first statement by the brake armed
        // in configurationDone, and this scriptParsed event was delivered to us
        // before that pause — so these binds always land before the body can run
        // past them. (For a mid-session source the body is long-running or idle;
        // binds apply to future executions, as with any post-parse arm.)
        const pending = this.pendingByPath.get(url);
        if (pending) {
            this.pendingByPath.delete(url);
            for (const key of pending) {
                const bp = this.lineBreakpoints.get(key);
                if (bp) void this.armLineBreakpoint(url, key, bp, entry.sourceMap);
            }
        }
    }

    private onBreakpointResolved(params: BreakpointResolvedEvent): void {
        // Placeholder resolved: the source is loaded. If a pending bp's mapped
        // location IS the placeholder's resolved statement, the placeholder's
        // upcoming pause stands in for that bp's hit (a bp bound at the pause
        // location cannot re-fire once resumed); record the alias and keep the
        // placeholder armed — its pause IS the hit (it shares coordinates with
        // the real bp, so removing it would disarm both). Non-aliased
        // placeholders are removed below so later passes hit normally.
        const record = this.placeholdersById.get(params.breakpointId);
        if (record) {
            this.placeholdersById.delete(record.jscId);
            const entry = this.sources.get(record.path);
            const map = entry?.sourceMap ?? null;
            let aliased = false;
            if (map) {
                for (const key of record.bpKeys) {
                    const bp = this.lineBreakpoints.get(key);
                    if (!bp) continue;
                    const gen = map.generatedLocation(bp.line, 0);
                    if (gen === null || gen.line - 1 !== params.location.lineNumber) continue;
                    if (map instanceof SourceMap && gen.column - 1 !== params.location.columnNumber) continue;
                    this.placeholderAlias.set(record.jscId, key);
                    if (!bp.verified) {
                        bp.verified = true;
                        this.host?.sendBreakpointChanged({ id: bp.id, line: bp.line, verified: true });
                    }
                    aliased = true;
                    break;
                }
            }
            // Only remove the placeholder when it is NOT standing in for a bp: it
            // shares coordinates with the aliased bp, so removing it would disarm
            // the real breakpoint too. When aliased, the placeholder's pause IS the
            // hit and is disarmed when that pause is consumed in onPaused. The id
            // stays in placeholderJscIds either way (late pauses get swallowed).
            // Also skip the removal while the placeholder doubles as the entry bp
            // (configurationDone reuses it when Path B already occupies the entry
            // file's line 0): the entry stop has not happened yet — the body is
            // frozen by the brake, so this resolution raced ahead of execution and
            // removing now would disarm the entry bp before its pause. The entry
            // stop's disarm block in onPaused removes it instead.
            if (!aliased && record.jscId !== this.entryBreakpointId) {
                this.requireInspector()
                    .send('Debugger.removeBreakpoint', { breakpointId: record.jscId })
                    .catch(() => undefined);
            }
            return;
        }
        const key = this.jscToDapKey.get(params.breakpointId);
        if (!key) return;
        const bp = this.lineBreakpoints.get(key);
        if (bp === undefined) return;
        if (bp.jscBreakpointId === params.breakpointId && !bp.verified) {
            bp.verified = true;
            this.host?.sendBreakpointChanged({ id: bp.id, line: bp.line, verified: true });
        }
    }
    /**
     * Arm one line breakpoint against JSC at its source-mapped generated
     * coordinates. JSC resolves URL-keyed breakpoints against GENERATED
     * lines/columns (0-based), never TS lines, so the arm must go through the
     * map (statement-start column scan) — the root cause of the deferred-bp
     * misalignment fixed alongside this. A `locations: []` result is NOT a
     * failure: the bp can still resolve later via its own
     * `Debugger.breakpointResolved`, which flips verified in onBreakpointResolved.
     */
    private async armLineBreakpoint(
        absPath: string,
        key: string,
        bp: LineBreakpoint,
        map: AnySourceMap
    ): Promise<void> {
        const gen = map.generatedLocation(bp.line, 0);
        if (gen === null) {
            bp.message = 'breakpoint does not map to generated code';
            this.host?.sendBreakpointChanged({ id: bp.id, line: bp.line, verified: false, message: bp.message });
            return;
        }
        const params: SetBreakpointByUrlParams = {
            url: absPath,
            lineNumber: gen.line - 1,
            condition: bp.condition,
            isAutoContinue: false,
        };
        // NoopSourceMap columns are a constant identity (1) that can point at
        // leading whitespace (`locations: []`); omit and let JSC snap per line.
        if (map instanceof SourceMap) params.columnNumber = gen.column - 1;
        try {
            const result = await this.requireInspector().send('Debugger.setBreakpointByUrl', params);
            // Liveness re-check: a full-replacement setBreakpoints (or any path
            // that clears the bp) can race this await. If the slot now holds a
            // different bp (or is empty), the JSC bp we just armed is orphaned —
            // nothing in lineBreakpoints references it. Dispose it and bail before
            // stamping the id, so no stale jscToDapKey entry is left either.
            if (this.lineBreakpoints.get(key) !== bp) {
                this.requireInspector()
                    .send('Debugger.removeBreakpoint', { breakpointId: result.breakpointId })
                    .catch(() => undefined);
                return;
            }
            bp.jscBreakpointId = result.breakpointId;
            this.jscToDapKey.set(result.breakpointId, key);
            if (!bp.verified && result.locations.length > 0) {
                bp.verified = true;
                this.host?.sendBreakpointChanged({ id: bp.id, line: bp.line, verified: true });
            }
        } catch (err) {
            bp.message = (err as Error).message;
            this.host?.sendBreakpointChanged({ id: bp.id, line: bp.line, verified: false, message: bp.message });
        }
    }

    // ------------------------------------------------------------------
    // Breakpoints
    // ------------------------------------------------------------------

    async setBreakpoints(args: {
        source: { path?: string };
        breakpoints: DebugProtocol.SourceBreakpoint[];
    }): Promise<DebugProtocol.Breakpoint[]> {
        const path = args.source.path;
        if (!path) throw new Error('setBreakpoints: missing source.path');
        // Canonicalize: Bun reports script URLs with symlinks resolved (macOS
        // /tmp → /private/tmp), so a client-supplied path through a symlink must
        // be canonicalized before it can match sources / arm JSC breakpoints.
        let absPath: string;
        try {
            absPath = realpathSync(resolve(path));
        } catch {
            absPath = resolve(path); // not on disk (yet) — JSC matches the literal URL
        }

        // Remove previously armed breakpoints for this source (full replacement),
        // including any not-yet-armed deferred requests and their placeholder.
        this.pendingByPath.delete(absPath);
        for (const [pid, record] of [...this.placeholdersById]) {
            if (record.path !== absPath) continue;
            this.placeholdersById.delete(pid);
            // id stays in placeholderJscIds: late placeholder pauses get swallowed.
            this.requireInspector()
                .send('Debugger.removeBreakpoint', { breakpointId: pid })
                .catch(() => undefined);
        }
        for (const [key, bp] of [...this.lineBreakpoints]) {
            if (bp.path !== absPath) continue;
            this.lineBreakpoints.delete(key);
            if (bp.jscBreakpointId) {
                this.jscToDapKey.delete(bp.jscBreakpointId);
                this.requireInspector()
                    .send('Debugger.removeBreakpoint', { breakpointId: bp.jscBreakpointId })
                    .catch(() => undefined);
            }
        }

        const out: DebugProtocol.Breakpoint[] = [];
        const entry = this.sources.get(absPath);

        for (const requested of args.breakpoints) {
            const key = `line-${++this.nextBpKey}`;
            const bp: LineBreakpoint = {
                id: this.nextDapBpId++,
                path: absPath,
                line: requested.line,
                condition: requested.condition,
                hitCondition: requested.hitCondition,
                logMessage: requested.logMessage,
                verified: false,
                hitCount: 0,
            };
            this.lineBreakpoints.set(key, bp);

            if (requested.hitCondition !== undefined) {
                const n = Number(requested.hitCondition);
                if (!Number.isInteger(n) || n < 1) {
                    bp.message = 'hitCondition must be a positive integer';
                    out.push({ id: bp.id, verified: false, line: bp.line, message: bp.message });
                    continue;
                }
            }

            if (entry) {
                // The script is parsed (its map decoded synchronously at scriptParsed):
                // arm immediately at the mapped generated coordinates.
                await this.armLineBreakpoint(absPath, key, bp, entry.sourceMap ?? new NoopSourceMap());
            } else {
                // The script is not parsed yet (the normal launch flow — setBreakpoints
                // arrives while the --inspect-wait gate holds parsing). Defer: the bp
                // arms at mapped coordinates from onScriptParsed. Surface a
                // `pending script load` message so clients (and the agent-usage doc)
                // can show why verified is still false. Register the pending key
                // SYNCHRONOUSLY, before the placeholder RPC below awaits — the gate
                // may already be open, in which case the scriptParsed event can be
                // processed while this call is in flight; if pending were set only
                // after the await, onScriptParsed would see an empty registry and
                // the bp would never arm.
                bp.message = 'pending script load';
                let keys = this.pendingByPath.get(absPath);
                if (!keys) this.pendingByPath.set(absPath, (keys = []));
                keys.push(key);

                // Arm a line-0 placeholder for the source — upstream's design — as a
                // module-scoped engine brake: JSC pauses at the source's first
                // pausable statement when it executes, holding the module body until
                // the mapped arms bind. Without it, a program with imports runs its
                // whole synchronous body (self-tests, setup) before the arm RPC is
                // processed, and the breakpoint verifies only after it never fires.
                let record: PlaceholderRecord | undefined;
                for (const candidate of this.placeholdersById.values()) {
                    if (candidate.path === absPath) {
                        record = candidate;
                        break;
                    }
                }
                if (!record) {
                    try {
                        const result = await this.requireInspector().send('Debugger.setBreakpointByUrl', {
                            url: absPath,
                            lineNumber: 0,
                        });
                        record = { jscId: result.breakpointId, path: absPath, bpKeys: [] };
                        this.placeholdersById.set(result.breakpointId, record);
                        this.placeholderJscIds.add(result.breakpointId);
                    } catch (err) {
                        bp.message = (err as Error).message;
                    }
                }
                if (record) {
                    record.bpKeys.push(key);
                } else {
                    // Placeholder arm failed: without the brake the body may run past
                    // the bp, but the pending registration above still lets
                    // onScriptParsed arm it — best effort.
                }
            }
            out.push({ id: bp.id, verified: bp.verified, line: bp.line, message: bp.message });
        }
        return out;
    }

    async setFunctionBreakpoints(args: {
        breakpoints: DebugProtocol.FunctionBreakpoint[];
    }): Promise<DebugProtocol.Breakpoint[]> {
        const inspector = this.requireInspector();
        let requested = args.breakpoints;

        // Remove symbolic breakpoints not in the new set, and ones whose gating
        // changed: a re-sent symbol with a new condition/hitCondition must be
        // re-armed, not silently kept with the old gating.
        for (const [key, bp] of [...this.functionBreakpoints]) {
            const incoming = requested.find((r) => r.name === bp.symbol);
            if (
                incoming !== undefined &&
                incoming.condition === bp.condition &&
                incoming.hitCondition === bp.hitCondition
            ) {
                requested = requested.filter((r) => r.name !== bp.symbol); // keep existing
                continue;
            }
            this.functionBreakpoints.delete(key);
            this.jscToDapKey.delete(bp.jscBreakpointId);
            inspector.send('Debugger.removeSymbolicBreakpoint', { symbol: bp.symbol }).catch(() => undefined);
        }

        const out: DebugProtocol.Breakpoint[] = [];
        for (const b of requested) {
            const key = `fn-${++this.nextBpKey}`;
            const bp: FunctionBreakpoint = {
                id: this.nextDapBpId++,
                symbol: b.name,
                condition: b.condition,
                hitCondition: b.hitCondition,
                jscBreakpointId: '',
                verified: true,
                hitCount: 0,
            };
            if (b.hitCondition !== undefined) {
                const n = Number(b.hitCondition);
                if (!Number.isInteger(n) || n < 1) {
                    out.push({ id: bp.id, verified: false, message: 'hitCondition must be a positive integer' });
                    continue;
                }
            }
            try {
                const result = await inspector.send('Debugger.addSymbolicBreakpoint', {
                    symbol: b.name,
                    isRegex: false,
                    condition: b.condition,
                });
                bp.jscBreakpointId = result.breakpointId;
                this.functionBreakpoints.set(key, bp);
                this.jscToDapKey.set(result.breakpointId, key);
                out.push({ id: bp.id, verified: true });
            } catch (err) {
                out.push({ id: bp.id, verified: false, message: (err as Error).message });
            }
        }
        return out;
    }

    async setExceptionBreakpoints(args: { filters?: string[] }): Promise<void> {
        const filters = args.filters ?? [];
        // WebKit/Bun's Debugger.setPauseOnExceptions takes a `state` string, not a boolean.
        const state =
            filters.includes('all') || filters.includes('pauseOnExceptions')
                ? 'all'
                : filters.includes('uncaught')
                  ? 'uncaught'
                  : 'none';
        await this.requireInspector().send('Debugger.setPauseOnExceptions', { state });
    }

    // ------------------------------------------------------------------
    // Pause handling + stepping
    // ------------------------------------------------------------------

    private async onPaused(params: PausedEvent): Promise<void> {
        this.paused = true;
        this.currentCallFrames = params.callFrames;
        this.currentFrameIds = [];
        this.frameIdToCallFrame.clear();
        // New pause = new object generation: JSC invalidates remote objects from
        // earlier pauses on resume, but does NOT reliably send Debugger.resumed
        // (never after stepNext), so stale refs must be dropped HERE. Without
        // this, variables requests carrying old refs fetched old objectIds and
        // JSC answered -32000 Internal error — nvim's variables disappearing
        // after every step.
        this.objectRefs.reset();
        for (const frame of this.currentCallFrames) {
            const id = this.nextFrameId++;
            this.currentFrameIds.push(id);
            this.frameIdToCallFrame.set(id, frame);
        }

        // JSC/Bun reports the hit id(s) variably: an array under `data.breakpoints`
        // or a single string under `data.breakpointId` (observed in Bun 1.3.x).
        const data = params.data;
        const hitIds: string[] = [];
        if (data) {
            if (Array.isArray(data.breakpoints)) {
                hitIds.push(...data.breakpoints.filter((id): id is string => typeof id === 'string'));
            }
            if (typeof data.breakpointId === 'string') hitIds.push(data.breakpointId);
        }
        // Entry brake (configurationDone sent a pre-gate Debugger.pause because
        // deferred breakpoints were pending): the VM froze at the FIRST executed
        // statement of the whole process, with deferred bps armed meanwhile. If a
        // bp targets THIS statement, the pause is that bp's hit — a bp bound at
        // the pause location does not re-fire once resumed — so its keys join the
        // normal gating below and the stop reports as 'breakpoint'. Otherwise the
        let brakeKeys: string[] = [];
        // pause is the brake's own — but only when JSC reports it as such
        // (PauseOnNextStatement, or 'other' when stepping from a bp pause). A
        // Breakpoint/exception/FunctionCall pause arriving while brakePending is
        // a genuine hit that raced the brake's own pause — process it normally
        // and KEEP brakePending so the real brake pause still gets its
        // swallow/stop decision when it arrives.
        if (this.brakePending && (params.reason === 'PauseOnNextStatement' || params.reason === 'other')) {
            this.brakePending = false;
            brakeKeys = this.brakeOverlapKeys();
            const topFrame = this.currentCallFrames[0];
            const topEntry = topFrame?.location ? this.sourcesByScriptId.get(topFrame.location.scriptId) : undefined;
            const inEntryFile = topEntry?.path === this.entryBreakpointUrl;
            if (brakeKeys.length === 0 && (!this.firstPausePending || !inEntryFile)) {
                await this.resumeInternal();
                return;
            }
        }
        // Placeholder pauses (per-source line-0 brake, see setBreakpoints Path B):
        // the source's first pausable statement executed. If the placeholder was
        // aliased to a bp (its resolved location IS that bp's target), the pause
        // is that bp's hit — join the normal gating below. Otherwise it is
        // internal machinery: swallow it (the mapped arms already bound at
        // onScriptParsed, so execution resumes into them) — UNLESS stopOnEntry
        // is still owed and the pause is in the entry file: the entry bp and the
        // placeholder are both armed at line 0 of the same source and resolve to
        // the same statement, so JSC reports a single pause carrying whichever
        // id it picked. When that id is the placeholder's, swallowing here would
        // eat the entry stop; fall through so it surfaces as 'entry' (the
        // entry-bp disarm below also runs, once).
        let hitKeysAlias: string[] = [];
        // JSC reports the hit id variably (data.breakpoints array OR a single
        // data.breakpointId string). Test EVERY id for placeholder membership,
        // not just data.breakpointId — a placeholder pause reported via the
        // array form was slipping past this block and surfacing as a spurious
        // stop. Aliased ids join gating; non-aliased ones are swallowed unless
        // the entry stop is still owed here.
        const placeholderIds = hitIds.filter((id) => this.placeholderJscIds.has(id));
        if (placeholderIds.length > 0) {
            let aliasedId: string | undefined;
            let aliasKey: string | undefined;
            for (const id of placeholderIds) {
                const key = this.placeholderAlias.get(id);
                if (key !== undefined) {
                    aliasedId = id;
                    aliasKey = key;
                    break;
                }
            }
            if (aliasKey !== undefined && aliasedId !== undefined) {
                this.placeholderAlias.delete(aliasedId);
                // When the placeholder doubles as the entry bp (stopOnEntry + Path B),
                // do NOT remove it here: the first pause reports the bp hit, but the
                // entry stop is still owed and fires on continue when the entry
                // statement re-executes. Removing would disarm the entry bp (JSC
                // dedupes same-location breakpoints), leaving no way to surface the
                // entry stop. The real bp also cannot re-arm yet (JSC rejects a
                // duplicate at the same location); it re-arms after the entry stop
                // disarms the placeholder below. Keep firstPausePending true.
                if (aliasedId !== this.entryBreakpointId) {
                    this.requireInspector()
                        .send('Debugger.removeBreakpoint', { breakpointId: aliasedId })
                        .catch(() => undefined);
                    const bp = this.lineBreakpoints.get(aliasKey);
                    if (bp) {
                        bp.jscBreakpointId = undefined;
                        const map = this.sources.get(bp.path)?.sourceMap ?? new NoopSourceMap();
                        void this.armLineBreakpoint(bp.path, aliasKey, bp, map);
                    }
                }
                hitKeysAlias = [aliasKey];
            } else {
                const top = this.currentCallFrames[0];
                const topSource = top?.location ? this.sourcesByScriptId.get(top.location.scriptId) : undefined;
                const entryOwedHere = this.firstPausePending && topSource?.path === this.entryBreakpointUrl;
                if (!entryOwedHere) {
                    await this.resumeInternal();
                    return;
                }
            }
        }
        const hitKeys = [
            ...new Set([
                ...hitIds.map((id) => this.jscToDapKey.get(id)).filter((k): k is string => !!k),
                ...brakeKeys,
                ...hitKeysAlias,
            ]),
        ];
        const hitLineEntries = hitKeys.map((k) => this.lineBreakpoints.get(k)).filter((b): b is LineBreakpoint => !!b);
        const hitFnEntries = hitKeys
            .map((k) => this.functionBreakpoints.get(k))
            .filter((b): b is FunctionBreakpoint => !!b);
        for (const bp of hitLineEntries) bp.hitCount++;
        for (const bp of hitFnEntries) bp.hitCount++;

        // Hit-count gating (line + function). Several bps can share one pause
        // (two bps on one statement via the brake overlap, or an alias), so stop
        // when ANY hit bp has reached its count. The old loop resumed on the
        // FIRST failing bp, silently suppressing the siblings' stop. With
        // no hit bp at all (pure exception/entry stop) there is nothing to gate.
        const gated = [...hitLineEntries, ...hitFnEntries];
        if (gated.length > 0) {
            const anyCountPass = gated.some(
                (bp) => bp.hitCondition === undefined || bp.hitCount >= Number(bp.hitCondition)
            );
            if (!anyCountPass) {
                await this.resumeInternal();
                return;
            }

            // Condition gating: a falsy condition resumes without stopping — but
            // only when EVERY hit bp fails; an unconditioned sibling passes
            // trivially and its stop must not be eaten by a failing conditioned one
            // Evaluate conditioned bps until one passes; short-circuit on
            // the first unconditioned bp (it always passes).
            const topFrame = this.currentCallFrames[0];
            let anyConditionPass = false;
            for (const bp of gated) {
                if (bp.condition === undefined) {
                    anyConditionPass = true;
                    break;
                }
                if (await this.evalCondition(bp.condition, topFrame)) {
                    anyConditionPass = true;
                    break;
                }
            }
            if (!anyConditionPass) {
                await this.resumeInternal();
                return;
            }
        }

        // Logpoints: emit EVERY hit logpoint's interpolated message, then resume
        // — never stop. A single pause can carry several logpoints (e.g. two
        // deferred logpoints on the same statement, surfaced together via the
        // brake overlap); logging only the first (the old `find`) dropped the
        // rest. Logpoints never stop regardless of sibling non-logpoint bps.
        const logBps = hitLineEntries.filter((b) => b.logMessage !== undefined);
        if (logBps.length > 0) {
            for (const lp of logBps) await this.evalLogpoint(lp);
            await this.resumeInternal();
            return;
        }

        // A bp hit — from the brake overlap, a placeholder alias, OR a real
        // armed bp matched via jscToDapKey — always reports 'breakpoint', even
        // when it coincides with the first pause. Otherwise stopOnEntry + a bp at
        // the entry statement surfaces as a spurious 'entry' stop at the bp's line
        // (the old mapStopReason firstPausePending mapping). A pure entry stop
        // (the entry bp/placeholder firing with NO aliased real bp) has no entry
        // in hitKeys, so hasBpHit is false and mapStopReason reports 'entry'.
        const hasBpHit =
            brakeKeys.length > 0 || hitKeysAlias.length > 0 || hitLineEntries.length > 0 || hitFnEntries.length > 0;
        const stop = hasBpHit ? { reason: 'breakpoint' } : this.mapStopReason(params.reason, params);
        const hitDapIds = hitKeys
            .map((k) => (this.lineBreakpoints.get(k) ?? this.functionBreakpoints.get(k))?.id)
            .filter((id): id is number => id !== undefined);
        this.emitStopped(stop, hitDapIds);
        // The entry breakpoint is one-shot: disarm it only after a stop that was
        // genuinely the entry stop (reason 'entry'), so a coinciding bp stop
        // does not eat the entry stop the client is still owed on continue.
        if (this.entryBreakpointId !== undefined && stop.reason === 'entry') {
            const id = this.entryBreakpointId;
            this.entryBreakpointId = undefined;
            this.firstPausePending = false;
            try {
                void this.requireInspector()
                    .send('Debugger.removeBreakpoint', { breakpointId: id })
                    .catch(() => undefined);
            } catch {
                // inspector already gone
            }
        }
    }

    private async evalCondition(expression: string, frame: CallFrame | undefined): Promise<boolean> {
        const inspector = this.requireInspector();
        try {
            const res =
                frame !== undefined
                    ? await inspector.send('Debugger.evaluateOnCallFrame', {
                          callFrameId: frame.callFrameId,
                          expression,
                          returnByValue: true,
                          silent: true,
                      })
                    : await inspector.send('Runtime.evaluate', { expression, returnByValue: true, silent: true });
            if (res.wasThrown) {
                this.emitOutput('console', `condition threw: ${describeException(res.result)}\n`);
                return false;
            }
            const result = res.result;
            // JSC does not serialize bigint/symbol `value` (it is undefined), so the
            // `!!result.value` truthiness check treated every nonzero bigint and
            // every symbol as falsy — a condition like `n` on `0n`/`42n`/`Symbol()`
            // never stopped. Recover truthiness from the type/description: bigint
            // from its description (`0n` is falsy, any other is truthy), symbol as
            // always-truthy. Other types fall back to the serialized value.
            if (result.type === 'bigint') {
                const desc = result.description ?? '';
                try {
                    return BigInt(desc.replace(/n$/, '')) !== 0n;
                } catch {
                    return true;
                }
            }
            if (result.type === 'symbol') return true;
            return !!result.value;
        } catch (err) {
            this.emitOutput('console', `condition eval error: ${(err as Error).message}\n`);
            return false;
        }
    }

    private async evalLogpoint(bp: LineBreakpoint): Promise<void> {
        const expression = compileLogMessage(bp.logMessage ?? '');
        const inspector = this.requireInspector();
        const frame = this.currentCallFrames[0]?.callFrameId;
        try {
            const res =
                frame !== undefined
                    ? await inspector.send('Debugger.evaluateOnCallFrame', {
                          callFrameId: frame,
                          expression,
                          returnByValue: true,
                          silent: true,
                      })
                    : await inspector.send('Runtime.evaluate', { expression, returnByValue: true, silent: true });
            const raw = res.result.value ?? '';
            const text = res.wasThrown
                ? `<logpoint error: ${describeException(res.result)}>`
                : typeof raw === 'string'
                  ? raw
                  : JSON.stringify(raw);
            this.emitOutput('console', text + '\n');
        } catch (err) {
            this.emitOutput('console', `<logpoint error: ${(err as Error).message}>\n`);
        }
    }

    private mapStopReason(reason: string, params: PausedEvent): { reason: string; exceptionText?: string } {
        switch (reason) {
            case 'breakpoint':
            case 'Breakpoint': // WebKit capitalization as emitted by Bun
            case 'DebuggerStatement':
                return { reason: this.firstPausePending ? 'entry' : 'breakpoint' };
            case 'FunctionCall':
                return { reason: 'function breakpoint' };
            case 'exception':
            case 'exceptionWithStackTrace':
                return { reason: 'exception', exceptionText: describePausedData(params.data) };
            case 'PauseOnNextStatement':
            case 'AlreadyPaused':
            case 'entry':
                // A next/stepIn/stepOut request is in flight: this pause is its
                // completion. Checked before firstPausePending — the brake swallow
                // and placeholder branches run before mapStopReason and never reach
                // here with stepPending set, so the flag belongs to a real step.
                if (this.stepPending) {
                    this.stepPending = false;
                    return { reason: 'step' };
                }
                if (this.firstPausePending) {
                    this.firstPausePending = false;
                    return { reason: 'entry' };
                }
                return { reason: 'pause' };
            default:
                // Bun reports step completion as reason 'other' on some versions
                // (probe-verified: next from a bp pause). Upstream's #stopped flag
                // wins over ANY unrecognized reason the same way; a bp hit mid-step
                // maps 'breakpoint' above and leaves the flag for the completion.
                if (this.stepPending) {
                    this.stepPending = false;
                    return { reason: 'step' };
                }
                return { reason: 'pause' };
        }
    }

    /**
     * Bps whose source-mapped target IS the brake pause's statement. A bp that
     * got bound at the exact location the brake paused on can never fire
     * (resume does not re-trigger the pause statement), so the brake's pause
     * stands in for the hit. Column-exact for real source maps; line-only for
     * NoopSourceMap, whose identity column is unreliable.
     */
    private brakeOverlapKeys(): string[] {
        const frame = this.currentCallFrames[0];
        const loc = frame?.location;
        if (!loc) return [];
        const entry = this.sourcesByScriptId.get(loc.scriptId);
        const map = entry?.sourceMap;
        if (!entry || !map) return [];
        const keys: string[] = [];
        for (const [key, bp] of this.lineBreakpoints) {
            if (bp.path !== entry.path) continue;
            const gen = map.generatedLocation(bp.line, 0);
            if (gen === null || gen.line - 1 !== loc.lineNumber) continue;
            if (map instanceof SourceMap && gen.column - 1 !== loc.columnNumber) continue;
            keys.push(key);
        }
        return keys;
    }

    private clearPauseState(): void {
        // Mirror onResumed: drop paused + the frame tables so a stackTrace
        // between the resume/step RPC and the next pause does not return stale
        // frames. JSC does not reliably emit Debugger.resumed (never after
        // stepNext — probe-verified), so the resume/step send paths must clear
        // this themselves; onResumed still does it for the cases JSC does emit.
        // objectRefs is NOT reset here — only at onPaused start, the one
        // transition every pause passes through.
        this.paused = false;
        this.currentCallFrames = [];
        this.currentFrameIds = [];
        this.frameIdToCallFrame.clear();
    }

    private onResumed(): void {
        this.clearPauseState();
    }

    /**
     * Release the object groups `evaluate` creates ('repl'/'debugger'/'console')
     * so the DEBUGGEE does not retain evaluated objects across pauses: JSC
     * keeps group members alive until releaseObjectGroup, so without this
     * every evaluate leaks the referenced objects in the target. Called
     * on every resume/step path — the groups are dead weight once running.
     * Fire-and-forget: a failure (e.g. inspector mid-close) must not block the
     * resume. Sent before the resume RPC so JSC releases while still paused.
     */
    private releaseObjectGroups(): void {
        const inspector = this.inspector;
        if (!inspector) return;
        for (const objectGroupName of ['repl', 'debugger', 'console']) {
            void inspector.send('Runtime.releaseObjectGroup', { objectGroupName }).catch(() => undefined);
        }
    }

    private async resumeInternal(): Promise<void> {
        // Swallow resumes (brake/placeholder/logpoint/condition-gating) cancel
        // any in-flight step just like a client continue: every swallow path is
        // reached from a bp-family pause (or the brake), and probe-verified JSC
        // behavior is that a bp hit mid-step cancels the step — no completion
        // pause follows. Keeping the flag would strand it and mislabel a later
        // manual pause as 'step' (logstep probe: stepping over a logpoint line
        // printed the log and ran free, step absorbed).
        this.stepPending = false;
        this.releaseObjectGroups();
        await this.requireInspector().send('Debugger.resume');
        this.clearPauseState();
    }

    private emitStopped(stop: { reason: string; exceptionText?: string }, hitBreakpointIds: number[]): void {
        const emit = () =>
            this.host?.sendStopped(
                stop.reason,
                1,
                stop.exceptionText,
                hitBreakpointIds.length > 0 ? hitBreakpointIds : undefined
            );
        if (!this.initializedSent) {
            this.stoppedBuffer.push(emit);
            return;
        }
        emit();
    }

    // ------------------------------------------------------------------
    // Stack / scopes / variables / evaluate
    // ------------------------------------------------------------------

    threads(): { id: number; name: string }[] {
        return [{ id: 1, name: 'main' }];
    }

    stackTrace(args: { threadId: number; startFrame: number; levels?: number }): {
        stackFrames: DebugProtocol.StackFrame[];
        totalFrames: number;
    } {
        if (!this.paused || this.currentCallFrames.length === 0) return { stackFrames: [], totalFrames: 0 };
        const start = args.startFrame;
        // DAP: missing levels = all frames (nvim-dap omits it; VS Code pages).
        const levels = args.levels ?? this.currentCallFrames.length;
        const stackFrames: DebugProtocol.StackFrame[] = [];
        for (let i = start; i < start + levels && i < this.currentCallFrames.length; i++) {
            const frame = this.currentCallFrames[i];
            if (frame === undefined) break;
            const id = this.currentFrameIds[i] ?? 0;
            const entry = this.sourcesByScriptId.get(frame.location.scriptId);
            const map = entry?.sourceMap ?? null;
            const jsLine = frame.location.lineNumber;
            const jsCol = frame.location.columnNumber;
            let line = jsLine + 1;
            let column = 1;
            const sourcePath = entry?.path ?? '';
            const orig = map ? map.originalLocation(jsLine, jsCol) : null;
            if (orig && map instanceof SourceMap && orig.source !== '<unmapped>') {
                line = orig.line;
                column = orig.column;
            }
            const internal = sourcePath === '' || !isAbsolute(sourcePath);
            stackFrames.push({
                id,
                name: frame.functionName || (internal ? '<internal>' : '<anonymous>'),
                source: internal ? undefined : { name: basename(sourcePath), path: sourcePath },
                line,
                column,
                presentationHint: internal ? 'subtle' : undefined,
            });
        }
        return { stackFrames, totalFrames: this.currentCallFrames.length };
    }

    async scopes(frameId: number): Promise<DebugProtocol.Scope[]> {
        const frame = this.frameIdToCallFrame.get(frameId);
        if (!frame) return [];
        const scopes: DebugProtocol.Scope[] = frame.scopeChain.map((scope, i) => {
            const objectId = scope.object.objectId;
            const variablesReference = objectId
                ? this.objectRefs.create({ objectId, callFrameId: frame.callFrameId })
                : 0;
            // JSC/Bun emits the function-local scope as the first `closure`; DAP
            // clients look for a `Local` scope to show locals.
            const isLocal = i === 0 && (scope.type === 'local' || scope.type === 'closure');
            return {
                name: isLocal ? 'Local' : scopeName(scope.type),
                variablesReference,
                expensive: scope.type === 'global' || scope.type === 'script',
            };
        });
        // Advertise indexedVariables on scopes whose property list exceeds the
        // paging threshold, so a compliant DAP client sends start/count and the
        // start/count slicing becomes reachable (otherwise the client never pages and
        // the whole collection is dumped in one response). The count is free —
        // Runtime.getDisplayableProperties returns the full list each call.
        await Promise.all(
            scopes.map(async (s) => {
                if (s.variablesReference <= 0) return;
                const n = await this.childCount(this.objectRefs.get(s.variablesReference)?.objectId ?? '');
                if (n >= PAGING_THRESHOLD) s.indexedVariables = n;
            })
        );
        return scopes;
    }

    /**
     * Count the displayable properties of `objectId` (own + internal). Returns
     * 0 for a missing/empty result OR a failed fetch (e.g. inspector mid-close),
     * so callers (scopes/variables/evaluate) can gate the indexedVariables
     * advertisement without guarding the fetch — a count failure just means we
     * skip the advertisement, never that the parent operation rejects.
     */
    private async childCount(objectId: string): Promise<number> {
        if (!objectId) return 0;
        try {
            const props = await this.requireInspector().send('Runtime.getDisplayableProperties', {
                objectId,
                ownProperties: true,
                generatePreview: true,
            });
            return [...(props.properties ?? props.result ?? []), ...(props.internalProperties ?? [])].length;
        } catch {
            return 0;
        }
    }

    async variables(args: {
        variablesReference: number;
        start?: number;
        count?: number;
    }): Promise<DebugProtocol.Variable[]> {
        const entry = this.objectRefs.get(args.variablesReference);
        if (!entry) return [];
        const props = await this.requireInspector().send('Runtime.getDisplayableProperties', {
            objectId: entry.objectId,
            ownProperties: true,
            generatePreview: true,
        });
        const all = [...(props.properties ?? props.result ?? []), ...(props.internalProperties ?? [])];
        // DAP paging: honor the client's start/count window so a huge
        // collection is not dumped in one response; slice before formatting so
        // only the requested rows get object refs.
        const start = args.start ?? 0;
        const count = args.count ?? all.length;
        const page = all.slice(start, start + count);
        const vars: DebugProtocol.Variable[] = page.map((p) => {
            const variable = {
                name: p.name,
                value: formatRemoteObject(p.value),
                variablesReference: p.value?.objectId ? this.objectRefs.create({ objectId: p.value.objectId }) : 0,
                type: p.value?.type,
            };
            return variable;
        });

        // Advertise indexedVariables on each returned child that itself has an
        // objectId (hence children), so the client pages ITS children. The count
        // is fetched per expandable child; this is N fetches only for children the
        // client is about to expand anyway. Parallelized to avoid serial latency.
        await Promise.all(
            vars.map(async (v) => {
                if (v.variablesReference <= 0) return;
                const childId = this.objectRefs.get(v.variablesReference)?.objectId ?? '';
                const n = await this.childCount(childId);
                if (n >= PAGING_THRESHOLD) v.indexedVariables = n;
            })
        );
        return vars;
    }

    async evaluate(args: {
        expression: string;
        frameId?: number;
        context?: string;
    }): Promise<DebugProtocol.EvaluateResponse['body']> {
        const inspector = this.requireInspector();
        let res: { result: RemoteObject; wasThrown?: boolean };
        if (this.paused && args.frameId !== undefined) {
            // A frameId the client believes is current but we never issued (or
            // already cleared on resume) is stale: evaluating it in global scope
            // would silently lose the frame context and misreport locals. Refuse.
            const frame = this.frameIdToCallFrame.get(args.frameId);
            if (!frame)
                throw new Error(`evaluate: unknown frameId ${args.frameId} (the target is no longer paused there)`);
            res = await inspector.send('Debugger.evaluateOnCallFrame', {
                callFrameId: frame.callFrameId,
                expression: args.expression,
                objectGroup: args.context === 'repl' ? 'repl' : 'debugger',
                includeCommandLineAPI: true,
                returnByValue: false,
                generatePreview: true,
            });
        } else {
            res = await inspector.send('Runtime.evaluate', {
                expression: args.expression,
                objectGroup: 'console',
                includeCommandLineAPI: true,
                returnByValue: false,
                generatePreview: true,
            });
        }
        if (res.wasThrown) {
            throw new Error(res.result.description ?? describeException(res.result));
        }
        const variablesReference = res.result.objectId ? this.objectRefs.create({ objectId: res.result.objectId }) : 0;
        const body: DebugProtocol.EvaluateResponse['body'] = {
            result: formatRemoteObject(res.result),
            variablesReference,
        };
        // Advertise the child count on a large evaluate result so the client
        // pages its properties (same contract as scopes/variables).
        if (variablesReference > 0 && res.result.objectId) {
            const n = await this.childCount(res.result.objectId);
            if (n >= PAGING_THRESHOLD) body.indexedVariables = n;
        }
        return body;
    }

    // ------------------------------------------------------------------
    // Session-facing teardown
    // ------------------------------------------------------------------

    /**
     * Terminate the debuggee: SIGTERM the launched child (SIGKILL after 3 s),
     * or — in attach mode — resolve the target's pid over the inspector and
     * SIGTERM it (same escalation). Attach-mode termination is refused for
     * non-loopback targets: a remote pid cannot be killed safely from here.
     */
    async terminateDebuggee(): Promise<void> {
        const child = this.launchedChild;
        if (child) {
            if (child.exitCode !== null || child.killed) return;
            child.kill('SIGTERM');
            await new Promise<void>((done) => {
                const t = setTimeout(() => {
                    child.kill('SIGKILL');
                    done();
                }, 3000);
                child.once('exit', () => {
                    clearTimeout(t);
                    done();
                });
            });
            return;
        }
        if (this.mode !== 'attach' || this.attachHost === null || !this.inspector?.isConnected) return;
        const h = this.attachHost;
        if (!isLoopbackHost(h)) {
            throw new Error(`attach: terminateDebuggee is only supported for loopback targets (got ${h})`);
        }
        const res = await this.inspector.send('Runtime.evaluate', {
            expression: 'process.pid',
            returnByValue: true,
            silent: true,
        });
        const pid = typeof res.result.value === 'number' ? res.result.value : undefined;
        if (pid === undefined) throw new Error('attach: could not resolve the target pid for termination');
        try {
            process.kill(pid, 'SIGTERM');
        } catch {
            // already gone
        }
        const t = setTimeout(() => {
            try {
                process.kill(pid, 0); // throws ESRCH once the target is gone
                process.kill(pid, 'SIGKILL');
            } catch {
                // target exited on its own
            }
        }, 3000);
        t.unref();
    }

    /** Detach from the inspector; the debuggee keeps running (attach mode). */
    closeInspector(): void {
        this.inspector?.close();
        this.inspector = null;
    }

    get isLaunched(): boolean {
        return this.mode === 'launch';
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function scopeName(type: string): string {
    switch (type) {
        case 'local':
            return 'Local';
        case 'closure':
            return 'Closure';
        case 'global':
        case 'script':
        case 'globalLexicalEnvironment':
            return 'Global';
        case 'with':
            return 'With Block';
        case 'catch':
            return 'Catch';
        case 'block':
            return 'Block';
        case 'eval':
            return 'Eval';
        default:
            return 'Scope';
    }
}

/** Compile a DAP logMessage template (`${expr}` interpolation) into an
 * expression. Walks the template tracking brace depth so `${...}` captures
 * balanced braces — a naive `[^}]+` split truncated interpolations at the
 * first inner `}` (object literals `${ {a:1} }`, nested template literals
 * `${ `x ${y}` }`), producing invalid or wrong expressions. Exported for
 * unit testing. */
export function compileLogMessage(template: string): string {
    const segments: string[] = [];
    let i = 0;
    for (;;) {
        const start = template.indexOf('${', i);
        if (start < 0) {
            segments.push(template.slice(i));
            break;
        }
        segments.push(template.slice(i, start));
        // Walk past `${` tracking brace depth so a `}` inside the expression
        // (object literals, nested templates) does not end the interpolation
        // early. depth returns to 0 at the matching closing brace.
        let depth = 1;
        let j = start + 2;
        while (j < template.length && depth > 0) {
            const c = template.charAt(j);
            if (c === '{') depth++;
            else if (c === '}') depth--;
            j++;
        }
        if (depth !== 0) {
            // Unbalanced — best effort: take the rest as the expression.
            segments.push(template.slice(start + 2));
            segments.push('');
            break;
        }
        segments.push(template.slice(start + 2, j - 1));
        i = j;
    }
    // Even indices are literal text, odd indices are interpolated expressions
    // — the same alternation the original split produced, but brace-safe.
    let out = '';
    for (let k = 0; k < segments.length; k++) {
        const seg = segments[k] ?? '';
        if (k % 2 === 0) {
            out += JSON.stringify(seg.replace(/\n/g, '\\n'));
        } else {
            out += ` + (${seg}) + `;
        }
    }
    // Collapse the leading/trailing concatenation artifacts into one expression.
    return out.replace(/^\s*""\s*\+/, '').replace(/\+\s*""\s*$/, '');
}

/** Render a JSC exception/remote object to a readable string. */
function describeException(obj: RemoteObject | undefined | null): string {
    if (!obj) return '';
    if (obj.description) return obj.description;
    if (obj.value !== undefined) return typeof obj.value === 'string' ? obj.value : JSON.stringify(obj.value);
    return obj.type;
}
/** Render a thrown exception carried by a `Debugger.paused` event's data. */
function describePausedData(data: PausedEvent['data']): string {
    if (!data) return '';
    if (typeof data === 'object') {
        const d = data as Record<string, unknown>;
        if (typeof d.description === 'string') return d.description;
        if (d.value !== undefined && d.value !== null) {
            return typeof d.value === 'string' ? d.value : JSON.stringify(d.value);
        }
        if (typeof d.exception === 'object' && d.exception !== null) {
            const ex = d.exception as Record<string, unknown>;
            if (typeof ex.description === 'string') return ex.description;
            if (typeof ex.value === 'string') return ex.value;
        }
    }
    const s = JSON.stringify(data);
    return s && s.length > 300 ? s.slice(0, 300) : s;
}
