/**
 * ChainEditor — the shared transition-matrix editor used by V3 (classification
 * inspector) and V4 (stationary finder), and driven programmatically by V5's
 * "custom" chain. Built once per the lesson's biggest engineering decision.
 *
 * The model is a K×K row-stochastic matrix. Cells are numeric inputs. Editing
 * one entry auto-renormalizes the rest of its row (proportionally) so the row
 * keeps summing to 1 — unless "lock row" mode is on, in which case the row sum
 * is shown with a validity badge and left to the user. States can be added or
 * removed within [minStates, maxStates]. Every mutation emits a `change` with
 * the full matrix; setMatrix(P, {silent}) lets an external controller drive it
 * without a feedback loop.
 */

export interface ChainEditorOptions {
  P: number[][];
  maxStates?: number;
  minStates?: number;
  lockRow?: boolean;
  allowResize?: boolean;
  stateNames?: string[];
  onChange?: (P: number[][]) => void;
}

const EPS = 1e-9;

export class ChainEditor {
  private P: number[][];
  private maxStates: number;
  private minStates: number;
  private lockRow: boolean;
  private allowResize: boolean;
  private stateNames?: string[];
  private onChange?: (P: number[][]) => void;

  private wrap: HTMLDivElement;
  private gridEl!: HTMLDivElement;

  constructor(container: HTMLElement, opts: ChainEditorOptions) {
    this.P = opts.P.map((r) => r.slice());
    this.maxStates = opts.maxStates ?? 6;
    this.minStates = opts.minStates ?? 2;
    this.lockRow = opts.lockRow ?? false;
    this.allowResize = opts.allowResize ?? true;
    this.stateNames = opts.stateNames;
    this.onChange = opts.onChange;

    this.wrap = document.createElement("div");
    this.wrap.className = "mc-editor";
    container.appendChild(this.wrap);
    this.render();
  }

  get K(): number {
    return this.P.length;
  }

  getMatrix(): number[][] {
    return this.P.map((r) => r.slice());
  }

  /** Replace the matrix programmatically. Emits change unless { silent }. */
  setMatrix(P: number[][], opts: { silent?: boolean } = {}): void {
    this.P = P.map((r) => r.slice());
    this.render();
    if (!opts.silent) this.emit();
  }

  setLockRow(b: boolean): void {
    this.lockRow = b;
    this.render();
  }

  private name(i: number): string {
    return this.stateNames?.[i] ?? String(i);
  }

  private emit(): void {
    this.onChange?.(this.getMatrix());
  }

  private rowSum(i: number): number {
    return this.P[i].reduce((a, b) => a + b, 0);
  }

  /**
   * Set entry (i,j)=v, then (unless locked) renormalize the rest of row i to
   * absorb 1−v proportionally to the other entries' current weights.
   */
  private setCell(i: number, j: number, v: number): void {
    v = Math.max(0, Math.min(1, v));
    if (this.lockRow) {
      this.P[i][j] = v;
      this.emit();
      this.refreshRowSum(i);
      return;
    }
    this.P[i][j] = v;
    const others = this.P[i].map((_, k) => k).filter((k) => k !== j);
    const otherSum = others.reduce((a, k) => a + this.P[i][k], 0);
    const remaining = 1 - v;
    if (otherSum > EPS) {
      for (const k of others) this.P[i][k] = (this.P[i][k] / otherSum) * remaining;
    } else {
      const share = remaining / others.length;
      for (const k of others) this.P[i][k] = share;
    }
    this.emit();
    this.render();
  }

  private addState(): void {
    if (this.K >= this.maxStates) return;
    const K = this.K;
    for (let i = 0; i < K; i++) this.P[i].push(0); // existing rows: 0 into new column
    const newRow = new Array<number>(K + 1).fill(0);
    newRow[K] = 1; // new state is absorbing until edited (keeps row stochastic)
    this.P.push(newRow);
    this.render();
    this.emit();
  }

  private removeState(): void {
    if (this.K <= this.minStates) return;
    this.P.pop();
    for (const row of this.P) row.pop();
    // Renormalize rows that lost mass.
    for (let i = 0; i < this.K; i++) {
      const s = this.rowSum(i);
      if (s > EPS) for (let j = 0; j < this.K; j++) this.P[i][j] /= s;
      else this.P[i][i] = 1;
    }
    this.render();
    this.emit();
  }

  private refreshRowSum(i: number): void {
    const badge = this.gridEl.querySelector<HTMLElement>(`.mc-editor__rowsum[data-row="${i}"]`);
    if (!badge) return;
    const s = this.rowSum(i);
    const ok = Math.abs(s - 1) < 1e-6;
    badge.textContent = `Σ ${s.toFixed(2)}`;
    badge.classList.toggle("is-bad", !ok);
    badge.classList.toggle("is-ok", ok);
  }

  private render(): void {
    this.wrap.innerHTML = "";
    const K = this.K;

    const grid = document.createElement("div");
    grid.className = "mc-editor__grid";
    grid.style.gridTemplateColumns = `auto repeat(${K}, 1fr) auto`;
    this.gridEl = grid;

    // header row
    grid.appendChild(this.cornerCell("from \\ to"));
    for (let j = 0; j < K; j++) grid.appendChild(this.headCell(this.name(j), j));
    grid.appendChild(this.cornerCell(""));

    for (let i = 0; i < K; i++) {
      grid.appendChild(this.headCell(this.name(i), i, true));
      for (let j = 0; j < K; j++) grid.appendChild(this.inputCell(i, j));
      // row-sum badge
      const badge = document.createElement("div");
      badge.className = "mc-editor__rowsum";
      badge.dataset.row = String(i);
      const s = this.rowSum(i);
      const ok = Math.abs(s - 1) < 1e-6;
      badge.textContent = `Σ ${s.toFixed(2)}`;
      badge.classList.add(ok ? "is-ok" : "is-bad");
      grid.appendChild(badge);
    }

    this.wrap.appendChild(grid);
    this.wrap.appendChild(this.buildToolbar());
  }

  private cornerCell(text: string): HTMLElement {
    const d = document.createElement("div");
    d.className = "mc-editor__corner";
    d.textContent = text;
    return d;
  }

  private headCell(text: string, idx: number, isRow = false): HTMLElement {
    const d = document.createElement("div");
    d.className = "mc-editor__head" + (isRow ? " is-row" : "");
    const dot = document.createElement("span");
    dot.className = "mc-editor__dot";
    dot.style.background = `var(--mc-state-${(idx % 8) + 1})`;
    d.append(dot, document.createTextNode(text));
    return d;
  }

  private inputCell(i: number, j: number): HTMLElement {
    const cell = document.createElement("div");
    cell.className = "mc-editor__cell" + (i === j ? " is-diag" : "");
    const input = document.createElement("input");
    input.type = "number";
    input.min = "0";
    input.max = "1";
    input.step = "0.05";
    input.value = this.fmt(this.P[i][j]);
    input.setAttribute("aria-label", `P from ${this.name(i)} to ${this.name(j)}`);
    input.addEventListener("change", () => {
      const v = parseFloat(input.value);
      this.setCell(i, j, Number.isFinite(v) ? v : 0);
    });
    cell.appendChild(input);
    return cell;
  }

  private fmt(x: number): string {
    // Trim to 2dp but drop trailing zeros for clean editing.
    return (Math.round(x * 100) / 100).toString();
  }

  private buildToolbar(): HTMLElement {
    const bar = document.createElement("div");
    bar.className = "mc-editor__toolbar rl-controls";

    if (this.allowResize) {
      const add = document.createElement("button");
      add.textContent = "+ state";
      add.disabled = this.K >= this.maxStates;
      add.addEventListener("click", () => this.addState());

      const rem = document.createElement("button");
      rem.textContent = "− state";
      rem.disabled = this.K <= this.minStates;
      rem.addEventListener("click", () => this.removeState());
      bar.append(add, rem);
    }

    const lockWrap = document.createElement("label");
    const lock = document.createElement("input");
    lock.type = "checkbox";
    lock.checked = this.lockRow;
    lock.addEventListener("change", () => this.setLockRow(lock.checked));
    lockWrap.append(lock, " lock row (no auto-renormalize)");
    bar.append(lockWrap);

    return bar;
  }

  destroy(): void {
    this.wrap.remove();
  }
}
