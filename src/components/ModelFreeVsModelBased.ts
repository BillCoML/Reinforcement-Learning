/** V1 — Model-Free vs Model-Based side-by-side. Placeholder until Step 4. */
class ModelFreeVsModelBased extends HTMLElement {
  connectedCallback() {
    this.innerHTML = `<div class="panel-placeholder" style="height:280px;display:flex;align-items:center;justify-content:center;background:var(--rl-surface);border:1px dashed var(--rl-border);border-radius:6px;color:var(--rl-ink-muted);font-family:var(--rl-mono)">V1 · ModelFreeVsModelBased — coming in Step 4</div>`;
  }
}
customElements.define("model-free-vs-model-based", ModelFreeVsModelBased);
