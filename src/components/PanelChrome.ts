/**
 * Shared chrome for every interactive panel: thin border, rounded corners, a
 * monospace header (`panel: <id> | <status>`), and a padded body. Components
 * call `createPanel` and append their own SVG / controls into `body`.
 */

export interface PanelHandle {
  panel: HTMLElement;
  body: HTMLElement;
  header: HTMLElement;
  /** Update the right-hand status text in the header (e.g. `t=247 / 5000`). */
  setStatus(text: string): void;
}

export interface PanelOptions {
  id: string; // shown after "panel: "
  arena?: boolean; // wider breakout for the centerpiece
  heavy?: boolean; // V4–V7: show a mobile "view on desktop" notice
  mobileNotice?: string;
}

export function createPanel(opts: PanelOptions): PanelHandle {
  const panel = document.createElement("div");
  panel.className = "rl-panel" + (opts.arena ? " arena" : "");
  if (opts.heavy) panel.dataset.heavy = "true";

  const header = document.createElement("div");
  header.className = "rl-panel__header";
  const idSpan = document.createElement("span");
  idSpan.className = "panel-id";
  idSpan.textContent = opts.id;
  const statusSpan = document.createElement("span");
  statusSpan.className = "panel-status rl-mono";
  header.append(idSpan, statusSpan);

  const body = document.createElement("div");
  body.className = "rl-panel__body";

  panel.append(header, body);

  if (opts.heavy) {
    const notice = document.createElement("div");
    notice.className = "rl-mobile-notice";
    notice.textContent =
      opts.mobileNotice ?? "View on a wider screen for the full interactive experience.";
    panel.appendChild(notice);
  }

  return {
    panel,
    body,
    header,
    setStatus(text: string) {
      statusSpan.textContent = text;
    },
  };
}
