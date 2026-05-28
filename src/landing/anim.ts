/**
 * Shared helpers for landing-page mini animations.
 * Each animation uses requestAnimationFrame that self-terminates when its
 * canvas leaves the DOM (so SPA navigation away from the landing GC's cleanly).
 */

export function loop(el: HTMLElement, fn: (tSec: number) => void): void {
  const t0 = performance.now();
  const tick = (now: number) => {
    if (!el.isConnected) return;
    fn((now - t0) / 1000);
    requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
}

export function dpr(): number {
  return Math.min(window.devicePixelRatio || 1, 2);
}

/** Create an HiDPI-aware canvas inside `host`. Returns the 2D context. */
export function mkCanvas(host: HTMLElement, w: number, h: number): {
  c: HTMLCanvasElement;
  g: CanvasRenderingContext2D;
  W: number;
  H: number;
} {
  const c = document.createElement("canvas");
  const r = dpr();
  c.width = w * r;
  c.height = h * r;
  c.style.width = `${w}px`;
  c.style.height = `${h}px`;
  c.style.display = "block";
  host.appendChild(c);
  const g = c.getContext("2d")!;
  g.scale(r, r);
  return { c, g, W: w, H: h };
}

/** Read a CSS variable from :root. */
export function tok(name: string): string {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

/** Linear interpolation. */
export function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

/** Wrap an animation canvas in a labelled "tile" with title + caption + click-to-nav. */
export function makeTile(opts: {
  lesson: string;
  title: string;
  caption: string;
  slug: string;
  build: (host: HTMLElement) => void;
}): HTMLElement {
  const tile = document.createElement("a");
  tile.className = "landing-tile";
  tile.href = `#${opts.slug}`;

  const eyebrow = document.createElement("div");
  eyebrow.className = "landing-tile__eyebrow";
  eyebrow.textContent = opts.lesson;

  const stage = document.createElement("div");
  stage.className = "landing-tile__stage";

  const meta = document.createElement("div");
  meta.className = "landing-tile__meta";
  meta.innerHTML = `
    <div class="landing-tile__title">${opts.title}</div>
    <div class="landing-tile__caption">${opts.caption}</div>
  `;

  tile.appendChild(eyebrow);
  tile.appendChild(stage);
  tile.appendChild(meta);
  opts.build(stage);
  return tile;
}
