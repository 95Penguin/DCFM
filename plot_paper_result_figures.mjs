/**
 * Generate paper-ready result figures without a Python plotting dependency.
 * Run with:
 *   node plot_paper_result_figures.mjs
 *
 * Outputs SVG (vector, recommended for Word) and 600-dpi-equivalent PNG files
 * to result/plot/paper_figures.
 */

import fs from "node:fs";
import path from "node:path";
import sharp from "/Users/95penguin/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/sharp/lib/index.js";

const outputDir = path.resolve("result/plot/paper_figures");
const datasets = ["Solar-AL", "SDWPF", "Electricity", "PJM"];
const blue = "#1F5FD0";
const orange = "#D97706";
const red = "#D83A3A";
const gray = "#777777";

const metrics = {
  MAE: { dcfm: [0.0794, 0.2423, 0.0154, 0.0294], baseline: [0.0846, 0.2478, 0.0163, 0.0302] },
  RMSE: { dcfm: [0.2021, 0.4088, 0.1125, 0.0568], baseline: [0.2187, 0.4037, 0.1169, 0.0603] },
  CRPS: { dcfm: [0.0562, 0.1812, 0.0118, 0.0217], baseline: [0.0604, 0.1676, 0.0121, 0.0218] },
};

const intervals = {
  TSDiff: { picp: [0.9240, 0.9185, 0.9175, 0.8524], pinaw: [0.0678, 0.3150, 0.0215, 0.0275] },
  TSFlow: { picp: [0.9455, 0.9272, 0.9377, 0.8865], pinaw: [0.0504, 0.3265, 0.0230, 0.0187] },
  DCFM: { picp: [0.9491, 0.9347, 0.9484, 0.9564], pinaw: [0.0399, 0.3796, 0.0160, 0.0245] },
};

const esc = (text) => String(text).replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;");
const svgOpen = (width, height) => `<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">
<rect width="100%" height="100%" fill="white"/>
<style>text{font-family:Arial,'Helvetica Neue',sans-serif;fill:#202124}.axis{font-size:18px}.tick{font-size:16px;fill:#444}.title{font-size:22px;font-weight:700}.note{font-size:16px;fill:#4a4a4a}</style>`;

async function save(svg, baseName, width) {
  fs.mkdirSync(outputDir, { recursive: true });
  fs.writeFileSync(path.join(outputDir, `${baseName}.svg`), svg);
  await sharp(Buffer.from(svg)).png({ compressionLevel: 9 }).toFile(path.join(outputDir, `${baseName}.png`));
  console.log(`Saved ${baseName}.svg and ${baseName}.png (${width}px wide)`);
}

function makeFig3() {
  const width = 3300;
  const height = 1000;
  const left = 150;
  const right = 80;
  const top = 95;
  const bottom = 215;
  const panelGap = 105;
  const panelW = (width - left - right - 2 * panelGap) / 3;
  const plotH = height - top - bottom;
  const yMap = (value) => top + ((10 - value) / 20) * plotH;
  let s = svgOpen(width, height);

  Object.entries(metrics).forEach(([metric, values], panel) => {
    const x0 = left + panel * (panelW + panelGap);
    const x1 = x0 + panelW;
    const improvement = values.dcfm.map((value, i) => (values.baseline[i] - value) / values.baseline[i] * 100);
    [10, 5, 0, -5, -10].forEach((tick) => {
      const y = yMap(tick);
      s += `<line x1="${x0}" y1="${y}" x2="${x1}" y2="${y}" stroke="${tick === 0 ? "#3F3F3F" : "#D9DDE3"}" stroke-width="${tick === 0 ? 2 : 1}"/>`;
      if (panel === 0) s += `<text class="tick" x="${x0 - 17}" y="${y + 6}" text-anchor="end">${tick}</text>`;
    });
    s += `<text class="title" x="${(x0 + x1) / 2}" y="54" text-anchor="middle">${metric}</text>`;
    improvement.forEach((value, i) => {
      const cx = x0 + panelW * (i + 0.5) / datasets.length;
      const barW = 96;
      const zero = yMap(0);
      const y = yMap(value);
      const fill = value >= 0 ? blue : orange;
      s += `<rect x="${cx - barW / 2}" y="${Math.min(y, zero)}" width="${barW}" height="${Math.abs(zero - y)}" rx="3" fill="${fill}"/>`;
      s += `<text class="axis" x="${cx}" y="${value >= 0 ? y - 15 : y + 29}" text-anchor="middle" font-weight="700" fill="${fill}">${value >= 0 ? "+" : ""}${value.toFixed(1)}%</text>`;
      s += `<text class="tick" x="${cx}" y="${top + plotH + 38}" text-anchor="middle">${datasets[i]}</text>`;
    });
    s += `<line x1="${x0}" y1="${top}" x2="${x0}" y2="${top + plotH}" stroke="#333" stroke-width="1.4"/>`;
    s += `<line x1="${x0}" y1="${top + plotH}" x2="${x1}" y2="${top + plotH}" stroke="#333" stroke-width="1.4"/>`;
  });

  s += `<text class="axis" x="34" y="${top + plotH / 2}" transform="rotate(-90 34 ${top + plotH / 2})" text-anchor="middle">Relative improvement over best baseline (%)</text>`;
  s += `<rect x="${width / 2 - 630}" y="${height - 115}" width="20" height="20" fill="${blue}" rx="3"/><text class="note" x="${width / 2 - 600}" y="${height - 98}">Improved</text>`;
  s += `<rect x="${width / 2 - 420}" y="${height - 115}" width="20" height="20" fill="${orange}" rx="3"/><text class="note" x="${width / 2 - 390}" y="${height - 98}">Decreased</text>`;
  s += `<text class="note" x="${width / 2}" y="${height - 52}" text-anchor="middle">Positive values indicate that DCFM obtains a lower value (better performance).</text>`;
  return `${s}</svg>`;
}

function markerShape(method, x, y, color) {
  if (method === "TSDiff") return `<rect x="${x - 12}" y="${y - 12}" width="24" height="24" fill="${color}" stroke="white" stroke-width="3"/>`;
  if (method === "TSFlow") return `<path d="M ${x} ${y - 15} L ${x + 14} ${y + 12} L ${x - 14} ${y + 12} Z" fill="${color}" stroke="white" stroke-width="3"/>`;
  return `<circle cx="${x}" cy="${y}" r="14" fill="${color}" stroke="white" stroke-width="3"/>`;
}

function makeFig4() {
  const width = 3480;
  const height = 945;
  const left = 140;
  const right = 55;
  const top = 80;
  const bottom = 185;
  const gap = 90;
  const panelW = (width - left - right - 3 * gap) / 4;
  const plotH = height - top - bottom;
  const yMin = 0.84;
  const yMax = 0.97;
  const yMap = (v) => top + (yMax - v) / (yMax - yMin) * plotH;
  const styles = { TSDiff: gray, TSFlow: red, DCFM: blue };
  let s = svgOpen(width, height);

  datasets.forEach((dataset, idx) => {
    const x0 = left + idx * (panelW + gap);
    const x1 = x0 + panelW;
    const widths = Object.values(intervals).map((v) => v.pinaw[idx]);
    const pad = Math.max((Math.max(...widths) - Math.min(...widths)) * 0.3, 0.0035);
    const xmin = Math.min(...widths) - pad;
    const xmax = Math.max(...widths) + pad;
    const xMap = (v) => x0 + (v - xmin) / (xmax - xmin) * panelW;
    [0.85, 0.90, 0.95].forEach((tick) => {
      const y = yMap(tick);
      s += `<line x1="${x0}" y1="${y}" x2="${x1}" y2="${y}" stroke="${tick === 0.95 ? "#4A4A4A" : "#D9DDE3"}" stroke-width="${tick === 0.95 ? 2 : 1}" ${tick === 0.95 ? "stroke-dasharray=\"8 6\"" : ""}/>`;
      if (idx === 0) s += `<text class="tick" x="${x0 - 17}" y="${y + 6}" text-anchor="end">${tick.toFixed(2)}</text>`;
    });
    s += `<text class="title" x="${(x0 + x1) / 2}" y="47" text-anchor="middle">${dataset}</text>`;
    s += `<text class="note" x="${x1 - 7}" y="${yMap(0.95) - 12}" text-anchor="end">target = 0.95</text>`;
    Object.entries(intervals).forEach(([method, values]) => {
      const x = xMap(values.pinaw[idx]);
      const y = yMap(values.picp[idx]);
      s += markerShape(method, x, y, styles[method]);
      const dx = method === "TSDiff" ? 16 : 18;
      const dy = method === "TSFlow" ? -12 : -13;
      s += `<text class="tick" x="${x + dx}" y="${y + dy}" fill="${styles[method]}" font-weight="${method === "DCFM" ? "700" : "400"}">${method}</text>`;
    });
    const tickValues = [xmin, (xmin + xmax) / 2, xmax];
    tickValues.forEach((tick) => {
      const x = xMap(tick);
      s += `<line x1="${x}" y1="${top + plotH}" x2="${x}" y2="${top + plotH + 7}" stroke="#333" stroke-width="1.4"/>`;
      s += `<text class="tick" x="${x}" y="${top + plotH + 32}" text-anchor="middle">${tick.toFixed(3)}</text>`;
    });
    s += `<line x1="${x0}" y1="${top}" x2="${x0}" y2="${top + plotH}" stroke="#333" stroke-width="1.4"/>`;
    s += `<line x1="${x0}" y1="${top + plotH}" x2="${x1}" y2="${top + plotH}" stroke="#333" stroke-width="1.4"/>`;
    s += `<text class="axis" x="${(x0 + x1) / 2}" y="${height - 86}" text-anchor="middle">PINAW (lower is better)</text>`;
  });
  s += `<text class="axis" x="37" y="${top + plotH / 2}" transform="rotate(-90 37 ${top + plotH / 2})" text-anchor="middle">PICP (target = 0.95)</text>`;
  s += `<text class="note" x="${width / 2}" y="${height - 34}" text-anchor="middle">Points closer to the dashed target line with a smaller PINAW are preferred.</text>`;
  return `${s}</svg>`;
}

await save(makeFig3(), "fig3_relative_performance_improvement", 3300);
await save(makeFig4(), "fig4_picp_pinaw_tradeoff", 3480);
