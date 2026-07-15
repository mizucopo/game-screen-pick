// PROTOTYPE: Three report layouts, switchable via ?variant=A|B|C.
const variants = [
  { key: "A", name: "Gallery first" },
  { key: "B", name: "Timeline first" },
  { key: "C", name: "Audit first" },
];

const run = {
  id: "run_20260714T093214Z_7f3a2c",
  requested: 6,
  selectedCount: 5,
  duration: "03:44:18.420",
  candidateMoments: 52,
  annotated: 42,
  status: "completed_with_warnings",
  warning:
    "要求6枚に対して5枚を選択しました。残りはVisual Near-Duplicateまたはtitle上限のため採用できませんでした。",
  selected: [
    {
      index: 1,
      id: "frm_38f1a9c2",
      file: "0001_opening_38f1a9c2e642.webp",
      title: "旅立ちを示すタイトル画面",
      type: "title",
      scene: "opening",
      sceneLabel: "オープニング",
      video: "vid_a1e9c4",
      videoLabel: "01-opening.mp4",
      time: "00:00:02.010",
      progress: 0.0001,
      utility: 0.835,
      base: 0.785,
      coverage: 0.05,
      spoilerPenalty: 0,
      temporalPenalty: 0,
      quality: 0.81,
      spoiler: "none",
      spoilerEvidence: null,
      context: "unavailable",
      summary: "作品名と旅立ちの雰囲気を示すタイトル画面。",
      frameReason: "logoと背景が同時に読みやすいframe。",
      decisionExplanation: "作品を識別できるtitle候補のうち最も品質が高く、title未採用bonusを得た。",
      colorA: "#0f172a",
      colorB: "#334155",
    },
    {
      index: 2,
      id: "frm_6a4d812e",
      file: "0002_exploration_6a4d812e6241.webp",
      title: "広い遺跡を探索する通常play",
      type: "normal_gameplay",
      scene: "exploration",
      sceneLabel: "探索",
      video: "vid_a1e9c4",
      videoLabel: "01-opening.mp4",
      time: "00:12:34.567",
      progress: 0.056,
      utility: 0.915,
      base: 0.815,
      coverage: 0.1,
      spoilerPenalty: 0,
      temporalPenalty: 0,
      quality: 0.9,
      spoiler: "none",
      spoilerEvidence: null,
      context: "weak",
      summary: "遺跡の広さと探索中のHUDが分かる通常play。",
      frameReason: "遺跡の構造とplayer位置が最も明瞭なframe。",
      decisionExplanation: "高い画質と説明価値を持ち、normal_gameplayのcoverageを満たす。",
      colorA: "#365314",
      colorB: "#a16207",
    },
    {
      index: 3,
      id: "frm_b7206e55",
      file: "0003_conversation_b7206e55f3aa.webp",
      title: "次の目的地が示される会話event",
      type: "event",
      scene: "conversation",
      sceneLabel: "会話",
      video: "vid_c7bd10",
      videoLabel: "02-forest.mp4",
      time: "00:45:10.120",
      progress: 0.47,
      utility: 0.881,
      base: 0.791,
      coverage: 0.1,
      spoilerPenalty: -0.01,
      temporalPenalty: 0,
      quality: 0.84,
      spoiler: "low",
      spoilerEvidence: "次の目的地を示す軽微な進行情報。",
      context: "strong",
      summary: "仲間との会話で次の目的地が示されるevent。",
      frameReason: "話者と目的地の背景が同時に分かるframe。",
      decisionExplanation: "説明価値highと強いContext Cue relevanceがevent coverageに寄与した。",
      colorA: "#312e81",
      colorB: "#be185d",
    },
    {
      index: 4,
      id: "frm_18d0ab44",
      file: "0004_equipment_18d0ab449cc7.webp",
      title: "装備構成が分かるmenu",
      type: "menu",
      scene: "equipment",
      sceneLabel: "装備",
      video: "vid_c7bd10",
      videoLabel: "02-forest.mp4",
      time: "01:11:02.333",
      progress: 0.593,
      utility: 0.792,
      base: 0.71,
      coverage: 0.1,
      spoilerPenalty: 0,
      temporalPenalty: -0.018,
      quality: 0.82,
      spoiler: "none",
      spoilerEvidence: null,
      context: "none",
      summary: "装備の種類と構成が読み取れるmenu。",
      frameReason: "項目とcharacter previewが明瞭なframe。",
      decisionExplanation: "menuのsoft coverageを満たし、既選択画像との視覚重複が小さい。",
      colorA: "#164e63",
      colorB: "#0f766e",
    },
    {
      index: 5,
      id: "frm_d9c3f271",
      file: "0005_battle_d9c3f271dd12.webp",
      title: "終盤の特徴的なboss戦",
      type: "normal_gameplay",
      scene: "battle",
      sceneLabel: "戦闘",
      video: "vid_f4c223",
      videoLabel: "03-citadel.mp4",
      time: "00:18:42.900",
      progress: 0.918,
      utility: 0.863,
      base: 0.803,
      coverage: 0.1,
      spoilerPenalty: -0.04,
      temporalPenalty: 0,
      quality: 0.92,
      spoiler: "medium",
      spoilerEvidence: "固有bossと終盤固有areaが表示される。",
      context: "strong",
      summary: "終盤areaで固有bossと戦う通常play。",
      frameReason: "boss、player、battle HUDが同時に明瞭なframe。",
      decisionExplanation: "後半位置自体は減点せず、medium spoiler penalty後も高い品質と説明価値が残った。",
      colorA: "#7f1d1d",
      colorB: "#c2410c",
    },
  ],
  videos: [
    {
      id: "vid_a1e9c4",
      label: "01-opening.mp4",
      order: 1,
      duration: "00:58:04.200",
      selected: [1, 2],
      positions: [0.001, 0.217],
    },
    {
      id: "vid_c7bd10",
      label: "02-forest.mp4",
      order: 2,
      duration: "01:34:11.700",
      selected: [3, 4],
      positions: [0.48, 0.754],
    },
    {
      id: "vid_f4c223",
      label: "03-citadel.mp4",
      order: 3,
      duration: "01:12:02.520",
      selected: [5],
      positions: [0.26],
    },
  ],
  typeMix: [
    { label: "normal_gameplay", actual: 2, target: 4 },
    { label: "event", actual: 1, target: 2 },
    { label: "menu", actual: 1, target: 0 },
    { label: "title", actual: 1, target: 0 },
  ],
  nearMisses: [
    {
      id: "frm_10a83f09",
      label: "別のタイトル画面",
      utility: 0.824,
      type: "title",
      base: 0.774,
      coverage: 0.05,
      spoilerPenalty: 0,
      temporalPenalty: 0,
      reason: "title_limit",
    },
    {
      id: "frm_74e2150b",
      label: "遺跡探索の近似frame",
      utility: 0.902,
      type: "normal_gameplay",
      base: 0.802,
      coverage: 0.1,
      spoilerPenalty: 0,
      temporalPenalty: 0,
      reason: "visual_near_duplicate (0.997)",
    },
    {
      id: "frm_885c4d13",
      label: "同じboss戦の直後",
      utility: 0.858,
      type: "normal_gameplay",
      base: 0.798,
      coverage: 0.1,
      spoilerPenalty: -0.04,
      temporalPenalty: 0,
      reason: "similarity_ceiling (0.985 > 0.98)",
    },
  ],
  stages: [
    { name: "video_scan", fingerprint: "stg_4118…", cache: "3 / 3 hit", duration: "0.42s" },
    { name: "context_cues", fingerprint: "stg_dccc…", cache: "2 / 3 hit", duration: "4.81s" },
    { name: "scene_catalog", fingerprint: "stg_3112…", cache: "hit", duration: "0.03s" },
    { name: "candidate_annotation", fingerprint: "stg_7624…", cache: "38 / 42 hit", duration: "9.31s" },
    { name: "final_selection", fingerprint: "stg_d973…", cache: "miss", duration: "0.08s" },
  ],
};

const app = document.querySelector("#app");
const label = document.querySelector("#variant-label");

function selectedByIndex(index) {
  return run.selected.find((item) => item.index === index);
}

function header(kicker, title, description) {
  return `
    <header class="page-header">
      <div>
        <p class="eyebrow">${kicker}</p>
        <h1>${title}</h1>
        <p class="muted">${description}</p>
        <p class="muted mono">${run.id}</p>
      </div>
      <span class="status">warningあり</span>
    </header>`;
}

function stats() {
  return `
    <section class="stat-grid">
      <div class="stat"><strong>${run.selectedCount}/${run.requested}</strong><span>選択画像</span></div>
      <div class="stat"><strong>${run.videos.length}</strong><span>source videos</span></div>
      <div class="stat"><strong>${run.duration}</strong><span>Video Set duration</span></div>
      <div class="stat"><strong>${run.candidateMoments}</strong><span>Candidate Moments</span></div>
    </section>`;
}

function warning() {
  return `<aside class="warning-box"><span>⚠</span><div><strong>Selection Shortfall</strong>${run.warning}</div></aside>`;
}

function tags(item) {
  const spoilerClass = item.spoiler === "medium" ? "spoiler-medium" : "spoiler-low";
  return `
    <div class="tag-row">
      <span class="tag">${item.type}</span>
      <span class="tag">${item.sceneLabel}</span>
      <span class="tag ${spoilerClass}">spoiler: ${item.spoiler}</span>
      <span class="tag">utility ${item.utility.toFixed(3)}</span>
    </div>`;
}

function imageCard(item) {
  return `
    <article class="image-card">
      <div class="thumb" style="--thumb-a:${item.colorA};--thumb-b:${item.colorB}">
        <span>placeholder ${String(item.index).padStart(2, "0")}</span>
      </div>
      <div class="image-card-body">
        ${tags(item)}
        <h3>${String(item.index).padStart(2, "0")} — ${item.title}</h3>
        <p class="reason"><strong>画像の説明（モデル）</strong><br>${item.summary}</p>
        <p class="reason"><strong>代表frameの理由（モデル）</strong><br>${item.frameReason}</p>
        <p class="reason"><strong>採用理由（selector）</strong><br>${item.decisionExplanation}</p>
        ${item.spoilerEvidence ? `<details><summary>Spoiler evidence（モデル）</summary><p>${item.spoilerEvidence}</p></details>` : ""}
        <div class="meta-grid">
          <span>${item.videoLabel} · <span class="mono">${item.video}</span></span><span>${item.time}</span>
          <span class="mono">${item.id}</span><span>${item.file}</span>
        </div>
      </div>
    </article>`;
}

function typeMix() {
  return run.typeMix
    .map((item) => {
      const max = Math.max(item.target, item.actual, 1);
      const width = Math.round((item.actual / max) * 100);
      return `<div class="bar-row"><span>${item.label}</span><div class="bar-track"><div class="bar-fill" style="width:${width}%"></div></div><strong>${item.actual}/${item.target}</strong></div>`;
    })
    .join("");
}

function nearMisses() {
  return `<ul class="near-miss-list">${run.nearMisses
    .map(
      (item) => `<li><span><strong>${item.label}</strong><br><span class="muted mono">${item.id}</span></span><span><strong>${item.utility.toFixed(3)}</strong><br><span class="muted">${item.reason}</span></span></li>`,
    )
    .join("")}</ul>`;
}

function stageList() {
  return `<ul class="stage-list">${run.stages
    .map(
      (stage) => `<li><span class="mono">${stage.name}<br><span class="muted">${stage.fingerprint}</span></span><span><strong>${stage.cache}</strong><br><span class="muted">${stage.duration}</span></span></li>`,
    )
    .join("")}</ul>`;
}

function renderGallery() {
  app.innerHTML = `
    <div class="page">
      ${header("Variant A — Gallery first", "選ばれた画像から確認する", "ブログに使う画像と採用理由を主役にし、改善診断は後半へ置く構成。")}
      ${stats()}
      ${warning()}
      <section class="section">
        <div class="section-heading"><h2>選択画像</h2><a href="variant-a-gallery.md">Markdown sample</a></div>
        <div class="gallery">${run.selected.map(imageCard).join("")}</div>
      </section>
      <section class="section split">
        <div class="panel"><h2>Blog Image Type mix</h2>${typeMix()}</div>
        <div class="panel"><h2>Near misses</h2>${nearMisses()}</div>
      </section>
      <section class="section panel">
        <div class="section-heading"><h2>再現情報</h2><a href="report.sample.json">JSON sample</a></div>
        <p class="muted">model、schema、cache、stage timingは利用者の主作業を邪魔しないappendixとしてまとめる。</p>
        ${stageList()}
      </section>
    </div>`;
}

function videoSection(video) {
  const items = video.selected.map(selectedByIndex);
  const markers = items
    .map(
      (item, i) => `<div class="marker" style="left:${video.positions[i] * 100}%"><span>${item.time}</span></div>`,
    )
    .join("");
  return `
    <article class="video-section">
      <div class="video-title">
        <div><p class="eyebrow">Video ${video.order}</p><h2>${video.label}</h2><p class="muted mono">${video.id}</p></div>
        <strong>${video.duration}</strong>
      </div>
      <div class="timeline"><div class="timeline-track"></div>${markers}</div>
      <div class="timeline-items">${items
        .map(
          (item) => `<div class="timeline-item"><strong>${String(item.index).padStart(2, "0")} ${item.title}</strong>${item.time} · ${item.type} · spoiler ${item.spoiler}</div>`,
        )
        .join("")}</div>
    </article>`;
}

function renderTimeline() {
  app.innerHTML = `
    <div class="page">
      ${header("Variant B — Timeline first", "どの動画の、どの時点かを見る", "Video Orderとsource timeを主役にし、長い録画全体からのcoverageを確認する構成。")}
      ${stats()}
      ${warning()}
      <section class="section">
        <div class="section-heading"><h2>Video Set timeline</h2><a href="variant-b-timeline.md">Markdown sample</a></div>
        ${run.videos.map(videoSection).join("")}
      </section>
      <section class="section split">
        <div class="panel"><h2>全体進行の採用位置</h2><p class="muted">0%から100%までのVideo Set Progress</p><div class="timeline"><div class="timeline-track"></div>${run.selected
          .map((item) => `<div class="marker" style="left:${item.progress * 100}%"><span>${item.index}</span></div>`)
          .join("")}</div></div>
        <div class="panel"><h2>Near misses</h2>${nearMisses()}</div>
      </section>
    </div>`;
}

function funnel() {
  const values = [
    [run.candidateMoments, "Candidate Moments"],
    [42, "valid frames"],
    [run.annotated, "annotated"],
    [37, "hard-excluded"],
    [run.selectedCount, "selected"],
  ];
  return `<div class="funnel">${values
    .map(([value, text]) => `<div class="funnel-step"><strong>${value}</strong><span>${text}</span></div>`)
    .join("")}</div>`;
}

function auditRows() {
  const selected = run.selected.map((item) => ({
    id: item.id,
    decision: "selected",
    type: item.type,
    base: item.base,
    coverage: item.coverage,
    spoiler: item.spoilerPenalty,
    temporal: item.temporalPenalty,
    final: item.utility,
  }));
  const rejected = run.nearMisses.map((item) => ({
    id: item.id,
    decision: item.reason,
    type: item.type,
    base: item.base,
    coverage: item.coverage,
    spoiler: item.spoilerPenalty,
    temporal: item.temporalPenalty,
    final: item.utility,
  }));
  return [...selected, ...rejected]
    .map(
      (row) => `<tr><td class="mono">${row.id}</td><td class="${row.decision === "selected" ? "decision-selected" : "decision-rejected"}">${row.decision}</td><td>${row.type}</td><td>${row.base.toFixed(3)}</td><td>${row.coverage.toFixed(3)}</td><td>${row.spoiler.toFixed(3)}</td><td>${row.temporal.toFixed(3)}</td><td><strong>${row.final.toFixed(3)}</strong></td></tr>`,
    )
    .join("");
}

function renderAudit() {
  app.innerHTML = `
    <div class="page">
      ${header("Variant C — Audit first", "selectorの判断を監査する", "候補がどの段階で減り、どの加点・減点で採否が決まったかを主役にする構成。")}
      ${stats()}
      ${warning()}
      <section class="section panel">
        <div class="section-heading"><h2>Selection funnel</h2><a href="variant-c-audit.md">Markdown sample</a></div>
        ${funnel()}
      </section>
      <section class="section">
        <h2>Decision ledger</h2>
        <div class="audit-table-wrap"><table class="audit-table"><thead><tr><th>Frame ID</th><th>Decision</th><th>Type</th><th>Base</th><th>Coverage</th><th>Spoiler</th><th>Temporal</th><th>Marginal</th></tr></thead><tbody>${auditRows()}</tbody></table></div>
      </section>
      <section class="section split">
        <div class="panel"><h2>Stage provenance</h2>${stageList()}</div>
        <div class="panel"><h2>Public boundary</h2><div class="code-block">absolute_paths: omitted\nreasoning_trace: omitted\ngenerated_screen_quote: omitted\ncontext_cue_text: processing_cache_only\nrelative_source_path: included</div></div>
      </section>
    </div>`;
}

function currentVariant() {
  const value = new URLSearchParams(window.location.search).get("variant")?.toUpperCase();
  return variants.some((variant) => variant.key === value) ? value : "A";
}

function render() {
  const key = currentVariant();
  const variant = variants.find((item) => item.key === key);
  label.textContent = `${variant.key} — ${variant.name}`;
  if (key === "B") renderTimeline();
  else if (key === "C") renderAudit();
  else renderGallery();
  document.title = `${variant.key} — ${variant.name} — Report Prototype`;
}

function move(delta) {
  const index = variants.findIndex((variant) => variant.key === currentVariant());
  const next = variants[(index + delta + variants.length) % variants.length];
  const url = new URL(window.location.href);
  url.searchParams.set("variant", next.key);
  window.history.replaceState({}, "", url);
  render();
}

document.querySelector("#previous").addEventListener("click", () => move(-1));
document.querySelector("#next").addEventListener("click", () => move(1));
window.addEventListener("keydown", (event) => {
  const target = event.target;
  if (target instanceof HTMLElement && (target.matches("input, textarea") || target.isContentEditable)) return;
  if (event.key === "ArrowLeft") move(-1);
  if (event.key === "ArrowRight") move(1);
});
window.addEventListener("popstate", render);
render();
