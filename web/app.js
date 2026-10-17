/* ShelfSense Edge — dashboard client (tanpa dependensi eksternal) */

const $ = (s) => document.querySelector(s);
const video = $("#video");
const videoBg = $("#videoBg");
const overlay = $("#overlay");
const chart = $("#chart");
const scanZone = $("#scanZone");
const camEmpty = $("#camEmpty");
const lockChip = $("#lockChip");
const resultPanel = $("#resultPanel");

let ws = null;
let last = null;
let lastFrameSize = [1280, 720];
let basePrice = 2000;
let chartDirty = false;
let lastChartDraw = 0;

const SIGNAL_LABELS = { mold: "Jamur", bruise: "Memar", anomaly: "Permukaan", wrinkle: "Keriput", dullness: "Kusam" };

/* ---------------- koneksi ---------------- */
function connect() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${proto}://${location.host}/ws`);

  ws.onopen = () => {
    setCamStatus(true, "kamera aktif");
  };
  ws.onclose = () => {
    setCamStatus(false, "terputus — menyambung ulang");
    setTimeout(connect, 1200);
  };
  ws.onerror = () => setCamStatus(false, "error koneksi");
  ws.onmessage = (ev) => {
    let msg;
    try { msg = JSON.parse(ev.data); } catch { return; }
    if (msg.type !== "frame") return;
    if (msg.image) {
      const src = `data:image/jpeg;base64,${msg.image}`;
      video.src = src;
      videoBg.src = src;
      camEmpty.classList.add("hidden");
    }
    if (Array.isArray(msg.frame)) lastFrameSize = msg.frame;
    last = msg.state || {};
    render(last);
  };
}

function setCamStatus(ok, text) {
  const dot = $("#camDot");
  dot.classList.toggle("ok", ok);
  dot.classList.toggle("err", !ok);
  $("#camText").textContent = text;
}

/* ---------------- render ---------------- */
function render(s) {
  if (!s) return;

  $("#fpsText").textContent = `${Math.round(s.fps || 0)} fps`;
  $("#engineText").textContent =
    s.detector === "yolo" ? "YOLO" : s.detector === "color" ? "Color-blob" : s.detector === "roi" ? "ROI fallback" : "—";
  scanZone.classList.toggle("active", s.detector === "roi");

  if (!s.ok) {
    resultPanel.style.setProperty("--grade", "#b9a892");
    $("#condNum").textContent = "—";
    $("#bigValue").classList.add("empty");
    $("#gradePill").textContent = "—";
    $("#gradeLabel").textContent = s.error || "";
    lockChip.textContent = "Menunggu objek";
    lockChip.className = "badge";
    camEmpty.innerHTML = "<strong></strong><span>Letakkan tomat di tengah bingkai</span>";
    camEmpty.querySelector("strong").textContent = s.error || "Arahkan kamera ke tomat";
    camEmpty.classList.remove("hidden");
    drawOverlay(null, s.color);
    return;
  }

  camEmpty.classList.add("hidden");
  $("#bigValue").classList.remove("empty");

  const color = s.color || "#34d399";
  resultPanel.style.setProperty("--grade", color);
  resultPanel.style.setProperty("--grade-soft", hexA(color, 0.18));

  $("#condNum").textContent = Math.round(s.condition_pct);
  $("#gradePill").textContent = s.grade;
  $("#gradeLabel").textContent = s.grade_label || "";

  $("#discVal").textContent = s.hold ? "—" : `${s.discount}%`;
  const priceEl = $("#priceVal");
  const priceText = `Rp ${fmt(s.price)}`;
  if (priceEl.textContent !== priceText) {
    priceEl.textContent = priceText;
    priceEl.classList.remove("flash");
    void priceEl.offsetWidth;
    priceEl.classList.add("flash");
  }
  $("#baseVal").textContent = s.hold ? "" : `dari Rp ${fmt(s.base_price)}`;
  const saved = s.base_price - s.price;
  $("#saveVal").textContent = !s.hold && saved > 0 ? `Hemat Rp ${fmt(saved)}` : "";

  $("#matBar").style.width = `${Math.min(100, s.maturity_pct)}%`;
  $("#defBar").style.width = `${Math.min(100, s.defect_pct)}%`;
  $("#matVal").textContent = `${Math.round(s.maturity_pct)}%`;
  $("#defVal").textContent = `${Math.round(s.defect_pct)}%`;

  renderSignals(s.signal_contrib || {});
  renderNextBand(s);
  renderTimeline(s);

  $("#remainVal").textContent = fmtHours(s.remaining_hours);
  $("#remainVal").classList.toggle("warn-text", s.remaining_hours < 12);
  $("#confVal").textContent = `${Math.round((s.confidence || 0) * 100)}%`;
  $("#stageVal").textContent = s.stage_label || "—";

  if (s.locked) {
    lockChip.textContent = "Terbaca";
    lockChip.className = "badge locked";
  } else if (s.needs_verify) {
    lockChip.textContent = "Perlu dicek";
    lockChip.className = "badge warn";
  } else {
    lockChip.textContent = "Memindai…";
    lockChip.className = "badge";
  }

  // tab consumer mengikuti harga live
  $("#notifText").textContent =
    s.hold
      ? `${s.profile.display_name} belum matang — ditahan dari rak`
      : `${s.profile.display_name} Grade ${s.grade} — diskon ${s.discount}%`;
  $("#miniTitle").textContent = `${s.profile.display_name} · ${s.grade}`;
  $("#miniPrice").textContent = `Rp ${fmt(s.price)}`;

  drawOverlay(s.bbox, color);

  const now = performance.now();
  if (now - lastChartDraw > 180) {
    lastChartDraw = now;
    drawChart(s);
  }
}

function renderSignals(contrib) {
  const ul = $("#signals");
  const keys = Object.keys(SIGNAL_LABELS);
  if (ul.dataset.built !== "1") {
    ul.innerHTML = keys
      .map(
        (k) => `<li>
          <span>${SIGNAL_LABELS[k]}</span>
          <span class="sbar"><i data-k="${k}"></i></span>
          <span class="sval" data-v="${k}">0%</span>
        </li>`
      )
      .join("");
    ul.dataset.built = "1";
  }
  keys.forEach((k) => {
    const v = contrib[k] ?? 0;
    const bar = ul.querySelector(`i[data-k="${k}"]`);
    const val = ul.querySelector(`span[data-v="${k}"]`);
    if (bar) bar.style.width = `${Math.min(100, v * 3)}%`;
    if (val) val.textContent = `${v.toFixed(0)}%`;
  });
}

function renderNextBand(s) {
  const el = $("#nextBand");
  if (s.hold) {
    el.innerHTML = `<b>HOLD</b> — kirim ke ruang pematangan, bukan didiskon.`;
    return;
  }
  const nb = s.next_band;
  if (!nb) {
    el.innerHTML = `<b>Diskon maksimum</b> sudah tercapai untuk komoditas ini.`;
    return;
  }
  el.innerHTML =
    `Turun ke <b>${nb.min_condition}%</b> → diskon naik ke <b>${nb.discount}%</b>` +
    ` <span style="opacity:.6">(sisa ${nb.remaining_to_drop} poin)</span>`;
}

function renderTimeline(s) {
  const el = $("#timeline");
  const gt = s.grade_times || {};
  const colors = {};
  (s.profile.grades || []).forEach((g) => (colors[g.grade] = g.color));
  const items = ["B", "C", "D"]
    .filter((g) => gt[g] !== undefined)
    .map((g) => {
      const t = gt[g];
      const label = t >= 24 ? `${(t / 24).toFixed(1)} hari` : `${Math.round(t)} jam`;
      return `<span class="tl-item" style="--c:${colors[g] || "#888"}"><b>${g}</b>+${label}</span>`;
    });
  el.innerHTML = items.length
    ? `<span class="tl-label">Turun grade</span>${items.join("")}`
    : "";
}

/* ---------------- overlay bbox ---------------- */
function drawOverlay(bbox, color = "#34d399") {
  const dpr = window.devicePixelRatio || 1;
  const rect = overlay.getBoundingClientRect();
  if (rect.width === 0) return;
  if (overlay.width !== Math.round(rect.width * dpr)) {
    overlay.width = Math.round(rect.width * dpr);
    overlay.height = Math.round(rect.height * dpr);
  }
  const ctx = overlay.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, rect.width, rect.height);
  if (!bbox) return;

  const [fw, fh] = lastFrameSize;
  const scale = Math.min(rect.width / fw, rect.height / fh);
  const dw = fw * scale;
  const dh = fh * scale;
  const ox = (rect.width - dw) / 2;
  const oy = (rect.height - dh) / 2;

  const [x, y, w, h] = bbox;
  const X = ox + x * scale;
  const Y = oy + y * scale;
  const W = w * scale;
  const H = h * scale;

  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.shadowColor = color;
  ctx.shadowBlur = 14;
  roundRect(ctx, X, Y, W, H, 10);
  ctx.stroke();

  // sudut tebal
  const c = Math.min(22, W * 0.18, H * 0.18);
  ctx.lineWidth = 4;
  ctx.shadowBlur = 0;
  corner(ctx, X, Y, c, 1, 1);
  corner(ctx, X + W, Y, c, -1, 1);
  corner(ctx, X, Y + H, c, 1, -1);
  corner(ctx, X + W, Y + H, c, -1, -1);

  ctx.restore();
}

function corner(ctx, x, y, len, dx, dy) {
  ctx.beginPath();
  ctx.moveTo(x + dx * len, y);
  ctx.lineTo(x, y);
  ctx.lineTo(x, y + dy * len);
  ctx.stroke();
}

function roundRect(ctx, x, y, w, h, r) {
  const rr = Math.min(r, w / 2, h / 2);
  ctx.beginPath();
  ctx.moveTo(x + rr, y);
  ctx.arcTo(x + w, y, x + w, y + h, rr);
  ctx.arcTo(x + w, y + h, x, y + h, rr);
  ctx.arcTo(x, y + h, x, y, rr);
  ctx.arcTo(x, y, x + w, y, rr);
  ctx.closePath();
}

/* ---------------- chart ---------------- */
function drawChart(s) {
  const dpr = window.devicePixelRatio || 1;
  const rect = chart.getBoundingClientRect();
  if (rect.width < 10) return;
  if (chart.width !== Math.round(rect.width * dpr)) {
    chart.width = Math.round(rect.width * dpr);
    chart.height = Math.round(rect.height * dpr);
  }
  const ctx = chart.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, rect.width, rect.height);

  const traj = s.trajectory || [];
  if (traj.length < 2) return;

  const padL = 34, padR = 40, padT = 12, padB = 22;
  const W = rect.width - padL - padR;
  const H = rect.height - padT - padB;
  const xOf = (t) => padL + (t / 72) * W;
  const yOf = (c) => padT + (1 - c) * H;

  const grades = (s.profile.grades || []).slice().sort((a, b) => b.min_condition - a.min_condition);
  const prime = grades.find((g) => g.grade === "B");
  const primeThr = (prime ? prime.min_condition : 70) / 100;

  // grid horizontal + label grade
  ctx.font = "9px ui-monospace, monospace";
  ctx.textBaseline = "middle";
  grades.forEach((g) => {
    if (g.min_condition <= 0) return;
    const y = yOf(g.min_condition / 100);
    ctx.strokeStyle = "rgba(243,236,225,.09)";
    ctx.setLineDash([3, 4]);
    ctx.beginPath();
    ctx.moveTo(padL, y);
    ctx.lineTo(padL + W, y);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = hexA(g.color, 0.9);
    ctx.textAlign = "right";
    ctx.fillText(`${g.grade} ${g.min_condition}`, padL - 6, y);
  });

  // prime window: rentang waktu kualitas masih di atas ambang grade B
  let t0 = null;
  let t1 = null;
  traj.forEach((p) => {
    if (p.condition >= primeThr) {
      if (t0 === null) t0 = p.t;
      t1 = p.t;
    }
  });
  if (t0 !== null) {
    ctx.fillStyle = "rgba(197,138,46,.12)";
    ctx.fillRect(xOf(t0), padT, Math.max(1, xOf(t1) - xOf(t0)), H);
    ctx.fillStyle = "rgba(197,138,46,.80)";
    ctx.textAlign = "left";
    ctx.font = "9px ui-monospace, monospace";
    ctx.fillText("prime window", xOf(t0) + 6, padT + 10);
  }

  // garis trajektori
  ctx.beginPath();
  traj.forEach((p, i) => {
    const x = xOf(p.t), y = yOf(p.condition);
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  });
  ctx.strokeStyle = "#f3ece1";
  ctx.lineWidth = 2;
  ctx.lineJoin = "round";
  ctx.stroke();

  // titik sekarang
  const off = s.config ? s.config.time_offset_h : 0;
  let cur = traj[0];
  for (const p of traj) if (p.t <= off + 0.01) cur = p;
  const cx = xOf(cur.t), cy = yOf(cur.condition);
  ctx.strokeStyle = "rgba(243,236,225,.28)";
  ctx.setLineDash([2, 3]);
  ctx.beginPath();
  ctx.moveTo(cx, padT);
  ctx.lineTo(cx, padT + H);
  ctx.stroke();
  ctx.setLineDash([]);

  ctx.beginPath();
  ctx.arc(cx, cy, 4.5, 0, Math.PI * 2);
  ctx.fillStyle = s.color || "#34d399";
  ctx.shadowColor = s.color || "#34d399";
  ctx.shadowBlur = 12;
  ctx.fill();
  ctx.shadowBlur = 0;

  ctx.fillStyle = "#f3ece1";
  ctx.textAlign = "left";
  ctx.font = "10px ui-monospace, monospace";
  ctx.fillText(`${Math.round(cur.condition * 100)}%`, cx + 8, cy - 9);

  // sumbu x
  ctx.fillStyle = "#6d6359";
  ctx.font = "9px ui-monospace, monospace";
  ctx.textAlign = "center";
  [0, 24, 48, 72].forEach((t) => {
    ctx.fillText(t === 0 ? "sekarang" : `+${t}j`, xOf(t), padT + H + 11);
  });

  // garis dasar
  ctx.strokeStyle = "rgba(243,236,225,.12)";
  ctx.beginPath();
  ctx.moveTo(padL, padT + H);
  ctx.lineTo(padL + W, padT + H);
  ctx.stroke();
}

/* ---------------- helpers ---------------- */
function fmt(n) {
  return new Intl.NumberFormat("id-ID").format(Math.round(n || 0));
}
function fmtHours(h) {
  if (h == null) return "—";
  if (h >= 72) return `${(h / 24).toFixed(1)} hari`;
  return `${Math.round(h)} jam`;
}
function hexA(hex, a) {
  const h = hex.replace("#", "");
  const n = parseInt(h.length === 3 ? h.split("").map((c) => c + c).join("") : h, 16);
  return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
}
function toast(msg) {
  const el = $("#toast");
  el.textContent = msg;
  el.classList.add("show");
  clearTimeout(el._t);
  el._t = setTimeout(() => el.classList.remove("show"), 2200);
}
function sleep(ms) {
  return new Promise((res) => setTimeout(res, ms));
}

/* ---------------- kontrol ---------------- */
let envTimer = null;
function pushEnv(patch) {
  clearTimeout(envTimer);
  envTimer = setTimeout(async () => {
    try {
      await fetch("/api/env", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(patch),
      });
    } catch { /* diamkan */ }
  }, 140);
}

const tempEl = $("#temp");
const timeEl = $("#time");

tempEl.addEventListener("input", () => {
  const v = Number(tempEl.value);
  $("#tempVal").textContent = `${v}°C`;
  $("#chillNote").classList.toggle("warn", v < 12);
  pushEnv({ temp_c: v });
});

timeEl.addEventListener("input", () => {
  const v = Number(timeEl.value);
  $("#timeVal").textContent = v === 0 ? "sekarang" : `+${v} jam`;
  pushEnv({ time_offset_h: v });
});

$("#calibrateBtn").addEventListener("click", async () => {
  const btn = $("#calibrateBtn");
  btn.disabled = true;
  toast("Arahkan kertas putih ke tengah layar…");
  await sleep(2500);
  try {
    const r = await fetch("/api/calibrate", { method: "POST" });
    const j = await r.json();
    if (j.ok) {
      toast(`Kalibrasi selesai — gain ${j.gains.join(" / ")}`);
      const cal = $("#calText");
      cal.textContent = "✓ terkalibrasi";
      cal.classList.add("ok");
    } else {
      toast(j.error || "Kalibrasi gagal");
    }
  } catch {
    toast("Kalibrasi gagal");
  } finally {
    btn.disabled = false;
  }
});

document.querySelectorAll(".profile-chip").forEach((btn) => {
  btn.addEventListener("click", async () => {
    document.querySelectorAll(".profile-chip").forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    try {
      const r = await fetch("/api/profile", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ id: btn.dataset.id }),
      });
      const j = await r.json();
      if (j.ok) {
        basePrice = j.profile.base_price;
        toast(`Profil diganti: ${j.profile.display_name}`);
      }
    } catch {
      toast("Gagal mengganti profil");
    }
  });
});

document.querySelectorAll(".tab-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    document.querySelectorAll(".tab-body").forEach((b) => b.classList.add("hidden"));
    $(`#tab-${btn.dataset.tab}`).classList.remove("hidden");
  });
});

/* ---------------- boot ---------------- */
fetch("/api/status")
  .then((r) => r.json())
  .then((s) => {
    setCamStatus(!!s.camera, s.camera ? "kamera aktif" : s.error || "kamera tidak aktif");
    $("#engineText").textContent = s.detector || "—";
    if (s.calibrated) {
      const cal = $("#calText");
      cal.textContent = "✓ terkalibrasi";
      cal.classList.add("ok");
    }
    if (s.profiles) {
      document.querySelectorAll(".profile-chip").forEach((b) => {
        b.classList.toggle("active", b.dataset.id === s.profile);
      });
    }
  })
  .catch(() => {});

/* ---------------- mode demo (tanpa kamera) ---------------- */
function drawDemoFrame() {
  const c = document.createElement("canvas");
  c.width = 1280;
  c.height = 720;
  const x = c.getContext("2d");
  const bg = x.createLinearGradient(0, 0, 0, 720);
  bg.addColorStop(0, "#1d1813");
  bg.addColorStop(1, "#120f0c");
  x.fillStyle = bg;
  x.fillRect(0, 0, 1280, 720);
  x.fillStyle = "#2a231c";
  x.fillRect(0, 470, 1280, 250);

  const g = x.createRadialGradient(590, 300, 30, 640, 360, 205);
  g.addColorStop(0, "#e8634f");
  g.addColorStop(0.7, "#cf3a2b");
  g.addColorStop(1, "#a8241a");
  x.fillStyle = g;
  x.beginPath();
  x.arc(640, 360, 190, 0, Math.PI * 2);
  x.fill();

  x.fillStyle = "rgba(255,255,255,.30)";
  x.beginPath();
  x.arc(578, 286, 48, 0, Math.PI * 2);
  x.fill();

  x.fillStyle = "#5f241b";
  x.beginPath();
  x.arc(712, 408, 56, 0, Math.PI * 2);
  x.fill();

  x.fillStyle = "rgba(210,210,200,.85)";
  x.beginPath();
  x.arc(700, 500, 30, 0, Math.PI * 2);
  x.fill();

  return c.toDataURL("image/jpeg", 0.9);
}

function buildDemoState(profile) {
  const traj = [];
  for (let t = 0; t <= 72; t += 2) {
    const c = 0.764 - 0.0018 * t - 0.000012 * t * t;
    traj.push({ t, condition: Math.max(0.2, c), maturity: 1, defect: 1 - c });
  }
  return {
    ok: true,
    camera: true,
    locked: true,
    needs_verify: false,
    confidence: 0.81,
    condition_pct: 76.4,
    maturity_pct: 100,
    defect_pct: 23.6,
    stage: "red_ripe",
    stage_label: "Merah matang",
    signals: { mold: 0.05, bruise: 0.42, wrinkle: 0.0, dullness: 0.12 },
    signal_contrib: { mold: 1.7, bruise: 10.5, wrinkle: 0.0, dullness: 2.4 },
    grade: "B",
    grade_label: "Jual cepat",
    color: "#c08a2e",
    discount: 15,
    price: 1700,
    base_price: 2000,
    hold: false,
    reason: "kondisi 76%",
    remaining_hours: 45,
    next_band: { grade: "C", discount: 30, min_condition: 50, remaining_to_drop: 26.4, progress: 1 },
    grade_times: { A: 0, B: 19.5, C: 45 },
    trajectory: traj,
    config: { temp_c: 24, rh: 88, time_offset_h: 0 },
    profile,
    detector: "demo",
    bbox: [450, 170, 380, 380],
    fps: 12,
  };
}

if (location.search.includes("demo")) {
  fetch("/api/profiles")
    .then((r) => r.json())
    .then((j) => {
      const p = (j.profiles || []).find((x) => x.id === "tomato") || (j.profiles || [])[0];
      lastFrameSize = [1280, 720];
      video.src = drawDemoFrame();
      videoBg.src = video.src;
      const st = buildDemoState(p);
      last = st;
      render(st);
      setCamStatus(true, "kamera aktif");
      $("#engineText").textContent = "DEMO";
      camEmpty.classList.add("hidden");
    })
    .catch(() => {});
} else {
  connect();
}
window.addEventListener("resize", () => { if (last) drawChart(last); });
