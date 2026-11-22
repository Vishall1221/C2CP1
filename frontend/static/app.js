// Content to Commerce - web UI: video picker, live progress of the 5 steps, characters, outfits.
const STAGES = [
  ["scan", "1. Frame extraction & person tracking", "PyAV decode · shot-boundary detection · YOLO11 persons · ByteTrack · YuNet/SFace face embeddings"],
  ["cast", "2. Main-character clustering", "face-embedding clustering · ranked by screen time"],
  ["wardrobe", "3. Outfit segmentation & clustering", "clothing segmentation · DINOv2 + colour features · outfit grouping"],
  ["gemini", "4. Gemini requests (Vertex AI)", "one structured outfit description per request"],
  ["export", "5. Export JSON / CSV", "outfits.json · outfits.csv saved in output/"]];
let MAX_CHARS = 3;
const $ = s => document.querySelector(s);
const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;"})[c]);
const fmt = s => s == null ? "" : s < 60 ? `${s.toFixed(0)} s` : `${Math.floor(s / 60)} m ${String(Math.round(s % 60)).padStart(2, "0")} s`;
let key = null, es = null, clock = null;
const live = {};   // stage -> {status, started, ended, wall}

$("#stages").innerHTML = STAGES.map(([id, name, sub]) =>
  `<div class="name">${name}<small>${sub}</small></div><div class="st" id="st-${id}">waiting</div>
   <div><div class="bar"><i id="bar-${id}"></i></div><div class="msg" id="m-${id}"></div></div>
   <div class="time" id="t-${id}"></div>`).join("");

const api = (p, o) => fetch(p, o).then(async r => {
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText); return r.json(); });
const fileUrl = p => `/files/${key}/${p.split("/").map(encodeURIComponent).join("/")}`;

function setState(s) { const el = $("#state"); el.textContent = s; el.className = "pill " + s; }

function finish(id, ts) {
  const L = live[id]; if (!L || L.status !== "running") return;
  L.status = "done"; L.ended = ts;
  $("#st-" + id).textContent = "done"; $("#st-" + id).className = "st done";
  $("#bar-" + id).style.width = "100%";
}

function setStage(e) {
  const id = e.stage; if (!$("#st-" + id)) return;
  const order = STAGES.map(s => s[0]);
  for (const prev of order.slice(0, order.indexOf(id))) finish(prev, e.started_ts ?? e.ts);   // single-event stages (cast, export)
  const label = ({started: "running", running: "running", done: "done", skipped: "cached", failed: "failed"})[e.status] || e.status;
  const L = live[id] = live[id] || {};
  if (e.status === "started" || L.started == null) L.started = e.started_ts ?? e.ts;
  L.status = label; L.ended = label === "running" ? null : e.ts;
  const st = $("#st-" + id); st.textContent = label; st.className = "st " + label;
  const bar = $("#bar-" + id);
  if (e.total) bar.style.width = `${Math.min(100, 100 * e.done / e.total)}%`;
  if (label === "done" || label === "cached") bar.style.width = "100%";
  if (e.message && !["started", "done"].includes(e.message)) $("#m-" + id).textContent = e.message;
  tick();
}

function tick() {
  let total = 0;
  for (const [id] of STAGES) {
    const L = live[id]; if (!L || L.started == null) continue;
    const secs = L.wall ?? ((L.ended ?? Date.now() / 1000) - L.started);
    if (L.status !== "cached") total += secs;
    $("#t-" + id).textContent = L.status === "cached" ? "–" : fmt(secs);
  }
  $("#total").textContent = total ? `total ${fmt(total)}` : "";
}

function resetStages() {
  for (const [id] of STAGES) {
    delete live[id];
    $("#st-" + id).textContent = "waiting"; $("#st-" + id).className = "st";
    $("#bar-" + id).style.width = "0"; $("#m-" + id).textContent = ""; $("#t-" + id).textContent = "";
  }
  $("#err").textContent = ""; $("#total").textContent = "";
}

function showTimings(t) {
  if (!t || !Object.keys(t).length) { $("#details").textContent = "No timings yet."; return; }
  for (const [id] of STAGES) if (t[id] && live[id] && live[id].status !== "running") { live[id].wall = t[id].wall_s; }
  const lines = [];
  if (t._run) for (const [k, v] of Object.entries(t._run)) lines.push(`${k}: ${v}`);
  for (const [stage, v] of Object.entries(t)) {
    if (stage.startsWith("_")) continue;
    lines.push(`\n${stage}: ${v.wall_s} s wall, ${v.cpu_s} s cpu, ${v.peak_rss_mb} MB peak`);
    for (const [c, n] of Object.entries(v.counters || {})) lines.push(`  ${c}: ${n}`);
  }
  $("#details").textContent = lines.join("\n").trim();
  tick();
}

async function loadVideos(select) {
  const vids = await api("/api/videos");
  $("#video").innerHTML = vids.length
    ? vids.map(v => `<option value="${esc(v.name)}" data-key="${v.key}">${esc(v.name)} · ${v.size_mb} MB · ${v.state}</option>`).join("")
    : `<option value="">No videos in video/ — upload one</option>`;
  $("#run").disabled = !vids.length;
  if (select) $("#video").value = select;
  if (vids.length) await attach();
}

async function attach() {
  const opt = $("#video").selectedOptions[0]; if (!opt || !opt.dataset.key) return;
  key = opt.dataset.key; if (es) es.close();
  resetStages();
  $("#outfits").innerHTML = `<div class="empty">Appears when the run finishes.</div>`;
  $("#cast").innerHTML = `<div class="empty">Appears after the scan.</div>`;
  $("#dl").innerHTML = ""; $("#count").textContent = ""; $("#details").textContent = "No run yet.";
  let s;
  try { s = await api(`/api/runs/${key}`); }
  catch (_) { setState("new"); $("#run").textContent = "Run"; return; }
  setState(s.state);
  $("#run").textContent = {interrupted: "Resume", error: "Resume", done: "Run again", running: "Running…"}[s.state] || "Run";
  $("#run").disabled = s.state === "running";
  Object.values(s.stages).sort((a, b) => a.ts - b.ts).forEach(setStage);
  if (s.state !== "running") for (const [id] of STAGES) finish(id, live[id] && (s.stages[id] || {}).ts);
  showTimings(s.timings);
  if (s.output_dir) $("#saved").textContent = `Saved in ${s.output_dir}`;
  await showResult();
  if (s.state === "running") listen();
}

function listen() {
  if (es) es.close();
  clearInterval(clock); clock = setInterval(tick, 1000);
  $("#run").disabled = true; $("#run").textContent = "Running…"; setState("running");
  es = new EventSource(`/api/runs/${key}/events`);
  es.onmessage = m => {
    const e = JSON.parse(m.data); setStage(e);
    if (e.stage === "pipeline" && e.status === "error") $("#err").textContent = e.message;
    if (["cast", "wardrobe", "gemini", "export"].includes(e.stage) && e.status !== "running") showResult();
  };
  es.addEventListener("end", async () => {
    es.close(); clearInterval(clock);
    for (const [id] of STAGES) finish(id, Date.now() / 1000);
    const v = $("#video").value; await loadVideos(v);
  });
}

function garmentTags(o) {
  return (o.garments || []).map(g => {
    const bits = [g.type, (g.colors || []).join("/"), g.pattern, g.fabric].filter(Boolean);
    return `<span class="tag">${esc(bits.join(" · "))}</span>`;
  }).join("");
}

async function showResult() {
  let r; try { r = await api(`/api/runs/${key}/result`); } catch (_) { return; }
  if (r.cast.length)
    $("#cast").innerHTML = r.cast.map(c => `<div class="person">
        <img src="${c.portraits[0] ? fileUrl(c.portraits[0]) : ""}" alt="">
        <div><b>Character ${c.character_id}</b><span class="muted">${fmt(c.screen_time_s)} on screen · ${c.shots} shots</span>
        <div class="muted small">${esc(c.cast_id)}</div></div></div>`).join("");

  const by = {}; r.outfits.forEach(o => (by[o.character_id] = by[o.character_id] || []).push(o));
  if (r.outfits.length) {
    $("#count").textContent = `${r.outfits.length} outfits`;
    $("#outfits").innerHTML = Object.keys(by).sort((a, b) => a - b).map(ch =>
      `<h3>Character ${ch} <span class="muted small">· ${by[ch].length} outfits</span></h3>` +
      by[ch].map(o => `<div class="outfit">
          <a href="${fileUrl(o.image)}" target="_blank"><img loading="lazy" src="${fileUrl(o.image)}" alt=""></a>
          <div><div class="meta"><span class="ts">${esc(o.timestamp)}</span>
            <span class="muted small">${o.appearances} appearance${o.appearances === 1 ? "" : "s"}</span></div>
            ${o.status === "ok" ? `<div>${esc(o.description)}</div><div class="tags">${garmentTags(o)}</div>`
                                : `<div class="bad">${esc(o.status)}${o.error ? ": " + esc(o.error) : ""}</div>`}
          </div></div>`).join("")).join("");
  }
  if (r.outfits.some(o => o.status === "ok"))
    $("#dl").innerHTML = `<a href="${fileUrl("out/outfits.json")}" download>outfits.json</a>
                          <a href="${fileUrl("out/outfits.csv")}" download>outfits.csv</a>` +
      (r.complete ? "" : `<span class="muted small">some outfits pending — press Resume</span>`);

  const c = await api(`/api/runs/${key}/candidates`).catch(() => []);
  const chosen = new Set(r.cast.map(x => x.cast_id));
  $("#cands").innerHTML = c.map(x => `<label class="person">
      <input type="checkbox" value="${esc(x.cast_id)}" ${chosen.has(x.cast_id) ? "checked" : ""}>
      <img src="${x.portraits[0] ? fileUrl(x.portraits[0]) : ""}" alt="">
      <div><b>${esc(x.cast_id)}</b><span class="muted small">${fmt(x.screen_time_s)} · ${x.shots} shots</span></div></label>`).join("");
}

async function start(castIds) {
  $("#err").textContent = "";
  try {
    const body = {video: $("#video").value};
    if (castIds) body.cast_ids = castIds;
    const r = await api("/api/runs", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)});
    key = r.key; resetStages(); listen();
  } catch (e) { $("#err").textContent = e.message; }
}

$("#run").onclick = () => start();
$("#video").onchange = attach;
$("#recast").onclick = () => {
  const ids = [...document.querySelectorAll("#cands input:checked")].map(i => i.value);
  if (!ids.length || ids.length > MAX_CHARS) { $("#err").textContent = `Choose 1 to ${MAX_CHARS} people.`; return; }
  start(ids);
};
$("#file").onchange = () => {
  const f = $("#file").files[0]; if (!f) return;
  const x = new XMLHttpRequest();
  x.open("PUT", "/api/upload?name=" + encodeURIComponent(f.name));
  x.upload.onprogress = e => { $("#up").textContent = `Uploading ${f.name}: ${(100 * e.loaded / e.total).toFixed(0)}%`; };
  x.onload = async () => {
    if (x.status !== 200) { $("#err").textContent = "Upload failed: " + x.responseText; return; }
    $("#up").textContent = `Uploaded ${f.name}. Starting…`;
    await loadVideos(f.name); start();
  };
  x.send(f);
};
api("/api/config").then(c => {
  MAX_CHARS = c.main_characters;
  $("#cfg").textContent = `${c.main_characters} main characters · ${c.sample_fps} fps · ${c.model} (${c.location})`;
  $("#pickhint").textContent = `Pick 1–${c.main_characters}. Only the outfit and Gemini steps run again.`;
}).catch(e => { $("#err").textContent = "config.env: " + e.message; });
loadVideos();
