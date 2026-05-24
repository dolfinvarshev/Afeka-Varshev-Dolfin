/* script.js - CNN forward pass + full visualization */

var model = null;
var CW = 1440,
  CH = 5000;
var animTimers = [];

/* ── Init ─────────────────────────────────────────────── */
function setModelStatus(m, ok) {
  var e = document.getElementById("model-status");
  if (e) {
    e.innerHTML = m;
    e.className = ok ? "status-ok" : "status-error";
  }
}

window.onload = function () {
  try {
    var stored = loadModelFromStorage();
    if (stored) {
      model = stored;
      setModelStatus("Model loaded from localStorage. Ready to test a new image.", true);
    } else {
      saveModelToStorage(MODEL_JSON);
      model = MODEL_JSON;
      setModelStatus("Model saved to localStorage. Ready to test a new image.", true);
    }
  } catch (e) {
    model = MODEL_JSON;
    setModelStatus("Model ready.", true);
  }
  try {
    draw(null);
  } catch (drawErr) {
    setModelStatus("Draw error: " + drawErr.message, false);
  }
};

/* ══════════════════ IMAGE UPLOAD ══════════════════════ */
function onImageChange(el) {
  if (!el.files[0]) return;
  var rd = new FileReader();
  rd.onload = function (e) {
    var p = document.getElementById("img-preview");
    p.src = e.target.result;
    p.style.display = "block";
    draw(null);
  };
  rd.readAsDataURL(el.files[0]);
}

/* ══════════════════ CNN OPS ═══════════════════════════ */
/* input [C,H,W], filters [F,C,kH,kW] → output [F,H',W'] */
function conv2d(input, filters, biases, padding) {
  var inC = input.length,
    inH = input[0].length,
    inW = input[0][0].length;
  var F = filters.length,
    kH = filters[0][0].length,
    kW = filters[0][0][0].length;
  var pad = padding === "same" ? Math.floor(kH / 2) : 0;
  var oH = inH + 2 * pad - kH + 1,
    oW = inW + 2 * pad - kW + 1;
  /* pad input - for loop replaces .map() */
  var P = [];
  for (var c = 0; c < inC; c++) {
    var pc = [];
    for (var i = 0; i < inH + 2 * pad; i++) {
      var row = [];
      for (var j = 0; j < inW + 2 * pad; j++)
        row.push(
          i >= pad && i < inH + pad && j >= pad && j < inW + pad
            ? input[c][i - pad][j - pad]
            : 0,
        );
      pc.push(row);
    }
    P.push(pc);
  }
  var out = [];
  for (var f = 0; f < F; f++) {
    var fm = [];
    for (var i = 0; i < oH; i++) {
      var row = [];
      for (var j = 0; j < oW; j++) {
        var s = biases[f];
        for (var c = 0; c < inC; c++)
          for (var ki = 0; ki < kH; ki++)
            for (var kj = 0; kj < kW; kj++)
              s += P[c][i + ki][j + kj] * filters[f][c][ki][kj];
        row.push(s);
      }
      fm.push(row);
    }
    out.push(fm);
  }
  return out;
}
function relu3d(g) {
  var out = [];
  for (var f = 0; f < g.length; f++) {
    var fm = [];
    for (var i = 0; i < g[f].length; i++) {
      var row = [];
      for (var j = 0; j < g[f][i].length; j++)
        row.push(g[f][i][j] > 0 ? g[f][i][j] : 0);
      fm.push(row);
    }
    out.push(fm);
  }
  return out;
}
function pool2d(g) {
  var out = [];
  for (var f = 0; f < g.length; f++) {
    var fm = g[f],
      H = fm.length,
      W = fm[0].length,
      pfm = [];
    for (var i = 0; i < Math.floor(H / 2); i++) {
      var row = [];
      for (var j = 0; j < Math.floor(W / 2); j++) {
        var mx = Math.max(
          fm[i * 2][j * 2],
          fm[i * 2][j * 2 + 1],
          fm[i * 2 + 1][j * 2],
          fm[i * 2 + 1][j * 2 + 1],
        );
        row.push(mx);
      }
      pfm.push(row);
    }
    out.push(pfm);
  }
  return out;
}
function flatten(g) {
  /* Matches Keras channels-last order: for h, for w, for c */
  var C = g.length,
    H = g[0].length,
    W = g[0][0].length,
    o = [];
  for (var h = 0; h < H; h++)
    for (var w = 0; w < W; w++) for (var c = 0; c < C; c++) o.push(g[c][h][w]);
  return o;
}
function dense(inp, W, b, relu) {
  var out = [];
  for (var n = 0; n < W.length; n++) {
    var z = b[n];
    for (var i = 0; i < inp.length; i++) z += W[n][i] * inp[i];
    out.push(relu && z < 0 ? 0 : z);
  }
  return out;
}
function softmax(arr) {
  var mx = arr[0];
  for (var i = 1; i < arr.length; i++) if (arr[i] > mx) mx = arr[i];
  var ex = [],
    s = 0;
  for (var i = 0; i < arr.length; i++) {
    var e = Math.exp(arr[i] - mx);
    ex.push(e);
    s += e;
  }
  var out = [];
  for (var i = 0; i < ex.length; i++) out.push(ex[i] / s);
  return out;
}
function forward(pgrid) {
  var L = model.layers;
  var c1 = relu3d(conv2d(pgrid, L[0].filters, L[0].biases, "same"));
  var p1 = pool2d(c1);
  var c2 = relu3d(conv2d(p1, L[1].filters, L[1].biases, "same"));
  var p2 = pool2d(c2);
  var fl = flatten(p2);
  var d1 = dense(fl, L[2].weights, L[2].biases, true);
  var raw = dense(d1, L[3].weights, L[3].biases, false);
  return {
    pgrid: pgrid,
    c1: c1,
    p1: p1,
    c2: c2,
    p2: p2,
    fl: fl,
    d1: d1,
    probs: softmax(raw),
  };
}

/* Read image → [1,8,8] pixel tensor with contrast normalisation.
   Normalisation ensures that even low-contrast or thin-line drawings
   produce a full 0–1 range, fixing the "always predicts Square" problem
   that occurs when inputs are all near zero. */
function readPixels(src, cb) {
  var hc = document.getElementById("hidden-canvas");
  hc.width = 8;
  hc.height = 8;
  var ctx = hc.getContext("2d"),
    tmp = new Image();
  tmp.onload = function () {
    ctx.drawImage(tmp, 0, 0, 8, 8);
    var d = ctx.getImageData(0, 0, 8, 8).data;
    var raw = [];
    for (var i = 0; i < 64; i++) {
      var k = i * 4;
      /* NO inversion - Colab trains with white=1.0, black=0.0, so we must match */
      raw.push((d[k] + d[k + 1] + d[k + 2]) / (3 * 255));
    }
    /* contrast normalisation: stretch so any image fills [0,1]
       (safe for pure black/white: min=0,max=1 → unchanged) */
    var mn = raw[0],
      mx = raw[0];
    for (var i = 1; i < 64; i++) {
      if (raw[i] < mn) mn = raw[i];
      if (raw[i] > mx) mx = raw[i];
    }
    var rng = mx - mn;
    if (rng > 0.05) {
      for (var i = 0; i < 64; i++) raw[i] = (raw[i] - mn) / rng;
    }
    var g = [[]];
    for (var i = 0; i < 8; i++) {
      g[0].push([]);
      for (var j = 0; j < 8; j++) g[0][i].push(raw[i * 8 + j]);
    }
    cb(g);
  };
  tmp.src = src;
}

/* ══════════════════ BUTTONS ═══════════════════════════ */
function onTest() {
  for (var t = 0; t < animTimers.length; t++) clearTimeout(animTimers[t]);
  animTimers = [];
  var prev = document.getElementById("img-preview"),
    res = document.getElementById("result-text");
  if (!prev.src || prev.style.display === "none") {
    alert("Please upload an image first.");
    return;
  }
  res.innerHTML = "Running CNN…";
  res.className = "";
  readPixels(prev.src, function (pgrid) {
    var r = forward(pgrid);
    var wi = 0;
    for (var i = 1; i < r.probs.length; i++)
      if (r.probs[i] > r.probs[wi]) wi = i;
    var names = ["Triangle", "Heart", "Square"],
      icons = ["▲", "♥", "■"];
    res.innerHTML =
      icons[wi] +
      " &nbsp; Result: <strong>" +
      names[wi] +
      "</strong> &nbsp; (" +
      (r.probs[wi] * 100).toFixed(1) +
      "% confidence)";
    res.className = "result-" + names[wi].toLowerCase();
    /* step-by-step animation */
    var st = {
      pgrid: pgrid,
      c1: null,
      p1: null,
      c2: null,
      p2: null,
      fl: null,
      d1: null,
      probs: null,
    };
    draw(st);
    animTimers.push(setTimeout(function () {
      st.c1 = r.c1;
      st.p1 = r.p1;
      draw(st);
      animTimers.push(setTimeout(function () {
        st.c2 = r.c2;
        st.p2 = r.p2;
        draw(st);
        animTimers.push(setTimeout(function () {
          st.fl = r.fl;
          st.d1 = r.d1;
          st.probs = r.probs;
          draw(st);
        }, 700));
      }, 700));
    }, 600));
  });
}
function onReset() {
  for (var t = 0; t < animTimers.length; t++) clearTimeout(animTimers[t]);
  animTimers = [];
  document.getElementById("file-input").value = "";
  var p = document.getElementById("img-preview");
  p.src = "";
  p.style.display = "none";
  document.getElementById("result-text").innerHTML = "";
  document.getElementById("result-text").className = "";
  draw(null);
}

/* ══════════════════ DRAWING ═══════════════════════════ */

/* map 0→white, positive→blue  -  rgb() only (sources: Internet2_css_2012) */
function blue(v, mx) {
  if (!mx || v <= 0) return "#e8ecf4";
  var t = Math.min(v / mx, 1);
  return (
    "rgb(" +
    Math.round(232 - t * 206) +
    "," +
    Math.round(236 - t * 125) +
    "," +
    Math.round(244 - t * 48) +
    ")"
  );
}
/* connection line colour by weight magnitude - rgb() only, no rgba() */
function connClr(w, strength) {
  var t = Math.min(strength, 1);
  if (w >= 0)
    return (
      "rgb(" +
      Math.round(220 - t * 194) +
      "," +
      Math.round(235 - t * 124) +
      "," +
      Math.round(248 - t * 52) +
      ")"
    );
  return (
    "rgb(" +
    Math.round(248 - t * 52) +
    "," +
    Math.round(220 - t * 190) +
    "," +
    Math.round(220 - t * 190) +
    ")"
  );
}

/* kernel weight → colour: negative=red, zero=white, positive=blue */
function wclr(w) {
  var t = Math.max(-1, Math.min(1, w / 0.25));
  if (t >= 0)
    return (
      "rgb(" +
      Math.round(255 * (1 - t * 0.8)) +
      "," +
      Math.round(255 * (1 - t * 0.8)) +
      ",255)"
    );
  return (
    "rgb(255," +
    Math.round(255 * (1 + t * 0.8)) +
    "," +
    Math.round(255 * (1 + t * 0.8)) +
    ")"
  );
}

/* draw a 3×3 kernel grid with labeled values */
function drawKernel(ctx, kernel, x, y, cell, title) {
  if (title) {
    ctx.fillStyle = "#333";
    ctx.font = "bold 9px Arial";
    ctx.textAlign = "center";
    ctx.fillText(title, x + cell * 1.5, y - 4);
  }
  for (var r = 0; r < 3; r++)
    for (var c = 0; c < 3; c++) {
      var v = kernel[r][c];
      ctx.fillStyle = wclr(v);
      ctx.fillRect(x + c * cell, y + r * cell, cell - 1, cell - 1);
      ctx.fillStyle = Math.abs(v) > 0.12 ? "#fff" : "#333";
      /* scale font with cell size so values are always readable */
      ctx.font = "bold " + Math.max(8, Math.round(cell * 0.45)) + "px Arial";
      ctx.textAlign = "center";
      ctx.fillText(
        v.toFixed(2),
        x + c * cell + cell / 2,
        y + r * cell + cell / 2 + 2,
      );
    }
  ctx.strokeStyle = "#666";
  ctx.lineWidth = 1;
  ctx.strokeRect(x, y, 3 * cell, 3 * cell);
}

/* draw 2D feature map as coloured cells */
function drawFmap(ctx, fm, x, y, cell, mx) {
  var H = fm.length,
    W = fm[0].length;
  for (var i = 0; i < H; i++)
    for (var j = 0; j < W; j++) {
      ctx.fillStyle = blue(fm[i][j], mx || 1);
      ctx.fillRect(x + j * cell, y + i * cell, cell - 1, cell - 1);
    }
  ctx.strokeStyle = "#aaa";
  ctx.lineWidth = 0.5;
  ctx.strokeRect(x, y, W * cell, H * cell);
}

/* draw 4 feature maps in a 2×2 arrangement, return {w,h} used */
function draw4maps(ctx, maps, x, y, cell, active, mx) {
  var sz = maps[0].length * cell,
    gap = 3;
  var positions = [
    [0, 0],
    [sz + gap, 0],
    [0, sz + gap],
    [sz + gap, sz + gap],
  ];
  for (var f = 0; f < 4; f++) {
    if (active)
      drawFmap(
        ctx,
        maps[f],
        x + positions[f][0],
        y + positions[f][1],
        cell,
        mx,
      );
    else {
      ctx.fillStyle = "#dde";
      ctx.fillRect(x + positions[f][0], y + positions[f][1], sz, sz);
    }
    ctx.strokeStyle = active ? "#1a6fc4" : "#bbb";
    ctx.lineWidth = 0.7;
    ctx.strokeRect(x + positions[f][0], y + positions[f][1], sz, sz);
    ctx.fillStyle = "#666";
    ctx.font = "6px Arial";
    ctx.textAlign = "center";
    ctx.fillText(
      "f" + f,
      x + positions[f][0] + sz / 2,
      y + positions[f][1] + sz + 8,
    );
  }
  return { w: 2 * sz + gap, h: 2 * sz + gap };
}

/* horizontal arrow with optional multi-line label */
function arrow(ctx, x1, x2, y, lbl) {
  ctx.beginPath();
  ctx.moveTo(x1, y);
  ctx.lineTo(x2 - 8, y);
  ctx.strokeStyle = "#888";
  ctx.lineWidth = 2;
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(x2, y);
  ctx.lineTo(x2 - 9, y - 5);
  ctx.lineTo(x2 - 9, y + 5);
  ctx.fillStyle = "#888";
  ctx.fill();
  if (lbl) {
    var mid = (x1 + x2) / 2;
    ctx.fillStyle = "#444";
    ctx.font = "bold 10px Arial";
    ctx.textAlign = "center";
    var lines = lbl.split("\n");
    for (var li = 0; li < lines.length; li++)
      ctx.fillText(lines[li], mid, y - 8 + li * 13);
  }
}

/* neuron circle - no shadow (not in sources); winner shown with thick gold border + second ring */
function neuron(ctx, x, y, r, label, val, fill, stroke, glow) {
  if (glow) {
    ctx.beginPath();
    ctx.arc(x, y, r + 5, 0, 2 * Math.PI);
    ctx.strokeStyle = "#e6a800";
    ctx.lineWidth = 3;
    ctx.stroke();
  }
  ctx.beginPath();
  ctx.arc(x, y, r, 0, 2 * Math.PI);
  ctx.fillStyle = fill;
  ctx.fill();
  ctx.strokeStyle = stroke;
  ctx.lineWidth = glow ? 4 : 1.8;
  ctx.stroke();
  ctx.fillStyle = glow ? "#5a3a00" : "#222";
  ctx.font = "bold 10px Arial";
  ctx.textAlign = "center";
  ctx.fillText(label, x, y - r - 4);
  if (val !== null) {
    ctx.font = "bold 9px Arial";
    ctx.fillText(val, x, y + 4);
  }
}

/* connection line with weight label - connClr() replaces rgba() */
function wconn(ctx, x1, y1, x2, y2, w, r1, r2, active) {
  var dx = x2 - x1,
    dy = y2 - y1,
    d = Math.sqrt(dx * dx + dy * dy);
  var nx = dx / d,
    ny = dy / d,
    aw = Math.abs(w);
  var strength = active ? Math.min(0.9, 0.2 + aw * 0.4) : 0.12;
  var lw = active ? Math.min(4, Math.max(0.8, aw * 1.8)) : 0.5;
  ctx.beginPath();
  ctx.moveTo(x1 + nx * r1, y1 + ny * r1);
  ctx.lineTo(x2 - nx * r2, y2 - ny * r2);
  ctx.strokeStyle = connClr(w, strength);
  ctx.lineWidth = lw;
  ctx.stroke();
  if (active) {
    var tx = x1 + dx * 0.45,
      ty = y1 + dy * 0.45,
      lbl = w.toFixed(2);
    ctx.font = "bold 9px Arial";
    var tw = ctx.measureText(lbl).width;
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(tx - tw / 2 - 3, ty - 7, tw + 6, 13);
    ctx.fillStyle = w >= 0 ? "#0b2860" : "#6e0b0d";
    ctx.textAlign = "center";
    ctx.fillText(lbl, tx, ty + 4);
  }
}

/* section header - 2× font */
function secHead(ctx, cx, y, line1, line2) {
  ctx.fillStyle = "#1a3a5c";
  ctx.font = "bold 13px Arial";
  ctx.textAlign = "center";
  ctx.fillText(line1, cx, y);
  ctx.fillStyle = "#555";
  ctx.font = "10px Arial";
  ctx.fillText(line2, cx, y + 15);
}

/* max value in [F,H,W] tensor */
function fmax(maps) {
  var mx = 0;
  for (var f = 0; f < maps.length; f++)
    for (var i = 0; i < maps[f].length; i++)
      for (var j = 0; j < maps[f][i].length; j++)
        if (maps[f][i][j] > mx) mx = maps[f][i][j];
  return mx || 1;
}

/* ══════════ HELPERS ══════════════════════════════════ */

/* Full-width dark section banner */
function banner(ctx, y, title, sub) {
  ctx.fillStyle = "#1a3a5c";
  ctx.fillRect(0, y, CW, 42);
  ctx.fillStyle = "#fff";
  ctx.font = "bold 17px Arial";
  ctx.textAlign = "left";
  ctx.fillText(title, 22, y + 27);
  ctx.fillStyle = "#aed6f1";
  ctx.font = "13px Arial";
  ctx.textAlign = "right";
  ctx.fillText(sub, CW - 22, y + 27);
}

/* Vertical down-arrow centred at x=CW/2 */
function downArrow(ctx, y1, y2, label) {
  var cx = CW / 2;
  ctx.beginPath();
  ctx.moveTo(cx, y1);
  ctx.lineTo(cx, y2 - 12);
  ctx.strokeStyle = "#555";
  ctx.lineWidth = 2.5;
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(cx, y2);
  ctx.lineTo(cx - 8, y2 - 14);
  ctx.lineTo(cx + 8, y2 - 14);
  ctx.fillStyle = "#555";
  ctx.fill();
  if (label) {
    ctx.fillStyle = "#333";
    ctx.font = "bold 13px Arial";
    ctx.textAlign = "left";
    ctx.fillText(label, cx + 14, (y1 + y2) / 2 + 5);
  }
}

/* Draw single kernel 3×3 at (x,y), cell size KCL, with title */
function kern(ctx, kernel, x, y, KCL, title) {
  ctx.fillStyle = "#222";
  ctx.font = "bold 13px Arial";
  ctx.textAlign = "center";
  ctx.fillText(title, x + 1.5 * KCL, y - 6);
  for (var r = 0; r < 3; r++)
    for (var c = 0; c < 3; c++) {
      var v = kernel[r][c];
      ctx.fillStyle = wclr(v);
      ctx.fillRect(x + c * KCL, y + r * KCL, KCL - 2, KCL - 2);
      ctx.fillStyle = Math.abs(v) > 0.12 ? "#fff" : "#111";
      ctx.font = "bold " + Math.round(KCL * 0.38) + "px Arial";
      ctx.textAlign = "center";
      ctx.fillText(
        v.toFixed(2),
        x + c * KCL + KCL / 2,
        y + r * KCL + KCL / 2 + 4,
      );
    }
  ctx.strokeStyle = "#444";
  ctx.lineWidth = 1.2;
  ctx.strokeRect(x, y, 3 * KCL, 3 * KCL);
}

/* Draw one feature map at (x,y) with cell size C.
   Shows numeric value inside each cell when cells are large enough. */
function fmap(ctx, fm, x, y, C, mx, active) {
  var H = fm.length,
    W = fm[0].length;
  var showNums = active && C >= 16;
  var fontSize = Math.min(16, Math.max(8, Math.round(C * 0.42)));
  for (var i = 0; i < H; i++)
    for (var j = 0; j < W; j++) {
      var v = fm[i][j];
      ctx.fillStyle = active ? blue(v, mx || 1) : "#d8dce8";
      ctx.fillRect(x + j * C, y + i * C, C - 1, C - 1);
      if (showNums) {
        ctx.fillStyle = v / (mx || 1) > 0.55 ? "#fff" : "#111";
        ctx.font = "bold " + fontSize + "px Arial";
        ctx.textAlign = "center";
        ctx.fillText(
          v.toFixed(2),
          x + j * C + C / 2,
          y + i * C + C / 2 + fontSize * 0.38,
        );
      }
    }
  ctx.strokeStyle = active ? "#1a6fc4" : "#aaa";
  ctx.lineWidth = 1.2;
  ctx.strokeRect(x, y, W * C, H * C);
}

/* Draw 4 feature maps in a SINGLE ROW centred on the canvas */
function row4maps(ctx, maps, y, C, gap, mx, active) {
  var n = 4,
    sz = maps[0].length * C,
    totalW = n * sz + (n - 1) * gap;
  var sx = Math.round((CW - totalW) / 2);
  for (var f = 0; f < n; f++) {
    var fx = sx + f * (sz + gap);
    fmap(ctx, maps[f], fx, y, C, mx, active);
    ctx.fillStyle = "#333";
    ctx.font = "bold 13px Arial";
    ctx.textAlign = "center";
    ctx.fillText("Filter " + f, fx + sz / 2, y + sz + 18);
  }
  return y + sz + 24; /* bottom edge */
}

/* Draw 4 kernels in a SINGLE ROW centred on the canvas */
function row4kernels(ctx, filters, y, KCL, titles) {
  var n = 4,
    sz = 3 * KCL,
    gap = 50,
    totalW = n * sz + (n - 1) * gap;
  var sx = Math.round((CW - totalW) / 2);
  for (var f = 0; f < n; f++) {
    kern(ctx, filters[f], sx + f * (sz + gap), y, KCL, titles[f]);
  }
  return y + 3 * KCL + 24; /* bottom edge */
}

/* neuron + circle (unchanged) */
/* wconn (unchanged) */

/* ══════════ MASTER DRAW - vertical layout ════════════ */
function draw(st) {
  var canvas = document.getElementById("nn-canvas");
  var ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, CW, CH);
  ctx.fillStyle = "#f5f7fb";
  ctx.fillRect(0, 0, CW, CH);

  var hasImg = st && st.pgrid,
    hasC1 = st && st.c1,
    hasP1 = st && st.p1,
    hasC2 = st && st.c2,
    hasP2 = st && st.p2,
    hasFlat = st && st.fl,
    hasD1 = st && st.d1,
    hasOut = st && st.probs;
  var wi = -1;
  if (hasOut) {
    wi = 0;
    for (var i = 1; i < st.probs.length; i++)
      if (st.probs[i] > st.probs[wi]) wi = i;
  }

  /* ── Sizes ── */
  var KCL = 58; /* kernel cell px - large for readability */
  var KSZ = 3 * KCL; /* 174px per 3×3 kernel */
  var K4G = 55; /* gap between 4 kernels in a row */
  var K4W = 4 * KSZ + 3 * K4G; /* total width of 4-kernel row */
  var K4X = Math.round((CW - K4W) / 2); /* left x of first kernel */

  /* Feature-map cell sizes - large enough to display numeric values */
  var C1C = 27,
    C1G = 48,
    C1S =
      8 *
      C1C; /* Conv1 8×8 maps: 27px/cell → 216px/map (1.5× for readability) */
  var P1C = 28,
    P1G = 50,
    P1S = 4 * P1C; /* Pool1 4×4 maps: 28px/cell → 112px/map */
  var P2C = 56,
    P2G = 70,
    P2S = 2 * P2C; /* Pool2 2×2 maps: 56px/cell → 112px/map */
  var C1X = Math.round((CW - (4 * C1S + 3 * C1G)) / 2);
  var P1X = Math.round((CW - (4 * P1S + 3 * P1G)) / 2);
  var P2X = Math.round((CW - (4 * P2S + 3 * P2G)) / 2);

  /* Dense-1 neurons - count read from model so it works for both 8 and 16 neurons */
  var D1N = model.layers[2].units; /* 8 or 16 depending on trained model */
  var D1R = 18;
  var D1SP = Math.floor(
    (CW - 2 * D1R) / D1N,
  ); /* auto-spacing across full width */
  var D1X0 = Math.round((CW - (D1N - 1) * D1SP) / 2);
  var d1Xs = [];
  for (var n = 0; n < D1N; n++) d1Xs.push(D1X0 + n * D1SP);
  var L2 = model.layers[2],
    L3 = model.layers[3];

  /* Output neurons (3, horizontal row) */
  var OUTR = 40,
    OUTSP = 270;
  var OUTX0 = Math.round((CW - (2 * OUTSP + 2 * OUTR)) / 2) + OUTR;
  var outXs = [OUTX0, OUTX0 + OUTSP, OUTX0 + 2 * OUTSP];
  var icons = ["▲", "♥", "■"],
    names = ["Triangle", "Heart", "Square"];

  /* ── Shared drawing sub-helpers (var = expressions, safe in all modes) ── */

  var slbl = function (txt, y2) {
    ctx.fillStyle = "#1a3a5c";
    ctx.font = "bold 15px Arial";
    ctx.textAlign = "left";
    ctx.fillText(txt, 30, y2);
  };

  var note = function (txt, y2) {
    ctx.fillStyle = "#555";
    ctx.font = "14px Arial";
    ctx.textAlign = "center";
    ctx.fillText(txt, CW / 2, y2);
  };

  var k4row = function (filters, titles, biases, y2) {
    for (var f = 0; f < 4; f++) {
      var kx = K4X + f * (KSZ + K4G);
      ctx.fillStyle = "#1a3a5c";
      ctx.font = "bold 14px Arial";
      ctx.textAlign = "center";
      ctx.fillText(titles[f], kx + KSZ / 2, y2 + 14);
      kern(ctx, filters[f], kx, y2 + 24, KCL, "");
      ctx.fillStyle = "#555";
      ctx.font = "13px Arial";
      ctx.textAlign = "center";
      ctx.fillText(
        "bias = " + biases[f].toFixed(3),
        kx + KSZ / 2,
        y2 + 24 + KSZ + 20,
      );
    }
    return y2 + 24 + KSZ + 36;
  };

  var fm4row = function (maps, active, mx, sx, fsz, fcell, fgap, y2) {
    for (var f = 0; f < 4; f++) {
      var fx = sx + f * (fsz + fgap);
      if (active && maps && maps[f]) {
        fmap(ctx, maps[f], fx, y2, fcell, mx, true);
      } else {
        ctx.fillStyle = "#d8dce8";
        ctx.fillRect(fx, y2, fsz, fsz);
        ctx.strokeStyle = "#bbb";
        ctx.lineWidth = 1;
        ctx.strokeRect(fx, y2, fsz, fsz);
      }
      ctx.fillStyle = "#333";
      ctx.font = "bold 13px Arial";
      ctx.textAlign = "center";
      ctx.fillText("Map " + f, fx + fsz / 2, y2 + fsz + 18);
    }
    return y2 + fsz + 30;
  };

  /* ── Y cursor - every element placed explicitly ── */
  var y = 0;

  /* ══════════════════════════════════════════════
     LAYER 0 - INPUT
  ══════════════════════════════════════════════ */
  banner(
    ctx,
    y,
    "Layer 0 - Input",
    "Any image, any size - resized to 8×8 grayscale pixels",
  );
  y += 44;

  y += 18;
  note(
    "The uploaded image is drawn to a hidden 8×8 canvas.  " +
      "Dark pixels → 0.0  ·  White pixels → 1.0",
    y,
  );
  y += 30;

  /* Image centred */
  var IMSZ = 200,
    CELLIM = 25;
  var imgX = Math.round((CW - IMSZ) / 2);
  ctx.strokeStyle = hasImg ? "#1a6fc4" : "#bbb";
  ctx.lineWidth = 2;
  ctx.strokeRect(imgX, y, IMSZ, IMSZ);
  var prev = document.getElementById("img-preview");
  if (prev && prev.src && prev.style.display !== "none") {
    ctx.drawImage(prev, imgX, y, IMSZ, IMSZ);
  } else {
    ctx.fillStyle = "#d8dfe8";
    ctx.fillRect(imgX + 1, y + 1, IMSZ - 2, IMSZ - 2);
    ctx.fillStyle = "#888";
    ctx.font = "16px Arial";
    ctx.textAlign = "center";
    ctx.fillText("Upload image", imgX + IMSZ / 2, y + IMSZ / 2 - 8);
    ctx.fillText("(any size)", imgX + IMSZ / 2, y + IMSZ / 2 + 14);
  }
  if (hasImg) {
    var g8 = st.pgrid[0];
    for (var i = 0; i < 8; i++)
      for (var j = 0; j < 8; j++) {
        var gv = g8[i][j];
        /* gv=1.0 → white background → display light; gv=0.0 → black shape → display dark */
        var gc = Math.round(gv * 240);
        ctx.fillStyle = "rgb(" + gc + "," + gc + "," + gc + ")";
        ctx.fillRect(imgX + j * CELLIM, y + i * CELLIM, CELLIM - 1, CELLIM - 1);
        /* value label - white text on dark cells, dark text on light cells */
        ctx.fillStyle = gv < 0.5 ? "#fff" : "#111";
        ctx.font = "bold 8px Arial";
        ctx.textAlign = "center";
        ctx.fillText(
          gv.toFixed(2),
          imgX + j * CELLIM + CELLIM / 2,
          y + i * CELLIM + CELLIM / 2 + 3,
        );
      }
  }
  y += IMSZ + 22;
  note(
    "8 × 8 pixel grid  (" +
      (hasImg
        ? "actual normalised values shown above"
        : "upload to see values") +
      ")",
    y,
  );
  y += 22;
  /* Drawing tip */
  ctx.fillStyle = "#e67e22";
  ctx.font = "bold 13px Arial";
  ctx.textAlign = "center";
  ctx.fillText(
    "Tip: draw a large FILLED dark shape on a plain white background - fill most of the image for best results.",
    CW / 2,
    y,
  );
  y += 28;

  downArrow(ctx, y, y + 65, "Convolution  +  ReLU");
  y += 65;

  /* ══════════════════════════════════════════════
     LAYER 1 — CONV2D-1
  ══════════════════════════════════════════════ */
  banner(
    ctx,
    y,
    "Layer 1 - Conv2D-1  →  ReLU",
    "4 filters · 3×3 kernels · same padding · 4 bias values",
  );
  y += 44;

  y += 16;
  note(
    "Each 3×3 filter slides across the 8×8 image, computing a dot product at every position.  ReLU zeroes negatives.",
    y,
  );
  y += 20;
  note(
    "The value in each kernel cell is the exact learned filter coefficient - 9 per filter, 36 total across 4 filters.",
    y,
  );
  y += 20;
  /* ── Disclaimer 1 ── */
  ctx.fillStyle = "#b05000";
  ctx.font = "bold 13px Arial";
  ctx.textAlign = "center";
  ctx.fillText(
    "⚠ Disclaimer: Layer 0 has 64 neurons; Layer 1 has 4×8×8 = 256 neurons - 64×256 = 16,384 potential connections.",
    CW / 2,
    y,
  );
  y += 18;
  ctx.fillStyle = "#555";
  ctx.font = "13px Arial";
  ctx.fillText(
    "Connections are not drawn. Conv filters use weight sharing: the same 9-value kernel is reused at every position - unlike Dense layers where every pair has its own unique weight.",
    CW / 2,
    y,
  );
  y += 22;

  /* divider line */
  ctx.strokeStyle = "#ccc";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(20, y);
  ctx.lineTo(CW - 20, y);
  ctx.stroke();
  y += 16;

  slbl(
    "Learned filter kernels - 4 filters × 9 coefficients each (Blue = positive  ·  Red = negative):",
    y,
  );
  y += 26;

  var L0 = model.layers[0];
  var kt0 = [
    "Filter 0 - horiz. edge",
    "Filter 1 - vert. edge",
    "Filter 2 - diagonal",
    "Filter 3 - avg / density",
  ];
  var L0f = [];
  for (var fi = 0; fi < L0.filters.length; fi++) L0f.push(L0.filters[fi][0]);
  y = k4row(L0f, kt0, L0.biases, y);
  y += 20;

  ctx.strokeStyle = "#ccc";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(20, y);
  ctx.lineTo(CW - 20, y);
  ctx.stroke();
  y += 16;

  slbl("Output feature maps after ReLU  (8 × 8 pixels per map):", y);
  y += 26;
  var c1mx = hasC1 ? fmax(st.c1) : 1;
  y = fm4row(hasC1 ? st.c1 : null, hasC1, c1mx, C1X, C1S, C1C, C1G, y);
  y += 24;

  downArrow(ctx, y, y + 55, "Max-Pool  2×2");
  y += 55;

  /* ══════════════════════════════════════════════
     LAYER 2 — MAXPOOL-1
  ══════════════════════════════════════════════ */
  banner(
    ctx,
    y,
    "Layer 2 - MaxPool2D  (2×2)",
    "Spatial size halved: 8×8 → 4×4  ·  Keeps maximum of each 2×2 window",
  );
  y += 44;

  y += 16;
  note("No learned weights - MaxPool is a fixed mathematical operation.", y);
  y += 20;
  note("Output: 4 feature maps × 4×4 pixels = 64 activation values.", y);
  y += 20;
  ctx.fillStyle = "#b05000";
  ctx.font = "bold 13px Arial";
  ctx.textAlign = "center";
  ctx.fillText(
    "⚠ Disclaimer: Layer 1 (Conv2D-1) has 4×8×8 = 256 neurons; Layer 2 (MaxPool-1) has 4×4×4 = 64 neurons.",
    CW / 2,
    y,
  );
  y += 18;
  ctx.fillStyle = "#555";
  ctx.font = "13px Arial";
  ctx.fillText(
    "Drawing all 256×64 = 16,384 individual connections is not feasible - no connection diagram or weight table is shown.",
    CW / 2,
    y,
  );
  y += 18;
  ctx.fillText(
    "MaxPool picks the max of each 2×2 window - no learned weights - so no weight lines are drawn.",
    CW / 2,
    y,
  );
  y += 22;

  ctx.strokeStyle = "#ccc";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(20, y);
  ctx.lineTo(CW - 20, y);
  ctx.stroke();
  y += 16;

  slbl("Output feature maps after max-pooling  (4 × 4 pixels per map):", y);
  y += 26;
  var p1mx = hasP1 ? fmax(st.p1) : 1;
  y = fm4row(hasP1 ? st.p1 : null, hasP1, p1mx, P1X, P1S, P1C, P1G, y);
  y += 24;

  downArrow(ctx, y, y + 65, "Convolution  +  ReLU");
  y += 65;

  /* ══════════════════════════════════════════════
     LAYER 3 — CONV2D-2
  ══════════════════════════════════════════════ */
  banner(
    ctx,
    y,
    "Layer 3 - Conv2D-2  →  ReLU",
    "4 filters · 3×3 kernels · 4 input channels · 144 learned coefficients total · grid = average across 4 input channels",
  );
  y += 44;

  y += 16;
  note(
    "Second convolutional layer - operates on 4 input channels (the feature maps from MaxPool-1).",
    y,
  );
  y += 20;
  note(
    "Each output filter learns a separate 3×3 kernel per input channel → 4×9 = 36 coefficients per filter, 144 total.",
    y,
  );
  y += 20;
  note(
    "Since showing all 16 channel kernels (4 filters × 4 channels) would be unreadable, the grid displays the average across input channels.",
    y,
  );
  y += 20;
  /* ── Disclaimer 2 ── */
  ctx.fillStyle = "#b05000";
  ctx.font = "bold 13px Arial";
  ctx.textAlign = "center";
  ctx.fillText(
    "⚠ Disclaimer: Layer 2 (MaxPool-1) has 4×4×4 = 64 neurons; Layer 3 (Conv2D-2) has 4×4×4 = 64 neurons - 64×64 = 4,096 potential connections.",
    CW / 2,
    y,
  );
  y += 18;
  ctx.fillStyle = "#555";
  ctx.font = "13px Arial";
  ctx.fillText(
    "Connections are not drawn - too many to display individually.",
    CW / 2,
    y,
  );
  y += 18;
  ctx.fillText(
    "The averaged kernel grids above show the learned filter coefficients for this layer.",
    CW / 2,
    y,
  );
  y += 22;

  ctx.strokeStyle = "#ccc";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(20, y);
  ctx.lineTo(CW - 20, y);
  ctx.stroke();
  y += 16;

  slbl(
    "Learned filter kernels - channel-averaged view (Blue = positive  ·  Red = negative):",
    y,
  );
  y += 26;

  var L1 = model.layers[1];
  var kt2 = [
    "Filter 0 - triangle",
    "Filter 1 - heart",
    "Filter 2 - square",
    "Filter 3 - general",
  ];
  var avgF = [];
  for (var fi = 0; fi < L1.filters.length; fi++) {
    var a = [
      [0, 0, 0],
      [0, 0, 0],
      [0, 0, 0],
    ];
    for (var ch = 0; ch < 4; ch++)
      for (var r = 0; r < 3; r++)
        for (var c = 0; c < 3; c++) a[r][c] += L1.filters[fi][ch][r][c] / 4;
    avgF.push(a);
  }
  y = k4row(avgF, kt2, L1.biases, y);
  y += 20;

  ctx.strokeStyle = "#ccc";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(20, y);
  ctx.lineTo(CW - 20, y);
  ctx.stroke();
  y += 16;

  slbl("Output feature maps after ReLU  (4 × 4 pixels per map):", y);
  y += 26;
  var c2mx = hasC2 ? fmax(st.c2) : 1;
  y = fm4row(hasC2 ? st.c2 : null, hasC2, c2mx, P1X, P1S, P1C, P1G, y);
  y += 24;

  downArrow(ctx, y, y + 65, "Max-Pool  2×2");
  y += 65;

  /* ══════════════════════════════════════════════
     LAYER 4 — MAXPOOL-2   (separate layer — NOT merged with Flatten)
     4 feature maps 2×2 = 16 neurons total.
     No learned weights - only keeps the maximum of each 2×2 window.
  ══════════════════════════════════════════════ */
  banner(
    ctx,
    y,
    "Layer 4 - MaxPool2D  (2×2)  +  Flatten",
    "4×4 → 2×2 per map  ·  16 neurons  ·  Flatten = reshape to 1D  ·  no learned weights in either operation",
  );
  y += 44;

  y += 16;
  note(
    "No learned weights - both MaxPool and Flatten are fixed operations with no trainable parameters.",
    y,
  );
  y += 20;
  note(
    "MaxPool halves spatial size by keeping the max of each 2×2 block.  Flatten reshapes [4×2×2] → [16]: same values, 1D format.",
    y,
  );
  y += 20;
  ctx.fillStyle = "#b05000";
  ctx.font = "bold 13px Arial";
  ctx.textAlign = "center";
  ctx.fillText(
    "⚠ Disclaimer: Layer 3 (Conv2D-2) has 4×4×4 = 64 neurons; Layer 4 (MaxPool-2) has 4×2×2 = 16 neurons.",
    CW / 2,
    y,
  );
  y += 18;
  ctx.fillStyle = "#555";
  ctx.font = "13px Arial";
  ctx.fillText(
    "Drawing all 64×16 = 1,024 individual connections is not feasible - no connection diagram or weight table is shown.",
    CW / 2,
    y,
  );
  y += 18;
  ctx.fillText(
    "MaxPool picks the max of each 2×2 window - no learned weights - so no weight lines are drawn.",
    CW / 2,
    y,
  );
  y += 22;

  ctx.strokeStyle = "#ccc";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(20, y);
  ctx.lineTo(CW - 20, y);
  ctx.stroke();
  y += 16;

  slbl("Output feature maps after max-pooling  (2 × 2 pixels per map):", y);
  y += 26;
  var p2mx = hasP2 ? fmax(st.p2) : 1;
  y = fm4row(hasP2 ? st.p2 : null, hasP2, p2mx, P2X, P2S, P2C, P2G, y);
  y += 24;

  ctx.strokeStyle = "#ccc";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(20, y);
  ctx.lineTo(CW - 20, y);
  ctx.stroke();
  y += 16;

  /* Flatten — same 16 values shown as 1D vector, merged into Layer 4 */
  var FL_N = model.layers[2].weights[0].length; /* 16 when 4 filters */
  slbl(
    "Flatten - same 16 values reshaped to 1D  (reshape only, no new computation)  ·  value inside = activation:",
    y,
  );
  y += 28;

  var FL_R = 13;
  var FL_SP = Math.floor((CW - 2 * FL_R) / FL_N);
  var fl_xs = [];
  for (var i = 0; i < FL_N; i++)
    fl_xs.push(Math.round((CW - (FL_N - 1) * FL_SP) / 2) + i * FL_SP);

  var fl_y = y + FL_R;
  var flat16 = hasP2 ? flatten(st.p2) : null;
  var flat16max = 1;
  if (flat16) {
    for (var i = 0; i < flat16.length; i++) {
      var av = flat16[i] < 0 ? -flat16[i] : flat16[i];
      if (av > flat16max) flat16max = av;
    }
  }

  for (var i = 0; i < FL_N; i++) {
    var fv = flat16 ? flat16[i] : null;
    neuron(
      ctx,
      fl_xs[i],
      fl_y,
      FL_R,
      "[" + i + "]",
      fv !== null ? fv.toFixed(2) : null,
      fv !== null ? blue(Math.abs(fv), flat16max) : "#e4e8f4",
      hasFlat || hasD1 ? "#1a6fc4" : "#99aabb",
      false,
    );
  }
  y = fl_y + FL_R + 30;

  downArrow(ctx, y, y + 55, "Fully Connected  +  ReLU");
  y += 55;

  /* ══════════════════════════════════════════════
     LAYER 6 — DENSE-1
     Shown exactly like the Output layer:
     connections enter from the top, neurons below, weight table at the bottom.
  ══════════════════════════════════════════════ */
  banner(
    ctx,
    y,
    "Layer 5 - Dense-1  (Fully Connected)",
    FL_N +
      "×" +
      D1N +
      " = " +
      FL_N * D1N +
      " weights + " +
      D1N +
      " biases  ·  ReLU  ·  Blue=positive · Red=negative · thickness ∝ |weight|",
  );
  y += 44;

  y += 14;
  note(
    "Each of the " +
      D1N +
      " neurons multiplies every one of the " +
      FL_N +
      " flattened inputs by its own unique learned weight, adds a bias, then applies ReLU.",
    y,
  );
  y += 20;
  ctx.fillStyle = "#555";
  ctx.font = "13px Arial";
  ctx.textAlign = "center";
  ctx.fillText(
    "Each line below = one unique learned weight connecting a specific Flatten neuron to a specific Dense-1 neuron.",
    CW / 2,
    y,
  );
  y += 18;
  ctx.fillText(
    "Unlike Conv filters (shared across all positions), every Dense connection has its own independent value - line colour and thickness encode the weight.",
    CW / 2,
    y,
  );
  y += 28;

  ctx.strokeStyle = "#ccc";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(20, y);
  ctx.lineTo(CW - 20, y);
  ctx.stroke();
  y += 16;

  slbl(
    "All " +
      FL_N * D1N +
      " weight connections  Flatten → Dense-1  (Blue=positive · Red=negative · thickness ∝ |weight|):",
    y,
  );
  y += 26;

  /* Pre-compute Dense-1 neuron y so connection lines can be drawn first */
  var d1CY = y + D1R + 130;

  /* Draw all FL_N × D1N connection lines first (neurons on top) */
  for (var i = 0; i < FL_N; i++) {
    for (var n = 0; n < D1N; n++) {
      var wfl = L2.weights[n][i],
        awfl = Math.abs(wfl);
      var sfl =
        hasFlat || hasD1
          ? Math.min(0.9, 0.28 + awfl * 0.48)
          : Math.min(0.78, 0.18 + awfl * 0.38);
      var lwfl =
        hasFlat || hasD1
          ? Math.min(5.5, Math.max(0.8, awfl * 2.2))
          : Math.min(4, Math.max(0.5, awfl * 1.6));
      ctx.beginPath();
      ctx.moveTo(fl_xs[i], y + 10);
      ctx.lineTo(d1Xs[n], d1CY - D1R - 6);
      ctx.strokeStyle = connClr(wfl, sfl);
      ctx.lineWidth = lwfl;
      ctx.stroke();
    }
  }

  /* Dense-1 neuron circles on top */
  var d1mx = 1;
  if (hasD1) {
    for (var i = 0; i < st.d1.length; i++) if (st.d1[i] > d1mx) d1mx = st.d1[i];
  }
  for (var n = 0; n < D1N; n++) {
    var dv = hasD1 ? st.d1[n] : null;
    neuron(
      ctx,
      d1Xs[n],
      d1CY,
      D1R,
      "N" + n,
      dv !== null ? dv.toFixed(2) : null,
      hasD1 ? blue(dv, d1mx) : "#e4e8f4",
      hasD1 ? "#1a6fc4" : "#99aabb",
      false,
    );
    ctx.fillStyle = "#666";
    ctx.font = "11px Arial";
    ctx.textAlign = "center";
    ctx.fillText("b=" + L2.biases[n].toFixed(2), d1Xs[n], d1CY + D1R + 16);
  }
  y = d1CY + D1R + 40;

  ctx.strokeStyle = "#ccc";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(20, y);
  ctx.lineTo(CW - 20, y);
  ctx.stroke();
  y += 16;

  /* Weight matrix heatmap — auto-sized for D1N × FL_N */
  slbl(
    "Dense-1 weight matrix  [" +
      D1N +
      " neurons × " +
      FL_N +
      " inputs]  -  Blue=positive · Red=negative · value inside each cell:",
    y,
  );
  y += 26;
  var wW = Math.min(52, Math.floor((CW - 80) / FL_N) - 3);
  var wH = 34,
    wG = 3;
  var wmX = Math.round((CW - (FL_N * (wW + wG) - wG)) / 2);
  var wmMax = 0;
  for (var ni = 0; ni < D1N; ni++)
    for (var ii = 0; ii < FL_N; ii++)
      if (Math.abs(L2.weights[ni][ii]) > wmMax)
        wmMax = Math.abs(L2.weights[ni][ii]);
  for (var ni = 0; ni < D1N; ni++) {
    ctx.fillStyle = "#333";
    ctx.font = "bold 12px Arial";
    ctx.textAlign = "right";
    ctx.fillText("N" + ni, wmX - 10, y + ni * (wH + wG) + wH / 2 + 5);
    for (var ii = 0; ii < FL_N; ii++) {
      var wv = L2.weights[ni][ii],
        wt = wmMax ? wv / wmMax : 0;
      var wr, wg2, wb2;
      if (wt >= 0) {
        wr = Math.round(235 * (1 - wt * 0.85));
        wg2 = Math.round(235 * (1 - wt * 0.85));
        wb2 = 235;
      } else {
        wr = 235;
        wg2 = Math.round(235 * (1 + wt * 0.85));
        wb2 = Math.round(235 * (1 + wt * 0.85));
      }
      ctx.fillStyle = "rgb(" + wr + "," + wg2 + "," + wb2 + ")";
      ctx.fillRect(wmX + ii * (wW + wG), y + ni * (wH + wG), wW, wH);
      ctx.fillStyle = Math.abs(wt) > 0.5 ? "#fff" : "#111";
      ctx.font = "bold " + Math.max(8, Math.round(wW * 0.22)) + "px Arial";
      ctx.textAlign = "center";
      ctx.fillText(
        wv.toFixed(2),
        wmX + ii * (wW + wG) + wW / 2,
        y + ni * (wH + wG) + wH / 2 + 4,
      );
    }
  }
  for (var ii = 0; ii < FL_N; ii++) {
    ctx.fillStyle = "#555";
    ctx.font = "11px Arial";
    ctx.textAlign = "center";
    ctx.fillText(
      "[" + ii + "]",
      wmX + ii * (wW + wG) + wW / 2,
      y + D1N * (wH + wG) + 14,
    );
  }
  y += D1N * (wH + wG) + 32;

  downArrow(ctx, y, y + 65, "Fully Connected  +  Softmax");
  y += 65;

  /* ══════════════════════════════════════════════
     LAYER 7 — OUTPUT   (Dense → Softmax)
     D1N × 3 = 24 connection lines, all with weight labels.
  ══════════════════════════════════════════════ */
  banner(
    ctx,
    y,
    "Layer 6 - Output Layer  (Fully Connected  →  Softmax)",
    D1N +
      " × 3 = " +
      D1N * 3 +
      " weights + 3 biases  ·  Blue=positive · Red=negative · thickness ∝ |weight|  ·  Winner glows gold",
  );
  y += 44;

  y += 16;
  note(
    "Each output neuron computes: score = dot product of Dense-1 activations × its weights + bias.",
    y,
  );
  y += 20;
  note(
    "Softmax converts the 3 raw scores to probabilities summing to 100% - the highest probability is the prediction.",
    y,
  );
  y += 26;

  ctx.strokeStyle = "#ccc";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(20, y);
  ctx.lineTo(CW - 20, y);
  ctx.stroke();
  y += 16;

  slbl(
    "All 24 weight connections  Dense-1 → Output  (Blue=positive · Red=negative · thickness ∝ |weight|):",
    y,
  );
  y += 26;

  var outCY = y + OUTR + 130; /* Output neuron centres */

  /* ── Dense-1 → Output weight lines (D1N × 3) ── */
  for (var o = 0; o < 3; o++) {
    for (var n = 0; n < D1N; n++) {
      var ww = L3.weights[o][n],
        aw = Math.abs(ww);
      var x1 = d1Xs[n],
        y1 = y + 10;
      var x2 = outXs[o],
        y2 = outCY - OUTR - 6;
      var so =
        hasD1 || hasOut
          ? Math.min(0.95, 0.42 + aw * 0.55)
          : Math.min(0.82, 0.28 + aw * 0.48);
      var lw =
        hasD1 || hasOut
          ? Math.min(7, Math.max(1.5, aw * 3.0))
          : Math.min(5.5, Math.max(1.0, aw * 2.4));
      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
      ctx.strokeStyle = connClr(ww, so);
      ctx.lineWidth = lw;
      ctx.stroke();
    }
  }

  /* ── Output neuron circles (drawn on top of lines) ── */
  for (var o = 0; o < 3; o++) {
    var prob = hasOut ? st.probs[o] : null,
      isW = hasOut && o === wi;
    neuron(
      ctx,
      outXs[o],
      outCY,
      OUTR,
      icons[o] + "  " + names[o],
      prob !== null ? (prob * 100).toFixed(1) + " %" : null,
      isW ? "#FFD700" : hasOut ? blue(prob, 1) : "#e4e8f4",
      isW ? "#c8900a" : hasOut ? "#1a6fc4" : "#99aabb",
      isW,
    );
    ctx.fillStyle = "#555";
    ctx.font = "12px Arial";
    ctx.textAlign = "center";
    ctx.fillText(
      "bias = " + L3.biases[o].toFixed(3),
      outXs[o],
      outCY + OUTR + 20,
    );
    if (prob !== null) {
      ctx.fillStyle = isW ? "#b07000" : "#1a3a5c";
      ctx.font = "bold 20px Arial";
      ctx.textAlign = "center";
      ctx.fillText((prob * 100).toFixed(1) + " %", outXs[o], outCY + OUTR + 46);
    }
  }
  y = outCY + OUTR + 70;

  /* ── Weight matrix table: rows = Output neurons, cols = Dense-1 neurons ──
     This is the standard paper-style way to show all 24 weights clearly.
     Each cell: colour = sign, value = exact weight.
  ───────────────────────────────────────────────────────────────────────── */
  ctx.strokeStyle = "#ccc";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(20, y);
  ctx.lineTo(CW - 20, y);
  ctx.stroke();
  y += 16;

  slbl(
    "Output layer weight matrix  [3 output neurons × 8 Dense-1 inputs]  -  read: row = neuron, column = Dense-1 input:",
    y,
  );
  y += 26;

  /* Layout: auto-size columns based on D1N */
  var mRowH = 46,
    mGap = 4;
  var mColW = Math.min(80, Math.floor((CW - 120) / D1N) - mGap);
  var mRows = ["▲ Triangle", "♥ Heart", "■ Square"];
  var mCols = [];
  for (var ci = 0; ci < D1N; ci++) mCols.push("N" + ci);
  var mLabelW = 100;
  var mTotalW = mLabelW + D1N * (mColW + mGap) - mGap;
  var mX = Math.round((CW - mTotalW) / 2);

  /* column headers */
  ctx.fillStyle = "#1a3a5c";
  ctx.font = "bold 12px Arial";
  ctx.textAlign = "center";
  for (var ci = 0; ci < D1N; ci++) {
    ctx.fillText(
      mCols[ci],
      mX + mLabelW + ci * (mColW + mGap) + mColW / 2,
      y + 14,
    );
  }
  y += 22;

  /* find max absolute weight for colour scaling */
  var mMax = 0;
  for (var ri = 0; ri < 3; ri++)
    for (var ci = 0; ci < D1N; ci++)
      if (Math.abs(L3.weights[ri][ci]) > mMax)
        mMax = Math.abs(L3.weights[ri][ci]);

  for (var ri = 0; ri < 3; ri++) {
    /* row label */
    ctx.fillStyle = "#222";
    ctx.font = "bold 14px Arial";
    ctx.textAlign = "right";
    ctx.fillText(mRows[ri], mX + mLabelW - 8, y + mRowH / 2 + 5);

    for (var ci = 0; ci < D1N; ci++) {
      var mw = L3.weights[ri][ci],
        mt = mMax ? mw / mMax : 0;
      var mr, mg, mb;
      if (mt >= 0) {
        /* positive → blue */
        mr = Math.round(235 * (1 - mt * 0.82));
        mg = Math.round(235 * (1 - mt * 0.82));
        mb = 235;
      } else {
        /* negative → red */
        mr = 235;
        mg = Math.round(235 * (1 + mt * 0.82));
        mb = Math.round(235 * (1 + mt * 0.82));
      }
      var cx2 = mX + mLabelW + ci * (mColW + mGap),
        cy2 = y;
      ctx.fillStyle = "rgb(" + mr + "," + mg + "," + mb + ")";
      ctx.fillRect(cx2, cy2, mColW, mRowH);
      ctx.strokeStyle = "#bbb";
      ctx.lineWidth = 0.8;
      ctx.strokeRect(cx2, cy2, mColW, mRowH);

      /* weight value */
      ctx.fillStyle = Math.abs(mt) > 0.55 ? "#fff" : "#111";
      ctx.font = "bold 14px Arial";
      ctx.textAlign = "center";
      ctx.fillText(mw.toFixed(3), cx2 + mColW / 2, cy2 + mRowH / 2 + 5);
    }
    y += mRowH + mGap;
  }
  /* column axis label */
  ctx.fillStyle = "#777";
  ctx.font = "12px Arial";
  ctx.textAlign = "center";
  ctx.fillText(
    "← Dense-1 neuron inputs (N0 … N" + (D1N - 1) + ") →",
    mX + mLabelW + (D1N * mColW + (D1N - 1) * mGap) / 2,
    y + 14,
  );
  y += 32;

  /* ── Legend ── */
  ctx.fillStyle = "#eef0f6";
  ctx.fillRect(0, y, CW, 44);
  ctx.strokeStyle = "#ccd3db";
  ctx.lineWidth = 1;
  ctx.strokeRect(0, y, CW, 44);
  ctx.fillStyle = "#0b2860";
  ctx.fillRect(24, y + 15, 36, 8);
  ctx.fillStyle = "#222";
  ctx.font = "13px Arial";
  ctx.textAlign = "left";
  ctx.fillText("Positive weight", 66, y + 23);
  ctx.fillStyle = "#6e0b0d";
  ctx.fillRect(230, y + 15, 36, 8);
  ctx.fillText("Negative weight", 272, y + 23);
  ctx.fillStyle = "#666";
  ctx.font = "12px Arial";
  ctx.fillText(
    "  ·  Line thickness & opacity ∝ |weight|  ·  Kernel cell value = exact trained weight  ·  Heat colour = activation magnitude",
    430,
    y + 23,
  );
  ctx.textAlign = "right";
  ctx.fillStyle = "#888";
  ctx.fillText(
    hasOut
      ? "✓ Forward pass complete - winner glows gold"
      : hasImg
        ? "Click Test to run the CNN - scroll down to see all layers"
        : "Upload image → click Test",
    CW - 16,
    y + 23,
  );
}
