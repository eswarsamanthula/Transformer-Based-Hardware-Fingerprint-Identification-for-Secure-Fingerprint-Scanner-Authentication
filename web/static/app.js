(function () {
  "use strict";

  var NUM_FMT = function (v, d) { return v == null ? "\u2014" : (v * 100).toFixed(d == null ? 1 : d) + "%"; };
  var CAT = function (m) {
    return ["efficientnet_b3","mobilenet","resnet50","resnet152","vgg16","lenet5"].indexOf(m) >= 0 ? "cnn" : "vit";
  };

  function fail(msg) {
    var el = document.getElementById("err");
    el.hidden = false;
    el.innerHTML = "<em style='color:var(--bad)'>" + msg + "</em>";
  }

  function countUp(el, target, decimals) {
    var ms = 620, t0 = null;
    function frame(t) {
      if (!t0) t0 = t;
      var p = Math.min(1, (t - t0) / ms);
      el.textContent = (target * p * 100).toFixed(decimals) + "%";
      if (p < 1) requestAnimationFrame(frame);
    }
    requestAnimationFrame(frame);
  }

  var cssVars = getComputedStyle(document.documentElement);
  var COL_1 = cssVars.getPropertyValue("--amber").trim() || "#c94810";
  var COL_2 = cssVars.getPropertyValue("--cyan").trim() || "#24517b";
  function alpha(hex, a) {
    return "rgba(" + parseInt(hex.slice(1, 3), 16) + "," +
      parseInt(hex.slice(3, 5), 16) + "," + parseInt(hex.slice(5, 7), 16) + "," + a + ")";
  }

  if (typeof Chart !== "undefined") {
    Chart.defaults.font.family = "IBM Plex Mono";
    Chart.defaults.color = cssVars.getPropertyValue("--ink-2").trim() || "#4c4637";
    Chart.defaults.borderColor = cssVars.getPropertyValue("--rule").trim() || "rgba(0,0,0,0.08)";

    Promise.all([
      fetch("/api/multiclass").then(function (r) { return r.json(); }),
      fetch("/api/binary").then(function (r) { return r.json(); }),
      fetch("/api/curves").then(function (r) { return r.json(); })
    ]).then(function (data) {
      var mc = data[0], bin = data[1], curves = data[2];

      if (mc.error) { fail(mc.error); return; }
      renderMulti(mc.models);
      renderBinary(bin.rows);
      renderCurves(curves);
    })["catch"](function (e) { fail("Failed to load results: " + e); });
  }
  var accChart = null, lossChart = null, acc2Chart = null, binChart = null;
  var RUNS = {};

  function styleChart(c, lean) {
    c.options.scales = {
      x: { ticks: { maxRotation: 45, font: { size: 10 } }, grid: { display: false } },
      y: { beginAtZero: true, ticks: { font: { size: 10 } } }
    };
    if (lean === "pct") {
      c.options.scales.y.ticks.callback = function (v) { return (v * 100).toFixed(0) + "%"; };
    }
    c.update();
  }

  /* ---------------- multiclass ---------------- */

  function renderMulti(models) {
    models.sort(function (a, b) {
      var ta = a.test.accuracy == null ? -1 : a.test.accuracy;
      var tb = b.test.accuracy == null ? -1 : b.test.accuracy;
      return tb - ta || (a.model < b.model ? -1 : 1);
    });

    var best = -1, bestGray = -1, bestRgb = -1, bestM, bestGM, bestRM;
    models.forEach(function (m) {
      var t = m.test.accuracy;
      if (t > best) { best = t; bestM = m; }
      if (m.channel === "grayscale" && t > bestGray) { bestGray = t; bestGM = m; }
      if (m.channel === "rgb" && t > bestRgb) { bestRgb = t; bestRM = m; }
    });

    countUp(document.getElementById("kpi-best"), best, 1);
    document.getElementById("kpi-best-sub").textContent = bestM.model + " \u00b7 " + bestM.channel;
    countUp(document.getElementById("kpi-gray"), bestGray, 1);
    document.getElementById("kpi-gray-sub").textContent = bestGM.model;
    countUp(document.getElementById("kpi-rgb"), bestRgb, 1);
    document.getElementById("kpi-rgb-sub").textContent = bestRM.model;

    var tbody = document.querySelector("#tbl-multi tbody");
    models.forEach(function (m) {
      var tr = document.createElement("tr");
      if (m.test.accuracy === best) tr.className = "best best-line";
      else if (m.test.accuracy === bestGray && m.channel === "grayscale") tr.className = "next";
      else if (m.test.accuracy === bestRgb && m.channel === "rgb") tr.className = "next";

      var c = CAT(m.model);
      tr.innerHTML =
        "<td class='model'>" + m.model + "<span class='badge " + c + "'>" + (c === "cnn" ? "CNN" : "ViT") + "</span></td>" +
        "<td class='chan " + m.channel + "'>" + m.channel + "</td>" +
        "<td class='num " + highlight(m.val, bestGray, "gray") + "'>" + cell(m.val.accuracy) + "</td>" +
        "<td class='num'>" + cell(m.val.f1) + "</td>" +
        "<td class='num'>" + cell(m.val.auc) + "</td>" +
        "<td class='num " + hl(m.test, best, bestGray, bestRgb, m.channel) + "'>" + cell(m.test.accuracy) + "</td>" +
        "<td class='num'>" + cell(m.test.f1) + "</td>" +
        "<td class='num'>" + cell(m.test.auc) + "</td>";
      tbody.appendChild(tr);
    });

    renderAccChart(models);
  }

  function hl(t, best, bestGray, bestRgb, channel) {
    if (t.accuracy == null) return "";
    if (t.accuracy === best) return "hl";
    if (channel === "grayscale" && t.accuracy === bestGray) return "hl";
    if (channel === "rgb" && t.accuracy === bestRgb) return "hl";
    return "";
  }
  function highlight(_, b, ch) { return ""; } // val cells never highlighted

  function cell(v) { return v == null ? "\u2014" : NUM_FMT(v, 1); }

  function renderAccChart(models) {
    var ctx = document.getElementById("chart-acc").getContext("2d");
    var labels = models.map(function (m) { return m.model + "\n" + m.channel; });
    accChart = new Chart(ctx, {
      type: "bar",
      data: {
        labels: labels,
        datasets: [
          { label: "grayscale", data: models.map(function (m) { return m.channel === "grayscale" ? m.test.accuracy : 0; }),
            backgroundColor: alpha(COL_1, 0.75), borderColor: COL_1, borderWidth: 1,
            borderRadius: 3, maxBarThickness: 20 },
          { label: "rgb", data: models.map(function (m) { return m.channel === "rgb" ? m.test.accuracy : 0; }),
            backgroundColor: alpha(COL_2, 0.75), borderColor: COL_2, borderWidth: 1,
            borderRadius: 3, maxBarThickness: 20 }
        ]
      },
      options: { responsive: true, maintainAspectRatio: false }
    });
    styleChart(accChart, "pct");
  }

  /* ---------------- binary ---------------- */

  function renderBinary(rows) {
    if (!rows || !rows.length) return;
    var test = rows.filter(function (r) { return r.split === "test"; });
    var tbody = document.querySelector("#tbl-binary tbody");
    test.forEach(function (r) {
      var tr = document.createElement("tr");
      if (String(r.scanner) === "mean") tr.className = "mean-line";
      tr.innerHTML =
        "<td class='model'>" + r.scanner + "</td>" +
        "<td class='chan " + r.channel + "'>" + r.channel + "</td>" +
        "<td class='num'>" + NUM_FMT(r.accuracy, 1) + "</td>" +
        "<td class='num'>" + NUM_FMT(r.f1, 1) + "</td>" +
        "<td class='num'>" + NUM_FMT(r.auc, 1) + "</td>" +
        heat(r.far, true) + heat(r.frr, false);
      tbody.appendChild(tr);
    });
  }

  function heat(v, isFAR) {
    if (v == null) return "<td class='num'>\u2014</td>";
    var w = Math.min(100, v * 100).toFixed(0);
    var cls = v <= 0.10 ? "good" : v <= 0.30 ? "" : "bad";
    return "<td class='num heat " + cls + "'>" + NUM_FMT(v, 1) +
      "<span class='bar'><i style='width:" + w + "%" + (isFAR ? ";background:var(--amber)" : "") + "'></i></span></td>";
  }

  /* ---------------- curves ---------------- */

  function renderCurves(curves) {
    RUNS = curves;
    var keys = Object.keys(curves).sort();
    if (!keys.length) return;
    var sel = document.getElementById("run");
    keys.forEach(function (k) {
      var o = document.createElement("option");
      o.value = k;
      o.textContent = k;
      sel.appendChild(o);
    });
    sel.addEventListener("change", function () { drawRun(sel.value); });
    drawRun(keys[0]);
  }

  function drawRun(name) {
    var rows = RUNS[name];
    if (!rows || !rows.length) return;
    var epochs = rows.map(function (r) { return r.epoch; });
    document.getElementById("run-meta").textContent =
      rows.length + " epochs \u00b7 last val_acc " + NUM_FMT(rows[rows.length - 1].val_acc, 1);

    draw("chart-loss", "loss", { train_loss: COL_1, val_loss: COL_2 }, false);
    draw("chart-acc-2", "accuracy", { train_acc: COL_1, val_acc: COL_2 }, true);

    function draw(id, label, series, pct) {
      var chart = id === "chart-loss" ? lossChart : acc2Chart;
      var ctx = document.getElementById(id).getContext("2d");
      if (chart) chart.destroy();
      var datasets = Object.keys(series).map(function (key) {
        return {
          label: key,
          data: rows.map(function (r) { return r[key]; }),
          borderColor: series[key],
          backgroundColor: series[key] + "22",
          fill: true, tension: 0.25, borderWidth: 1.6,
          pointRadius: 0
        };
      });
      var c = new Chart(ctx, { type: "line", data: { labels: epochs, datasets: datasets }, options: { responsive: true, maintainAspectRatio: false } });
      if (id === "chart-loss") { c.options.scales.y.title = { display: true, text: "loss" }; lossChart = c; }
      else { acc2Chart = c; }
      styleChart(c, pct ? "pct" : null);
    }
  }
function loadAcceptMatrix() {
    if (!loadAcceptMatrix._p) {
      loadAcceptMatrix._p = fetch("/static/matrix.json", { cache: "no-store" })
        .then(function (r) {
          if (!r.ok) throw new Error(r.status);
          return r.json();
        })
        .then(function (d) { return { per_scanner: d.per_scanner || {}, samples: d.samples || [] }; })
        .catch(function () { return { per_scanner: {}, samples: [] }; });
    }
    return loadAcceptMatrix._p;
  }

  function buildSampleGroups(row, onPick) {
    // This shared script is loaded independently of each page's local helpers.
    // Do not rely on a page-scoped fetchJSON() here: it leaves every picker empty.
    fetch("/api/samples", { cache: "no-store" }).then(function (r) {
      if (!r.ok) throw new Error(r.status);
      return r.json();
    }).then(function (d) {
      loadAcceptMatrix().then(function (mx) {
        var good = {};
        mx.samples.forEach(function (n) { good[n] = true; });
        var keep = d.samples.filter(function (s) { return good[s.name]; });
        row.textContent = "";
        keep.forEach(function (s) {
          var el = document.createElement("button");
          el.type = "button";
          el.className = "s";
          el.textContent = "S" + s.scanner + " \u00b7 " + s.name.replace(/\.bmp$/, "");
          el.onclick = function () {
            for (var i = 0; i < row.children.length; i++) row.children[i].classList.remove("on");
            el.classList.add("on");
            onPick(s);
          };
          row.appendChild(el);
        });
      });
    }).catch(function (e) { row.textContent = "samples unavailable: " + e.message; });
  }
  window.buildSampleGroups = buildSampleGroups;
})();
