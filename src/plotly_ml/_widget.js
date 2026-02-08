/* Pairplot crossfilter – anywidget ESM module
 *
 * Linked lasso / box selection across scatter subplots.
 *
 * Strategy: on every selection event we collect the selected row-IDs
 * from ``customdata``, then set per-point ``marker.opacity`` arrays
 * on every scatter trace via ``Plotly.restyle``.  We simultaneously
 * clear ``selectedpoints`` so Plotly's native selection rendering
 * does not conflict with our custom opacity.
 */

/* ------------------------------------------------------------------ */
/*  Plotly.js loader                                                   */
/* ------------------------------------------------------------------ */
let _plotlyPromise = null;

function ensurePlotly() {
  if (window.Plotly) return Promise.resolve(window.Plotly);
  if (_plotlyPromise) return _plotlyPromise;

  _plotlyPromise = new Promise(function (resolve, reject) {
    var existing = document.querySelector(
      'script[src*="cdn.plot.ly/plotly"]'
    );
    if (existing) {
      var check = setInterval(function () {
        if (window.Plotly) {
          clearInterval(check);
          resolve(window.Plotly);
        }
      }, 50);
      return;
    }

    var s = document.createElement("script");
    s.src = "https://cdn.plot.ly/plotly-2.35.2.min.js";
    s.onload = function () {
      resolve(window.Plotly);
    };
    s.onerror = function () {
      reject(new Error("Failed to load Plotly.js from CDN"));
    };
    document.head.appendChild(s);
  });
  return _plotlyPromise;
}

/* ------------------------------------------------------------------ */
/*  Crossfilter logic                                                  */
/* ------------------------------------------------------------------ */

function setupCrossfilter(gd, Plotly, statusEl, combineMode) {
  function log(msg) {
    console.log("[pairplot-xf]", msg);
    if (statusEl) statusEl.textContent = "XF: " + msg;
  }

  // Avoid duplicate handlers if setupCrossfilter is called multiple times
  // (e.g., after Plotly.react). Plotly graph divs expose Node-style
  // event emitter methods in most builds.
  try {
    if (gd && gd.removeAllListeners) {
      gd.removeAllListeners("plotly_selected");
      gd.removeAllListeners("plotly_deselect");
      gd.removeAllListeners("plotly_doubleclick");
    }
  } catch (e) {
    /* best-effort */
  }

  /* Gather scatter-marker traces with customdata (row-IDs). */
  var scatterIndices = [];
  for (var i = 0; i < gd.data.length; i++) {
    var t = gd.data[i];
    if (
      t &&
      t.mode &&
      t.mode.indexOf("markers") !== -1 &&
      t.customdata &&
      t.customdata.length > 0
    ) {
      scatterIndices.push(i);
    }
  }

  log(scatterIndices.length + " linked traces \u2013 lasso or box select");
  if (scatterIndices.length === 0) return;

  /* Guard: prevent plotly_deselect feedback loop after our restyle. */
  var _busy = false;
  var _currentIds = null; /* Set|null */

  function toKey(v) {
    return Array.isArray(v) ? String(v[0]) : String(v);
  }

  function pointId(pt) {
    if (!pt) return null;
    if (pt.customdata != null) return toKey(pt.customdata);

    // Some Plotly builds omit point.customdata; fall back to trace customdata.
    var curve = pt.curveNumber;
    var idx =
      pt.pointNumber != null
        ? pt.pointNumber
        : pt.pointIndex != null
          ? pt.pointIndex
          : null;
    if (curve == null || idx == null) return null;

    var trace = (gd && gd.data) ? gd.data[curve] : null;
    if (!trace || !trace.customdata || idx >= trace.customdata.length) return null;
    return toKey(trace.customdata[idx]);
  }

  /**
   * Apply linked selection by setting Plotly-native `selectedpoints` on every
   * scatter trace. This is more reliable than trying to drive per-point
   * `marker.opacity` arrays.
   * @param {Set|null} selectedIds  Row-IDs to highlight, or null to clear.
   */
  function applySelection(selectedIds) {
    _busy = true;

    var sps = []; /* per-trace selectedpoints arrays (or null) */
    for (var k = 0; k < scatterIndices.length; k++) {
      if (!selectedIds) {
        sps.push(null);
        continue;
      }

      var cd = gd.data[scatterIndices[k]].customdata;
      var idxs = [];
      for (var j = 0; j < cd.length; j++) {
        if (selectedIds.has(toKey(cd[j]))) idxs.push(j);
      }
      sps.push(idxs);
    }

    var update;
    if (!selectedIds) {
      // Clear selection and ensure no dimming remains.
      update = {
        selectedpoints: sps,
        selected: { marker: { opacity: 1.0 } },
        unselected: { marker: { opacity: 1.0 } },
      };
    } else {
      update = {
        selectedpoints: sps,
        selected: { marker: { opacity: 1.0 } },
        unselected: { marker: { opacity: 0.1 } },
      };
    }

    Plotly.restyle(gd, update, scatterIndices)
      .then(function () {
        log(
          selectedIds
            ? selectedIds.size + " rows highlighted"
            : "selection cleared"
        );
        setTimeout(function () {
          _busy = false;
        }, 200);
      })
      .catch(function (err) {
        log(
          "restyle error: " +
            (err && err.message ? err.message : String(err))
        );
        _busy = false;
      });
  }

  /* --- event wiring ----------------------------------------------- */

  gd.on("plotly_selected", function (eventData) {
    if (_busy) return;
    if (
      !eventData ||
      !eventData.points ||
      eventData.points.length === 0
    ) {
      applySelection(null);
      return;
    }
    var ids = new Set();
    eventData.points.forEach(function (pt) {
      var pid = pointId(pt);
      if (pid != null) ids.add(pid);
    });
    var next = ids.size > 0 ? ids : null;
    if (!next) {
      _currentIds = null;
      applySelection(null);
      return;
    }

    var mode = (combineMode || "replace").toLowerCase();
    if (!_currentIds || mode === "replace") {
      _currentIds = next;
    } else if (mode === "union") {
      var u = new Set(_currentIds);
      next.forEach(function (v) {
        u.add(v);
      });
      _currentIds = u;
    } else if (mode === "intersection") {
      var inter = new Set();
      _currentIds.forEach(function (v) {
        if (next.has(v)) inter.add(v);
      });
      _currentIds = inter.size ? inter : null;
    } else if (mode === "difference") {
      // A\\B: remove newly selected points from existing selection
      var diff = new Set(_currentIds);
      next.forEach(function (v) {
        diff.delete(v);
      });
      _currentIds = diff.size ? diff : null;
    } else {
      _currentIds = next;
    }

    log(
      "selected: " +
        (_currentIds ? _currentIds.size : 0) +
        " rows (mode=" +
        mode +
        ")"
    );
    applySelection(_currentIds);
  });

  gd.on("plotly_deselect", function () {
    if (_busy) return;
    _currentIds = null;
    applySelection(null);
  });
  gd.on("plotly_doubleclick", function () {
    if (_busy) return;
    _currentIds = null;
    applySelection(null);
  });
}

/* ------------------------------------------------------------------ */
/*  anywidget render entry-point                                       */
/* ------------------------------------------------------------------ */

export default {
  async render({ model, el }) {
    var Plotly = await ensurePlotly();

    var status = null;

    var container = document.createElement("div");
    container.style.width = "100%";
    el.appendChild(container);

    var fig = JSON.parse(model.get("fig_json"));

    var meta = (fig && fig.layout && fig.layout.meta) ? fig.layout.meta : {};
    var combineMode = meta && meta.pairplot_xf_combine ? meta.pairplot_xf_combine : "replace";
    var showStatus = !!(meta && meta.pairplot_xf_show_status);

    if (showStatus) {
      /* Optional status badge – hidden by default. */
      status = document.createElement("div");
      status.style.cssText =
        "font-size:12px;color:#333;padding:4px 8px;font-family:monospace;" +
        "background:#f5f5f5;border:1px solid #ddd;border-radius:6px;" +
        "margin:2px 0 6px 0;display:inline-block;";
      el.appendChild(status);
    }

    await Plotly.newPlot(container, fig.data, fig.layout, {
      responsive: true,
      scrollZoom: true,
    });

    setupCrossfilter(container, Plotly, status, combineMode);

    model.on("change:fig_json", async function () {
      var newFig = JSON.parse(model.get("fig_json"));
      await Plotly.react(container, newFig.data, newFig.layout);
      var newMeta = (newFig && newFig.layout && newFig.layout.meta)
        ? newFig.layout.meta
        : {};
      var newCombineMode = newMeta && newMeta.pairplot_xf_combine
        ? newMeta.pairplot_xf_combine
        : "replace";
      setupCrossfilter(container, Plotly, status, newCombineMode);
    });

    return function () {
      Plotly.purge(container);
    };
  },
};
