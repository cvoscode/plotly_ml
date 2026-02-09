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

function ensurePlotly(plotlyJsText) {
  if (window.Plotly) return Promise.resolve(window.Plotly);
  if (_plotlyPromise) return _plotlyPromise;

  function tryRequire() {
    return new Promise(function (resolve, reject) {
      var req = window.requirejs || window.require;
      if (!req) {
        reject(new Error("requirejs not available"));
        return;
      }

      var tried = [];
      function attempt(names, pick) {
        return new Promise(function (res, rej) {
          tried.push(names.join(","));
          try {
            req(names, function () {
              try {
                var args = Array.prototype.slice.call(arguments);
                var maybe = pick.apply(null, args);
                if (maybe) {
                  res(maybe);
                } else {
                  rej(new Error("module loaded but Plotly not found"));
                }
              } catch (e) {
                rej(e);
              }
            }, function (err) {
              rej(err || new Error("require() failed"));
            });
          } catch (e) {
            rej(e);
          }
        });
      }

      // Best-effort: different frontends expose Plotly differently.
      // - Some register a 'plotly' module that is the Plotly object.
      // - Some expose Plotly as a property on plotlywidget.
      // - Some only set window.Plotly.
      attempt(["plotly"], function (m) {
        return m && (m.Plotly || m.default || m);
      })
        .catch(function () {
          return attempt(["plotlywidget"], function (w) {
            return w && (w.Plotly || (w.default && w.default.Plotly) || null);
          });
        })
        .then(function (Plotly) {
          if (Plotly) {
            window.Plotly = window.Plotly || Plotly;
            resolve(Plotly);
          } else {
            reject(new Error("Plotly not found via require(): " + tried.join(" | ")));
          }
        })
        .catch(function (err) {
          err = err || new Error("Plotly not found via require()");
          err.message = "Plotly not found via require(): " + tried.join(" | ") + "\n" + err.message;
          reject(err);
        });
    });
  }

  _plotlyPromise = new Promise(function (resolve, reject) {
    // Attempt 1: Jupyter/requirejs-provided Plotly (no remote scripts)
    tryRequire()
      .then(function (Plotly) {
        resolve(Plotly);
      })
      .catch(function () {
        // Attempt 2: load embedded Plotly.js from the widget model.
        // This avoids network access and works in restricted environments.

        function loadFromEmbedded(text) {
          return new Promise(function (res, rej) {
            if (!text || typeof text !== "string" || text.length < 1000) {
              rej(new Error("No embedded Plotly.js available"));
              return;
            }

            try {
              var blob = new Blob([text], { type: "text/javascript" });
              var url = URL.createObjectURL(blob);
              var s = document.createElement("script");
              s.src = url;
              s.async = true;
              s.onload = function () {
                // Revoke URL after load; Plotly should now be on window.
                try {
                  URL.revokeObjectURL(url);
                } catch (e) {
                  /* best-effort */
                }
                if (window.Plotly) {
                  res(window.Plotly);
                } else {
                  rej(new Error("Embedded Plotly.js loaded but window.Plotly is missing"));
                }
              };
              s.onerror = function () {
                try {
                  URL.revokeObjectURL(url);
                } catch (e) {
                  /* best-effort */
                }
                rej(new Error("Failed to load embedded Plotly.js"));
              };
              document.head.appendChild(s);
            } catch (e) {
              rej(e);
            }
          });
        }

        loadFromEmbedded(plotlyJsText)
          .then(function (Plotly) {
            resolve(Plotly);
          })
          .catch(function () {
            // Attempt 3: load from a CDN via <script>.
        // VS Code's Jupyter widget sandbox may block some hosts unless allowed
        // via the "Jupyter: Widget Script Sources" setting. jsDelivr/unpkg are
        // commonly allowed by default.

        function waitForPlotly(timeoutMs) {
          return new Promise(function (res, rej) {
            var start = Date.now();
            var t = setInterval(function () {
              if (window.Plotly) {
                clearInterval(t);
                res(window.Plotly);
              } else if (Date.now() - start > (timeoutMs || 8000)) {
                clearInterval(t);
                rej(new Error("Timed out waiting for window.Plotly"));
              }
            }, 50);
          });
        }

        function loadScript(url) {
          return new Promise(function (res, rej) {
            // If already present, just wait for Plotly.
            var existing = document.querySelector('script[src="' + url + '"]');
            if (existing) {
              waitForPlotly(8000).then(res).catch(rej);
              return;
            }

            var s = document.createElement("script");
            s.src = url;
            s.async = true;
            s.onload = function () {
              waitForPlotly(8000).then(res).catch(rej);
            };
            s.onerror = function () {
              rej(new Error("Failed to load script: " + url));
            };
            document.head.appendChild(s);
          });
        }

        var urls = [
          "https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js",
          "https://unpkg.com/plotly.js-dist-min@2.35.2/plotly.min.js",
          "https://cdn.plot.ly/plotly-2.35.2.min.js",
        ];

        (function tryNext(i, lastErr) {
          if (i >= urls.length) {
            reject(
              new Error(
                "Failed to load Plotly.js from any CDN. Last error: " +
                  (lastErr && lastErr.message ? lastErr.message : String(lastErr))
              )
            );
            return;
          }
          loadScript(urls[i])
            .then(function (Plotly) {
              resolve(Plotly);
            })
            .catch(function (err) {
              tryNext(i + 1, err);
            });
        })(0, null);
          });
      });

    // Note: reject is handled by the CDN loader branch.
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
    var Plotly;
    try {
      var plotlyJsText = model.get("plotly_js");
      Plotly = await ensurePlotly(plotlyJsText);
    } catch (err) {
      var msg =
        "Plotly ML pairplot widget failed to load Plotly.js.\n\n" +
        "This usually happens when the notebook frontend blocks remote scripts.\n" +
        "The widget tries to load Plotly.js from the embedded bundle first, then (jsDelivr / unpkg / cdn.plot.ly).\n\n" +
        "Fixes:\n" +
        "- Trust the notebook/workspace (VS Code restricted mode disables widget JS).\n" +
        "- Ensure VS Code Jupyter widget rendering is enabled (Jupyter + Jupyter Renderers).\n" +
        "- In VS Code settings, allow the needed host(s) in 'Jupyter: Widget Script Sources' (try allowing jsDelivr / unpkg), then restart the kernel.\n\n" +
        "- If you are offline: restart the kernel so the embedded Plotly.js is sent to the widget.\n\n" +
        "Error: " +
        (err && err.message ? err.message : String(err));

      var pre = document.createElement("pre");
      pre.textContent = msg;
      pre.style.whiteSpace = "pre-wrap";
      pre.style.fontFamily = "ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace";
      pre.style.fontSize = "12px";
      pre.style.padding = "10px";
      pre.style.border = "1px solid #ddd";
      pre.style.borderRadius = "8px";
      pre.style.background = "#fafafa";
      el.appendChild(pre);
      return function () {};
    }

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
