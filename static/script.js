(() => {
  const MAX_BYTES = 5 * 1024 * 1024;
  const ALLOWED = ["jpg", "jpeg", "png", "bmp", "tif", "tiff"];
  const $ = (id) => document.getElementById(id);
  const fileInput = $("file"), dropzone = $("dropzone"), preview = $("preview");
  const btn = $("predict-btn"), errBox = $("error"), resultBox = $("result");
  let selected = null;

  function showError(msg) { errBox.textContent = msg; errBox.hidden = false; resultBox.hidden = true; }
  function clearError() { errBox.hidden = true; }

  function setFile(file) {
    clearError(); resultBox.hidden = true; selected = null; btn.disabled = true;
    if (!file) return;
    const ext = file.name.split(".").pop().toLowerCase();
    if (!ALLOWED.includes(ext)) return showError("Unsupported file type. Use JPG, PNG, BMP or TIFF.");
    if (file.size === 0) return showError("The selected file is empty.");
    if (file.size > MAX_BYTES) return showError("File is too large (max 5 MB).");
    selected = file;
    btn.disabled = false;
    preview.src = URL.createObjectURL(file);
    preview.hidden = false;
    $("dz-text").hidden = true;
  }

  fileInput.addEventListener("change", () => setFile(fileInput.files[0]));
  ["dragenter", "dragover"].forEach((e) => dropzone.addEventListener(e, (ev) => { ev.preventDefault(); dropzone.classList.add("drag"); }));
  ["dragleave", "drop"].forEach((e) => dropzone.addEventListener(e, (ev) => { ev.preventDefault(); dropzone.classList.remove("drag"); }));
  dropzone.addEventListener("drop", (ev) => { if (ev.dataTransfer.files.length) setFile(ev.dataTransfer.files[0]); });

  btn.addEventListener("click", async () => {
    if (!selected) return;
    clearError(); btn.disabled = true; btn.textContent = "Analyzing…";
    const form = new FormData();
    form.append("file", selected);
    try {
      const res = await fetch("/api/predict", { method: "POST", body: form });
      const data = await res.json().catch(() => ({ error: "Unexpected server response." }));
      if (!res.ok) throw new Error(data.error || "Request failed.");
      render(data);
    } catch (e) {
      showError(e.message || "Network error.");
    } finally {
      btn.disabled = false; btn.textContent = "Classify";
    }
  });

  function render(d) {
    const label = $("res-label");
    label.textContent = d.label;
    label.classList.toggle("uncertain", d.uncertain);
    $("res-conf").textContent = (d.confidence * 100).toFixed(1) + "% confidence";
    $("warning").hidden = !d.uncertain;
    const bars = $("bars"); bars.innerHTML = "";
    Object.entries(d.probabilities).sort((a, b) => b[1] - a[1]).forEach(([name, p]) => {
      const row = document.createElement("div"); row.className = "bar-row";
      const n = document.createElement("span"); n.className = "name"; n.textContent = name;
      const track = document.createElement("div"); track.className = "track";
      const fill = document.createElement("div"); fill.className = "fill"; fill.style.width = (p * 100) + "%";
      track.appendChild(fill);
      const pct = document.createElement("span"); pct.className = "pct"; pct.textContent = (p * 100).toFixed(1) + "%";
      row.append(n, track, pct); bars.appendChild(row);
    });
    resultBox.hidden = false;
  }
})();
