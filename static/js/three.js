(async function () {
  const stage = new NGL.Stage("viewer", { backgroundColor: "white" });

  // Handle HiDPI & resizing
  window.addEventListener("resize", () => stage.handleResize(), false);

  // Load your multi-model PDB served by Flask
  const pdbUrl = "/file/data/multi_model_anomaly.pdb";
  const comp = await stage.loadFile(pdbUrl, { ext: "pdb" });

  // Cartoon by B-factor (your anomaly score)
  let rep = comp.addRepresentation("cartoon", {
    colorScheme: "bfactor",
    colorScale: "RdYlBu", // NGL builtin diverging scale
    opacity: 1.0,
    aspectRatio: 3.0,
    smoothSheet: true,
  });

  // Fit to screen
  comp.autoView();

  // Figure out how many models we have (multi-model PDB)
  // NGL sets atom.modelIndex; extract max to set slider max.
  const atomStore = comp.structure.getView(new NGL.Selection("")).structure.atomStore;
  let maxModel = 0;
  for (let i = 0; i < atomStore.count; i++) {
    const m = atomStore.modelIndex[i];
    if (m > maxModel) maxModel = m;
  }

  const slider = document.getElementById("modelSlider");
  const modelVal = document.getElementById("modelVal");
  slider.max = String(maxModel);
  slider.value = "0";
  modelVal.textContent = "0";

  // Function to isolate a single model
  function setModel(m) {
    // Update representation selection to only show given model
    const sel = new NGL.Selection(`@MODEL ${m + 1}`); // PDB MODEL records are 1-based
    comp.removeAllRepresentations();
    rep = comp.addRepresentation("cartoon", {
      sele: sel.string,
      colorScheme: document.getElementById("scheme").value,
      colorScale: "RdYlBu",
      opacity: 1.0,
      aspectRatio: 3.0,
      smoothSheet: true,
    });
    comp.autoView(sel);
  }

  // slider events
  slider.addEventListener("input", () => {
    const m = parseInt(slider.value, 10);
    modelVal.textContent = String(m);
    setModel(m);
  });

  // color scheme switcher
  document.getElementById("scheme").addEventListener("change", () => {
    const m = parseInt(slider.value, 10);
    setModel(m);
  });

  // Start at model 0
  setModel(0);
})();
