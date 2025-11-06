// static/js/interactive_viewer.js - Interactive molecular viewer with slicing and selection

class MolecularViewer {
  constructor() {
    this.scene = null;
    this.camera = null;
    this.renderer = null;
    this.controls = null;
    
    this.atomMeshes = [];
    this.bondLines = null;
    this.selectedAtomIndex = null;
    
    this.clippingPlane = new THREE.Plane(new THREE.Vector3(0, 0, -1), 0);
    this.clippingEnabled = false;
    
    this.metadata = null;
    this.residueMap = [];
    this.hotspotData = {};
    this.rmsfData = {};
    this.currentFrame = 0;
    this.colorMode = 'hotspot';
    
    this.raycaster = new THREE.Raycaster();
    this.mouse = new THREE.Vector2();
    
    this.init();
  }

  async init() {
    this.setupScene();
    this.setupControls();
    this.setupEventListeners();
    
    try {
      await this.loadData();
      this.animate();
      this.updateStatus('Ready');
    } catch (error) {
      console.error('Initialization error:', error);
      this.updateStatus(`Error: ${error.message}`);
    }
  }

  setupScene() {
    // Scene
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x0a0e27);

    // Camera
    const container = document.getElementById('viewer');
    this.camera = new THREE.PerspectiveCamera(
      45,
      container.clientWidth / container.clientHeight,
      0.1,
      1000
    );
    this.camera.position.set(30, 30, 50);

    // Renderer
    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setSize(container.clientWidth, container.clientHeight);
    this.renderer.setPixelRatio(window.devicePixelRatio);
    this.renderer.localClippingEnabled = true;
    container.appendChild(this.renderer.domElement);

    // Lights
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
    this.scene.add(ambientLight);

    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.6);
    directionalLight.position.set(10, 10, 10);
    this.scene.add(directionalLight);

    const directionalLight2 = new THREE.DirectionalLight(0xffffff, 0.3);
    directionalLight2.position.set(-10, -10, -10);
    this.scene.add(directionalLight2);

    // Handle window resize
    window.addEventListener('resize', () => {
      const container = document.getElementById('viewer');
      this.camera.aspect = container.clientWidth / container.clientHeight;
      this.camera.updateProjectionMatrix();
      this.renderer.setSize(container.clientWidth, container.clientHeight);
    });
  }

  setupControls() {
    // OrbitControls - using simpler version for compatibility
    const controls = {
      enabled: true,
      target: new THREE.Vector3(0, 0, 0),
      update: () => {}
    };
    
    let isDragging = false;
    let previousMousePosition = { x: 0, y: 0 };
    
    this.renderer.domElement.addEventListener('mousedown', (e) => {
      if (e.button === 2) { // Right click
        isDragging = true;
        previousMousePosition = { x: e.clientX, y: e.clientY };
      }
    });
    
    this.renderer.domElement.addEventListener('mousemove', (e) => {
      if (isDragging) {
        const deltaX = e.clientX - previousMousePosition.x;
        const deltaY = e.clientY - previousMousePosition.y;
        
        const deltaRotationQuaternion = new THREE.Quaternion()
          .setFromEuler(new THREE.Euler(
            deltaY * 0.01,
            deltaX * 0.01,
            0,
            'XYZ'
          ));
        
        this.camera.position.applyQuaternion(deltaRotationQuaternion);
        this.camera.lookAt(this.scene.position);
        
        previousMousePosition = { x: e.clientX, y: e.clientY };
      }
    });
    
    this.renderer.domElement.addEventListener('mouseup', () => {
      isDragging = false;
    });
    
    this.renderer.domElement.addEventListener('wheel', (e) => {
      e.preventDefault();
      const delta = e.deltaY * 0.01;
      this.camera.position.multiplyScalar(1 + delta * 0.05);
    });
    
    this.controls = controls;
  }

  setupEventListeners() {
    // Color mode
    document.getElementById('colorMode').addEventListener('change', (e) => {
      this.colorMode = e.target.value;
      this.updateColors();
    });

    // Frame slider
    document.getElementById('frameSlider').addEventListener('input', (e) => {
      this.currentFrame = parseInt(e.target.value);
      document.getElementById('frameValue').textContent = this.currentFrame;
      this.loadFrame(this.currentFrame);
    });

    // Clipping controls
    document.getElementById('toggleClipping').addEventListener('click', () => {
      this.clippingEnabled = !this.clippingEnabled;
      const btn = document.getElementById('toggleClipping');
      const controls = document.getElementById('clippingControls');
      
      if (this.clippingEnabled) {
        btn.textContent = 'Disable Clipping';
        btn.classList.add('active');
        controls.style.display = 'block';
      } else {
        btn.textContent = 'Enable Clipping';
        btn.classList.remove('active');
        controls.style.display = 'none';
      }
      
      this.updateClipping();
    });

    document.getElementById('clipPosition').addEventListener('input', (e) => {
      const value = parseFloat(e.target.value);
      document.getElementById('clipPosValue').textContent = value.toFixed(1);
      this.clippingPlane.constant = value;
    });

    document.getElementById('clipAxis').addEventListener('change', (e) => {
      const axis = e.target.value;
      if (axis === 'x') {
        this.clippingPlane.normal.set(1, 0, 0);
      } else if (axis === 'y') {
        this.clippingPlane.normal.set(0, 1, 0);
      } else {
        this.clippingPlane.normal.set(0, 0, 1);
      }
    });

    // Click detection for atom selection
    this.renderer.domElement.addEventListener('click', (e) => {
      if (e.button === 2) return; // Ignore right clicks
      this.onAtomClick(e);
    });

    // Hover for tooltip
    this.renderer.domElement.addEventListener('mousemove', (e) => {
      this.onAtomHover(e);
    });
  }

  async loadData() {
    this.updateStatus('Loading metadata...');
    
    // Load metadata
    const metaResp = await fetch('/api/trajectory/meta');
    this.metadata = await metaResp.json();
    
    document.getElementById('infoFrames').textContent = this.metadata.n_frames;
    document.getElementById('infoAtoms').textContent = this.metadata.n_atoms;
    document.getElementById('infoResidues').textContent = this.metadata.n_residues;
    
    // Setup frame slider
    const slider = document.getElementById('frameSlider');
    slider.max = this.metadata.n_frames - 1;
    document.getElementById('frameMax').textContent = this.metadata.n_frames - 1;
    
    // Load residue map
    this.updateStatus('Loading residue map...');
    const mapResp = await fetch('/api/trajectory/residue_map');
    const mapData = await mapResp.json();
    this.residueMap = mapData.resnums;
    
    // Load hotspot data
    this.updateStatus('Loading hotspot data...');
    const hotspotResp = await fetch('/api/hotspots');
    const hotspotData = await hotspotResp.json();
    this.hotspotData = hotspotData.hotspots || {};
    
    // Load RMSF data
    this.updateStatus('Loading RMSF data...');
    const rmsfResp = await fetch('/api/rmsf');
    const rmsfData = await rmsfResp.json();
    this.rmsfData = rmsfData.rmsf || {};
    
    // Load first frame
    await this.loadFrame(0);
  }

  async loadFrame(frameIdx) {
    this.updateStatus(`Loading frame ${frameIdx}...`);
    
    const resp = await fetch(`/api/trajectory/frame/${frameIdx}`);
    const data = await resp.json();
    
    if (data.error) {
      this.updateStatus(`Error: ${data.error}`);
      return;
    }
    
    this.renderMolecule(data.xyz);
    this.updateStatus(`Frame ${frameIdx} loaded`);
  }

  renderMolecule(coordinates) {
    // Clear existing meshes
    this.atomMeshes.forEach(mesh => this.scene.remove(mesh));
    this.atomMeshes = [];
    
    if (this.bondLines) {
      this.scene.remove(this.bondLines);
      this.bondLines = null;
    }
    
    // Create atoms
    const atomGeometry = new THREE.SphereGeometry(0.3, 16, 16);
    
    coordinates.forEach((pos, idx) => {
      const material = new THREE.MeshPhongMaterial({
        color: 0xcccccc,
        clippingPlanes: this.clippingEnabled ? [this.clippingPlane] : []
      });
      
      const mesh = new THREE.Mesh(atomGeometry, material);
      mesh.position.set(pos[0], pos[1], pos[2]);
      mesh.userData = { atomIndex: idx };
      
      this.scene.add(mesh);
      this.atomMeshes.push(mesh);
    });
    
    // Update colors based on current mode
    this.updateColors();
    
    // Create bonds (simple approach: connect nearby atoms)
    this.createBonds(coordinates);
  }

  createBonds(coordinates) {
    const bondMaterial = new THREE.LineBasicMaterial({
      color: 0x555555,
      clippingPlanes: this.clippingEnabled ? [this.clippingPlane] : []
    });
    
    const bondGeometry = new THREE.BufferGeometry();
    const positions = [];
    
    // Simple bonding: connect atoms within threshold distance
    const bondThreshold = 1.8;
    
    for (let i = 0; i < coordinates.length; i++) {
      for (let j = i + 1; j < Math.min(i + 5, coordinates.length); j++) {
        const dist = Math.sqrt(
          Math.pow(coordinates[i][0] - coordinates[j][0], 2) +
          Math.pow(coordinates[i][1] - coordinates[j][1], 2) +
          Math.pow(coordinates[i][2] - coordinates[j][2], 2)
        );
        
        if (dist < bondThreshold) {
          positions.push(coordinates[i][0], coordinates[i][1], coordinates[i][2]);
          positions.push(coordinates[j][0], coordinates[j][1], coordinates[j][2]);
        }
      }
    }
    
    bondGeometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    this.bondLines = new THREE.LineSegments(bondGeometry, bondMaterial);
    this.scene.add(this.bondLines);
  }

  updateColors() {
    this.atomMeshes.forEach((mesh, idx) => {
      const resid = this.residueMap[idx];
      let color;
      
      switch (this.colorMode) {
        case 'hotspot':
          color = this.getHotspotColor(resid);
          break;
        case 'rmsf':
          color = this.getRMSFColor(resid);
          break;
        case 'residue':
          color = this.getResidueTypeColor(resid);
          break;
        case 'chain':
          color = this.getChainColor(idx);
          break;
        default:
          color = 0xcccccc;
      }
      
      mesh.material.color.setHex(color);
    });
  }

  getHotspotColor(resid) {
    const hotspot = this.hotspotData[resid];
    if (!hotspot || !hotspot.delta_err) {
      return 0x888888; // Gray for no data
    }
    
    // Map delta_err to color: blue (low) -> white (medium) -> red (high)
    const value = Math.min(Math.max(hotspot.delta_err, 0), 0.3) / 0.3;
    
    if (value < 0.5) {
      // Blue to white
      const t = value * 2;
      return this.interpolateColor(0x3b82f6, 0xffffff, t);
    } else {
      // White to red
      const t = (value - 0.5) * 2;
      return this.interpolateColor(0xffffff, 0xef4444, t);
    }
  }

  getRMSFColor(resid) {
    const rmsf = this.rmsfData[resid];
    if (!rmsf || rmsf.rmsf === null) {
      return 0x888888;
    }
    
    // Map RMSF to color
    const value = Math.min(Math.max(rmsf.rmsf, 0), 5) / 5;
    
    if (value < 0.5) {
      const t = value * 2;
      return this.interpolateColor(0x3b82f6, 0xffffff, t);
    } else {
      const t = (value - 0.5) * 2;
      return this.interpolateColor(0xffffff, 0xef4444, t);
    }
  }

  getResidueTypeColor(resid) {
    // Simple color by residue number
    const hue = (resid * 137.5) % 360;
    return new THREE.Color(`hsl(${hue}, 70%, 60%)`).getHex();
  }

  getChainColor(atomIdx) {
    // Color by position for now
    const segment = Math.floor((atomIdx / this.atomMeshes.length) * 5);
    const colors = [0x3b82f6, 0x10b981, 0xf59e0b, 0xef4444, 0x8b5cf6];
    return colors[segment % colors.length];
  }

  interpolateColor(color1, color2, t) {
    const c1 = new THREE.Color(color1);
    const c2 = new THREE.Color(color2);
    return c1.lerp(c2, t).getHex();
  }

  updateClipping() {
    this.atomMeshes.forEach(mesh => {
      mesh.material.clippingPlanes = this.clippingEnabled ? [this.clippingPlane] : [];
      mesh.material.needsUpdate = true;
    });
    
    if (this.bondLines) {
      this.bondLines.material.clippingPlanes = this.clippingEnabled ? [this.clippingPlane] : [];
      this.bondLines.material.needsUpdate = true;
    }
  }

  onAtomClick(event) {
    const rect = this.renderer.domElement.getBoundingClientRect();
    this.mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    this.mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
    
    this.raycaster.setFromCamera(this.mouse, this.camera);
    const intersects = this.raycaster.intersectObjects(this.atomMeshes);
    
    if (intersects.length > 0) {
      const atomIdx = intersects[0].object.userData.atomIndex;
      this.selectAtom(atomIdx);
    }
  }

  async selectAtom(atomIdx) {
    this.selectedAtomIndex = atomIdx;
    const resid = this.residueMap[atomIdx];
    
    // Highlight selected atom
    this.atomMeshes.forEach((mesh, idx) => {
      if (idx === atomIdx) {
        mesh.material.emissive.setHex(0x444444);
      } else {
        mesh.material.emissive.setHex(0x000000);
      }
    });
    
    // Fetch residue details
    const resp = await fetch(`/api/residue/${resid}`);
    const data = await resp.json();
    
    this.displayResidueInfo(data);
  }

  displayResidueInfo(data) {
    const container = document.getElementById('residueInfo');
    
    let html = `
      <div class="info-item">
        <div class="info-label">Residue</div>
        <div class="info-value">${data.resname} ${data.resid}</div>
      </div>
      <div class="info-item">
        <div class="info-label">Chain</div>
        <div class="info-value">${data.chain || 'N/A'}</div>
      </div>
    `;
    
    if (data.hotspot && data.hotspot.delta_err !== undefined) {
      html += `
        <div class="info-item">
          <div class="info-label">Hotspot Score</div>
          <div class="info-value">${data.hotspot.delta_err.toFixed(4)}</div>
        </div>
      `;
    }
    
    if (data.rmsf_data && data.rmsf_data.rmsf !== null) {
      html += `
        <div class="info-item">
          <div class="info-label">RMSF (Flexibility)</div>
          <div class="info-value">${data.rmsf_data.rmsf.toFixed(3)} Å</div>
        </div>
      `;
    }
    
    if (data.rmsf_data && data.rmsf_data.sasa !== null) {
      html += `
        <div class="info-item">
          <div class="info-label">SASA</div>
          <div class="info-value">${data.rmsf_data.sasa.toFixed(1)} Ų</div>
        </div>
      `;
    }
    
    container.innerHTML = html;
  }

  onAtomHover(event) {
    const rect = this.renderer.domElement.getBoundingClientRect();
    this.mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    this.mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
    
    this.raycaster.setFromCamera(this.mouse, this.camera);
    const intersects = this.raycaster.intersectObjects(this.atomMeshes);
    
    const tooltip = document.getElementById('tooltip');
    
    if (intersects.length > 0) {
      const atomIdx = intersects[0].object.userData.atomIndex;
      const resid = this.residueMap[atomIdx];
      
      tooltip.style.display = 'block';
      tooltip.style.left = (event.clientX + 10) + 'px';
      tooltip.style.top = (event.clientY + 10) + 'px';
      tooltip.textContent = `Atom ${atomIdx} • Residue ${resid}`;
    } else {
      tooltip.style.display = 'none';
    }
  }

  updateStatus(message) {
    document.getElementById('status').textContent = message;
  }

  animate() {
    requestAnimationFrame(() => this.animate());
    
    if (this.controls.update) {
      this.controls.update();
    }
    
    this.renderer.render(this.scene, this.camera);
  }
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
  new MolecularViewer();
});
