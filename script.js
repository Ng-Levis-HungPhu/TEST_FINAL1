const canvas = document.getElementById("rocketCanvas");
canvas.width = 1200;
canvas.height = 400;

const aspect = canvas.width / canvas.height;
const zoom = 1.75;
const camera = new THREE.OrthographicCamera(
  -canvas.width / zoom, canvas.width / zoom,
  canvas.height / zoom, -canvas.height / zoom,
  1, 2000
);
camera.position.set(300, 80, 350);
camera.lookAt(0, 0, 0);

const scene = new THREE.Scene();

const light = new THREE.PointLight(0xffffff, 1.2);
camera.add(light);
scene.add(camera);
const ambientLight = new THREE.AmbientLight(0xffffff, 0.3);
scene.add(ambientLight);

const backLight = new THREE.DirectionalLight(0x8888ff, 0.4);
backLight.position.set(-100, -100, -200);
scene.add(backLight);

const renderer = new THREE.WebGLRenderer({ canvas, alpha: true });
renderer.setSize(canvas.width, canvas.height);

const controls = new THREE.OrbitControls(camera, renderer.domElement);
controls.target.set(0, 0, 0);
controls.enableDamping = true;
controls.dampingFactor = 0.1;
controls.rotateSpeed = 0.5;
controls.zoomSpeed = 0.5;
controls.enablePan = false;
controls.update();

const axesHelper = new THREE.AxesHelper(100);
axesHelper.position.set(0, 50, 0);
scene.add(axesHelper);
const boxGeometry = new THREE.BoxGeometry(10, 10, 10);
const boxMaterial = new THREE.MeshStandardMaterial({ color: 0xff0000 });
const redBox = new THREE.Mesh(boxGeometry, boxMaterial);
redBox.position.set(0, 50, 0);
scene.add(redBox);


let rocketMesh = null;
let rocketGroup = new THREE.Group();
scene.add(rocketGroup);

function createWing(root, tip, height) {
  const shape = new THREE.Shape();
 
  shape.moveTo(0, 0);
  shape.lineTo(root, 0);
  shape.lineTo(tip, height);
  shape.lineTo(0, height);
  shape.lineTo(0, 0);

  shape.autoClose = true;

  const extrudeSettings = { depth: 0.75, bevelEnabled: false};
  const geometry = new THREE.ExtrudeGeometry(shape, extrudeSettings);
  const material = new THREE.MeshPhysicalMaterial({ color: 0x999999});
  return new THREE.Mesh(geometry, material);
}

function createFin1(root, tip, height, swept) {
  const shape = new THREE.Shape();

  const x1 = Math.tan(swept * Math.PI/180)
  const x2 = Math.tan(16.54 * Math.PI/180)
 
  shape.moveTo(0, 0);
  shape.lineTo(root, 0);
  shape.lineTo(tip - height*(x1 - x2), height);
  shape.lineTo(- height*(x1 - x2), height);
  shape.lineTo(0, 0);

  shape.autoClose = true;

  const extrudeSettings = { depth: 0.75, bevelEnabled: false };
  const geometry = new THREE.ExtrudeGeometry(shape, extrudeSettings);
  const material = new THREE.MeshPhysicalMaterial({ color: 0x666666 });
  return new THREE.Mesh(geometry, material);
}

function createFin2(root, tip, height, swept) {
  const shape = new THREE.Shape();
 
  shape.moveTo(0, 0);
  shape.lineTo(root, 0);
  shape.lineTo(tip, height);
  shape.lineTo(0, height);
  shape.lineTo(0, 0);

  shape.autoClose = true;

  const extrudeSettings = { depth: 0.75, bevelEnabled: false };
  const geometry = new THREE.ExtrudeGeometry(shape, extrudeSettings);
  const material = new THREE.MeshPhysicalMaterial({ color: 0x666666 });
  return new THREE.Mesh(geometry, material);
}

function createRocketModel() {
  const cssVars = getComputedStyle(document.body);

  const diameter = parseFloat(cssVars.getPropertyValue('--variable-collection-diameter'));
  const length = parseFloat(cssVars.getPropertyValue('--variable-collection-length'));

  const ln = parseFloat(document.getElementById('lnInput').value);
  const swept = parseFloat(document.getElementById('sweptInput').value);

  const lw = parseFloat(cssVars.getPropertyValue('--variable-collection-LW'));
  const crw = parseFloat(cssVars.getPropertyValue('--variable-collection-CRW'));
  const ctw = parseFloat(cssVars.getPropertyValue('--variable-collection-CTW'));
  const bw = parseFloat(cssVars.getPropertyValue('--variable-collection-BW'));

  const lf = parseFloat(cssVars.getPropertyValue('--variable-collection-LF'));
  const crf = parseFloat(cssVars.getPropertyValue('--variable-collection-CRF'));
  const ctf = parseFloat(cssVars.getPropertyValue('--variable-collection-CTF'));
  const bf = parseFloat(cssVars.getPropertyValue('--variable-collection-BF'));

  const mode = document.getElementById('dropdownBtn').textContent.split(": ")[1];

  const bodyLength = length - ln;

  const scaledDiameter = diameter * 10;
  const scaledBodyLength = bodyLength * 10;
  const scaledNoseLength = ln * 10;

  const scaledLW = lw * 10;
  const scaledCRW = crw * 10;
  const scaledCTW = ctw * 10;
  const scaledBW = bw * 10;

  const scaledLF = lf * 10;
  const scaledCRF = crf * 10;
  const scaledCTF = ctf * 10;
  const scaledBF = bf * 10;


  if (!rocketGroup) {
    rocketGroup = new THREE.Group();
    scene.add(rocketGroup);
  } else {
    rocketGroup.clear();
  }

  // === THÂN TÊN LỬA ===
  const bodyGeometry = new THREE.CylinderGeometry(
    scaledDiameter / 2,
    scaledDiameter / 2,
    scaledBodyLength,
    32
  );
  const bodyMaterial = new THREE.MeshPhysicalMaterial({ color: 0x00ccff });
  const bodyMesh = new THREE.Mesh(bodyGeometry, bodyMaterial);
  bodyMesh.rotation.set(0, 0, Math.PI / 2);
  bodyMesh.position.set(scaledBodyLength/2, 0, 0);
  rocketGroup.add(bodyMesh);

  // === MŨI OGIVE ===
  const points = [];
  const segments = 50;
  const noseRadius = scaledDiameter / 2;
  for (let i = 0; i <= segments; i++) {
    const t = i / segments;
    const x = noseRadius * Math.sqrt(1 - Math.pow(t - 1, 2));
    const y = t * scaledNoseLength;
    points.push(new THREE.Vector2(x, y));
  }

  const noseGeometry = new THREE.LatheGeometry(points, 64);
  const noseMaterial = new THREE.MeshPhysicalMaterial({ color: 0x0077be });
  const noseMesh = new THREE.Mesh(noseGeometry, noseMaterial);
  noseMesh.rotation.set(0, 0, Math.PI / 2);
  noseMesh.position.set(scaledNoseLength+scaledBodyLength, 0, 0);
  rocketGroup.add(noseMesh);

  // === GẮN 1 CANARD LÊN THÂN ===
  const wingX = scaledBodyLength+scaledNoseLength-scaledCRW-scaledLW;
  const wingDistance = scaledDiameter / 2;
  
  const wingOffsets = [
    { y: wingDistance, z: 0 },
    { y: 0, z:  wingDistance },
    { y: -wingDistance, z: 0 },
    { y: 0, z: -wingDistance }
  ];
  
  wingOffsets.forEach((offset, index) => {
    const wing = createWing(scaledCRW, scaledCTW, scaledBW);
    wing.rotation.set(index * Math.PI / 2, 0, 0);
    wing.position.set(wingX, offset.y, offset.z);
    rocketGroup.add(wing);
  });
  // Gắn FIN
  const finX = scaledBodyLength+scaledNoseLength-scaledCRF-scaledLF;
  const finDistance = scaledDiameter / 2;
  
  const finOffsets = [
    { y: finDistance, z: 0 },
    { y: 0, z:  finDistance },
    { y: -finDistance, z: 0 },
    { y: 0, z: -finDistance }
  ];
  
  let createFinFunc;

  if (mode === "NASA") {
    createFinFunc = createFin1;
  } else {
    createFinFunc = createFin2;
  }

  finOffsets.forEach((offset, index) => {
    const fin = createFinFunc(scaledCRF, scaledCTF, scaledBF, swept);
    fin.rotation.set(index * Math.PI / 2, 0, 0);
    fin.position.set(finX, offset.y, offset.z);
    rocketGroup.add(fin);
  });

  // === CĂN MÔ HÌNH GIỮA KHUNG ===
  rocketGroup.position.set(-500, 0, 0);
}

function updateBodyLength() {
  const cssVars = getComputedStyle(document.body);
  let length = parseFloat(cssVars.getPropertyValue('--variable-collection-length').replace("px", "").trim());
  const ln = parseFloat(document.getElementById('lnInput').value);

  if (isNaN(length) || isNaN(ln)) {
    document.getElementById('lengthInput').value = '0.00';
    return;
  }

  const bodyLength = length - ln;
  document.getElementById('lengthInput').value = bodyLength.toFixed(2);
  createRocketModel();
}

document.getElementById('lnInput').addEventListener('input', updateBodyLength);
document.getElementById('sweptInput').addEventListener('input', updateBodyLength);
updateBodyLength();

const dropdownBtn = document.getElementById("dropdownBtn");
const dropdownList = document.getElementById("dropdownList");
const body = document.body;

dropdownBtn.addEventListener("click", () => {
  dropdownList.style.display = dropdownList.style.display === "block" ? "none" : "block";
});

dropdownList.querySelectorAll("a").forEach(item => {
  item.addEventListener("click", e => {
    e.preventDefault();
    const mode = item.dataset.mode;
    body.setAttribute("data-variable-collection-mode", mode);
    dropdownBtn.textContent = `Model: ${mode}`;
    dropdownList.style.display = "none";
    document.querySelectorAll("input[type='range'], input[type='number']").forEach(input => {
      input.value = input.defaultValue;
    });
    updateDisplayedValues();
    updateFieldLocking(mode);
    updateBodyLength();
  });
});

document.addEventListener("click", e => {
  if (!e.target.closest(".dropdown")) dropdownList.style.display = "none";
});

function updateDisplayedValues() {
  const cssVars = getComputedStyle(body);
  const getVar = name => parseFloat(cssVars.getPropertyValue(name))?.toFixed(2) || "0.00";

  const values = {
    length: getVar("--variable-collection-length"),
    diameter: getVar("--variable-collection-diameter"),
    LW: getVar("--variable-collection-LW"),
    LF: getVar("--variable-collection-LF"),
    CTW: getVar("--variable-collection-CTW"),
    CRW: getVar("--variable-collection-CRW"),
    CTF: getVar("--variable-collection-CTF"),
    CRF: getVar("--variable-collection-CRF"),
    BW: getVar("--variable-collection-BW"),
    BF: getVar("--variable-collection-BF"),
  };

  document.querySelectorAll(".text-wrapper-3")[0].textContent = values.length;
  document.querySelectorAll(".text-wrapper-3")[1].textContent = values.LF;
  document.querySelectorAll(".text-wrapper-3")[2].textContent = values.LW;
  document.querySelector(".text-wrapper-6").textContent = values.diameter;
  document.querySelectorAll(".text-wrapper-8")[0].textContent = values.CTW;
  document.querySelectorAll(".text-wrapper-6")[1].textContent = values.CRW;
  document.querySelectorAll(".text-wrapper-8")[1].textContent = values.CTF;
  document.querySelectorAll(".text-wrapper-8")[2].textContent = values.CRF;
  document.querySelectorAll(".text-wrapper-8")[3].textContent = values.BW;
  document.querySelectorAll(".text-wrapper-8")[4].textContent = values.BF;
}

updateDisplayedValues();

document.querySelector('.overlap-3').addEventListener('click', async () => {
  try {
    const mach = parseFloat(document.getElementById('machInput').value);
    const aoa = parseFloat(document.getElementById('aoaInput').value);
    const ln = parseFloat(document.getElementById('lnInput').value);
    const swept = parseFloat(document.getElementById('sweptInput').value);
    const lln = parseFloat(document.getElementById('lengthInput').value);
    const mode = document.getElementById('dropdownBtn').textContent.split(": ")[1];

    if (isNaN(mach) || isNaN(aoa) || isNaN(ln)|| isNaN(swept)|| isNaN(lln)) {
      alert("Vui lòng nhập đủ cả 5 thông số!");
      return;
    }

    const response = await fetch("https://test-final1-sd9x.onrender.com/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({mode, mach, aoa, ln, swept, lln})
    });
    
    const contentType = response.headers.get("content-type") || "";
    const rawText = await response.text();
    
    if (!response.ok) {
      console.error("Lỗi server:", rawText);
      alert("Server lỗi: " + rawText);
      return;
    }
    
    let data;
    try {
      if (!contentType.includes("application/json")) {
        throw new Error("Server không trả về JSON.");
      }
      data = JSON.parse(rawText);
    } catch (e) {
      console.error("Lỗi parse JSON:", e.message, rawText);
      alert("Không thể đọc phản hồi từ server: " + e.message);
      return;
    }

    document.querySelector(".text-wrapper-18").textContent = data.cl;
    document.querySelector(".text-wrapper-16").textContent = data.cd;

    if (data.warning) {
      alert(data.warning);
    }
  } catch (err) {
    console.error("Lỗi:", err);
    alert("Không kết nối được với server: " + err.message);
  }
});

function updateFieldLocking(mode) {
  const lnInput = document.getElementById("lnInput");
  const sweptInput = document.getElementById("sweptInput");

  const lnSlider = document.getElementById("lnSlider");
  const sweptSlider = document.getElementById("sweptSlider");

  const isNASA = mode === "NASA";

  lnInput.readOnly = !isNASA;
  sweptInput.readOnly = !isNASA;

  lnSlider.disabled = !isNASA;
  sweptSlider.disabled = !isNASA;

  if (!isNASA) {
    machInput.value = 0.00;
    aoaInput.value = 0.00;
    lnInput.value = 19.32;
    sweptInput.value = 16.54;
    updateBodyLength();
  }

  if (mode == "Missile Shape 1") {
    sweptInput.value = 4.11;
    sweptSlider.value = 4.11;
  }

  if (mode == "Missile Shape 2") {
    sweptInput.value = 19.73;
    sweptSlider.value = 19.73;
  }

  if (mode == "Missile Shape 3") {
    sweptInput.value = 10.45;
    sweptSlider.value = 10.45;
  }
}

function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}
animate();

function linkSliderAndInput(sliderId, inputId) {
  const slider = document.getElementById(sliderId);
  const input = document.getElementById(inputId);

  slider.addEventListener('input', () => {
    input.value = slider.value;
    updateBodyLength();
  });

  input.addEventListener('input', () => {
    slider.value = input.value;
    updateBodyLength();
  });
}

linkSliderAndInput('lnSlider', 'lnInput');
linkSliderAndInput('sweptSlider', 'sweptInput');
linkSliderAndInput('aoaSlider', 'aoaInput');
linkSliderAndInput('machSlider', 'machInput');
