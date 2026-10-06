// LeKiwi in the browser: physics by MuJoCo's official WebAssembly build, picture by three.js.
// model/ holds the training simulation's model (same physics). The 3 wheel motors are locked; only the arm moves.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { ArmKinematics, pitchOf } from './kinematics.js';

const MUJOCO_VERSION = '3.14.0';  // keep equal to the Python MuJoCo that exported the model (manifest.json)
const MUJOCO_URL = `https://cdn.jsdelivr.net/npm/@mujoco/mujoco@${MUJOCO_VERSION}/mujoco.js`;
const MODEL_URL = 'model/';
const ROOT = '/lekiwi';            // folder of the model files inside MuJoCo's in-memory file system
const D2R = Math.PI / 180;
const DRAG = { k: 500, c: 8, fmax: 40 };  // mouse spring: N/m, N*s/m, N
const PIN = { k: 2e4, c: 300, kr: 1000, cr: 8 };  // chassis holder: N/m, N*s/m, N*m/rad, N*m*s/rad
const $ = (id) => document.getElementById(id);
const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
const f1 = (v) => (Math.abs(v) < 0.05 ? 0 : v).toFixed(1);  // one decimal, never "-0.0"
const minus = (s) => String(s).replace(/-/g, '−');          // typographic minus for display

// ------------------------------------------------------------------ loading screen / errors
function loading(text, frac) {  // index.html's bar creeps toward the next step; the text is for screen readers
  $('loading').setAttribute('aria-label', text);
  if (frac !== undefined) window.lekiwiLoad = frac;
}
function fail(msg) {
  $('loading').classList.remove('done');
  $('loading').classList.add('error');
  $('loadText').textContent = msg;
  console.error(msg);
}

// ------------------------------------------------------------------ model files
async function fetchJSON(name) {
  const r = await fetch(MODEL_URL + name, { cache: 'no-cache' });
  if (!r.ok) throw new Error(`${name}: HTTP ${r.status}`);
  return r.json();
}
async function fetchBytes(url, onProgress) {
  const r = await fetch(url, { cache: 'no-cache' });
  if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
  const total = Number(r.headers.get('content-length')) || 0;
  if (!r.body || !total || !onProgress) return new Uint8Array(await r.arrayBuffer());
  const reader = r.body.getReader(), parts = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    parts.push(value); got += value.length; onProgress(got / total);
  }
  const out = new Uint8Array(got);
  let o = 0;
  for (const p of parts) { out.set(p, o); o += p.length; }
  return out;
}
async function gunzip(bytes) {  // the server may already have decoded it (Content-Encoding): check the gzip magic
  if (bytes[0] !== 0x1f || bytes[1] !== 0x8b) return bytes;
  const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}
function unpack(bytes) {  // model.pack.gz: [uint32 header length][JSON header][file bytes ...]
  const hlen = new DataView(bytes.buffer, bytes.byteOffset, 4).getUint32(0, true);
  const header = JSON.parse(new TextDecoder().decode(bytes.subarray(4, 4 + hlen)));
  const files = [];
  let off = 4 + hlen;
  for (const f of header.files) { files.push([f.path, bytes.subarray(off, off + f.size)]); off += f.size; }
  return files;
}
async function loadModelFiles(manifest) {  // every model file in one download: model.pack.gz
  const mb = manifest.pack_bytes / 1e6;
  const packed = await fetchBytes(MODEL_URL + manifest.pack, (f) => loading(`Downloading the model ${(f * mb).toFixed(1)} / ${mb.toFixed(1)} MB`, 0.15 + 0.45 * f));
  return unpack(await gunzip(packed));
}
function writeTree(mj, files) {
  const FS = mj.FS;
  const mkdirp = (dir) => {
    let cur = '';
    for (const part of dir.split('/').filter(Boolean)) {
      cur += '/' + part;
      if (!FS.analyzePath(cur).exists) FS.mkdir(cur);
    }
  };
  for (const [path, bytes] of files) {
    const full = `${ROOT}/${path}`;
    mkdirp(full.slice(0, full.lastIndexOf('/')));
    FS.writeFile(full, bytes);
  }
}

// ------------------------------------------------------------------ state
let mj, model, data, C, AF, MANIFEST, KIN;
const S = {
  running: true, teach: true, showFrames: true, showLabels: true, showTCP: false, ghost: false, framesOnly: false,
  autoView: true, simBudget: 0, lastWall: 0, demo: false, demoT0: 0, demoFrom: null,
};
let goal = new Array(6).fill(0);       // arm servo goals (rad, sim joint angles)
let motors = new Array(6).fill(0);     // the same in LeRobot units (slider values)
const armV = new Array(6).fill(0);     // servo profile state: current speed per arm motor
let drag = null;
let chassis = -1, pin0 = null;  // chassis body id, its pose after reset

// LeRobot units <-> sim joint angles (kinematics.js, the training environment's rule); targets kept in range
const motorsToCtrl = (m) => KIN.toQ(m).map((v, i) => clamp(v, C.ctrlrange[i][0], C.ctrlrange[i][1]));
const qToMotors = (q) => KIN.toLeRobot(q);
function armQ() { const qpos = data.qpos; return C.arm_qadr.map((a) => qpos[a]); }

// ------------------------------------------------------------------ physics
function resetSim(m6 = C.rest_motors) {  // as in training: default pose, arm at m6, settle 0.3 s
  setDemo(false);
  mj.mj_resetData(model, data);
  goal = motorsToCtrl(m6);
  motors = qToMotors(goal);
  const qpos = data.qpos, ctrl = data.ctrl;
  for (let i = 0; i < 6; i++) { qpos[C.arm_qadr[i]] = goal[i]; ctrl[C.arm_act[i]] = goal[i]; armV[i] = 0; }
  for (const a of C.wheel_act) ctrl[a] = 0;
  mj.mj_forward(model, data);
  const n = Math.round(C.settle_s / model.opt.timestep);
  for (let k = 0; k < n; k++) mj.mj_step(model, data);
  data.time = 0;
  const qp = data.qpos;
  pin0 = Array.from(qp.slice(0, 7));
  endDrag();
  syncSliders(true);
}
function setGoal(m6) { goal = motorsToCtrl(m6); motors = qToMotors(goal); if (rows.length) syncSliders(true); }

// demo: a smooth periodic motion (LeRobot units), blended in from the current pose over 3 s, on simulation time
// (pausing the sim pauses it). wrist_roll stays within -60..80 deg, above its -67.2 deg limit.
// Checked collision-free from the rest pose.
const DEMO_DELAY = [1.5, 0.4, 0, 0, 1.5, 0];  // elbow / wrist unfold first, then lift, pan and wrist_roll turn last
function demoMotors(t, from) {
  const d = [55 * Math.sin(0.6 * t), -15 + 25 * Math.sin(0.45 * t + 1), 12.5 + 27.5 * Math.sin(0.5 * t),
             -7.5 + 32.5 * Math.sin(0.7 * t + 0.5), 10 + 70 * Math.sin(0.4 * t), 40 + 35 * Math.sin(1.1 * t)];
  const b = tt => 0.5 - 0.5 * Math.cos(Math.PI * Math.min(Math.max(tt, 0) / 3, 1));
  return d.map((v, i) => from[i] + b(t - DEMO_DELAY[i]) * (v - from[i]));
}
function setDemo(on) {
  S.demo = on;
  if (on) { S.demoT0 = data.time; S.demoFrom = motors.slice(); }
  $('bDemo').setAttribute('aria-pressed', String(on));
  $('bDemo').classList.toggle('on', on);
}

// one control step, as in training: wheels held at 0, the arm ctrl moves toward the goal with the servo speed and
// acceleration profile fitted to the real servos (control.json servo_profile)
function servoStep(h) {
  const ctrl = data.ctrl;
  for (const a of C.wheel_act) ctrl[a] = 0;
  if (drag && S.teach) {
    // teach drag: each servo only cancels gravity (target = current angle + gravity torque / kp), so the arm
    // floats and follows the mouse; where it is let go it stays (the goal is set to that pose)
    const qpos = data.qpos, bias = data.qfrc_bias, gain = model.actuator_gainprm;
    for (let i = 0; i < 6; i++) {
      const a = C.arm_act[i], q = qpos[C.arm_qadr[i]], kp = gain[10 * a];
      ctrl[a] = clamp(q + bias[C.arm_dofadr[i]] / kp, C.ctrlrange[i][0], C.ctrlrange[i][1]);
      goal[i] = q; armV[i] = 0;
    }
    return;
  }
  if (S.demo) goal = motorsToCtrl(demoMotors(data.time - S.demoT0, S.demoFrom));
  const VM = C.servo_profile.max_speed_radps, AC = C.servo_profile.acceleration_radps2;
  for (let i = 0; i < 6; i++) {
    const a = C.arm_act[i], err = goal[i] - ctrl[a];
    const vmax = VM[i] === null ? Infinity : VM[i];
    if (AC === null) {
      armV[i] = clamp(err / h, -vmax, vmax);
    } else {
      const acc = AC[i], vdes = Math.sign(err) * Math.min(vmax, Math.sqrt(2 * acc * Math.abs(err)));
      armV[i] += clamp(vdes - armV[i], -acc * h, acc * h);
    }
    const stepq = armV[i] * h, done = Math.abs(stepq) >= Math.abs(err);
    ctrl[a] = done ? goal[i] : ctrl[a] + stepq;
    if (done && Math.abs(err) < 1e-9) armV[i] = 0;
  }
}

// mouse spring on the grabbed point (force + torque about the body's centre of mass, world frame)
function dragForce(h) {
  if (!drag) return;
  const b = drag.body, xpos = data.xpos, xmat = data.xmat, xipos = data.xipos, f = data.xfrc_applied;
  const o = 3 * b, r = 9 * b, [lx, ly, lz] = drag.local;
  const p = [xpos[o] + xmat[r] * lx + xmat[r + 1] * ly + xmat[r + 2] * lz,
             xpos[o + 1] + xmat[r + 3] * lx + xmat[r + 4] * ly + xmat[r + 5] * lz,
             xpos[o + 2] + xmat[r + 6] * lx + xmat[r + 7] * ly + xmat[r + 8] * lz];
  const v = drag.prev ? p.map((x, i) => (x - drag.prev[i]) / h) : [0, 0, 0];
  drag.prev = p; drag.point = p;
  let F = [0, 1, 2].map((i) => DRAG.k * (drag.target[i] - p[i]) - DRAG.c * v[i]);
  const n = Math.hypot(...F);
  if (n > DRAG.fmax) F = F.map((x) => x * DRAG.fmax / n);
  const d = [p[0] - xipos[o], p[1] - xipos[o + 1], p[2] - xipos[o + 2]];
  const T = [d[1] * F[2] - d[2] * F[1], d[2] * F[0] - d[0] * F[2], d[0] * F[1] - d[1] * F[0]];
  for (let i = 0; i < 3; i++) { f[6 * b + i] = F[i]; f[6 * b + 3 + i] = T[i]; }
}

// The chassis is held: a stiff spring-damper pulls it back to its pose after reset (an external force, not part
// of the model), so pulling the arm does not slide the robot on its rollers or tip it over.
function holdChassis() {
  const f = data.xfrc_applied, o = 6 * chassis;
  if (!pin0) { for (let i = 0; i < 6; i++) f[o + i] = 0; return; }
  const q = data.qpos, v = data.qvel, R = data.xmat, r = 9 * chassis;
  for (let i = 0; i < 3; i++) f[o + i] = -PIN.k * (q[i] - pin0[i]) - PIN.c * v[i];
  // orientation error: rotation vector of q * conj(q0)
  const [w, x, y, z] = [q[3], q[4], q[5], q[6]], [w0, x0, y0, z0] = [pin0[3], -pin0[4], -pin0[5], -pin0[6]];
  let ew = w * w0 - x * x0 - y * y0 - z * z0;
  let e = [w * x0 + x * w0 + y * z0 - z * y0, w * y0 - x * z0 + y * w0 + z * x0, w * z0 + x * y0 - y * x0 + z * w0];
  if (ew < 0) { ew = -ew; e = e.map((c) => -c); }
  const wl = [v[3], v[4], v[5]];  // free-joint angular velocity is in the body frame
  const ww = [0, 1, 2].map((i) => R[r + 3 * i] * wl[0] + R[r + 3 * i + 1] * wl[1] + R[r + 3 * i + 2] * wl[2]);
  for (let i = 0; i < 3; i++) f[o + 3 + i] = -PIN.kr * 2 * e[i] - PIN.cr * ww[i];
}

function stepPhysics(h) {
  servoStep(h);
  dragForce(h);
  holdChassis();
  mj.mj_step(model, data);
}
function physics(wallDt) {
  if (!S.running) { S.simBudget = 0; return; }
  const h = model.opt.timestep;
  S.simBudget += Math.min(wallDt, 0.1);
  let n = 0;
  for (; S.simBudget >= h && n < 120; n++) { stepPhysics(h); S.simBudget -= h; }
  if (n === 120) S.simBudget = 0;  // this computer is slower than real time: do not try to catch up
}

// ------------------------------------------------------------------ three.js scene
let renderer, scene, camera, controls, sun;
const geomViews = [];      // {g, obj, arm}
const armMeshes = [];      // pickable
const frameViews = [];     // {f, body, offset, group, el (tag), lead (its leader line), shown, o, ends}
let skeleton, dragLine;
const tmpM = new THREE.Matrix4(), tmpM2 = new THREE.Matrix4();

function srgb(r, g, b) { return new THREE.Color().setRGB(r, g, b, THREE.SRGBColorSpace); }

function checkerTexture() {  // MuJoCo's "groundplane": 2x2 checker with marked tile edges
  const cv = document.createElement('canvas');
  cv.width = cv.height = 256;
  const g = cv.getContext('2d');
  const c1 = `rgb(${0.2 * 255},${0.3 * 255},${0.4 * 255})`, c2 = `rgb(${0.1 * 255},${0.2 * 255},${0.3 * 255})`;
  g.fillStyle = c1; g.fillRect(0, 0, 256, 256);
  g.fillStyle = c2; g.fillRect(0, 0, 128, 128); g.fillRect(128, 128, 128, 128);
  g.strokeStyle = 'rgba(204,204,204,0.9)'; g.lineWidth = 3; g.strokeRect(0, 0, 256, 256);
  const t = new THREE.CanvasTexture(cv);
  t.colorSpace = THREE.SRGBColorSpace;
  t.wrapS = t.wrapT = THREE.RepeatWrapping;
  t.anisotropy = 8;
  return t;
}

function meshGeometry(id, cache) {
  if (cache.has(id)) return cache.get(id);
  const vadr = model.mesh_vertadr[id], fadr = model.mesh_faceadr[id], nf = model.mesh_facenum[id];
  const V = model.mesh_vert, F = model.mesh_face, pos = new Float32Array(nf * 9);
  for (let f = 0; f < nf; f++) {
    for (let k = 0; k < 3; k++) {
      const v = 3 * (vadr + F[3 * (fadr + f) + k]);
      pos[9 * f + 3 * k] = V[v]; pos[9 * f + 3 * k + 1] = V[v + 1]; pos[9 * f + 3 * k + 2] = V[v + 2];
    }
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  geo.computeVertexNormals();  // non-indexed -> flat facets, like the CAD parts
  geo.computeBoundingSphere();
  cache.set(id, geo);
  return geo;
}

function buildGeoms() {
  const armIds = new Set(Object.values(C.body_ids));
  const cache = new Map();
  const type = model.geom_type, group = model.geom_group, matid = model.geom_matid, size = model.geom_size;
  const grgba = model.geom_rgba, mrgba = model.mat_rgba, bodyid = model.geom_bodyid, dataid = model.geom_dataid;
  for (let g = 0; g < model.ngeom; g++) {
    if (group[g] > 2) continue;  // collision-only groups
    const mi = matid[g], rgba = mi >= 0 ? [mrgba[4 * mi], mrgba[4 * mi + 1], mrgba[4 * mi + 2], mrgba[4 * mi + 3]]
                                        : [grgba[4 * g], grgba[4 * g + 1], grgba[4 * g + 2], grgba[4 * g + 3]];
    if (rgba[3] === 0) continue;  // invisible (wheel contact capsules)
    const s = [size[3 * g], size[3 * g + 1], size[3 * g + 2]];
    let geo, mat;
    switch (type[g]) {
      case 0: {  // plane (the floor): MuJoCo's checker, 5 tiles per metre
        const L = 6;
        geo = new THREE.PlaneGeometry(2 * L, 2 * L);
        const tex = checkerTexture();
        tex.repeat.set(2 * L * 5, 2 * L * 5);
        mat = new THREE.MeshStandardMaterial({ map: tex, roughness: 0.85, metalness: 0.0 });
        break;
      }
      case 2: geo = new THREE.SphereGeometry(s[0], 24, 16); break;
      case 3: geo = new THREE.CapsuleGeometry(s[0], 2 * s[1], 8, 16).rotateX(Math.PI / 2); break;
      case 4: geo = new THREE.SphereGeometry(1, 24, 16).scale(s[0], s[1], s[2]); break;
      case 5: geo = new THREE.CylinderGeometry(s[0], s[0], 2 * s[1], 24).rotateX(Math.PI / 2); break;
      case 6: geo = new THREE.BoxGeometry(2 * s[0], 2 * s[1], 2 * s[2]); break;
      case 7: geo = meshGeometry(dataid[g], cache); break;
      default: continue;
    }
    const arm = armIds.has(bodyid[g]);
    mat = mat || new THREE.MeshStandardMaterial({ color: srgb(rgba[0], rgba[1], rgba[2]), roughness: 0.55, metalness: 0.05,
                                                transparent: rgba[3] < 1, opacity: rgba[3] });
    const obj = new THREE.Mesh(geo, mat);
    obj.matrixAutoUpdate = false;
    obj.castShadow = type[g] !== 0;
    obj.receiveShadow = true;
    obj.userData = { geom: g, body: bodyid[g], arm, baseOpacity: rgba[3] };
    scene.add(obj);
    geomViews.push({ g, obj, arm, floor: type[g] === 0 });
    if (arm) armMeshes.push(obj);
  }
}

function arrow(color, len, radius) {
  const m = new THREE.MeshBasicMaterial({ color, depthTest: false, transparent: true });
  const shaft = new THREE.Mesh(new THREE.CylinderGeometry(radius, radius, len * 0.78, 10), m);
  shaft.position.y = len * 0.39;
  const head = new THREE.Mesh(new THREE.ConeGeometry(radius * 2.6, len * 0.22, 14), m);
  head.position.y = len * 0.89;
  const g = new THREE.Group();
  g.add(shaft, head);
  g.traverse((o) => { o.renderOrder = 10; });
  return g;
}

// every frame (F0..F6, TCP) gets 21 mm axes, so the closest pairs (F4-F5 42.4 mm, F0-F1 46.1 mm apart) never overlap
const FRAME_LEN = 0.021, FRAME_R = 0.0011;  // m: axis length, shaft radius

function buildFrames() {
  for (const f of AF.frames) {
    const body = C.body_ids[f.body];
    if (body === undefined) continue;
    const off = f.offset_in_body, q = off.quat_wxyz;
    const offset = new THREE.Matrix4().compose(new THREE.Vector3(...off.pos), new THREE.Quaternion(q[1], q[2], q[3], q[0]), new THREE.Vector3(1, 1, 1));
    const group = new THREE.Group();
    group.matrixAutoUpdate = false;
    const L = FRAME_LEN, r = FRAME_R;
    const ax = arrow(0xe5484d, L, r); ax.rotation.z = -Math.PI / 2;  // +y -> +x
    const ay = arrow(0x30a46c, L, r);                                 // +y
    const az = arrow(0x3e63dd, L, r); az.rotation.x = Math.PI / 2;    // +y -> +z
    const dot = new THREE.Mesh(new THREE.SphereGeometry(r * 2.2, 12, 8), new THREE.MeshBasicMaterial({ color: 0xffffff, depthTest: false }));
    dot.renderOrder = 11;
    group.add(ax, ay, az, dot);
    scene.add(group);
    const el = Object.assign(document.createElement('div'), { className: 'tag', textContent: f.key });
    const lead = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    $('tags').appendChild(el);
    $('tags').firstElementChild.appendChild(lead);
    frameViews.push({ f, body, offset, group, el, lead });
  }
  skeleton = new THREE.Line(new THREE.BufferGeometry().setFromPoints(new Array(7).fill(0).map(() => new THREE.Vector3())),
                            new THREE.LineBasicMaterial({ color: 0xc8ccd4, depthTest: false, transparent: true, opacity: 0.85 }));
  skeleton.renderOrder = 9;
  skeleton.frustumCulled = false;
  scene.add(skeleton);
  dragLine = new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]),
                            new THREE.LineBasicMaterial({ color: 0xff5050, depthTest: false }));
  dragLine.renderOrder = 12;
  dragLine.frustumCulled = false;
  dragLine.visible = false;
  scene.add(dragLine);
}

function bodyMatrix(b, xpos, xmat, out) {
  const o = 3 * b, r = 9 * b;
  return out.set(xmat[r], xmat[r + 1], xmat[r + 2], xpos[o],
                 xmat[r + 3], xmat[r + 4], xmat[r + 5], xpos[o + 1],
                 xmat[r + 6], xmat[r + 7], xmat[r + 8], xpos[o + 2], 0, 0, 0, 1);
}

function syncScene() {
  // body poses of the current qpos (mj_step computes them before integrating); no effect on the dynamics
  mj.mj_kinematics(model, data);
  const gx = data.geom_xpos, gm = data.geom_xmat;
  for (const v of geomViews) {
    const g = v.g, o = 3 * g, r = 9 * g;
    v.obj.matrix.set(gm[r], gm[r + 1], gm[r + 2], gx[o], gm[r + 3], gm[r + 4], gm[r + 5], gx[o + 1],
                     gm[r + 6], gm[r + 7], gm[r + 8], gx[o + 2], 0, 0, 0, 1);
    v.obj.matrixWorldNeedsUpdate = true;
    v.obj.visible = !(S.framesOnly && !v.floor);
  }
  const xpos = data.xpos, xmat = data.xmat, pts = [];
  const world = {};
  for (const fv of frameViews) {
    const show = S.showFrames && (fv.f.default || (S.showTCP && fv.f.key === 'TCP'));
    fv.group.visible = fv.shown = show;
    bodyMatrix(fv.body, xpos, xmat, tmpM);
    fv.group.matrix.multiplyMatrices(tmpM, fv.offset);
    fv.group.matrixWorldNeedsUpdate = true;
    const m = fv.group.matrix.elements, o = new THREE.Vector3(m[12], m[13], m[14]);
    world[fv.f.key] = fv.group.matrix;
    fv.o = o;
    fv.ends = [0, 1, 2].map((c) => o.clone().addScaledVector(new THREE.Vector3(m[4 * c], m[4 * c + 1], m[4 * c + 2]), FRAME_LEN));
    if (fv.f.default) pts.push(o);
  }
  skeleton.visible = S.showFrames && S.framesOnly;
  if (skeleton.visible) { skeleton.geometry.setFromPoints(pts); skeleton.geometry.attributes.position.needsUpdate = true; }
  dragLine.visible = !!(drag && drag.point);
  if (dragLine.visible) {
    dragLine.geometry.setFromPoints([new THREE.Vector3(...drag.point), new THREE.Vector3(...drag.target)]);
  }
  return world;
}

// ------------------------------------------------------------------ frame tags
// Short "F0".."F6" tags as in the Frames video: each where it covers no other tag, no frame origin and no axis,
// inside the view, with a thin leader line when it had to move away; they glide instead of jumping.
const TAG = { gap: 4, lead: 8 };
const ANGLES = Array.from({ length: 16 }, (_, i) => (i * Math.PI) / 8);
const RADII = [0, 7, 15, 26];  // extra distance (px) beyond TAG.gap
const area = (a) => Math.max(0, a[2] - a[0]) * Math.max(0, a[3] - a[1]);
const cut = (a, b) => [Math.max(a[0], b[0]), Math.max(a[1], b[1]), Math.min(a[2], b[2]), Math.min(a[3], b[3])];
function insideLength(p, q, b) {  // length of segment p-q inside box b (Liang-Barsky)
  let t0 = 0, t1 = 1;
  const dx = q[0] - p[0], dy = q[1] - p[1];
  for (const [num, den] of [[p[0] - b[0], -dx], [b[2] - p[0], dx], [p[1] - b[1], -dy], [b[3] - p[1], dy]]) {
    if (den === 0) { if (num < 0) return 0; continue; }
    const t = num / den;
    if (den < 0) t0 = Math.max(t0, t); else t1 = Math.min(t1, t);
    if (t0 > t1) return 0;
  }
  return (t1 - t0) * Math.hypot(dx, dy);
}
function nearDot(c, rad, b) {  // circle (centre c) touches box b
  const x = clamp(c[0], b[0], b[2]), y = clamp(c[1], b[1], b[3]);
  return Math.hypot(c[0] - x, c[1] - y) < rad;
}
function placeTags(items, W, H, prev) {  // items in drawing order; prev: key -> last target offset
  const placed = [], view = [3, 3, W - 3, H - 3], segs = [];
  for (const it of items) for (const t of it.axes) segs.push([it.p, t, 26]);  // axes: avoid strongly
  const chain = items.filter((it) => it.chain);
  for (let k = 1; k < chain.length; k++) segs.push([chain[k - 1].p, chain[k].p, 9]);  // the F0..F6 line: mildly
  for (const it of items) {
    let best = null;
    for (const ang of ANGLES) {
      const dx = Math.cos(ang), dy = Math.sin(ang), ext = Math.abs(dx) * it.w / 2 + Math.abs(dy) * it.h / 2;
      for (const extra of RADII) {
        const r = TAG.gap + extra, cx = it.p[0] + dx * (r + ext), cy = it.p[1] + dy * (r + ext);
        const box = [cx - it.w / 2, cy - it.h / 2, cx + it.w / 2, cy + it.h / 2];
        let cost = 2.2 * extra + 9 * (1 - Math.cos(ang + Math.PI / 6));  // close, preferably up-right
        cost += 25 * (area(box) - area(cut(box, view)));                  // inside the view
        for (const q of placed) cost += 12 * area(cut(box, q.box));      // not on another tag
        for (const o of items) if (nearDot(o.p, 4.5, box)) cost += 600;  // not on an origin
        for (const [a, b, wgt] of segs) cost += wgt * insideLength(a, b, box);
        const off = [cx - it.p[0], cy - it.p[1]];
        if (prev[it.key]) cost += 1.4 * Math.hypot(off[0] - prev[it.key][0], off[1] - prev[it.key][1]);
        if (!best || cost < best.cost) best = { cost, off };
      }
    }
    const box = [it.p[0] + best.off[0] - it.w / 2, it.p[1] + best.off[1] - it.h / 2,
                 it.p[0] + best.off[0] + it.w / 2, it.p[1] + best.off[1] + it.h / 2];
    placed.push({ ...it, off: best.off, box });
  }
  return placed;
}
const tagTarget = {}, tagDrawn = {}, tmpV = new THREE.Vector3();
function toPx(v, W, H) {
  tmpV.copy(v).project(camera);
  return tmpV.z > 1 ? null : [(tmpV.x + 1) / 2 * W, (1 - tmpV.y) / 2 * H];
}
function layoutTags(dt) {
  const layer = $('tags'), on = S.showFrames && S.showLabels;
  layer.style.display = on ? '' : 'none';
  if (!on) return;
  const W = layer.clientWidth, H = layer.clientHeight, items = [];
  for (const fv of frameViews) {
    const p = fv.shown && fv.o ? toPx(fv.o, W, H) : null;
    if (!p) { fv.el.style.display = fv.lead.style.display = 'none'; continue; }
    fv.el.style.display = '';
    if (!fv.w) { fv.w = fv.el.offsetWidth; fv.h = fv.el.offsetHeight; }
    items.push({ key: fv.f.key, fv, chain: fv.f.default, w: fv.w, h: fv.h, p, axes: fv.ends.map((e) => toPx(e, W, H) || p) });
  }
  const k = 1 - Math.exp(-Math.min(dt, 0.1) / 0.03);  // glide toward the new place
  for (const t of placeTags(items, W, H, tagTarget)) {
    tagTarget[t.key] = t.off;
    const d = tagDrawn[t.key], off = d ? [d[0] + (t.off[0] - d[0]) * k, d[1] + (t.off[1] - d[1]) * k] : t.off.slice();
    tagDrawn[t.key] = off;
    const x0 = t.p[0] + off[0] - t.w / 2, y0 = t.p[1] + off[1] - t.h / 2;
    t.fv.el.style.transform = `translate(${x0.toFixed(1)}px, ${y0.toFixed(1)}px)`;
    const nx = clamp(t.p[0], x0, x0 + t.w), ny = clamp(t.p[1], y0, y0 + t.h), dist = Math.hypot(nx - t.p[0], ny - t.p[1]);
    const ln = t.fv.lead;
    ln.style.display = dist > TAG.lead ? '' : 'none';
    if (dist <= TAG.lead) continue;
    const s = 3 / dist;  // from just off the origin dot to the tag
    ln.setAttribute('x1', (t.p[0] + (nx - t.p[0]) * s).toFixed(1)); ln.setAttribute('y1', (t.p[1] + (ny - t.p[1]) * s).toFixed(1));
    ln.setAttribute('x2', nx.toFixed(1)); ln.setAttribute('y2', ny.toFixed(1));
  }
}

function setGhost(on) {
  for (const v of geomViews) {
    if (!v.arm) continue;
    const m = v.obj.material, base = v.obj.userData.baseOpacity;
    m.transparent = on || base < 1;
    m.opacity = on ? 0.28 : base;
    m.depthWrite = !on;
    m.needsUpdate = true;
  }
}

// ------------------------------------------------------------------ default view
// From the robot's front-left, framed so that the chassis and every demo pose stay in the picture; refitted on
// resize until the view is moved by hand (a double-click on the background returns to it).
const VIEW_DIR = new THREE.Vector3(0.5, 0.5, 0.27).normalize();  // from the look-at point toward the camera
const VIEW_FILL = 0.9;  // use 90 % of the picture: a margin for the frame labels
let viewPts = null;     // what the default view must show (world, m): chassis + arm over a whole demo period

function supportPoints(g) {  // a few points (geom frame) that bound the geom's outline seen from any side
  const pts = [];
  if (model.geom_type[g] === 7) {  // mesh: its outermost vertex along each of 26 directions
    const id = model.geom_dataid[g], va = model.mesh_vertadr[id], nv = model.mesh_vertnum[id], V = model.mesh_vert;
    const pick = new Set();
    for (let i = -1; i <= 1; i++) for (let j = -1; j <= 1; j++) for (let k = -1; k <= 1; k++) {
      if (!i && !j && !k) continue;
      let best = 0, bs = -Infinity;
      for (let n = 0; n < nv; n++) {
        const o = 3 * (va + n), s = i * V[o] + j * V[o + 1] + k * V[o + 2];
        if (s > bs) { bs = s; best = n; }
      }
      pick.add(best);
    }
    for (const n of pick) { const o = 3 * (va + n); pts.push(V[o], V[o + 1], V[o + 2]); }
  } else {  // primitive: the corners of its bounding box
    const A = model.geom_aabb, o = 6 * g;
    for (let k = 0; k < 8; k++) {
      pts.push(A[o] + (k & 1 ? A[o + 3] : -A[o + 3]), A[o + 1] + (k & 2 ? A[o + 4] : -A[o + 4]),
               A[o + 2] + (k & 4 ? A[o + 5] : -A[o + 5]));
    }
  }
  return pts;
}

function buildViewEnvelope() {  // kinematics only (a separate MjData): the simulation is not touched
  const parts = geomViews.filter((v) => !v.floor).map((v) => ({ g: v.g, arm: v.arm, pts: supportPoints(v.g) }));
  const arm = parts.filter((p) => p.arm);
  const d2 = new mj.MjData(model);
  d2.qpos.set(data.qpos);  // the chassis where it stands after the reset
  const seen = new Set(), out = [];
  const put = (list) => {
    const X = d2.geom_xpos, R = d2.geom_xmat;
    for (const p of list) {
      const o = 3 * p.g, r = 9 * p.g, P = p.pts;
      for (let i = 0; i < P.length; i += 3) {
        const x = X[o] + R[r] * P[i] + R[r + 1] * P[i + 1] + R[r + 2] * P[i + 2];
        const y = X[o + 1] + R[r + 3] * P[i] + R[r + 4] * P[i + 1] + R[r + 5] * P[i + 2];
        const z = X[o + 2] + R[r + 6] * P[i] + R[r + 7] * P[i + 1] + R[r + 8] * P[i + 2];
        const key = Math.round(x * 200) + 2048 + 4096 * (Math.round(y * 200) + 2048 + 4096 * (Math.round(z * 200) + 2048));
        if (!seen.has(key)) { seen.add(key); out.push(x, y, z); }  // one point per 5 mm cell
      }
    }
  };
  const pose = (m6) => {
    const q = motorsToCtrl(m6);
    for (let i = 0; i < 6; i++) d2.qpos[C.arm_qadr[i]] = q[i];
    mj.mj_kinematics(model, d2);
  };
  pose(C.rest_motors); put(parts);
  pose(C.zero_motors); put(arm);
  // the demo repeats every 2 pi / 0.05 s = 125.7 s (all its frequencies are multiples of 0.05 rad/s); from rest
  const T = 3 + Math.max(...DEMO_DELAY) + 2 * Math.PI / 0.05;
  for (let t = 0; t <= T; t += 0.05) { pose(demoMotors(t, C.rest_motors)); put(arm); }
  d2.delete();
  viewPts = new Float32Array(out);
}

// Camera on VIEW_DIR, as close as it can be with every point of viewPts in the picture. In the camera basis
// (right r, up u, forward f; coordinates x, y, z about the points' mean) a point is visible when
// |x - cx| <= (D + z) tx and |y - cy| <= (D + z) ty, D = camera distance, (cx, cy) = sideways shift of the camera:
// that gives the smallest D and the (cx, cy) that centres the points directly.
function fitView() {
  if (!viewPts) return;
  const f = VIEW_DIR.clone().negate(), r = new THREE.Vector3().crossVectors(f, camera.up).normalize();
  const u = new THREE.Vector3().crossVectors(r, f);
  const ty = Math.tan(camera.fov * D2R / 2) * VIEW_FILL, tx = ty * camera.aspect;
  const P = viewPts, c = new THREE.Vector3();
  for (let i = 0; i < P.length; i += 3) { c.x += P[i]; c.y += P[i + 1]; c.z += P[i + 2]; }
  c.divideScalar(P.length / 3);
  let ax = -Infinity, bx = Infinity, ay = -Infinity, by = Infinity;
  for (let i = 0; i < P.length; i += 3) {
    const dx = P[i] - c.x, dy = P[i + 1] - c.y, dz = P[i + 2] - c.z;
    const x = dx * r.x + dy * r.y + dz * r.z, y = dx * u.x + dy * u.y + dz * u.z, z = dx * f.x + dy * f.y + dz * f.z;
    ax = Math.max(ax, x - z * tx); bx = Math.min(bx, x + z * tx);
    ay = Math.max(ay, y - z * ty); by = Math.min(by, y + z * ty);
  }
  const D = Math.max((ax - bx) / (2 * tx), (ay - by) / (2 * ty));
  controls.target.copy(c).addScaledVector(r, (ax + bx) / 2).addScaledVector(u, (ay + by) / 2);
  camera.position.copy(controls.target).addScaledVector(VIEW_DIR, D);
  controls.update();
}

// ------------------------------------------------------------------ mouse: drag arm parts, orbit otherwise
const raycaster = new THREE.Raycaster();
function ndc(e) {
  const r = renderer.domElement.getBoundingClientRect();
  return new THREE.Vector2(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
}
function onPointerDown(e) {
  if (e.button !== 0 || !data || S.framesOnly || !S.running) return;
  raycaster.setFromCamera(ndc(e), camera);
  const hit = raycaster.intersectObjects(armMeshes, false)[0];
  if (!hit) return;  // not on the arm: OrbitControls rotates the view
  const b = hit.object.userData.body, xpos = data.xpos, xmat = data.xmat, o = 3 * b, r = 9 * b;
  const d = [hit.point.x - xpos[o], hit.point.y - xpos[o + 1], hit.point.z - xpos[o + 2]];
  const local = [0, 1, 2].map((j) => xmat[r + j] * d[0] + xmat[r + 3 + j] * d[1] + xmat[r + 6 + j] * d[2]);  // R^T d
  const n = new THREE.Vector3();
  camera.getWorldDirection(n);
  setDemo(false);
  drag = { body: b, local, target: [hit.point.x, hit.point.y, hit.point.z], pointerId: e.pointerId,
           plane: new THREE.Plane().setFromNormalAndCoplanarPoint(n, hit.point), prev: null, point: null };
  controls.enabled = false;
  try { renderer.domElement.setPointerCapture(e.pointerId); } catch (err) { /* synthetic events */ }
  e.stopPropagation();
  e.preventDefault();
}
function onPointerMove(e) {
  if (!drag || e.pointerId !== drag.pointerId) return;
  raycaster.setFromCamera(ndc(e), camera);
  const p = new THREE.Vector3();
  if (raycaster.ray.intersectPlane(drag.plane, p)) drag.target = [p.x, p.y, p.z];
}
function endDrag(e) {
  if (!drag || (e && e.pointerId !== drag.pointerId)) return;
  const f = data.xfrc_applied;
  for (let i = 0; i < 6; i++) f[6 * drag.body + i] = 0;
  if (S.teach) { setGoal(qToMotors(armQ())); armV.fill(0); }  // the arm stays where it is
  drag = null;
  controls.enabled = true;
}

// ------------------------------------------------------------------ panel
const JNAMES = ['Pan', 'Lift', 'Elbow', 'Wrist flex', 'Wrist roll', 'Gripper'];
const unit = (i) => (i === 5 ? '%' : '°');
const rows = [];
const frac = (r, v) => clamp((v - r.lo) / (r.hi - r.lo), 0, 1);
const along = (f) => `calc(7px + ${f.toFixed(4)} * (100% - 14px))`;  // where the slider handle's centre is at f
function buildPanel() {
  const box = $('joints');
  const LO = qToMotors(C.ctrlrange.map((r) => r[0])), HI = qToMotors(C.ctrlrange.map((r) => r[1]));
  for (let i = 0; i < 6; i++) {
    const lo = LO[i], hi = HI[i];
    const row = document.createElement('div');
    row.className = 'joint';
    row.innerHTML = `<div class="top"><span class="name" title="${C.motors[i]}">${JNAMES[i]}<span class="ftag">F${i + 1}</span></span>` +
      `<input class="val" type="text" inputmode="decimal" spellcheck="false" autocomplete="off" title="Type a target, Enter" aria-label="${JNAMES[i]} target"></div>` +
      `<input type="range" min="${lo.toFixed(1)}" max="${hi.toFixed(1)}" step="0.1" aria-label="${JNAMES[i]} target"><output class="bubble"></output>`;
    const inp = row.querySelector('input[type=range]'), val = row.querySelector('.val');
    const target = (v) => { setDemo(false); const m = motors.slice(); m[i] = v; goal = motorsToCtrl(m); motors = qToMotors(goal); syncSliders(v !== +inp.value); };
    inp.addEventListener('input', () => target(+inp.value));
    // the number shows the actual angle; click it to type a target (Enter sets it, Esc cancels)
    val.addEventListener('focus', () => { val.value = val.dataset.start = f1(motors[i]); val.select(); });
    val.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') val.blur();
      else if (e.key === 'Escape') { val.value = val.dataset.start; val.blur(); }
    });
    val.addEventListener('blur', () => {
      const v = parseFloat(val.value.replace('−', '-'));
      if (val.value !== val.dataset.start && Number.isFinite(v)) target(v);
      val.value = minus(f1(qToMotors(armQ())[i])) + unit(i);
    });
    const active = (on) => row.classList.toggle('active', on);  // the target value shows above the handle while set
    inp.addEventListener('pointerdown', () => active(true));
    inp.addEventListener('keydown', () => active(true));
    for (const ev of ['pointerup', 'pointercancel', 'blur']) inp.addEventListener(ev, () => active(false));
    box.appendChild(row);
    rows.push({ inp, lo, hi, val, bubble: row.querySelector('.bubble') });
  }
  $('bRest').onclick = () => { setDemo(false); setGoal(C.rest_motors); };
  $('bZero').onclick = () => { setDemo(false); setGoal(C.zero_motors); };
  $('bDemo').onclick = () => setDemo(!S.demo);
  $('bReset').onclick = () => resetSim();
  $('bPause').onclick = togglePause;
  const bind = (id, key, fn) => { const el = $(id); el.checked = S[key]; el.addEventListener('change', () => { S[key] = el.checked; fn && fn(); }); };
  bind('oFrames', 'showFrames'); bind('oLabels', 'showLabels'); bind('oTCP', 'showTCP');
  bind('oGhost', 'ghost', () => setGhost(S.ghost));
  bind('oFramesOnly', 'framesOnly');
  for (const r of document.querySelectorAll('input[name=dragmode]')) r.addEventListener('change', () => { S.teach = $('mTeach').checked; });
  if (document.fonts) document.fonts.ready.then(() => { for (const fv of frameViews) fv.w = 0; });  // re-measure the tags
}
function syncSliders(all) {  // handles = targets
  for (let i = 0; i < 6; i++) {
    const r = rows[i];
    if (all || document.activeElement !== r.inp) r.inp.value = motors[i];
    r.bubble.textContent = minus(f1(motors[i])) + unit(i);
    r.bubble.style.left = along(frac(r, motors[i]));
  }
}
function updateBars() {  // every frame: the fill runs from 0 to the actual angle; in the demo and while the arm is
  // dragged in Teach mode the handles move along with the arm
  if (S.demo || (drag && S.teach)) { motors = qToMotors(S.demo ? goal : armQ()); syncSliders(true); }
  const m = qToMotors(armQ());
  for (let i = 0; i < 6; i++) {
    const r = rows[i], a = frac(r, m[i]), z = frac(r, 0);
    r.inp.style.setProperty('--a', along(Math.min(a, z)));
    r.inp.style.setProperty('--b', along(Math.max(a, z)));
  }
}
function togglePause() {
  S.running = !S.running;
  const b = $('bPause');
  b.classList.toggle('paused', !S.running);
  b.title = S.running ? 'Pause (Space)' : 'Resume (Space)';
  b.setAttribute('aria-label', S.running ? 'Pause' : 'Resume');
  if (!S.running) endDrag();
}
let lastPanel = 0;
function updatePanel(now, world) {
  if (now - lastPanel < 100) return;
  lastPanel = now;
  const m = qToMotors(armQ());
  for (let i = 0; i < 6; i++) if (document.activeElement !== rows[i].val) rows[i].val.value = minus(f1(m[i])) + unit(i);
  $('status').classList.toggle('live', S.running);
  $('statustext').textContent = !S.running ? 'Paused' : S.demo ? 'Demo' : 'Running';
  if (world.F0 && world.TCP) {
    tmpM2.copy(world.F0).invert().multiply(world.TCP);
    const e = tmpM2.elements, cells = $('tcp').querySelectorAll('span');
    for (let c = 0; c < 3; c++) cells[c].textContent = minus(f1(e[12 + c] * 1000));
  }
}

// ------------------------------------------------------------------ kinematics panel (FK / IK)
const M = (s) => `<span class="m">${s}</span>`;  // math notation: italic symbols, subscripts
const sym = (base, sub) => M(`${base}<sub>${sub}</sub>`);
const fx = (v, d) => minus((Math.abs(v) < 0.5 * 10 ** -d ? 0 : v).toFixed(d));
const deg = (v) => (Math.abs(v - Math.round(v)) < 0.005 ? fx(v, 0) : fx(v, 2)) + '°';
function buildKinematicsPanel() {
  const I = '<var>i</var>';
  const F = (k) => `F<sub>${k.slice(1)}</sub>`;  // F3 -> F with subscript 3
  const rowsHtml = AF.frames.filter((f) => f.parent && f.default).map((f) => {
    const p = f.from_parent;
    return `<tr><td>${M(`${F(f.parent)} → ${F(f.key)}`)}</td><td>${f.short.replace('_', ' ')}</td><td>${fx(p.dist_mm, 1)}</td><td>${p.xyz_mm.map((v) => fx(v, 1)).join(', ')}</td></tr>`;
  });
  $('chainTbl').innerHTML = `<tr><th>${M(`F<sub>${I}−1</sub> → F<sub>${I}</sub>`)}</th><th>${sym('<var>q</var>', I)}</th>` +
    `<th>${M(`‖<var>p</var><sub>${I}</sub>‖`)}</th><th>${sym('<var>p</var>', I)}</th></tr>` + rowsHtml.join('');
  $('chainNote').innerHTML = `${sym('<var>p</var>', I)} = origin of ${M(`F<sub>${I}</sub>`)} in ${M(`F<sub>${I}−1</sub>`)} <span class="dim">· lengths in mm</span>`;
  if (AF.mdh) {
    $('dhTbl').innerHTML = `<tr><th>${M(I)}</th><th>${sym('<var>α</var>', `${I}−1`)}</th><th>${sym('<var>a</var>', `${I}−1`)}</th>` +
      `<th>${sym('Δ<var>θ</var>', I)}</th><th>${sym('<var>d</var>', I)}</th></tr>` +
      AF.mdh.rows.map((r, i) => `<tr><td>${i + 1} <span class="dim">${JNAMES[i].toLowerCase()}</span></td><td>${deg(r.alpha_prev_deg)}</td>` +
        `<td>${fx(r.a_prev_mm, 2)}</td><td>${fx(r.theta_offset_deg, 2)}°</td><td>${fx(r.d_mm, 2)}</td></tr>`).join('');
    $('dhTool').innerHTML = `<div>${M(`<var>θ</var><sub>${I}</sub> = <var>q</var><sub>${I}</sub> + Δ<var>θ</var><sub>${I}</sub>`)} <span class="dim">· lengths in mm</span></div>` +
      `<div class="tool"><span>${M('M<sub>5</sub>')} → TCP</span><span>${M('<var>p</var>')} = (${AF.mdh.tool_xyz_mm.map((v) => fx(v, 2)).join(', ')}) mm</span>` +
      `<span></span><span>rpy = (${AF.mdh.tool_rpy_deg.map(deg).join(', ')})</span></div>`;
  }
  $('bIkFill').onclick = () => {
    const T = KIN.fk(armQ()).TCP;
    $('ikX').value = f1(T[3] * 1000); $('ikY').value = f1(T[7] * 1000); $('ikZ').value = f1(T[11] * 1000);
    $('ikP').value = f1(pitchOf(T) / D2R); $('ikR').value = f1(qToMotors(armQ())[4]);
  };
  $('bIkGo').onclick = () => {
    const v = ['ikX', 'ikY', 'ikZ', 'ikP', 'ikR'].map((id) => parseFloat($(id).value));
    if (v.some((x) => !Number.isFinite(x))) { $('ikOut').textContent = 'Enter x, y, z, pitch and roll.'; return; }
    const seed = armQ();
    const sol = KIN.ik({ pos: [v[0] / 1000, v[1] / 1000, v[2] / 1000], pitch: v[3] * D2R, roll: KIN.toQ([0, 0, 0, 0, v[4], 0])[4] }, seed);
    const m = qToMotors(sol.q);
    m[5] = motors[5];  // keep the gripper
    if (sol.ok) {
      setDemo(false);
      setGoal(m);
      $('ikOut').innerHTML = `<span class="ok">Solved</span> · ${sol.iters} iterations`;
    } else if (sol.reason === 'roll') {
      $('ikOut').innerHTML = `<span class="bad">Roll out of range</span> · ${fx(qToMotors(KIN.range.map((r) => r[0]))[4], 1)}° … ${fx(qToMotors(KIN.range.map((r) => r[1]))[4], 1)}°`;
    } else {
      $('ikOut').innerHTML = `<span class="bad">No solution found</span> · ${fx(sol.posErr * 1000, 1)} mm, ${fx(sol.pitchErr / D2R, 1)}° off`;
    }
  };
}
// ------------------------------------------------------------------ main
async function main() {
  loading('Loading MuJoCo (WebAssembly, about 2.5 MB)…', 0.05);
  const t0 = performance.now();
  try {
    const { default: loadMujoco } = await import(MUJOCO_URL);
    mj = await loadMujoco();
  } catch (e) {
    fail(`MuJoCo could not be downloaded (${MUJOCO_URL}): check that this computer can reach cdn.jsdelivr.net. Error: ${e && e.message ? e.message : e}`);
    return;
  }
  const version = mj.mj_versionString();
  loading('Reading the model manifest…', 0.12);
  [MANIFEST, C, AF] = await Promise.all([fetchJSON('manifest.json'), fetchJSON('control.json'), fetchJSON('arm_frames.json')]);
  KIN = new ArmKinematics(AF);
  const edits = await fetchJSON('edits.json');
  const files = await loadModelFiles(MANIFEST);
  loading('Compiling the model (about 1 s)…', 0.65);
  await new Promise((r) => setTimeout(r, 20));
  writeTree(mj, files);
  const tc = performance.now();
  model = mj.MjModel.mj_loadXML(`${ROOT}/${MANIFEST.entry}`);
  const compileMs = performance.now() - tc;
  // the training model's calibrated values (edits.json), then the derived constants
  for (const [k, v] of Object.entries(edits.fields)) model[k].set(v);
  for (const [k, v] of Object.entries(edits.geometry_fields || {})) model[k].set(v);
  data = new mj.MjData(model);
  mj.mj_setConst(model, data);
  chassis = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY.value, C.chassis_body);
  for (const [k, v] of Object.entries(edits.geometry_fields || {})) model[k].set(v);
  model.stat.center.set(edits.stat.center);
  model.stat.extent = edits.stat.extent;
  model.opt.timestep = edits.timestep;

  loading('Building the 3D scene…', 0.85);
  const view = $('view');
  renderer = new THREE.WebGLRenderer({ canvas: $('c'), antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  scene = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(42, 1, 0.01, 30);
  camera.up.set(0, 0, 1);
  camera.position.set(0.62, 0.5, 0.45);  // front-left; fitView() frames it once the model is loaded
  scene.add(camera);
  scene.add(new THREE.HemisphereLight(0xffffff, 0x3a4550, 1.5));
  sun = new THREE.DirectionalLight(0xffffff, 1.8);
  sun.position.set(0.5, 0.3, 1.5);  // from the viewer's side, so the side we look at is lit
  sun.castShadow = true;
  sun.shadow.mapSize.set(2048, 2048);
  Object.assign(sun.shadow.camera, { left: -0.6, right: 0.6, top: 0.6, bottom: -0.6, near: 0.2, far: 4 });
  sun.shadow.bias = -0.0005;
  scene.add(sun);
  const head = new THREE.DirectionalLight(0xffffff, 0.6);
  head.position.set(0, 0, 1);
  camera.add(head);
  controls = new OrbitControls(camera, renderer.domElement);
  controls.target.set(0.12, 0, 0.18);
  controls.enableDamping = true;
  controls.addEventListener('start', () => { S.autoView = false; });  // moved by hand: keep that view
  controls.update();
  buildGeoms();
  buildFrames();
  buildPanel();
  buildKinematicsPanel();
  const canvas = renderer.domElement;
  canvas.addEventListener('pointerdown', onPointerDown, { capture: true });  // before OrbitControls
  canvas.addEventListener('pointermove', onPointerMove);
  canvas.addEventListener('pointerup', endDrag);
  canvas.addEventListener('pointercancel', endDrag);
  canvas.addEventListener('dblclick', (e) => {  // on the background: back to the default view
    raycaster.setFromCamera(ndc(e), camera);
    if (S.framesOnly || !raycaster.intersectObjects(armMeshes, false).length) { S.autoView = true; fitView(); }
  });
  window.addEventListener('keydown', (e) => {
    if (e.code === 'Space' && !['INPUT', 'BUTTON'].includes(document.activeElement.tagName)) { e.preventDefault(); togglePause(); }
  });
  const resize = () => {
    const w = view.clientWidth, h = view.clientHeight;
    renderer.setSize(w, h, false);
    camera.aspect = w / Math.max(1, h);
    camera.updateProjectionMatrix();
    if (S.autoView) fitView();
  };
  new ResizeObserver(resize).observe(view);
  resize();
  resetSim();
  loading('Fitting the default view to the whole demo motion…', 0.95);
  await new Promise((r) => setTimeout(r, 20));
  buildViewEnvelope();
  fitView();

  const faces = model.nmeshface;
  console.info(`MuJoCo ${version} (WebAssembly) · ${model.nbody} bodies · ${model.nu} motors (3 wheels locked) · ` +
    `${(faces / 1000).toFixed(0)}k triangles · loaded in ${((performance.now() - t0) / 1000).toFixed(1)} s, compiled in ${(compileMs / 1000).toFixed(1)} s`);
  if (version !== MANIFEST.mujoco) console.info(`note: the model was exported with MuJoCo ${MANIFEST.mujoco}`);
  window.lekiwiLoad = 1;
  $('loadBar').style.width = '100%';
  setTimeout(() => $('loading').classList.add('done'), 350);
  // for the browser console: LEKIWI.setGoal([0,-40,60,20,0,50]); LEKIWI.advance(1.0) steps 1 s at once
  const advance = (sec) => { for (let k = Math.round(sec / model.opt.timestep); k > 0; k--) stepPhysics(model.opt.timestep); syncScene(); };
  window.LEKIWI = { mj, model, data, C, S, KIN, setGoal, resetSim, advance, qToMotors, armQ, THREE, camera, scene, renderer, controls, fitView, get viewPts() { return viewPts; }, get drag() { return drag; } };

  S.lastWall = performance.now();
  const frame = (now) => {
    requestAnimationFrame(frame);
    const dt = (now - S.lastWall) / 1000;
    S.lastWall = now;
    physics(dt);
    controls.update();
    const world = syncScene();
    renderer.render(scene, camera);
    layoutTags(dt);
    updateBars();
    updatePanel(now, world);
  };
  requestAnimationFrame(frame);
  if (new URLSearchParams(location.search).has('demo')) setDemo(true);  // index.html?demo starts moving
}

main().catch((e) => fail('Something went wrong: ' + (e && e.message ? e.message : e)));
