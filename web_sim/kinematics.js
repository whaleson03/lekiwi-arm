// Kinematics of the LeKiwi follower arm (SO-101) from model/arm_frames.json. No MuJoCo needed, no dependencies:
// use it in the page, in Node, or as the reference to check your own FK / IK against.
//
//   import { ArmKinematics } from './kinematics.js';
//   const kin = new ArmKinematics(await (await fetch('model/arm_frames.json')).json());
//   const q = kin.toQ([0, -97, 92, 66, 0, 9]);          // LeRobot units -> sim joint angles (rad)
//   const T = kin.fk(q).TCP;                             // TCP pose in F0: 4x4 row-major, metres
//   const sol = kin.ik({ pos: [0.2, 0, 0.1], pitch: Math.PI / 2, roll: 0 }, q);   // gripper pointing down
//
// Conventions: matrices are 4x4 row-major arrays of 16 numbers, lengths in metres, angles in
// radians. q = the 6 sim joint angles [pan, lift, elbow, wrist_flex, wrist_roll, gripper]; with the zero offsets
// of this robot (0) q_i = LeRobot degrees * pi/180 for the first 5. Every joint frame's z is its joint axis in the
// positive direction, so FK is T_F0_k = T_F0_parent * A_k * Rz(q_k) down the chain F0 -> F1 ... F6, TCP.

export function mul(A, B) {
  const C = new Array(16);
  for (let i = 0; i < 4; i++) {
    for (let j = 0; j < 4; j++) {
      C[4 * i + j] = A[4 * i] * B[j] + A[4 * i + 1] * B[4 + j] + A[4 * i + 2] * B[8 + j] + A[4 * i + 3] * B[12 + j];
    }
  }
  return C;
}
export function rotZ(q) { const c = Math.cos(q), s = Math.sin(q); return [c, -s, 0, 0, s, c, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]; }
export function invRigid(T) {  // inverse of a rotation + translation
  const R = [T[0], T[4], T[8], T[1], T[5], T[9], T[2], T[6], T[10]];  // R^T
  const p = [T[3], T[7], T[11]];
  return [R[0], R[1], R[2], -(R[0] * p[0] + R[1] * p[1] + R[2] * p[2]),
          R[3], R[4], R[5], -(R[3] * p[0] + R[4] * p[1] + R[5] * p[2]),
          R[6], R[7], R[8], -(R[6] * p[0] + R[7] * p[1] + R[8] * p[2]), 0, 0, 0, 1];
}
export const position = (T) => [T[3], T[7], T[11]];
export const axis = (T, i) => [T[i], T[4 + i], T[8 + i]];          // column i: x, y or z axis of the frame
export const pitchOf = (T) => Math.asin(Math.max(-1, Math.min(1, -T[10])));  // approach (z) below horizontal
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];

export class ArmKinematics {
  constructor(json) {
    this.json = json;
    this.joints = json.joints;
    this.frames = json.frames.map((f) => ({ ...f, A: f.A.flat() }));
    this.range = [];       // commandable range per joint (rad), the LeRobot calibrated range
    for (const f of this.frames) if (f.joint) this.range[f.joint_index] = f.ctrl_range_rad || f.range_rad;
    const L = json.lerobot || {};
    this.zeroOffDeg = L.zero_offset_deg || [0, 0, 0, 0, 0];
    this.gripClosed = L.gripper_closed_rad;
    this.gripOpen = L.gripper_open_rad;
    if (json.mdh) {
      this.dh = json.mdh.rows.map((r) => [r.alpha_prev_rad, r.a_prev_m, r.theta_offset_rad, r.d_m]);
      this.tool = json.mdh.tool.flat();
    }
  }

  // ------------------------------------------------------------ units
  toQ(m6) {  // LeRobot [deg x5, gripper %] -> sim angles (rad); x * (pi / 180) like Python's radians()
    const q = m6.slice(0, 5).map((v, i) => (v + this.zeroOffDeg[i]) * (Math.PI / 180));
    q.push(this.gripClosed + (m6[5] ?? 0) / 100 * (this.gripOpen - this.gripClosed));
    return q;
  }
  toLeRobot(q) {
    const m = q.slice(0, 5).map((v, i) => v / (Math.PI / 180) - this.zeroOffDeg[i]);
    m.push(((q[5] ?? this.gripClosed) - this.gripClosed) / (this.gripOpen - this.gripClosed) * 100);
    return m;
  }

  // ------------------------------------------------------------ forward kinematics
  fk(q) {  // every frame (F0 ... F6, TCP) in F0
    const out = {};
    for (const f of this.frames) {
      let T = f.parent ? mul(out[f.parent], f.A) : f.A.slice();
      if (f.joint) T = mul(T, rotZ(q[f.joint_index]));
      out[f.key] = T;
    }
    return out;
  }
  fkDH(q) {  // TCP in F0 from the modified DH table (Craig): same result as fk(q).TCP
    let T = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
    for (let i = 0; i < 5; i++) {
      const [al, a, th, d] = this.dh[i], ca = Math.cos(al), sa = Math.sin(al);
      const Rx = [1, 0, 0, a, 0, ca, -sa, 0, 0, sa, ca, 0, 0, 0, 0, 1];
      const c = Math.cos(q[i] + th), s = Math.sin(q[i] + th);
      T = mul(mul(T, Rx), [c, -s, 0, 0, s, c, 0, 0, 0, 0, 1, d, 0, 0, 0, 1]);
    }
    return mul(T, this.tool);
  }

  // ------------------------------------------------------------ inverse kinematics (reference, numerical)
  // The arm has 5 joints for the TCP, so a target is: TCP position (3) + approach pitch (1) + wrist roll (1).
  // pos: [x, y, z] in F0 (m); pitch: approach angle below horizontal (rad, pi/2 = pointing straight down);
  // roll: the wrist_roll joint angle (rad), taken as given; outside its range the target is refused (ok false,
  // reason 'roll'). Solves pan, lift, elbow, wrist_flex by damped Gauss-Newton (Levenberg-Marquardt) inside the
  // commandable ranges, from several starts (seed, default q0 below, plus analytic guesses); ok false with no
  // reason means no start converged, not that the target is out of reach.
  // Returns { q (6, gripper copied from the seed), ok, reason, posErr (m), pitchErr (rad), iters }.
  ik(target, seed, opts = {}) {
    const W = opts.pitchWeight ?? 0.1;  // metres per radian when mixing the errors
    const tolP = opts.tolPos ?? 1e-6, tolA = opts.tolPitch ?? 1e-5, maxIt = opts.maxIter ?? 200;
    seed = seed || this.toQ([0, -97, 92, 66, 0, 0]);
    const rr = this.range[4];
    if (target.roll !== undefined && (target.roll < rr[0] - 1e-9 || target.roll > rr[1] + 1e-9)) {
      return { q: seed.slice(), ok: false, reason: 'roll', posErr: NaN, pitchErr: NaN, iters: 0, cost: Infinity };
    }
    const grip = (s) => [...s.slice(0, 5), seed[5] ?? 0];  // every start keeps the seed's gripper
    const starts = [...(this.dh ? this.ikGuess(target) : []), seed,
                    ...(opts.extraSeeds ?? [this.toQ([0, -60, 60, 60, 0, 0]), this.toQ([0, 0, 0, 0, 0, 0])])].map(grip);
    const sols = starts.map((s) => this._solve(target, s, W, tolP, tolA, maxIt));
    const good = sols.filter((r) => r.ok);
    const dist = (q) => q.slice(0, 4).reduce((a, v, i) => a + (v - seed[i]) ** 2, 0);
    if (good.length) return good.reduce((b, r) => (dist(r.q) < dist(b.q) ? r : b));  // nearest to the seed
    return sols.reduce((b, r) => (r.cost < b.cost ? r : b));
  }

  // Analytic first guesses by wrist-centre decoupling, read off the DH table:
  //  - the TCP target (position, approach pitch, roll) gives the wrist centre W = TCP - R5 * tool translation;
  //  - pan turns the arm plane toward W (pan positive = clockwise from above, so pan = -heading), or away from
  //    it with the arm reaching back over the top;
  //  - lift and elbow are a planar 2-link arm from the lift axis (a1 forward of the pan axis, h1 above F0) to W,
  //    links L1 = a2 and L2 = a3; in-plane angles: upper arm = -(theta2_off + q2), forearm = upper arm -
  //    (theta3_off + q3); the approach direction (in the arm plane) = forearm - (theta4_off + q4) - 90 deg.
  // 8 candidates: arm facing W or reaching back x gripper pointing forward or backward in the arm plane (same
  // pitch) x elbow up or down. They ignore the 0.18 mm lateral offset of the roll axis; ik() refines them.
  ikGuess(target) {
    const [, r2, r3, r4] = this.dh;
    const h1 = -this.dh[0][3], a1 = r2[1], L1 = r3[1], L2 = r4[1];
    const th2 = r2[2], th3 = r3[2], th4 = r4[2];
    const t = [this.tool[3], this.tool[7], this.tool[11]];  // TCP in M5 (m)
    const p = target.pos, pit = target.pitch, q5 = target.roll ?? 0;
    const wrap = (x) => Math.atan2(Math.sin(x), Math.cos(x));
    const out = [];
    for (const back of [0, 1]) {
      for (const theta of [pit, Math.PI - pit]) {  // approach angle below the arm's forward direction
        let psi = Math.atan2(p[1], p[0]) + back * Math.PI, Wc = p;
        for (let it = 0; it < 3; it++) {  // the heading and W depend on each other a little
          const a = [Math.cos(psi) * Math.cos(theta), Math.sin(psi) * Math.cos(theta), -Math.sin(theta)];
          const n = [Math.cos(psi) * Math.sin(theta), Math.sin(psi) * Math.sin(theta), Math.cos(theta)];
          const z5 = a.map((v) => -v), y0 = cross(z5, n);
          const x5 = n.map((v, i) => Math.cos(q5) * v + Math.sin(q5) * y0[i]);
          const y5 = n.map((v, i) => -Math.sin(q5) * v + Math.cos(q5) * y0[i]);
          Wc = p.map((v, i) => v - (t[0] * x5[i] + t[1] * y5[i] + t[2] * z5[i]));
          psi = Math.atan2(Wc[1], Wc[0]) + back * Math.PI;
        }
        const r = Wc[0] * Math.cos(psi) + Wc[1] * Math.sin(psi);  // signed, along the arm's heading
        const dr = r - a1, dh = Wc[2] - h1;
        const c = (dr * dr + dh * dh - L1 * L1 - L2 * L2) / (2 * L1 * L2);
        for (const sgn of [1, -1]) {
          const beta = sgn * Math.acos(Math.max(-1, Math.min(1, c)));  // forearm relative to the upper arm
          const phiU = Math.atan2(dh, dr) + Math.atan2(L2 * Math.sin(beta), L1 + L2 * Math.cos(beta));
          const q2 = -th2 - phiU, q3 = beta - th3, phiF = phiU - beta;
          const q4 = wrap(phiF - th4 - (Math.PI / 2 - theta));  // x4 sits 90 deg above the approach
          out.push([wrap(-psi), wrap(q2), wrap(q3), q4, q5, 0]);
        }
      }
    }
    return out;
  }

  _solve(target, seed, W, tolP, tolA, maxIt) {
    const clampJ = (x) => x.map((v, i) => Math.min(this.range[i][1], Math.max(this.range[i][0], v)));
    const roll = Math.min(this.range[4][1], Math.max(this.range[4][0], target.roll ?? seed[4]));
    const full = (x) => [x[0], x[1], x[2], x[3], roll, seed[5] ?? 0];
    const resid = (x) => {
      const T = this.fk(full(x)).TCP;
      return [T[3] - target.pos[0], T[7] - target.pos[1], T[11] - target.pos[2], W * (pitchOf(T) - target.pitch)];
    };
    const norm2 = (r) => r.reduce((s, v) => s + v * v, 0);
    let x = clampJ(seed.slice(0, 4)), r = resid(x), cost = norm2(r), lam = 1e-3, it = 0;
    for (; it < maxIt; it++) {
      if (Math.hypot(r[0], r[1], r[2]) < tolP && Math.abs(r[3] / W) < tolA) break;
      const J = [];  // 4 x 4, numerical
      for (let j = 0; j < 4; j++) {
        const h = 1e-6, xp = x.slice();
        xp[j] += h;
        const rp = resid(xp);
        for (let i = 0; i < 4; i++) (J[i] ||= [])[j] = (rp[i] - r[i]) / h;
      }
      const JtJ = [0, 1, 2, 3].map((a) => [0, 1, 2, 3].map((b) => J.reduce((s, row) => s + row[a] * row[b], 0)));
      const Jtr = [0, 1, 2, 3].map((a) => J.reduce((s, row, i) => s + row[a] * r[i], 0));
      let improved = false;
      for (let k = 0; k < 10 && !improved; k++) {
        const Mx = JtJ.map((row, a) => row.map((v, b) => v + (a === b ? lam * (1 + v) : 0)));
        const dx = solve4(Mx, Jtr.map((v) => -v));
        const xn = clampJ(x.map((v, i) => v + dx[i]));
        const rn = resid(xn), cn = norm2(rn);
        if (cn < cost) { x = xn; r = rn; cost = cn; lam = Math.max(lam / 3, 1e-9); improved = true; } else lam *= 4;
      }
      if (!improved) break;  // stuck: a joint limit or an unreachable target
    }
    const posErr = Math.hypot(r[0], r[1], r[2]), pitchErr = Math.abs(r[3] / W);
    return { q: full(x), ok: posErr < Math.max(tolP, 1e-5) && pitchErr < Math.max(tolA, 1e-4), posErr, pitchErr, iters: it, cost };
  }
}

function solve4(A, b) {  // Gaussian elimination with partial pivoting (4 x 4)
  const M = A.map((row, i) => [...row, b[i]]), n = 4;
  for (let c = 0; c < n; c++) {
    let p = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
    [M[c], M[p]] = [M[p], M[c]];
    const piv = M[c][c] || 1e-12;
    for (let r = c + 1; r < n; r++) {
      const f = M[r][c] / piv;
      for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k];
    }
  }
  const x = new Array(n).fill(0);
  for (let r = n - 1; r >= 0; r--) {
    let s = M[r][n];
    for (let k = r + 1; k < n; k++) s -= M[r][k] * x[k];
    x[r] = s / (M[r][r] || 1e-12);
  }
  return x;
}
