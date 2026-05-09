const canvas = document.getElementById("arena");
const ctx = canvas.getContext("2d");

const state = {
  tool: "select",
  arenaHalf: 3.5,
  mapName: "custom_map_01",
  baseTask: "open_clutter",
  start: { x: -1.6, y: -1.2, yaw: 0.0 },
  goal: { x: 1.6, y: 1.2 },
  obstacles: [],
  selected: { kind: "start", id: null },
  dragging: false,
  wallDraft: null,
  filePath: "not saved",
  jitter: { enabled: false, start_std: 0, goal_std: 0, yaw_std: 0 },
  safety: {
    robotRadius: 0.18,
    goalRadius: 0.24,
    approachMargin: 0.12,
    minObstacleGap: 0.32,
    showBuffers: true,
  },
  analysis: null,
};

const els = {
  mapName: document.getElementById("mapName"),
  baseTask: document.getElementById("baseTask"),
  arenaHalf: document.getElementById("arenaHalf"),
  status: document.getElementById("status"),
  cursor: document.getElementById("cursorReadout"),
  mapList: document.getElementById("mapList"),
  obstacleCount: document.getElementById("obstacleCount"),
  goalDistance: document.getElementById("goalDistance"),
  nearestGap: document.getElementById("nearestGap"),
  robotCorridor: document.getElementById("robotCorridor"),
  startClearance: document.getElementById("startClearance"),
  goalClearance: document.getElementById("goalClearance"),
  directRouteClearance: document.getElementById("directRouteClearance"),
  selectedGap: document.getElementById("selectedGap"),
  safetyWarnings: document.getElementById("safetyWarnings"),
  filePath: document.getElementById("filePath"),
  selectionKind: document.getElementById("selectionKind"),
  fieldX: document.getElementById("fieldX"),
  fieldY: document.getElementById("fieldY"),
  fieldA: document.getElementById("fieldA"),
  fieldB: document.getElementById("fieldB"),
  fieldC: document.getElementById("fieldC"),
  fieldALabel: document.getElementById("fieldALabel"),
  fieldBLabel: document.getElementById("fieldBLabel"),
  fieldCLabel: document.getElementById("fieldCLabel"),
  jitterEnabled: document.getElementById("jitterEnabled"),
  jitterStart: document.getElementById("jitterStart"),
  jitterGoal: document.getElementById("jitterGoal"),
  jitterYaw: document.getElementById("jitterYaw"),
  robotRadius: document.getElementById("robotRadius"),
  approachMargin: document.getElementById("approachMargin"),
  minObstacleGap: document.getElementById("minObstacleGap"),
  showSafetyBuffers: document.getElementById("showSafetyBuffers"),
};

function setStatus(text) {
  els.status.textContent = text;
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function worldToCanvas(x, y) {
  const half = state.arenaHalf;
  const px = ((x + half) / (2 * half)) * canvas.width;
  const py = canvas.height - ((y + half) / (2 * half)) * canvas.height;
  return { x: px, y: py };
}

function canvasToWorld(px, py) {
  const half = state.arenaHalf;
  return {
    x: clamp((px / canvas.width) * 2 * half - half, -half, half),
    y: clamp(((canvas.height - py) / canvas.height) * 2 * half - half, -half, half),
  };
}

function scaleMeters(value) {
  return (value / (2 * state.arenaHalf)) * canvas.width;
}

function formatMeters(value) {
  if (!Number.isFinite(value)) return "--";
  return `${value.toFixed(2)} m`;
}

function setMetric(el, value, level = "ok", detail = "") {
  el.textContent = detail || formatMeters(value);
  el.classList.remove("metric-ok", "metric-warn", "metric-bad");
  if (detail === "--" || !Number.isFinite(value)) return;
  el.classList.add(level === "bad" ? "metric-bad" : level === "warn" ? "metric-warn" : "metric-ok");
}

function syncSafetyFromInputs() {
  state.safety.robotRadius = clamp(Number(els.robotRadius.value || 0.18), 0.05, state.arenaHalf);
  state.safety.approachMargin = clamp(Number(els.approachMargin.value || 0), 0, state.arenaHalf);
  state.safety.minObstacleGap = clamp(Number(els.minObstacleGap.value || 0), 0, state.arenaHalf * 2);
  state.safety.showBuffers = Boolean(els.showSafetyBuffers.checked);
}

function syncSafetyInputs() {
  els.robotRadius.value = Number(state.safety.robotRadius).toFixed(2);
  els.approachMargin.value = Number(state.safety.approachMargin).toFixed(2);
  els.minObstacleGap.value = Number(state.safety.minObstacleGap).toFixed(2);
  els.showSafetyBuffers.checked = Boolean(state.safety.showBuffers);
}

function applySafetyDefaults(config) {
  const robotRadius = Number(config?.robot?.radius);
  const goalRadius = Number(config?.goal?.radius);
  const minObstacleGap = Number(config?.obstacles?.min_clearance);
  if (Number.isFinite(robotRadius) && robotRadius > 0) state.safety.robotRadius = robotRadius;
  if (Number.isFinite(goalRadius) && goalRadius > 0) state.safety.goalRadius = goalRadius;
  if (Number.isFinite(minObstacleGap) && minObstacleGap >= 0) state.safety.minObstacleGap = minObstacleGap;
  syncSafetyInputs();
}

function selectedObstacle() {
  if (state.selected.kind !== "obstacle") return null;
  return state.obstacles.find((item) => item.id === state.selected.id) || null;
}

function defaultStartGoal() {
  const offset = state.arenaHalf * 0.58;
  return {
    start: { x: -offset, y: -offset, yaw: 0.0 },
    goal: { x: offset, y: offset },
  };
}

function clearObstacles() {
  if (!state.obstacles.length) {
    setStatus("No obstacles to clear");
    return;
  }
  const removed = state.obstacles.length;
  state.obstacles = [];
  state.selected = { kind: "none", id: null };
  state.wallDraft = null;
  state.dragging = false;
  setStatus(`Cleared ${removed} obstacles`);
  updateInspector();
  draw();
}

function newBlankMap() {
  state.arenaHalf = Number(els.arenaHalf.value || state.arenaHalf);
  state.baseTask = els.baseTask.value || state.baseTask;
  state.mapName = "custom_map_new";
  const defaults = defaultStartGoal();
  state.start = defaults.start;
  state.goal = defaults.goal;
  state.obstacles = [];
  state.selected = { kind: "start", id: null };
  state.wallDraft = null;
  state.dragging = false;
  state.filePath = "not saved";
  state.jitter = { enabled: false, start_std: 0, goal_std: 0, yaw_std: 0 };
  els.mapName.value = state.mapName;
  els.jitterEnabled.checked = false;
  els.jitterStart.value = 0;
  els.jitterGoal.value = 0;
  els.jitterYaw.value = 0;
  setStatus("Blank map ready");
  updateInspector();
  draw();
}

function localBoxPoint(obs, point) {
  const dx = point.x - obs.x;
  const dy = point.y - obs.y;
  const c = Math.cos(obs.yaw || 0);
  const s = Math.sin(obs.yaw || 0);
  return { x: c * dx + s * dy, y: -s * dx + c * dy };
}

function boxCorners(obs) {
  const hx = obs.half_x;
  const hy = obs.half_y;
  const yaw = obs.yaw || 0;
  const c = Math.cos(yaw);
  const s = Math.sin(yaw);
  return [
    { x: -hx, y: -hy },
    { x: hx, y: -hy },
    { x: hx, y: hy },
    { x: -hx, y: hy },
  ].map((p) => ({ x: obs.x + c * p.x - s * p.y, y: obs.y + s * p.x + c * p.y }));
}

function dot(a, b) {
  return a.x * b.x + a.y * b.y;
}

function pointSegmentDistance(point, a, b) {
  const ab = { x: b.x - a.x, y: b.y - a.y };
  const denom = dot(ab, ab);
  const t = denom <= 1e-9 ? 0 : clamp(dot({ x: point.x - a.x, y: point.y - a.y }, ab) / denom, 0, 1);
  const closest = { x: a.x + ab.x * t, y: a.y + ab.y * t };
  return { distance: Math.hypot(point.x - closest.x, point.y - closest.y), closest };
}

function orientation(a, b, c) {
  return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
}

function onSegment(a, b, p) {
  return (
    Math.min(a.x, b.x) - 1e-9 <= p.x &&
    p.x <= Math.max(a.x, b.x) + 1e-9 &&
    Math.min(a.y, b.y) - 1e-9 <= p.y &&
    p.y <= Math.max(a.y, b.y) + 1e-9
  );
}

function segmentsIntersect(a, b, c, d) {
  const o1 = orientation(a, b, c);
  const o2 = orientation(a, b, d);
  const o3 = orientation(c, d, a);
  const o4 = orientation(c, d, b);
  if (Math.abs(o1) < 1e-9 && onSegment(a, b, c)) return true;
  if (Math.abs(o2) < 1e-9 && onSegment(a, b, d)) return true;
  if (Math.abs(o3) < 1e-9 && onSegment(c, d, a)) return true;
  if (Math.abs(o4) < 1e-9 && onSegment(c, d, b)) return true;
  return o1 * o2 < 0 && o3 * o4 < 0;
}

function segmentDistance(a, b, c, d) {
  if (segmentsIntersect(a, b, c, d)) {
    return { distance: 0, pointA: a, pointB: a };
  }
  const candidates = [
    { ...pointSegmentDistance(a, c, d), pointA: a },
    { ...pointSegmentDistance(b, c, d), pointA: b },
    { ...pointSegmentDistance(c, a, b), pointB: c },
    { ...pointSegmentDistance(d, a, b), pointB: d },
  ].map((item) => ({
    distance: item.distance,
    pointA: item.pointA || item.closest,
    pointB: item.pointB || item.closest,
  }));
  return candidates.reduce((best, item) => (item.distance < best.distance ? item : best), candidates[0]);
}

function polygonSegments(points) {
  return points.map((point, idx) => [point, points[(idx + 1) % points.length]]);
}

function pointInPolygon(point, points) {
  const axes = polygonAxes(points);
  return axes.every((axis) => {
    const projected = projectPolygon(points, axis);
    const p = dot(point, axis);
    return p >= projected.min - 1e-9 && p <= projected.max + 1e-9;
  });
}

function polygonAxes(points) {
  return polygonSegments(points).map(([a, b]) => {
    const edge = { x: b.x - a.x, y: b.y - a.y };
    const length = Math.hypot(edge.x, edge.y) || 1;
    return { x: -edge.y / length, y: edge.x / length };
  });
}

function projectPolygon(points, axis) {
  let min = Infinity;
  let max = -Infinity;
  points.forEach((point) => {
    const value = dot(point, axis);
    min = Math.min(min, value);
    max = Math.max(max, value);
  });
  return { min, max };
}

function polygonSignedDistance(a, b) {
  let overlap = Infinity;
  for (const axis of [...polygonAxes(a), ...polygonAxes(b)]) {
    const pa = projectPolygon(a, axis);
    const pb = projectPolygon(b, axis);
    const separated = Math.min(pa.max, pb.max) - Math.max(pa.min, pb.min);
    if (separated < 0) {
      let distance = Infinity;
      for (const [a0, a1] of polygonSegments(a)) {
        for (const [b0, b1] of polygonSegments(b)) {
          distance = Math.min(distance, segmentDistance(a0, a1, b0, b1).distance);
        }
      }
      return distance;
    }
    overlap = Math.min(overlap, separated);
  }
  return -overlap;
}

function obstaclePairClearance(a, b) {
  if (a.shape !== "box" && b.shape !== "box") {
    return Math.hypot(a.x - b.x, a.y - b.y) - Number(a.radius || 0) - Number(b.radius || 0);
  }
  if (a.shape === "box" && b.shape === "box") {
    return polygonSignedDistance(boxCorners(a), boxCorners(b));
  }
  const box = a.shape === "box" ? a : b;
  const circle = a.shape === "box" ? b : a;
  return obstacleDistance(box, circle) - Number(circle.radius || 0);
}

function nearestObstaclePair(obstacles = state.obstacles) {
  let best = null;
  for (let i = 0; i < obstacles.length; i += 1) {
    for (let j = i + 1; j < obstacles.length; j += 1) {
      const gap = obstaclePairClearance(obstacles[i], obstacles[j]);
      if (!best || gap < best.gap) {
        best = { gap, a: obstacles[i], b: obstacles[j] };
      }
    }
  }
  return best;
}

function nearestSelectedPair(obs) {
  let best = null;
  state.obstacles.forEach((other) => {
    if (other.id === obs.id) return;
    const gap = obstaclePairClearance(obs, other);
    if (!best || gap < best.gap) {
      best = { gap, a: obs, b: other };
    }
  });
  return best;
}

function nearestPointClearance(point, bodyRadius) {
  let best = null;
  state.obstacles.forEach((obs) => {
    const clearance = obstacleDistance(obs, point) - bodyRadius;
    if (!best || clearance < best.clearance) {
      best = { clearance, obstacle: obs };
    }
  });
  return best;
}

function segmentToObstacleClearance(start, end, obs, bodyRadius) {
  if (obs.shape !== "box") {
    return pointSegmentDistance({ x: obs.x, y: obs.y }, start, end).distance - Number(obs.radius || 0) - bodyRadius;
  }
  const poly = boxCorners(obs);
  if (pointInPolygon(start, poly) || pointInPolygon(end, poly)) {
    return -bodyRadius;
  }
  let distance = Infinity;
  for (const [a, b] of polygonSegments(poly)) {
    const item = segmentDistance(start, end, a, b);
    distance = Math.min(distance, item.distance);
    if (distance <= 0) break;
  }
  return distance - bodyRadius;
}

function directRouteClearance() {
  if (!state.obstacles.length) return null;
  const start = { x: state.start.x, y: state.start.y };
  const end = { x: state.goal.x, y: state.goal.y };
  let best = null;
  state.obstacles.forEach((obs) => {
    const clearance = segmentToObstacleClearance(start, end, obs, state.safety.robotRadius);
    if (!best || clearance < best.clearance) {
      best = { clearance, obstacle: obs };
    }
  });
  return best;
}

function analyzeMap() {
  syncSafetyFromInputs();
  const warnings = [];
  const startGoalDistance = Math.hypot(state.goal.x - state.start.x, state.goal.y - state.start.y);
  const nearestPair = nearestObstaclePair();
  const startClearance = nearestPointClearance(state.start, state.safety.robotRadius);
  const goalClearance = nearestPointClearance(state.goal, state.safety.goalRadius);
  const directRoute = directRouteClearance();
  const selected = selectedObstacle();
  const selectedPair = selected ? nearestSelectedPair(selected) : null;
  const requiredCorridor = 2 * (state.safety.robotRadius + state.safety.approachMargin);

  if (startGoalDistance < 0.4) warnings.push({ level: "invalid", text: "Start and goal are too close." });
  if (nearestPair) {
    if (nearestPair.gap < 0) {
      warnings.push({ level: "invalid", text: `${nearestPair.a.id} overlaps ${nearestPair.b.id}.` });
    } else if (nearestPair.gap < requiredCorridor) {
      warnings.push({
        level: "warn",
        text: `${nearestPair.a.id} - ${nearestPair.b.id} gap is ${formatMeters(nearestPair.gap)}; robot+margin needs ${formatMeters(requiredCorridor)}.`,
      });
    } else if (nearestPair.gap < state.safety.minObstacleGap) {
      warnings.push({
        level: "warn",
        text: `${nearestPair.a.id} - ${nearestPair.b.id} gap is below ${formatMeters(state.safety.minObstacleGap)}.`,
      });
    }
  }
  if (startClearance) {
    if (startClearance.clearance < 0) {
      warnings.push({ level: "invalid", text: `Start overlaps ${startClearance.obstacle.id}.` });
    } else if (startClearance.clearance < state.safety.approachMargin) {
      warnings.push({
        level: "warn",
        text: `Start clearance to ${startClearance.obstacle.id} is ${formatMeters(startClearance.clearance)}.`,
      });
    }
  }
  if (goalClearance) {
    if (goalClearance.clearance < 0) {
      warnings.push({ level: "invalid", text: `Goal overlaps ${goalClearance.obstacle.id}.` });
    } else if (goalClearance.clearance < state.safety.approachMargin) {
      warnings.push({
        level: "warn",
        text: `Goal clearance to ${goalClearance.obstacle.id} is ${formatMeters(goalClearance.clearance)}.`,
      });
    }
  }
  if (directRoute && directRoute.clearance < 0) {
    warnings.push({ level: "warn", text: `Straight start-goal route intersects ${directRoute.obstacle.id}.` });
  } else if (directRoute && directRoute.clearance < state.safety.approachMargin) {
    warnings.push({
      level: "warn",
      text: `Straight route clearance to ${directRoute.obstacle.id} is ${formatMeters(directRoute.clearance)}.`,
    });
  }

  return {
    startGoalDistance,
    nearestPair,
    robotCorridor: nearestPair ? nearestPair.gap - 2 * state.safety.robotRadius : Infinity,
    startClearance,
    goalClearance,
    directRoute,
    selectedPair,
    requiredCorridor,
    warnings,
  };
}

function currentSelection() {
  if (state.selected.kind === "start") return state.start;
  if (state.selected.kind === "goal") return state.goal;
  return selectedObstacle();
}

function hitTest(world) {
  const startDistance = Math.hypot(world.x - state.start.x, world.y - state.start.y);
  if (startDistance < state.safety.robotRadius) return { kind: "start", id: null };
  const goalDistance = Math.hypot(world.x - state.goal.x, world.y - state.goal.y);
  if (goalDistance < state.safety.goalRadius) return { kind: "goal", id: null };
  for (let i = state.obstacles.length - 1; i >= 0; i--) {
    const obs = state.obstacles[i];
    if (obs.shape === "box") {
      const local = localBoxPoint(obs, world);
      if (Math.abs(local.x) <= obs.half_x && Math.abs(local.y) <= obs.half_y) {
        return { kind: "obstacle", id: obs.id };
      }
    } else if (Math.hypot(world.x - obs.x, world.y - obs.y) <= obs.radius) {
      return { kind: "obstacle", id: obs.id };
    }
  }
  return { kind: "none", id: null };
}

function addObstacle(shape, world) {
  const id = `${shape}_${Date.now()}`;
  if (shape === "box") {
    state.obstacles.push({ id, shape, x: world.x, y: world.y, half_x: 0.22, half_y: 0.22, radius: 0.22, yaw: 0.0, kind: "box" });
  } else {
    state.obstacles.push({ id, shape: "cylinder", x: world.x, y: world.y, radius: 0.22, half_x: 0.22, half_y: 0.22 });
  }
  state.selected = { kind: "obstacle", id };
  updateInspector();
  draw();
}

function finalizeWall(start, end) {
  const dx = end.x - start.x;
  const dy = end.y - start.y;
  const length = Math.hypot(dx, dy);
  if (length < 0.12) {
    setStatus("Wall too short");
    return;
  }
  const id = `wall_${Date.now()}`;
  const halfX = Math.max(0.08, length * 0.5);
  const halfY = 0.08;
  state.obstacles.push({
    id,
    shape: "box",
    kind: "wall",
    x: (start.x + end.x) * 0.5,
    y: (start.y + end.y) * 0.5,
    half_x: halfX,
    half_y: halfY,
    radius: halfX,
    yaw: Math.atan2(dy, dx),
  });
  state.selected = { kind: "obstacle", id };
  setStatus("Wall added");
}

function drawGrid() {
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  const half = state.arenaHalf;
  ctx.strokeStyle = "#e1e6ec";
  ctx.lineWidth = 1;
  for (let x = Math.ceil(-half); x <= half; x += 1) {
    const p0 = worldToCanvas(x, -half);
    const p1 = worldToCanvas(x, half);
    ctx.beginPath();
    ctx.moveTo(p0.x, p0.y);
    ctx.lineTo(p1.x, p1.y);
    ctx.stroke();
  }
  for (let y = Math.ceil(-half); y <= half; y += 1) {
    const p0 = worldToCanvas(-half, y);
    const p1 = worldToCanvas(half, y);
    ctx.beginPath();
    ctx.moveTo(p0.x, p0.y);
    ctx.lineTo(p1.x, p1.y);
    ctx.stroke();
  }
  ctx.strokeStyle = "#64748b";
  ctx.lineWidth = 2;
  ctx.strokeRect(1, 1, canvas.width - 2, canvas.height - 2);
}

function drawSafetyBuffers() {
  if (!state.safety.showBuffers || !state.obstacles.length) return;
  const margin = state.safety.robotRadius + state.safety.approachMargin;
  ctx.save();
  ctx.setLineDash([10, 7]);
  ctx.fillStyle = "rgba(37, 99, 235, 0.07)";
  ctx.strokeStyle = "rgba(37, 99, 235, 0.45)";
  ctx.lineWidth = 2;
  state.obstacles.forEach((obs) => {
    if (obs.shape === "box") {
      const expanded = { ...obs, half_x: obs.half_x + margin, half_y: obs.half_y + margin };
      const corners = boxCorners(expanded).map((point) => worldToCanvas(point.x, point.y));
      ctx.beginPath();
      ctx.moveTo(corners[0].x, corners[0].y);
      corners.slice(1).forEach((point) => ctx.lineTo(point.x, point.y));
      ctx.closePath();
      ctx.fill();
      ctx.stroke();
    } else {
      const p = worldToCanvas(obs.x, obs.y);
      const r = scaleMeters(Number(obs.radius || 0) + margin);
      ctx.beginPath();
      ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
    }
  });
  ctx.restore();
}

function drawNearestGap(analysis) {
  const pair = analysis?.nearestPair;
  if (!pair) return;
  const a = worldToCanvas(pair.a.x, pair.a.y);
  const b = worldToCanvas(pair.b.x, pair.b.y);
  const alert = pair.gap < analysis.requiredCorridor;
  ctx.save();
  ctx.setLineDash([8, 6]);
  ctx.strokeStyle = alert ? "#dc2626" : "#2563eb";
  ctx.lineWidth = alert ? 3 : 2;
  ctx.beginPath();
  ctx.moveTo(a.x, a.y);
  ctx.lineTo(b.x, b.y);
  ctx.stroke();
  ctx.setLineDash([]);

  const label = `gap ${pair.gap.toFixed(2)} m`;
  const mx = (a.x + b.x) * 0.5;
  const my = (a.y + b.y) * 0.5;
  ctx.font = "12px Inter, sans-serif";
  const width = ctx.measureText(label).width + 12;
  ctx.fillStyle = alert ? "rgba(254, 242, 242, 0.95)" : "rgba(239, 246, 255, 0.95)";
  ctx.strokeStyle = alert ? "#dc2626" : "#2563eb";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.roundRect(mx - width * 0.5, my - 11, width, 22, 6);
  ctx.fill();
  ctx.stroke();
  ctx.fillStyle = alert ? "#991b1b" : "#174ea6";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(label, mx, my);
  ctx.restore();
}

function drawObstacle(obs) {
  const p = worldToCanvas(obs.x, obs.y);
  const selected = state.selected.kind === "obstacle" && state.selected.id === obs.id;
  ctx.fillStyle = obs.shape === "box" ? "#c2762f" : "#bc4d38";
  ctx.strokeStyle = selected ? "#111827" : "#7c2d12";
  ctx.lineWidth = selected ? 4 : 2;
  if (obs.shape === "box") {
    const corners = boxCorners(obs).map((point) => worldToCanvas(point.x, point.y));
    ctx.beginPath();
    ctx.moveTo(corners[0].x, corners[0].y);
    corners.slice(1).forEach((point) => ctx.lineTo(point.x, point.y));
    ctx.closePath();
    ctx.fill();
    ctx.stroke();
  } else {
    const r = scaleMeters(obs.radius);
    ctx.beginPath();
    ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
  }
}

function drawStartGoal() {
  const start = worldToCanvas(state.start.x, state.start.y);
  const goal = worldToCanvas(state.goal.x, state.goal.y);

  ctx.strokeStyle = "#2563eb";
  ctx.lineWidth = 3;
  ctx.setLineDash([8, 8]);
  ctx.beginPath();
  ctx.moveTo(start.x, start.y);
  ctx.lineTo(goal.x, goal.y);
  ctx.stroke();
  ctx.setLineDash([]);

  ctx.fillStyle = "#facc15";
  ctx.strokeStyle = state.selected.kind === "start" ? "#111827" : "#854d0e";
  ctx.lineWidth = state.selected.kind === "start" ? 4 : 2;
  ctx.beginPath();
  ctx.arc(start.x, start.y, scaleMeters(state.safety.robotRadius), 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();
  const nose = worldToCanvas(state.start.x + Math.cos(state.start.yaw) * 0.34, state.start.y + Math.sin(state.start.yaw) * 0.34);
  ctx.strokeStyle = "#111827";
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.moveTo(start.x, start.y);
  ctx.lineTo(nose.x, nose.y);
  ctx.stroke();

  ctx.fillStyle = "#16a34a";
  ctx.strokeStyle = state.selected.kind === "goal" ? "#111827" : "#14532d";
  ctx.lineWidth = state.selected.kind === "goal" ? 4 : 2;
  ctx.beginPath();
  ctx.arc(goal.x, goal.y, scaleMeters(state.safety.goalRadius), 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();
  ctx.fillStyle = "#ffffff";
  ctx.beginPath();
  ctx.arc(goal.x, goal.y, scaleMeters(0.07), 0, Math.PI * 2);
  ctx.fill();
}

function draw() {
  const analysis = analyzeMap();
  state.analysis = analysis;
  drawGrid();
  drawSafetyBuffers();
  state.obstacles.forEach(drawObstacle);
  drawNearestGap(analysis);
  if (state.wallDraft) {
    const start = worldToCanvas(state.wallDraft.start.x, state.wallDraft.start.y);
    const end = worldToCanvas(state.wallDraft.end.x, state.wallDraft.end.y);
    ctx.strokeStyle = "#111827";
    ctx.lineWidth = Math.max(3, scaleMeters(0.16));
    ctx.lineCap = "round";
    ctx.beginPath();
    ctx.moveTo(start.x, start.y);
    ctx.lineTo(end.x, end.y);
    ctx.stroke();
    ctx.lineCap = "butt";
  }
  drawStartGoal();
  updateMetrics(analysis);
}

function updateMetrics(analysis = analyzeMap()) {
  els.obstacleCount.textContent = String(state.obstacles.length);
  els.goalDistance.textContent = formatMeters(analysis.startGoalDistance);
  if (analysis.nearestPair) {
    const nearestLevel = analysis.nearestPair.gap < 0 ? "bad" : analysis.nearestPair.gap < analysis.requiredCorridor ? "warn" : "ok";
    setMetric(els.nearestGap, analysis.nearestPair.gap, nearestLevel);
    const corridorLevel = analysis.robotCorridor < 0 ? "bad" : analysis.robotCorridor < state.safety.approachMargin * 2 ? "warn" : "ok";
    setMetric(els.robotCorridor, analysis.robotCorridor, corridorLevel);
  } else {
    setMetric(els.nearestGap, Infinity, "ok", "--");
    setMetric(els.robotCorridor, Infinity, "ok", "--");
  }
  if (analysis.startClearance) {
    const level = analysis.startClearance.clearance < 0 ? "bad" : analysis.startClearance.clearance < state.safety.approachMargin ? "warn" : "ok";
    setMetric(els.startClearance, analysis.startClearance.clearance, level);
  } else {
    setMetric(els.startClearance, Infinity, "ok", "--");
  }
  if (analysis.goalClearance) {
    const level = analysis.goalClearance.clearance < 0 ? "bad" : analysis.goalClearance.clearance < state.safety.approachMargin ? "warn" : "ok";
    setMetric(els.goalClearance, analysis.goalClearance.clearance, level);
  } else {
    setMetric(els.goalClearance, Infinity, "ok", "--");
  }
  if (analysis.directRoute) {
    const level = analysis.directRoute.clearance < 0 ? "bad" : analysis.directRoute.clearance < state.safety.approachMargin ? "warn" : "ok";
    setMetric(els.directRouteClearance, analysis.directRoute.clearance, level);
  } else {
    setMetric(els.directRouteClearance, Infinity, "ok", "--");
  }
  if (analysis.selectedPair) {
    const other = analysis.selectedPair.a.id === state.selected.id ? analysis.selectedPair.b : analysis.selectedPair.a;
    const level = analysis.selectedPair.gap < 0 ? "bad" : analysis.selectedPair.gap < analysis.requiredCorridor ? "warn" : "ok";
    setMetric(els.selectedGap, analysis.selectedPair.gap, level, `${analysis.selectedPair.gap.toFixed(2)} m to ${other.id}`);
  } else {
    setMetric(els.selectedGap, Infinity, "ok", "--");
  }
  els.filePath.textContent = state.filePath;
  updateWarnings(analysis);
}

function updateWarnings(analysis) {
  els.safetyWarnings.innerHTML = "";
  if (!analysis.warnings.length) {
    const item = document.createElement("li");
    item.className = "ok";
    item.textContent = "No safety warnings.";
    els.safetyWarnings.append(item);
    return;
  }
  analysis.warnings.forEach((warning) => {
    const item = document.createElement("li");
    item.className = warning.level === "invalid" ? "invalid" : "";
    item.textContent = warning.text;
    els.safetyWarnings.append(item);
  });
}

function updateInspector() {
  const sel = currentSelection();
  els.selectionKind.textContent = state.selected.kind === "none" ? "None" : state.selected.kind;
  const disabled = !sel;
  [els.fieldX, els.fieldY, els.fieldA, els.fieldB, els.fieldC].forEach((input) => {
    input.disabled = disabled;
  });
  if (!sel) return;
  els.fieldX.value = Number(sel.x).toFixed(2);
  els.fieldY.value = Number(sel.y).toFixed(2);
  if (state.selected.kind === "start") {
    els.fieldALabel.textContent = "Radius";
    els.fieldBLabel.textContent = "Yaw";
    els.fieldA.value = Number(state.safety.robotRadius).toFixed(2);
    els.fieldA.disabled = true;
    els.fieldB.value = Number(state.start.yaw).toFixed(2);
    els.fieldCLabel.textContent = "Angle";
    els.fieldC.value = "0.00";
    els.fieldC.disabled = true;
  } else if (state.selected.kind === "goal") {
    els.fieldALabel.textContent = "Radius";
    els.fieldBLabel.textContent = "Yaw";
    els.fieldA.value = Number(state.safety.goalRadius).toFixed(2);
    els.fieldB.value = "0.00";
    els.fieldA.disabled = true;
    els.fieldB.disabled = true;
    els.fieldCLabel.textContent = "Angle";
    els.fieldC.value = "0.00";
    els.fieldC.disabled = true;
  } else if (sel.shape === "box") {
    els.fieldALabel.textContent = "Half X";
    els.fieldBLabel.textContent = "Half Y";
    els.fieldCLabel.textContent = "Yaw";
    els.fieldA.disabled = false;
    els.fieldB.disabled = false;
    els.fieldC.disabled = false;
    els.fieldA.value = Number(sel.half_x).toFixed(2);
    els.fieldB.value = Number(sel.half_y).toFixed(2);
    els.fieldC.value = Number(sel.yaw || 0).toFixed(2);
  } else {
    els.fieldALabel.textContent = "Radius";
    els.fieldBLabel.textContent = "Yaw";
    els.fieldA.disabled = false;
    els.fieldB.disabled = true;
    els.fieldCLabel.textContent = "Angle";
    els.fieldC.disabled = true;
    els.fieldA.value = Number(sel.radius).toFixed(2);
    els.fieldB.value = "0.00";
    els.fieldC.value = "0.00";
  }
}

function applyInspector() {
  const sel = currentSelection();
  if (!sel) return;
  sel.x = clamp(Number(els.fieldX.value || 0), -state.arenaHalf, state.arenaHalf);
  sel.y = clamp(Number(els.fieldY.value || 0), -state.arenaHalf, state.arenaHalf);
  if (state.selected.kind === "start") {
    state.start.yaw = Number(els.fieldB.value || 0);
  } else if (state.selected.kind === "obstacle" && sel.shape === "box") {
    sel.half_x = clamp(Number(els.fieldA.value || 0.05), 0.05, state.arenaHalf);
    sel.half_y = clamp(Number(els.fieldB.value || 0.05), 0.05, state.arenaHalf);
    sel.yaw = Number(els.fieldC.value || 0);
    sel.radius = Math.max(sel.half_x, sel.half_y);
  } else if (state.selected.kind === "obstacle") {
    sel.radius = clamp(Number(els.fieldA.value || 0.05), 0.05, state.arenaHalf);
    sel.half_x = sel.radius;
    sel.half_y = sel.radius;
  }
  draw();
}

function validateMap() {
  const analysis = analyzeMap();
  updateMetrics(analysis);
  const invalid = analysis.warnings.find((warning) => warning.level === "invalid");
  if (invalid) {
    setStatus(`Invalid: ${invalid.text}`);
    return false;
  }
  const risky = analysis.warnings.find((warning) => warning.level === "warn");
  setStatus(risky ? `Risk: ${risky.text}` : "Map valid");
  return true;
}

function obstacleDistance(obs, point) {
  if (obs.shape === "box") {
    const local = localBoxPoint(obs, point);
    const dx = Math.abs(local.x) - obs.half_x;
    const dy = Math.abs(local.y) - obs.half_y;
    return Math.hypot(Math.max(dx, 0), Math.max(dy, 0)) + Math.min(Math.max(dx, dy), 0);
  }
  return Math.hypot(point.x - obs.x, point.y - obs.y) - obs.radius;
}

function payload() {
  syncSafetyFromInputs();
  return {
    name: els.mapName.value,
    base_task: els.baseTask.value,
    arena_half: Number(els.arenaHalf.value),
    start: state.start,
    goal: state.goal,
    obstacles: state.obstacles,
    jitter: {
      enabled: els.jitterEnabled.checked,
      start_std: Number(els.jitterStart.value || 0),
      goal_std: Number(els.jitterGoal.value || 0),
      yaw_std: Number(els.jitterYaw.value || 0),
    },
    safety: {
      robot_radius: state.safety.robotRadius,
      goal_radius: state.safety.goalRadius,
      obstacle_min_gap: state.safety.minObstacleGap,
    },
  };
}

async function saveMap() {
  state.mapName = els.mapName.value;
  state.arenaHalf = Number(els.arenaHalf.value || 3.5);
  if (!validateMap()) return;
  const response = await fetch("/api/maps", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload()),
  });
  const data = await response.json();
  if (!response.ok) {
    setStatus(data.error || "Save failed");
    return;
  }
  state.filePath = data.path;
  setStatus(`Saved ${data.name}`);
  await refreshMaps();
  draw();
}

function loadConfig(config, name, path) {
  const map = config.map || {};
  applySafetyDefaults(config);
  state.mapName = name || map.name || config.name || "custom_map";
  state.baseTask = map.base_task || state.baseTask;
  state.arenaHalf = Number(config.arena?.half_size || state.arenaHalf);
  state.start = { x: Number(map.start?.[0] || -1), y: Number(map.start?.[1] || -1), yaw: Number(map.start?.[2] || 0) };
  state.goal = { x: Number(map.goal?.[0] || 1), y: Number(map.goal?.[1] || 1) };
  state.obstacles = (map.obstacles || []).map((item, idx) => ({ yaw: 0.0, id: item.id || `${item.shape}_${idx}`, ...item }));
  state.filePath = path || "not saved";
  const jitter = map.jitter || {};
  els.jitterEnabled.checked = Boolean(jitter.enabled);
  els.jitterStart.value = Number(jitter.start_std || 0);
  els.jitterGoal.value = Number(jitter.goal_std || 0);
  els.jitterYaw.value = Number(jitter.yaw_std || 0);
  els.mapName.value = state.mapName;
  els.arenaHalf.value = state.arenaHalf;
  els.baseTask.value = state.baseTask;
  state.selected = { kind: "start", id: null };
  updateInspector();
  draw();
}

async function refreshMaps() {
  const response = await fetch("/api/maps");
  const data = await response.json();
  els.mapList.innerHTML = "";
  data.maps.forEach((name) => {
    const option = document.createElement("option");
    option.value = name;
    option.textContent = name;
    els.mapList.append(option);
  });
}

async function loadSelectedMap() {
  const name = els.mapList.value;
  if (!name) return;
  const response = await fetch(`/api/maps/${encodeURIComponent(name)}`);
  const data = await response.json();
  if (!response.ok) {
    setStatus(data.error || "Load failed");
    return;
  }
  loadConfig(data.config, data.name, data.path);
  setStatus(`Loaded ${data.name}`);
}

async function deleteSelectedMap() {
  const name = els.mapList.value;
  if (!name) return;
  const response = await fetch(`/api/maps/${encodeURIComponent(name)}`, { method: "DELETE" });
  const data = await response.json();
  setStatus(data.ok ? `Removed ${name}` : data.error || "Remove failed");
  await refreshMaps();
}

async function loadBaseTasks() {
  const response = await fetch("/api/base-tasks");
  const data = await response.json();
  els.baseTask.innerHTML = "";
  data.tasks.forEach((name) => {
    const option = document.createElement("option");
    option.value = name;
    option.textContent = name;
    if (name === state.baseTask) option.selected = true;
    els.baseTask.append(option);
  });
  if (els.baseTask.value) {
    await loadBaseTaskDefaults(els.baseTask.value);
  }
}

async function loadBaseTaskDefaults(name) {
  try {
    const response = await fetch(`/api/base-tasks/${encodeURIComponent(name)}`);
    if (!response.ok) return;
    const data = await response.json();
    applySafetyDefaults(data.config || {});
  } catch (_error) {
    // Base task defaults are advisory; the editor remains usable without them.
  }
}

canvas.addEventListener("pointerdown", (event) => {
  const rect = canvas.getBoundingClientRect();
  const world = canvasToWorld((event.clientX - rect.left) * (canvas.width / rect.width), (event.clientY - rect.top) * (canvas.height / rect.height));
  if (state.tool === "start") {
    state.start.x = world.x;
    state.start.y = world.y;
    state.selected = { kind: "start", id: null };
  } else if (state.tool === "goal") {
    state.goal.x = world.x;
    state.goal.y = world.y;
    state.selected = { kind: "goal", id: null };
  } else if (state.tool === "cylinder" || state.tool === "box") {
    addObstacle(state.tool, world);
  } else if (state.tool === "wall") {
    state.wallDraft = { start: world, end: world };
    state.dragging = true;
    state.selected = { kind: "none", id: null };
  } else {
    state.selected = hitTest(world);
    state.dragging = state.selected.kind !== "none";
  }
  updateInspector();
  draw();
});

canvas.addEventListener("pointermove", (event) => {
  const rect = canvas.getBoundingClientRect();
  const world = canvasToWorld((event.clientX - rect.left) * (canvas.width / rect.width), (event.clientY - rect.top) * (canvas.height / rect.height));
  els.cursor.textContent = `x ${world.x.toFixed(2)}, y ${world.y.toFixed(2)}`;
  if (state.wallDraft) {
    state.wallDraft.end = world;
    draw();
    return;
  }
  if (!state.dragging) return;
  const sel = currentSelection();
  if (!sel) return;
  sel.x = world.x;
  sel.y = world.y;
  updateInspector();
  draw();
});

canvas.addEventListener("pointerup", () => {
  if (state.wallDraft) {
    finalizeWall(state.wallDraft.start, state.wallDraft.end);
    state.wallDraft = null;
    state.dragging = false;
    updateInspector();
    draw();
    return;
  }
  state.dragging = false;
});

document.querySelectorAll(".tool[data-tool]").forEach((button) => {
  button.addEventListener("click", () => {
    state.tool = button.dataset.tool;
    document.querySelectorAll(".tool[data-tool]").forEach((item) => item.classList.toggle("active", item === button));
  });
});

document.getElementById("deleteSelected").addEventListener("click", () => {
  const obs = selectedObstacle();
  if (obs) {
    state.obstacles = state.obstacles.filter((item) => item.id !== obs.id);
    state.selected = { kind: "none", id: null };
    updateInspector();
    draw();
  }
});
document.getElementById("clearObstacles").addEventListener("click", clearObstacles);
document.getElementById("newMap").addEventListener("click", newBlankMap);

[els.fieldX, els.fieldY, els.fieldA, els.fieldB, els.fieldC].forEach((input) => input.addEventListener("input", applyInspector));
els.arenaHalf.addEventListener("input", () => {
  state.arenaHalf = Number(els.arenaHalf.value || 3.5);
  draw();
});
els.baseTask.addEventListener("change", async () => {
  state.baseTask = els.baseTask.value || state.baseTask;
  await loadBaseTaskDefaults(state.baseTask);
  draw();
});
[els.robotRadius, els.approachMargin, els.minObstacleGap, els.showSafetyBuffers].forEach((input) => {
  input.addEventListener("input", draw);
  input.addEventListener("change", draw);
});
document.getElementById("validateMap").addEventListener("click", validateMap);
document.getElementById("saveMap").addEventListener("click", saveMap);
document.getElementById("refreshMaps").addEventListener("click", refreshMaps);
document.getElementById("loadMap").addEventListener("click", loadSelectedMap);
document.getElementById("deleteMap").addEventListener("click", deleteSelectedMap);

loadBaseTasks();
refreshMaps();
updateInspector();
draw();
