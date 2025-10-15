import http from "k6/http";
import { check, sleep } from "k6";

// Prefer env var: k6 run -e ALB=http://your-alb k6.js
const ALB =
  __ENV.ALB || "http://traffic-ai-alb-484326813.us-east-1.elb.amazonaws.com";

// payloads
const routeBody = JSON.stringify({
  origin: [1.3521, 103.8198],
  destination: [1.2903, 103.8519],
  filters: {},
});

const predictBody = JSON.stringify({
  origin: [1.3521, 103.8198],
  destination: [1.2903, 103.8519],
});

const json = { "Content-Type": "application/json" };

export const options = {
  thresholds: {
    http_req_failed: ["rate<0.01"], // <1% errors
    "http_req_duration{ep:healthz}": ["p(95)<200"], // infra quick
    "http_req_duration{ep:route}": ["p(95)<800"], // Google route
    "http_req_duration{ep:predict}": ["p(95)<1500"], // SageMaker adds latency
  },
  // For your ramp scenario, uncomment this and remove CLI stages:
  // stages: [
  //   { duration: "30s", target: 1 },    // warmup
  //   { duration: "60s", target: 50 },   // ramp
  //   { duration: "60s", target: 100 },  // peak
  //   { duration: "30s", target: 0 },    // ramp down
  // ],
};

export default function () {
  // 1) Healthz
  const r1 = http.get(`${ALB}/healthz`, { tags: { ep: "healthz" } });
  check(r1, { "healthz 200": (res) => res.status === 200 });

  // 2) Route (Google Directions)
  const r2 = http.post(`${ALB}/route`, routeBody, {
    headers: json,
    tags: { ep: "route" },
  });
  check(r2, { "route ok": (res) => res.status === 200 });

  // 3) Predict (SageMaker)
  const r3 = http.post(`${ALB}/predict_route`, predictBody, {
    headers: json,
    tags: { ep: "predict" },
  });
  check(r3, { "predict ok": (res) => res.status === 200 });

  sleep(1);
}
