import http from "k6/http";
import { check, sleep } from "k6";

export const options = {
  // gentle ramp so we can observe behavior
  stages: [
    { duration: "10s", target: 10 },
    { duration: "20s", target: 20 },
    { duration: "10s", target: 0 },
  ],
  thresholds: {
    http_req_failed: ["rate<0.05"], // allow some failures during debugging
    http_req_duration: ["p(95)<2000"], // 95% under 2s
  },
};

const payload = JSON.stringify({
  origin: [1.3531, 103.9457],
  destination: [1.2834, 103.8607],
  mode: "bus",
});

const params = { headers: { "Content-Type": "application/json" } };

export default function () {
  const host = __ENV.HOST; // pass HOST=... on the CLI
  const url = `http://${host}:8080/predict_route`;

  const res = http.post(url, payload, params);

  check(res, {
    "status is 200": (r) => r.status === 200,
    "body not empty": (r) => (r.body || "").length > 0,
  });

  sleep(0.5); // tiny pacing
}
