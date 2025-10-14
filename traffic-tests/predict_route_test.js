import http from "k6/http";
import { check, sleep } from "k6";

export const options = {
  stages: [
    { duration: "30s", target: 1 }, // warmup
    { duration: "60s", target: 50 }, // ramp
    { duration: "60s", target: 100 }, // peak
    { duration: "30s", target: 0 }, // ramp down
  ],
  thresholds: {
    http_req_failed: ["rate<0.01"],
    // set loose first, tighten after tuning:
    http_req_duration: ["p(95)<5000"],
  },
};

const URL = "http://44.196.227.203:8080/predict_route";
const BODY = JSON.stringify({
  origin: [1.3521, 103.8198],
  destination: [1.2903, 103.8519],
});

export default function () {
  const res = http.post(URL, BODY, {
    headers: { "Content-Type": "application/json" },
  });
  check(res, {
    "status 200": (r) => r.status === 200,
    "non-empty": (r) => (r.body || "").length > 2,
  });
  sleep(1);
}
