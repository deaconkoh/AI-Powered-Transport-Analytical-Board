import http from "k6/http";
import { check, sleep } from "k6";

export const options = {
  stages: [
    { duration: "30s", target: 25 },
    { duration: "60s", target: 50 },
    { duration: "60s", target: 100 },
    { duration: "30s", target: 0 },
  ],
  thresholds: {
    http_req_failed: ["rate<0.02"],
    http_req_duration: ["p(95)<1500"],
  },
};

const HOST = __ENV.HOST || "localhost";
const BASE = `http://${HOST}:8080`;
const headers = { "Content-Type": "application/json" };
const payloads = JSON.parse(open("./payloads.json"));

export default function () {
  const body = JSON.stringify(
    payloads[Math.floor(Math.random() * payloads.length)]
  );
  const res = http.post(`${BASE}/predict_route`, body, { headers });

  check(res, {
    "status is 200": (r) => r.status === 200,
    "body not empty": (r) => r && r.body && r.body.length > 0,
  });

  sleep(1);
}
