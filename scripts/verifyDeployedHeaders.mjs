const url = process.env.DEPLOY_URL ?? "https://3d-life-sim.pages.dev/";
const required = [
  "content-security-policy",
  "x-frame-options",
  "referrer-policy",
  "x-content-type-options",
  "strict-transport-security",
  "permissions-policy"
];

let missing = required;
for (let attempt = 1; attempt <= 6; attempt++) {
  const response = await fetch(url, { headers: { "cache-control": "no-cache" } });
  missing = required.filter((name) => !response.headers.has(name));
  if (missing.length === 0) {
    console.log(`Security headers verified at ${url}`);
    process.exit(0);
  }
  if (attempt < 6) await new Promise((resolve) => setTimeout(resolve, 5000));
}

console.error(`Deployed site at ${url} is missing security headers: ${missing.join(", ")}`);
process.exit(1);
