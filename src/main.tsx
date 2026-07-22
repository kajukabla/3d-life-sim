import React from "react";
import { createRoot } from "react-dom/client";
import { App } from "./App";
import { SimSpeedDebugLab } from "./SimSpeedDebugLab";
import { isCoarsePointerClient } from "./mobileLaunch";
import "./styles.css";

function DesktopOnly() {
  return (
    <div className="desktop-only" data-testid="desktop-only">
      <strong>Desktop only</strong>
      <span>3D Life Sim needs a desktop browser with WebGPU. Come back on a computer.</span>
    </div>
  );
}

const searchParams = new URLSearchParams(window.location.search);
const Root = import.meta.env.DEV && searchParams.get("simSpeedLab") === "1"
  ? SimSpeedDebugLab
  : isCoarsePointerClient()
    ? DesktopOnly
    : App;

createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <Root />
  </React.StrictMode>
);
