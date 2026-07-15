import React from "react";
import { createRoot } from "react-dom/client";
import { App } from "./App";
import { SimSpeedDebugLab } from "./SimSpeedDebugLab";
import "./styles.css";

const searchParams = new URLSearchParams(window.location.search);
const Root = searchParams.get("simSpeedLab") === "1" ? SimSpeedDebugLab : App;

createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <Root />
  </React.StrictMode>
);
