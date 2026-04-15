"use client";
import { useEffect } from "react";

export default function Callback() {
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const code = params.get("code");
    if (code) {
      localStorage.setItem("spotify_auth_code", code);
    }
    // Redirect back to same origin
    window.location.replace(window.location.origin);
  }, []);

  return (
    <div style={{ background: "#000", color: "#fff", height: "100vh", display: "flex", alignItems: "center", justifyContent: "center" }}>
      <p>Connecting to Spotify...</p>
    </div>
  );
}