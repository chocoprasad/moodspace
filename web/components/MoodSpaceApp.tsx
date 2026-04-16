"use client";

import { useEffect, useRef, useState, useId, useCallback } from "react";
import * as THREE from "three";
import {
  Home, Video, Camera, Share2, Heart, Sparkles, User, Sun, Settings,
  Bell, Star, Gift, Download, Mail, HelpCircle, LogOut, ExternalLink,
  Brain, ArrowRight, Eye, Music, Wifi, Thermometer, Activity, Cpu, Zap,
  Moon, Monitor, Lock, ChevronRight, X, Check, Volume2, SkipForward,
  SkipBack, Pause, Play, Repeat, Shuffle, Radio, AlertCircle, Droplets,
  Apple, Clock, Pill, Wind, Coffee,
} from "lucide-react";

/* ══════════════════════════════════════════════════════
   API CONFIG
   ══════════════════════════════════════════════════════ */
const API_URL = process.env.NODE_ENV === "production" 
  ? "https://moodspace-api.onrender.com"
  : "http://localhost:8000";

/* ══════════════════════════════════════════════════════
   SPOTIFY CONFIG — fill in your credentials
   ══════════════════════════════════════════════════════ */
const SPOTIFY_CLIENT_ID = "c9f01c55ea814376b599edf076ffb29e"; // Your Spotify Client ID
const SPOTIFY_REDIRECT_URI = process.env.NODE_ENV === "production"
  ? "https://moodspace-api.onrender.com/spotify/callback"
  : "http://127.0.0.1:8000/spotify/callback";
const SPOTIFY_SCOPES = [
  "streaming",
  "user-read-playback-state",
  "user-modify-playback-state",
  "user-read-currently-playing",
  "playlist-read-private",
  "user-library-read",
].join(" ");

/* ══════════════════════════════════════════════════════
   AUTH HELPERS
   ══════════════════════════════════════════════════════ */
function getToken(): string | null {
  if (typeof window === "undefined") return null;
  return localStorage.getItem("moodspace_token");
}
function setToken(token: string) {
  localStorage.setItem("moodspace_token", token);
}
function clearToken() {
  localStorage.removeItem("moodspace_token");
  localStorage.removeItem("moodspace_user");
}
function getSavedUser(): UserAccount | null {
  if (typeof window === "undefined") return null;
  const s = localStorage.getItem("moodspace_user");
  if (s) try { return JSON.parse(s); } catch { return null; }
  return null;
}
function saveUser(user: UserAccount) {
  localStorage.setItem("moodspace_user", JSON.stringify(user));
}

async function authFetch(path: string, options: RequestInit = {}) {
  const token = getToken();
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    ...(options.headers as Record<string, string> || {}),
  };
  if (token) headers["Authorization"] = `Bearer ${token}`;
  const res = await fetch(`${API_URL}${path}`, { ...options, headers });
  if (res.status === 401) {
    clearToken();
    throw new Error("Session expired. Please log in again.");
  }
  return res;
}

/* ══════════════════════════════════════════════════════
   SPOTIFY HELPERS — PKCE Auth Flow
   ══════════════════════════════════════════════════════ */
function getSpotifyToken(): string | null {
  if (typeof window === "undefined") return null;
  const data = localStorage.getItem("spotify_token_data");
  if (!data) return null;
  try {
    const parsed = JSON.parse(data);
    if (parsed.expires_at && Date.now() > parsed.expires_at) {
      localStorage.removeItem("spotify_token_data");
      return null;
    }
    return parsed.access_token;
  } catch { return null; }
}

function saveSpotifyToken(accessToken: string, expiresIn: number) {
  localStorage.setItem("spotify_token_data", JSON.stringify({
    access_token: accessToken,
    expires_at: Date.now() + expiresIn * 1000,
  }));
}

function clearSpotifyToken() {
  localStorage.removeItem("spotify_token_data");
  localStorage.removeItem("spotify_code_verifier");
}

// PKCE helpers
function generateRandomString(length: number): string {
  const possible = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
  const values = crypto.getRandomValues(new Uint8Array(length));
  return values.reduce((acc, x) => acc + possible[x % possible.length], "");
}

async function sha256(plain: string): Promise<ArrayBuffer> {
  const encoder = new TextEncoder();
  const data = encoder.encode(plain);
  return window.crypto.subtle.digest("SHA-256", data);
}

function base64encode(input: ArrayBuffer): string {
  return btoa(String.fromCharCode(...new Uint8Array(input)))
    .replace(/=/g, "").replace(/\+/g, "-").replace(/\//g, "_");
}

async function startSpotifyAuth() {
  const codeVerifier = generateRandomString(64);
  localStorage.setItem("spotify_code_verifier", codeVerifier);
  const hashed = await sha256(codeVerifier);
  const codeChallenge = base64encode(hashed);

  const params = new URLSearchParams({
    client_id: SPOTIFY_CLIENT_ID,
    response_type: "code",
    redirect_uri: SPOTIFY_REDIRECT_URI,
    scope: SPOTIFY_SCOPES,
    code_challenge_method: "S256",
    code_challenge: codeChallenge,
  });
  window.location.href = `https://accounts.spotify.com/authorize?${params}`;
}

async function exchangeSpotifyCode(code: string): Promise<boolean> {
  const codeVerifier = localStorage.getItem("spotify_code_verifier");
  if (!codeVerifier) return false;

  try {
    const res = await fetch("https://accounts.spotify.com/api/token", {
      method: "POST",
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
      body: new URLSearchParams({
        client_id: SPOTIFY_CLIENT_ID,
        grant_type: "authorization_code",
        code,
        redirect_uri: SPOTIFY_REDIRECT_URI,
        code_verifier: codeVerifier,
      }),
    });
    const data = await res.json();
    if (data.access_token) {
      saveSpotifyToken(data.access_token, data.expires_in || 3600);
      localStorage.removeItem("spotify_code_verifier");
      return true;
    }
  } catch (e) {
    console.error("Spotify token exchange failed:", e);
  }
  return false;
}

async function spotifyFetch(endpoint: string, options: RequestInit = {}) {
  const token = getSpotifyToken();
  if (!token) throw new Error("Not connected to Spotify");
  const res = await fetch(`https://api.spotify.com/v1${endpoint}`, {
    ...options,
    headers: {
      Authorization: `Bearer ${token}`,
      "Content-Type": "application/json",
      ...(options.headers as Record<string, string> || {}),
    },
  });
  if (res.status === 401) {
    clearSpotifyToken();
    throw new Error("Spotify session expired");
  }
  return res;
}

/* ══════════════════════════════════════════════════════
   SHADER ANIMATION BACKGROUND
   ══════════════════════════════════════════════════════ */
function ShaderAnimation() {
  const containerRef = useRef<HTMLDivElement>(null);
  const sceneRef = useRef<{
    camera: THREE.Camera;
    scene: THREE.Scene;
    renderer: THREE.WebGLRenderer;
    uniforms: { time: { value: number }; resolution: { value: THREE.Vector2 } };
    animationId: number;
  } | null>(null);

  useEffect(() => {
    if (!containerRef.current) return;
    const container = containerRef.current;

    const vertexShader = `void main() { gl_Position = vec4(position, 1.0); }`;
    const fragmentShader = `
      #define TWO_PI 6.2831853072
      #define PI 3.14159265359
      precision highp float;
      uniform vec2 resolution;
      uniform float time;
      void main(void) {
        vec2 uv = (gl_FragCoord.xy * 2.0 - resolution.xy) / min(resolution.x, resolution.y);
        float t = time * 0.05;
        float lineWidth = 0.002;
        vec3 color = vec3(0.0);
        for(int j = 0; j < 3; j++){
          for(int i = 0; i < 5; i++){
            color[j] += lineWidth * float(i*i) / abs(fract(t - 0.01*float(j) + float(i)*0.01)*5.0 - length(uv) + mod(uv.x+uv.y, 0.2));
          }
        }
        gl_FragColor = vec4(color[0], color[1], color[2], 1.0);
      }
    `;

    const camera = new THREE.Camera();
    camera.position.z = 1;
    const scene = new THREE.Scene();
    const geometry = new THREE.PlaneGeometry(2, 2);
    const uniforms = {
      time: { value: 1.0 },
      resolution: { value: new THREE.Vector2() },
    };
    const material = new THREE.ShaderMaterial({ uniforms, vertexShader, fragmentShader });
    scene.add(new THREE.Mesh(geometry, material));

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(window.devicePixelRatio);
    container.appendChild(renderer.domElement);

    const onResize = () => {
      renderer.setSize(container.clientWidth, container.clientHeight);
      uniforms.resolution.value.x = renderer.domElement.width;
      uniforms.resolution.value.y = renderer.domElement.height;
    };
    onResize();
    window.addEventListener("resize", onResize, false);

    const animate = () => {
      const id = requestAnimationFrame(animate);
      uniforms.time.value += 0.05;
      renderer.render(scene, camera);
      if (sceneRef.current) sceneRef.current.animationId = id;
    };
    sceneRef.current = { camera, scene, renderer, uniforms, animationId: 0 };
    animate();

    return () => {
      window.removeEventListener("resize", onResize);
      if (sceneRef.current) {
        cancelAnimationFrame(sceneRef.current.animationId);
        if (container && renderer.domElement.parentNode === container)
          container.removeChild(renderer.domElement);
        renderer.dispose();
        geometry.dispose();
        material.dispose();
      }
    };
  }, []);

  return (
    <div
      ref={containerRef}
      className="absolute inset-0 w-full h-full"
      style={{ opacity: 0.45, mixBlendMode: "screen", pointerEvents: "none" }}
    />
  );
}

/* ══════════════════════════════════════════════════════
   BOOT SHADER
   ══════════════════════════════════════════════════════ */
function BootShader() {
  const containerRef = useRef<HTMLDivElement>(null);
  const sceneRef = useRef<{
    camera: any; scene: any; renderer: any; uniforms: any; animationId: number | null;
  }>({ camera: null, scene: null, renderer: null, uniforms: null, animationId: null });

  useEffect(() => {
    const script = document.createElement("script");
    script.src = "https://cdnjs.cloudflare.com/ajax/libs/three.js/89/three.min.js";
    script.onload = () => { if (containerRef.current && (window as any).THREE) initThreeJS(); };
    document.head.appendChild(script);
    return () => {
      if (sceneRef.current.animationId) cancelAnimationFrame(sceneRef.current.animationId);
      if (sceneRef.current.renderer) sceneRef.current.renderer.dispose();
      if (document.head.contains(script)) document.head.removeChild(script);
    };
  }, []);

  const initThreeJS = () => {
    if (!containerRef.current || !(window as any).THREE) return;
    const THREE = (window as any).THREE;
    const container = containerRef.current;
    container.innerHTML = "";
    const camera = new THREE.Camera();
    camera.position.z = 1;
    const scene = new THREE.Scene();
    const geometry = new THREE.PlaneBufferGeometry(2, 2);
    const uniforms = {
      time: { type: "f", value: 1.0 },
      resolution: { type: "v2", value: new THREE.Vector2() },
    };
    const material = new THREE.ShaderMaterial({
      uniforms,
      vertexShader: `void main() { gl_Position = vec4(position, 1.0); }`,
      fragmentShader: `
        precision highp float;
        uniform vec2 resolution;
        uniform float time;
        float random(in float x) { return fract(sin(x)*1e4); }
        void main(void) {
          vec2 uv = (gl_FragCoord.xy * 2.0 - resolution.xy) / min(resolution.x, resolution.y);
          vec2 fMosaicScal = vec2(4.0, 2.0);
          vec2 vScreenSize = vec2(256.0, 256.0);
          uv.x = floor(uv.x * vScreenSize.x / fMosaicScal.x) / (vScreenSize.x / fMosaicScal.x);
          uv.y = floor(uv.y * vScreenSize.y / fMosaicScal.y) / (vScreenSize.y / fMosaicScal.y);
          float t = time*0.06 + random(uv.x)*0.4;
          float lineWidth = 0.0008;
          vec3 color = vec3(0.0);
          for(int j = 0; j < 3; j++){
            for(int i = 0; i < 5; i++){
              color[j] += lineWidth*float(i*i) / abs(fract(t - 0.01*float(j)+float(i)*0.01)*1.0 - length(uv));
            }
          }
          gl_FragColor = vec4(color[2],color[1],color[0],1.0);
        }
      `,
    });
    scene.add(new THREE.Mesh(geometry, material));
    const renderer = new THREE.WebGLRenderer();
    renderer.setPixelRatio(window.devicePixelRatio);
    container.appendChild(renderer.domElement);
    sceneRef.current = { camera, scene, renderer, uniforms, animationId: null };
    const onResize = () => {
      const rect = container.getBoundingClientRect();
      renderer.setSize(rect.width, rect.height);
      uniforms.resolution.value.x = renderer.domElement.width;
      uniforms.resolution.value.y = renderer.domElement.height;
    };
    onResize();
    window.addEventListener("resize", onResize, false);
    const animate = () => {
      sceneRef.current.animationId = requestAnimationFrame(animate);
      uniforms.time.value += 0.05;
      renderer.render(scene, camera);
    };
    animate();
  };

  return <div ref={containerRef} className="absolute inset-0 w-full h-full" />;
}

/* ══════════════════════════════════════════════════════
   BOOT SCREEN — no Mitra AI
   ══════════════════════════════════════════════════════ */
function BootScreen({ onContinue }: { onContinue: () => void }) {
  const [phase, setPhase] = useState(0);
  const audioRef = useRef<AudioContext | null>(null);

  useEffect(() => {
    const t1 = setTimeout(() => setPhase(1), 300);
    const t2 = setTimeout(() => setPhase(2), 2600);
    return () => { clearTimeout(t1); clearTimeout(t2); };
  }, []);

  useEffect(() => {
    if (phase !== 1) return;
    try {
      const AudioCtx = window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
      const ctx = new AudioCtx();
      audioRef.current = ctx;
      const tone = (freq: number, start: number, dur: number, vol = 0.1, type: OscillatorType = "sine") => {
        const o = ctx.createOscillator(), g = ctx.createGain();
        o.type = type;
        o.frequency.setValueAtTime(freq, ctx.currentTime + start);
        g.gain.setValueAtTime(0, ctx.currentTime + start);
        g.gain.linearRampToValueAtTime(vol, ctx.currentTime + start + 0.08);
        g.gain.setValueAtTime(vol, ctx.currentTime + start + dur - 0.3);
        g.gain.linearRampToValueAtTime(0, ctx.currentTime + start + dur);
        o.connect(g); g.connect(ctx.destination);
        o.start(ctx.currentTime + start); o.stop(ctx.currentTime + start + dur);
      };
      tone(65, 0, 3.5, 0.05); tone(130, 0, 3.5, 0.03);
      tone(261.63, 0.1, 2.8, 0.09); tone(329.63, 0.3, 2.6, 0.08);
      tone(392, 0.5, 2.4, 0.07); tone(523.25, 0.7, 2.2, 0.06);
      tone(659.25, 0.9, 2.0, 0.05); tone(1046.5, 1.2, 1.5, 0.035);
      tone(130.81, 2.0, 1.5, 0.07);
    } catch (_) {}
    return () => { audioRef.current?.close().catch(() => {}); };
  }, [phase]);

  return (
    <div className="fixed inset-0 z-[100] flex flex-col items-center justify-center" style={{ background: "#020205" }}>
      <BootShader />
      <style>{`
        @keyframes bGlow{0%{opacity:0;filter:blur(24px);transform:scale(0.7)}40%{opacity:1;filter:blur(0);transform:scale(1.02)}100%{opacity:1;filter:blur(0);transform:scale(1)}}
        @keyframes bSub{0%{opacity:0;transform:translateY(12px)}100%{opacity:0.5;transform:translateY(0)}}
        @keyframes bLine{0%{width:0;opacity:0}50%{opacity:1}100%{width:200px;opacity:0.3}}
        @keyframes bCont{0%{opacity:0;transform:translateY(20px)}100%{opacity:1;transform:translateY(0)}}
        @keyframes cPulse{0%,100%{opacity:0.6;box-shadow:0 0 20px rgba(169,85,255,0.2)}50%{opacity:1;box-shadow:0 0 40px rgba(169,85,255,0.4)}}
        @keyframes ambOrb{0%,100%{transform:translate(0,0) scale(1);opacity:0.08}50%{transform:translate(20px,-15px) scale(1.1);opacity:0.15}}
        .b-logo{animation:bGlow 2s cubic-bezier(.16,1,.3,1) .5s both}
        .b-sub{animation:bSub 1s ease-out 1.5s both}
        .b-line{animation:bLine 1.5s ease-out 1.2s both}
        .b-cont{animation:bCont .8s ease-out 2.6s both}
        .c-pulse{animation:cPulse 2s ease-in-out infinite}
      `}</style>
      <div className="absolute w-[400px] h-[400px] rounded-full" style={{ background: "radial-gradient(circle,rgba(169,85,255,0.1) 0%,transparent 70%)", left: "30%", top: "30%", animation: "ambOrb 8s ease-in-out infinite" }} />
      <div className="absolute w-[300px] h-[300px] rounded-full" style={{ background: "radial-gradient(circle,rgba(234,81,255,0.06) 0%,transparent 70%)", right: "25%", bottom: "25%", animation: "ambOrb 10s ease-in-out 2s infinite" }} />
      {phase >= 1 && (
        <div className="relative flex flex-col items-center">
          <div className="b-logo w-20 h-20 rounded-2xl flex items-center justify-center mb-8" style={{ background: "linear-gradient(135deg,#a955ff,#ea51ff)", boxShadow: "0 0 60px rgba(169,85,255,0.5)" }}>
            <Brain size={40} className="text-white" />
          </div>
          <h1 className="b-logo text-6xl sm:text-8xl font-extrabold tracking-tighter" style={{ color: "#fff" }}>
            Mood<span className="bg-clip-text text-transparent" style={{ backgroundImage: "linear-gradient(135deg,#a955ff,#ea51ff,#56CCF2)" }}>Space</span>
          </h1>
          <div className="b-line h-[1px] mt-6" style={{ background: "linear-gradient(90deg,transparent,#a955ff,transparent)" }} />
          <p className="b-sub mt-5 text-sm tracking-[0.4em] uppercase" style={{ color: "rgba(255,255,255,0.5)" }}>Adaptive Mood Intelligence</p>
        </div>
      )}
      {phase >= 2 && (
        <button onClick={onContinue} className="b-cont c-pulse absolute bottom-16 px-8 py-3 rounded-full border border-white/10 bg-white/[0.06] backdrop-blur-sm text-white/80 text-sm tracking-[0.2em] uppercase transition-all duration-300 hover:bg-white/15 hover:scale-105 focus:outline-none cursor-pointer">
          Press to Continue
        </button>
      )}
    </div>
  );
}

/* ══════════════════════════════════════════════════════
   GRADIENT MENU
   ══════════════════════════════════════════════════════ */
const gItems = [
  { title: "Home", icon: Home, from: "#a955ff", to: "#ea51ff" },
  { title: "Video", icon: Video, from: "#56CCF2", to: "#2F80ED" },
  { title: "Photo", icon: Camera, from: "#FF9966", to: "#FF5E62" },
  { title: "Share", icon: Share2, from: "#80FF72", to: "#7EE8FA" },
  { title: "Mood", icon: Heart, from: "#ffa9c6", to: "#f434e2" },
];

function GradMenu() {
  return (
    <ul className="flex gap-3 sm:gap-4 flex-wrap justify-center">
      {gItems.map(({ title, icon: I, from, to }, i) => (
        <li key={i} className="relative w-[50px] h-[50px] sm:w-[58px] sm:h-[58px] bg-white/10 backdrop-blur-md border border-white/15 rounded-full flex items-center justify-center transition-all duration-500 hover:w-[145px] sm:hover:w-[170px] group cursor-pointer">
          <span className="absolute inset-0 rounded-full opacity-0 transition-all duration-500 group-hover:opacity-100" style={{ background: `linear-gradient(135deg,${from},${to})` }} />
          <span className="absolute top-[10px] inset-x-0 h-full rounded-full blur-[15px] opacity-0 -z-10 transition-all duration-500 group-hover:opacity-50" style={{ background: `linear-gradient(135deg,${from},${to})` }} />
          <span className="relative z-10 transition-all duration-500 group-hover:scale-0"><I size={20} className="text-white/60" /></span>
          <span className="absolute text-white uppercase tracking-[0.2em] text-[10px] font-semibold transition-all duration-500 scale-0 group-hover:scale-100 whitespace-nowrap">{title}</span>
        </li>
      ))}
    </ul>
  );
}

/* ══════════════════════════════════════════════════════
   ACCOUNT SYSTEM
   ══════════════════════════════════════════════════════ */
interface UserAccount {
  id?: string;
  name: string;
  username: string;
  email: string;
  status: "online" | "focus" | "offline";
  theme: "dark" | "light" | "auto";
  notifications: boolean;
  musicAutoplay: boolean;
  iotConnected: boolean;
  spotifyConnected?: boolean;
}

const defaultUser: UserAccount = {
  name: "MoodSpace User",
  username: "@moodspace_user",
  email: "user@moodspace.app",
  status: "online",
  theme: "dark",
  notifications: true,
  musicAutoplay: true,
  iotConnected: false,
};

/* ══════════════════════════════════════════════════════
   AUTH SCREEN
   ══════════════════════════════════════════════════════ */
function AuthScreen({ onAuth }: { onAuth: (user: UserAccount) => void }) {
  const [mode, setMode] = useState<"login" | "signup">("login");
  const [name, setName] = useState("");
  const [username, setUsername] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const inputCls = "w-full bg-white/[0.06] border border-white/10 rounded-lg px-4 py-3 text-white text-sm focus:outline-none focus:border-violet-500/50 transition-colors placeholder:text-white/20";

  const handleSubmit = async () => {
    setError("");
    setLoading(true);
    try {
      if (mode === "signup") {
        if (!name || !username || !email || !password) { setError("All fields are required"); setLoading(false); return; }
        if (password.length < 6) { setError("Password must be at least 6 characters"); setLoading(false); return; }
        if (password !== confirmPassword) { setError("Passwords don't match"); setLoading(false); return; }
        const res = await fetch(`${API_URL}/auth/signup`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ name, username, email, password }) });
        const data = await res.json();
        if (!res.ok) { setError(data.detail || "Signup failed"); setLoading(false); return; }
        setToken(data.token); saveUser(data.user); onAuth(data.user);
      } else {
        if (!email || !password) { setError("Email and password are required"); setLoading(false); return; }
        const res = await fetch(`${API_URL}/auth/login`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ email, password }) });
        const data = await res.json();
        if (!res.ok) { setError(data.detail || "Login failed"); setLoading(false); return; }
        setToken(data.token); saveUser(data.user); onAuth(data.user);
      }
    } catch (e: any) { setError("Server not reachable. Make sure the API is running on localhost:8000"); }
    setLoading(false);
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center" style={{ background: "rgba(0,0,0,0.85)", backdropFilter: "blur(20px)" }}>
      <div className="w-[95%] max-w-md rounded-2xl overflow-hidden" style={{ background: "rgba(15,15,22,0.98)", border: "1px solid rgba(255,255,255,0.08)" }}>
        <div className="p-6 text-center border-b border-white/[0.06]">
          <div className="w-14 h-14 rounded-2xl mx-auto mb-4 flex items-center justify-center" style={{ background: "linear-gradient(135deg,#a955ff,#ea51ff)", boxShadow: "0 4px 30px rgba(169,85,255,0.4)" }}>
            <Brain size={28} className="text-white" />
          </div>
          <h2 className="text-white text-xl font-bold">{mode === "login" ? "Welcome back" : "Create account"}</h2>
          <p className="text-white/30 text-xs mt-1">{mode === "login" ? "Sign in to your MoodSpace account" : "Join MoodSpace and start your journey"}</p>
        </div>
        <div className="p-5 space-y-3">
          {error && (<div className="flex items-center gap-2 p-3 rounded-xl bg-red-500/10 border border-red-500/20"><AlertCircle size={14} className="text-red-400 flex-shrink-0" /><p className="text-red-400 text-xs">{error}</p></div>)}
          {mode === "signup" && (<><input value={name} onChange={e => setName(e.target.value)} placeholder="Full name" className={inputCls} /><input value={username} onChange={e => setUsername(e.target.value)} placeholder="Username" className={inputCls} /></>)}
          <input value={email} onChange={e => setEmail(e.target.value)} placeholder="Email" type="email" className={inputCls} />
          <input value={password} onChange={e => setPassword(e.target.value)} placeholder="Password" type="password" className={inputCls} onKeyDown={e => { if (e.key === "Enter" && mode === "login") handleSubmit(); }} />
          {mode === "signup" && (<input value={confirmPassword} onChange={e => setConfirmPassword(e.target.value)} placeholder="Confirm password" type="password" className={inputCls} onKeyDown={e => { if (e.key === "Enter") handleSubmit(); }} />)}
          <button onClick={handleSubmit} disabled={loading} className="w-full py-3 rounded-xl text-sm font-semibold text-white cursor-pointer transition-all hover:scale-[1.02] disabled:opacity-50" style={{ background: "linear-gradient(135deg,#a955ff,#ea51ff)" }}>
            {loading ? "..." : mode === "login" ? "Sign In" : "Create Account"}
          </button>
          <p className="text-center text-white/30 text-xs pt-2">
            {mode === "login" ? "Don't have an account? " : "Already have an account? "}
            <button onClick={() => { setMode(mode === "login" ? "signup" : "login"); setError(""); }} className="text-violet-400 hover:text-violet-300 cursor-pointer font-semibold">
              {mode === "login" ? "Sign up" : "Sign in"}
            </button>
          </p>
        </div>
      </div>
    </div>
  );
}

/* ══════════════════════════════════════════════════════
   ACCOUNT SETTINGS PANEL
   ══════════════════════════════════════════════════════ */
function AccountPanel({ user, setUser, onClose }: { user: UserAccount; setUser: (u: UserAccount) => void; onClose: () => void }) {
  const [tab, setTab] = useState<"profile" | "appearance" | "notifications" | "security">("profile");
  const [editName, setEditName] = useState(user.name);
  const [editEmail, setEditEmail] = useState(user.email);
  const [saved, setSaved] = useState(false);
  const [error, setError] = useState("");
  const [currentPw, setCurrentPw] = useState("");
  const [newPw, setNewPw] = useState("");
  const [confirmPw, setConfirmPw] = useState("");
  const [pwMsg, setPwMsg] = useState("");
  const [deleting, setDeleting] = useState(false);

  const save = async () => {
    setError("");
    try {
      const res = await authFetch("/user/update", { method: "PUT", body: JSON.stringify({ name: editName, email: editEmail }) });
      const data = await res.json();
      if (!res.ok) { setError(data.detail || "Update failed"); return; }
      setUser(data.user); saveUser(data.user); setSaved(true);
      setTimeout(() => setSaved(false), 2000);
    } catch (e: any) { setError(e.message || "Failed to save"); }
  };

  const changePassword = async () => {
    setPwMsg("");
    if (newPw.length < 6) { setPwMsg("Password must be at least 6 characters"); return; }
    if (newPw !== confirmPw) { setPwMsg("Passwords don't match"); return; }
    try {
      const res = await authFetch("/user/change-password", { method: "POST", body: JSON.stringify({ current_password: currentPw, new_password: newPw }) });
      const data = await res.json();
      if (!res.ok) { setPwMsg(data.detail || "Failed"); return; }
      setPwMsg("Password updated!"); setCurrentPw(""); setNewPw(""); setConfirmPw("");
    } catch (e: any) { setPwMsg(e.message); }
  };

  const deleteAccount = async () => {
    try { await authFetch("/user/delete", { method: "DELETE" }); clearToken(); window.location.reload(); } catch (_) {}
  };

  const updateSetting = async (key: string, value: any) => {
    const updated = { ...user, [key]: value }; setUser(updated); saveUser(updated);
    try { await authFetch("/user/update", { method: "PUT", body: JSON.stringify({ [key]: value }) }); } catch (_) {}
  };

  const inputCls = "w-full bg-white/[0.06] border border-white/10 rounded-lg px-3 py-2.5 text-white text-sm focus:outline-none focus:border-violet-500/50 transition-colors";
  const toggleCls = (on: boolean) => `relative w-10 h-5 rounded-full transition-colors cursor-pointer ${on ? "bg-violet-500" : "bg-white/10"}`;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center" style={{ background: "rgba(0,0,0,0.8)", backdropFilter: "blur(12px)" }}>
      <div className="w-[95%] max-w-lg rounded-2xl overflow-hidden max-h-[85vh] overflow-y-auto" style={{ background: "rgba(15,15,22,0.98)", border: "1px solid rgba(255,255,255,0.08)" }}>
        <div className="flex items-center justify-between p-4 border-b border-white/[0.06] sticky top-0 z-10" style={{ background: "rgba(15,15,22,0.98)" }}>
          <div className="flex items-center gap-2"><Settings size={18} className="text-violet-400" /><span className="text-white text-sm font-semibold">Account Settings</span></div>
          <button onClick={onClose} className="w-7 h-7 rounded-lg bg-white/[0.05] hover:bg-white/10 flex items-center justify-center cursor-pointer"><X size={14} className="text-white/50" /></button>
        </div>
        <div className="flex gap-1 px-4 pt-3">
          {(["profile", "appearance", "notifications", "security"] as const).map(t => (
            <button key={t} onClick={() => setTab(t)} className={`px-3 py-1.5 rounded-lg text-[11px] font-semibold uppercase tracking-wider transition-all cursor-pointer ${tab === t ? "bg-violet-500/20 text-violet-400" : "text-white/30 hover:text-white/50"}`}>{t}</button>
          ))}
        </div>
        <div className="p-4 space-y-4">
          {tab === "profile" && (<>
            {error && <div className="p-2 rounded-lg bg-red-500/10 border border-red-500/20"><p className="text-red-400 text-xs">{error}</p></div>}
            <div className="flex items-center gap-4 p-3 rounded-xl bg-white/[0.03] border border-white/[0.06]">
              <div className="w-14 h-14 rounded-full bg-gradient-to-br from-violet-500 to-fuchsia-500 flex items-center justify-center text-white font-bold text-lg flex-shrink-0">{editName.slice(0, 2).toUpperCase()}</div>
              <div className="flex-1"><p className="text-white text-sm font-semibold">{editName}</p><p className="text-white/30 text-xs">@{user.username}</p></div>
            </div>
            <div><label className="text-[10px] text-white/30 block mb-1 uppercase tracking-wider">Display Name</label><input value={editName} onChange={e => setEditName(e.target.value)} className={inputCls} /></div>
            <div><label className="text-[10px] text-white/30 block mb-1 uppercase tracking-wider">Email</label><input value={editEmail} onChange={e => setEditEmail(e.target.value)} className={inputCls} /></div>
            <div><label className="text-[10px] text-white/30 block mb-1 uppercase tracking-wider">Status</label>
              <div className="flex gap-2">
                {(["online", "focus", "offline"] as const).map(s => (
                  <button key={s} onClick={() => updateSetting("status", s)} className={`flex items-center gap-1.5 px-3 py-2 rounded-lg text-xs transition-all cursor-pointer ${user.status === s ? "bg-white/10 text-white border border-white/20" : "text-white/30 hover:text-white/50 bg-white/[0.03]"}`}>
                    <span className={`w-2 h-2 rounded-full ${s === "online" ? "bg-green-500" : s === "focus" ? "bg-yellow-400" : "bg-gray-500"}`} />{s}
                  </button>
                ))}
              </div>
            </div>
            <button onClick={save} className="w-full py-2.5 rounded-xl text-sm font-semibold text-white cursor-pointer transition-all hover:scale-[1.02]" style={{ background: "linear-gradient(135deg,#a955ff,#ea51ff)" }}>
              {saved ? <span className="flex items-center justify-center gap-2"><Check size={14} />Saved!</span> : "Save Changes"}
            </button>
          </>)}
          {tab === "appearance" && (<>
            <div><label className="text-[10px] text-white/30 block mb-2 uppercase tracking-wider">Theme</label>
              <div className="flex gap-2">
                {(["dark", "light", "auto"] as const).map(t => (
                  <button key={t} onClick={() => updateSetting("theme", t)} className={`flex items-center gap-2 px-4 py-3 rounded-xl text-xs transition-all cursor-pointer flex-1 ${user.theme === t ? "bg-violet-500/20 text-violet-400 border border-violet-500/30" : "text-white/30 bg-white/[0.03] border border-white/[0.06]"}`}>
                    {t === "dark" ? <Moon size={14} /> : t === "light" ? <Sun size={14} /> : <Monitor size={14} />}{t}
                  </button>
                ))}
              </div>
            </div>
            <div className="flex items-center justify-between p-3 rounded-xl bg-white/[0.03] border border-white/[0.06]">
              <div><p className="text-white text-sm">Music Autoplay</p><p className="text-white/25 text-[10px]">Auto-play mood music on detection</p></div>
              <div onClick={() => updateSetting("musicAutoplay", !user.musicAutoplay)} className={toggleCls(user.musicAutoplay)}>
                <div className={`absolute top-0.5 w-4 h-4 rounded-full bg-white transition-all ${user.musicAutoplay ? "left-[22px]" : "left-0.5"}`} />
              </div>
            </div>
          </>)}
          {tab === "notifications" && (<>
            <div className="flex items-center justify-between p-3 rounded-xl bg-white/[0.03] border border-white/[0.06]">
              <div><p className="text-white text-sm">Push Notifications</p><p className="text-white/25 text-[10px]">Mood alerts and reminders</p></div>
              <div onClick={() => updateSetting("notifications", !user.notifications)} className={toggleCls(user.notifications)}>
                <div className={`absolute top-0.5 w-4 h-4 rounded-full bg-white transition-all ${user.notifications ? "left-[22px]" : "left-0.5"}`} />
              </div>
            </div>
          </>)}
          {tab === "security" && (<>
            <div className="p-3 rounded-xl bg-white/[0.03] border border-white/[0.06]">
              <div className="flex items-center gap-2 mb-2"><Lock size={14} className="text-white/40" /><p className="text-white text-sm">Change Password</p></div>
              <input type="password" placeholder="Current password" value={currentPw} onChange={e => setCurrentPw(e.target.value)} className={inputCls + " mb-2"} />
              <input type="password" placeholder="New password" value={newPw} onChange={e => setNewPw(e.target.value)} className={inputCls + " mb-2"} />
              <input type="password" placeholder="Confirm new password" value={confirmPw} onChange={e => setConfirmPw(e.target.value)} className={inputCls} />
              {pwMsg && <p className={`text-xs mt-2 ${pwMsg.includes("updated") ? "text-green-400" : "text-red-400"}`}>{pwMsg}</p>}
            </div>
            <button onClick={changePassword} className="w-full py-2.5 rounded-xl text-sm font-semibold text-white cursor-pointer bg-white/[0.06] hover:bg-white/10 transition-colors">Update Password</button>
            <div className="p-3 rounded-xl bg-red-500/[0.06] border border-red-500/20">
              <p className="text-red-400 text-sm font-semibold mb-1">Danger Zone</p>
              <p className="text-white/25 text-[10px] mb-2">Permanently delete your account and all data</p>
              {!deleting ? (
                <button onClick={() => setDeleting(true)} className="px-4 py-2 rounded-lg text-xs text-red-400 border border-red-500/20 hover:bg-red-500/10 transition-colors cursor-pointer">Delete Account</button>
              ) : (
                <div className="flex gap-2">
                  <button onClick={deleteAccount} className="px-4 py-2 rounded-lg text-xs text-white bg-red-600 hover:bg-red-700 cursor-pointer">Yes, delete everything</button>
                  <button onClick={() => setDeleting(false)} className="px-4 py-2 rounded-lg text-xs text-white/40 hover:text-white/60 cursor-pointer">Cancel</button>
                </div>
              )}
            </div>
          </>)}
        </div>
      </div>
    </div>
  );
}

/* ══════════════════════════════════════════════════════
   USER DROPDOWN
   ══════════════════════════════════════════════════════ */
function UserDD({ user, onSettings, onLogout }: { user: UserAccount; onSettings: () => void; onLogout: () => void }) {
  const [open, setOpen] = useState(false);
  const r = useRef<HTMLDivElement>(null);
  useEffect(() => { const h = (e: MouseEvent) => { if (r.current && !r.current.contains(e.target as Node)) setOpen(false); }; document.addEventListener("mousedown", h); return () => document.removeEventListener("mousedown", h); }, []);
  const sc: Record<string, string> = { online: "bg-green-500", focus: "bg-yellow-400", offline: "bg-gray-500" };

  const menuClick = (action: string) => {
    setOpen(false);
    if (["settings", "profile", "appearance", "notifications"].includes(action)) onSettings();
    if (action === "logout") onLogout();
  };

  return (
    <div className="relative" ref={r}>
      <button onClick={() => setOpen(!open)} className="relative w-9 h-9 rounded-full overflow-hidden border-2 border-white/20 hover:border-white/50 transition-all focus:outline-none cursor-pointer">
        <div className="w-full h-full bg-gradient-to-br from-violet-500 to-fuchsia-500 flex items-center justify-center text-white font-bold text-[11px]">{user.name.slice(0, 2).toUpperCase()}</div>
        <span className={`absolute bottom-0 right-0 w-2.5 h-2.5 ${sc[user.status]} rounded-full border-2 border-black`} />
      </button>
      {open && (
        <div className="absolute right-0 top-12 w-[260px] rounded-2xl overflow-hidden z-50" style={{ background: "rgba(12,12,18,0.96)", backdropFilter: "blur(24px)", border: "1px solid rgba(255,255,255,0.08)", boxShadow: "0 25px 60px rgba(0,0,0,0.7)" }}>
          <div className="p-3 flex items-center gap-3 border-b border-white/[0.06]">
            <div className="relative w-9 h-9 rounded-full bg-gradient-to-br from-violet-500 to-fuchsia-500 flex items-center justify-center text-white font-bold text-[11px] flex-shrink-0">{user.name.slice(0, 2).toUpperCase()}<span className={`absolute -bottom-0.5 -right-0.5 w-3 h-3 ${sc[user.status]} rounded-full border-2 border-[#0c0c12]`} /></div>
            <div className="flex-1 min-w-0"><p className="text-white text-xs font-semibold truncate">{user.name}</p><p className="text-white/35 text-[10px] truncate">{user.email}</p></div>
          </div>
          {[{ icon: User, label: "Your profile", action: "profile" }, { icon: Sun, label: "Appearance", action: "appearance" }, { icon: Settings, label: "Settings", action: "settings" }, { icon: Bell, label: "Notifications", action: "notifications" }].map((it, i) => (
            <button key={i} onClick={() => menuClick(it.action)} className="flex items-center gap-2.5 w-full px-4 py-2 text-[12px] text-white/50 hover:text-white hover:bg-white/[0.05] transition-all cursor-pointer"><it.icon size={15} className="text-white/30" />{it.label}</button>
          ))}
          <div className="h-px bg-white/[0.04] mx-2" />
          <button onClick={() => menuClick("logout")} className="flex items-center gap-2.5 w-full px-4 py-2 text-[12px] text-white/35 hover:text-red-400 hover:bg-red-500/[0.06] transition-all cursor-pointer"><LogOut size={15} />Log out</button>
        </div>
      )}
    </div>
  );
}

/* ══════════════════════════════════════════════════════
   SPOTIFY PLAYER OVERLAY — Real Playback
   ══════════════════════════════════════════════════════ */
const MOOD_SEARCH_QUERIES: Record<string, string[]> = {
  stressed: ["calming piano music", "ambient relaxation", "nature sounds meditation"],
  relaxed:  ["chill acoustic vibes", "feel good indie", "lazy sunday afternoon"],
  focused:  ["lofi hip hop beats", "deep focus ambient", "study music concentration"],
  fatigued: ["upbeat morning energy", "happy pop songs", "motivational workout"],
};

interface SpotifyTrack {
  id: string;
  name: string;
  artist: string;
  album: string;
  albumArt: string;
  uri: string;
  duration_ms: number;
}

function SpotifyOverlay({ mood, visible, onClose }: { mood: string; visible: boolean; onClose: () => void }) {
  const [connected, setConnected] = useState(false);
  const [tracks, setTracks] = useState<SpotifyTrack[]>([]);
  const [currentTrack, setCurrentTrack] = useState<SpotifyTrack | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [progress, setProgress] = useState(0);
  const [deviceId, setDeviceId] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const playerRef = useRef<any>(null);
  const progressRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const meta = MOOD_META[mood] || { emoji: "\u{1F916}", color: "#a78bfa" };

  // Check Spotify connection on mount
  useEffect(() => {
    const token = getSpotifyToken();
    setConnected(!!token);
  }, [visible]);

  // Initialize Spotify Web Playback SDK
  useEffect(() => {
    if (!connected || !visible) return;
    const token = getSpotifyToken();
    if (!token) return;

    // Load SDK
    if (!(window as any).Spotify) {
      const script = document.createElement("script");
      script.src = "https://sdk.scdn.co/spotify-player.js";
      document.body.appendChild(script);
    }

    (window as any).onSpotifyWebPlaybackSDKReady = () => {
      const player = new (window as any).Spotify.Player({
        name: "MoodSpace Player",
        getOAuthToken: (cb: (t: string) => void) => { cb(token!); },
        volume: 0.5,
      });

      player.addListener("ready", ({ device_id }: { device_id: string }) => {
        setDeviceId(device_id);
        playerRef.current = player;
      });

      player.addListener("player_state_changed", (state: any) => {
        if (!state) return;
        setIsPlaying(!state.paused);
        setProgress(state.position);
        if (state.track_window?.current_track) {
          const t = state.track_window.current_track;
          setCurrentTrack({
            id: t.id,
            name: t.name,
            artist: t.artists.map((a: any) => a.name).join(", "),
            album: t.album.name,
            albumArt: t.album.images[0]?.url || "",
            uri: t.uri,
            duration_ms: t.duration_ms,
          });
        }
      });

      player.connect();
    };

    // If SDK already loaded
    if ((window as any).Spotify) {
      (window as any).onSpotifyWebPlaybackSDKReady();
    }

    return () => {
      if (progressRef.current) clearInterval(progressRef.current);
      playerRef.current?.disconnect();
    };
  }, [connected, visible]);

  // Search tracks based on mood — and auto-play first result when mood changes
  const prevMoodRef = useRef(mood);
  useEffect(() => {
    if (!connected || !visible) return;
    const moodChanged = prevMoodRef.current !== mood;
    prevMoodRef.current = mood;
    searchMoodTracks(moodChanged);
  }, [mood, connected, visible]);

  // Progress ticker
  useEffect(() => {
    if (progressRef.current) clearInterval(progressRef.current);
    if (isPlaying && currentTrack) {
      progressRef.current = setInterval(() => {
        setProgress(p => Math.min(p + 1000, currentTrack.duration_ms));
      }, 1000);
    }
    return () => { if (progressRef.current) clearInterval(progressRef.current); };
  }, [isPlaying, currentTrack]);

  const searchMoodTracks = async (autoPlay = false) => {
    setLoading(true);
    setError("");
    try {
      const queries = MOOD_SEARCH_QUERIES[mood] || MOOD_SEARCH_QUERIES.relaxed;
      const query = queries[Math.floor(Math.random() * queries.length)];
      const res = await spotifyFetch(`/search?q=${encodeURIComponent(query)}&type=track&limit=8`);
      const data = await res.json();
      const items = data.tracks?.items || [];
      const parsed: SpotifyTrack[] = items.map((t: any) => ({
        id: t.id,
        name: t.name,
        artist: t.artists.map((a: any) => a.name).join(", "),
        album: t.album.name,
        albumArt: t.album.images?.[1]?.url || t.album.images?.[0]?.url || "",
        uri: t.uri,
        duration_ms: t.duration_ms,
      }));
      setTracks(parsed);
      // Auto-play first track when mood changes
      if (autoPlay && parsed.length > 0 && deviceId) {
        playTrack(parsed[0]);
      }
    } catch (e: any) {
      if (e.message.includes("expired")) setConnected(false);
      else setError("Failed to load tracks");
    }
    setLoading(false);
  };

  const playTrack = async (track: SpotifyTrack) => {
    if (!deviceId) { setError("No playback device ready. Refresh and try again."); return; }
    try {
      await spotifyFetch(`/me/player/play?device_id=${deviceId}`, {
        method: "PUT",
        body: JSON.stringify({ uris: [track.uri] }),
      });
      setCurrentTrack(track);
      setIsPlaying(true);
      setProgress(0);
    } catch (e: any) {
      setError("Playback failed. Make sure Spotify Premium is active.");
    }
  };

  const togglePlay = () => { playerRef.current?.togglePlay(); };
  const nextTrack = () => { playerRef.current?.nextTrack(); };
  const prevTrack = () => { playerRef.current?.previousTrack(); };

  const formatTime = (ms: number) => {
    const s = Math.floor(ms / 1000);
    return `${Math.floor(s / 60)}:${(s % 60).toString().padStart(2, "0")}`;
  };

  if (!visible) return null;

  return (
    <div className="fixed bottom-4 right-4 z-40 w-[340px] rounded-2xl overflow-hidden" style={{ background: "rgba(12,12,18,0.95)", backdropFilter: "blur(24px)", border: "1px solid rgba(255,255,255,0.08)", boxShadow: "0 20px 50px rgba(0,0,0,0.5)" }}>
      {/* Header */}
      <div className="p-3 flex items-center justify-between border-b border-white/[0.04]">
        <div className="flex items-center gap-2">
          <div className="w-5 h-5 rounded-full bg-green-500 flex items-center justify-center"><Music size={10} className="text-black" /></div>
          <span className="text-[10px] text-green-400 font-semibold uppercase tracking-wider">Spotify</span>
          {connected && <span className="w-1.5 h-1.5 rounded-full bg-green-400" />}
        </div>
        <button onClick={onClose} className="w-5 h-5 rounded bg-white/[0.05] hover:bg-white/10 flex items-center justify-center cursor-pointer"><X size={10} className="text-white/40" /></button>
      </div>

      {!connected ? (
        /* Connect to Spotify */
        <div className="p-5 text-center">
          <div className="w-16 h-16 rounded-full bg-green-500/10 flex items-center justify-center mx-auto mb-4">
            <Music size={28} className="text-green-400" />
          </div>
          <p className="text-white text-sm font-semibold mb-1">Connect Spotify</p>
          <p className="text-white/30 text-[11px] mb-4">Play mood-matched music directly from your Spotify Premium account</p>
          <button onClick={startSpotifyAuth} className="w-full py-2.5 rounded-xl text-sm font-semibold text-black cursor-pointer transition-all hover:scale-[1.02] bg-green-500 hover:bg-green-400">
            Connect with Spotify
          </button>
        </div>
      ) : (
        <>
          {/* Mood context */}
          <div className="flex items-center gap-2 px-3 py-2 border-b border-white/[0.03]">
            <span className="text-sm">{meta.emoji}</span>
            <span className="text-[10px] text-white/30">Playing for <span className="font-semibold" style={{ color: meta.color }}>{mood}</span> mood</span>
            <button onClick={() => searchMoodTracks()} className="ml-auto text-[9px] text-white/20 hover:text-white/40 cursor-pointer">Refresh</button>
          </div>

          {/* Now playing bar */}
          {currentTrack && (
            <div className="p-3 border-b border-white/[0.04]">
              <div className="flex items-center gap-3 mb-2">
                {currentTrack.albumArt && <img src={currentTrack.albumArt} alt="" className="w-11 h-11 rounded-lg object-cover" />}
                <div className="flex-1 min-w-0">
                  <p className="text-white text-xs font-semibold truncate">{currentTrack.name}</p>
                  <p className="text-white/30 text-[10px] truncate">{currentTrack.artist}</p>
                </div>
              </div>
              {/* Progress bar */}
              <div className="flex items-center gap-2">
                <span className="text-[9px] text-white/20 w-8 text-right">{formatTime(progress)}</span>
                <div className="flex-1 h-1 rounded-full bg-white/10 overflow-hidden">
                  <div className="h-full rounded-full bg-green-400 transition-all" style={{ width: `${(progress / currentTrack.duration_ms) * 100}%` }} />
                </div>
                <span className="text-[9px] text-white/20 w-8">{formatTime(currentTrack.duration_ms)}</span>
              </div>
              {/* Controls */}
              <div className="flex items-center justify-center gap-4 mt-2">
                <button onClick={prevTrack} className="text-white/30 hover:text-white cursor-pointer"><SkipBack size={16} /></button>
                <button onClick={togglePlay} className="w-8 h-8 rounded-full bg-white flex items-center justify-center cursor-pointer hover:scale-105 transition-transform">
                  {isPlaying ? <Pause size={14} className="text-black" /> : <Play size={14} className="text-black ml-0.5" />}
                </button>
                <button onClick={nextTrack} className="text-white/30 hover:text-white cursor-pointer"><SkipForward size={16} /></button>
              </div>
            </div>
          )}

          {/* Track list */}
          <div className="max-h-[240px] overflow-y-auto p-2">
            {loading ? (
              <div className="py-6 text-center"><p className="text-white/20 text-xs">Finding tracks...</p></div>
            ) : error ? (
              <div className="py-4 text-center"><p className="text-red-400 text-xs">{error}</p></div>
            ) : (
              tracks.map((track) => (
                <div key={track.id} onClick={() => playTrack(track)} className={`flex items-center gap-3 px-2 py-2 rounded-xl hover:bg-white/[0.04] transition-colors cursor-pointer group ${currentTrack?.id === track.id ? "bg-white/[0.04]" : ""}`}>
                  <div className="relative w-9 h-9 rounded-lg overflow-hidden flex-shrink-0">
                    {track.albumArt ? <img src={track.albumArt} alt="" className="w-full h-full object-cover" /> : <div className="w-full h-full bg-white/10 flex items-center justify-center"><Music size={14} className="text-white/20" /></div>}
                    <div className="absolute inset-0 bg-black/40 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity">
                      <Play size={12} className="text-white ml-0.5" />
                    </div>
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="text-white text-[11px] font-medium truncate">{track.name}</p>
                    <p className="text-white/25 text-[10px] truncate">{track.artist}</p>
                  </div>
                  {currentTrack?.id === track.id && isPlaying && (
                    <div className="flex gap-0.5 items-end h-3">
                      <div className="w-0.5 bg-green-400 rounded-full" style={{ height: "40%", animation: "eqBar 0.4s ease-in-out infinite alternate" }} />
                      <div className="w-0.5 bg-green-400 rounded-full" style={{ height: "70%", animation: "eqBar 0.4s ease-in-out 0.1s infinite alternate" }} />
                      <div className="w-0.5 bg-green-400 rounded-full" style={{ height: "50%", animation: "eqBar 0.4s ease-in-out 0.2s infinite alternate" }} />
                    </div>
                  )}
                </div>
              ))
            )}
          </div>

          {/* Disconnect */}
          <div className="px-3 py-2 border-t border-white/[0.04]">
            <button onClick={() => { clearSpotifyToken(); setConnected(false); setTracks([]); setCurrentTrack(null); }} className="text-[9px] text-white/15 hover:text-red-400 cursor-pointer transition-colors">Disconnect Spotify</button>
          </div>
        </>
      )}
    </div>
  );
}

/* ══════════════════════════════════════════════════════
   AI HEALTH INSIGHTS PANEL
   ══════════════════════════════════════════════════════ */
interface HealthTip {
  icon: any;
  title: string;
  tip: string;
  color: string;
  type: "mood" | "general";
}

const MOOD_HEALTH_TIPS: Record<string, HealthTip[]> = {
  stressed: [
    { icon: Wind, title: "Box Breathing", tip: "Inhale 4s, hold 4s, exhale 4s, hold 4s. Repeat 4 times to activate your parasympathetic nervous system.", color: "#60a5fa", type: "mood" },
    { icon: Droplets, title: "Hydrate Now", tip: "Stress increases cortisol which dehydrates you. Drink a full glass of water right now.", color: "#22d3ee", type: "mood" },
    { icon: Eye, title: "20-20-20 Rule", tip: "Look at something 20 feet away for 20 seconds every 20 minutes to reduce eye strain and mental fatigue.", color: "#a78bfa", type: "mood" },
  ],
  focused: [
    { icon: Droplets, title: "Stay Hydrated", tip: "Drink water every 30 minutes to maintain cognitive performance. Dehydration reduces focus by up to 25%.", color: "#22d3ee", type: "mood" },
    { icon: Clock, title: "Pomodoro Check", tip: "If you've been focused for 25+ minutes, take a 5-minute break. Stand up, stretch, look away from the screen.", color: "#facc15", type: "mood" },
    { icon: Apple, title: "Brain Fuel", tip: "Snack on walnuts, dark chocolate, or blueberries — they boost BDNF and improve sustained attention.", color: "#4ade80", type: "mood" },
  ],
  relaxed: [
    { icon: Activity, title: "Light Movement", tip: "A relaxed state is perfect for a 10-minute walk. It boosts serotonin and prevents post-relaxation sluggishness.", color: "#4ade80", type: "mood" },
    { icon: Droplets, title: "Green Tea", tip: "L-theanine in green tea maintains calm focus without drowsiness. Great for transitioning from relaxed to productive.", color: "#22d3ee", type: "mood" },
  ],
  fatigued: [
    { icon: Sun, title: "Sunlight Exposure", tip: "Get 5-10 minutes of direct sunlight. Indians commonly have low Vitamin D — sunlight resets your circadian rhythm and boosts energy.", color: "#fbbf24", type: "mood" },
    { icon: Coffee, title: "Strategic Caffeine", tip: "If it's before 2 PM, a small coffee can help. After 2 PM, try cold water on your face instead — it triggers the diving reflex and boosts alertness.", color: "#f97316", type: "mood" },
    { icon: Wind, title: "Power Nap", tip: "A 15-20 minute nap now is more effective than pushing through. Set an alarm — longer naps cause grogginess.", color: "#94a3b8", type: "mood" },
  ],
};

const GENERAL_HEALTH_TIPS: HealthTip[] = [
  { icon: Pill, title: "B12 Deficiency Alert", tip: "Up to 47% of Indians are B12 deficient (especially vegetarians). Symptoms: fatigue, brain fog, mood swings. Consider methylcobalamin supplements or fortified foods.", color: "#f472b6", type: "general" },
  { icon: Sun, title: "Vitamin D Crisis", tip: "70-90% of Indians have insufficient Vitamin D despite sunny climate. Get 15 mins of midday sun with arms exposed, or supplement 1000-2000 IU daily.", color: "#fbbf24", type: "general" },
  { icon: Droplets, title: "Hydration Reminder", tip: "Drink water every 30 minutes. By the time you feel thirsty, you're already 1-2% dehydrated — enough to impair mood and cognition.", color: "#22d3ee", type: "general" },
  { icon: Apple, title: "Iron Intake", tip: "Anemia affects ~50% of Indian women and 25% of men. Pair iron-rich foods (spinach, lentils, jaggery) with Vitamin C for 6x better absorption.", color: "#4ade80", type: "general" },
  { icon: Moon, title: "Sleep Hygiene", tip: "Screen blue light suppresses melatonin by 55%. Enable night mode after sunset and avoid screens 30 mins before bed.", color: "#818cf8", type: "general" },
  { icon: Activity, title: "Sedentary Alert", tip: "Sitting for 8+ hours increases heart disease risk by 147%. Stand and move for 2 minutes every 30 minutes.", color: "#fb923c", type: "general" },
];

function HealthInsightsPanel({ mood, visible, onClose }: { mood: string; visible: boolean; onClose: () => void }) {
  const [tab, setTab] = useState<"mood" | "general">("mood");
  const meta = MOOD_META[mood] || { emoji: "\u{1F916}", color: "#a78bfa" };

  // Auto-rotate tips every 15 minutes
  const [moodTipIdx, setMoodTipIdx] = useState(0);
  const [generalTipIdx, setGeneralTipIdx] = useState(0);
  const [fadeKey, setFadeKey] = useState(0);

  const allMoodTips = MOOD_HEALTH_TIPS[mood] || MOOD_HEALTH_TIPS.relaxed;
  const ROTATE_MS = 15 * 60 * 1000; // 15 minutes

  useEffect(() => {
    const interval = setInterval(() => {
      setMoodTipIdx(prev => (prev + 1) % allMoodTips.length);
      setGeneralTipIdx(prev => (prev + 1) % GENERAL_HEALTH_TIPS.length);
      setFadeKey(k => k + 1);
    }, ROTATE_MS);
    return () => clearInterval(interval);
  }, [allMoodTips.length]);

  // Reset mood tip index when mood changes
  useEffect(() => {
    setMoodTipIdx(0);
    setFadeKey(k => k + 1);
  }, [mood]);

  // Show 2 tips at a time, cycling through
  const visibleMood = [
    allMoodTips[moodTipIdx % allMoodTips.length],
    allMoodTips[(moodTipIdx + 1) % allMoodTips.length],
  ].filter((v, i, a) => a.indexOf(v) === i); // dedupe if only 2 tips

  const visibleGeneral = [
    GENERAL_HEALTH_TIPS[generalTipIdx % GENERAL_HEALTH_TIPS.length],
    GENERAL_HEALTH_TIPS[(generalTipIdx + 1) % GENERAL_HEALTH_TIPS.length],
    GENERAL_HEALTH_TIPS[(generalTipIdx + 2) % GENERAL_HEALTH_TIPS.length],
  ];

  // Time until next rotation
  const [countdown, setCountdown] = useState(ROTATE_MS);
  useEffect(() => {
    setCountdown(ROTATE_MS);
    const t = setInterval(() => setCountdown(c => Math.max(0, c - 1000)), 1000);
    return () => clearInterval(t);
  }, [fadeKey]);
  const minsLeft = Math.ceil(countdown / 60000);

  if (!visible) return null;

  return (
    <div className="fixed bottom-4 left-4 z-40 w-[340px] rounded-2xl overflow-hidden" style={{ background: "rgba(12,12,18,0.95)", backdropFilter: "blur(24px)", border: "1px solid rgba(255,255,255,0.08)", boxShadow: "0 20px 50px rgba(0,0,0,0.5)" }}>
      <div className="p-3 flex items-center justify-between border-b border-white/[0.04]">
        <div className="flex items-center gap-2">
          <div className="w-5 h-5 rounded-full bg-emerald-500/20 flex items-center justify-center"><Heart size={10} className="text-emerald-400" /></div>
          <span className="text-[10px] text-emerald-400 font-semibold uppercase tracking-wider">Health Insights</span>
          <span className="text-[8px] text-white/15">refreshes in {minsLeft}m</span>
        </div>
        <button onClick={onClose} className="w-5 h-5 rounded bg-white/[0.05] hover:bg-white/10 flex items-center justify-center cursor-pointer"><X size={10} className="text-white/40" /></button>
      </div>

      <div className="flex gap-1 px-3 pt-2">
        <button onClick={() => setTab("mood")} className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-[10px] font-semibold uppercase tracking-wider transition-all cursor-pointer ${tab === "mood" ? "bg-emerald-500/15 text-emerald-400" : "text-white/25 hover:text-white/40"}`}>
          <span className="text-xs">{meta.emoji}</span> For {mood}
        </button>
        <button onClick={() => setTab("general")} className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-[10px] font-semibold uppercase tracking-wider transition-all cursor-pointer ${tab === "general" ? "bg-emerald-500/15 text-emerald-400" : "text-white/25 hover:text-white/40"}`}>
          General
        </button>
      </div>

      <div key={fadeKey} className="p-3 space-y-2 max-h-[320px] overflow-y-auto" style={{ animation: "fadeIn 0.5s ease-out" }}>
        {(tab === "mood" ? visibleMood : visibleGeneral).map((tip, i) => (
          <div key={`${tip.title}-${i}`} className="p-3 rounded-xl bg-white/[0.02] border border-white/[0.06] hover:bg-white/[0.04] transition-colors">
            <div className="flex items-center gap-2 mb-1.5">
              <div className="w-6 h-6 rounded-lg flex items-center justify-center" style={{ background: `${tip.color}15`, border: `1px solid ${tip.color}25` }}>
                <tip.icon size={12} style={{ color: tip.color }} />
              </div>
              <span className="text-white text-xs font-semibold">{tip.title}</span>
            </div>
            <p className="text-white/40 text-[11px] leading-relaxed">{tip.tip}</p>
          </div>
        ))}

        <button onClick={() => {
          if (tab === "mood") setMoodTipIdx(prev => (prev + 1) % allMoodTips.length);
          else setGeneralTipIdx(prev => (prev + 1) % GENERAL_HEALTH_TIPS.length);
          setFadeKey(k => k + 1);
        }} className="w-full py-2 text-[10px] text-white/20 hover:text-white/40 cursor-pointer transition-colors flex items-center justify-center gap-1">
          <Shuffle size={10} /> Show different tips
        </button>
      </div>
    </div>
  );
}

/* ══════════════════════════════════════════════════════
   IOT SENSOR PLACEHOLDER
   ══════════════════════════════════════════════════════ */
function IoTPanel({ visible, onClose }: { visible: boolean; onClose: () => void }) {
  if (!visible) return null;
  const sensors = [
    { name: "Heart Rate", icon: Activity, value: "-- bpm", status: "waiting", color: "#f87171" },
    { name: "Temperature", icon: Thermometer, value: "-- °C", status: "waiting", color: "#facc15" },
    { name: "GSR / EDA", icon: Zap, value: "-- µS", status: "waiting", color: "#60a5fa" },
    { name: "SpO2", icon: Heart, value: "-- %", status: "waiting", color: "#4ade80" },
    { name: "Accelerometer", icon: Activity, value: "-- g", status: "waiting", color: "#c084fc" },
    { name: "Ambient Light", icon: Sun, value: "-- lux", status: "waiting", color: "#fbbf24" },
  ];
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center" style={{ background: "rgba(0,0,0,0.8)", backdropFilter: "blur(12px)" }}>
      <div className="w-[95%] max-w-lg rounded-2xl overflow-hidden" style={{ background: "rgba(15,15,22,0.98)", border: "1px solid rgba(255,255,255,0.08)" }}>
        <div className="flex items-center justify-between p-4 border-b border-white/[0.06]">
          <div className="flex items-center gap-2"><Cpu size={18} className="text-cyan-400" /><span className="text-white text-sm font-semibold">IoT Sensor Hub</span></div>
          <button onClick={onClose} className="w-7 h-7 rounded-lg bg-white/[0.05] hover:bg-white/10 flex items-center justify-center cursor-pointer"><X size={14} className="text-white/50" /></button>
        </div>
        <div className="p-4">
          <div className="flex items-center gap-2 p-3 rounded-xl bg-yellow-500/[0.06] border border-yellow-500/20 mb-4">
            <Wifi size={16} className="text-yellow-400" />
            <div><p className="text-yellow-400 text-xs font-semibold">Hardware Not Connected</p><p className="text-white/25 text-[10px]">Connect your Arduino/ESP32 via USB or Bluetooth</p></div>
          </div>
          <div className="grid grid-cols-2 gap-3">
            {sensors.map((s, i) => (
              <div key={i} className="p-3 rounded-xl bg-white/[0.02] border border-white/[0.06]">
                <div className="flex items-center gap-2 mb-2"><s.icon size={14} style={{ color: s.color }} /><span className="text-[10px] text-white/40 uppercase tracking-wider">{s.name}</span></div>
                <p className="text-white/20 text-lg font-bold">{s.value}</p>
                <div className="flex items-center gap-1 mt-1"><span className="w-1.5 h-1.5 rounded-full bg-yellow-500" /><span className="text-[9px] text-yellow-500/60">{s.status}</span></div>
              </div>
            ))}
          </div>
          <button className="w-full mt-4 py-2.5 rounded-xl text-sm font-semibold text-white cursor-pointer transition-all hover:scale-[1.02]" style={{ background: "linear-gradient(135deg,#22d3ee,#3b82f6)" }}>Scan for Devices</button>
        </div>
      </div>
    </div>
  );
}

/* ══════════════════════════════════════════════════════
   LIVE MOOD OVERLAY
   ══════════════════════════════════════════════════════ */
const MOOD_META: Record<string, { emoji: string; color: string }> = {
  stressed: { emoji: "\u{1F624}", color: "#f87171" },
  relaxed:  { emoji: "\u{1F60C}", color: "#4ade80" },
  focused:  { emoji: "\u{1F3AF}", color: "#facc15" },
  fatigued: { emoji: "\u{1F634}", color: "#94a3b8" },
};

const MOOD_ACTIONS: Record<string, string> = {
  stressed: "AC: 20°C Fan High | Music: Calm",
  relaxed:  "AC: 24°C Fan Low  | Music: Ambient",
  focused:  "AC: 22°C Fan Med  | Music: Lofi",
  fatigued: "AC: 23°C Fan Med  | Music: Energetic",
};

function LiveMoodOverlay({ visible, onClose, onMoodChange }: { visible: boolean; onClose: () => void; onMoodChange: (mood: string) => void }) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [topMood, setTopMood] = useState("relaxed");
  const [topConf, setTopConf] = useState(0);
  const [scores, setScores] = useState<Record<string, number>>({ stressed: 25, relaxed: 25, focused: 25, fatigued: 25 });
  const [action, setAction] = useState(MOOD_ACTIONS.relaxed);
  const [faceDetected, setFaceDetected] = useState(false);
  const [apiOnline, setApiOnline] = useState(false);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    if (!visible) return;
    let s: MediaStream | null = null;
    navigator.mediaDevices.getUserMedia({ video: { facingMode: "user", width: 640, height: 480 } })
      .then((ms) => { s = ms; setStream(ms); if (videoRef.current) videoRef.current.srcObject = ms; })
      .catch(() => {});
    return () => { s?.getTracks().forEach(t => t.stop()); setStream(null); };
  }, [visible]);

  useEffect(() => {
    if (!visible) return;
    fetch(`${API_URL}/health`).then(r => r.json()).then(() => setApiOnline(true)).catch(() => setApiOnline(false));
  }, [visible]);

  useEffect(() => {
    if (!visible || !apiOnline || !stream) return;
    let busy = false;
    const detect = async () => {
      if (busy || !videoRef.current || !canvasRef.current) return;
      const video = videoRef.current;
      if (video.readyState < 2 || video.videoWidth === 0) return;
      busy = true;
      const canvas = canvasRef.current;
      canvas.width = 480; canvas.height = 360;
      const ctx = canvas.getContext("2d");
      if (!ctx) { busy = false; return; }
      ctx.drawImage(video, 0, 0, 480, 360);
      const dataUrl = canvas.toDataURL("image/jpeg", 0.8);
      try {
        const controller = new AbortController();
        const timeout = setTimeout(() => controller.abort(), 3000);
        const res = await fetch(`${API_URL}/detect`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ image: dataUrl }), signal: controller.signal });
        clearTimeout(timeout);
        if (!res.ok) { busy = false; return; }
        const data = await res.json();
        if (!data.error && data.mood) {
          setTopMood(data.mood); setTopConf(typeof data.confidence === "number" ? data.confidence : 0);
          setScores(data.scores || { focused: 25, relaxed: 25, stressed: 25, fatigued: 25 });
          setAction(data.action || MOOD_ACTIONS[data.mood] || ""); setFaceDetected(!!data.face_detected);
          onMoodChange(data.mood);
        }
      } catch (_) {}
      busy = false;
    };
    intervalRef.current = setInterval(detect, 1000);
    return () => { if (intervalRef.current) clearInterval(intervalRef.current); };
  }, [visible, apiOnline, stream, onMoodChange]);

  if (!visible) return null;
  const meta = MOOD_META[topMood] || { emoji: "\u{1F916}", color: "#a78bfa" };

  return (
    <div className="fixed inset-0 z-50" style={{ background: "#000" }}>
      <canvas ref={canvasRef} style={{ display: "none" }} />
      <video ref={videoRef} autoPlay playsInline muted className="absolute inset-0 w-full h-full object-cover" />
      <div className="absolute inset-0 bg-black/10" />
      <div className="absolute top-0 left-0 right-0 flex items-center justify-between px-5 py-3 z-20">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 rounded-lg flex items-center justify-center" style={{ background: "linear-gradient(135deg,#a955ff,#ea51ff)" }}><Brain size={16} className="text-white" /></div>
          <span className="text-white text-sm font-bold">MoodSpace Live</span>
        </div>
        <div className="flex items-center gap-3">
          <span className={`flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[10px] font-semibold uppercase tracking-wider ${apiOnline ? "bg-green-500/20 text-green-400 border border-green-500/30" : "bg-red-500/20 text-red-400 border border-red-500/30"}`}>
            <span className={`w-1.5 h-1.5 rounded-full ${apiOnline ? "bg-green-400" : "bg-red-400"}`} />{apiOnline ? "Live" : "Offline"}
          </span>
          <span className={`px-2.5 py-1 rounded-full text-[10px] font-semibold ${faceDetected ? "bg-blue-500/20 text-blue-400 border border-blue-500/30" : "bg-yellow-500/20 text-yellow-400 border border-yellow-500/30"}`}>
            {faceDetected ? "Face ✓" : "No Face"}
          </span>
          <button onClick={onClose} className="w-8 h-8 rounded-full bg-white/10 hover:bg-white/20 flex items-center justify-center cursor-pointer"><X size={16} className="text-white" /></button>
        </div>
      </div>
      <div className="absolute top-16 left-4 z-20 space-y-2" style={{ width: 200 }}>
        {(["focused", "relaxed", "stressed", "fatigued"] as const).map(m => {
          const s = scores?.[m] ?? 25; const mt = MOOD_META[m]; const isTop = m === topMood;
          return (
            <div key={m}>
              <div className="flex items-center gap-2 mb-0.5">
                <span className="text-sm">{mt.emoji}</span>
                <span className={`text-[11px] font-semibold uppercase tracking-wider ${isTop ? "text-white" : "text-white/50"}`}>{m}</span>
                <span className="text-[11px] font-bold ml-auto" style={{ color: mt.color }}>{Math.round(s)}%</span>
              </div>
              <div className="h-3 rounded-full overflow-hidden" style={{ background: "rgba(255,255,255,0.08)" }}>
                <div className="h-full rounded-full transition-all duration-500" style={{ width: `${s}%`, background: mt.color, boxShadow: isTop ? `0 0 12px ${mt.color}60` : "none" }} />
              </div>
            </div>
          );
        })}
      </div>
      <div className="absolute top-16 right-4 z-20" style={{ width: 170 }}>
        <div className="rounded-2xl p-4 text-center" style={{ background: "rgba(0,0,0,0.6)", backdropFilter: "blur(16px)", border: `1px solid ${meta.color}30` }}>
          <span className="text-5xl block mb-2">{meta.emoji}</span>
          <p className="text-white text-lg font-bold uppercase">{topMood}</p>
          <p className="text-2xl font-extrabold mt-1" style={{ color: meta.color }}>{Math.round(topConf)}%</p>
        </div>
      </div>
      <div className="absolute bottom-0 left-0 right-0 z-20 px-5 py-3 flex items-center justify-between" style={{ background: meta.color, transition: "background 0.5s" }}>
        <div className="flex items-center gap-3"><span className="text-xl">{meta.emoji}</span><span className="text-black/80 text-sm font-semibold">{action}</span></div>
        <span className="text-black/50 text-xs">MoodSpace</span>
      </div>
      {!apiOnline && (
        <div className="absolute inset-0 flex items-center justify-center z-30" style={{ background: "rgba(0,0,0,0.7)" }}>
          <div className="text-center p-8 rounded-2xl max-w-md" style={{ background: "rgba(20,20,30,0.95)", border: "1px solid rgba(255,255,255,0.1)" }}>
            <div className="text-4xl mb-4">{"\u{1F50C}"}</div>
            <p className="text-white text-lg font-bold mb-2">API Server Not Running</p>
            <p className="text-white/40 text-sm mb-4">Start the server for live detection:</p>
            <div className="rounded-lg p-3 text-left" style={{ background: "rgba(255,255,255,0.05)" }}>
              <code className="text-green-400 text-xs block">cd &quot;E:\moodspace\server&quot;</code>
              <code className="text-green-400 text-xs block mt-1">python server.py</code>
            </div>
            <button onClick={onClose} className="mt-4 px-6 py-2 rounded-full text-sm text-white/60 border border-white/10 hover:bg-white/5 cursor-pointer">Go Back</button>
          </div>
        </div>
      )}
    </div>
  );
}

/* ══════════════════════════════════════════════════════
   HOMEPAGE MOOD MUSIC RECOMMENDATIONS
   ══════════════════════════════════════════════════════ */
const MOOD_SONG_RECS: Record<string, { title: string; artist: string; reason: string }[]> = {
  stressed: [
    { title: "Weightless", artist: "Marconi Union", reason: "Scientifically proven to reduce anxiety by 65%" },
    { title: "Clair de Lune", artist: "Debussy", reason: "Slow tempo calms your heart rate" },
    { title: "Sunset Lover", artist: "Petit Biscuit", reason: "Ambient textures ease racing thoughts" },
    { title: "River Flows in You", artist: "Yiruma", reason: "Gentle piano melody for deep breathing" },
    { title: "Nuvole Bianche", artist: "Ludovico Einaudi", reason: "Meditative minimalism to slow your mind" },
  ],
  relaxed: [
    { title: "Banana Pancakes", artist: "Jack Johnson", reason: "Warm acoustic for your chill mood" },
    { title: "Put It All on Me", artist: "Ed Sheeran", reason: "Feel-good rhythm to keep the vibe" },
    { title: "Here Comes the Sun", artist: "The Beatles", reason: "Timeless positivity" },
    { title: "Better Together", artist: "Jack Johnson", reason: "Laid-back sunday energy" },
    { title: "Three Little Birds", artist: "Bob Marley", reason: "Keep the relaxed flow going" },
  ],
  focused: [
    { title: "Lofi Study Beats", artist: "ChilledCow", reason: "Steady BPM keeps you in the zone" },
    { title: "Interstellar Theme", artist: "Hans Zimmer", reason: "Epic focus without lyrics to distract" },
    { title: "Experience", artist: "Ludovico Einaudi", reason: "Builds concentration gradually" },
    { title: "Time", artist: "Hans Zimmer", reason: "Layered crescendo for deep work" },
    { title: "Intro", artist: "The xx", reason: "Minimal electronic — zero distraction" },
  ],
  fatigued: [
    { title: "Wake Me Up", artist: "Avicii", reason: "High energy to snap out of fatigue" },
    { title: "On Top of the World", artist: "Imagine Dragons", reason: "Upbeat anthem to boost energy" },
    { title: "Happy", artist: "Pharrell Williams", reason: "Instant mood and energy lift" },
    { title: "Walking on Sunshine", artist: "Katrina & The Waves", reason: "Pure energy injection" },
    { title: "Don't Stop Me Now", artist: "Queen", reason: "The ultimate fatigue killer" },
  ],
};

function MoodMusicRecs({ mood, onOpenSpotify }: { mood: string; onOpenSpotify: () => void }) {
  const songs = MOOD_SONG_RECS[mood] || MOOD_SONG_RECS.relaxed;
  const meta = MOOD_META[mood] || { emoji: "\u{1F916}", color: "#a78bfa" };
  const spotifyConnected = !!getSpotifyToken();

  // Show 3 songs, rotate every 30 seconds for variety
  const [offset, setOffset] = useState(0);
  useEffect(() => {
    const t = setInterval(() => setOffset(o => (o + 1) % songs.length), 30000);
    return () => clearInterval(t);
  }, [songs.length]);

  // Reset offset when mood changes
  useEffect(() => { setOffset(0); }, [mood]);

  const visible = [
    songs[offset % songs.length],
    songs[(offset + 1) % songs.length],
    songs[(offset + 2) % songs.length],
  ];

  return (
    <div className="as w-full max-w-lg mx-auto mb-2">
      <div className="rounded-2xl overflow-hidden" style={{ background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.06)", backdropFilter: "blur(12px)" }}>
        <div className="px-4 py-2.5 flex items-center justify-between border-b border-white/[0.04]">
          <div className="flex items-center gap-2">
            <span className="text-sm">{meta.emoji}</span>
            <span className="text-[10px] text-white/40">
              Feeling <span className="font-semibold" style={{ color: meta.color }}>{mood}</span>? Try these
            </span>
          </div>
          <button onClick={onOpenSpotify} className="flex items-center gap-1 px-2 py-1 rounded-full text-[9px] font-semibold text-green-400 bg-green-500/10 border border-green-500/20 hover:bg-green-500/20 transition-colors cursor-pointer">
            <Music size={9} /> {spotifyConnected ? "Open Player" : "Connect Spotify"}
          </button>
        </div>
        <div className="px-3 py-2 space-y-1">
          {visible.map((song, i) => (
            <div key={`${song.title}-${i}`} onClick={onOpenSpotify} className="flex items-center gap-3 px-2 py-2 rounded-xl hover:bg-white/[0.04] transition-colors cursor-pointer group" style={{ animation: `fadeIn 0.4s ease-out ${i * 0.1}s both` }}>
              <div className="w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0" style={{ background: `${meta.color}12`, border: `1px solid ${meta.color}20` }}>
                <Play size={12} style={{ color: meta.color }} className="ml-0.5 opacity-50 group-hover:opacity-100 transition-opacity" />
              </div>
              <div className="flex-1 min-w-0">
                <p className="text-white text-[11px] font-medium truncate">{song.title} <span className="text-white/25">— {song.artist}</span></p>
                <p className="text-white/20 text-[9px] truncate">{song.reason}</p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

/* ══════════════════════════════════════════════════════
   PARTICLES
   ══════════════════════════════════════════════════════ */
function Particles() {
  const d = useRef(Array.from({ length: 14 }, (_, i) => ({ id: i, x: Math.random() * 100, y: Math.random() * 100, s: Math.random() * 2 + 0.8, d: Math.random() * 22 + 14, dl: Math.random() * -18, o: Math.random() * 0.2 + 0.05 }))).current;
  return (<div className="absolute inset-0 pointer-events-none overflow-hidden">{d.map(p => (<div key={p.id} className="absolute rounded-full bg-white" style={{ left: `${p.x}%`, top: `${p.y}%`, width: `${p.s}px`, height: `${p.s}px`, opacity: 0, animation: `pf ${p.d}s ease-in-out ${p.dl}s infinite`, ["--po" as string]: p.o }} />))}</div>);
}

/* ══════════════════════════════════════════════════════
   MAIN APP
   ══════════════════════════════════════════════════════ */
export default function MoodSpaceApp() {
  const [booted, setBooted] = useState(false);
  const [show, setShow] = useState(false);
  const [showMoodCam, setShowMoodCam] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [showIoT, setShowIoT] = useState(false);
  const [showMusic, setShowMusic] = useState(false);
  const [showHealth, setShowHealth] = useState(false);
  const [showAuth, setShowAuth] = useState(false);
  const [currentMood, setCurrentMood] = useState("relaxed");
  const [user, setUser] = useState<UserAccount | null>(null);
  const [isLoggedIn, setIsLoggedIn] = useState(false);

  // Check for Spotify OAuth callback (PKCE flow returns ?code= in URL)
  const [spotifyReady, setSpotifyReady] = useState(false);
  useEffect(() => {
    if (typeof window === "undefined") return;
    const params = new URLSearchParams(window.location.search);
    let code = params.get("code") || params.get("spotify_code");
    if (!code) {
      code = localStorage.getItem("spotify_auth_code");
      if (code) localStorage.removeItem("spotify_auth_code");
    }
    if (code) {
      exchangeSpotifyCode(code).then((success) => {
        if (success) setSpotifyReady(true);
        // Clean URL
        window.history.replaceState({}, document.title, window.location.pathname);
      });
    } else if (getSpotifyToken()) {
      setSpotifyReady(true);
    }
  }, []);

  // Check for existing session on mount
  useEffect(() => {
    const token = getToken();
    const saved = getSavedUser();
    if (token && saved) {
      setUser(saved); setIsLoggedIn(true);
      authFetch("/auth/me").then(res => {
        if (res.ok) return res.json();
        throw new Error("expired");
      }).then(data => { setUser(data.user); saveUser(data.user); }).catch(() => {});
    }
  }, []);

  const handleContinue = () => { setBooted(true); setTimeout(() => setShow(true), 500); };
  const handleMoodChange = useCallback((mood: string) => { setCurrentMood(mood); }, []);
  const handleAuth = (u: UserAccount) => { setUser(u); setIsLoggedIn(true); setShowAuth(false); };
  const handleLogout = () => { clearToken(); setUser(null); setIsLoggedIn(false); };
  const displayUser = user || defaultUser;

  return (
    <div className="w-full h-screen overflow-hidden" style={{ background: "#000" }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@200;300;400;600;700;800&display=swap');
        @keyframes pf{0%,100%{transform:translateY(0) translateX(0);opacity:0}10%{opacity:var(--po)}90%{opacity:var(--po)}50%{transform:translateY(-70px) translateX(25px)}}
        @keyframes fadeIn{from{opacity:0}to{opacity:1}}
        @keyframes slideDown{from{opacity:0;transform:translateY(-24px)}to{opacity:1;transform:translateY(0)}}
        @keyframes slideUp{from{opacity:0;transform:translateY(30px)}to{opacity:1;transform:translateY(0)}}
        @keyframes scaleUp{from{opacity:0;transform:scale(0.85)}to{opacity:1;transform:scale(1)}}
        @keyframes glowPulse{0%,100%{text-shadow:0 0 20px rgba(169,85,255,0.2),0 0 50px rgba(234,81,255,0.06)}50%{text-shadow:0 0 35px rgba(169,85,255,0.4),0 0 80px rgba(234,81,255,0.12)}}
        @keyframes breathe{0%,100%{opacity:0.4}50%{opacity:1}}
        @keyframes eqBar{0%{height:20%}100%{height:100%}}
        .pg{animation:fadeIn 1s ease-out both}
        .an{animation:slideDown .7s cubic-bezier(.16,1,.3,1) .3s both}
        .ah{animation:slideUp .9s cubic-bezier(.16,1,.3,1) .5s both}
        .as{animation:slideUp .9s cubic-bezier(.16,1,.3,1) .7s both}
        .am{animation:scaleUp .7s cubic-bezier(.16,1,.3,1) .9s both}
        .ac{animation:slideUp .7s cubic-bezier(.16,1,.3,1) 1.1s both}
        .gw{animation:glowPulse 4s ease-in-out infinite}
        .pd{animation:breathe 2.5s ease-in-out infinite}
        *{font-family:'Outfit',sans-serif}
      `}</style>

      {!booted && <BootScreen onContinue={handleContinue} />}
      {booted && (
        <div className={`relative w-full h-full ${show ? "pg" : "opacity-0"}`}>
          <div className="absolute inset-0 bg-black" />
          <ShaderAnimation />
          <Particles />
          <div className="absolute bottom-0 left-1/2 -translate-x-1/2 w-[500px] h-[160px] pointer-events-none" style={{ background: "radial-gradient(ellipse,rgba(169,85,255,0.08) 0%,transparent 70%)", filter: "blur(40px)" }} />

          <div className="relative z-10 flex flex-col h-full">
            {/* NAV — no Mitra AI */}
            <nav className="an flex items-center justify-between px-5 sm:px-10 py-4">
              <div className="flex items-center gap-3">
                <div className="w-9 h-9 rounded-xl flex items-center justify-center flex-shrink-0" style={{ background: "linear-gradient(135deg,#a955ff,#ea51ff)", boxShadow: "0 4px 20px rgba(169,85,255,0.3)" }}><Brain size={18} className="text-white" /></div>
                <h1 className="text-white font-bold text-base tracking-tight leading-none">MoodSpace</h1>
              </div>
              <div className="hidden lg:flex items-center gap-6">
                {["Features", "Spaces", "Community"].map(t => (<a key={t} href="#" className="text-[12px] text-white/35 hover:text-white transition-colors duration-300 tracking-wide">{t}</a>))}
              </div>
              <div className="flex items-center gap-2">
                <button onClick={() => setShowMoodCam(true)} className="inline-flex items-center gap-1.5 px-3 py-2 rounded-full text-[10px] font-semibold text-white uppercase tracking-wider transition-all hover:scale-105 cursor-pointer" style={{ background: "linear-gradient(135deg,#7c3aed,#a855f7)", boxShadow: "0 4px 12px rgba(124,58,237,0.3)" }}>
                  <Eye size={12} />Mood Cam
                </button>
                <button onClick={() => setShowIoT(true)} className="inline-flex items-center gap-1.5 px-3 py-2 rounded-full text-[10px] font-semibold text-white uppercase tracking-wider transition-all hover:scale-105 cursor-pointer border border-cyan-500/30 bg-cyan-500/10">
                  <Cpu size={12} className="text-cyan-400" />IoT
                </button>
                <button onClick={() => setShowMusic(!showMusic)} className="inline-flex items-center gap-1.5 px-3 py-2 rounded-full text-[10px] font-semibold text-white uppercase tracking-wider transition-all hover:scale-105 cursor-pointer border border-green-500/30 bg-green-500/10">
                  <Music size={12} className="text-green-400" />Spotify
                </button>
                <button onClick={() => setShowHealth(!showHealth)} className="inline-flex items-center gap-1.5 px-3 py-2 rounded-full text-[10px] font-semibold text-white uppercase tracking-wider transition-all hover:scale-105 cursor-pointer border border-emerald-500/30 bg-emerald-500/10">
                  <Heart size={12} className="text-emerald-400" />Health
                </button>
                {isLoggedIn ? (
                  <UserDD user={displayUser} onSettings={() => setShowSettings(true)} onLogout={handleLogout} />
                ) : (
                  <button onClick={() => setShowAuth(true)} className="inline-flex items-center gap-1.5 px-4 py-2 rounded-full text-[10px] font-semibold text-white uppercase tracking-wider transition-all hover:scale-105 cursor-pointer border border-white/20 bg-white/[0.06] hover:bg-white/10">
                    <User size={12} />Sign In
                  </button>
                )}
              </div>
            </nav>

            {/* HERO — no Mitra AI */}
            <div className="flex-1 flex flex-col items-center justify-center px-5 -mt-4">
              <div className="ah text-center max-w-3xl mx-auto mb-5">
                <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full border border-white/[0.06] bg-white/[0.03] backdrop-blur-sm mb-6">
                  <span className="w-1.5 h-1.5 rounded-full bg-green-400 pd" />
                  <span className="text-[10px] text-white/40 tracking-wide">AI-powered mood detection + adaptive environment</span>
                </div>
                <h2 className="text-5xl sm:text-7xl lg:text-8xl font-extrabold text-white tracking-tighter leading-[0.85] gw">
                  Mood<span className="bg-clip-text text-transparent" style={{ backgroundImage: "linear-gradient(135deg,#a955ff,#ea51ff,#56CCF2)" }}>Space</span>
                </h2>
              </div>
              <p className="as text-white/30 text-center text-sm max-w-md leading-relaxed mb-6">Navigate your emotional landscape with AI-powered insights. Capture moments, share feelings, discover patterns.</p>

              {/* Mood Music Recommendations on Homepage */}
              <MoodMusicRecs mood={currentMood} onOpenSpotify={() => setShowMusic(true)} />

              <div className="am mt-4"><GradMenu /></div>
              <div className="ac mt-8 flex items-center gap-4 flex-wrap justify-center">
                <button onClick={() => setShowMoodCam(true)} className="group px-7 py-3 rounded-full text-sm font-semibold text-white tracking-wide transition-all hover:scale-105 cursor-pointer" style={{ background: "linear-gradient(135deg,#a955ff,#ea51ff)", boxShadow: "0 8px 32px rgba(169,85,255,0.25)" }}>
                  <span className="flex items-center gap-2">Detect Mood<Eye size={15} /></span>
                </button>
                <button onClick={() => setShowIoT(true)} className="group px-7 py-3 rounded-full text-sm font-semibold text-white tracking-wide transition-all hover:scale-105 cursor-pointer border border-white/20 hover:border-white/40" style={{ background: "rgba(255,255,255,0.05)" }}>
                  <span className="flex items-center gap-2"><Cpu size={15} className="text-cyan-400" />IoT Sensors</span>
                </button>
              </div>
            </div>
          </div>

          {/* Overlays */}
          {showAuth && <AuthScreen onAuth={handleAuth} />}
          {showMoodCam && <LiveMoodOverlay visible={showMoodCam} onClose={() => setShowMoodCam(false)} onMoodChange={handleMoodChange} />}
          {showSettings && isLoggedIn && <AccountPanel user={displayUser} setUser={(u) => { setUser(u); saveUser(u); }} onClose={() => setShowSettings(false)} />}
          {showIoT && <IoTPanel visible={showIoT} onClose={() => setShowIoT(false)} />}
          <SpotifyOverlay mood={currentMood} visible={showMusic} onClose={() => setShowMusic(false)} />
          <HealthInsightsPanel mood={currentMood} visible={showHealth} onClose={() => setShowHealth(false)} />
        </div>
      )}
    </div>
  );
}
