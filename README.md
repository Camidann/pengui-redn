<!DOCTYPE html>
<html>
<style>
  body { margin: 0; background: transparent; font-family: 'Courier New', monospace; }
  .wrap { background: #0d1117; border-radius: 12px; overflow: hidden; position: relative; width: 100%; }
  .stars { position: absolute; top: 0; left: 0; width: 100%; height: 100%; pointer-events: none; }
  .content { display: flex; align-items: center; padding: 36px 40px; gap: 32px; position: relative; z-index: 2; }
  .left { flex-shrink: 0; }
  .right { flex: 1; }
  .title { font-size: 40px; font-weight: 700; color: #fff; letter-spacing: -1px; line-height: 1; margin: 0 0 6px; }
  .title span { color: #3b82f6; }
  .sub { font-size: 12px; letter-spacing: 4px; color: #3b82f6; margin: 0 0 14px; text-transform: uppercase; }
  .desc { font-size: 13px; color: #8b949e; line-height: 1.6; margin: 0 0 20px; max-width: 380px; font-family: 'Courier New', monospace; }
  .tags { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 20px; }
  .tag { background: #161b22; border: 0.5px solid #30363d; border-radius: 20px; padding: 4px 12px; font-size: 11px; color: #8b949e; }
  .tag.rust { color: #f97316; border-color: #f97316; }
  .tag.ml { color: #a78bfa; border-color: #a78bfa; }
  .tag.oss { color: #3fb950; border-color: #3fb950; }
  .btns { display: flex; gap: 10px; }
  .btn { padding: 8px 20px; border-radius: 6px; font-size: 13px; font-family: 'Courier New', monospace; cursor: pointer; border: none; }
  .btn-primary { background: #2563eb; color: white; }
  .btn-secondary { background: transparent; color: #8b949e; border: 0.5px solid #30363d; }
  .footer-bar { background: #161b22; border-top: 0.5px solid #21262d; padding: 12px 40px; display: flex; gap: 28px; align-items: center; position: relative; z-index: 2; }
  .stat { display: flex; align-items: center; gap: 6px; font-size: 12px; color: #8b949e; }
  .stat-dot { width: 8px; height: 8px; border-radius: 50%; }
</style>
<div class="wrap">
  <svg class="stars" viewBox="0 0 680 260" xmlns="http://www.w3.org/2000/svg">
    <!-- background glow -->
    <radialGradient id="glow1" cx="50%" cy="50%" r="50%">
      <stop offset="0%" stop-color="#1d4ed8" stop-opacity="0.08"/>
      <stop offset="100%" stop-color="#1d4ed8" stop-opacity="0"/>
    </radialGradient>
    <ellipse cx="340" cy="130" rx="300" ry="130" fill="url(#glow1)"/>
    <!-- stars -->
    <circle cx="55" cy="22" r="1.2" fill="#fff" opacity="0.35"/>
    <circle cx="130" cy="48" r="0.8" fill="#fff" opacity="0.2"/>
    <circle cx="210" cy="18" r="1" fill="#fff" opacity="0.28"/>
    <circle cx="350" cy="30" r="1.2" fill="#fff" opacity="0.22"/>
    <circle cx="440" cy="12" r="0.8" fill="#fff" opacity="0.3"/>
    <circle cx="520" cy="45" r="1" fill="#fff" opacity="0.2"/>
    <circle cx="610" cy="25" r="1.2" fill="#fff" opacity="0.28"/>
    <circle cx="655" cy="70" r="0.8" fill="#fff" opacity="0.2"/>
    <circle cx="30" cy="100" r="1" fill="#fff" opacity="0.18"/>
    <circle cx="645" cy="190" r="1" fill="#fff" opacity="0.2"/>
    <!-- neural net background right -->
    <g opacity="0.1">
      <circle cx="560" cy="80" r="5" fill="none" stroke="#60a5fa" stroke-width="1"/>
      <circle cx="600" cy="60" r="5" fill="none" stroke="#60a5fa" stroke-width="1"/>
      <circle cx="600" cy="100" r="5" fill="none" stroke="#60a5fa" stroke-width="1"/>
      <circle cx="640" cy="80" r="5" fill="none" stroke="#60a5fa" stroke-width="1"/>
      <line x1="565" y1="80" x2="595" y2="62" stroke="#60a5fa" stroke-width="0.7"/>
      <line x1="565" y1="80" x2="595" y2="98" stroke="#60a5fa" stroke-width="0.7"/>
      <line x1="605" y1="62" x2="635" y2="79" stroke="#60a5fa" stroke-width="0.7"/>
      <line x1="605" y1="98" x2="635" y2="81" stroke="#60a5fa" stroke-width="0.7"/>
    </g>
    <g opacity="0.07">
      <circle cx="40" cy="190" r="5" fill="none" stroke="#60a5fa" stroke-width="1"/>
      <circle cx="75" cy="175" r="5" fill="none" stroke="#60a5fa" stroke-width="1"/>
      <circle cx="75" cy="205" r="5" fill="none" stroke="#60a5fa" stroke-width="1"/>
      <line x1="45" y1="190" x2="70" y2="177" stroke="#60a5fa" stroke-width="0.7"/>
      <line x1="45" y1="190" x2="70" y2="203" stroke="#60a5fa" stroke-width="0.7"/>
    </g>
  </svg>

  <div class="content">
    <div class="left">
      <!-- Penguin SVG -->
      <svg width="140" height="170" viewBox="0 0 140 170" xmlns="http://www.w3.org/2000/svg">
        <!-- Shadow -->
        <ellipse cx="70" cy="163" rx="38" ry="7" fill="#000" opacity="0.3"/>
        <!-- Body -->
        <polygon points="35,95 45,75 70,68 95,75 105,95 100,130 70,145 40,130" fill="#f0f4ff"/>
        <polygon points="42,90 48,73 70,67 92,73 98,90 94,125 70,138 46,125" fill="#fff"/>
        <!-- Blue outer body sides -->
        <polygon points="20,90 42,80 42,130 25,138" fill="#1a1fd4"/>
        <polygon points="120,90 98,80 98,130 115,138" fill="#1a1fd4"/>
        <!-- Head -->
        <polygon points="40,72 50,45 70,38 90,45 100,72 90,85 70,90 50,85" fill="#2525e8"/>
        <polygon points="45,68 54,46 70,40 86,46 95,68 87,80 70,85 53,80" fill="#3030f0"/>
        <!-- Shine on head -->
        <ellipse cx="62" cy="52" rx="8" ry="5" fill="#4444ff" opacity="0.4"/>
        <!-- Eyes -->
        <circle cx="54" cy="62" r="7" fill="white"/>
        <circle cx="86" cy="62" r="8" fill="white"/>
        <circle cx="54" cy="62" r="4" fill="#1a1fd4"/>
        <circle cx="86" cy="62" r="5" fill="#1a1fd4"/>
        <circle cx="53" cy="61" r="2.5" fill="#111"/>
        <circle cx="85" cy="60" r="3" fill="#111"/>
        <circle cx="51" cy="59" r="1" fill="white"/>
        <circle cx="83" cy="58" r="1.2" fill="white"/>
        <!-- Beak -->
        <polygon points="58,78 70,73 82,78 76,88 70,91 64,88" fill="#f0a800"/>
        <polygon points="62,78 70,74 78,78 73,85 70,88 67,85" fill="#f5bc20"/>
        <!-- Smile line -->
        <path d="M63 87 Q70 92 77 87" fill="none" stroke="#d08000" stroke-width="1" stroke-linecap="round"/>
        <!-- Wings -->
        <polygon  fill="#1a1fd4"/>
        <path d="M108,87 Q122,95 124,115 Q125,130 116,132 Q110,133 106,128 Q104,118 106,100 Z" fill="#1a1fd4"/>
        <path d="M32,90 Q21,98 19,115 Q18,127 25,129 Q31,130 33,124 Q35,114 33,98 Z" fill="#2222cc"/>
        <!-- Feet -->
        <polygon points="45,138 62,135 66,150 58,158 40,155 36,145" fill="#f0a800"/>
        <polygon points="75,135 92,133 96,148 88,157 72,154 69,144" fill="#f0a800"/>
        <polygon points="47,140 60,138 63,150 56,156 43,153 40,146" fill="#f5bc20"/>
        <polygon points="77,137 90,135 93,147 87,154 75,151 72,144" fill="#f5bc20"/>
      </svg>
    </div>

    <div class="right">
      <p class="sub">Neural Network Generator</p>
      <h1 class="title">pengui<span>-redn</span></h1>
      <p class="desc">Generador de arquitecturas de redes neuronales escrito en Rust. Crea, entrená y exportá modelos con una API simple y eficiente.</p>
      <div class="tags">
        <span class="tag rust">🦀 Rust</span>
        <span class="tag ml">⚡ neural nets</span>
        <span class="tag oss">✦ open source</span>
        <span class="tag">v0.1.0</span>
      </div>
      <div class="btns">
        <button class="btn btn-primary">cargo add pengui-redn</button>
      </div>
    </div>
  </div>

  <div class="footer-bar">
    <div class="stat">
      <div class="stat-dot" style="background:#f97316"></div>
      Rust 100%
    </div>
    <div class="stat" style="margin-left:auto">
      <svg width="14" height="14" viewBox="0 0 16 16" fill="#8b949e"><path d="M8 9.5a1.5 1.5 0 1 0 0-3 1.5 1.5 0 0 0 0 3Z"/><path d="M8 0a8 8 0 1 1 0 16A8 8 0 0 1 8 0ZM1.5 8a6.5 6.5 0 1 0 13 0 6.5 6.5 0 0 0-13 0Z"/></svg>
      Camidann
    </div>
  </div>
</div>

