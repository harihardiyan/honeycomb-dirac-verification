<svg width="460" height="140" viewBox="0 0 460 140" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="grad-bg" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#020617"/>
      <stop offset="100%" stop-color="#020617"/>
    </linearGradient>
    <linearGradient id="grad-hex" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#38bdf8"/>
      <stop offset="100%" stop-color="#a855f7"/>
    </linearGradient>
  </defs>

  <!-- Background -->
  <rect x="0" y="0" width="460" height="140" fill="url(#grad-bg)"/>

  <!-- Honeycomb / k-space motif -->
  <g transform="translate(80,70) scale(1.1)">
    <polygon points="0,-28 24,-14 24,14 0,28 -24,14 -24,-14"
             fill="none" stroke="url(#grad-hex)" stroke-width="2.2"/>
    <circle cx="0" cy="-14" r="3.2" fill="#f97316"/>
    <circle cx="0" cy="14" r="3.2" fill="#22c55e"/>
    <circle cx="24" cy="0" r="2.6" fill="#38bdf8"/>
    <circle cx="-24" cy="0" r="2.6" fill="#38bdf8"/>
    <circle cx="12" cy="21" r="2.4" fill="#a855f7"/>
    <circle cx="-12" cy="-21" r="2.4" fill="#a855f7"/>
  </g>

  <!-- Title -->
  <text x="150" y="60" fill="#e5e7eb"
        font-family="SF Mono, Menlo, Consolas, monospace"
        font-size="18" letter-spacing="0.08em">
    Graphene Massive Dirac Auditor
  </text>

  <!-- Subtitle -->
  <text x="150" y="84" fill="#9ca3af"
        font-family="SF Mono, Menlo, Consolas, monospace"
        font-size="11" letter-spacing="0.20em">
    SYMMETRY • BERRY PHASE • SCALING • JAX
  </text>
</svg>
