export const theme = {
  colors: {
    background: '#fff8f0',
    surface: '#ffffff',
    surfaceAlt: '#fff1e0',
    text: '#2e1a0a',
    muted: '#8c6f5a',
    success: '#1ca86a',
    neon: '#ff7a1a',
    neonAlt: '#c25100',
    danger: '#ff4f4f',
  },
  fonts: {
    heading: '"Space Grotesk", "Inter", sans-serif',
    body: '"Inter", "SF Pro Display", sans-serif',
  },
  radii: {
    md: '16px',
    lg: '32px',
  },
  shadows: {
    glow: '0 22px 45px rgba(194, 81, 0, 0.18)',
  },
  spacing: (factor: number) => `${factor * 8}px`,
  durations: {
    quick: '180ms',
    normal: '360ms',
    slow: '580ms',
  },
};

export type Theme = typeof theme;
