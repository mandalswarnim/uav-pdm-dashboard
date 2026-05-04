import type { Config } from 'tailwindcss';

const config: Config = {
  content: ['./app/**/*.{ts,tsx}', './components/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        hud: {
          bg: '#02060d',
          panel: '#06101c',
          cyan: '#00e5ff',
          blue: '#1e90ff',
          amber: '#ffb000',
          red: '#ff2a2a',
          green: '#00ff9c',
          dim: '#6ea8c4',
        },
      },
      fontFamily: {
        mono: ['ui-monospace', 'SFMono-Regular', 'Menlo', 'monospace'],
      },
      boxShadow: {
        glow: '0 0 12px rgba(0,229,255,0.45), inset 0 0 12px rgba(0,229,255,0.15)',
        glowAmber: '0 0 14px rgba(255,176,0,0.55), inset 0 0 12px rgba(255,176,0,0.2)',
        glowRed: '0 0 18px rgba(255,42,42,0.7), inset 0 0 12px rgba(255,42,42,0.25)',
      },
      animation: {
        scan: 'scan 6s linear infinite',
        pulseRed: 'pulseRed 1.4s ease-in-out infinite',
        flicker: 'flicker 3.2s linear infinite',
      },
      keyframes: {
        scan: {
          '0%': { transform: 'translateY(-100%)' },
          '100%': { transform: 'translateY(100%)' },
        },
        pulseRed: {
          '0%,100%': { boxShadow: '0 0 8px rgba(255,42,42,0.5)' },
          '50%': { boxShadow: '0 0 22px rgba(255,42,42,0.95)' },
        },
        flicker: {
          '0%,98%,100%': { opacity: '1' },
          '99%': { opacity: '0.6' },
        },
      },
    },
  },
  plugins: [],
};
export default config;
