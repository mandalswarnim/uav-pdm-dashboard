'use client';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { useEffect, useState } from 'react';

const links = [
  { href: '/', label: 'CMD' },
  { href: '/armory', label: 'ARMORY' },
  { href: '/mission', label: 'MISSION' },
  { href: '/diagnostics', label: 'DIAGNOSTICS' },
  { href: '/lab', label: 'LAB' },
];

export default function TopNav() {
  const path = usePathname();
  const [time, setTime] = useState('--:--:--');

  useEffect(() => {
    const tick = () => setTime(new Date().toISOString().slice(11, 19) + 'Z');
    tick();
    const id = setInterval(tick, 1000);
    return () => clearInterval(id);
  }, []);

  return (
    <header className="relative z-20 border-b border-hud-cyan/30 bg-hud-bg/80 px-4 py-2">
      <div className="flex items-center gap-6">
        <div className="flex items-center gap-2">
          <div className="h-3 w-3 animate-pulseRed rounded-sm bg-hud-red" />
          <span className="text-sm font-bold uppercase tracking-[0.3em] text-hud-cyan glitch-text">
            ARES // PdM
          </span>
        </div>
        <nav className="flex items-center gap-1">
          {links.map((l) => {
            const active = l.href === '/' ? path === '/' : path?.startsWith(l.href);
            return (
              <Link
                key={l.href}
                href={l.href}
                className={`px-3 py-1 text-[11px] uppercase tracking-[0.25em] transition-all ${
                  active
                    ? 'border border-hud-cyan/70 bg-hud-cyan/10 text-hud-cyan shadow-glow'
                    : 'border border-transparent text-hud-dim hover:text-hud-cyan'
                }`}
              >
                {l.label}
              </Link>
            );
          })}
        </nav>
        <div className="ml-auto flex items-center gap-4 text-[11px] uppercase tracking-[0.25em] text-hud-dim">
          <span>OPS-CLOCK <span className="text-hud-cyan">{time}</span></span>
          <span>NET <span className="text-hud-green">●</span> SECURE</span>
          <span>SAT-LINK <span className="text-hud-green">●</span> 4/4</span>
        </div>
      </div>
    </header>
  );
}
