import './globals.css';
import type { Metadata } from 'next';
import TopNav from '@/components/HUD/TopNav';
import AppBootstrap from '@/components/AppBootstrap';

export const metadata: Metadata = {
  title: 'TACTICAL // UAV PdM Digital Twin',
  description: 'AI-driven Predictive Maintenance dashboard for tactical UAVs and missiles.',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen">
        <div className="relative z-10 flex h-screen flex-col">
          <AppBootstrap />
          <TopNav />
          <main className="relative flex-1 overflow-hidden">{children}</main>
          <footer className="border-t border-hud-cyan/30 bg-hud-bg/80 px-4 py-1 text-[10px] uppercase tracking-[0.3em] text-hud-dim">
            <div className="flex justify-between">
              <span>// SYS::ONLINE</span>
              <span>CLASSIFIED // FOR DEMONSTRATION ONLY</span>
              <span>BUILD 0.1.0-ALPHA</span>
            </div>
          </footer>
        </div>
      </body>
    </html>
  );
}
