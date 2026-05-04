'use client';
import { Suspense, useState } from 'react';
import SceneEnv from '@/components/3D/SceneEnv';
import UAVModel from '@/components/3D/UAVModel';
import { useDash } from '@/lib/store';
import type { Asset } from '@/lib/assets';
import { Html } from '@react-three/drei';

export default function AssetCarousel() {
  const { assets, selectedId, select } = useDash();
  const radius = 6;
  const [hoverId, setHoverId] = useState<string | null>(null);

  return (
    <SceneEnv cameraPos={[0, 4, 12]} autoRotate gizmo>
      <Suspense fallback={null}>
        {/* Holo platform */}
        <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -1.3, 0]}>
          <ringGeometry args={[radius - 0.6, radius + 0.6, 64]} />
          <meshBasicMaterial color="#00e5ff" transparent opacity={0.15} />
        </mesh>
        <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -1.31, 0]}>
          <ringGeometry args={[radius + 0.6, radius + 0.7, 64]} />
          <meshBasicMaterial color="#00e5ff" />
        </mesh>

        {assets.map((a, i) => {
          const angle = (i / assets.length) * Math.PI * 2;
          const x = Math.cos(angle) * radius;
          const z = Math.sin(angle) * radius;
          const isSel = a.id === selectedId;
          return (
            <group key={a.id} position={[x, 0, z]}>
              <group
                onClick={(e) => { e.stopPropagation(); select(a.id); }}
                onPointerOver={(e) => { e.stopPropagation(); setHoverId(a.id); document.body.style.cursor = 'pointer'; }}
                onPointerOut={() => { setHoverId(null); document.body.style.cursor = 'auto'; }}
              >
                <UAVModel asset={a} spin scale={isSel ? 0.95 : 0.7} />
              </group>
              {/* Pulsing red aura for critical */}
              {a.status === 'CRITICAL' && (
                <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -1.15, 0]}>
                  <ringGeometry args={[1.2, 1.5, 32]} />
                  <meshBasicMaterial color="#ff2a2a" transparent opacity={0.55} />
                </mesh>
              )}
              {(hoverId === a.id || isSel) && (
                <Html position={[0, 1.6, 0]} center distanceFactor={8} style={{ pointerEvents: 'none' }}>
                  <Tooltip a={a} selected={isSel} />
                </Html>
              )}
            </group>
          );
        })}
      </Suspense>
    </SceneEnv>
  );
}

function Tooltip({ a, selected }: { a: Asset; selected: boolean }) {
  const color = a.status === 'CRITICAL' ? '#ff2a2a' : a.status === 'WARNING' ? '#ffb000' : '#00e5ff';
  return (
    <div
      style={{ borderColor: color, color, boxShadow: `0 0 12px ${color}` }}
      className="min-w-[160px] border bg-black/85 px-3 py-2 font-mono text-[11px] uppercase tracking-[0.18em]"
    >
      <div className="font-bold">{a.name}</div>
      <div className="text-[10px] opacity-80">{a.id} · {a.class}</div>
      <div className="mt-1">HEALTH: <span className="font-bold">{a.rul.toFixed(0)}%</span></div>
      <div className="text-[10px] opacity-80">{a.data_source}</div>
      {selected && <div className="mt-1 text-[10px] opacity-80">[SELECTED]</div>}
    </div>
  );
}
