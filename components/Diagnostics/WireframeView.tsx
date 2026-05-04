'use client';
import { Suspense } from 'react';
import SceneEnv from '@/components/3D/SceneEnv';
import UAVModel from '@/components/3D/UAVModel';
import { Html } from '@react-three/drei';
import type { Asset } from '@/lib/assets';

export default function WireframeView({ asset }: { asset: Asset }) {
  return (
    <SceneEnv cameraPos={[0, 1.5, 6]} gizmo>
      <Suspense fallback={null}>
        {/* Solid base */}
        <UAVModel asset={asset} spin={false} highlightComponent={asset.anomaly?.component ?? null} scale={1} />
        {/* Wireframe ghost overlay */}
        <UAVModel
          asset={asset}
          spin={false}
          wireframe
          highlightComponent={asset.anomaly?.component ?? null}
          scale={1.02}
        />
        {asset.anomaly && (
          <Html position={[1.6, 1.2, -0.6]} distanceFactor={6} style={{ pointerEvents: 'none' }}>
            <div className="border border-hud-red/80 bg-black/85 px-2 py-1 font-mono text-[10px] uppercase tracking-[0.18em] text-hud-red shadow-glowRed">
              ⚠ {asset.anomaly.component}
              <div className="text-[9px] opacity-80">SEV {(asset.anomaly.severity * 100).toFixed(0)}%</div>
            </div>
          </Html>
        )}
        {/* Ground plane */}
        <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -1.6, 0]}>
          <planeGeometry args={[20, 20, 20, 20]} />
          <meshBasicMaterial color="#00e5ff" wireframe transparent opacity={0.12} />
        </mesh>
      </Suspense>
    </SceneEnv>
  );
}
