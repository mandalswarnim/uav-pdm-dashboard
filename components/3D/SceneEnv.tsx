'use client';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Stars, GizmoHelper, GizmoViewport } from '@react-three/drei';
import { ReactNode } from 'react';

interface Props {
  children: ReactNode;
  cameraPos?: [number, number, number];
  controls?: boolean;
  gizmo?: boolean;
  autoRotate?: boolean;
}

export default function SceneEnv({
  children,
  cameraPos = [4, 2.5, 6],
  controls = true,
  gizmo = false,
  autoRotate = false,
}: Props) {
  return (
    <Canvas camera={{ position: cameraPos, fov: 45 }} dpr={[1, 2]}>
      <color attach="background" args={['#02060d']} />
      <fog attach="fog" args={['#02060d', 8, 28]} />
      <ambientLight intensity={0.25} />
      <directionalLight position={[6, 8, 4]} intensity={0.6} color="#9fd6ff" />
      <pointLight position={[-6, -2, -4]} intensity={0.5} color="#1e90ff" />
      <Stars radius={60} depth={40} count={2000} factor={2} fade speed={0.4} />
      {children}
      {controls && <OrbitControls enablePan={false} enableZoom autoRotate={autoRotate} autoRotateSpeed={0.6} />}
      {gizmo && (
        <GizmoHelper alignment="bottom-right" margin={[60, 60]}>
          <GizmoViewport axisColors={['#ff2a2a', '#00ff9c', '#00e5ff']} labelColor="white" />
        </GizmoHelper>
      )}
    </Canvas>
  );
}
