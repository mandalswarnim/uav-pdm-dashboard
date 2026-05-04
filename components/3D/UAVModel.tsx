'use client';
import { useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import type { Asset } from '@/lib/assets';

const STATUS_COLOR: Record<Asset['status'], string> = {
  NOMINAL: '#00e5ff',
  WARNING: '#ffb000',
  CRITICAL: '#ff2a2a',
};

interface Props {
  asset: Asset;
  spin?: boolean;
  wireframe?: boolean;
  highlightComponent?: string | null;
  scale?: number;
}

/**
 * Procedural UAV / missile geometry. Different classes get different silhouettes.
 * No external GLTF assets required.
 */
export default function UAVModel({
  asset,
  spin = true,
  wireframe = false,
  highlightComponent = null,
  scale = 1,
}: Props) {
  const ref = useRef<THREE.Group>(null);
  const color = STATUS_COLOR[asset.status];

  useFrame((_, dt) => {
    if (spin && ref.current) ref.current.rotation.y += dt * 0.4;
  });

  const matProps = {
    color,
    emissive: color,
    emissiveIntensity: wireframe ? 0.6 : 0.35,
    wireframe,
    transparent: true,
    opacity: wireframe ? 0.85 : 1,
  };

  const isMissile = asset.class.startsWith('MISSILE');

  return (
    <group ref={ref} scale={scale}>
      {isMissile ? <MissileBody color={color} matProps={matProps} highlight={highlightComponent} />
                 : <DroneBody color={color} matProps={matProps} highlight={highlightComponent} />}
    </group>
  );
}

function DroneBody({
  color, matProps, highlight,
}: { color: string; matProps: any; highlight: string | null }) {
  const rotorRefs = [useRef<THREE.Mesh>(null), useRef<THREE.Mesh>(null), useRef<THREE.Mesh>(null), useRef<THREE.Mesh>(null)];
  useFrame((_, dt) => rotorRefs.forEach((r) => r.current && (r.current.rotation.y += dt * 12)));

  const isHL = (name: string) => highlight && highlight.toLowerCase().includes(name);

  return (
    <group>
      {/* Fuselage */}
      <mesh>
        <capsuleGeometry args={[0.4, 1.6, 6, 16]} />
        <meshStandardMaterial {...matProps} />
      </mesh>
      {/* Wings */}
      <mesh position={[0, 0, 0]} rotation={[0, 0, Math.PI / 2]}>
        <boxGeometry args={[0.05, 4.2, 0.5]} />
        <meshStandardMaterial {...matProps} />
      </mesh>
      {/* Tail */}
      <mesh position={[0, 0.4, -1.05]}>
        <boxGeometry args={[0.05, 0.7, 0.4]} />
        <meshStandardMaterial {...matProps} />
      </mesh>
      <mesh position={[0, 0, -1.1]} rotation={[0, 0, Math.PI / 2]}>
        <boxGeometry args={[0.05, 1.2, 0.3]} />
        <meshStandardMaterial {...matProps} />
      </mesh>
      {/* Rotor pylons + spinning rotors (colored red on the highlighted one) */}
      {[
        { pos: [-1.6, 0.05, 0.6] as const, name: 'rotor 1' },
        { pos: [ 1.6, 0.05, 0.6] as const, name: 'rotor 2' },
        { pos: [-1.6, 0.05,-0.6] as const, name: 'rotor 3' },
        { pos: [ 1.6, 0.05,-0.6] as const, name: 'rotor 4' },
      ].map((r, i) => {
        const hl = isHL(r.name);
        return (
          <group key={i} position={r.pos as unknown as [number, number, number]}>
            <mesh>
              <cylinderGeometry args={[0.08, 0.08, 0.25, 12]} />
              <meshStandardMaterial
                color={hl ? '#ff2a2a' : color}
                emissive={hl ? '#ff2a2a' : color}
                emissiveIntensity={hl ? 1.2 : 0.4}
                wireframe={matProps.wireframe}
              />
            </mesh>
            <mesh ref={rotorRefs[i]} position={[0, 0.18, 0]}>
              <boxGeometry args={[1.1, 0.02, 0.08]} />
              <meshStandardMaterial
                color={hl ? '#ff2a2a' : color}
                emissive={hl ? '#ff2a2a' : color}
                emissiveIntensity={hl ? 1.5 : 0.5}
                transparent opacity={0.85}
              />
            </mesh>
            {hl && (
              <pointLight color="#ff2a2a" intensity={2} distance={3} />
            )}
          </group>
        );
      })}
      {/* Nose camera ball */}
      <mesh position={[0, -0.15, 1.05]}>
        <sphereGeometry args={[0.2, 16, 16]} />
        <meshStandardMaterial color="#031019" emissive="#00e5ff" emissiveIntensity={0.4} />
      </mesh>
    </group>
  );
}

function MissileBody({
  color, matProps, highlight,
}: { color: string; matProps: any; highlight: string | null }) {
  const isHL = (name: string) => highlight && highlight.toLowerCase().includes(name);
  const gyroHL = isHL('gyro') || isHL('imu');
  return (
    <group rotation={[0, 0, Math.PI / 2]}>
      {/* Body */}
      <mesh>
        <cylinderGeometry args={[0.3, 0.3, 3, 24]} />
        <meshStandardMaterial {...matProps} />
      </mesh>
      {/* Nose cone */}
      <mesh position={[0, 1.7, 0]}>
        <coneGeometry args={[0.3, 0.6, 24]} />
        <meshStandardMaterial {...matProps} />
      </mesh>
      {/* Tail fins */}
      {[0, 1, 2, 3].map((i) => (
        <mesh key={i} position={[0, -1.4, 0]} rotation={[0, (i * Math.PI) / 2, 0]}>
          <boxGeometry args={[0.05, 0.5, 0.6]} />
          <meshStandardMaterial {...matProps} />
        </mesh>
      ))}
      {/* IMU bay (highlight if anomaly) */}
      <mesh position={[0, 0.4, 0]}>
        <torusGeometry args={[0.32, 0.04, 8, 32]} />
        <meshStandardMaterial
          color={gyroHL ? '#ff2a2a' : color}
          emissive={gyroHL ? '#ff2a2a' : color}
          emissiveIntensity={gyroHL ? 1.4 : 0.5}
        />
      </mesh>
      {gyroHL && <pointLight color="#ff2a2a" intensity={2} distance={3} position={[0, 0.4, 0]} />}
      {/* Exhaust */}
      <mesh position={[0, -1.7, 0]}>
        <cylinderGeometry args={[0.2, 0.28, 0.25, 16]} />
        <meshStandardMaterial color="#031019" emissive="#1e90ff" emissiveIntensity={0.6} />
      </mesh>
    </group>
  );
}
