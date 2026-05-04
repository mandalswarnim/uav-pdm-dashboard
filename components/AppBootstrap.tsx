'use client';
import { useEffect } from 'react';
import { useDash } from '@/lib/store';

/** Hydrates the fleet roster on first render. Mounted once in the root layout. */
export default function AppBootstrap() {
  const loadFleet = useDash((s) => s.loadFleet);
  useEffect(() => { loadFleet(); }, [loadFleet]);
  return null;
}
