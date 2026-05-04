export type AssetClass = 'UAV-RECON' | 'UAV-STRIKE' | 'UAV-SWARM' | 'MISSILE-AGM' | 'MISSILE-CRUISE';

export interface Asset {
  id: string;
  name: string;
  class: AssetClass;
  rul: number;            // 0..100 (% of useful life remaining)
  status: 'NOMINAL' | 'WARNING' | 'CRITICAL';
  anomaly?: {
    component: string;
    note: string;
    severity: number;
    top_sensor_idx?: number;
    predicted_fault?: string;
    true_fault?: string;
  } | null;
  data_source?: 'C-MAPSS' | 'UAV-Synth';
}

export const rulToStatus = (r: number): Asset['status'] =>
  r < 35 ? 'CRITICAL' : r < 65 ? 'WARNING' : 'NOMINAL';
