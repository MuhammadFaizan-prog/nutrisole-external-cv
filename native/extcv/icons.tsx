// Icons used by the External-CV scan screens (imported directly so the shared app icon set stays unchanged).
import { ChevronDown, ChevronRight, CircleAlert, Cpu, Info, Pause, Play, Scan, ScanSearch, Settings, ShoppingCart, X } from 'lucide-react-native';
const set = { ChevronDown, ChevronRight, CircleAlert, Cpu, Info, Pause, Play, Scan, ScanSearch, Settings, ShoppingCart, X };
export type ExtIconName = keyof typeof set;
export default function ExtIcon({ name, size = 24, color = '#123f2c' }: { name: ExtIconName; size?: number; color?: string }) {
  const C = set[name];
  return <C size={size} color={color} strokeWidth={1.8} />;
}
