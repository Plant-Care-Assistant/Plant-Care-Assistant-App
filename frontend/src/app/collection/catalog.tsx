import { PlantCatalogScreen } from '@/components/screens/PlantCatalogScreen';
import { useTheme } from '@/providers';

export default function CatalogPage() {
  const { theme } = useTheme();
  return <PlantCatalogScreen darkMode={theme === 'dark'} />;
}
