import { FilterOption } from '@/components/features/collection/CollectionFilters';

export interface Plant {
  id: string;
  name: string;
  species?: string;
  lastWatered?: string;
  nextWatering?: string;
  lightLevel?: 'low' | 'medium' | 'high';
  health: 'healthy' | 'needs-attention' | 'critical';
  imageUrl?: string;
  location?: string;
  wateringFrequency?: number;
  aiIdentified?: boolean;
  confidence?: number;
}

/**
 * Mock plants data for development and testing
 * Replace with actual API data in production
 */
export const MOCK_PLANTS: Plant[] = [
  {
    id: '1',
    name: 'Monstera Deliciosa',
    species: 'Monstera deliciosa',
    lastWatered: '2 days ago',
    health: 'healthy',
    lightLevel: 'medium',
    imageUrl: '/1.jpg',
  },
  {
    id: '2',
    name: 'Snake Plant',
    species: 'Sansevieria trifasciata',
    lastWatered: '1 week ago',
    health: 'healthy',
    lightLevel: 'low',
    imageUrl: '/2.jpg',
  },
  {
    id: '3',
    name: 'Pothos',
    species: 'Epipremnum aureum',
    lastWatered: '3 days ago',
    health: 'needs-attention',
    lightLevel: 'medium',
    imageUrl: '/3.jpg',
  },
  {
    id: '4',
    name: 'Fiddle Leaf Fig',
    species: 'Ficus lyrata',
    lastWatered: '4 days ago',
    health: 'healthy',
    lightLevel: 'high',
    imageUrl: '/4.jpg',
  },
  {
    id: '5',
    name: 'Peace Lily',
    species: 'Spathiphyllum',
    lastWatered: '5 days ago',
    health: 'needs-attention',
    lightLevel: 'low',
    imageUrl: '/5.jpg',
  },
  {
    id: '6',
    name: 'Spider Plant',
    species: 'Chlorophytum comosum',
    lastWatered: '3 days ago',
    health: 'healthy',
    lightLevel: 'medium',
    imageUrl: '/6.jpg',
  },
  {
    id: '7',
    name: 'Rubber Plant',
    species: 'Ficus elastica',
    lastWatered: '6 days ago',
    health: 'critical',
    lightLevel: 'high',
    imageUrl: '/7.jpg',
  },
  {
    id: '8',
    name: 'Aloe Vera',
    species: 'Aloe barbadensis',
    lastWatered: '1 week ago',
    health: 'healthy',
    lightLevel: 'high',
    imageUrl: '/8.jpg',
  },
];

/**
 * Filters and searches plants based on the active filter and search query
 */
export function filterAndSearchPlants(
  plants: Plant[],
  activeFilter: FilterOption,
  searchQuery: string
): Plant[] {
  let filtered = [...plants];

  // Apply filter
  if (activeFilter === 'healthy') {
    filtered = filtered.filter(p => p.health === 'healthy');
  } else if (activeFilter === 'need-care') {
    filtered = filtered.filter(p => p.health === 'needs-attention' || p.health === 'critical');
  } else if (activeFilter === 'recent') {
    // Sort by most recently watered (for demo, just reverse the array)
    // In production, you'd sort by actual date
    filtered = filtered.reverse();
  }

  // Apply search
  if (searchQuery.trim()) {
    const query = searchQuery.toLowerCase();
    filtered = filtered.filter(p => 
      p.name.toLowerCase().includes(query) || 
      p.species?.toLowerCase().includes(query)
    );
  }

  return filtered;
}

/**
 * Filters plants by health status
 */
export function filterPlantsByHealth(
  plants: Plant[],
  health: 'healthy' | 'needs-attention' | 'critical'
): Plant[] {
  return plants.filter(p => p.health === health);
}

/**
 * Searches plants by name or species
 */
export function searchPlants(plants: Plant[], query: string): Plant[] {
  if (!query.trim()) return plants;
  
  const searchTerm = query.toLowerCase();
  return plants.filter(p => 
    p.name.toLowerCase().includes(searchTerm) || 
    p.species?.toLowerCase().includes(searchTerm)
  );
}

/**
 * Sorts plants by last watered date (most recent first)
 * Note: This is a placeholder - implement actual date sorting in production
 */
export function sortPlantsByRecent(plants: Plant[]): Plant[] {
  return [...plants].reverse();
}
