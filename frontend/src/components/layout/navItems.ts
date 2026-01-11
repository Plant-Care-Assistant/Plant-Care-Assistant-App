import type { LucideIcon } from 'lucide-react';
import { Camera, Grid, Home, User } from 'lucide-react';

export type NavScreen = 'home' | 'scan' | 'collection' | 'profile';

export type NavItem = {
  screen: NavScreen;
  label: string;
  icon: LucideIcon;
};

export const navItems: NavItem[] = [
  { screen: 'home', label: 'Home', icon: Home },
  { screen: 'scan', label: 'Scan', icon: Camera },
  { screen: 'collection', label: 'Collection', icon: Grid },
  { screen: 'profile', label: 'Profile', icon: User },
];
