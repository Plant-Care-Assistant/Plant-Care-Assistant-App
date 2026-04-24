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

export const SCREEN_ROUTES: Record<NavScreen, string> = {
  home: '/',
  scan: '/scan',
  collection: '/collection',
  profile: '/profile',
};

export function screenFromPathname(pathname: string | null): NavScreen {
  if (!pathname) return 'home';
  if (pathname.startsWith('/scan')) return 'scan';
  if (pathname.startsWith('/plant') || pathname.startsWith('/collection')) return 'collection';
  if (pathname.startsWith('/profile')) return 'profile';
  return 'home';
}
