'use client';

import { createContext, useCallback, useContext, useEffect, useRef, useState, ReactNode } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import { Zap, Trophy } from 'lucide-react';

export type ToastKind = 'xp' | 'achievement';

export interface ToastItem {
  id: number;
  kind: ToastKind;
  title: string;
  subtitle?: string;
  amount?: number;
}

interface ToastContextType {
  showXpToast: (args: { amount: number; title: string; subtitle?: string }) => void;
  showAchievementToast: (args: { title: string; subtitle?: string }) => void;
}

const ToastContext = createContext<ToastContextType | undefined>(undefined);

const TOAST_DURATION_MS = 2800;

export function ToastProvider({ children }: { children: ReactNode }) {
  const [toasts, setToasts] = useState<ToastItem[]>([]);
  const nextId = useRef(1);
  const timers = useRef<Map<number, ReturnType<typeof setTimeout>>>(new Map());

  const dismiss = useCallback((id: number) => {
    const t = timers.current.get(id);
    if (t) {
      clearTimeout(t);
      timers.current.delete(id);
    }
    setToasts((prev) => prev.filter((toast) => toast.id !== id));
  }, []);

  const push = useCallback(
    (item: Omit<ToastItem, 'id'>) => {
      const id = nextId.current++;
      setToasts((prev) => [...prev, { ...item, id }]);
      const handle = setTimeout(() => dismiss(id), TOAST_DURATION_MS);
      timers.current.set(id, handle);
    },
    [dismiss],
  );

  useEffect(() => {
    const handles = timers.current;
    return () => {
      handles.forEach(clearTimeout);
      handles.clear();
    };
  }, []);

  const showXpToast = useCallback<ToastContextType['showXpToast']>(
    ({ amount, title, subtitle }) => push({ kind: 'xp', title, subtitle, amount }),
    [push],
  );

  const showAchievementToast = useCallback<ToastContextType['showAchievementToast']>(
    ({ title, subtitle }) => push({ kind: 'achievement', title, subtitle }),
    [push],
  );

  return (
    <ToastContext.Provider value={{ showXpToast, showAchievementToast }}>
      {children}
      <div className="fixed top-4 right-4 z-[9999] flex flex-col gap-2 pointer-events-none max-w-sm">
        <AnimatePresence initial={false}>
          {toasts.map((toast) => {
            const isXp = toast.kind === 'xp';
            return (
              <motion.div
                key={toast.id}
                layout
                initial={{ opacity: 0, y: -8, scale: 0.98 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                exit={{ opacity: 0, y: -8, scale: 0.98 }}
                transition={{ type: 'spring', stiffness: 380, damping: 32 }}
                onClick={() => dismiss(toast.id)}
                className="pointer-events-auto flex items-center gap-3 rounded-2xl px-4 py-3 cursor-pointer border shadow-sm bg-white border-neutral-200 dark:bg-neutral-800 dark:border-neutral-700"
              >
                <div
                  className={`flex-shrink-0 w-10 h-10 rounded-xl flex items-center justify-center ${
                    isXp ? 'bg-nature/15 text-nature' : 'bg-accent2/15 text-accent2'
                  }`}
                >
                  {isXp ? (
                    <Zap size={18} fill="currentColor" />
                  ) : (
                    <Trophy size={18} fill="currentColor" />
                  )}
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-baseline gap-2">
                    <span className="font-semibold text-sm text-neutral-900 dark:text-white">
                      {toast.title}
                    </span>
                    {typeof toast.amount === 'number' && (
                      <span className="font-bold text-sm text-nature">+{toast.amount} XP</span>
                    )}
                  </div>
                  {toast.subtitle && (
                    <p className="text-xs mt-0.5 truncate text-neutral-500 dark:text-neutral-400">
                      {toast.subtitle}
                    </p>
                  )}
                </div>
              </motion.div>
            );
          })}
        </AnimatePresence>
      </div>
    </ToastContext.Provider>
  );
}

export function useToast() {
  const ctx = useContext(ToastContext);
  if (!ctx) throw new Error('useToast must be used within a ToastProvider');
  return ctx;
}
