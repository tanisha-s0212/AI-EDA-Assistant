'use client';

import { useSyncExternalStore } from 'react';
import { useThemeContext } from '@/lib/theme-context';
import { Moon, Sun, Monitor } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { motion, AnimatePresence } from 'framer-motion';

export function ThemeToggle() {
  const { theme, setTheme, mounted } = useThemeContext();
  const hydrated = useSyncExternalStore(
    () => () => {},
    () => true,
    () => false,
  );

  const themes = [
    { value: 'light' as const, icon: Sun, label: 'Light' },
    { value: 'dark' as const, icon: Moon, label: 'Dark' },
    { value: 'system' as const, icon: Monitor, label: 'System' },
  ];

  const currentIndex = themes.findIndex((t) => t.value === theme);
  const isReady = hydrated && mounted;
  const current = isReady ? (themes[currentIndex] || themes[0]) : themes[2];

  const cycleTheme = () => {
    const activeIndex = currentIndex >= 0 ? currentIndex : 2;
    const nextIndex = (activeIndex + 1) % themes.length;
    setTheme(themes[nextIndex].value);
  };

  return (
    <Button
      variant="ghost"
      size="icon"
      onClick={cycleTheme}
      className="relative h-9 w-9 rounded-lg"
      title={`Theme: ${current.label} (click to switch)`}
    >
      <AnimatePresence mode="wait">
        <motion.div
          key={current.value}
          initial={{ y: -10, opacity: 0, rotate: -90 }}
          animate={{ y: 0, opacity: 1, rotate: 0 }}
          exit={{ y: 10, opacity: 0, rotate: 90 }}
          transition={{ duration: 0.2 }}
          className="flex items-center justify-center"
        >
          <current.icon className="h-4 w-4" />
        </motion.div>
      </AnimatePresence>
    </Button>
  );
}
