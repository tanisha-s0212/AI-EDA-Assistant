'use client';

import React from 'react';
import { Coins } from 'lucide-react';
import { cn } from '@/lib/utils';
import type { CurrencyCode } from '@/lib/currency-format';
import { useAppStore } from '@/lib/store';

const OPTIONS: CurrencyCode[] = ['USD', 'INR'];

export default function CurrencySelector({ className }: { className?: string }) {
  const displayCurrency = useAppStore((state) => state.displayCurrency);
  const setDisplayCurrency = useAppStore((state) => state.setDisplayCurrency);

  return (
    <div className={cn('flex items-center gap-2 rounded-full border border-border/70 bg-background/90 px-2 py-1.5 text-xs', className)}>
      <Coins className="h-3.5 w-3.5 text-muted-foreground" />
      <span className="font-medium text-muted-foreground">Currency</span>
      <div className="flex items-center gap-1">
        {OPTIONS.map((option) => (
          <button
            key={option}
            type="button"
            onClick={() => setDisplayCurrency(option)}
            className={cn(
              'rounded-full px-2.5 py-1 font-semibold transition-colors',
              displayCurrency === option
                ? 'bg-primary text-primary-foreground shadow-sm'
                : 'text-muted-foreground hover:bg-muted hover:text-foreground',
            )}
          >
            {option}
          </button>
        ))}
      </div>
    </div>
  );
}
