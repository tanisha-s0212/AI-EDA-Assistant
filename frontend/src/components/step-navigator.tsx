'use client';

import React, { useCallback, useRef } from 'react';
import { useAppStore, TabId } from '@/lib/store';
import {
  ChevronLeft,
  ChevronRight,
  Upload,
  Database,
  Sparkles,
  BarChart3,
  BrainCircuit,
  Target,
  LineChart,
  FileText,
  CheckCircle2,
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';
import { motion } from 'framer-motion';

const tabs: { id: TabId; label: string; shortLabel: string; icon: React.ElementType; step: number }[] = [
  { id: 'upload', label: 'Data Upload', shortLabel: 'Upload', icon: Upload, step: 1 },
  { id: 'understanding', label: 'Understanding', shortLabel: 'Understand', icon: Database, step: 2 },
  { id: 'cleaning', label: 'Cleaning', shortLabel: 'Clean', icon: Sparkles, step: 3 },
  { id: 'eda', label: 'EDA', shortLabel: 'EDA', icon: BarChart3, step: 4 },
  { id: 'forecast_ts', label: 'Forecast TS', shortLabel: 'TS', icon: LineChart, step: 5 },
  { id: 'forecast_ml', label: 'Forecast ML', shortLabel: 'MLF', icon: LineChart, step: 6 },
  { id: 'ml', label: 'ML Assistant', shortLabel: 'ML', icon: BrainCircuit, step: 7 },
  { id: 'prediction', label: 'Prediction', shortLabel: 'Predict', icon: Target, step: 8 },
  { id: 'report', label: 'Report', shortLabel: 'Report', icon: FileText, step: 9 },
];

const mlSteps = [
  { step: 1, label: 'Target', shortLabel: 'Target' },
  { step: 2, label: 'Model', shortLabel: 'Model' },
  { step: 3, label: 'Config', shortLabel: 'Config' },
  { step: 4, label: 'Train', shortLabel: 'Train' },
  { step: 5, label: 'Compare', shortLabel: 'Compare' },
  { step: 6, label: 'Summary', shortLabel: 'Summary' },
];

type StepNavigatorProps = {
  showTabs?: boolean;
  showControls?: boolean;
  showSwipeHint?: boolean;
  className?: string;
};

export default function StepNavigator({
  showTabs = true,
  showControls = true,
  showSwipeHint = true,
  className,
}: StepNavigatorProps) {
  const { activeTab, setActiveTab, rawData, modelTrained, mlWorkflowStep, setMlWorkflowStep } = useAppStore();
  const touchStartX = useRef<number | null>(null);
  const touchEndX = useRef<number | null>(null);
  const currentIndex = tabs.findIndex((tab) => tab.id === activeTab);
  const isMlTab = activeTab === 'ml';

  const isTabEnabled = useCallback((tabId: TabId): boolean => {
    if (tabId === 'upload') return true;
    if (!rawData) return false;
    if (tabId === 'prediction' && !modelTrained) return false;
    return true;
  }, [modelTrained, rawData]);

  const goToTab = useCallback((tabId: TabId) => {
    if (isTabEnabled(tabId)) {
      setActiveTab(tabId);
    }
  }, [isTabEnabled, setActiveTab]);

  const goPrev = useCallback(() => {
    if (isMlTab) {
      setMlWorkflowStep(mlWorkflowStep - 1);
      return;
    }

    if (currentIndex <= 0) return;
    const prev = tabs[currentIndex - 1];
    if (prev && isTabEnabled(prev.id)) {
      goToTab(prev.id);
    }
  }, [currentIndex, goToTab, isMlTab, isTabEnabled, mlWorkflowStep, setMlWorkflowStep]);

  const goNext = useCallback(() => {
    if (isMlTab) {
      if (mlWorkflowStep < 6) {
        setMlWorkflowStep(mlWorkflowStep + 1);
        return;
      }
      if (modelTrained) {
        setActiveTab('prediction');
      }
      return;
    }

    if (currentIndex >= tabs.length - 1) return;
    const nextEnabledTab = tabs.slice(currentIndex + 1).find((tab) => isTabEnabled(tab.id));
    if (nextEnabledTab) {
      goToTab(nextEnabledTab.id);
    }
  }, [currentIndex, goToTab, isMlTab, isTabEnabled, mlWorkflowStep, modelTrained, setActiveTab, setMlWorkflowStep]);

  const onTouchStart = useCallback((e: React.TouchEvent) => {
    touchEndX.current = null;
    touchStartX.current = e.targetTouches[0].clientX;
  }, []);

  const onTouchMove = useCallback((e: React.TouchEvent) => {
    touchEndX.current = e.targetTouches[0].clientX;
  }, []);

  const onTouchEnd = useCallback(() => {
    if (touchStartX.current === null || touchEndX.current === null) return;
    const diff = touchStartX.current - touchEndX.current;
    const minSwipe = 60;

    if (Math.abs(diff) > minSwipe) {
      if (diff > 0) {
        goNext();
      } else {
        goPrev();
      }
    }

    touchStartX.current = null;
    touchEndX.current = null;
  }, [goNext, goPrev]);

  const nextTab = tabs.slice(currentIndex + 1).find((tab) => isTabEnabled(tab.id)) ?? null;
  const mlNextDisabledReason = mlWorkflowStep === 6 && !modelTrained
    ? 'Train a model in ML Assistant before opening Prediction.'
    : null;
  const canGoPrev = isMlTab ? mlWorkflowStep > 1 : currentIndex > 0 && isTabEnabled(tabs[currentIndex - 1]?.id);
  const canGoNext = isMlTab ? (mlWorkflowStep < 6 || modelTrained) : Boolean(nextTab);
  const nextDisabledReason = isMlTab ? mlNextDisabledReason : null;
  const nextButtonLabel = isMlTab
    ? mlWorkflowStep < 6
      ? `Next: ${mlSteps[mlWorkflowStep]?.shortLabel ?? 'Next'}`
      : 'Next: Predict'
    : nextTab?.label ? `Next: ${nextTab.shortLabel}` : 'Next';

  return (
    <div className={cn('mb-5 space-y-3', className)}>
      {showTabs && (
        <div
          onTouchStart={onTouchStart}
          onTouchMove={onTouchMove}
          onTouchEnd={onTouchEnd}
          className="relative"
        >
          <div className="absolute inset-0 pointer-events-none rounded-xl overflow-hidden">
            <div className="absolute left-0 top-0 bottom-0 w-8 bg-gradient-to-r from-background/80 to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />
            <div className="absolute right-0 top-0 bottom-0 w-8 bg-gradient-to-r from-transparent to-background/80 opacity-0 group-hover:opacity-100 transition-opacity" />
          </div>

          <div className="flex items-center gap-1.5 overflow-x-auto pb-1 scrollbar-none">
            {tabs.map((tab, index) => {
              const isActive = activeTab === tab.id;
              const enabled = isTabEnabled(tab.id);
              const Icon = tab.icon;
              const isPast = currentIndex > index;

              return (
                <button
                  key={tab.id}
                  onClick={() => goToTab(tab.id)}
                  disabled={!enabled}
                  className={cn(
                    'flex shrink-0 items-center gap-1.5 rounded-full border px-3 py-2 text-xs font-medium whitespace-nowrap transition-all duration-200',
                    isActive && enabled && 'border-primary bg-primary text-primary-foreground shadow-md shadow-primary/20',
                    !isActive && enabled && isPast && 'border-border bg-secondary text-secondary-foreground',
                    !isActive && enabled && !isPast && 'border-border bg-background text-muted-foreground hover:border-primary/30 hover:text-primary',
                    !enabled && 'cursor-not-allowed border-muted/50 bg-muted/30 text-muted-foreground/30',
                  )}
                >
                  <Icon
                    className={cn(
                      'h-3.5 w-3.5',
                      isActive && enabled && 'text-white',
                      !isActive && !enabled && 'opacity-40',
                    )}
                  />
                  <span className="hidden sm:inline">{tab.label}</span>
                  <span className="sm:hidden">{tab.shortLabel}</span>
                  {isPast && enabled && <CheckCircle2 className="h-3 w-3 text-primary" />}
                  <span
                    className={cn(
                      'text-[10px] font-mono',
                      isActive ? 'text-white/80' : 'text-muted-foreground/60',
                    )}
                  >
                    {tab.step}
                  </span>
                </button>
              );
            })}
          </div>
        </div>
      )}

      {showControls && (
        <div className="flex items-center justify-between gap-3">
          <Button
            variant="outline"
            size="sm"
            onClick={goPrev}
            disabled={!canGoPrev}
            className={cn(
              'gap-1.5 text-xs',
              canGoPrev && 'hover:border-primary/30 hover:text-primary',
            )}
          >
            <ChevronLeft className="h-3.5 w-3.5" />
            <span className="hidden sm:inline">Previous</span>
          </Button>

          <div className="flex items-center gap-1.5">
            {isMlTab ? (
              mlSteps.map((step) => {
                const isActive = mlWorkflowStep === step.step;
                const enabled = step.step <= mlWorkflowStep || step.step < 6 || modelTrained;
                const isPast = mlWorkflowStep > step.step;

                return (
                  <button
                    key={`ml-step-${step.step}`}
                    onClick={() => setMlWorkflowStep(step.step)}
                    disabled={!enabled}
                    className={cn(
                      'rounded-full transition-all duration-300',
                      isActive
                        ? 'h-2 w-6 bg-primary'
                        : isPast && enabled
                          ? 'h-2 w-2 bg-primary/60'
                          : enabled
                            ? 'h-2 w-2 bg-muted-foreground/30 hover:bg-primary/60'
                            : 'h-2 w-2 bg-muted/20',
                      !enabled && 'cursor-not-allowed',
                    )}
                    title={step.label}
                  />
                );
              })
            ) : (
              tabs.map((tab, index) => {
                const isActive = activeTab === tab.id;
                const enabled = isTabEnabled(tab.id);
                const isPast = currentIndex > index;

                return (
                  <button
                    key={tab.id}
                    onClick={() => goToTab(tab.id)}
                    disabled={!enabled}
                    className={cn(
                      'rounded-full transition-all duration-300',
                      isActive
                        ? 'h-2 w-6 bg-primary'
                        : isPast && enabled
                          ? 'h-2 w-2 bg-primary/60'
                          : enabled
                            ? 'h-2 w-2 bg-muted-foreground/30 hover:bg-primary/60'
                            : 'h-2 w-2 bg-muted/20',
                      !enabled && 'cursor-not-allowed',
                    )}
                    title={tab.label}
                  />
                );
              })
            )}
          </div>

          <Button
            variant="outline"
            size="sm"
            onClick={goNext}
            disabled={!canGoNext}
            title={nextDisabledReason ?? undefined}
            className={cn(
              'gap-1.5 text-xs',
              canGoNext && 'hover:border-primary/30 hover:text-primary',
            )}
          >
            <span className="hidden sm:inline">{nextButtonLabel}</span>
            <ChevronRight className="h-3.5 w-3.5" />
          </Button>
        </div>
      )}

      {nextDisabledReason && showControls && (
        <p className="text-center text-xs text-muted-foreground">{nextDisabledReason}</p>
      )}

      {showSwipeHint && showTabs && showControls && (
        <motion.p
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 2, duration: 0.5 }}
          className="text-center text-[10px] text-muted-foreground/50 sm:hidden"
        >
          Swipe to navigate
        </motion.p>
      )}
    </div>
  );
}
