'use client';

import React from 'react';
import { useAppStore, TabId } from '@/lib/store';
import { ThemeToggle } from '@/components/theme-toggle';
import StepNavigator from '@/components/step-navigator';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import { Sheet, SheetContent, SheetTrigger } from '@/components/ui/sheet';
import {
  Upload,
  Database,
  Sparkles,
  BarChart3,
  BrainCircuit,
  Target,
  LineChart,
  FileText,
  Menu,
  Bot,
  ChevronRight,
} from 'lucide-react';
import { motion } from 'framer-motion';
import { cn } from '@/lib/utils';

import UploadTab from '@/components/tabs/upload-tab';
import UnderstandingTab from '@/components/tabs/understanding-tab';
import CleaningTab from '@/components/tabs/cleaning-tab';
import EdaTab from '@/components/tabs/eda-tab';
import MlTab from '@/components/tabs/ml-tab';
import PredictionTab from '@/components/tabs/prediction-tab';
import SalesForecastTab from '@/components/tabs/sales-forecast-tab';
import ReportTab from '@/components/tabs/report-tab';

const tabs: { id: TabId; label: string; icon: React.ElementType; step: number }[] = [
  { id: 'upload', label: 'Data Upload', icon: Upload, step: 1 },
  { id: 'understanding', label: 'Understanding', icon: Database, step: 2 },
  { id: 'cleaning', label: 'Cleaning', icon: Sparkles, step: 3 },
  { id: 'eda', label: 'EDA', icon: BarChart3, step: 4 },
  { id: 'sales_forecast', label: 'Sales Forecast', icon: LineChart, step: 5 },
  { id: 'ml', label: 'ML Assistant', icon: BrainCircuit, step: 6 },
  { id: 'prediction', label: 'Prediction', icon: Target, step: 7 },
  { id: 'report', label: 'Report', icon: FileText, step: 8 },
];

function SidebarContent({ onNavigate }: { onNavigate?: (id: TabId) => void }) {
  const { activeTab, setActiveTab, rawData, cleaningDone, modelTrained, previewLoaded, loadedRowCount, totalRows } = useAppStore();

  const isTabEnabled = (tabId: TabId) => {
    if (tabId === 'upload') return true;
    if (!rawData) return false;
    if (tabId === 'prediction' && !modelTrained) return false;
    return true;
  };

  return (
    <div className="flex flex-col h-full">
      {/* Logo */}
      <div className="p-6 pb-4">
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-br from-emerald-500 to-teal-600 text-white shadow-lg shadow-emerald-500/25">
            <Bot className="h-5 w-5" />
          </div>
          <div>
            <h1 className="text-base font-bold tracking-tight">Intelligent Data Assistant</h1>
            <p className="text-xs text-muted-foreground">Universal ML Workspace</p>
          </div>
        </div>
      </div>
      <Separator className="opacity-50" />

      {/* Navigation */}
      <div className="flex-1 overflow-y-auto px-3 py-4">
        <nav className="flex flex-col gap-1">
          {tabs.map((tab, index) => {
            const isActive = activeTab === tab.id;
            const enabled = isTabEnabled(tab.id);
            const Icon = tab.icon;

            return (
              <React.Fragment key={tab.id}>
                <motion.button
                  whileHover={enabled ? { x: 4 } : undefined}
                  whileTap={enabled ? { scale: 0.98 } : undefined}
                  onClick={() => {
                    if (enabled) {
                      setActiveTab(tab.id);
                      onNavigate?.(tab.id);
                    }
                  }}
                  className={cn(
                    'group relative flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium transition-all duration-200',
                    isActive && enabled && 'bg-gradient-to-r from-emerald-500/10 to-teal-500/5 text-emerald-600 dark:text-emerald-400',
                    !isActive && enabled && 'text-muted-foreground hover:bg-accent hover:text-foreground',
                    !enabled && 'cursor-not-allowed text-muted-foreground/40',
                  )}
                >
                  {isActive && enabled && (
                    <motion.div
                      layoutId="activeTab"
                      className="absolute left-0 top-1/2 -translate-y-1/2 h-6 w-1 rounded-r-full bg-gradient-to-b from-emerald-500 to-teal-500"
                      transition={{ type: 'spring', stiffness: 300, damping: 30 }}
                    />
                  )}
                  <div className={cn(
                    'flex h-7 w-7 shrink-0 items-center justify-center rounded-md text-xs font-bold transition-colors',
                    isActive && enabled && 'bg-emerald-500 text-white shadow-sm shadow-emerald-500/30',
                    !isActive && enabled && 'bg-muted text-muted-foreground',
                    !enabled && 'bg-muted/50 text-muted-foreground/40',
                  )}>
                    {tab.step}
                  </div>
                  <span className="truncate">{tab.label}</span>
                  {isActive && enabled && (
                    <ChevronRight className="ml-auto h-4 w-4 text-emerald-500" />
                  )}
                </motion.button>
                {index < tabs.length - 1 && index < 1 && (
                  <div className="flex items-center px-5 py-1">
                    <div className="h-2 w-px bg-border" />
                  </div>
                )}
              </React.Fragment>
            );
          })}
        </nav>
      </div>

      {/* Footer */}
      <div className="p-4 border-t">
        <div className="rounded-lg bg-gradient-to-r from-emerald-500/5 to-teal-500/5 p-3">
          <p className="text-xs font-medium text-emerald-600 dark:text-emerald-400">
            AI-Powered Analysis
          </p>
          <p className="text-xs text-muted-foreground mt-0.5">
            {rawData
              ? previewLoaded
                ? `${loadedRowCount.toLocaleString()} preview rows of ${totalRows.toLocaleString()} total`
                : `${totalRows.toLocaleString()} rows loaded`
              : 'Upload data to start'}
          </p>
        </div>
      </div>
    </div>
  );
}

export default function HomePage() {
  const { activeTab, rawData, previewLoaded, loadedRowCount, totalRows } = useAppStore();

  const renderTab = () => {
    switch (activeTab) {
      case 'upload': return <UploadTab />;
      case 'understanding': return <UnderstandingTab />;
      case 'cleaning': return <CleaningTab />;
      case 'eda': return <EdaTab />;
      case 'ml': return <MlTab />;
      case 'sales_forecast': return <SalesForecastTab />;
      case 'prediction': return <PredictionTab />;
      case 'report': return <ReportTab />;
      default: return <UploadTab />;
    }
  };

  return (
    <div className="flex min-h-screen">
      {/* Desktop Sidebar */}
      <aside className="hidden lg:flex lg:w-64 lg:flex-col lg:border-r bg-card/50 backdrop-blur-sm">
        <SidebarContent />
      </aside>

      {/* Main Content */}
      <div className="flex flex-1 flex-col min-w-0">
        {/* Header */}
        <header className="sticky top-0 z-40 flex h-14 items-center gap-4 border-b bg-background/80 backdrop-blur-md px-4 lg:px-6">
          {/* Mobile menu */}
          <Sheet>
            <SheetTrigger asChild>
              <Button variant="ghost" size="icon" className="lg:hidden">
                <Menu className="h-5 w-5" />
              </Button>
            </SheetTrigger>
            <SheetContent side="left" className="w-64 p-0">
              <SidebarContent />
            </SheetContent>
          </Sheet>

          {/* Breadcrumb */}
          <div className="flex items-center gap-2">
            <span className="text-sm text-muted-foreground">Step</span>
            <Badge variant="secondary" className="bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border-emerald-500/20">
              {tabs.find(t => t.id === activeTab)?.step ?? 1}
            </Badge>
            <span className="text-sm font-medium">
              {tabs.find(t => t.id === activeTab)?.label ?? 'Upload'}
            </span>
          </div>

          <div className="ml-auto flex items-center gap-2">
            {rawData && (
              <Badge variant="outline" className="text-xs">
                <Database className="mr-1 h-3 w-3" />
                {previewLoaded
                  ? `${loadedRowCount.toLocaleString()} preview / ${totalRows.toLocaleString()} total`
                  : `${totalRows.toLocaleString()} rows`}
              </Badge>
            )}
            <ThemeToggle />
          </div>
        </header>

        {/* Content */}
        <main className="flex-1 overflow-auto">
          <div className="p-4 lg:p-6 max-w-7xl mx-auto">
            <StepNavigator showControls={false} />
            <div>
              {renderTab()}
            </div>
            <StepNavigator showTabs={false} showSwipeHint={false} className="mt-8 mb-2" />
          </div>
        </main>

        {/* Footer */}
        <footer className="border-t mt-auto">
          <div className="flex h-12 items-center justify-between px-4 lg:px-6 text-xs text-muted-foreground">
            <span>Intelligent Data Assistant</span>
            <span>Universal Dataset Support</span>
          </div>
        </footer>
      </div>
    </div>
  );
}
