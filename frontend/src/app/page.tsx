'use client';

import React from 'react';
import { useAppStore, TabId } from '@/lib/store';
import { apiClient } from '@/lib/api';
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
  ChevronRight,
  Bot,
  ShieldCheck,
  Building2,
  History,
  CheckCircle2,
  AlertCircle,
  RotateCcw,
  RefreshCw,
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

const DESKTOP_SIDEBAR_WIDTH = 'lg:pl-72';

type ActivityResponse = {
  activities: Array<{
    action: string;
    detail: string | null;
    createdAt: string;
    datasetId: string | null;
    status: string;
  }>;
  count: number;
};

function formatActivityTimestamp(value: string | null) {
  if (!value) return 'No recent sync';
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return 'Recent sync recorded';
  return new Intl.DateTimeFormat(undefined, {
    dateStyle: 'medium',
    timeStyle: 'short',
  }).format(parsed);
}

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
      <div className="px-5 pb-4 pt-5 sm:px-6 sm:pt-6">
        <div className="flex items-center gap-3">
          <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-primary p-2 text-primary-foreground shadow-lg shadow-primary/20">
            <Bot className="h-6 w-6" />
          </div>
          <div className="min-w-0">
            <h1 className="whitespace-nowrap text-[15px] font-bold tracking-tight text-foreground sm:text-base">
              Intelligent Data Assistant
            </h1>
            <p className="text-xs text-muted-foreground">AI-guided dataset understanding, analysis, and modeling.</p>
          </div>
        </div>
      </div>
      <Separator className="opacity-50" />

      {/* Navigation */}
      <div className="flex-1 overflow-y-auto px-3 py-3 sm:py-4 [scrollbar-width:none] [-ms-overflow-style:none] [&::-webkit-scrollbar]:hidden">
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
                    'group relative flex w-full items-center gap-3 rounded-xl border border-transparent px-3 py-3 text-left text-sm font-medium transition-all duration-200',
                    isActive && enabled && 'border-primary/30 bg-primary/10 text-primary shadow-sm',
                    !isActive && enabled && 'bg-transparent text-muted-foreground hover:border-border hover:bg-accent hover:text-accent-foreground',
                    !enabled && 'cursor-not-allowed text-muted-foreground/40',
                  )}
                >
                  {isActive && enabled && (
                    <motion.div
                      layoutId="activeTab"
                      className="absolute left-0 top-2 bottom-2 w-1 rounded-r-full bg-primary"
                      transition={{ type: 'spring', stiffness: 300, damping: 30 }}
                    />
                  )}
                  <div className={cn(
                    'flex h-9 w-9 shrink-0 items-center justify-center rounded-lg transition-colors',
                    isActive && enabled && 'bg-primary text-primary-foreground shadow-sm shadow-primary/25',
                    !isActive && enabled && 'bg-secondary text-secondary-foreground',
                    !enabled && 'bg-muted/50 text-muted-foreground/40',
                  )}>
                    <Icon className="h-4 w-4" />
                  </div>
                  <span className="truncate">{tab.label}</span>
                  {isActive && enabled && (
                    <ChevronRight className="ml-auto h-4 w-4 text-primary" />
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
      <div className="border-t p-4 sm:p-5">
        <div className="rounded-xl border border-border bg-secondary/60 p-3">
          <p className="text-xs font-semibold text-secondary-foreground">
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
  const {
    activeTab,
    rawData,
    fileName,
    columns,
    previewLoaded,
    loadedRowCount,
    totalRows,
    setActiveTab,
    resetWorkspace,
    hasHydrated,
  } = useAppStore();
  const [isRefreshingActivity, setIsRefreshingActivity] = React.useState(false);
  const [recentActivity, setRecentActivity] = React.useState<ActivityResponse['activities'][number] | null>(null);
  const activeTabMeta = tabs.find((t) => t.id === activeTab) ?? tabs[0];
  const hasWorkspace = Boolean(rawData?.length);
  const datasetStatus = rawData
    ? previewLoaded
      ? `${loadedRowCount.toLocaleString()} preview rows of ${totalRows.toLocaleString()}`
      : `${totalRows.toLocaleString()} rows ready`
    : 'No dataset loaded';
  const workflowReadiness = !hasWorkspace
    ? 'Resume your previous workspace or start fresh with a new dataset.'
    : previewLoaded
      ? 'Large dataset connected with preview rendering in the browser and full-fidelity coverage preserved in the backend.'
      : 'Workspace is fully loaded and ready for analysis, forecasting, modeling, and reporting.';
  const sessionSummary = hasWorkspace
    ? `Continue from ${activeTabMeta.label.toLowerCase()} with ${columns.length.toLocaleString()} profiled columns in scope.`
    : 'No active workspace is loaded in memory yet.';
  const recentSyncLabel = formatActivityTimestamp(recentActivity?.createdAt ?? null);
  const activityLabel = recentActivity
    ? recentActivity.action.replace(/_/g, ' ')
    : 'Waiting for backend activity';

  const refreshRecentActivity = React.useCallback(async () => {
    setIsRefreshingActivity(true);
    try {
      const response = await apiClient.get<ActivityResponse>('/activities', {
        params: { limit: 1 },
      });
      setRecentActivity(response.data.activities[0] ?? null);
    } catch {
      setRecentActivity(null);
    } finally {
      setIsRefreshingActivity(false);
    }
  }, []);

  React.useEffect(() => {
    if (!hasHydrated) return;
    void refreshRecentActivity();
  }, [hasHydrated, refreshRecentActivity]);

  const handleResumeWorkspace = React.useCallback(() => {
    if (!hasWorkspace) {
      setActiveTab('upload');
      return;
    }
    setActiveTab(activeTab === 'upload' ? 'understanding' : activeTab);
  }, [activeTab, hasWorkspace, setActiveTab]);

  const handleFreshStart = React.useCallback(() => {
    resetWorkspace();
  }, [resetWorkspace]);

  const renderTab = () => {
    switch (activeTab) {
      case 'upload': return <UploadTab />;
      case 'understanding': return <UnderstandingTab />;
      case 'cleaning': return <CleaningTab />;
      case 'eda': return <EdaTab />;
      case 'sales_forecast': return <SalesForecastTab />;
      case 'ml': return <MlTab />;
      case 'prediction': return <PredictionTab />;
      case 'report': return <ReportTab />;
      default: return <UploadTab />;
    }
  };

  return (
    <div className="min-h-screen bg-background">
      {/* Desktop Sidebar */}
      <aside className="fixed inset-y-0 left-0 z-40 hidden h-screen w-72 flex-col border-r border-border bg-sidebar/95 shadow-[0_24px_80px_-28px_rgba(15,23,42,0.18)] backdrop-blur-xl lg:flex">
        <SidebarContent />
      </aside>

      {/* Main Content */}
      <div className={cn('flex min-w-0 flex-1 flex-col', DESKTOP_SIDEBAR_WIDTH)}>
        {/* Content */}
        <main className="flex-1 overflow-y-auto overflow-x-hidden [scrollbar-width:none] [-ms-overflow-style:none] [&::-webkit-scrollbar]:hidden">
          <div className="mx-auto max-w-7xl px-4 pb-6 pt-3 sm:px-6 sm:pt-4 lg:px-8">
            <div className="sticky top-0 z-30 -mx-4 mb-5 border-b border-border/70 bg-[linear-gradient(180deg,rgba(248,250,252,0.97),rgba(244,247,251,0.94))] px-4 py-4 backdrop-blur-xl dark:bg-[linear-gradient(180deg,rgba(15,23,42,0.96),rgba(15,23,42,0.92))] sm:-mx-6 sm:mb-6 sm:px-6 sm:py-5 lg:-mx-8 lg:px-8">
              <div className="mx-auto max-w-7xl">
                <div className="overflow-hidden rounded-[28px] border border-slate-800/80 bg-[linear-gradient(135deg,#0f172a_0%,#162338_55%,#1e293b_100%)] p-4 text-white shadow-[0_26px_90px_-38px_rgba(15,23,42,0.72)] sm:p-5">
                  <div className="flex flex-col gap-5">
                    <div className="flex flex-col gap-4 xl:flex-row xl:items-start xl:justify-between">
                      <div className="flex items-start gap-3">
                        <Sheet>
                          <SheetTrigger asChild>
                            <Button variant="ghost" size="icon" className="mt-0.5 shrink-0 text-white hover:bg-white/10 hover:text-white lg:hidden">
                              <Menu className="h-5 w-5" />
                            </Button>
                          </SheetTrigger>
                          <SheetContent side="left" className="w-72 p-0">
                            <SidebarContent />
                          </SheetContent>
                        </Sheet>
                        <div className="min-w-0">
                          <div className="flex items-center gap-3">
                            <div className="flex h-12 w-12 shrink-0 items-center justify-center rounded-2xl bg-white/10 text-white ring-1 ring-white/15">
                              <Building2 className="h-5 w-5" />
                            </div>
                            <div className="min-w-0">
                              <p className="text-[11px] font-semibold uppercase tracking-[0.28em] text-slate-300">Workspace Session</p>
                              <p className="text-xl font-semibold tracking-tight text-white">Intelligent Data Assistant</p>
                              <p className="text-sm text-slate-300">
                                Standardized analytics workspace with session continuity, guided modeling, and report-ready outputs.
                              </p>
                            </div>
                          </div>
                          <div className="mt-4 flex flex-wrap items-center gap-2">
                            <Badge variant="outline" className="rounded-full border-white/15 bg-white/10 px-3 py-1 text-white">
                              {hasWorkspace ? <CheckCircle2 className="mr-2 h-3.5 w-3.5 text-emerald-300" /> : <AlertCircle className="mr-2 h-3.5 w-3.5 text-amber-300" />}
                              {hasWorkspace ? 'Workspace in progress' : 'Awaiting dataset'}
                            </Badge>
                            <Badge variant="outline" className="rounded-full border-white/15 bg-white/10 px-3 py-1 text-white">
                              <ShieldCheck className="mr-2 h-3.5 w-3.5 text-sky-300" />
                              PostgreSQL activity tracking connected
                            </Badge>
                          </div>
                        </div>
                      </div>

                      <div className="flex flex-wrap items-center gap-2 xl:justify-end">
                        <Button size="sm" className="h-10 rounded-full border border-white/10 bg-white text-slate-950 px-4 shadow-sm hover:bg-slate-100" onClick={handleResumeWorkspace}>
                          <History className="mr-2 h-4 w-4" />
                          {hasWorkspace ? 'Resume Previous Workspace' : 'Open Saved Workspace'}
                        </Button>
                        <Button size="sm" variant="outline" className="h-10 rounded-full border-white/20 bg-white/5 px-4 text-white hover:bg-white/10 hover:text-white" onClick={handleFreshStart}>
                          <RotateCcw className="mr-2 h-4 w-4" />
                          Fresh Start
                        </Button>
                        <Button size="sm" variant="ghost" className="h-10 rounded-full px-3 text-slate-200 hover:bg-white/10 hover:text-white" onClick={() => void refreshRecentActivity()}>
                          <RefreshCw className={cn('mr-2 h-4 w-4', isRefreshingActivity && 'animate-spin')} />
                          Refresh Sync
                        </Button>
                        <ThemeToggle />
                      </div>
                    </div>

                    <div className="grid gap-3 border-t border-white/10 pt-4 lg:grid-cols-[minmax(0,1.35fr)_minmax(360px,1fr)]">
                      <div className="min-w-0">
                        <p className="text-[11px] font-semibold uppercase tracking-[0.24em] text-slate-300">Current Focus</p>
                        <h1 className="mt-2 text-2xl font-semibold tracking-tight text-white">
                          {activeTabMeta.label}
                        </h1>
                        <p className="mt-3 max-w-3xl text-sm leading-6 text-slate-300">
                          {workflowReadiness}
                        </p>
                        <p className="mt-3 text-sm text-slate-400">{sessionSummary}</p>
                      </div>

                      <div className="grid gap-3 sm:grid-cols-2">
                        <div className="rounded-2xl border border-white/10 bg-white/6 px-4 py-4 shadow-sm backdrop-blur-sm">
                          <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-300">
                            Dataset Summary
                          </p>
                          <p className="mt-2 text-lg font-semibold tracking-tight text-white">{datasetStatus}</p>
                          <p className="mt-1 truncate text-sm text-slate-300">
                            {fileName ? fileName : 'No source selected'}
                          </p>
                          <div className="mt-4 flex flex-wrap gap-2">
                            <span className="inline-flex items-center gap-1 rounded-full bg-white/10 px-2.5 py-1 text-xs text-slate-200">
                              <Database className="h-3 w-3" />
                              {hasWorkspace ? totalRows.toLocaleString() : 0} rows
                            </span>
                            <span className="inline-flex items-center gap-1 rounded-full bg-white/10 px-2.5 py-1 text-xs text-slate-200">
                              {columns.length.toLocaleString()} columns
                            </span>
                            <span className="inline-flex items-center gap-1 rounded-full bg-white/10 px-2.5 py-1 text-xs text-slate-200">
                              {previewLoaded ? 'Preview + cached' : 'Fully loaded'}
                            </span>
                          </div>
                          <p className="mt-3 text-xs text-slate-400">
                            {previewLoaded ? 'Responsive preview is active while backend processing continues on the complete dataset.' : 'Workspace data is ready for the full guided workflow.'}
                          </p>
                        </div>

                        <div className="rounded-2xl border border-white/10 bg-white/6 px-4 py-4 shadow-sm backdrop-blur-sm">
                          <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-300">
                            Session Continuity
                          </p>
                          <p className="mt-2 text-lg font-semibold tracking-tight text-white">{recentSyncLabel}</p>
                          <p className="mt-1 text-sm capitalize text-slate-300">{activityLabel}</p>
                          <div className="mt-4 flex items-start gap-3 rounded-2xl border border-white/10 bg-slate-950/25 p-3">
                            <div className="mt-0.5 flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-white/10 text-white">
                              <History className="h-4 w-4" />
                            </div>
                            <div className="min-w-0">
                              <p className="text-sm font-medium text-white">
                                {recentActivity?.detail || 'The workspace can continue from the last saved browser state and backend activity trail.'}
                              </p>
                              <p className="mt-1 text-xs text-slate-400">
                                {recentActivity?.datasetId ? `Dataset ${recentActivity.datasetId} was the latest backend-linked session.` : 'Load a dataset to establish a persisted working session.'}
                              </p>
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
            <div className="relative">
              {renderTab()}
            </div>
            <StepNavigator showTabs={false} showSwipeHint={false} className="mt-8 mb-2" />
          </div>
        </main>

        {/* Footer */}
        <footer className="mt-auto border-t border-border/70 bg-[linear-gradient(180deg,rgba(248,250,252,0.97),rgba(244,247,251,0.94))] px-4 py-4 backdrop-blur-xl dark:bg-[linear-gradient(180deg,rgba(15,23,42,0.96),rgba(15,23,42,0.92))] sm:px-6 lg:px-8">
          <div className="mx-auto max-w-7xl">
            <div className="flex flex-col gap-2 overflow-hidden rounded-[28px] border border-slate-800/80 bg-[linear-gradient(135deg,#0f172a_0%,#162338_55%,#1e293b_100%)] px-6 py-4 text-base font-bold text-white shadow-[0_26px_90px_-38px_rgba(15,23,42,0.72)] lg:flex-row lg:items-center lg:justify-between">
              <span>Intelligent Data Assistant</span>
              <span className="text-sm font-bold text-slate-300">
                AI-guided dataset understanding, analysis, and predictive modeling.
              </span>
            </div>
          </div>
        </footer>
      </div>
    </div>
  );
}
