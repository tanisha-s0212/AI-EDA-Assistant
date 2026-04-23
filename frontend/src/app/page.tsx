'use client';

import React from 'react';
import { useAppStore, TabId, AuthenticatedUser } from '@/lib/store';
import { apiClient, getApiErrorMessage } from '@/lib/api';
import { ThemeToggle } from '@/components/theme-toggle';
import LoginPage from '@/components/login-page';
import StepNavigator from '@/components/step-navigator';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import { useToast } from '@/hooks/use-toast';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
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
  ShieldCheck,
  History,
  CheckCircle2,
  AlertCircle,
  RotateCcw,
  RefreshCw,
  Orbit,
  Radar,
  Cpu,
  LogOut,
  UserRound,
  Mail,
} from 'lucide-react';
import { motion } from 'framer-motion';
import { cn } from '@/lib/utils';

import UploadTab from '@/components/tabs/upload-tab';
import UnderstandingTab from '@/components/tabs/understanding-tab';
import CleaningTab from '@/components/tabs/cleaning-tab';
import EdaTab from '@/components/tabs/eda-tab';
import MlTab from '@/components/tabs/ml-tab';
import PredictionTab from '@/components/tabs/prediction-tab';
import TimeSeriesForecastTab from '@/components/tabs/time-series-forecast-tab';
import MlForecastTab from '@/components/tabs/ml-forecast-tab';
import ReportTab from '@/components/tabs/report-tab';

const tabs: { id: TabId; label: string; icon: React.ElementType }[] = [
  { id: 'upload', label: 'Data Upload', icon: Upload },
  { id: 'understanding', label: 'Data Understanding', icon: Database },
  { id: 'eda', label: 'Exploratory Data Analysis', icon: BarChart3 },
  { id: 'cleaning', label: 'Data Cleaning', icon: Sparkles },
  { id: 'forecast_ts', label: 'Time Series Forecast', icon: LineChart },
  { id: 'forecast_ml', label: 'Machine Learning Forecast', icon: LineChart },
  { id: 'ml', label: 'ML Assistant', icon: BrainCircuit },
  { id: 'prediction', label: 'Prediction', icon: Target },
  { id: 'report', label: 'Report', icon: FileText },
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

type AuthResponse = {
  user: AuthenticatedUser;
};

type DatasetPreviewResponse = {
  datasetId: string;
  fileName: string | null;
  data: Array<Record<string, string | number | boolean | null>>;
  columns: Array<{
    name: string;
    dtype: string;
    nonNull: number;
    nullCount: number;
    uniqueCount: number;
    role: string;
    sample?: string[];
  }>;
  rowCount: number;
  loadedRowCount: number;
  previewLoaded: boolean;
  duplicates: number;
};

const INDIA_TIMEZONE = 'Asia/Kolkata';

function formatActivityTimestamp(value: string | null) {
  const parsed = value ? new Date(value) : new Date();
  if (Number.isNaN(parsed.getTime())) return 'Current time unavailable';
  const formatted = new Intl.DateTimeFormat('en-IN', {
    dateStyle: 'medium',
    timeStyle: 'short',
    timeZone: INDIA_TIMEZONE,
  }).format(parsed);
  return `${formatted} IST`;
}

function formatIndiaTime(value: Date | string) {
  const parsed = value instanceof Date ? value : new Date(value);
  if (Number.isNaN(parsed.getTime())) return 'Time unavailable';
  return new Intl.DateTimeFormat('en-IN', {
    hour: 'numeric',
    minute: '2-digit',
    hour12: true,
    timeZone: INDIA_TIMEZONE,
  }).format(parsed);
}

function formatIndiaDate(value: Date | string) {
  const parsed = value instanceof Date ? value : new Date(value);
  if (Number.isNaN(parsed.getTime())) return 'Date unavailable';
  return new Intl.DateTimeFormat('en-IN', {
    weekday: 'short',
    day: 'numeric',
    month: 'short',
    year: 'numeric',
    timeZone: INDIA_TIMEZONE,
  }).format(parsed);
}

function getRelativeActivityAge(value: string | null) {
  if (!value) return 'Awaiting backend activity';
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return 'Timestamp unavailable';

  const diffMinutes = Math.max(0, Math.round((Date.now() - parsed.getTime()) / 60000));
  if (diffMinutes < 1) return 'Updated just now';
  if (diffMinutes < 60) return `Updated ${diffMinutes} min ago`;

  const diffHours = Math.round(diffMinutes / 60);
  if (diffHours < 24) return `Updated ${diffHours} hr ago`;

  const diffDays = Math.round(diffHours / 24);
  return `Updated ${diffDays} day${diffDays === 1 ? '' : 's'} ago`;
}

function getSessionContinuityLabel(value: string | null) {
  const parsed = value ? new Date(value) : new Date();
  const timeLabel = Number.isNaN(parsed.getTime())
    ? 'Time unavailable'
    : new Intl.DateTimeFormat('en-IN', {
        hour: 'numeric',
        minute: '2-digit',
        hour12: true,
        timeZone: INDIA_TIMEZONE,
      }).format(parsed);
  const dateLabel = Number.isNaN(parsed.getTime())
    ? 'Date unavailable'
    : new Intl.DateTimeFormat('en-IN', {
        weekday: 'short',
        day: 'numeric',
        month: 'short',
        year: 'numeric',
        timeZone: INDIA_TIMEZONE,
      }).format(parsed);

  if (value) {
    return {
      timestamp: formatActivityTimestamp(value),
      status: 'Last synced activity',
      timezone: 'India Standard Time (IST) | India UTC+5:30',
      timeLabel,
      dateLabel,
      freshness: getRelativeActivityAge(value),
    };
  }

  return {
    timestamp: formatActivityTimestamp(null),
    status: 'Current IST shown until the first recorded activity',
    timezone: 'India Standard Time (IST) | India UTC+5:30',
    timeLabel,
    dateLabel,
    freshness: getRelativeActivityAge(null),
  };
}

function BrandMark({ compact = false }: { compact?: boolean }) {
  return (
    <div
      className={cn(
        'relative flex shrink-0 items-center justify-center overflow-hidden rounded-[22px] border border-white/12 bg-[linear-gradient(145deg,#08111f_0%,#0f2747_52%,#0b7f8f_100%)] text-white shadow-[0_24px_60px_-28px_rgba(14,116,144,0.62)]',
        compact ? 'h-11 w-11' : 'h-12 w-12'
      )}
    >
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_20%_18%,rgba(255,255,255,0.2),transparent_32%)]" />
      <div className="absolute inset-0 bg-[linear-gradient(180deg,rgba(255,255,255,0.08),transparent_42%,rgba(8,145,178,0.24)_100%)]" />
      <div className="absolute inset-[18%] rounded-[18px] border border-white/10 bg-slate-950/30" />
      <div className="absolute inset-x-[26%] top-[24%] h-px bg-cyan-200/50" />
      <div className="absolute inset-x-[26%] bottom-[24%] h-px bg-cyan-200/35" />
      <div className="absolute inset-y-[26%] left-[24%] w-px bg-cyan-200/40" />
      <div className="absolute inset-y-[26%] right-[24%] w-px bg-cyan-200/25" />
      <div className={cn('absolute rounded-md border border-cyan-200/25 bg-cyan-300/12', compact ? 'left-[19%] top-[19%] h-2 w-2' : 'left-[18%] top-[18%] h-2.5 w-2.5')} />
      <div className={cn('absolute rounded-md border border-cyan-200/20 bg-cyan-300/10', compact ? 'bottom-[18%] right-[18%] h-2.5 w-2.5' : 'bottom-[18%] right-[18%] h-3 w-3')} />
      <Orbit className={cn('absolute text-cyan-100/20', compact ? 'h-8 w-8' : 'h-9 w-9')} strokeWidth={1.5} />
      <Cpu className={cn('absolute text-white', compact ? 'h-4.5 w-4.5' : 'h-5 w-5')} strokeWidth={2.2} />
      <Radar className={cn('absolute text-cyan-100/85', compact ? 'h-3.5 w-3.5' : 'h-4 w-4')} strokeWidth={1.9} />
    </div>
  );
}

function BrandWordmark({
  compact = false,
  inverted = false,
}: {
  compact?: boolean;
  inverted?: boolean;
}) {
  return (
    <div className={cn('min-w-0', compact && 'max-w-[180px]')}>
      <h1 className={cn(
        'font-black tracking-[-0.03em]',
        compact ? 'text-[1.05rem] leading-[1.18] sm:text-[1.15rem]' : 'text-[1.95rem] leading-tight sm:text-[2.35rem]',
        inverted ? 'text-white' : 'text-slate-950'
      )}>
        <span className={cn(
          'bg-clip-text text-transparent',
          inverted
            ? 'bg-[linear-gradient(135deg,#ffffff_0%,#d7f9ff_42%,#67e8f9_100%)]'
            : 'bg-[linear-gradient(135deg,#082f49_0%,#0f766e_46%,#0284c7_100%)]'
        )}>
          Intelligent Data Assistant
        </span>
      </h1>
    </div>
  );
}

function SidebarContent({
  onNavigate,
  currentUser,
  onLogout,
}: {
  onNavigate?: (id: TabId) => void;
  currentUser?: AuthenticatedUser | null;
  onLogout?: () => void;
}) {
  const { activeTab, setActiveTab, rawData, modelTrained, totalRows } = useAppStore();
  const hasDatasetContext = Boolean(rawData?.length || totalRows > 0);

  const isTabEnabled = (tabId: TabId) => {
    if (tabId === 'upload') return true;
    if (!hasDatasetContext) return false;
    if (tabId === 'prediction' && !modelTrained) return false;
    return true;
  };

  return (
    <div className="flex h-full flex-col">
      {/* Logo */}
      <div className="px-5 pb-4 pt-5 sm:px-6 sm:pt-6">
        <div className="flex items-start gap-3">
          <BrandMark compact />
          <BrandWordmark compact />
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
                  whileHover={enabled ? { x: 4, scale: 1.01 } : undefined}
                  whileTap={enabled ? { scale: 0.98 } : undefined}
                  onClick={() => {
                    if (enabled) {
                      setActiveTab(tab.id);
                      onNavigate?.(tab.id);
                    }
                  }}
                  className={cn(
                    'group relative flex w-full items-center gap-3 overflow-hidden rounded-2xl border px-3 py-3 text-left text-sm font-medium transition-all duration-300',
                    isActive && enabled && 'border-primary/25 bg-[linear-gradient(135deg,rgba(59,130,246,0.16),rgba(125,211,252,0.08))] text-primary shadow-[0_18px_40px_-24px_rgba(37,99,235,0.5)]',
                    !isActive && enabled && 'border-transparent bg-transparent text-muted-foreground hover:border-border/70 hover:bg-card/80 hover:text-foreground hover:shadow-[0_14px_36px_-28px_rgba(15,23,42,0.2)]',
                    !enabled && 'cursor-not-allowed text-muted-foreground/40',
                  )}
                >
                  <div className="pointer-events-none absolute inset-0 opacity-0 transition-opacity duration-300 group-hover:opacity-100" aria-hidden>
                    <div className="absolute inset-y-0 left-0 w-20 bg-gradient-to-r from-primary/8 to-transparent" />
                  </div>
                  {isActive && enabled && (
                    <motion.div
                      layoutId="activeTab"
                      className="absolute left-0 top-2 bottom-2 w-1 rounded-r-full bg-primary"
                      transition={{ type: 'spring', stiffness: 300, damping: 30 }}
                    />
                  )}
                  <div className={cn(
                    'flex h-9 w-9 shrink-0 items-center justify-center rounded-xl transition-all duration-300',
                    isActive && enabled && 'bg-primary text-primary-foreground shadow-sm shadow-primary/25 ring-4 ring-primary/10',
                    !isActive && enabled && 'bg-secondary text-secondary-foreground group-hover:scale-105 group-hover:bg-primary/10 group-hover:text-primary',
                    !enabled && 'bg-muted/50 text-muted-foreground/40',
                  )}>
                    <Icon className="h-4 w-4" />
                  </div>
                  <div className="min-w-0 flex-1">
                    <span className="truncate">{tab.label}</span>
                  </div>
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
        <div className="rounded-[26px] border border-border/70 bg-[linear-gradient(180deg,rgba(255,255,255,0.78),rgba(248,250,252,0.96))] p-4 shadow-[0_18px_50px_-32px_rgba(15,23,42,0.25)] dark:bg-[linear-gradient(180deg,rgba(30,41,59,0.8),rgba(15,23,42,0.96))]">
          <div className="flex items-start gap-3">
            <div className="flex h-11 w-11 shrink-0 items-center justify-center rounded-2xl bg-[linear-gradient(135deg,#0f172a_0%,#0f766e_100%)] text-white shadow-[0_16px_40px_-26px_rgba(15,23,42,0.55)]">
              <UserRound className="h-5 w-5" />
            </div>
            <div className="min-w-0 flex-1">
              <p className="text-[11px] font-semibold uppercase tracking-[0.2em] text-muted-foreground">
                Signed In Profile
              </p>
              <p className="mt-1 truncate text-sm font-semibold text-foreground">
                {currentUser?.username ?? 'Workspace User'}
              </p>
              <p className="mt-1 flex items-center gap-1.5 truncate text-xs text-muted-foreground">
                <Mail className="h-3.5 w-3.5 shrink-0" />
                <span className="truncate">{currentUser?.email ?? 'No email available'}</span>
              </p>
            </div>
          </div>
          <Button
            variant="outline"
            className="mt-4 h-10 w-full justify-center rounded-2xl border-border/70 bg-background/80 font-semibold"
            onClick={onLogout}
          >
            <LogOut className="mr-2 h-4 w-4" />
            Logout
          </Button>
        </div>
      </div>
    </div>
  );
}

export default function HomePage() {
  const {
    activeTab,
    activeDatasetKey,
    datasets,
    datasetOrder,
    rawData,
    fileName,
    columns,
    previewLoaded,
    loadedRowCount,
    totalRows,
    selectDataset,
    setActiveTab,
    requestUploadPicker,
    resetWorkspace,
    currentUser,
    isAuthenticated,
    setCurrentUser,
    logoutUser,
    hasHydrated,
  } = useAppStore();
  const { toast } = useToast();
  const [isResolvingAuth, setIsResolvingAuth] = React.useState(true);
  const [isRefreshingActivity, setIsRefreshingActivity] = React.useState(false);
  const [isRestoringWorkspace, setIsRestoringWorkspace] = React.useState(false);
  const [recentActivity, setRecentActivity] = React.useState<ActivityResponse['activities'][number] | null>(null);
  const [currentTime, setCurrentTime] = React.useState(() => new Date());
  const lastRestoreDatasetIdRef = React.useRef<string | null>(null);
  const activeTabMeta = tabs.find((t) => t.id === activeTab) ?? tabs[0];
  const availableDatasets = React.useMemo(
    () => datasetOrder.map((key) => datasets[key]).filter(Boolean),
    [datasetOrder, datasets]
  );
  const activeDataset = React.useMemo(
    () => (activeDatasetKey ? datasets[activeDatasetKey] ?? null : null),
    [activeDatasetKey, datasets]
  );
  const displayFileName = fileName ?? activeDataset?.fileName ?? null;
  const displayColumns = columns.length || activeDataset?.columns.length || 0;
  const displayTotalRows = totalRows || activeDataset?.totalRows || 0;
  const displayLoadedRowCount = loadedRowCount || activeDataset?.loadedRowCount || 0;
  const displayPreviewLoaded = previewLoaded || activeDataset?.previewLoaded || false;
  const activeDatasetId = activeDataset?.datasetId ?? null;
  const hasWorkspace = Boolean(rawData?.length || activeDatasetId || displayTotalRows);
  const hasDatasetLibrary = availableDatasets.length > 0;
  const sessionContinuity = getSessionContinuityLabel(recentActivity?.createdAt ?? null);
  const liveIndiaTime = formatIndiaTime(currentTime);
  const liveIndiaDate = formatIndiaDate(currentTime);

  const refreshRecentActivity = React.useCallback(async () => {
    setIsRefreshingActivity(true);
    try {
      const response = await apiClient.get<ActivityResponse>('/activities', {
        params: { limit: 1, dataset_id: activeDatasetId ?? undefined },
      });
      setRecentActivity(response.data.activities[0] ?? null);
    } catch {
      setRecentActivity(null);
    } finally {
      setIsRefreshingActivity(false);
    }
  }, [activeDatasetId]);

  React.useEffect(() => {
    if (!hasHydrated) return;

    let isMounted = true;
    setIsResolvingAuth(true);

    void apiClient
      .get<AuthResponse>('/auth/me')
      .then((response) => {
        if (!isMounted) return;
        setCurrentUser(response.data.user);
      })
      .catch(() => {
        if (!isMounted) return;
        logoutUser();
      })
      .finally(() => {
        if (isMounted) {
          setIsResolvingAuth(false);
        }
      });

    return () => {
      isMounted = false;
    };
  }, [hasHydrated, logoutUser, setCurrentUser]);

  React.useEffect(() => {
    if (!hasHydrated) return;
    void refreshRecentActivity();
  }, [hasHydrated, refreshRecentActivity]);

  React.useEffect(() => {
    const interval = window.setInterval(() => {
      setCurrentTime(new Date());
    }, 30000);
    return () => window.clearInterval(interval);
  }, []);

  React.useEffect(() => {
    if (!hasHydrated) return;

    const interval = window.setInterval(() => {
      void refreshRecentActivity();
    }, 60000);

    const handleVisibilityChange = () => {
      if (document.visibilityState === 'visible') {
        void refreshRecentActivity();
      }
    };

    document.addEventListener('visibilitychange', handleVisibilityChange);
    return () => {
      window.clearInterval(interval);
      document.removeEventListener('visibilitychange', handleVisibilityChange);
    };
  }, [hasHydrated, refreshRecentActivity]);

  React.useEffect(() => {
    if (!hasHydrated) return;
    if (rawData?.length) return;
    if (!activeDataset?.datasetId) return;
    if (lastRestoreDatasetIdRef.current === activeDataset.datasetId) return;

    lastRestoreDatasetIdRef.current = activeDataset.datasetId;
    setIsRestoringWorkspace(true);

    void apiClient
      .get<DatasetPreviewResponse>('/dataset-preview', {
        params: { dataset_id: activeDataset.datasetId },
      })
      .then((response) => {
        const result = response.data;
        const currentState = useAppStore.getState();
        useAppStore.setState({
          fileName: result.fileName ?? currentState.fileName,
          datasetId: result.datasetId,
          rawData: result.data,
          cleanedData: currentState.cleaningDone ? result.data : null,
          columns: result.columns.map((column) => ({
            ...column,
            role: column.role as 'identifier' | 'numeric' | 'categorical' | 'boolean' | 'datetime' | 'unknown',
            sample: Array.isArray(column.sample) ? column.sample : [],
          })),
          totalRows: result.rowCount,
          loadedRowCount: result.loadedRowCount ?? result.data.length,
          previewLoaded: !!result.previewLoaded,
          duplicates: result.duplicates ?? currentState.duplicates,
          reportUrl: null,
        });
      })
      .catch(() => {
        lastRestoreDatasetIdRef.current = null;
      })
      .finally(() => {
        setIsRestoringWorkspace(false);
      });
  }, [activeDataset, hasHydrated, rawData]);

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

  const handleAddDataset = React.useCallback(() => {
    requestUploadPicker(activeTab);
  }, [activeTab, requestUploadPicker]);

  const handleAuthSuccess = React.useCallback((user: AuthenticatedUser) => {
    setCurrentUser(user);
  }, [setCurrentUser]);

  const handleLogout = React.useCallback(async () => {
    try {
      await apiClient.post('/auth/logout');
    } catch (error) {
      toast({
        title: 'Logout warning',
        description: getApiErrorMessage(error, 'The backend session could not be closed cleanly, but the local session was cleared.'),
        variant: 'destructive',
      });
    } finally {
      logoutUser();
    }
  }, [logoutUser, toast]);

  const renderTab = () => {
    switch (activeTab) {
      case 'upload': return <UploadTab />;
      case 'understanding': return <UnderstandingTab />;
      case 'cleaning': return <CleaningTab />;
      case 'eda': return <EdaTab />;
      case 'forecast_ts': return <TimeSeriesForecastTab />;
      case 'forecast_ml': return <MlForecastTab />;
      case 'ml': return <MlTab />;
      case 'prediction': return <PredictionTab />;
      case 'report': return <ReportTab />;
      default: return <UploadTab />;
    }
  };

  if (!hasHydrated || isResolvingAuth) {
    return null;
  }

  if (!isAuthenticated || !currentUser) {
    return <LoginPage onAuthSuccess={handleAuthSuccess} />;
  }

  return (
    <div className="min-h-screen bg-background">
      <div className="pointer-events-none fixed inset-0 -z-10 overflow-hidden">
        <div className="app-grid-bg absolute inset-x-0 top-0 h-[520px] opacity-60" />
        <div className="absolute left-[-8rem] top-20 h-72 w-72 rounded-full bg-sky-400/10 blur-3xl" />
        <div className="absolute right-[-6rem] top-36 h-80 w-80 rounded-full bg-primary/10 blur-3xl" />
        <div className="absolute bottom-0 left-1/3 h-80 w-80 rounded-full bg-emerald-400/10 blur-3xl" />
      </div>
      {/* Desktop Sidebar */}
      <aside className="fixed inset-y-0 left-0 z-40 hidden h-screen w-72 flex-col border-r border-border/70 bg-sidebar/82 shadow-[0_24px_80px_-28px_rgba(15,23,42,0.18)] backdrop-blur-2xl lg:flex">
        <SidebarContent currentUser={currentUser} onLogout={() => void handleLogout()} />
      </aside>

      {/* Main Content */}
      <div className={cn('flex min-w-0 flex-1 flex-col', DESKTOP_SIDEBAR_WIDTH)}>
        {/* Content */}
        <main className="flex-1 overflow-y-auto overflow-x-hidden [scrollbar-width:none] [-ms-overflow-style:none] [&::-webkit-scrollbar]:hidden">
          <div className="mx-auto max-w-7xl px-4 pb-6 pt-3 sm:px-6 sm:pt-4 lg:px-8">
            <div className="sticky top-0 z-30 -mx-4 mb-5 border-b border-border/70 bg-[linear-gradient(180deg,rgba(248,250,252,0.97),rgba(244,247,251,0.94))] px-4 py-3 backdrop-blur-xl dark:bg-[linear-gradient(180deg,rgba(15,23,42,0.96),rgba(15,23,42,0.92))] sm:-mx-6 sm:mb-6 sm:px-6 sm:py-4 lg:-mx-8 lg:px-8">
              <div className="mx-auto max-w-7xl">
                <div className="group relative overflow-hidden rounded-[30px] border border-slate-800/80 bg-[linear-gradient(135deg,#08111f_0%,#13233b_48%,#1d3148_100%)] p-4 text-white shadow-[0_26px_90px_-38px_rgba(15,23,42,0.72)] sm:p-5">
                  <div className="pointer-events-none absolute inset-0 opacity-80">
                    <div className="absolute -left-12 top-8 h-28 w-28 rounded-full bg-sky-400/12 blur-3xl transition-transform duration-700 group-hover:scale-125" />
                    <div className="absolute right-0 top-0 h-36 w-36 rounded-full bg-blue-500/10 blur-3xl transition-transform duration-700 group-hover:translate-x-4 group-hover:-translate-y-2" />
                    <div className="absolute inset-y-0 right-[24%] w-px bg-white/8" />
                    <div className="absolute inset-x-0 top-16 h-px bg-gradient-to-r from-transparent via-white/10 to-transparent" />
                    <div className="absolute inset-x-0 bottom-0 h-px bg-gradient-to-r from-transparent via-white/30 to-transparent" />
                  </div>
                  <div className="flex flex-col gap-4">
                    <div className="flex flex-col gap-3 xl:flex-row xl:items-start xl:justify-between">
                      <div className="flex items-start gap-3">
                        <Sheet>
                          <SheetTrigger asChild>
                            <Button variant="ghost" size="icon" className="mt-0.5 shrink-0 text-white hover:bg-white/10 hover:text-white lg:hidden">
                              <Menu className="h-5 w-5" />
                            </Button>
                          </SheetTrigger>
                          <SheetContent side="left" className="w-72 p-0">
                            <SidebarContent currentUser={currentUser} onLogout={() => void handleLogout()} />
                          </SheetContent>
                        </Sheet>
                        <div className="min-w-0">
                          <div className="flex flex-wrap items-center gap-2">
                            <span className="rounded-full border border-white/12 bg-white/8 px-3 py-1 text-[10px] font-semibold uppercase tracking-[0.24em] text-slate-200">
                              Workspace Console
                            </span>
                            <span className="rounded-full border border-cyan-300/20 bg-cyan-400/10 px-3 py-1 text-[10px] font-medium uppercase tracking-[0.22em] text-cyan-100">
                              {activeTabMeta.label}
                            </span>
                          </div>
                          <div className="flex items-center gap-3">
                            <BrandMark />
                            <BrandWordmark inverted />
                          </div>
                          <p className="mt-3 max-w-2xl text-sm text-slate-300">
                            A focused analytics workspace for guided dataset intake, profiling, cleaning, and forecasting.
                          </p>
                          <div className="mt-4 flex flex-wrap items-center gap-2">
                            <Badge variant="outline" className="rounded-full border-white/15 bg-white/10 px-3 py-1 text-white">
                              {hasWorkspace ? <CheckCircle2 className="mr-2 h-3.5 w-3.5 text-emerald-300" /> : <AlertCircle className="mr-2 h-3.5 w-3.5 text-amber-300" />}
                              {isRestoringWorkspace ? 'Restoring workspace' : hasWorkspace ? 'Workspace in progress' : 'Awaiting dataset'}
                            </Badge>
                            <Badge variant="outline" className="rounded-full border-white/15 bg-white/10 px-3 py-1 text-white">
                              <ShieldCheck className="mr-2 h-3.5 w-3.5 text-sky-300" />
                              PostgreSQL activity tracking connected
                            </Badge>
                            <Badge variant="outline" className="rounded-full border-white/15 bg-white/10 px-3 py-1 text-white">
                              <RefreshCw className={cn('mr-2 h-3.5 w-3.5 text-cyan-300', isRefreshingActivity && 'animate-spin')} />
                              {sessionContinuity.freshness}
                            </Badge>
                          </div>
                        </div>
                      </div>

                      <div className="flex flex-col gap-3 xl:max-w-[62%] xl:items-end">
                        <div className="rounded-[24px] border border-white/12 bg-[linear-gradient(180deg,rgba(255,255,255,0.09),rgba(255,255,255,0.04))] px-4 py-3 text-white shadow-[0_18px_42px_-28px_rgba(15,23,42,0.58)] backdrop-blur-sm">
                          <div className="flex items-start justify-between gap-4">
                            <div>
                              <p className="text-[10px] font-semibold uppercase tracking-[0.22em] text-slate-300">Workspace Time</p>
                              <p className="mt-1 text-xl font-semibold tracking-tight text-white">{liveIndiaTime}</p>
                            </div>
                            <Badge variant="outline" className="rounded-full border-white/12 bg-white/10 px-2.5 py-1 text-[10px] uppercase tracking-[0.18em] text-slate-200">
                              IST
                            </Badge>
                          </div>
                          <div className="mt-2 flex flex-wrap items-center gap-2 text-xs text-slate-300">
                            <span>{liveIndiaDate}</span>
                            <span className="text-slate-500">|</span>
                            <span>{sessionContinuity.status}</span>
                          </div>
                        </div>
                        <div className="flex flex-wrap items-center gap-2 xl:justify-end">
                        <Button size="sm" className="h-9 rounded-full border border-white/10 bg-white px-4 text-slate-950 shadow-sm transition-all duration-300 hover:-translate-y-0.5 hover:bg-slate-100 hover:shadow-lg" onClick={handleResumeWorkspace}>
                          <History className="mr-2 h-4 w-4" />
                          {hasWorkspace ? 'Resume Workspace' : 'Open Workspace'}
                        </Button>
                        <Button size="sm" className="h-9 rounded-full border border-sky-300/20 bg-sky-400/15 px-4 text-white shadow-sm transition-all duration-300 hover:-translate-y-0.5 hover:bg-sky-400/22 hover:shadow-lg" onClick={handleAddDataset}>
                          <Upload className="mr-2 h-4 w-4" />
                          Add Dataset
                        </Button>
                        <Button size="sm" variant="outline" className="h-9 rounded-full border-white/20 bg-white/5 px-4 text-white transition-all duration-300 hover:-translate-y-0.5 hover:bg-white/10 hover:text-white" onClick={handleFreshStart}>
                          <RotateCcw className="mr-2 h-4 w-4" />
                          Fresh Start
                        </Button>
                        <Button size="sm" variant="ghost" className="h-9 rounded-full px-3 text-slate-200 transition-all duration-300 hover:-translate-y-0.5 hover:bg-white/10 hover:text-white" onClick={() => void refreshRecentActivity()}>
                          <RefreshCw className={cn('mr-2 h-4 w-4', isRefreshingActivity && 'animate-spin')} />
                          Sync
                        </Button>
                        <ThemeToggle />
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
            <div className="relative">
              {activeTab !== 'upload' && (
                <div className="hidden">
                  <UploadTab />
                </div>
              )}
              <div className="glass-panel rounded-[30px] border border-border/70 px-3 py-4 shadow-[0_30px_80px_-42px_rgba(15,23,42,0.32)] sm:px-5 sm:py-5">
                <div className="mb-5 flex flex-col gap-3 rounded-[24px] border border-border/70 bg-background/70 px-4 py-3 shadow-[0_18px_50px_-36px_rgba(15,23,42,0.2)] backdrop-blur-sm lg:flex-row lg:items-center lg:justify-between">
                  <div className="min-w-0">
                    <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-muted-foreground">Active Dataset</p>
                    <p className="mt-1 truncate text-base font-semibold text-foreground">
                      {displayFileName ?? 'No dataset selected'}
                    </p>
                    <p className="mt-1 text-xs text-muted-foreground">
                      {hasDatasetLibrary
                        ? `${availableDatasets.length} dataset${availableDatasets.length === 1 ? '' : 's'} available in this workspace.`
                        : 'Upload datasets to build a multi-dataset workspace.'}
                    </p>
                  </div>
                  <div className="flex flex-col gap-2 sm:flex-row sm:items-center">
                    <Select value={activeDatasetKey ?? undefined} onValueChange={selectDataset} disabled={!hasDatasetLibrary}>
                      <SelectTrigger className="w-full min-w-[260px] rounded-2xl border-border/70 bg-card/80 sm:w-[320px]">
                        <SelectValue placeholder="Choose a dataset" />
                      </SelectTrigger>
                      <SelectContent>
                        {availableDatasets.map((dataset) => (
                          <SelectItem key={dataset.key} value={dataset.key}>
                            <span className="flex min-w-0 flex-col">
                              <span className="truncate font-medium">{dataset.fileName ?? dataset.datasetId ?? dataset.key}</span>
                              <span className="text-xs text-muted-foreground">
                                {dataset.totalRows.toLocaleString()} rows | {dataset.columns.length.toLocaleString()} cols
                              </span>
                            </span>
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                    <Button type="button" variant="outline" className="rounded-2xl" onClick={handleAddDataset}>
                      <Upload className="mr-2 h-4 w-4" />
                      Upload Another
                    </Button>
                  </div>
                </div>
                {renderTab()}
              </div>
            </div>
            <StepNavigator showTabs={false} showSwipeHint={false} className="mt-8 mb-2" />
          </div>
        </main>

        {/* Footer */}
        <footer className="mt-auto border-t border-border/70 bg-[linear-gradient(180deg,rgba(248,250,252,0.97),rgba(244,247,251,0.94))] px-4 py-4 backdrop-blur-xl dark:bg-[linear-gradient(180deg,rgba(15,23,42,0.96),rgba(15,23,42,0.92))] sm:px-6 lg:px-8">
            <div className="mx-auto max-w-7xl">
              <div className="flex flex-col gap-2 overflow-hidden rounded-[28px] border border-slate-800/80 bg-[linear-gradient(135deg,#0f172a_0%,#162338_55%,#1e293b_100%)] px-6 py-4 text-base font-bold text-white shadow-[0_26px_90px_-38px_rgba(15,23,42,0.72)] transition-all duration-300 hover:shadow-[0_30px_100px_-40px_rgba(15,23,42,0.78)] lg:flex-row lg:items-center lg:justify-between">
              <span className="bg-[linear-gradient(135deg,#ffffff_0%,#d7f9ff_42%,#67e8f9_100%)] bg-clip-text text-transparent">Intelligent Data Assistant</span>
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
