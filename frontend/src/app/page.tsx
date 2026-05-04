'use client';

import React from 'react';
import Image from 'next/image';
import { useAppStore, TabId, AuthenticatedUser } from '@/lib/store';
import { apiClient, getApiErrorMessage } from '@/lib/api';
import { ThemeToggle } from '@/components/theme-toggle';
import LoginPage from '@/components/login-page';
import StepNavigator from '@/components/step-navigator';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Avatar, AvatarFallback, AvatarImage } from '@/components/ui/avatar';
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from '@/components/ui/dialog';
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
  LogOut,
  Camera,
  Pencil,
  Save,
  ExternalLink,
  TrendingDown,
  TrendingUp,
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
import LossForecastTab from '@/components/tabs/loss-forecast-tab';
import ProfitForecastTab from '@/components/tabs/profit-forecast-tab';
import ReportTab from '@/components/tabs/report-tab';

const tabs: { id: TabId; label: string; icon: React.ElementType }[] = [
  { id: 'upload', label: 'Data Upload', icon: Upload },
  { id: 'understanding', label: 'Data Understanding', icon: Database },
  { id: 'eda', label: 'Exploratory Data Analysis', icon: BarChart3 },
  { id: 'cleaning', label: 'Data Cleaning', icon: Sparkles },
  { id: 'forecast_ts', label: 'Time Series Forecast', icon: LineChart },
  { id: 'forecast_ml', label: 'Machine Learning Forecast', icon: LineChart },
  { id: 'loss_forecast', label: 'Loss Forecast', icon: TrendingDown },
  { id: 'profit_forecast', label: 'Profit Forecast', icon: TrendingUp },
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
  sheetSelection?: {
    availableSheets: Array<{ name: string; rowCount: number; columnCount: number; columns?: string[] }>;
    selectedSheets: string[];
    mergeMode: 'single' | 'stack';
    requiresSelection: boolean;
  } | null;
};

const INDIA_TIMEZONE = 'Asia/Kolkata';
const AROHA_WEBSITE_URL = 'https://aroha.co.in/';
const AROHA_LOGO_URL = 'https://aroha.co.in/wp-content/uploads/2024/08/aroha_logo.png';

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

function BrandWordmark({
  compact = false,
  inverted = false,
}: {
  compact?: boolean;
  inverted?: boolean;
}) {
  return (
    <div className={cn('min-w-0', compact && 'max-w-[168px]')}>
      <h1 className={cn(
        'font-black tracking-[-0.03em]',
        compact ? 'text-[1rem] leading-[1.16] sm:text-[1.08rem]' : 'text-[1.95rem] leading-tight sm:text-[2.35rem]',
        inverted ? 'text-white' : 'text-slate-950'
      )}>
        <span className={cn(
          'bg-clip-text text-transparent',
          inverted
            ? 'bg-[linear-gradient(135deg,#ffffff_0%,#e2efff_48%,#b9ddff_100%)]'
            : 'bg-[linear-gradient(135deg,#234e9e_0%,#2f5fa8_48%,#4cb8f0_100%)]'
        )}>
          Intelligent Data Assistant
        </span>
      </h1>
    </div>
  );
}

function CompanyLogo({ compact = false }: { compact?: boolean }) {
  const width = compact ? 112 : 148;
  const height = compact ? 40 : 52;

  return (
    <img
      src={AROHA_LOGO_URL}
      alt="Aroha Technologies logo"
      width={width}
      height={height}
      className="h-auto w-auto shrink-0 object-contain mix-blend-multiply dark:mix-blend-screen"
    />
  );
}

function ApplicationLogo({ compact = false }: { compact?: boolean }) {
  const size = compact ? 42 : 54;

  return (
    <div
      className={cn(
        'flex shrink-0 items-center justify-center overflow-hidden rounded-xl border border-white/70 bg-white/95 shadow-[0_18px_44px_-28px_rgba(31,95,168,0.55)] ring-1 ring-blue-100/70',
        compact ? 'h-[42px] w-[42px]' : 'h-[54px] w-[54px]'
      )}
    >
      <Image
        src="/app-logo.svg"
        alt="Intelligent Data Assistant logo"
        width={Math.round(size * 0.8)}
        height={Math.round(size * 0.8)}
        priority={false}
        sizes={`${size}px`}
        className="h-auto w-auto object-contain"
      />
    </div>
  );
}

function getUserInitials(user?: AuthenticatedUser | null) {
  const source = user?.username || user?.email || 'User';
  const parts = source.trim().split(/\s+/).filter(Boolean);
  if (parts.length >= 2) {
    return `${parts[0][0]}${parts[1][0]}`.toUpperCase();
  }
  return source.slice(0, 2).toUpperCase();
}

function UserAvatar({ user, className }: { user?: AuthenticatedUser | null; className?: string }) {
  return (
    <Avatar className={cn('border border-white/70 bg-white shadow-[0_14px_32px_-22px_rgba(31,95,168,0.55)]', className)}>
      <AvatarImage src={user?.profileImageDataUrl ?? undefined} alt={user?.username ?? 'User profile'} className="object-cover" />
      <AvatarFallback className="bg-[linear-gradient(135deg,#2f5fa8_0%,#4cb8f0_100%)] text-xs font-bold text-white">
        {getUserInitials(user)}
      </AvatarFallback>
    </Avatar>
  );
}

function UserProfileDialog({
  open,
  onOpenChange,
  currentUser,
  onProfileUpdated,
  onLogout,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  currentUser: AuthenticatedUser;
  onProfileUpdated: (user: AuthenticatedUser) => void;
  onLogout?: () => void;
}) {
  const { toast } = useToast();
  const [name, setName] = React.useState(currentUser.username);
  const [selectedFile, setSelectedFile] = React.useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = React.useState<string | null>(null);
  const [isEditing, setIsEditing] = React.useState(false);
  const [isSaving, setIsSaving] = React.useState(false);
  const fileInputRef = React.useRef<HTMLInputElement | null>(null);

  React.useEffect(() => {
    if (!open) return;
    setName(currentUser.username);
    setSelectedFile(null);
    setPreviewUrl(null);
    setIsEditing(false);
  }, [currentUser.username, open]);

  React.useEffect(() => {
    if (!selectedFile) {
      setPreviewUrl(null);
      return;
    }
    const objectUrl = URL.createObjectURL(selectedFile);
    setPreviewUrl(objectUrl);
    return () => URL.revokeObjectURL(objectUrl);
  }, [selectedFile]);

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (!isEditing) return;
    const file = event.target.files?.[0] ?? null;
    if (!file) return;
    if (!file.type.startsWith('image/')) {
      toast({
        title: 'Unsupported image',
        description: 'Choose a PNG, JPEG, WEBP, or GIF profile image.',
        variant: 'destructive',
      });
      return;
    }
    setSelectedFile(file);
  };

  const handleSaveProfile = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setIsSaving(true);
    try {
      const formData = new FormData();
      formData.append('username', name.trim());
      if (selectedFile) {
        formData.append('profile_image', selectedFile);
      }
      const response = await apiClient.put<AuthResponse>('/auth/profile', formData);
      onProfileUpdated(response.data.user);
      toast({
        title: 'Profile updated',
        description: 'Your profile details were saved successfully.',
      });
      setIsEditing(false);
      onOpenChange(false);
    } catch (error) {
      toast({
        title: 'Profile update failed',
        description: getApiErrorMessage(error, 'We could not update your profile.'),
        variant: 'destructive',
      });
    } finally {
      setIsSaving(false);
    }
  };

  const avatarPreviewUser = {
    ...currentUser,
    username: name || currentUser.username,
    profileImageDataUrl: previewUrl ?? currentUser.profileImageDataUrl,
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="overflow-hidden rounded-[2rem] border-[8px] border-white/80 bg-[#edf1f6] p-0 shadow-[0_34px_110px_-40px_rgba(8,24,58,0.55)] dark:border-white/12 dark:bg-[#122034] sm:max-w-2xl">
        <div className="relative bg-[linear-gradient(135deg,#2f5fa8_0%,#4e8ed3_55%,#67b3df_100%)] px-6 py-6 text-white dark:bg-[linear-gradient(135deg,#14284a_0%,#21558c_54%,#2d7eb2_100%)]">
          <div className="pointer-events-none absolute inset-0 bg-[linear-gradient(130deg,rgba(255,255,255,0.16)_0%,rgba(255,255,255,0.04)_44%,rgba(0,0,0,0.16)_100%)]" />
          <DialogHeader className="relative gap-2 text-left">
            <DialogTitle className="flex items-center gap-3 text-2xl font-bold">
              <UserAvatar user={avatarPreviewUser} className="size-12" />
              User Profile
            </DialogTitle>
            <DialogDescription className="text-white/82">
              Manage your profile information for the Intelligent Data Assistant workspace.
            </DialogDescription>
          </DialogHeader>
          <Button
            type="button"
            size="sm"
            variant="outline"
            className="absolute right-14 top-6 rounded-full border-white/28 bg-white/14 px-4 text-white hover:bg-white/22 hover:text-white"
            onClick={() => setIsEditing((current) => !current)}
          >
            <Pencil className="mr-2 h-4 w-4" />
            {isEditing ? 'Editing' : 'Edit Profile'}
          </Button>
        </div>

        <form onSubmit={handleSaveProfile} className="space-y-5 p-6">
          <div className="flex flex-col gap-5 rounded-2xl border border-white/70 bg-white/70 p-5 shadow-[0_20px_56px_-36px_rgba(31,95,168,0.32)] backdrop-blur-xl dark:border-white/10 dark:bg-white/8 sm:flex-row sm:items-center">
            <div className="relative">
              <UserAvatar user={avatarPreviewUser} className="size-24" />
              <button
                type="button"
                onClick={() => fileInputRef.current?.click()}
                disabled={!isEditing}
                className={cn(
                  'absolute -bottom-1 -right-1 flex h-9 w-9 items-center justify-center rounded-full border border-white/80 bg-[linear-gradient(135deg,#36a74b_0%,#49b653_100%)] text-white shadow-[0_14px_32px_-18px_rgba(34,139,64,0.65)] transition-all duration-300 hover:-translate-y-0.5 hover:brightness-105',
                  !isEditing && 'cursor-not-allowed opacity-55 hover:translate-y-0 hover:brightness-100'
                )}
                aria-label="Upload profile image"
              >
                <Camera className="h-4 w-4" />
              </button>
              <input ref={fileInputRef} type="file" accept="image/png,image/jpeg,image/webp,image/gif" className="hidden" onChange={handleFileChange} />
            </div>
            <div className="min-w-0 flex-1">
              <p className="text-xs font-semibold uppercase tracking-[0.18em] text-muted-foreground">Profile Picture</p>
              <p className="mt-2 text-sm leading-6 text-muted-foreground">
                {isEditing
                  ? 'Upload a square image for best results. PNG, JPEG, WEBP, and GIF files up to 1.5 MB are stored with your account.'
                  : 'Use Edit Profile to update your display name or profile picture.'}
              </p>
              <Button type="button" variant="outline" className="mt-3 rounded-sm" onClick={() => fileInputRef.current?.click()} disabled={!isEditing}>
                <Upload className="mr-2 h-4 w-4" />
                Choose Image
              </Button>
            </div>
          </div>

          <div className="grid gap-4 rounded-2xl border border-white/70 bg-white/70 p-5 shadow-[0_20px_56px_-36px_rgba(31,95,168,0.28)] backdrop-blur-xl dark:border-white/10 dark:bg-white/8 sm:grid-cols-2">
            <div className="space-y-2">
              <Label htmlFor="profile-name">Name</Label>
              <Input id="profile-name" value={name} onChange={(event) => setName(event.target.value)} minLength={3} maxLength={80} required disabled={!isEditing} className="rounded-sm" />
            </div>
            <div className="space-y-2">
              <Label htmlFor="profile-email">Email ID</Label>
              <Input id="profile-email" value={currentUser.email} disabled className="rounded-sm opacity-90" />
            </div>
          </div>

          <DialogFooter className="items-center justify-between gap-3 sm:flex-row">
            <Button type="button" variant="ghost" className="rounded-sm text-muted-foreground" onClick={onLogout}>
              <LogOut className="mr-2 h-4 w-4" />
              Logout
            </Button>
            {isEditing ? (
              <div className="flex flex-col gap-2 sm:flex-row">
                <Button
                  type="button"
                  variant="outline"
                  className="rounded-sm"
                  onClick={() => {
                    setName(currentUser.username);
                    setSelectedFile(null);
                    setPreviewUrl(null);
                    setIsEditing(false);
                  }}
                >
                  Cancel
                </Button>
                <Button type="submit" disabled={isSaving} className="rounded-sm bg-[linear-gradient(135deg,#36a74b_0%,#49b653_100%)] text-white">
                  <Save className="mr-2 h-4 w-4" />
                  {isSaving ? 'Saving...' : 'Save Profile'}
                </Button>
              </div>
            ) : (
              <Button type="button" className="rounded-sm bg-[linear-gradient(135deg,#2f5fa8_0%,#4cb8f0_100%)] text-white" onClick={() => setIsEditing(true)}>
                <Pencil className="mr-2 h-4 w-4" />
                Edit Profile
              </Button>
            )}
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
}

function SidebarContent({
  onNavigate,
  currentUser,
  onLogout,
  onProfileUpdated,
}: {
  onNavigate?: (id: TabId) => void;
  currentUser?: AuthenticatedUser | null;
  onLogout?: () => void;
  onProfileUpdated?: (user: AuthenticatedUser) => void;
}) {
  const { activeTab, setActiveTab, rawData, modelTrained, totalRows } = useAppStore();
  const [profileOpen, setProfileOpen] = React.useState(false);
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
        <div className="group rounded-2xl border border-white/55 bg-white/58 p-3 shadow-[0_18px_48px_-34px_rgba(31,95,168,0.35)] backdrop-blur-xl transition-all duration-300 hover:-translate-y-0.5 hover:bg-white/72 dark:border-white/10 dark:bg-white/8 dark:hover:bg-white/12">
          <a href={AROHA_WEBSITE_URL} target="_blank" rel="noreferrer" aria-label="Open Aroha Technologies website" className="flex items-center gap-3">
            <CompanyLogo compact />
            <ExternalLink className="ml-auto h-3.5 w-3.5 text-muted-foreground transition-transform duration-300 group-hover:-translate-y-0.5 group-hover:translate-x-0.5" />
          </a>
          <div className="mt-3">
            <p className="text-[11px] font-semibold uppercase tracking-[0.24em] text-muted-foreground">Aroha</p>
            <BrandWordmark compact />
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
      <div className="border-t border-white/40 p-4 sm:p-5 dark:border-white/10">
        <button
          type="button"
          onClick={() => setProfileOpen(true)}
          className="group flex w-full items-center gap-3 rounded-2xl border border-white/60 bg-[linear-gradient(135deg,rgba(255,255,255,0.74),rgba(226,239,255,0.66))] p-3 text-left shadow-[0_18px_50px_-32px_rgba(31,95,168,0.32)] backdrop-blur-xl transition-all duration-300 hover:-translate-y-0.5 hover:border-white/80 hover:shadow-[0_22px_58px_-32px_rgba(31,95,168,0.42)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring dark:border-white/10 dark:bg-[linear-gradient(135deg,rgba(255,255,255,0.1),rgba(76,184,240,0.08))]"
        >
          <UserAvatar user={currentUser} className="size-11" />
          <div className="min-w-0 flex-1">
            <p className="truncate text-sm font-semibold text-foreground">
              {currentUser?.username ?? 'Workspace User'}
            </p>
            <p className="mt-0.5 text-xs text-muted-foreground">View profile</p>
          </div>
          <ChevronRight className="h-4 w-4 text-muted-foreground transition-transform duration-300 group-hover:translate-x-1" />
        </button>
        {currentUser && (
          <UserProfileDialog
            open={profileOpen}
            onOpenChange={setProfileOpen}
            currentUser={currentUser}
            onProfileUpdated={(user) => onProfileUpdated?.(user)}
            onLogout={onLogout}
          />
        )}
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
          availableSheets: result.sheetSelection?.availableSheets ?? currentState.availableSheets,
          selectedSheets: result.sheetSelection?.selectedSheets ?? currentState.selectedSheets,
          sheetMergeMode: result.sheetSelection?.mergeMode ?? currentState.sheetMergeMode,
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
      case 'loss_forecast': return <LossForecastTab />;
      case 'profit_forecast': return <ProfitForecastTab />;
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
    <div className="workspace-shell min-h-screen bg-background">
      <div className="pointer-events-none fixed inset-0 -z-10 overflow-hidden">
        <div className="app-grid-bg absolute inset-x-0 top-0 h-[520px] opacity-40" />
        <div className="absolute left-[-8rem] top-20 h-72 w-72 rounded-full bg-blue-200/24 blur-3xl" />
        <div className="absolute right-[-6rem] top-36 h-80 w-80 rounded-full bg-sky-300/20 blur-3xl" />
        <div className="absolute bottom-0 left-1/3 h-80 w-80 rounded-full bg-blue-900/10 blur-3xl" />
      </div>
      {/* Desktop Sidebar */}
      <aside className="fixed inset-y-0 left-0 z-40 hidden h-screen w-72 flex-col border-r border-white/45 bg-sidebar/72 shadow-[0_24px_80px_-30px_rgba(31,95,168,0.22)] backdrop-blur-2xl dark:border-white/10 lg:flex">
        <SidebarContent currentUser={currentUser} onLogout={() => void handleLogout()} onProfileUpdated={setCurrentUser} />
      </aside>

      {/* Main Content */}
      <div className={cn('flex min-w-0 flex-1 flex-col', DESKTOP_SIDEBAR_WIDTH)}>
        {/* Content */}
        <main className="flex-1 overflow-y-auto overflow-x-hidden [scrollbar-width:none] [-ms-overflow-style:none] [&::-webkit-scrollbar]:hidden">
          <div className="mx-auto max-w-7xl px-4 pb-6 pt-3 sm:px-6 sm:pt-4 lg:px-8">
            <div className="sticky top-0 z-30 -mx-4 mb-5 border-b border-white/45 bg-[linear-gradient(180deg,rgba(237,241,246,0.9),rgba(237,241,246,0.72))] px-4 py-3 backdrop-blur-xl dark:border-white/10 dark:bg-[linear-gradient(180deg,rgba(18,32,52,0.9),rgba(18,32,52,0.74))] sm:-mx-6 sm:mb-6 sm:px-6 sm:py-4 lg:-mx-8 lg:px-8">
              <div className="mx-auto max-w-7xl">
                <div className="group relative overflow-hidden rounded-[2rem] border-[8px] border-white/80 bg-[linear-gradient(135deg,#2f5fa8_0%,#4e8ed3_56%,#67b3df_100%)] p-4 text-white shadow-[0_30px_100px_-42px_rgba(31,95,168,0.62)] transition-all duration-500 hover:shadow-[0_34px_110px_-42px_rgba(31,95,168,0.74)] dark:border-white/12 dark:bg-[linear-gradient(135deg,#14284a_0%,#21558c_54%,#2d7eb2_100%)] sm:p-5">
                  <div className="pointer-events-none absolute inset-0 opacity-80">
                    <div className="absolute -left-12 top-8 h-28 w-28 rounded-full bg-white/18 blur-3xl transition-transform duration-700 group-hover:scale-125" />
                    <div className="absolute right-0 top-0 h-36 w-36 rounded-full bg-blue-100/18 blur-3xl transition-transform duration-700 group-hover:translate-x-4 group-hover:-translate-y-2" />
                    <div className="absolute inset-y-0 right-[24%] w-px bg-white/12" />
                    <div className="absolute inset-x-0 top-16 h-px bg-gradient-to-r from-transparent via-white/18 to-transparent" />
                    <div className="absolute inset-x-0 bottom-0 h-px bg-gradient-to-r from-transparent via-white/30 to-transparent" />
                  </div>
                  <motion.div
                    className="pointer-events-none absolute -inset-y-8 -left-1/2 w-1/2 bg-[linear-gradient(90deg,transparent_0%,rgba(255,255,255,0.18)_50%,transparent_100%)]"
                    animate={{ x: ['0%', '300%'] }}
                    transition={{ duration: 7.5, repeat: Infinity, ease: 'linear' }}
                  />
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
                            <SidebarContent currentUser={currentUser} onLogout={() => void handleLogout()} onProfileUpdated={setCurrentUser} />
                          </SheetContent>
                        </Sheet>
                        <div className="min-w-0">
                          <div className="mb-3 flex flex-wrap items-center gap-3">
                            <span className="text-[11px] font-semibold uppercase tracking-[0.32em] text-white/86">
                              Aroha Intelligent System
                            </span>
                            <span className="hidden h-1 w-1 rounded-full bg-slate-400/70 sm:inline-block" />
                            <span className="text-xs font-medium text-white/76">
                              {activeTabMeta.label}
                            </span>
                          </div>
                          <div className="flex items-center gap-3">
                            <ApplicationLogo />
                            <div className="min-w-0">
                              <p className="text-sm font-semibold uppercase tracking-[0.18em] text-white/78">Aroha</p>
                              <BrandWordmark inverted />
                            </div>
                          </div>
                          <p className="mt-3 max-w-2xl text-sm leading-6 text-white/86">
                            Enterprise-ready analytics workspace for dataset intake, understanding, data preparation, and forecasting workflows.
                          </p>
                          <div className="mt-4 flex flex-wrap items-center gap-2">
                            <Badge variant="outline" className="rounded-full !border-white/15 !bg-white/10 px-3 py-1 !text-white">
                              {hasWorkspace ? <CheckCircle2 className="mr-2 h-3.5 w-3.5 text-emerald-300" /> : <AlertCircle className="mr-2 h-3.5 w-3.5 text-amber-300" />}
                              {isRestoringWorkspace ? 'Restoring workspace' : hasWorkspace ? 'Workspace in progress' : 'Awaiting dataset'}
                            </Badge>
                            <Badge variant="outline" className="rounded-full !border-white/15 !bg-white/10 px-3 py-1 !text-white">
                              <ShieldCheck className="mr-2 h-3.5 w-3.5 text-sky-300" />
                              PostgreSQL activity tracking connected
                            </Badge>
                            <Badge variant="outline" className="rounded-full !border-white/15 !bg-white/10 px-3 py-1 !text-white">
                              <RefreshCw className={cn('mr-2 h-3.5 w-3.5 text-cyan-300', isRefreshingActivity && 'animate-spin')} />
                              {sessionContinuity.freshness}
                            </Badge>
                          </div>
                        </div>
                      </div>

                      <div className="flex flex-col gap-3 xl:max-w-[62%] xl:items-end">
                        <div className="rounded-xl border border-white/25 bg-white/12 px-4 py-3 text-white shadow-[0_18px_42px_-28px_rgba(15,23,42,0.45)] backdrop-blur-md transition-all duration-300 hover:-translate-y-0.5 hover:bg-white/16">
                          <div className="flex items-start justify-between gap-4">
                            <div>
                              <p className="text-[10px] font-semibold uppercase tracking-[0.22em] text-slate-300">Workspace Time</p>
                              <p className="mt-1 text-xl font-semibold tracking-tight text-white">{liveIndiaTime}</p>
                            </div>
                            <Badge variant="outline" className="rounded-full !border-white/12 !bg-white/10 px-2.5 py-1 text-[10px] uppercase tracking-[0.18em] !text-slate-200">
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
                        <Button size="sm" className="h-9 rounded-sm border border-white/10 bg-white px-4 text-slate-950 shadow-sm transition-all duration-300 hover:-translate-y-0.5 hover:bg-slate-100 hover:shadow-lg" onClick={handleResumeWorkspace}>
                          <History className="mr-2 h-4 w-4" />
                          {hasWorkspace ? 'Resume Workspace' : 'Open Workspace'}
                        </Button>
                        <Button size="sm" className="h-9 rounded-sm border border-sky-100/25 bg-white/14 px-4 text-white shadow-sm transition-all duration-300 hover:-translate-y-0.5 hover:bg-white/20 hover:shadow-lg" onClick={handleAddDataset}>
                          <Upload className="mr-2 h-4 w-4" />
                          Add Dataset
                        </Button>
                        <Button size="sm" variant="outline" className="h-9 rounded-sm border-white/24 bg-white/8 px-4 text-white transition-all duration-300 hover:-translate-y-0.5 hover:bg-white/14 hover:text-white" onClick={handleFreshStart}>
                          <RotateCcw className="mr-2 h-4 w-4" />
                          Fresh Start
                        </Button>
                        <Button size="sm" variant="ghost" className="h-9 rounded-sm px-3 text-white/86 transition-all duration-300 hover:-translate-y-0.5 hover:bg-white/10 hover:text-white" onClick={() => void refreshRecentActivity()}>
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
              <div className="glass-panel rounded-[2rem] border-[8px] border-white/80 px-3 py-4 shadow-[0_30px_90px_-42px_rgba(31,95,168,0.34)] dark:border-white/10 sm:px-5 sm:py-5">
                <div className="mb-5 flex flex-col gap-3 rounded-xl border border-white/68 bg-white/66 px-4 py-3 shadow-[0_18px_50px_-36px_rgba(31,95,168,0.28)] backdrop-blur-sm transition-all duration-300 hover:bg-white/78 dark:border-white/10 dark:bg-white/8 lg:flex-row lg:items-center lg:justify-between">
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
                      <SelectTrigger className="w-full min-w-[260px] rounded-sm border-border/70 bg-card/80 sm:w-[320px]">
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
                    <Button type="button" variant="outline" className="rounded-sm" onClick={handleAddDataset}>
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
        <footer className="mt-auto border-t border-white/45 bg-[linear-gradient(180deg,rgba(237,241,246,0.96),rgba(230,238,248,0.92))] px-4 py-4 backdrop-blur-xl dark:border-white/10 dark:bg-[linear-gradient(180deg,rgba(18,32,52,0.94),rgba(18,32,52,0.88))] sm:px-6 lg:px-8">
          <div className="mx-auto max-w-7xl">
            <div className="grid gap-5 overflow-hidden rounded-2xl border border-white/75 bg-white/72 px-6 py-5 text-[#1f3340] shadow-[0_22px_70px_-42px_rgba(31,95,168,0.34)] backdrop-blur-xl transition-all duration-300 hover:shadow-[0_26px_80px_-44px_rgba(31,95,168,0.42)] dark:border-white/10 dark:bg-white/8 dark:text-slate-100 lg:grid-cols-[1fr_auto] lg:items-center">
              <div className="flex flex-col gap-3 text-left lg:flex-row lg:items-center">
                <ApplicationLogo />
                <div>
                  <p className="text-xs font-semibold uppercase tracking-[0.22em] text-muted-foreground">Aroha Intelligent Platform</p>
                  <p className="bg-[linear-gradient(135deg,#234e9e_0%,#2f5fa8_48%,#4cb8f0_100%)] bg-clip-text text-base font-bold text-transparent dark:bg-[linear-gradient(135deg,#ffffff_0%,#d7f9ff_42%,#67e8f9_100%)]">
                    Intelligent Data Assistant
                  </p>
                  <p className="mt-1 text-sm font-medium text-muted-foreground">
                    AI-guided dataset understanding, analysis, and predictive modeling.
                  </p>
                </div>
              </div>
              <div className="flex flex-col items-start gap-1.5 text-left text-sm font-semibold text-muted-foreground lg:min-w-[360px]">
                <p>
                  <span className="text-foreground">Location:</span> Bangalore
                </p>
                <p>
                  <span className="text-foreground">Contact:</span>{' '}
                  <a className="text-[#2f5fa8] transition-colors hover:text-[#234e9e] dark:text-cyan-300 dark:hover:text-cyan-200" href="mailto:hr@aroha.co.in">hr@aroha.co.in</a>
                </p>
                <p>
                  <span className="text-foreground">Phone:</span>{' '}
                  <a className="text-[#2f5fa8] transition-colors hover:text-[#234e9e] dark:text-cyan-300 dark:hover:text-cyan-200" href="tel:+919886228615">+91 9886228615</a>
                </p>
                <p className="text-xs uppercase tracking-[0.18em] text-muted-foreground">Bangalore | Aroha Technologies</p>
              </div>
            </div>
          </div>
        </footer>
      </div>
    </div>
  );
}
