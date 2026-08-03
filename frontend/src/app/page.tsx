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
  Mail,
  MapPin,
  Phone,
  Wrench,
  X,
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
// # AGENTIC LAYER START
import AgenticWorkspace from '@/components/agentic/agentic-workspace';
// # AGENTIC LAYER END

const tabs: { id: TabId; label: string; icon: React.ElementType }[] = [
  { id: 'upload', label: 'Data Upload', icon: Upload },
  { id: 'understanding', label: 'Data Understanding', icon: Database },
  { id: 'eda', label: 'Exploratory Data Analysis', icon: BarChart3 },
  { id: 'cleaning', label: 'Data Cleaning', icon: Wrench },
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
const AROHA_LOGO_URL = '/app-logo.svg';

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
        'font-black tracking-normal',
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

function CompanyLogo({ compact = false, mark = false }: { compact?: boolean; mark?: boolean }) {
  const width = mark ? 96 : compact ? 224 : 296;
  const height = mark ? 96 : compact ? 80 : 104;

  return (
    <img
      src={AROHA_LOGO_URL}
      alt="Aroha Technologies logo"
      width={width}
      height={height}
      className={cn(
        'shrink-0 object-contain',
        mark ? 'h-24 w-24' : 'h-auto w-auto'
      )}
    />
  );
}

function getUserInitials(user?: AuthenticatedUser | null) {
  const displayName = user?.username?.trim() || user?.email?.split('@')[0]?.trim() || 'User';
  const parts = displayName.split(/\s+/).filter(Boolean);
  const firstInitial = parts[0]?.[0] ?? 'U';
  const lastInitial = parts.length > 1 ? parts[parts.length - 1]?.[0] : '';
  return `${firstInitial}${lastInitial}`.toUpperCase();
}

function maskEmail(value?: string | null) {
  if (!value || !value.includes('@')) return 'Email hidden';
  const [name, domain] = value.split('@');
  const visibleName = name.length <= 2 ? `${name[0] ?? '*'}***` : `${name.slice(0, 2)}***`;
  const [domainName, ...domainRest] = domain.split('.');
  const visibleDomain = domainName ? `${domainName[0] ?? '*'}***` : '***';
  return `${visibleName}@${visibleDomain}${domainRest.length ? `.${domainRest.join('.')}` : ''}`;
}

function UserAvatar({
  user,
  className,
  fallbackClassName,
}: {
  user?: AuthenticatedUser | null;
  className?: string;
  fallbackClassName?: string;
}) {
  const profileImage = user?.profileImageDataUrl?.trim() || undefined;
  const displayName = user?.username?.trim() || user?.email?.split('@')[0]?.trim() || 'User profile';
  return (
    <Avatar className={cn('overflow-hidden rounded-full border border-white/70 bg-[#2f5fa8] shadow-[0_14px_32px_-22px_rgba(31,95,168,0.55)]', className)}>
      {profileImage ? <AvatarImage src={profileImage} alt={displayName} className="h-full w-full rounded-full object-cover" /> : null}
      <AvatarFallback className={cn('rounded-full bg-[#2f5fa8] text-xs font-bold uppercase text-white', fallbackClassName)}>
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
  const [email, setEmail] = React.useState(currentUser.email);
  const [selectedFile, setSelectedFile] = React.useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = React.useState<string | null>(null);
  const [isEditing, setIsEditing] = React.useState(false);
  const [isSaving, setIsSaving] = React.useState(false);
  const fileInputRef = React.useRef<HTMLInputElement | null>(null);

  React.useEffect(() => {
    if (!open) return;
    setName(currentUser.username);
    setEmail(currentUser.email);
    setSelectedFile(null);
    setPreviewUrl(null);
    setIsEditing(false);
  }, [currentUser.email, currentUser.profileImageDataUrl, currentUser.username, open]);

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
    if (file.size > 1_500_000) {
      toast({
        title: 'Image too large',
        description: 'Profile images must be 1.5 MB or smaller.',
        variant: 'destructive',
      });
      event.target.value = '';
      return;
    }
    setSelectedFile(file);
    setIsEditing(true);
    event.target.value = '';
  };

  const handleSaveProfile = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setIsSaving(true);
    try {
      const formData = new FormData();
      formData.append('username', name.trim());
      formData.append('email', email.trim());
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
  const displayName = name.trim() || currentUser.username;
  const displayEmail = email.trim() || currentUser.email;
  const privateEmailLabel = maskEmail(displayEmail);
  const hasProfileImage = Boolean(previewUrl || currentUser.profileImageDataUrl);
  const hasChanges =
    name.trim() !== currentUser.username ||
    email.trim() !== currentUser.email ||
    Boolean(selectedFile);
  const profileStatus = hasProfileImage ? 'Complete' : 'Needs photo';

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="overflow-hidden rounded-xl border border-white/80 bg-[#f5f8fc] p-0 shadow-[0_34px_110px_-40px_rgba(8,24,58,0.55)] dark:border-white/12 dark:bg-[#101b2c] sm:max-w-3xl">
        <div className="relative overflow-hidden bg-[linear-gradient(135deg,#123767_0%,#2565a7_58%,#3f9ccc_100%)] px-7 py-7 text-white dark:bg-[linear-gradient(135deg,#14284a_0%,#21558c_54%,#2d7eb2_100%)]">
          <div className="pointer-events-none absolute inset-0 bg-[linear-gradient(130deg,rgba(255,255,255,0.18)_0%,rgba(255,255,255,0.05)_42%,rgba(0,0,0,0.18)_100%)]" />
          <div className="pointer-events-none absolute -right-12 -top-16 h-44 w-44 rounded-full border border-white/18" />
          <div className="pointer-events-none absolute -bottom-24 left-20 h-52 w-52 rounded-full bg-cyan-200/12 blur-2xl" />
          <DialogHeader className="relative max-w-[29rem] gap-2 text-left">
            <DialogTitle className="text-2xl font-semibold tracking-normal">Account Profile</DialogTitle>
            <DialogDescription className="text-sm leading-6 text-white/82">
              Private workspace identity, access status, and profile image.
            </DialogDescription>
          </DialogHeader>
          <Button
            type="button"
            size="sm"
            variant="outline"
            className="absolute right-14 top-7 h-9 rounded-lg border-white/28 bg-white/14 px-4 text-white shadow-[0_14px_34px_-24px_rgba(255,255,255,0.95)] hover:bg-white/22 hover:text-white"
            onClick={() => setIsEditing((current) => !current)}
          >
            <Pencil className="mr-2 h-4 w-4" />
            {isEditing ? 'Editing' : 'Edit Profile'}
          </Button>
        </div>

        <form onSubmit={handleSaveProfile} className="space-y-5 p-6">
          <section className="grid gap-5 rounded-lg border border-slate-200/90 bg-white p-5 shadow-[0_20px_56px_-42px_rgba(31,95,168,0.42)] dark:border-white/10 dark:bg-white/8 md:grid-cols-[14rem_1fr]">
            <div className="flex flex-col items-center justify-center rounded-lg border border-slate-200/80 bg-[linear-gradient(180deg,#f8fbff,#eef5fb)] p-5 dark:border-white/10 dark:bg-white/5">
              <div className="relative">
                <UserAvatar user={avatarPreviewUser} className="size-28 border-4 border-white shadow-[0_22px_48px_-28px_rgba(15,23,42,0.72)]" fallbackClassName="text-3xl" />
                <button
                  type="button"
                  onClick={() => fileInputRef.current?.click()}
                  className="absolute -bottom-1 -right-1 flex h-10 w-10 items-center justify-center rounded-full border-2 border-white bg-[linear-gradient(135deg,#1f6fb8_0%,#45b3e7_100%)] text-white shadow-[0_16px_34px_-18px_rgba(31,111,184,0.8)] transition-all duration-300 hover:-translate-y-0.5 hover:brightness-105"
                  aria-label="Upload profile image"
                >
                  <Camera className="h-4 w-4" />
                </button>
                <input ref={fileInputRef} type="file" accept="image/png,image/jpeg,image/webp,image/gif" className="hidden" onChange={handleFileChange} aria-label="Upload profile image" />
              </div>
              <Badge className="mt-4 rounded-md border border-blue-100 bg-blue-50 text-blue-700 hover:bg-blue-50 dark:border-blue-400/20 dark:bg-blue-400/10 dark:text-blue-100">
                {profileStatus}
              </Badge>
              <p className="mt-3 text-center text-xs leading-5 text-muted-foreground">Profile details are shown with privacy controls by default.</p>
            </div>

            <div className="min-w-0 space-y-5">
              <div>
                <p className="text-xs font-semibold uppercase tracking-[0.18em] text-muted-foreground">Workspace Account</p>
                <h3 className="mt-2 truncate text-2xl font-semibold tracking-normal text-slate-950 dark:text-white">{displayName}</h3>
                <div className="mt-2 flex min-w-0 items-center gap-2 text-sm text-muted-foreground">
                  <ShieldCheck className="h-4 w-4 shrink-0 text-emerald-600" />
                  <span className="truncate">Email hidden for privacy ({privateEmailLabel})</span>
                </div>
              </div>

              <div className="grid gap-3 sm:grid-cols-3">
                <div className="rounded-lg border border-slate-200 bg-slate-50 px-4 py-3 dark:border-white/10 dark:bg-white/5">
                  <p className="text-[11px] font-semibold uppercase tracking-[0.16em] text-muted-foreground">Access</p>
                  <p className="mt-1 flex items-center gap-2 text-sm font-semibold text-slate-900 dark:text-white">
                    <ShieldCheck className="h-4 w-4 text-emerald-600" />
                    Secure workspace
                  </p>
                </div>
                <div className="rounded-lg border border-slate-200 bg-slate-50 px-4 py-3 dark:border-white/10 dark:bg-white/5">
                  <p className="text-[11px] font-semibold uppercase tracking-[0.16em] text-muted-foreground">Privacy</p>
                  <p className="mt-1 flex items-center gap-2 text-sm font-semibold text-slate-900 dark:text-white">
                    <CheckCircle2 className="h-4 w-4 text-blue-600" />
                    Email protected
                  </p>
                </div>
                <div className="rounded-lg border border-slate-200 bg-slate-50 px-4 py-3 dark:border-white/10 dark:bg-white/5">
                  <p className="text-[11px] font-semibold uppercase tracking-[0.16em] text-muted-foreground">Mode</p>
                  <p className="mt-1 flex items-center gap-2 text-sm font-semibold text-slate-900 dark:text-white">
                    <Pencil className="h-4 w-4 text-sky-600" />
                    {isEditing ? 'Editing' : 'Review'}
                  </p>
                </div>
              </div>

              <Button type="button" variant="outline" className="h-10 rounded-lg" onClick={() => fileInputRef.current?.click()}>
                <Upload className="mr-2 h-4 w-4" />
                Choose Image
              </Button>
            </div>
          </section>

          <section className="grid gap-4 rounded-lg border border-slate-200/90 bg-white p-5 shadow-[0_20px_56px_-42px_rgba(31,95,168,0.32)] dark:border-white/10 dark:bg-white/8 sm:grid-cols-2">
            <div className="space-y-2">
              <Label htmlFor="profile-name">Name</Label>
              <Input id="profile-name" value={name} onChange={(event) => setName(event.target.value)} minLength={3} maxLength={80} required disabled={!isEditing} className="h-11 rounded-lg bg-slate-50 disabled:opacity-100 dark:bg-white/5" />
            </div>
            <div className="space-y-2">
              <Label htmlFor="profile-email">Private Email</Label>
              <Input
                id="profile-email"
                type={isEditing ? 'email' : 'text'}
                value={isEditing ? email : privateEmailLabel}
                onChange={(event) => setEmail(event.target.value)}
                required
                disabled={!isEditing}
                className="h-11 rounded-lg bg-slate-50 disabled:opacity-100 dark:bg-white/5"
              />
            </div>
          </section>

          <DialogFooter className="items-center justify-between gap-3 border-t border-slate-200 pt-5 dark:border-white/10 sm:flex-row">
            <Button type="button" variant="ghost" className="h-10 rounded-lg text-muted-foreground hover:text-red-600" onClick={onLogout}>
              <LogOut className="mr-2 h-4 w-4" />
              Logout
            </Button>
            {isEditing ? (
              <div className="flex flex-col gap-2 sm:flex-row">
                <Button
                  type="button"
                  variant="outline"
                  className="h-10 rounded-lg"
                  onClick={() => {
                    setName(currentUser.username);
                    setEmail(currentUser.email);
                    setSelectedFile(null);
                    setPreviewUrl(null);
                    setIsEditing(false);
                  }}
                >
                  Cancel
                </Button>
                <Button type="submit" disabled={isSaving || !hasChanges} className="h-10 rounded-lg bg-[linear-gradient(135deg,#1f6fb8_0%,#45b3e7_100%)] text-white">
                  <Save className="mr-2 h-4 w-4" />
                  {isSaving ? 'Saving...' : 'Save Profile'}
                </Button>
              </div>
            ) : (
              <Button type="button" className="h-10 rounded-lg bg-[linear-gradient(135deg,#1f6fb8_0%,#45b3e7_100%)] text-white" onClick={() => setIsEditing(true)}>
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
  const displayName = currentUser?.username?.trim() || 'Workspace User';
  const accessLevel = 'Secure Workspace';

  const isTabEnabled = (tabId: TabId) => {
    if (tabId === 'upload') return true;
    if (!hasDatasetContext) return false;
    if (tabId === 'prediction' && !modelTrained) return false;
    return true;
  };

  return (
    <div className="relative flex h-full flex-col overflow-hidden">
      <div className="pointer-events-none absolute inset-x-0 top-0 h-44 bg-[radial-gradient(circle_at_24%_10%,rgba(96,165,250,0.22),transparent_44%),linear-gradient(180deg,rgba(255,255,255,0.58),transparent)] dark:bg-[radial-gradient(circle_at_24%_10%,rgba(96,165,250,0.16),transparent_44%),linear-gradient(180deg,rgba(255,255,255,0.08),transparent)]" />
      {/* Logo */}
      <div className="relative px-5 pb-4 pt-5 sm:px-6 sm:pt-6">
        <div className="group relative bg-transparent p-0 shadow-none transition-all duration-300 hover:-translate-y-0.5 dark:bg-transparent dark:shadow-none">
          <a href={AROHA_WEBSITE_URL} target="_blank" rel="noreferrer" aria-label="Open Aroha Technologies website" className="relative flex items-center gap-3">
            <span className="flex h-24 w-56 shrink-0 items-center justify-start bg-transparent shadow-none dark:bg-transparent dark:shadow-none">
              <CompanyLogo compact />
            </span>
            <span className="ml-auto flex h-8 w-8 items-center justify-center rounded-full border border-white/70 bg-white/58 text-muted-foreground shadow-sm transition-all duration-300 group-hover:-translate-y-0.5 group-hover:translate-x-0.5 group-hover:text-primary dark:border-white/10 dark:bg-white/8">
              <ExternalLink className="h-3.5 w-3.5" />
            </span>
          </a>
          <div className="relative mt-4 min-w-0">
            <p className="text-[10px] font-bold uppercase tracking-[0.28em] text-[#5f7288] dark:text-slate-300">Aroha Intelligent Platform</p>
            <BrandWordmark compact />
            <div className="mt-3 flex items-center gap-2 rounded-full border border-blue-100/80 bg-white/54 px-3 py-1.5 text-[11px] font-semibold text-[#2f5fa8] shadow-sm backdrop-blur-xl dark:border-white/10 dark:bg-white/8 dark:text-cyan-200">
              <ShieldCheck className="h-3.5 w-3.5" />
              Secure analytics workspace
            </div>
          </div>
        </div>
      </div>
      <Separator className="relative opacity-45" />

      {/* Navigation */}
      <div className="relative flex-1 overflow-y-auto px-3 py-3 sm:py-4 [scrollbar-width:none] [-ms-overflow-style:none] [&::-webkit-scrollbar]:hidden">
        <div className="mb-3 flex items-center justify-between px-3">
          <p className="text-[10px] font-bold uppercase tracking-[0.24em] text-muted-foreground">Workspace Flow</p>
          <span className="rounded-full border border-blue-100/80 bg-white/58 px-2 py-0.5 text-[10px] font-semibold text-[#2f5fa8] dark:border-white/10 dark:bg-white/8 dark:text-cyan-200">
            {hasDatasetContext ? 'Active' : 'Start'}
          </span>
        </div>
        <nav className="flex flex-col gap-1.5">
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
                    'group relative flex w-full items-center gap-3 overflow-hidden rounded-xl border px-3 py-3 text-left text-sm font-semibold transition-all duration-300',
                    isActive && enabled && 'border-blue-200/90 bg-[linear-gradient(135deg,rgba(47,95,168,0.18),rgba(76,184,240,0.12))] text-[#234e9e] shadow-[0_18px_44px_-26px_rgba(37,99,235,0.55)] ring-1 ring-blue-100/70 dark:border-white/12 dark:text-cyan-100 dark:ring-white/10',
                    !isActive && enabled && 'border-transparent bg-white/0 text-muted-foreground hover:border-white/70 hover:bg-white/64 hover:text-foreground hover:shadow-[0_14px_36px_-28px_rgba(15,23,42,0.22)] dark:hover:border-white/10 dark:hover:bg-white/8',
                    !enabled && 'cursor-not-allowed border-transparent text-muted-foreground/38',
                  )}
                >
                  <div className="pointer-events-none absolute inset-0 opacity-0 transition-opacity duration-300 group-hover:opacity-100" aria-hidden>
                    <div className="absolute inset-y-0 left-0 w-24 bg-gradient-to-r from-primary/10 to-transparent" />
                    <div className="absolute inset-x-4 top-0 h-px bg-white/70" />
                  </div>
                  {isActive && enabled && (
                    <motion.div
                      layoutId="activeTab"
                      className="absolute left-0 top-2 bottom-2 w-1 rounded-r-full bg-[linear-gradient(180deg,#2f5fa8,#4cb8f0)]"
                      transition={{ type: 'spring', stiffness: 300, damping: 30 }}
                    />
                  )}
                  <div className={cn(
                    'flex h-10 w-10 shrink-0 items-center justify-center rounded-xl border transition-all duration-300',
                    isActive && enabled && 'border-white/70 bg-[linear-gradient(135deg,#2f5fa8,#4cb8f0)] text-white shadow-[0_14px_28px_-18px_rgba(47,95,168,0.72)] ring-4 ring-blue-200/55 dark:border-white/12 dark:ring-cyan-300/10',
                    !isActive && enabled && 'border-white/70 bg-white/58 text-[#456176] shadow-sm group-hover:scale-105 group-hover:border-blue-100 group-hover:bg-blue-50/80 group-hover:text-[#2f5fa8] dark:border-white/10 dark:bg-white/8 dark:text-slate-300',
                    !enabled && 'bg-muted/50 text-muted-foreground/40',
                  )}>
                    <Icon className="h-4 w-4" />
                  </div>
                  <div className="min-w-0 flex-1">
                    <span className="truncate">{tab.label}</span>
                  </div>
                  <ChevronRight className={cn(
                    'ml-auto h-4 w-4 transition-all duration-300',
                    isActive && enabled ? 'translate-x-0 text-[#2f5fa8] opacity-100 dark:text-cyan-200' : 'text-muted-foreground opacity-0 group-hover:translate-x-0.5 group-hover:opacity-70',
                  )} />
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
      <div className="border-t border-white/40 bg-[linear-gradient(180deg,rgba(255,255,255,0.1),rgba(219,234,254,0.32))] p-4 sm:p-5 dark:border-white/10 dark:bg-[linear-gradient(180deg,rgba(255,255,255,0.02),rgba(59,130,246,0.08))]">
        <button
          type="button"
          onClick={() => setProfileOpen(true)}
          className="group relative w-full overflow-hidden rounded-2xl border border-white/75 bg-white/82 p-3 text-left shadow-[0_18px_52px_-34px_rgba(31,95,168,0.42)] ring-1 ring-blue-100/70 backdrop-blur-xl transition-all duration-300 hover:-translate-y-0.5 hover:border-blue-200 hover:bg-white/94 hover:shadow-[0_24px_64px_-36px_rgba(31,95,168,0.5)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring dark:border-white/10 dark:bg-white/8 dark:ring-white/10 dark:hover:bg-white/12"
        >
          <div className="pointer-events-none absolute inset-0 bg-[linear-gradient(135deg,rgba(255,255,255,0.62),transparent_48%,rgba(76,184,240,0.08))]" />
          <div className="relative flex items-center gap-3">
            <UserAvatar
              user={currentUser}
              className="h-8 w-8 shrink-0 border-0 shadow-[0_12px_26px_-18px_rgba(31,95,168,0.8)] ring-2 ring-white/80 dark:ring-white/15"
              fallbackClassName="text-xs"
            />
            <div className="min-w-0 flex-1">
              <p className="truncate text-sm font-medium leading-5 text-slate-900 dark:text-slate-50">
                {displayName}
              </p>
              <p className="truncate text-xs leading-4 text-slate-500 dark:text-slate-300">{accessLevel}</p>
            </div>
            <span className="shrink-0 text-xs font-medium text-[#2f5fa8] transition-colors duration-300 group-hover:text-[#244f8d] dark:text-cyan-100 dark:group-hover:text-cyan-50">
              Manage
            </span>
          </div>
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

const AGENTIC_TOOLTIP_STORAGE_KEY = 'ida_agent_tooltip_dismissed';

function AgenticCoreLauncher({ onOpen, isProcessing }: { onOpen: () => void; isProcessing: boolean }) {
  const [showTooltip, setShowTooltip] = React.useState(false);

  React.useEffect(() => {
    try {
      setShowTooltip(window.localStorage.getItem(AGENTIC_TOOLTIP_STORAGE_KEY) !== 'true');
    } catch {
      setShowTooltip(false);
    }
  }, []);

  const dismissTooltip = () => {
    try {
      window.localStorage.setItem(AGENTIC_TOOLTIP_STORAGE_KEY, 'true');
    } catch {
      // Ignore storage failures; the button should remain usable.
    }
    setShowTooltip(false);
  };

  return (
    <div className="fixed bottom-6 right-6 z-[9999] flex flex-col items-end gap-3">
      {showTooltip && (
        <div className="relative max-w-[260px] rounded-lg border border-slate-200 bg-white px-4 py-3 pr-10 text-sm font-medium leading-5 text-slate-800 shadow-[0_22px_58px_-26px_rgba(15,23,42,0.45)] ring-1 ring-slate-900/5 dark:border-white/10 dark:bg-slate-950 dark:text-slate-100 dark:ring-white/10">
          <button
            type="button"
            aria-label="Dismiss Agentic Core tip"
            onClick={dismissTooltip}
            className="absolute right-2 top-2 grid h-6 w-6 place-items-center rounded-full text-slate-400 transition-colors hover:bg-slate-100 hover:text-slate-700 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 dark:hover:bg-white/10 dark:hover:text-white"
          >
            <X className="h-3.5 w-3.5" />
          </button>
          Your AI agent — click to automate your full workflow
          <span className="absolute -bottom-1.5 right-8 h-3 w-3 rotate-45 border-b border-r border-slate-200 bg-white dark:border-white/10 dark:bg-slate-950" />
        </div>
      )}
      <button
        type="button"
        aria-label="Open IDA Agentic Core"
        onClick={onOpen}
        className="inline-flex min-h-12 items-center gap-2.5 rounded-full border border-slate-200/80 bg-white py-2 pl-2 pr-3.5 text-left text-slate-900 shadow-[0_20px_52px_-24px_rgba(17,24,39,0.44)] ring-1 ring-slate-900/5 transition-all duration-200 hover:-translate-y-0.5 hover:border-blue-200 hover:shadow-[0_24px_58px_-24px_rgba(17,24,39,0.54)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 dark:border-white/12 dark:bg-slate-950 dark:text-white dark:ring-white/10"
      >
        <span className="grid h-8 w-8 place-items-center rounded-full bg-[linear-gradient(135deg,#234e9e_0%,#2f5fa8_48%,#4cb8f0_100%)] text-[10px] font-extrabold tracking-normal text-white shadow-[0_10px_22px_-14px_rgba(47,95,168,0.9)]">
          IDA
        </span>
        <span className="text-[13px] font-medium leading-none">Agentic Core</span>
        {isProcessing ? (
          <span className="h-3.5 w-3.5 rounded-full border-2 border-blue-500 border-t-transparent animate-spin" aria-label="Agent processing" />
        ) : (
          <span className="relative flex h-3.5 w-3.5 items-center justify-center" aria-label="Agent ready">
            <span className="absolute h-3.5 w-3.5 animate-ping rounded-full bg-emerald-400/55" />
            <span className="relative h-2.5 w-2.5 rounded-full bg-emerald-500 ring-2 ring-emerald-100 dark:ring-emerald-400/20" />
          </span>
        )}
      </button>
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
    // # AGENTIC LAYER START
    agenticEnabled,
    agenticStepStatuses,
    // # AGENTIC LAYER END
  } = useAppStore();
  const { toast } = useToast();
  const [isResolvingAuth, setIsResolvingAuth] = React.useState(true);
  const [isRefreshingActivity, setIsRefreshingActivity] = React.useState(false);
  const [isRestoringWorkspace, setIsRestoringWorkspace] = React.useState(false);
  const [agenticWorkspaceOpen, setAgenticWorkspaceOpen] = React.useState(false);
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
  const isAgenticProcessing = Object.values(agenticStepStatuses).some((status) => status === 'running');
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
      <aside className="fixed inset-y-0 left-0 z-40 hidden h-screen w-72 flex-col border-r border-white/60 bg-[linear-gradient(180deg,rgba(237,241,246,0.92),rgba(225,235,247,0.82))] shadow-[18px_0_70px_-46px_rgba(31,95,168,0.5)] backdrop-blur-2xl dark:border-white/10 dark:bg-[linear-gradient(180deg,rgba(20,36,58,0.94),rgba(15,29,48,0.88))] lg:flex">
        <SidebarContent currentUser={currentUser} onLogout={() => void handleLogout()} onProfileUpdated={setCurrentUser} />
      </aside>
      {agenticEnabled && <AgenticCoreLauncher onOpen={() => setAgenticWorkspaceOpen(true)} isProcessing={isAgenticProcessing} />}
      <Dialog open={agenticWorkspaceOpen} onOpenChange={setAgenticWorkspaceOpen}>
        <DialogContent className="max-h-[88vh] overflow-y-auto rounded-xl border border-white/80 bg-[#f5f8fc] p-4 shadow-[0_34px_110px_-40px_rgba(8,24,58,0.55)] dark:border-white/12 dark:bg-[#101b2c] sm:max-w-6xl">
          <DialogHeader className="text-left">
            <DialogTitle>IDA Agentic Core</DialogTitle>
            <DialogDescription>
              Agentic workspace for the active dataset inside Intelligent Data Assistant.
            </DialogDescription>
          </DialogHeader>
          <AgenticWorkspace datasetId={activeDatasetId} fileName={displayFileName} />
        </DialogContent>
      </Dialog>

      {/* Main Content */}
      <div className={cn('flex min-w-0 flex-1 flex-col', DESKTOP_SIDEBAR_WIDTH)}>
        {/* Content */}
        <main className="flex-1 overflow-y-auto overflow-x-hidden [scrollbar-width:none] [-ms-overflow-style:none] [&::-webkit-scrollbar]:hidden">
          <div className="mx-auto max-w-7xl px-4 pb-6 pt-3 sm:px-6 sm:pt-4 lg:px-8">
            <div className="sticky top-0 z-30 -mx-4 mb-5 border-b border-white/55 bg-[linear-gradient(180deg,rgba(237,241,246,0.92),rgba(237,241,246,0.74))] px-4 py-3 backdrop-blur-2xl dark:border-white/10 dark:bg-[linear-gradient(180deg,rgba(18,32,52,0.92),rgba(18,32,52,0.76))] sm:-mx-6 sm:mb-6 sm:px-6 sm:py-4 lg:-mx-8 lg:px-8">
              <div className="mx-auto max-w-7xl">
                <div className="group relative overflow-hidden rounded-xl border border-white/70 bg-[linear-gradient(135deg,rgba(28,73,124,0.98)_0%,rgba(50,112,181,0.96)_56%,rgba(74,154,205,0.94)_100%)] p-4 text-white shadow-[0_28px_82px_-44px_rgba(31,95,168,0.72)] ring-1 ring-blue-100/30 backdrop-blur-2xl transition-all duration-500 hover:shadow-[0_32px_92px_-44px_rgba(31,95,168,0.8)] dark:border-white/12 dark:bg-[linear-gradient(135deg,rgba(20,40,74,0.96)_0%,rgba(33,85,140,0.92)_54%,rgba(45,126,178,0.9)_100%)] sm:p-5">
                  <div className="pointer-events-none absolute inset-0 opacity-80">
                    <div className="absolute -left-12 top-8 h-28 w-28 rounded-full bg-white/16 blur-3xl transition-transform duration-700 group-hover:scale-125" />
                    <div className="absolute right-0 top-0 h-36 w-36 rounded-full bg-blue-100/16 blur-3xl transition-transform duration-700 group-hover:translate-x-4 group-hover:-translate-y-2" />
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
                          <div className="mb-3 text-xs font-semibold text-white/78">{activeTabMeta.label}</div>
                          <div className="flex items-center gap-4">
                            <CompanyLogo mark />
                            <div className="min-w-0">
                              <p className="text-xs font-bold uppercase tracking-widest text-white/72">AROHA TECHNOLOGIES</p>
                              <h1 className="mt-1 text-3xl font-black leading-tight tracking-normal text-white sm:text-4xl">
                                Intelligent Data Assistant
                              </h1>
                            </div>
                          </div>
                          <p className="mt-3 max-w-2xl text-sm font-medium leading-6 text-white/84">
                            Enterprise-ready analytics workspace for dataset intake, understanding, data preparation, and forecasting workflows.
                          </p>
                          <div className="mt-4 flex flex-wrap items-center gap-2">
                            <Badge variant="outline" className="rounded-full !border-white/18 !bg-white/12 px-3 py-1 !text-white shadow-sm backdrop-blur-md">
                              {hasWorkspace ? <CheckCircle2 className="mr-2 h-3.5 w-3.5 text-emerald-300" /> : <AlertCircle className="mr-2 h-3.5 w-3.5 text-amber-300" />}
                              {isRestoringWorkspace ? 'Restoring workspace' : hasWorkspace ? 'Workspace in progress' : 'Awaiting dataset'}
                            </Badge>
                            <Badge variant="outline" className="rounded-full !border-white/18 !bg-white/12 px-3 py-1 !text-white shadow-sm backdrop-blur-md">
                              <ShieldCheck className="mr-2 h-3.5 w-3.5 text-sky-300" />
                              PostgreSQL activity tracking connected
                            </Badge>
                          </div>
                        </div>
                      </div>

                      <div className="flex flex-col gap-3 xl:max-w-[62%] xl:items-end">
                        <div className="rounded-xl border border-white/25 bg-white/12 px-4 py-3 text-white shadow-[0_18px_42px_-28px_rgba(15,23,42,0.45)] ring-1 ring-white/8 backdrop-blur-md transition-all duration-300 hover:-translate-y-0.5 hover:bg-white/16">
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
                          </div>
                        </div>
                        <div className="flex flex-wrap items-center gap-2 xl:justify-end">
                        <Button size="sm" className="h-9 rounded-sm border border-white/20 bg-white px-4 text-slate-950 shadow-sm transition-all duration-300 hover:-translate-y-0.5 hover:bg-slate-100 hover:shadow-lg" onClick={handleResumeWorkspace}>
                          <History className="mr-2 h-4 w-4" />
                          {hasWorkspace ? 'Resume Workspace' : 'Open Workspace'}
                        </Button>
                        <Button size="sm" className="h-9 rounded-sm border border-sky-100/25 bg-white/14 px-4 text-white shadow-sm backdrop-blur-md transition-all duration-300 hover:-translate-y-0.5 hover:bg-white/20 hover:shadow-lg" onClick={handleAddDataset}>
                          <Upload className="mr-2 h-4 w-4" />
                          Add Dataset
                        </Button>
                        <Button size="sm" variant="outline" className="h-9 rounded-sm border-white/24 bg-white/8 px-4 text-white backdrop-blur-md transition-all duration-300 hover:-translate-y-0.5 hover:bg-white/14 hover:text-white" onClick={handleFreshStart}>
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
              <div className="glass-panel rounded-xl border border-white/78 px-3 py-4 shadow-[0_26px_76px_-42px_rgba(31,95,168,0.36)] ring-1 ring-white/58 dark:border-white/10 dark:ring-white/8 sm:px-5 sm:py-5">
                <div className="mb-5 flex flex-col gap-3 rounded-xl border border-white/72 bg-white/70 px-4 py-3 shadow-[0_18px_50px_-36px_rgba(31,95,168,0.28)] ring-1 ring-white/54 backdrop-blur-xl transition-all duration-300 hover:bg-white/82 dark:border-white/10 dark:bg-white/8 dark:ring-white/8 lg:flex-row lg:items-center lg:justify-between">
                  <div className="min-w-0">
                    <p className="text-[11px] font-bold uppercase tracking-[0.2em] text-muted-foreground">Active Dataset</p>
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
                      <SelectTrigger className="w-full min-w-[260px] rounded-sm border-border/70 bg-card/90 shadow-sm sm:w-[320px]">
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
                    <Button type="button" variant="outline" className="rounded-sm bg-white/80 shadow-sm dark:bg-white/8" onClick={handleAddDataset}>
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
        <footer className="mt-auto rounded-t-xl border-t border-white/45 bg-[linear-gradient(180deg,rgba(237,241,246,0.96),rgba(230,238,248,0.92))] px-4 py-3 text-muted-foreground shadow-[0_-14px_46px_-34px_rgba(31,95,168,0.38)] backdrop-blur-xl dark:border-gray-700 dark:bg-gray-900 dark:text-gray-400 sm:px-6 lg:px-8">
          <div className="mx-auto flex h-12 max-w-7xl items-center gap-6 overflow-hidden">
            <div className="flex shrink-0 items-center gap-5 text-sm font-semibold">
              <span className="inline-flex items-center gap-2 whitespace-nowrap">
                <MapPin className="h-4 w-4 text-[#2f5fa8] dark:text-cyan-300" />
                Bangalore
              </span>
              <a className="inline-flex items-center gap-2 whitespace-nowrap text-[#2f5fa8] transition-colors hover:text-[#234e9e] dark:text-cyan-300 dark:hover:text-cyan-200" href="mailto:hr@aroha.co.in">
                <Mail className="h-4 w-4" />
                hr@aroha.co.in
              </a>
              <a className="inline-flex items-center gap-2 whitespace-nowrap text-[#2f5fa8] transition-colors hover:text-[#234e9e] dark:text-cyan-300 dark:hover:text-cyan-200" href="tel:+919886228615">
                <Phone className="h-4 w-4" />
                +91 9886228615
              </a>
            </div>
            <div className="relative min-w-0 flex-1 overflow-hidden py-2">
              <div className="pointer-events-none absolute inset-y-0 left-0 z-10 w-12 bg-[linear-gradient(90deg,rgba(237,241,246,0.96),rgba(237,241,246,0))] dark:bg-[linear-gradient(90deg,rgb(17,24,39),rgba(17,24,39,0))]" />
              <div className="pointer-events-none absolute inset-y-0 right-0 z-10 w-12 bg-[linear-gradient(270deg,rgba(230,238,248,0.92),rgba(230,238,248,0))] dark:bg-[linear-gradient(270deg,rgb(17,24,39),rgba(17,24,39,0))]" />
              <div className="footer-info-marquee flex w-max items-center gap-8 whitespace-nowrap px-4 text-xs font-semibold uppercase tracking-[0.18em] text-muted-foreground dark:text-gray-400">
                {Array.from({ length: 2 }).map((_, index) => (
                  <div key={index} className="flex items-center gap-8">
                    <span>Aroha Technologies</span>
                    <span className="h-1.5 w-1.5 rounded-full bg-[#2f5fa8]/55 dark:bg-cyan-300/70" />
                    <span>Intelligent Data Assistant</span>
                    <span className="h-1.5 w-1.5 rounded-full bg-[#2f5fa8]/55 dark:bg-cyan-300/70" />
                    <span>AI-guided dataset understanding, analysis, and predictive modeling</span>
                    <span className="h-1.5 w-1.5 rounded-full bg-[#2f5fa8]/55 dark:bg-cyan-300/70" />
                    <a className="inline-flex items-center gap-2 text-[#2f5fa8] transition-colors hover:text-[#234e9e] dark:text-blue-400 dark:hover:text-blue-300" href={AROHA_WEBSITE_URL} target="_blank" rel="noopener noreferrer">
                      Aroha Intelligent Platform
                      <ExternalLink className="h-3.5 w-3.5" />
                    </a>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </footer>
      </div>
    </div>
  );
}
