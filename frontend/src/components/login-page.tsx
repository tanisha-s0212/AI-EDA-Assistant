'use client';

import * as React from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import { apiClient, getApiErrorMessage } from '@/lib/api';
import type { AuthenticatedUser } from '@/lib/store';
import { useToast } from '@/hooks/use-toast';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Badge } from '@/components/ui/badge';
import {
  ShieldCheck,
  Database,
  ArrowRight,
  UserRound,
  Mail,
  LockKeyhole,
  Loader2,
  AlertCircle,
  CheckCircle2,
  Orbit,
  Cpu,
  Radar,
} from 'lucide-react';

type LoginPageProps = {
  onAuthSuccess: (user: AuthenticatedUser) => void;
};

type AuthMode = 'login' | 'register';

type AuthResponse = {
  user: AuthenticatedUser;
};

const initialForm = {
  username: '',
  email: '',
  password: '',
  confirmPassword: '',
};

function BrandMark() {
  return (
    <div className="relative flex h-14 w-14 shrink-0 items-center justify-center overflow-hidden rounded-[22px] border border-white/12 bg-[linear-gradient(145deg,#08111f_0%,#0f2747_52%,#0b7f8f_100%)] text-white shadow-[0_24px_60px_-28px_rgba(14,116,144,0.62)]">
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_20%_18%,rgba(255,255,255,0.2),transparent_32%)]" />
      <div className="absolute inset-0 bg-[linear-gradient(180deg,rgba(255,255,255,0.08),transparent_42%,rgba(8,145,178,0.24)_100%)]" />
      <div className="absolute inset-[18%] rounded-[18px] border border-white/10 bg-slate-950/30" />
      <div className="absolute inset-x-[26%] top-[24%] h-px bg-cyan-200/50" />
      <div className="absolute inset-x-[26%] bottom-[24%] h-px bg-cyan-200/35" />
      <div className="absolute inset-y-[26%] left-[24%] w-px bg-cyan-200/40" />
      <div className="absolute inset-y-[26%] right-[24%] w-px bg-cyan-200/25" />
      <div className="absolute left-[18%] top-[18%] h-2.5 w-2.5 rounded-md border border-cyan-200/25 bg-cyan-300/12" />
      <div className="absolute bottom-[18%] right-[18%] h-3 w-3 rounded-md border border-cyan-200/20 bg-cyan-300/10" />
      <Orbit className="absolute h-9 w-9 text-cyan-100/20" strokeWidth={1.5} />
      <Cpu className="absolute h-5 w-5 text-white" strokeWidth={2.2} />
      <Radar className="absolute h-4 w-4 text-cyan-100/85" strokeWidth={1.9} />
    </div>
  );
}

function BrandWordmark() {
  return (
    <div className="min-w-0">
      <p className="text-xs font-semibold uppercase tracking-[0.34em] text-slate-500">Aroha Intelligent Platform</p>
      <div className="mt-1.5 flex min-w-0 items-center gap-2">
        <p className="truncate text-[1.9rem] font-black tracking-[-0.03em] sm:text-[2.25rem]">
          <span className="bg-[linear-gradient(135deg,#082f49_0%,#0f766e_46%,#0284c7_100%)] bg-clip-text text-transparent">
            Intelligent Data Assistant
          </span>
        </p>
      </div>
    </div>
  );
}

export default function LoginPage({ onAuthSuccess }: LoginPageProps) {
  const { toast } = useToast();
  const [mode, setMode] = React.useState<AuthMode>('login');
  const [form, setForm] = React.useState(initialForm);
  const [isSubmitting, setIsSubmitting] = React.useState(false);
  const [errorMessage, setErrorMessage] = React.useState<string | null>(null);
  const [infoMessage, setInfoMessage] = React.useState<string | null>(null);
  const [showSuccess, setShowSuccess] = React.useState(false);

  const updateField = (field: keyof typeof initialForm) => (event: React.ChangeEvent<HTMLInputElement>) => {
    setErrorMessage(null);
    setInfoMessage(null);
    setForm((current) => ({ ...current, [field]: event.target.value }));
  };

  const switchMode = (nextMode: AuthMode) => {
    setMode(nextMode);
    setForm(initialForm);
    setErrorMessage(null);
    setInfoMessage(null);
    setShowSuccess(false);
  };

  const handleForgotPassword = () => {
    setErrorMessage(null);
    setInfoMessage('Password reset is not automated yet. Please contact your administrator or support team to reset your password.');
  };

  const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setErrorMessage(null);
    setInfoMessage(null);
    setShowSuccess(false);

    if (mode === 'register' && form.password !== form.confirmPassword) {
      const message = 'Password and confirm password must match before we can create your account.';
      setErrorMessage(message);
      toast({
        title: 'Password mismatch',
        description: message,
        variant: 'destructive',
      });
      return;
    }

    setIsSubmitting(true);

    try {
      const response = mode === 'register'
        ? await apiClient.post<AuthResponse>('/auth/register', {
            username: form.username,
            email: form.email,
            password: form.password,
          })
        : await apiClient.post<AuthResponse>('/auth/login', {
            email: form.email,
            password: form.password,
          });

      setShowSuccess(true);
      onAuthSuccess(response.data.user);
      toast({
        title: mode === 'register' ? 'Account created' : 'Login successful',
        description:
          mode === 'register'
            ? `Welcome, ${response.data.user.username}. Your account is secured and your session is active.`
            : `Welcome back, ${response.data.user.username}. Your secure session has been restored.`,
      });
    } catch (error) {
      const message = getApiErrorMessage(
        error,
        mode === 'register' ? 'We could not create your account.' : 'We could not sign you in. Please check your email and password and try again.'
      );
      setErrorMessage(message);
      toast({
        title: mode === 'register' ? 'Registration failed' : 'Login failed',
        description: message,
        variant: 'destructive',
      });
    } finally {
      setIsSubmitting(false);
    }
  };

  React.useEffect(() => {
    if (!showSuccess) return;
    const timer = window.setTimeout(() => setShowSuccess(false), 1600);
    return () => window.clearTimeout(timer);
  }, [showSuccess]);

  return (
    <div className="relative min-h-screen overflow-hidden bg-[linear-gradient(160deg,#eff6ff_0%,#f8fafc_42%,#ecfeff_100%)]">
      <div className="pointer-events-none absolute inset-0">
        <div className="absolute left-[-4rem] top-16 h-64 w-64 rounded-full bg-sky-400/15 blur-3xl" />
        <div className="absolute right-[-5rem] top-12 h-80 w-80 rounded-full bg-cyan-300/20 blur-3xl" />
        <div className="absolute bottom-[-6rem] left-1/3 h-96 w-96 rounded-full bg-emerald-300/15 blur-3xl" />
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_top,rgba(15,118,110,0.08),transparent_32%)]" />
      </div>

      <div className="relative mx-auto flex min-h-screen max-w-7xl items-center px-4 py-8 sm:px-6 lg:px-8">
        <div className="grid w-full gap-8 lg:grid-cols-[minmax(0,1.1fr)_520px] lg:items-center">
          <section className="max-w-2xl">
            <div className="mb-6 flex items-center gap-4">
              <BrandMark />
              <BrandWordmark />
            </div>
            <Badge className="rounded-full border-sky-200 bg-white/80 px-3 py-1 text-sky-900 shadow-sm">
              <ShieldCheck className="mr-2 h-3.5 w-3.5" />
              Secure workspace access
            </Badge>
            <h1 className="mt-6 text-4xl font-semibold tracking-tight text-slate-950 sm:text-5xl">
              Secure Access to Your Analytics Workspace
            </h1>
            <p className="mt-4 max-w-xl text-base leading-7 text-slate-600 sm:text-lg">
              Accounts are stored in PostgreSQL with hashed passwords and protected server-side sessions. Once signed in,
              the analytics workspace opens with a verified backend session.
            </p>

            <div className="mt-8 grid gap-4 sm:grid-cols-3">
              <div className="rounded-3xl border border-white/70 bg-white/75 p-4 shadow-[0_24px_70px_-42px_rgba(14,116,144,0.45)] backdrop-blur-xl">
                <UserRound className="h-5 w-5 text-sky-700" />
                <p className="mt-3 text-sm font-semibold text-slate-900">Account identity</p>
                <p className="mt-1 text-sm leading-6 text-slate-600">Each user gets a dedicated stored profile.</p>
              </div>
              <div className="rounded-3xl border border-white/70 bg-white/75 p-4 shadow-[0_24px_70px_-42px_rgba(14,116,144,0.45)] backdrop-blur-xl">
                <LockKeyhole className="h-5 w-5 text-cyan-700" />
                <p className="mt-3 text-sm font-semibold text-slate-900">Hashed passwords</p>
                <p className="mt-1 text-sm leading-6 text-slate-600">Credentials are protected with PBKDF2 hashing.</p>
              </div>
              <div className="rounded-3xl border border-white/70 bg-white/75 p-4 shadow-[0_24px_70px_-42px_rgba(14,116,144,0.45)] backdrop-blur-xl">
                <Database className="h-5 w-5 text-emerald-700" />
                <p className="mt-3 text-sm font-semibold text-slate-900">Server sessions</p>
                <p className="mt-1 text-sm leading-6 text-slate-600">Secure cookies keep the authenticated session on the backend.</p>
              </div>
            </div>
          </section>

          <Card className="overflow-hidden rounded-[32px] border-white/80 bg-white/88 py-0 shadow-[0_30px_100px_-36px_rgba(15,23,42,0.28)] backdrop-blur-2xl">
            <CardHeader className="border-b border-slate-200/70 bg-[linear-gradient(135deg,rgba(14,116,144,0.07),rgba(14,165,233,0.02))] px-8 py-8">
              <div className="mb-4 flex items-center gap-4">
                <BrandMark />
                <div className="min-w-0">
                  <p className="text-[11px] font-semibold uppercase tracking-[0.22em] text-slate-500">Workspace Authentication</p>
                  <p className="mt-1 text-[1.4rem] font-black tracking-[-0.03em] text-slate-950 sm:text-[1.55rem]">
                    <span className="bg-[linear-gradient(135deg,#082f49_0%,#0f766e_46%,#0284c7_100%)] bg-clip-text text-transparent">
                      Intelligent Data Assistant
                    </span>
                  </p>
                </div>
              </div>
              <div className="flex items-center gap-2 rounded-2xl bg-slate-100/80 p-1">
                <Button
                  type="button"
                  variant={mode === 'login' ? 'default' : 'ghost'}
                  className="h-10 flex-1 rounded-xl"
                  onClick={() => switchMode('login')}
                >
                  Sign In
                </Button>
                <Button
                  type="button"
                  variant={mode === 'register' ? 'default' : 'ghost'}
                  className="h-10 flex-1 rounded-xl"
                  onClick={() => switchMode('register')}
                >
                  Register
                </Button>
              </div>
              <CardTitle className="pt-4 text-2xl font-semibold tracking-tight text-slate-950">
                {mode === 'register' ? 'Create a secure account' : 'Sign in to your account'}
              </CardTitle>
              <CardDescription className="text-sm leading-6 text-slate-600">
                {mode === 'register'
                  ? 'Register with username, email, and password to create a protected analytics workspace.'
                  : 'Use your email and password to restore your authenticated workspace session.'}
              </CardDescription>
            </CardHeader>
            <CardContent className="px-8 py-8">
              <form className="space-y-5" onSubmit={handleSubmit}>
                <AnimatePresence mode="wait">
                  {errorMessage ? (
                    <motion.div
                      key={`error-${mode}`}
                      initial={{ opacity: 0, y: -6 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0, y: -6 }}
                    >
                      <Alert variant="destructive" className="border-destructive/25 bg-red-50/90">
                        <AlertCircle />
                        <AlertTitle>{mode === 'register' ? 'Registration error' : 'Login error'}</AlertTitle>
                        <AlertDescription>{errorMessage}</AlertDescription>
                      </Alert>
                    </motion.div>
                  ) : null}
                </AnimatePresence>

                <AnimatePresence mode="wait">
                  {infoMessage ? (
                    <motion.div
                      key={`info-${mode}`}
                      initial={{ opacity: 0, y: -6 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0, y: -6 }}
                    >
                      <Alert className="border-sky-200 bg-sky-50/90 text-slate-800">
                        <Mail />
                        <AlertTitle>Password help</AlertTitle>
                        <AlertDescription>{infoMessage}</AlertDescription>
                      </Alert>
                    </motion.div>
                  ) : null}
                </AnimatePresence>

                <AnimatePresence>
                  {showSuccess ? (
                    <motion.div
                      initial={{ opacity: 0, scale: 0.96 }}
                      animate={{ opacity: 1, scale: 1 }}
                      exit={{ opacity: 0, scale: 0.96 }}
                      className="rounded-2xl border border-emerald-200 bg-emerald-50/90 px-4 py-3"
                    >
                      <div className="flex items-center gap-3 text-emerald-800">
                        <motion.div initial={{ scale: 0.7, rotate: -12 }} animate={{ scale: 1, rotate: 0 }} transition={{ type: 'spring', stiffness: 280, damping: 18 }}>
                          <CheckCircle2 className="h-5 w-5" />
                        </motion.div>
                        <p className="text-sm font-medium">{mode === 'register' ? 'Account created successfully.' : 'Login successful. Opening your workspace...'}</p>
                      </div>
                    </motion.div>
                  ) : null}
                </AnimatePresence>

                {mode === 'register' && (
                  <div className="space-y-2">
                    <Label htmlFor="username">Username</Label>
                    <Input
                      id="username"
                      type="text"
                      autoComplete="username"
                      placeholder="Enter your username"
                      value={form.username}
                      onChange={updateField('username')}
                      disabled={isSubmitting}
                      required
                      className="h-11 rounded-2xl border-slate-200 bg-white/80"
                    />
                  </div>
                )}

                <div className="space-y-2">
                  <Label htmlFor="email">Email address</Label>
                  <Input
                    id="email"
                    type="email"
                    autoComplete="email"
                    placeholder="name@company.com"
                    value={form.email}
                    onChange={updateField('email')}
                    disabled={isSubmitting}
                    required
                    className="h-11 rounded-2xl border-slate-200 bg-white/80"
                  />
                </div>

                <div className="space-y-2">
                  <Label htmlFor="password">Password</Label>
                  <Input
                    id="password"
                    type="password"
                    autoComplete={mode === 'register' ? 'new-password' : 'current-password'}
                    placeholder="Enter your password"
                    value={form.password}
                    onChange={updateField('password')}
                    disabled={isSubmitting}
                    required
                    minLength={8}
                    className="h-11 rounded-2xl border-slate-200 bg-white/80"
                  />
                </div>

                {mode === 'register' && (
                  <div className="space-y-2">
                    <Label htmlFor="confirm-password">Confirm password</Label>
                    <Input
                      id="confirm-password"
                      type="password"
                      autoComplete="new-password"
                      placeholder="Re-enter your password"
                      value={form.confirmPassword}
                      onChange={updateField('confirmPassword')}
                      disabled={isSubmitting}
                      required
                      minLength={8}
                      className="h-11 rounded-2xl border-slate-200 bg-white/80"
                    />
                  </div>
                )}

                {mode === 'login' && (
                  <div className="flex items-center justify-between gap-3 text-sm">
                    <button
                      type="button"
                      onClick={handleForgotPassword}
                      className="font-medium text-sky-700 transition-colors hover:text-sky-900"
                    >
                      Forgot Password
                    </button>
                    <button
                      type="button"
                      onClick={() => switchMode('register')}
                      className="font-medium text-slate-600 transition-colors hover:text-slate-950"
                    >
                      Create account
                    </button>
                  </div>
                )}

                <div className="rounded-2xl border border-sky-100 bg-sky-50/80 px-4 py-3 text-sm leading-6 text-slate-600">
                  {mode === 'register'
                    ? 'Registration saves your user profile, hashes the password, and creates an authenticated session immediately.'
                    : 'Signing in validates your credentials and restores your active backend session through a secure cookie.'}
                </div>

                <Button
                  type="submit"
                  disabled={isSubmitting}
                  className="h-11 w-full rounded-2xl bg-[linear-gradient(135deg,#0f172a_0%,#0f766e_100%)] text-white shadow-[0_20px_50px_-24px_rgba(15,23,42,0.55)] transition-transform duration-300 hover:-translate-y-0.5 hover:opacity-95"
                >
                  {isSubmitting && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
                  {isSubmitting
                    ? mode === 'register' ? 'Creating account...' : 'Signing in...'
                    : mode === 'register' ? 'Create account' : 'Access Workspace'}
                  {!isSubmitting && <ArrowRight className="ml-2 h-4 w-4" />}
                </Button>

                {mode === 'login' && (
                  <p className="text-center text-sm text-slate-500">
                    New to the platform?{' '}
                    <button
                      type="button"
                      onClick={() => switchMode('register')}
                      className="font-semibold text-sky-700 transition-colors hover:text-sky-900"
                    >
                      Create account
                    </button>
                  </p>
                )}
              </form>
            </CardContent>
          </Card>
        </div>
      </div>

      <div className="pointer-events-none absolute inset-x-0 bottom-0 z-10 px-4 pb-5 sm:px-6 lg:px-8">
        <div className="mx-auto max-w-7xl">
          <p className="text-center text-[11px] font-medium tracking-[0.08em] text-slate-500">
            © 2026 Aroha Technologies. Secure Enterprise Platform.
          </p>
        </div>
      </div>
    </div>
  );
}
