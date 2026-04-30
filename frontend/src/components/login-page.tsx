'use client';

import * as React from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import { Loader2, AlertCircle, CheckCircle2, ArrowRight, LogIn, User, Lock, MapPin, Mail, Phone } from 'lucide-react';
import { apiClient, getApiErrorMessage } from '@/lib/api';
import type { AuthenticatedUser } from '@/lib/store';
import { useToast } from '@/hooks/use-toast';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';

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

const AROHA_WEBSITE_URL = 'https://aroha.co.in/';
const AROHA_CONTACT_URL = 'https://aroha.co.in/contact-us/';
const ANALYTICS_VIDEO_URL = 'https://assets.mixkit.co/videos/preview/mixkit-business-people-analyzing-marketing-data-46685-large.mp4';

export default function LoginPage({ onAuthSuccess }: LoginPageProps) {
  const { toast } = useToast();
  const [mode, setMode] = React.useState<AuthMode>('login');
  const [modeDirection, setModeDirection] = React.useState(1);
  const [form, setForm] = React.useState(initialForm);
  const [isSubmitting, setIsSubmitting] = React.useState(false);
  const [errorMessage, setErrorMessage] = React.useState<string | null>(null);
  const [infoMessage, setInfoMessage] = React.useState<string | null>(null);
  const [showSuccess, setShowSuccess] = React.useState(false);
  const logoSrc = React.useMemo(() => '/company-logo.png', []);

  const updateField = (field: keyof typeof initialForm) => (event: React.ChangeEvent<HTMLInputElement>) => {
    setErrorMessage(null);
    setInfoMessage(null);
    setForm((current) => ({ ...current, [field]: event.target.value }));
  };

  const switchMode = (nextMode: AuthMode) => {
    if (nextMode === mode) return;
    setModeDirection(nextMode === 'register' ? 1 : -1);
    setMode(nextMode);
    setForm(initialForm);
    setErrorMessage(null);
    setInfoMessage(null);
    setShowSuccess(false);
  };

  const handleForgotPassword = () => {
    setErrorMessage(null);
    setInfoMessage('Password reset is not automated yet. Please contact the support team for assistance.');
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
        description: mode === 'register' ? `Welcome, ${response.data.user.username}.` : `Welcome back, ${response.data.user.username}.`,
      });
    } catch (error) {
      const message = getApiErrorMessage(
        error,
        mode === 'register' ? 'We could not create your account.' : 'We could not sign you in. Please verify your credentials and try again.'
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
    const timer = window.setTimeout(() => setShowSuccess(false), 1800);
    return () => window.clearTimeout(timer);
  }, [showSuccess]);

  return (
    <div className="relative min-h-screen overflow-hidden bg-[linear-gradient(135deg,#2f5fa8_0%,#4e8ed3_55%,#67b3df_100%)] [font-family:'Trebuchet_MS',Inter,ui-sans-serif,system-ui,sans-serif] dark:bg-[linear-gradient(135deg,#0f1d30_0%,#1b4778_56%,#2d7eb2_100%)]">
      <AnimatePresence>
        {isSubmitting ? (
          <motion.div
            key="auth-loading-bar"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="absolute left-0 top-0 z-50 h-1 w-full bg-white/20"
          >
            <motion.div
              className="h-full bg-[linear-gradient(90deg,#9ec5ff_0%,#e2efff_40%,#2563eb_100%)] shadow-[0_0_18px_rgba(59,130,246,0.9)]"
              initial={{ x: '-45%', width: '35%' }}
              animate={{ x: '210%', width: '45%' }}
              transition={{ duration: 1.1, repeat: Infinity, ease: 'easeInOut' }}
            />
          </motion.div>
        ) : null}
      </AnimatePresence>

      <div className="pointer-events-none absolute inset-0">
        <div className="absolute left-[-8%] top-[12%] h-80 w-80 rounded-full bg-blue-200/18 blur-3xl dark:bg-blue-200/10" />
        <div className="absolute right-[-10%] bottom-[8%] h-96 w-96 rounded-full bg-blue-900/20 blur-3xl dark:bg-black/24" />
        <div className="absolute inset-0 bg-[linear-gradient(130deg,rgba(255,255,255,0.14)_0%,rgba(255,255,255,0.03)_44%,rgba(0,0,0,0.2)_100%)]" />
        <motion.div
          className="absolute left-[10%] top-[14%] h-20 w-20 rounded-full border border-white/20"
          animate={{ rotate: 360 }}
          transition={{ duration: 26, ease: 'linear', repeat: Infinity }}
        />
        <motion.div
          className="absolute left-[40%] bottom-[10%] h-12 w-12 rounded-full border border-white/15"
          animate={{ rotate: -360 }}
          transition={{ duration: 22, ease: 'linear', repeat: Infinity }}
        />
      </div>

      <div className="relative mx-auto flex min-h-screen w-full max-w-7xl items-center px-4 py-8 sm:px-6 lg:px-8">
        <div className="relative grid w-full overflow-hidden rounded-[2rem] border-[10px] border-white/88 bg-[#edf1f6] shadow-[0_34px_110px_-40px_rgba(8,24,58,0.55)] dark:border-white/12 dark:bg-[#122034] lg:grid-cols-[0.92fr_1fr]">
          <section className="flex items-center justify-center bg-[#edf1f6] p-6 dark:bg-[#122034] sm:p-8 lg:p-10">
            <motion.div
              initial={{ opacity: 0, y: 14 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.45, ease: 'easeOut' }}
              className="relative h-[560px] w-full max-w-md overflow-hidden rounded-2xl border border-white/75 bg-white/70 p-6 shadow-[0_24px_70px_-34px_rgba(15,23,42,0.32)] backdrop-blur-2xl dark:border-white/10 dark:bg-white/8 dark:shadow-[0_24px_80px_-34px_rgba(1,8,18,0.78)] sm:p-8"
            >
              <motion.div
                className="pointer-events-none absolute inset-0 bg-[linear-gradient(115deg,transparent_0%,rgba(56,189,248,0.08)_46%,transparent_70%)]"
                animate={{ x: ['-130%', '130%'] }}
                transition={{ duration: 5.2, repeat: Infinity, ease: 'linear' }}
              />
              <div className="mb-5 flex items-center justify-between">
                <div>
                  <p className="text-4xl font-bold leading-none text-[#234e9e] dark:text-[#d7ebff]">{mode === 'register' ? 'Sign Up' : 'Login'}</p>
                  <p className="mt-2 text-sm text-slate-600 dark:text-slate-300">Intelligent Data Assistant</p>
                </div>
                <div className="flex h-9 w-9 items-center justify-center rounded-full bg-[linear-gradient(135deg,#1d78d4_0%,#4cb8f0_100%)] text-white shadow-[0_8px_24px_-12px_rgba(37,99,235,0.65)]">
                  <ArrowRight className="h-4 w-4" />
                </div>
              </div>

              <div className="h-[430px] overflow-hidden">
                <AnimatePresence mode="wait" custom={modeDirection}>
                  <motion.form
                    key={mode}
                    custom={modeDirection}
                    initial={{ x: modeDirection > 0 ? 42 : -42, opacity: 0, filter: 'blur(2px)' }}
                    animate={{ x: 0, opacity: 1, filter: 'blur(0px)' }}
                    exit={{ x: modeDirection > 0 ? -42 : 42, opacity: 0, filter: 'blur(2px)' }}
                    transition={{ duration: 0.35, ease: 'easeOut' }}
                    className="space-y-4"
                    onSubmit={handleSubmit}
                  >
                    <AnimatePresence mode="wait">
                      {errorMessage ? (
                        <motion.div key={`error-${mode}`} initial={{ opacity: 0, y: -5 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -5 }}>
                          <Alert variant="destructive" className="border-red-300/80 bg-red-50/85 dark:border-red-400/30 dark:bg-red-950/28">
                            <AlertCircle />
                            <AlertTitle>{mode === 'register' ? 'Registration error' : 'Login error'}</AlertTitle>
                            <AlertDescription>{errorMessage}</AlertDescription>
                          </Alert>
                        </motion.div>
                      ) : null}
                    </AnimatePresence>

                    <AnimatePresence mode="wait">
                      {infoMessage ? (
                        <motion.div key={`info-${mode}`} initial={{ opacity: 0, y: -5 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -5 }}>
                          <Alert className="border-blue-300/60 bg-blue-50/85 text-blue-950 dark:border-blue-300/25 dark:bg-blue-950/28 dark:text-blue-100">
                            <AlertCircle />
                            <AlertTitle>Password help</AlertTitle>
                            <AlertDescription>{infoMessage}</AlertDescription>
                          </Alert>
                        </motion.div>
                      ) : null}
                    </AnimatePresence>

                    <AnimatePresence>
                      {showSuccess ? (
                        <motion.div initial={{ opacity: 0, scale: 0.98 }} animate={{ opacity: 1, scale: 1 }} exit={{ opacity: 0, scale: 0.98 }} className="rounded-md border border-emerald-300 bg-emerald-50 px-3 py-2 dark:border-emerald-300/30 dark:bg-emerald-950/28">
                          <div className="flex items-center gap-2 text-emerald-900 dark:text-emerald-100">
                            <CheckCircle2 className="h-4 w-4" />
                            <p className="text-sm font-medium">{mode === 'register' ? 'Account created successfully.' : 'Login successful. Opening workspace...'}</p>
                          </div>
                        </motion.div>
                      ) : null}
                    </AnimatePresence>

                    {mode === 'register' && (
                      <div className="space-y-1.5">
                        <Label htmlFor="username" className="text-sm text-[#1f3340] dark:text-slate-100">User Name</Label>
                        <div className="group relative overflow-hidden rounded-sm p-[1px]">
                          <motion.span
                            className="pointer-events-none absolute inset-y-[-80%] left-[-45%] w-1/2 rotate-12 bg-[linear-gradient(90deg,transparent_0%,rgba(76,184,240,0.9)_48%,rgba(255,255,255,0.92)_52%,transparent_100%)] opacity-0 blur-sm transition-opacity duration-300 group-hover:opacity-100 group-focus-within:opacity-100"
                            animate={{ x: ['0%', '330%'] }}
                            transition={{ duration: 2.8, repeat: Infinity, ease: 'linear' }}
                          />
                          <Input
                            id="username"
                            type="text"
                            autoComplete="username"
                            placeholder="Enter your user name"
                            value={form.username}
                            onChange={updateField('username')}
                            disabled={isSubmitting}
                            required
                            className="relative z-10 h-11 rounded-sm border-[#cad5e4] bg-white text-[#1a2f3a] caret-[#1d78d4] placeholder:text-[#8a96a8] focus-visible:ring-[#60a5fa] dark:border-white/15 dark:bg-[#0f1d30]/80 dark:text-slate-50 dark:placeholder:text-slate-400"
                          />
                        </div>
                      </div>
                    )}

                    <div className="space-y-1.5">
                      <Label htmlFor="email" className="text-sm text-[#1f3340] dark:text-slate-100">User Name / ID</Label>
                      <div className="group relative overflow-hidden rounded-sm p-[1px]">
                        <motion.span
                          className="pointer-events-none absolute inset-y-[-80%] left-[-45%] w-1/2 rotate-12 bg-[linear-gradient(90deg,transparent_0%,rgba(76,184,240,0.9)_48%,rgba(255,255,255,0.92)_52%,transparent_100%)] opacity-0 blur-sm transition-opacity duration-300 group-hover:opacity-100 group-focus-within:opacity-100"
                          animate={{ x: ['0%', '330%'] }}
                          transition={{ duration: 2.8, repeat: Infinity, ease: 'linear' }}
                        />
                        <div className="relative z-10">
                          <User className="pointer-events-none absolute left-3 top-1/2 z-20 h-4 w-4 -translate-y-1/2 text-slate-400" />
                          <Input
                            id="email"
                            type="email"
                            autoComplete="email"
                            placeholder="name@company.com"
                            value={form.email}
                            onChange={updateField('email')}
                            disabled={isSubmitting}
                            required
                            className="h-11 rounded-sm border-[#cad5e4] bg-white pl-10 text-[#1a2f3a] caret-[#1d78d4] placeholder:text-[#8a96a8] focus-visible:ring-[#60a5fa] dark:border-white/15 dark:bg-[#0f1d30]/80 dark:text-slate-50 dark:placeholder:text-slate-400"
                          />
                        </div>
                      </div>
                    </div>

                    <div className="space-y-1.5">
                      <Label htmlFor="password" className="text-sm text-[#1f3340] dark:text-slate-100">Password</Label>
                      <div className="group relative overflow-hidden rounded-sm p-[1px]">
                        <motion.span
                          className="pointer-events-none absolute inset-y-[-80%] left-[-45%] w-1/2 rotate-12 bg-[linear-gradient(90deg,transparent_0%,rgba(76,184,240,0.9)_48%,rgba(255,255,255,0.92)_52%,transparent_100%)] opacity-0 blur-sm transition-opacity duration-300 group-hover:opacity-100 group-focus-within:opacity-100"
                          animate={{ x: ['0%', '330%'] }}
                          transition={{ duration: 2.8, repeat: Infinity, ease: 'linear' }}
                        />
                        <div className="relative z-10">
                          <Lock className="pointer-events-none absolute left-3 top-1/2 z-20 h-4 w-4 -translate-y-1/2 text-slate-400" />
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
                            className="h-11 rounded-sm border-[#cad5e4] bg-white pl-10 text-[#1a2f3a] caret-[#1d78d4] placeholder:text-[#8a96a8] focus-visible:ring-[#60a5fa] dark:border-white/15 dark:bg-[#0f1d30]/80 dark:text-slate-50 dark:placeholder:text-slate-400"
                          />
                        </div>
                      </div>
                    </div>

                    {mode === 'register' && (
                      <div className="space-y-1.5">
                        <Label htmlFor="confirm-password" className="text-sm text-[#1f3340] dark:text-slate-100">Confirm Password</Label>
                        <div className="group relative overflow-hidden rounded-sm p-[1px]">
                          <motion.span
                            className="pointer-events-none absolute inset-y-[-80%] left-[-45%] w-1/2 rotate-12 bg-[linear-gradient(90deg,transparent_0%,rgba(76,184,240,0.9)_48%,rgba(255,255,255,0.92)_52%,transparent_100%)] opacity-0 blur-sm transition-opacity duration-300 group-hover:opacity-100 group-focus-within:opacity-100"
                            animate={{ x: ['0%', '330%'] }}
                            transition={{ duration: 2.8, repeat: Infinity, ease: 'linear' }}
                          />
                          <Input
                            id="confirm-password"
                            type="password"
                            autoComplete="new-password"
                            placeholder="Re-enter password"
                            value={form.confirmPassword}
                            onChange={updateField('confirmPassword')}
                            disabled={isSubmitting}
                            required
                            minLength={8}
                            className="relative z-10 h-11 rounded-sm border-[#cad5e4] bg-white text-[#1a2f3a] caret-[#1d78d4] placeholder:text-[#8a96a8] focus-visible:ring-[#60a5fa] dark:border-white/15 dark:bg-[#0f1d30]/80 dark:text-slate-50 dark:placeholder:text-slate-400"
                          />
                        </div>
                      </div>
                    )}

                    {mode === 'login' && (
                      <div className="flex items-center justify-between text-xs text-slate-600 dark:text-slate-300">
                        <span>Stay signed in</span>
                        <button type="button" onClick={handleForgotPassword} className="font-medium text-[#2f5fa8] hover:underline dark:text-[#b9ddff]">
                          Forgot password?
                        </button>
                      </div>
                    )}

                    <Button
                      type="submit"
                      disabled={isSubmitting}
                      className="group relative h-11 w-full overflow-hidden rounded-sm bg-[linear-gradient(135deg,#36a74b_0%,#49b653_100%)] text-white shadow-[0_16px_45px_-24px_rgba(34,139,64,0.45)] transition-all duration-300 hover:-translate-y-0.5 hover:brightness-105"
                    >
                      <motion.span
                        className="pointer-events-none absolute inset-y-0 -left-1/2 w-1/2 bg-[linear-gradient(90deg,transparent_0%,rgba(255,255,255,0.42)_50%,transparent_100%)]"
                        animate={isSubmitting ? { x: ['0%', '320%'] } : { x: '-160%' }}
                        transition={isSubmitting ? { duration: 0.9, repeat: Infinity, ease: 'linear' } : { duration: 0.3 }}
                      />
                      {isSubmitting && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
                      {isSubmitting ? (mode === 'register' ? 'Creating account...' : 'Signing in...') : (mode === 'register' ? 'Sign Up' : 'Login')}
                      {!isSubmitting && <LogIn className="ml-2 h-4 w-4" />}
                    </Button>

                    {mode === 'login' ? (
                      <p className="text-center text-xs text-slate-600 dark:text-slate-300">
                        New here?{' '}
                        <button type="button" onClick={() => switchMode('register')} className="font-semibold text-[#2f5fa8] hover:underline dark:text-[#b9ddff]">
                          Sign up
                        </button>
                      </p>
                    ) : (
                      <p className="text-center text-xs text-slate-600 dark:text-slate-300">
                        Already have an account?{' '}
                        <button type="button" onClick={() => switchMode('login')} className="font-semibold text-[#2f5fa8] hover:underline dark:text-[#b9ddff]">
                          Back to Login
                        </button>
                      </p>
                    )}
                  </motion.form>
                </AnimatePresence>
              </div>
            </motion.div>
          </section>

          <section className="group relative min-h-[560px] overflow-hidden border-t border-white/25 lg:border-l lg:border-t-0 lg:border-l-white/30">
            <video
              className="absolute inset-0 h-full w-full object-cover"
              src={ANALYTICS_VIDEO_URL}
              poster="https://images.unsplash.com/photo-1551288049-bebda4e38f71?auto=format&fit=crop&w=1400&q=80"
              autoPlay
              muted
              loop
              playsInline
              preload="metadata"
            />
            <div className="absolute inset-0 bg-[linear-gradient(115deg,rgba(17,57,118,0.88)_0%,rgba(29,120,212,0.72)_48%,rgba(76,184,240,0.52)_100%)]" />
            <div className="absolute inset-0 bg-[linear-gradient(180deg,rgba(5,18,42,0.16)_0%,rgba(5,18,42,0.36)_100%)]" />
            <motion.div
              className="pointer-events-none absolute -inset-y-8 -left-1/2 w-1/2 bg-[linear-gradient(90deg,transparent_0%,rgba(255,255,255,0.18)_50%,transparent_100%)]"
              animate={{ x: ['0%', '300%'] }}
              transition={{ duration: 7.5, repeat: Infinity, ease: 'linear' }}
            />
            <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(circle_at_20%_16%,rgba(255,255,255,0.24),transparent_40%),radial-gradient(circle_at_78%_84%,rgba(255,255,255,0.16),transparent_36%)]" />

            <div className="relative z-10 flex h-full flex-col justify-between p-8 text-white sm:p-10 lg:p-12">
              <div>
                <motion.div
                  whileHover={{ x: 3 }}
                  transition={{ type: 'spring', stiffness: 220, damping: 18 }}
                  className="mb-8 flex items-center gap-4"
                >
                  <motion.a
                    href={AROHA_WEBSITE_URL}
                    target="_blank"
                    rel="noreferrer"
                    whileHover={{ scale: 1.07, rotate: -1 }}
                    transition={{ type: 'spring', stiffness: 240, damping: 16 }}
                    className="flex h-24 w-24 items-center justify-center overflow-hidden rounded-xl border border-white/35 bg-transparent p-0 shadow-[0_18px_40px_-22px_rgba(15,23,42,0.55)] transition-all duration-300 group-hover:shadow-[0_16px_42px_-18px_rgba(255,255,255,0.35)]"
                  >
                    <img
                      src={logoSrc}
                      alt="Aroha Technologies Company Logo"
                      className="h-full w-full object-contain mix-blend-multiply drop-shadow-[0_10px_24px_rgba(0,0,0,0.25)] dark:mix-blend-screen"
                    />
                  </motion.a>
                  <div>
                    <p className="text-sm font-black uppercase tracking-[0.24em] text-white">Aroha Intelligent Platform</p>
                    <p className="mt-1 text-xl font-black uppercase tracking-[0.12em] text-white">Intelligent Data Assistant</p>
                  </div>
                </motion.div>

                <motion.p
                  className="max-w-xl text-4xl font-black uppercase leading-tight text-white drop-shadow-[0_8px_22px_rgba(8,24,58,0.34)] sm:text-5xl"
                  whileHover={{ scale: 1.01 }}
                  transition={{ duration: 0.25 }}
                >
                  Insight-driven data analysis for modern teams
                </motion.p>
                <p className="mt-5 max-w-lg text-base font-bold leading-7 text-white/92">
                  A unified application for data upload, profiling, exploratory analysis, forecasting, and model-driven business insights.
                </p>
              </div>

              <motion.div
                whileHover={{ y: -2 }}
                transition={{ duration: 0.25 }}
                className="space-y-3 rounded-xl border border-white/28 bg-white/14 p-4 font-bold backdrop-blur-md transition-all duration-300 hover:border-white/42 hover:bg-white/18 hover:shadow-[0_14px_36px_-18px_rgba(15,23,42,0.5)]"
              >
                <p className="flex items-center gap-2 text-sm sm:text-base"><MapPin className="h-4 w-4 transition-transform duration-300 group-hover:scale-110" /> Location: Bangalore</p>
                <p className="flex items-center gap-2 text-sm sm:text-base">
                  <Mail className="h-4 w-4" />
                  Contact Us:
                  <a
                    href={AROHA_CONTACT_URL}
                    target="_blank"
                    rel="noreferrer"
                    className="font-black underline underline-offset-4 hover:opacity-90"
                  >
                    Contact Page
                  </a>
                </p>
                <p className="flex items-center gap-2 text-sm sm:text-base"><Phone className="h-4 w-4" /> Phone: +91 9886228615</p>
              </motion.div>
            </div>
          </section>
        </div>
      </div>
    </div>
  );
}
