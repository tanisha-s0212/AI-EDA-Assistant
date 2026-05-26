'use client';

import * as React from 'react';
import { motion } from 'framer-motion';
import {
  AlertCircle,
  ArrowRight,
  BarChart3,
  BrainCircuit,
  CheckCircle2,
  Eye,
  EyeOff,
  LinkIcon,
  Loader2,
  Lock,
  Mail,
  MapPin,
  Phone,
  ShieldCheck,
  Sparkles,
  UploadCloud,
  UserRound,
} from 'lucide-react';
import { apiClient, getApiErrorMessage } from '@/lib/api';
import type { AuthenticatedUser } from '@/lib/store';
import { useToast } from '@/hooks/use-toast';

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
  remember: false,
  acceptedTerms: false,
};

const heroFeatures = [
  {
    title: 'Secure Workspace',
    benefit: 'Protected sessions for every dataset.',
    icon: ShieldCheck,
  },
  {
    title: 'Smart Upload',
    benefit: 'Bring files in with guided validation.',
    icon: UploadCloud,
  },
  {
    title: 'Auto Analysis',
    benefit: 'Surface patterns without manual setup.',
    icon: BarChart3,
  },
  {
    title: 'AI Forecasting',
    benefit: 'Project outcomes with assisted models.',
    icon: BrainCircuit,
  },
];

export default function LoginPage({ onAuthSuccess }: LoginPageProps) {
  const { toast } = useToast();
  const [showPassword, setShowPassword] = React.useState(false);
  const [authMode, setAuthMode] = React.useState<AuthMode>('login');
  const [form, setForm] = React.useState(initialForm);
  const [isSubmitting, setIsSubmitting] = React.useState(false);
  const [errorMessage, setErrorMessage] = React.useState<string | null>(null);
  const [infoMessage, setInfoMessage] = React.useState<string | null>(null);
  const [showSuccess, setShowSuccess] = React.useState(false);
  const isRegister = authMode === 'register';

  const updateField = (field: keyof typeof initialForm) => (event: React.ChangeEvent<HTMLInputElement>) => {
    setErrorMessage(null);
    setInfoMessage(null);
    const value = event.target.type === 'checkbox' ? event.target.checked : event.target.value;
    setForm((current) => ({ ...current, [field]: value }));
  };

  const switchMode = (nextMode: AuthMode) => {
    if (nextMode === authMode) return;
    setAuthMode(nextMode);
    setForm(initialForm);
    setErrorMessage(null);
    setInfoMessage(null);
    setShowSuccess(false);
  };

  const handleForgotPassword = () => {
    setErrorMessage(null);
    setInfoMessage('Password reset is handled by Aroha support. Use the contact link for assistance.');
  };

  const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setErrorMessage(null);
    setInfoMessage(null);
    setShowSuccess(false);

    if (isRegister && form.password !== form.confirmPassword) {
      const message = 'Password and confirm password must match before we can create your account.';
      setErrorMessage(message);
      toast({ title: 'Password mismatch', description: message, variant: 'destructive' });
      return;
    }

    if (isRegister && !form.acceptedTerms) {
      const message = 'Please agree to continue before creating your account.';
      setErrorMessage(message);
      toast({ title: 'Confirmation required', description: message, variant: 'destructive' });
      return;
    }

    setIsSubmitting(true);
    try {
      const response = isRegister
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
        title: isRegister ? 'Account created' : 'Login successful',
        description: isRegister
          ? `Welcome, ${response.data.user.username}.`
          : `Welcome back, ${response.data.user.username}.`,
      });
    } catch (error) {
      const message = getApiErrorMessage(
        error,
        isRegister
          ? 'We could not create your account.'
          : 'We could not sign you in. Please verify your credentials and try again.',
      );
      setErrorMessage(message);
      toast({
        title: isRegister ? 'Registration failed' : 'Login failed',
        description: message,
        variant: 'destructive',
      });
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <main
      className="login-page-shell relative h-screen max-h-screen w-screen overflow-hidden bg-background text-foreground"
    >
      <div className="flex h-full min-h-0 w-full flex-col overflow-hidden md:flex-row">
        <section className="relative flex min-h-0 w-full items-center justify-center overflow-hidden border-b border-white/30 bg-[radial-gradient(circle_at_16%_12%,rgba(255,255,255,0.82),transparent_30%),radial-gradient(circle_at_84%_74%,rgba(99,102,241,0.28),transparent_36%),linear-gradient(135deg,#f8fbff_0%,#dbeafe_46%,#eef2ff_100%)] px-6 py-8 md:h-full md:w-[40%] md:border-b-0 md:border-r">
          <div className="pointer-events-none absolute left-[-7rem] top-[-5rem] h-64 w-64 rounded-full bg-white/70 blur-3xl" />
          <div className="pointer-events-none absolute bottom-[-7rem] right-[-6rem] h-72 w-72 rounded-full bg-indigo-300/30 blur-3xl" />
          <div className="pointer-events-none absolute inset-0 bg-[linear-gradient(120deg,rgba(255,255,255,0.34)_0%,rgba(255,255,255,0.08)_42%,rgba(255,255,255,0.24)_100%)]" />
          <motion.div
            aria-hidden="true"
            className="absolute right-[16%] top-[18%] text-indigo-400/55"
            animate={{ y: [-8, 8, -8] }}
            transition={{ duration: 4.5, repeat: Infinity, ease: 'easeInOut' }}
          >
            <Sparkles className="h-10 w-10 drop-shadow-[0_0_24px_rgba(99,102,241,0.4)]" />
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 18 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.55, ease: 'easeOut' }}
            className="relative w-full max-w-md overflow-hidden rounded-[2rem] border border-white/70 bg-white/22 p-8 shadow-[0_34px_110px_-42px_rgba(30,41,59,0.62),inset_0_1px_0_rgba(255,255,255,0.75)] ring-1 ring-white/45 backdrop-blur-[34px]"
          >
            <div className="pointer-events-none absolute inset-0 bg-[linear-gradient(135deg,rgba(255,255,255,0.82)_0%,rgba(255,255,255,0.28)_46%,rgba(199,210,254,0.3)_100%)]" />
            <div className="pointer-events-none absolute -left-24 top-24 h-40 w-[140%] rotate-[-14deg] bg-white/42 blur-3xl" />
            <div className="pointer-events-none absolute inset-x-8 top-0 h-px bg-white/90" />
            <div className="absolute left-1/2 top-0 flex h-12 w-40 -translate-x-1/2 -translate-y-px items-center justify-center rounded-b-2xl border-x border-b border-white/70 bg-white/82 text-sm font-semibold text-slate-900 shadow-[0_16px_36px_-24px_rgba(15,23,42,0.55)] backdrop-blur-2xl">
              {isRegister ? 'Register' : 'Login'}
            </div>

            <div className="relative mt-9 flex items-center gap-4 rounded-2xl border border-white/72 bg-white/36 px-4 py-3 shadow-[0_20px_56px_-36px_rgba(67,56,202,0.46),inset_0_1px_0_rgba(255,255,255,0.82)] backdrop-blur-2xl">
              <a
                href="https://aroha.co.in/"
                target="_blank"
                rel="noopener noreferrer"
                className="flex h-32 w-56 shrink-0 items-center justify-center rounded-xl border border-white/72 bg-white/72 px-6 shadow-[inset_0_1px_0_rgba(255,255,255,0.78),0_18px_44px_-32px_rgba(67,56,202,0.58)] transition hover:bg-white focus:outline-none focus:ring-2 focus:ring-indigo-400/60"
                aria-label="Visit Aroha website"
              >
                <img
                  src="/company-logo.png"
                  alt="Aroha"
                  width={224}
                  height={96}
                  className="max-h-24 w-full object-contain drop-shadow-[0_14px_30px_rgba(67,56,202,0.2)]"
                />
              </a>
              <div className="min-w-0 text-left">
                <p className="text-[10px] font-semibold uppercase tracking-[0.22em] text-indigo-500/90">Aroha</p>
                <p className="mt-1 text-xl font-black leading-tight tracking-normal text-slate-950">
                  Aroha Intelligent Platform
                </p>
              </div>
            </div>

            <div className="relative mt-5 text-center">
              <h1 className="mt-3 text-3xl font-semibold tracking-normal text-slate-950">
                {isRegister ? 'Create Your Account' : 'Welcome Back'}
              </h1>
              <p className="mt-3 text-sm leading-6 text-slate-700">
                {isRegister
                  ? 'Register to start a secure AI analytics workspace for your business data.'
                  : 'Sign in to continue your secure analytics workspace and active data workflows.'}
              </p>
            </div>

            <form className="relative mt-7 space-y-4" onSubmit={handleSubmit}>
              {errorMessage ? (
                <div className="flex gap-2 rounded-2xl border border-red-200/90 bg-red-50/85 px-4 py-3 text-sm text-red-950">
                  <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />
                  <span>{errorMessage}</span>
                </div>
              ) : null}

              {infoMessage ? (
                <div className="flex gap-2 rounded-2xl border border-indigo-200/90 bg-indigo-50/85 px-4 py-3 text-sm text-indigo-950">
                  <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />
                  <span>{infoMessage}</span>
                </div>
              ) : null}

              {showSuccess ? (
                <div className="flex gap-2 rounded-2xl border border-emerald-200/90 bg-emerald-50/85 px-4 py-3 text-sm text-emerald-950">
                  <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0" />
                  <span>{isRegister ? 'Account created successfully.' : 'Login successful. Opening workspace...'}</span>
                </div>
              ) : null}

              {isRegister && (
                <label className="block">
                  <span className="mb-2 block text-sm font-semibold text-slate-800">Full name</span>
                  <span className="flex h-12 items-center gap-3 rounded-2xl border border-white/72 bg-white/38 px-4 text-slate-950 shadow-[inset_0_1px_0_rgba(255,255,255,0.82),0_14px_36px_-30px_rgba(67,56,202,0.46)] transition backdrop-blur-2xl focus-within:border-indigo-300/90 focus-within:bg-white/58 focus-within:ring-4 focus-within:ring-indigo-300/22">
                    <UserRound className="h-5 w-5 text-indigo-600" />
                    <input
                      type="text"
                      autoComplete="name"
                      className="h-full min-w-0 flex-1 bg-transparent text-sm text-slate-950 outline-none placeholder:text-slate-500"
                      placeholder="Your name"
                      value={form.username}
                      onChange={updateField('username')}
                      disabled={isSubmitting}
                      required
                    />
                  </span>
                </label>
              )}

              <label className="block">
                <span className="mb-2 block text-sm font-semibold text-slate-800">Email</span>
                <span className="flex h-12 items-center gap-3 rounded-2xl border border-white/72 bg-white/38 px-4 text-slate-950 shadow-[inset_0_1px_0_rgba(255,255,255,0.82),0_14px_36px_-30px_rgba(67,56,202,0.46)] transition backdrop-blur-2xl focus-within:border-indigo-300/90 focus-within:bg-white/58 focus-within:ring-4 focus-within:ring-indigo-300/22">
                  <Mail className="h-5 w-5 text-indigo-600" />
                  <input
                    type="email"
                    autoComplete="email"
                    className="h-full min-w-0 flex-1 bg-transparent text-sm text-slate-950 outline-none placeholder:text-slate-500"
                    placeholder="you@example.com"
                    value={form.email}
                    onChange={updateField('email')}
                    disabled={isSubmitting}
                    required
                  />
                </span>
              </label>

              <label className="block">
                <span className="mb-2 block text-sm font-semibold text-slate-800">Password</span>
                <span className="flex h-12 items-center gap-3 rounded-2xl border border-white/72 bg-white/38 px-4 text-slate-950 shadow-[inset_0_1px_0_rgba(255,255,255,0.82),0_14px_36px_-30px_rgba(67,56,202,0.46)] transition backdrop-blur-2xl focus-within:border-indigo-300/90 focus-within:bg-white/58 focus-within:ring-4 focus-within:ring-indigo-300/22">
                  <Lock className="h-5 w-5 text-indigo-600" />
                  <input
                    type={showPassword ? 'text' : 'password'}
                    autoComplete={isRegister ? 'new-password' : 'current-password'}
                    className="h-full min-w-0 flex-1 bg-transparent text-sm text-slate-950 outline-none placeholder:text-slate-500"
                    placeholder="Enter your password"
                    value={form.password}
                    onChange={updateField('password')}
                    disabled={isSubmitting}
                    minLength={8}
                    required
                  />
                  <button
                    type="button"
                    onClick={() => setShowPassword((current) => !current)}
                    className="rounded-lg p-1 text-slate-500 transition hover:bg-indigo-50 hover:text-indigo-700 focus:outline-none focus:ring-2 focus:ring-indigo-300/80"
                    aria-label={showPassword ? 'Hide password' : 'Show password'}
                  >
                    {showPassword ? <EyeOff className="h-5 w-5" /> : <Eye className="h-5 w-5" />}
                  </button>
                </span>
              </label>

              {isRegister && (
                <label className="block">
                  <span className="mb-2 block text-sm font-semibold text-slate-800">Confirm password</span>
                  <span className="flex h-12 items-center gap-3 rounded-2xl border border-white/72 bg-white/38 px-4 text-slate-950 shadow-[inset_0_1px_0_rgba(255,255,255,0.82),0_14px_36px_-30px_rgba(67,56,202,0.46)] transition backdrop-blur-2xl focus-within:border-indigo-300/90 focus-within:bg-white/58 focus-within:ring-4 focus-within:ring-indigo-300/22">
                    <Lock className="h-5 w-5 text-indigo-600" />
                    <input
                      type={showPassword ? 'text' : 'password'}
                      autoComplete="new-password"
                      className="h-full min-w-0 flex-1 bg-transparent text-sm text-slate-950 outline-none placeholder:text-slate-500"
                      placeholder="Confirm your password"
                      value={form.confirmPassword}
                      onChange={updateField('confirmPassword')}
                      disabled={isSubmitting}
                      minLength={8}
                      required
                    />
                  </span>
                </label>
              )}

              <div className="flex items-center justify-between gap-4 text-sm">
                <label className="flex min-w-0 items-center gap-2 text-slate-700">
                  <input
                    type="checkbox"
                    className="h-4 w-4 rounded border-slate-300 bg-white text-indigo-600 focus:ring-2 focus:ring-indigo-300/60"
                    checked={isRegister ? form.acceptedTerms : form.remember}
                    onChange={updateField(isRegister ? 'acceptedTerms' : 'remember')}
                    disabled={isSubmitting}
                  />
                  <span>{isRegister ? 'I agree to continue' : 'Remember me'}</span>
                </label>
                {!isRegister && (
                  <button
                    type="button"
                    onClick={handleForgotPassword}
                    className="shrink-0 font-semibold text-indigo-700 transition hover:text-indigo-950"
                  >
                    Forgot password?
                  </button>
                )}
              </div>

              <motion.button
                type="submit"
                whileTap={{ scale: 0.97 }}
                disabled={isSubmitting}
                className="flex h-12 w-full items-center justify-center rounded-2xl bg-gradient-to-r from-violet-500 to-indigo-600 text-sm font-semibold text-white shadow-[0_16px_40px_-18px_rgba(129,140,248,0.9)] transition hover:scale-[1.02] hover:shadow-[0_22px_54px_-18px_rgba(129,140,248,1)] focus:outline-none focus:ring-4 focus:ring-violet-300/35 disabled:cursor-not-allowed disabled:opacity-70 disabled:hover:scale-100"
              >
                {isSubmitting ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : null}
                {isSubmitting ? (isRegister ? 'Creating account...' : 'Signing in...') : isRegister ? 'Create Account' : 'Login'}
              </motion.button>
            </form>

            <p className="mt-8 text-center text-sm text-slate-600">
              {isRegister ? 'Already have an account?' : "Don't have an account?"}{' '}
              <button
                type="button"
                onClick={() => switchMode(isRegister ? 'login' : 'register')}
                className="font-semibold text-indigo-700 transition hover:text-indigo-950"
              >
                {isRegister ? 'Login' : 'Register'}
              </button>
            </p>
          </motion.div>
        </section>

        <section className="relative h-full min-h-0 w-full flex-1 overflow-hidden bg-slate-950 md:w-[60%]">
          <video
            className="absolute inset-0 h-full w-full object-cover"
            autoPlay
            muted
            loop
            playsInline
            preload="metadata"
            aria-hidden="true"
          >
            <source src="https://videos.pexels.com/video-files/7947451/7947451-hd_1920_1080_30fps.mp4" type="video/mp4" />
          </video>

          <div className="absolute inset-0 bg-black/60" aria-hidden="true" />
          <div
            className="absolute inset-0 bg-[radial-gradient(circle_at_18%_18%,rgba(99,102,241,0.24),transparent_32%),linear-gradient(90deg,rgba(2,6,23,0.62)_0%,rgba(2,6,23,0.3)_58%,rgba(2,6,23,0.56)_100%)]"
            aria-hidden="true"
          />

          <motion.div
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, ease: 'easeOut', delay: 0.1 }}
            className="relative z-10 flex h-full min-h-0 flex-col justify-between gap-6 p-8 md:p-10 xl:p-12"
          >
            <div className="max-w-3xl space-y-4">
              <p className="text-xs font-medium uppercase tracking-[0.28em] text-white/60">
                Aroha Intelligent Platform
              </p>
              <p className="max-w-3xl bg-[linear-gradient(135deg,#ffffff_0%,#c7d2fe_42%,#67e8f9_100%)] bg-clip-text text-3xl font-black uppercase leading-tight tracking-normal text-transparent drop-shadow-2xl [font-family:ui-rounded,Inter,ui-sans-serif,system-ui,sans-serif] xl:text-5xl">
                Intelligent Data Assistant
              </p>
              <h1 className="max-w-2xl text-4xl font-black uppercase leading-tight tracking-normal text-white drop-shadow-2xl [font-family:ui-rounded,Inter,ui-sans-serif,system-ui,sans-serif] xl:text-5xl">
                Turn Data Into Confident Decisions
              </h1>
              <p className="max-w-2xl text-base leading-7 text-white/74 xl:text-lg">
                A secure AI-powered analytics environment for uploading datasets, profiling data quality,
                generating exploratory insights, preparing models, and forecasting business outcomes from one unified workspace.
              </p>

              <motion.a
                href="#"
                whileTap={{ scale: 0.95 }}
                className="mt-4 inline-flex h-12 items-center gap-2 rounded-full bg-white px-6 text-sm font-medium text-black shadow-[0_18px_50px_-24px_rgba(255,255,255,0.85)] transition hover:scale-105 hover:bg-indigo-50 hover:shadow-[0_24px_64px_-24px_rgba(199,210,254,0.95)] focus:outline-none focus:ring-4 focus:ring-white/30"
              >
                Explore Platform
                <ArrowRight className="h-4 w-4" />
              </motion.a>
            </div>

            <motion.div
              className="grid max-w-lg grid-cols-2 gap-4"
              initial="hidden"
              animate="visible"
              variants={{
                hidden: {},
                visible: {
                  transition: {
                    staggerChildren: 0.09,
                    delayChildren: 0.26,
                  },
                },
              }}
            >
              {heroFeatures.map((feature) => {
                const Icon = feature.icon;

                return (
                  <motion.div
                    key={feature.title}
                    variants={{
                      hidden: { opacity: 0, y: 14 },
                      visible: { opacity: 1, y: 0 },
                    }}
                    transition={{ duration: 0.45, ease: 'easeOut' }}
                    className="group rounded-xl border border-white/10 bg-white/10 p-4 shadow-[0_18px_48px_-34px_rgba(255,255,255,0.5)] backdrop-blur-md transition duration-300 hover:-translate-y-1 hover:scale-[1.03] hover:border-white/20 hover:bg-white/14 hover:shadow-[0_22px_56px_-30px_rgba(129,140,248,0.78)]"
                  >
                    <Icon className="mb-3 h-5 w-5 text-white/80 transition group-hover:text-white" />
                    <h3 className="text-sm font-medium text-white">{feature.title}</h3>
                    <p className="mt-1 text-sm leading-5 text-white/60">{feature.benefit}</p>
                  </motion.div>
                );
              })}
            </motion.div>

            <div className="grid max-w-3xl gap-3 border-t border-white/10 pt-4 text-sm text-white/76 sm:grid-cols-3">
              <div className="flex items-center gap-2 rounded-xl border border-white/10 bg-white/8 px-3 py-2 backdrop-blur-md">
                <MapPin className="h-4 w-4 shrink-0 text-cyan-200" />
                <span className="font-medium">Bangalore</span>
              </div>
              <a
                href="https://aroha.co.in/contact-us/"
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-2 rounded-xl border border-white/10 bg-white/8 px-3 py-2 font-medium text-white/82 backdrop-blur-md transition hover:border-white/20 hover:bg-white/12 hover:text-white"
              >
                <LinkIcon className="h-4 w-4 shrink-0 text-cyan-200" />
                <span className="truncate">Contact Aroha</span>
              </a>
              <a
                href="tel:+919886228615"
                className="flex items-center gap-2 rounded-xl border border-white/10 bg-white/8 px-3 py-2 font-medium text-white/82 backdrop-blur-md transition hover:border-white/20 hover:bg-white/12 hover:text-white"
              >
                <Phone className="h-4 w-4 shrink-0 text-cyan-200" />
                <span className="whitespace-nowrap">+91 9886228615</span>
              </a>
            </div>
          </motion.div>
        </section>
      </div>
    </main>
  );
}
