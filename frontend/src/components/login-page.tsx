'use client';

import * as React from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import {
  AlertCircle,
  BarChart3,
  CheckCircle2,
  Database,
  LineChart,
  Loader2,
  Lock,
  LogIn,
  Mail,
  MapPin,
  Phone,
  ShieldCheck,
  User,
} from 'lucide-react';
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

const AROHA_CONTACT_URL = 'https://aroha.co.in/contact-us/';
const ANALYTICS_VIDEO_URL = 'https://assets.mixkit.co/videos/preview/mixkit-business-people-analyzing-marketing-data-46685-large.mp4';

const features = [
  { label: 'Secure Workspace', icon: ShieldCheck },
  { label: 'Smart Upload', icon: Database },
  { label: 'Auto Analysis', icon: BarChart3 },
  { label: 'AI Forecasting', icon: LineChart },
];

export default function LoginPage({ onAuthSuccess }: LoginPageProps) {
  const { toast } = useToast();
  const [mode, setMode] = React.useState<AuthMode>('login');
  const [modeDirection, setModeDirection] = React.useState(1);
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
    setInfoMessage('Password reset is handled by Aroha support. Use the support desk link for assistance.');
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
    <main className="relative min-h-screen overflow-hidden bg-[#f8fbff] font-['Plus_Jakarta_Sans','Inter','Segoe_UI',sans-serif] text-[#15263a]">
      <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(circle_at_17%_20%,rgba(99,102,241,0.1),transparent_30%),radial-gradient(circle_at_82%_42%,rgba(34,211,238,0.16),transparent_34%),linear-gradient(180deg,#ffffff_0%,#f8fbff_58%,#edf6ff_100%)]" />
      <motion.div
        className="pointer-events-none absolute left-[-16%] top-[14%] h-[32rem] w-[32rem] rounded-full border border-indigo-200/42"
        animate={{ rotate: 360, scale: [1, 1.05, 1] }}
        transition={{ rotate: { duration: 38, repeat: Infinity, ease: 'linear' }, scale: { duration: 7, repeat: Infinity } }}
      />
      <motion.div
        className="pointer-events-none absolute bottom-[-19%] right-[-8%] h-[30rem] w-[30rem] rounded-full border border-cyan-200/55"
        animate={{ rotate: -360, scale: [1, 1.08, 1] }}
        transition={{ rotate: { duration: 42, repeat: Infinity, ease: 'linear' }, scale: { duration: 8, repeat: Infinity } }}
      />

      <AnimatePresence>
        {isSubmitting ? (
          <motion.div
            key="auth-loading-bar"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="absolute left-0 top-0 z-50 h-1 w-full bg-slate-200"
          >
            <motion.div
              className="h-full bg-[linear-gradient(90deg,#6c63ff_0%,#22d3ee_52%,#35d08f_100%)] shadow-[0_0_18px_rgba(99,102,241,0.55)]"
              initial={{ x: '-45%', width: '35%' }}
              animate={{ x: '210%', width: '45%' }}
              transition={{ duration: 1.1, repeat: Infinity, ease: 'easeInOut' }}
            />
          </motion.div>
        ) : null}
      </AnimatePresence>

      <div className="relative mx-auto grid min-h-screen w-full max-w-[92rem] items-center gap-8 px-4 py-8 sm:px-8 lg:grid-cols-[0.84fr_1.16fr] lg:px-12">
          <section className="relative flex min-h-[660px] items-center justify-center overflow-hidden rounded-[2rem] bg-white/42 p-5 shadow-[0_24px_80px_-48px_rgba(15,23,42,0.35)] sm:p-8 lg:p-10">
            <div className="pointer-events-none absolute inset-0 bg-[linear-gradient(135deg,rgba(255,255,255,0.7)_0%,rgba(255,255,255,0.2)_45%,rgba(219,234,254,0.42)_100%)]" />
            <motion.div
              className="pointer-events-none absolute left-10 top-16 h-20 w-20 rounded-full border border-indigo-200/70"
              animate={{ y: [0, -12, 0], opacity: [0.45, 0.85, 0.45] }}
              transition={{ duration: 5.5, repeat: Infinity }}
            />
            <motion.div
              className="pointer-events-none absolute bottom-16 right-12 h-16 w-16 rounded-2xl border border-cyan-300/50"
              animate={{ rotate: [0, 90, 0], opacity: [0.35, 0.75, 0.35] }}
              transition={{ duration: 8, repeat: Infinity }}
            />

            <motion.div
              initial={{ opacity: 0, y: 18 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.55, ease: 'easeOut' }}
              className="relative w-full max-w-[30rem] overflow-hidden rounded-[1.6rem] border border-white/80 bg-white/58 px-6 py-8 shadow-[0_28px_70px_-38px_rgba(35,78,158,0.35),0_18px_42px_-34px_rgba(15,23,42,0.3),inset_0_1px_0_rgba(255,255,255,0.9)] backdrop-blur-2xl sm:px-8 sm:py-10"
            >
              <div className="pointer-events-none absolute inset-0 bg-[linear-gradient(120deg,rgba(255,255,255,0.85),rgba(255,255,255,0.28)_44%,rgba(219,234,254,0.44))]" />
              <motion.div
                className="pointer-events-none absolute -left-10 top-20 h-28 w-[118%] rotate-[-18deg] bg-white/60 blur-2xl"
                animate={{ x: ['-28%', '22%', '-28%'], opacity: [0.35, 0.72, 0.35] }}
                transition={{ duration: 7, repeat: Infinity, ease: 'easeInOut' }}
              />
              <motion.div
                className="pointer-events-none absolute inset-y-0 -left-1/2 w-1/2 bg-[linear-gradient(90deg,transparent_0%,rgba(255,255,255,0.78)_50%,transparent_100%)]"
                animate={{ x: ['0%', '315%'] }}
                transition={{ duration: 5.8, repeat: Infinity, ease: 'linear' }}
              />

              <div className="relative">
                <motion.div
                  className="mx-auto mb-8 flex h-24 w-24 items-center justify-center rounded-full border border-slate-200/80 bg-white/80 shadow-[0_18px_42px_-28px_rgba(99,102,241,0.45)]"
                  animate={{ y: [0, -5, 0] }}
                  transition={{ duration: 4, repeat: Infinity, ease: 'easeInOut' }}
                >
                  <div className="relative flex h-16 w-16 items-center justify-center rounded-full bg-[linear-gradient(135deg,#6c63ff_0%,#8b7cff_100%)]">
                    <User className="h-8 w-8 text-white" />
                    <span className="absolute -right-1 top-2 h-3 w-3 rounded-full bg-[#35d08f] ring-4 ring-white" />
                  </div>
                </motion.div>

                <div className="mb-9 text-center">
                  <p className="text-xs font-bold uppercase tracking-[0.3em] text-slate-500">Aroha Secure Access</p>
                  <h1 className="mt-4 bg-[linear-gradient(135deg,#5b55ff_0%,#7b6cff_48%,#22b8cf_100%)] bg-clip-text text-5xl font-black tracking-tight text-transparent">
                      {mode === 'register' ? 'Sign Up' : 'Login'}
                  </h1>
                  <p className="mt-5 text-base font-medium text-slate-600">Exploring more by connecting with us</p>
                </div>

                <div className="min-h-[360px] overflow-hidden">
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
                            <Alert variant="destructive" className="border-red-200 bg-red-50 text-red-950">
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
                            <Alert className="border-indigo-200 bg-indigo-50 text-indigo-950">
                              <AlertCircle />
                              <AlertTitle>Password help</AlertTitle>
                              <AlertDescription>{infoMessage}</AlertDescription>
                            </Alert>
                          </motion.div>
                        ) : null}
                      </AnimatePresence>

                      <AnimatePresence>
                        {showSuccess ? (
                          <motion.div
                            initial={{ opacity: 0, scale: 0.98 }}
                            animate={{ opacity: 1, scale: 1 }}
                            exit={{ opacity: 0, scale: 0.98 }}
                            className="rounded-lg border border-emerald-200 bg-emerald-50 px-3 py-2 text-emerald-950"
                          >
                            <div className="flex items-center gap-2">
                              <CheckCircle2 className="h-4 w-4" />
                              <p className="text-sm font-semibold">{mode === 'register' ? 'Account created successfully.' : 'Login successful. Opening workspace...'}</p>
                            </div>
                          </motion.div>
                        ) : null}
                      </AnimatePresence>

                      {mode === 'register' && (
                        <GlassInput
                          id="username"
                          label="User Name"
                          type="text"
                          autoComplete="username"
                          placeholder="Enter your user name"
                          value={form.username}
                          onChange={updateField('username')}
                          disabled={isSubmitting}
                          icon={User}
                          required
                        />
                      )}

                      <GlassInput
                        id="email"
                        label="User Name / ID"
                        type="email"
                        autoComplete="email"
                        placeholder="name@company.com"
                        value={form.email}
                        onChange={updateField('email')}
                        disabled={isSubmitting}
                        icon={User}
                        required
                      />

                      <GlassInput
                        id="password"
                        label="Password"
                        type="password"
                        autoComplete={mode === 'register' ? 'new-password' : 'current-password'}
                        placeholder="Enter your password"
                        value={form.password}
                        onChange={updateField('password')}
                        disabled={isSubmitting}
                        icon={Lock}
                        minLength={8}
                        required
                      />

                      {mode === 'register' && (
                        <GlassInput
                          id="confirm-password"
                          label="Confirm Password"
                          type="password"
                          autoComplete="new-password"
                          placeholder="Re-enter password"
                          value={form.confirmPassword}
                          onChange={updateField('confirmPassword')}
                          disabled={isSubmitting}
                          icon={Lock}
                          minLength={8}
                          required
                        />
                      )}

                      {mode === 'login' && (
                        <div className="flex items-center justify-between text-sm text-slate-600">
                          <label className="flex items-center gap-2">
                            <span className="h-3.5 w-3.5 rounded-[3px] border border-slate-300 bg-white shadow-inner" />
                            Stay signed in
                          </label>
                          <button type="button" onClick={handleForgotPassword} className="font-semibold text-[#5f55ff] hover:text-[#4238d3] hover:underline">
                            Forgot password?
                          </button>
                        </div>
                      )}

                      <Button
                        type="submit"
                        disabled={isSubmitting}
                        className="group relative h-12 w-full overflow-hidden rounded-sm bg-[linear-gradient(135deg,#6c63ff_0%,#786cff_100%)] text-white shadow-[0_18px_42px_-24px_rgba(99,102,241,0.72)] transition-all duration-300 hover:-translate-y-0.5 hover:brightness-105"
                      >
                        <motion.span
                          className="pointer-events-none absolute inset-y-0 -left-1/2 w-1/2 bg-[linear-gradient(90deg,transparent_0%,rgba(255,255,255,0.42)_50%,transparent_100%)]"
                          animate={isSubmitting ? { x: ['0%', '320%'] } : { x: '-160%' }}
                          transition={isSubmitting ? { duration: 0.9, repeat: Infinity, ease: 'linear' } : { duration: 0.3 }}
                        />
                        {isSubmitting && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
                        {isSubmitting ? (mode === 'register' ? 'Creating account...' : 'Signing in...') : (mode === 'register' ? 'Create Account' : 'Login')}
                        {!isSubmitting && <LogIn className="ml-2 h-4 w-4" />}
                      </Button>

                      <p className="text-center text-sm text-slate-600">
                        {mode === 'login' ? 'New here?' : 'Already have an account?'}{' '}
                        <button
                          type="button"
                          onClick={() => switchMode(mode === 'login' ? 'register' : 'login')}
                          className="font-black text-[#5f55ff] underline decoration-indigo-300 underline-offset-4"
                        >
                          {mode === 'login' ? 'Sign up' : 'Back to Login'}
                        </button>
                      </p>
                    </motion.form>
                  </AnimatePresence>
                </div>
              </div>
            </motion.div>
          </section>

          <section className="group relative min-h-[660px] overflow-hidden rounded-[2rem] bg-white shadow-[0_24px_90px_-54px_rgba(15,23,42,0.42)]">
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
            <div className="absolute inset-0 bg-[linear-gradient(90deg,rgba(255,255,255,0.16)_0%,rgba(18,78,128,0.52)_48%,rgba(5,31,58,0.72)_100%)]" />
            <div className="absolute inset-0 bg-[linear-gradient(180deg,rgba(255,255,255,0.2)_0%,rgba(0,0,0,0.28)_100%)]" />
            <motion.div
              className="pointer-events-none absolute -inset-y-8 -left-1/2 w-1/2 bg-[linear-gradient(90deg,transparent_0%,rgba(255,255,255,0.34)_50%,transparent_100%)]"
              animate={{ x: ['0%', '305%'] }}
              transition={{ duration: 7.5, repeat: Infinity, ease: 'linear' }}
            />

            <div className="relative z-10 flex h-full min-h-[660px] flex-col justify-between p-7 text-white sm:p-10 lg:p-12">
              <div>
                <motion.div
                  initial={{ opacity: 0, y: 18 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.7 }}
                  className="mb-9 flex items-center gap-4"
                >
                  <div className="flex h-24 w-24 items-center justify-center overflow-hidden rounded-2xl border border-white/45 bg-white/20 p-2 shadow-[0_18px_50px_-28px_rgba(0,0,0,0.7)] backdrop-blur-md">
                    <img src="/company-logo.png" alt="Aroha Technologies Company Logo" className="h-full w-full object-contain drop-shadow-[0_10px_24px_rgba(0,0,0,0.24)]" />
                  </div>
                  <div>
                    <p className="text-sm font-black uppercase tracking-[0.28em] text-cyan-50/90">Aroha Intelligent Platform</p>
                    <p className="mt-2 text-2xl font-black uppercase tracking-[0.12em]">Intelligent Data Assistant</p>
                  </div>
                </motion.div>

                <motion.div
                  initial={{ opacity: 0, x: -18 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ duration: 0.7, delay: 0.15 }}
                >
                  <h2 className="max-w-2xl text-4xl font-black uppercase leading-tight tracking-tight sm:text-5xl lg:text-6xl">
                    Insight-Driven Data Analysis for Modern Teams
                  </h2>
                  <p className="mt-5 max-w-xl text-lg font-semibold leading-8 text-white/92">
                    A unified application for secure data upload, profiling, exploratory analysis, cleaning, forecasting, and model-assisted business insight generation for faster decisions.
                  </p>
                </motion.div>

                <div className="mt-8 grid max-w-2xl grid-cols-2 gap-3">
                  {features.map((feature, index) => {
                    const Icon = feature.icon;
                    return (
                      <motion.div
                        key={feature.label}
                        initial={{ opacity: 0, y: 14 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.42, delay: 0.24 + index * 0.08 }}
                        whileHover={{ y: -3 }}
                        className="rounded-xl border border-white/30 bg-white/16 p-3 shadow-[0_16px_42px_-30px_rgba(0,0,0,0.58)] backdrop-blur-md"
                      >
                        <Icon className="mb-2 h-5 w-5 text-cyan-100" />
                        <p className="text-sm font-bold">{feature.label}</p>
                      </motion.div>
                    );
                  })}
                </div>
              </div>

              <motion.div
                initial={{ opacity: 0, y: 18 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.7, delay: 0.4 }}
                className="mt-8 rounded-2xl border border-white/36 bg-white/18 p-4 font-bold shadow-[0_18px_48px_-30px_rgba(0,0,0,0.65)] backdrop-blur-md"
              >
                <p className="flex items-center gap-2 text-sm sm:text-base">
                  <MapPin className="h-4 w-4" />
                  Location: Bangalore
                </p>
                <p className="mt-3 flex items-center gap-2 text-sm sm:text-base">
                  <Mail className="h-4 w-4" />
                  Contact:
                  <a href={AROHA_CONTACT_URL} target="_blank" rel="noreferrer" className="font-black underline underline-offset-4 hover:text-cyan-100">
                    https://aroha.co.in/contact-us/
                  </a>
                </p>
                <p className="mt-3 flex items-center gap-2 text-sm sm:text-base">
                  <Phone className="h-4 w-4" />
                  Phone: +91 9886228615
                </p>
              </motion.div>
            </div>
          </section>
      </div>
    </main>
  );
}

function GlassInput({
  id,
  label,
  icon: Icon,
  ...props
}: React.ComponentProps<typeof Input> & {
  id: string;
  label: string;
  icon: React.ElementType;
}) {
  return (
    <div className="space-y-2">
      <Label htmlFor={id} className="text-sm font-bold text-slate-700">
        {label}
      </Label>
      <div className="group relative overflow-hidden rounded-sm p-[1px]">
        <motion.span
          className="pointer-events-none absolute inset-y-[-90%] left-[-45%] w-1/2 rotate-12 bg-[linear-gradient(90deg,transparent_0%,rgba(99,102,241,0.7)_48%,rgba(255,255,255,0.95)_52%,transparent_100%)] opacity-0 blur-sm transition-opacity duration-300 group-hover:opacity-100 group-focus-within:opacity-100"
          animate={{ x: ['0%', '330%'] }}
          transition={{ duration: 2.8, repeat: Infinity, ease: 'linear' }}
        />
        <div className="relative z-10">
          <Icon className="pointer-events-none absolute right-4 top-1/2 z-20 h-4 w-4 -translate-y-1/2 text-slate-400" />
          <Input
            id={id}
            {...props}
            className="h-12 rounded-sm border-slate-200 bg-white/78 px-5 pr-12 text-slate-800 shadow-[0_12px_30px_-28px_rgba(15,23,42,0.48),inset_0_1px_0_rgba(255,255,255,0.9)] placeholder:text-slate-500 focus-visible:ring-[#6c63ff]/35"
          />
        </div>
      </div>
    </div>
  );
}
