'use client';

import * as React from 'react';
import { apiClient, getApiErrorMessage } from '@/lib/api';
import type { AuthenticatedUser } from '@/lib/store';
import { useToast } from '@/hooks/use-toast';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Badge } from '@/components/ui/badge';
import { ShieldCheck, Database, Sparkles, ArrowRight, UserRound, Mail, LockKeyhole } from 'lucide-react';

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

export default function LoginPage({ onAuthSuccess }: LoginPageProps) {
  const { toast } = useToast();
  const [mode, setMode] = React.useState<AuthMode>('login');
  const [form, setForm] = React.useState(initialForm);
  const [isSubmitting, setIsSubmitting] = React.useState(false);

  const updateField = (field: keyof typeof initialForm) => (event: React.ChangeEvent<HTMLInputElement>) => {
    setForm((current) => ({ ...current, [field]: event.target.value }));
  };

  const switchMode = (nextMode: AuthMode) => {
    setMode(nextMode);
    setForm(initialForm);
  };

  const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();

    if (mode === 'register' && form.password !== form.confirmPassword) {
      toast({
        title: 'Password mismatch',
        description: 'Password and confirm password must match before we can create the account.',
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

      onAuthSuccess(response.data.user);
      toast({
        title: mode === 'register' ? 'Account created' : 'Login successful',
        description:
          mode === 'register'
            ? `Welcome, ${response.data.user.username}. Your account is secured and your session is active.`
            : `Welcome back, ${response.data.user.username}. Your secure session has been restored.`,
      });
    } catch (error) {
      toast({
        title: mode === 'register' ? 'Registration failed' : 'Login failed',
        description: getApiErrorMessage(
          error,
          mode === 'register' ? 'We could not create your account.' : 'We could not sign you in.'
        ),
        variant: 'destructive',
      });
    } finally {
      setIsSubmitting(false);
    }
  };

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
            <Badge className="rounded-full border-sky-200 bg-white/80 px-3 py-1 text-sky-900 shadow-sm">
              <ShieldCheck className="mr-2 h-3.5 w-3.5" />
              Secure workspace access
            </Badge>
            <h1 className="mt-6 text-4xl font-semibold tracking-tight text-slate-950 sm:text-5xl">
              Authenticate before entering the Intelligent Data Assistant
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
              <div className="mb-4 inline-flex h-14 w-14 items-center justify-center rounded-3xl bg-[linear-gradient(145deg,#082f49_0%,#0f766e_100%)] text-white shadow-[0_18px_44px_-24px_rgba(8,47,73,0.72)]">
                <Sparkles className="h-6 w-6" />
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
                  {isSubmitting
                    ? mode === 'register' ? 'Creating account...' : 'Signing in...'
                    : mode === 'register' ? 'Create account' : 'Continue to workspace'}
                  <ArrowRight className="ml-2 h-4 w-4" />
                </Button>
              </form>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
