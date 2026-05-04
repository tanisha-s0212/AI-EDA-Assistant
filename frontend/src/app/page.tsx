"use client";

import { motion } from "framer-motion";
import {
  ArrowRight,
  BarChart3,
  BrainCircuit,
  LinkIcon,
  MapPin,
  Phone,
  Eye,
  EyeOff,
  Lock,
  Mail,
  ShieldCheck,
  Sparkles,
  UploadCloud,
} from "lucide-react";
import { useState } from "react";

function LeftLoginPanel() {
  const [showPassword, setShowPassword] = useState(false);
  const [authMode, setAuthMode] = useState<"login" | "register">("login");
  const isRegister = authMode === "register";

  return (
    <section className="relative flex min-h-0 w-full items-center justify-center overflow-hidden border-b border-white/20 bg-[radial-gradient(circle_at_18%_16%,rgba(255,255,255,0.62),transparent_30%),radial-gradient(circle_at_78%_72%,rgba(129,140,248,0.22),transparent_34%),linear-gradient(135deg,#ede9fe_0%,#dbeafe_48%,#eef2ff_100%)] px-6 py-8 md:h-full md:w-[40%] md:border-b-0 md:border-r">
      <motion.div
        aria-hidden="true"
        className="absolute right-[16%] top-[18%] text-indigo-400/55"
        animate={{ y: [-8, 8, -8] }}
        transition={{ duration: 4.5, repeat: Infinity, ease: "easeInOut" }}
      >
        <Sparkles className="h-10 w-10 drop-shadow-[0_0_24px_rgba(99,102,241,0.4)]" />
      </motion.div>

      <motion.div
        initial={{ opacity: 0, y: 18 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.55, ease: "easeOut" }}
        className="relative w-full max-w-md rounded-[2rem] border border-white/45 bg-white/18 p-8 shadow-[0_32px_100px_-44px_rgba(30,41,59,0.58)] ring-1 ring-white/25 backdrop-blur-2xl"
      >
        <div className="absolute left-1/2 top-0 flex h-12 w-40 -translate-x-1/2 -translate-y-px items-center justify-center rounded-b-2xl bg-white/86 text-sm font-semibold text-slate-900 shadow-[0_16px_36px_-24px_rgba(15,23,42,0.55)]">
          {isRegister ? "Register" : "Login"}
        </div>

        <a
          href="https://aroha.co.in/"
          target="_blank"
          rel="noreferrer"
          className="mx-auto mt-8 flex w-fit items-center rounded-xl focus:outline-none focus:ring-2 focus:ring-indigo-400/60 focus:ring-offset-2 focus:ring-offset-white/60"
          aria-label="Visit Aroha website"
        >
          <img
            src="/company-logo.png"
            alt="Aroha"
            className="h-32 w-auto object-contain drop-shadow-[0_18px_42px_rgba(67,56,202,0.24)]"
          />
        </a>

        <div className="mt-4 text-center">
          <p className="text-xs font-semibold uppercase tracking-[0.24em] text-indigo-700">
            Aroha Intelligent Platform
          </p>
          <h1 className="mt-3 text-3xl font-semibold tracking-normal text-slate-950">
            {isRegister ? "Create Your Account" : "Welcome Back"}
          </h1>
          <p className="mt-3 text-sm leading-6 text-slate-700">
            {isRegister
              ? "Register to start a secure AI analytics workspace for your business data."
              : "Sign in to continue your secure analytics workspace and active data workflows."}
          </p>
        </div>

        <div className="mt-6 grid grid-cols-2 rounded-full border border-white/35 bg-white/20 p-1 shadow-inner shadow-white/20 backdrop-blur-xl">
          <button
            type="button"
            onClick={() => setAuthMode("login")}
            className={`h-10 rounded-full text-sm font-semibold transition ${
              !isRegister ? "bg-white text-slate-950 shadow-sm" : "text-slate-700 hover:text-slate-950"
            }`}
          >
            Login
          </button>
          <button
            type="button"
            onClick={() => setAuthMode("register")}
            className={`h-10 rounded-full text-sm font-semibold transition ${
              isRegister ? "bg-white text-slate-950 shadow-sm" : "text-slate-700 hover:text-slate-950"
            }`}
          >
            Register
          </button>
        </div>

        <form className="mt-6 space-y-4">
          {isRegister && (
            <label className="block">
              <span className="mb-2 block text-sm font-semibold text-slate-800">Full name</span>
              <span className="flex h-12 items-center gap-3 rounded-2xl border border-white/45 bg-white/22 px-4 text-slate-950 shadow-inner shadow-white/20 transition backdrop-blur-xl focus-within:border-indigo-300/80 focus-within:bg-white/38 focus-within:ring-4 focus-within:ring-indigo-300/20">
                <Sparkles className="h-5 w-5 text-indigo-600" />
                <input
                  type="text"
                  autoComplete="name"
                  className="h-full min-w-0 flex-1 bg-transparent text-sm text-slate-950 outline-none placeholder:text-slate-500"
                  placeholder="Your name"
                />
              </span>
            </label>
          )}

          <label className="block">
            <span className="mb-2 block text-sm font-semibold text-slate-800">Email</span>
            <span className="flex h-12 items-center gap-3 rounded-2xl border border-white/45 bg-white/22 px-4 text-slate-950 shadow-inner shadow-white/20 transition backdrop-blur-xl focus-within:border-indigo-300/80 focus-within:bg-white/38 focus-within:ring-4 focus-within:ring-indigo-300/20">
              <Mail className="h-5 w-5 text-indigo-600" />
              <input
                type="email"
                autoComplete="email"
                className="h-full min-w-0 flex-1 bg-transparent text-sm text-slate-950 outline-none placeholder:text-slate-500"
                placeholder="you@example.com"
              />
            </span>
          </label>

          <label className="block">
            <span className="mb-2 block text-sm font-semibold text-slate-800">Password</span>
            <span className="flex h-12 items-center gap-3 rounded-2xl border border-white/45 bg-white/22 px-4 text-slate-950 shadow-inner shadow-white/20 transition backdrop-blur-xl focus-within:border-indigo-300/80 focus-within:bg-white/38 focus-within:ring-4 focus-within:ring-indigo-300/20">
              <Lock className="h-5 w-5 text-indigo-600" />
              <input
                type={showPassword ? "text" : "password"}
                autoComplete={isRegister ? "new-password" : "current-password"}
                className="h-full min-w-0 flex-1 bg-transparent text-sm text-slate-950 outline-none placeholder:text-slate-500"
                placeholder="Enter your password"
              />
              <button
                type="button"
                onClick={() => setShowPassword((current) => !current)}
                className="rounded-lg p-1 text-slate-500 transition hover:bg-indigo-50 hover:text-indigo-700 focus:outline-none focus:ring-2 focus:ring-indigo-300/80"
                aria-label={showPassword ? "Hide password" : "Show password"}
              >
                {showPassword ? <EyeOff className="h-5 w-5" /> : <Eye className="h-5 w-5" />}
              </button>
            </span>
          </label>

          {isRegister && (
            <label className="block">
              <span className="mb-2 block text-sm font-semibold text-slate-800">Confirm password</span>
              <span className="flex h-12 items-center gap-3 rounded-2xl border border-white/45 bg-white/22 px-4 text-slate-950 shadow-inner shadow-white/20 transition backdrop-blur-xl focus-within:border-indigo-300/80 focus-within:bg-white/38 focus-within:ring-4 focus-within:ring-indigo-300/20">
                <Lock className="h-5 w-5 text-indigo-600" />
                <input
                  type={showPassword ? "text" : "password"}
                  autoComplete="new-password"
                  className="h-full min-w-0 flex-1 bg-transparent text-sm text-slate-950 outline-none placeholder:text-slate-500"
                  placeholder="Confirm your password"
                />
              </span>
            </label>
          )}

          <div className="flex items-center justify-between gap-4 text-sm">
            <label className="flex min-w-0 items-center gap-2 text-slate-700">
              <input
                type="checkbox"
                className="h-4 w-4 rounded border-slate-300 bg-white text-indigo-600 focus:ring-2 focus:ring-indigo-300/60"
              />
              <span>{isRegister ? "I agree to continue" : "Remember me"}</span>
            </label>
            {!isRegister && <a href="#" className="shrink-0 font-semibold text-indigo-700 transition hover:text-indigo-950">
              Forgot password?
            </a>}
          </div>

          <motion.button
            type="submit"
            whileTap={{ scale: 0.97 }}
            className="h-12 w-full rounded-2xl bg-gradient-to-r from-violet-500 to-indigo-600 text-sm font-semibold text-white shadow-[0_16px_40px_-18px_rgba(129,140,248,0.9)] transition hover:scale-[1.02] hover:shadow-[0_22px_54px_-18px_rgba(129,140,248,1)] focus:outline-none focus:ring-4 focus:ring-violet-300/35"
          >
            {isRegister ? "Create Account" : "Login"}
          </motion.button>
        </form>

        <p className="mt-8 text-center text-sm text-slate-600">
          {isRegister ? "Already have an account?" : "Don't have an account?"}{" "}
          <button
            type="button"
            onClick={() => setAuthMode(isRegister ? "login" : "register")}
            className="font-semibold text-indigo-700 transition hover:text-indigo-950"
          >
            {isRegister ? "Login" : "Register"}
          </button>
        </p>
      </motion.div>
    </section>
  );
}

const heroFeatures = [
  {
    title: "Secure Workspace",
    benefit: "Protected sessions for every dataset.",
    icon: ShieldCheck,
  },
  {
    title: "Smart Upload",
    benefit: "Bring files in with guided validation.",
    icon: UploadCloud,
  },
  {
    title: "Auto Analysis",
    benefit: "Surface patterns without manual setup.",
    icon: BarChart3,
  },
  {
    title: "AI Forecasting",
    benefit: "Project outcomes with assisted models.",
    icon: BrainCircuit,
  },
];

function RightFeatureGrid() {
  return (
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
            transition={{ duration: 0.45, ease: "easeOut" }}
            className="group rounded-xl border border-white/10 bg-white/10 p-4 shadow-[0_18px_48px_-34px_rgba(255,255,255,0.5)] backdrop-blur-md transition duration-300 hover:-translate-y-1 hover:scale-[1.03] hover:border-white/20 hover:bg-white/14 hover:shadow-[0_22px_56px_-30px_rgba(129,140,248,0.78)]"
          >
            <Icon className="mb-3 h-5 w-5 text-white/80 transition group-hover:text-white" />
            <h3 className="text-sm font-medium text-white">{feature.title}</h3>
            <p className="mt-1 text-sm leading-5 text-white/60">{feature.benefit}</p>
          </motion.div>
        );
      })}
    </motion.div>
  );
}

function RightHeroPanel() {
  return (
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
        <source
          src="https://videos.pexels.com/video-files/7947451/7947451-hd_1920_1080_30fps.mp4"
          type="video/mp4"
        />
      </video>

      <div className="absolute inset-0 bg-black/60" aria-hidden="true" />
      <div
        className="absolute inset-0 bg-[radial-gradient(circle_at_18%_18%,rgba(99,102,241,0.24),transparent_32%),linear-gradient(90deg,rgba(2,6,23,0.62)_0%,rgba(2,6,23,0.3)_58%,rgba(2,6,23,0.56)_100%)]"
        aria-hidden="true"
      />

      <motion.div
        initial={{ opacity: 0, y: 16 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, ease: "easeOut", delay: 0.1 }}
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

        <RightFeatureGrid />

        <div className="flex items-center justify-between gap-4 border-t border-white/10 pt-4 text-sm text-white/70">
          <div className="flex items-center gap-2 whitespace-nowrap">
            <MapPin className="h-4 w-4" />
            <span>Bangalore</span>
          </div>

          <a
            href="https://aroha.co.in/contact-us/"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-2 whitespace-nowrap transition hover:text-white"
          >
            <LinkIcon className="h-4 w-4" />
            <span className="underline underline-offset-4">Contact</span>
          </a>

          <div className="flex items-center gap-2 whitespace-nowrap">
            <Phone className="h-4 w-4" />
            <span>+91 9886228615</span>
          </div>
        </div>
      </motion.div>
    </section>
  );
}

function FixedFooter() {
  const tickerText =
    "Aroha Technologies – Intelligent Data Assistant | AI-driven analytics, secure data processing, and enterprise insights platform";

  return (
    <footer className="fixed inset-x-0 bottom-0 z-40 overflow-hidden border-t border-white/10 bg-black/40 text-white shadow-[0_-18px_54px_-34px_rgba(0,0,0,0.9)] backdrop-blur-md">
      <div className="flex h-11 items-center justify-start gap-5 overflow-hidden px-5 py-2 text-xs font-medium text-white/86 sm:px-8 sm:text-sm">
        <span className="inline-flex items-center gap-2 whitespace-nowrap">
          <MapPin className="h-4 w-4 text-indigo-100" />
          Bangalore
        </span>
        <a
          href="mailto:hr@aroha.co.in"
          className="inline-flex items-center gap-2 whitespace-nowrap transition hover:text-white"
        >
          <Mail className="h-4 w-4 text-indigo-100" />
          hr@aroha.co.in
        </a>
        <a
          href="tel:+919886228615"
          className="inline-flex items-center gap-2 whitespace-nowrap transition hover:text-white"
        >
          <Phone className="h-4 w-4 text-indigo-100" />
          +91 9886228615
        </a>
      </div>

      <div className="overflow-hidden border-t border-white/10 bg-white/10 py-2">
        <div className="footer-ticker-track flex w-max whitespace-nowrap text-xs font-medium uppercase tracking-[0.18em] text-white/70">
          <span className="pr-16">{tickerText.replace("\u00e2\u20ac\u201c", "-")}</span>
          <span className="pr-16" aria-hidden="true">
            {tickerText.replace("\u00e2\u20ac\u201c", "-")}
          </span>
        </div>
      </div>

      <style jsx>{`
        .footer-ticker-track {
          animation: footerTicker 28s linear infinite;
        }

        @keyframes footerTicker {
          from {
            transform: translateX(0);
          }
          to {
            transform: translateX(-50%);
          }
        }
      `}</style>
    </footer>
  );
}

export default function Home() {
  return (
    <div
      className="relative h-screen max-h-screen w-screen overflow-hidden bg-background text-foreground"
      style={{
        fontFamily:
          "'Satoshi', 'Clash Display', 'General Sans', Inter, ui-sans-serif, system-ui, sans-serif",
      }}
    >
      <main className="flex h-full min-h-0 w-full flex-col overflow-hidden md:flex-row">
        <LeftLoginPanel />

        <RightHeroPanel />
      </main>
    </div>
  );
}
