"use client";

import { useState } from "react";


// Extract FastAPI error messages for fields and non-fields
function parseFastApiErrors(detail: any) {
  const fieldErrors: Record<string, string[]> = {};
  let nonFieldErrors: string[] = [];
  if (typeof detail === 'string') {
    nonFieldErrors.push(detail);
  } else if (Array.isArray(detail)) {
    for (const err of detail) {
      if (err && typeof err === 'object' && err.loc && Array.isArray(err.loc) && err.msg) {
        // loc: ["body", "email"] or ["body", "password"]
        const field = err.loc[1] || err.loc[0];
        if (field === 'email' || field === 'password') {
          if (!fieldErrors[field]) fieldErrors[field] = [];
          fieldErrors[field].push(err.msg);
        } else {
          nonFieldErrors.push(err.msg);
        }
      } else if (err && err.msg) {
        nonFieldErrors.push(err.msg);
      } else if (typeof err === 'string') {
        nonFieldErrors.push(err);
      }
    }
  } else if (detail && typeof detail === 'object' && detail.msg) {
    nonFieldErrors.push(detail.msg);
  }
  return { fieldErrors, nonFieldErrors };
}
import Link from "next/link";
import { useRouter } from "next/navigation";
import Image from "next/image";
import { useForm } from "react-hook-form";
import { useAuth } from "@/providers";
import {
  Form,
  FormControl,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
  Button,
  Input,
} from "@/components/ui";
import { Mail, Lock } from "lucide-react";

interface LoginFormData {
  email: string;
  password: string;
}



export function LoginForm() {
  const router = useRouter();
  const { login, isLoading, error } = useAuth();
  const [serverError, setServerError] = useState<string | null>(null);
  const [fieldErrors, setFieldErrors] = useState<Record<string, string[]>>({});
  const form = useForm<LoginFormData>({
    defaultValues: { email: "", password: "" },
  });

  const onSubmit = async (data: LoginFormData) => {
    setServerError(null);
    setFieldErrors({});
    // Enforce frontend validation for empty fields
    if (!data.email || !data.email.trim()) {
      form.setError('email', { type: 'manual', message: 'Email is required' });
      return;
    }
    if (!data.password || !data.password.trim()) {
      form.setError('password', { type: 'manual', message: 'Password is required' });
      return;
    }
    try {
      // Debug: log what is being sent to backend
      const loginPayload = { username: data.email, password: data.password };
      console.log('Login payload:', loginPayload);
      await login(loginPayload);
      router.push("/dashboard");
    } catch (err: any) {
      // error is handled in AuthProvider, but we show it here too
      let detail = err?.response?.data?.detail;
      const { fieldErrors, nonFieldErrors } = parseFastApiErrors(detail);
      setFieldErrors(fieldErrors);
      setServerError(nonFieldErrors.length ? nonFieldErrors.join(' | ') : "Login failed. Please try again.");
      // Optionally, set react-hook-form field errors for UI feedback
      Object.entries(fieldErrors).forEach(([field, messages]) => {
        // Defensive: always pass a string, never an object
        const msg = messages.map(m => typeof m === 'string' ? m : (m && m.msg ? m.msg : JSON.stringify(m))).join(' | ');
        form.setError(field as keyof LoginFormData, { type: 'server', message: msg });
      });
    }
  };

  return (
    <div className="rounded-3xl bg-white p-8 shadow-lg dark:bg-neutral-900 dark:shadow-2xl">
      <div className="space-y-8">
        {/* Logo & Header */}
        <div className="flex flex-col items-center gap-6">
          <div className="flex h-24 w-24 items-center justify-center rounded-3xl shadow-lg" style={{ background: 'linear-gradient(to bottom right, #8FBC8F, #4A90A4, #87CEEB)' }}>
            <div className="relative h-16 w-16">
              <Image
                src="/logo.png"
                alt="Plant Care Assistant"
                fill
                sizes="64px"
                className="object-contain drop-shadow-lg"
                priority
              />
            </div>
          </div>
          <div className="text-center">
            <h1 className="text-2xl font-bold text-neutral-900 dark:text-white">Welcome back</h1>
            <p className="mt-2 text-sm text-neutral-600 dark:text-neutral-400">
              Sign in to continue caring for your plants
            </p>
          </div>
        </div>

        {/* Login Form */}
        <Form {...form}>
          <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-5">
            {/* Email Field */}
            <FormField
              control={form.control}
              name="email"
              rules={{
                required: "Email is required",
                pattern: {
                  value: /^[^\s@]+@[^\s@]+\.[^\s@]+$/,
                  message: "Please enter a valid email",
                },
              }}
              render={({ field }) => (
                <FormItem>
                  <FormLabel className="text-neutral-700 dark:text-neutral-200">Email address (used as username)</FormLabel>
                  <FormControl>
                    <div className="relative">
                      <Mail className="absolute left-4 top-1/2 h-5 w-5 -translate-y-1/2 text-neutral-400" aria-hidden="true" />
                      <Input
                        {...field}
                        type="email"
                        placeholder="your@email.com"
                        className="h-12 pl-12 text-base"
                        disabled={isLoading}
                        aria-label="Email address (used as username)"
                      />
                    </div>
                  </FormControl>
                  {/* Show FastAPI field error if present */}
                  {fieldErrors.email && fieldErrors.email.length > 0 && (
                    <div className="text-red-500 text-xs mt-1">{fieldErrors.email.join(' | ')}</div>
                  )}
                  {/* Only pass string to FormMessage, never object */}
                  {typeof form.formState.errors.email?.message === 'string' ? (
                    <FormMessage />
                  ) : null}
                </FormItem>
              )}
            />

            {/* Password Field */}
            <FormField
              control={form.control}
              name="password"
              rules={{
                required: "Password is required",
                minLength: {
                  value: 6,
                  message: "Password must be at least 6 characters",
                },
              }}
              render={({ field }) => (
                <FormItem>
                  <div className="flex items-center justify-between">
                    <FormLabel className="text-neutral-700 dark:text-neutral-200">Password</FormLabel>
                    <Link
                      href="/auth/forgot-password"
                      className="text-xs text-primary hover:underline dark:text-primary"
                    >
                      Forgot?
                    </Link>
                  </div>
                  <FormControl>
                    <div className="relative">
                      <Lock className="absolute left-4 top-1/2 h-5 w-5 -translate-y-1/2 text-neutral-400" aria-hidden="true" />
                      <Input
                        {...field}
                        type="password"
                        placeholder="Enter your password"
                        className="h-12 pl-12 text-base"
                        disabled={isLoading}
                        aria-label="Password"
                      />
                    </div>
                  </FormControl>
                  {/* Show FastAPI field error if present */}
                  {fieldErrors.password && fieldErrors.password.length > 0 && (
                    <div className="text-red-500 text-xs mt-1">{fieldErrors.password.join(' | ')}</div>
                  )}
                  {/* Only pass string to FormMessage, never object */}
                  {typeof form.formState.errors.password?.message === 'string' ? (
                    <FormMessage />
                  ) : null}
                </FormItem>
              )}
            />


            {/* Non-field (general) errors */}
            {(serverError || error) && (
              <div className="rounded-lg bg-red-50 p-4 text-sm text-red-600 dark:bg-red-900/20 dark:text-red-400 mb-2">
                {serverError || error}
              </div>
            )}

            {/* Sign In Button */}
            <Button
              type="submit"
              variant="secondary"
              size="lg"
              className="h-12 w-full text-base"
              disabled={isLoading}
            >
              {isLoading ? "Signing in..." : "Sign in"}
              {!isLoading && <span className="ml-2">→</span>}
            </Button>
          </form>
        </Form>

        {/* Sign Up Link */}
        <p className="text-center text-sm text-neutral-600 dark:text-neutral-400">
          Don't have an account?{" "}
          <Link href="/auth/signup" className="font-medium text-primary hover:underline">
            Sign up
          </Link>
        </p>
      </div>
    </div>
  );
}

export default LoginForm;
