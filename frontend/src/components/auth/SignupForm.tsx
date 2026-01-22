"use client";

import { useState } from "react";
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
import { Mail, Lock, User } from "lucide-react";

interface SignupFormData {
  username: string;
  email: string;
  password: string;
  confirmPassword: string;
}

export function SignupForm() {
  const router = useRouter();
  const { register, isLoading } = useAuth();
  const [serverError, setServerError] = useState<string | null>(null);
  const form = useForm<SignupFormData>({
    defaultValues: { username: "", email: "", password: "", confirmPassword: "" },
  });

  const onSubmit = async (data: SignupFormData) => {
    setServerError(null);

    if (data.password !== data.confirmPassword) {
      setServerError("Passwords do not match");
      return;
    }

    try {
      await register({
        username: data.username,
        email: data.email,
        password: data.password,
      });
      router.push("/");
    } catch (error) {
      setServerError(
        error instanceof Error ? error.message : "Signup failed. Please try again.",
      );
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
                className="object-contain drop-shadow-lg"
                priority
              />
            </div>
          </div>
          <div className="text-center">
            <h1 className="text-2xl font-bold text-neutral-900 dark:text-white">Create account</h1>
            <p className="mt-2 text-sm text-neutral-600 dark:text-neutral-400">
              Join us and start caring for your plants
            </p>
          </div>
        </div>

        {/* Signup Form */}
        <Form {...form}>
          <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-5">
            {/* Full Name Field */}
            <FormField
              control={form.control}
              name="username"
              rules={{
                required: "Full name is required",
                minLength: { value: 3, message: "Name must be at least 3 characters" },
              }}
              render={({ field }) => (
                <FormItem>
                  <FormLabel className="text-neutral-700 dark:text-neutral-200">Full name</FormLabel>
                  <FormControl>
                    <div className="relative">
                      <User className="absolute left-4 top-1/2 h-5 w-5 -translate-y-1/2 text-neutral-400" aria-hidden="true" />
                      <Input
                        {...field}
                        placeholder="John Doe"
                        className="h-12 pl-12 text-base"
                        disabled={isLoading}
                        aria-label="Full name"
                      />
                    </div>
                  </FormControl>
                  <FormMessage />
                </FormItem>
              )}
            />

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
                  <FormLabel className="text-neutral-700 dark:text-neutral-200">Email address</FormLabel>
                  <FormControl>
                    <div className="relative">
                      <Mail className="absolute left-4 top-1/2 h-5 w-5 -translate-y-1/2 text-neutral-400" aria-hidden="true" />
                      <Input
                        {...field}
                        type="email"
                        placeholder="your@email.com"
                        className="h-12 pl-12 text-base"
                        disabled={isLoading}
                        aria-label="Email address"
                      />
                    </div>
                  </FormControl>
                  <FormMessage />
                </FormItem>
              )}
            />

            {/* Password Field */}
            <FormField
              control={form.control}
              name="password"
              rules={{
                required: "Password is required",
                minLength: { value: 6, message: "Password must be at least 6 characters" },
              }}
              render={({ field }) => (
                <FormItem>
                  <FormLabel className="text-neutral-700 dark:text-neutral-200">Password</FormLabel>
                  <FormControl>
                    <div className="relative">
                      <Lock className="absolute left-4 top-1/2 h-5 w-5 -translate-y-1/2 text-neutral-400" aria-hidden="true" />
                      <Input
                        {...field}
                        type="password"
                        placeholder="Create a password"
                        className="h-12 pl-12 text-base"
                        disabled={isLoading}
                        aria-label="Password"
                      />
                    </div>
                  </FormControl>
                  <FormMessage />
                </FormItem>
              )}
            />

            {/* Confirm Password Field */}
            <FormField
              control={form.control}
              name="confirmPassword"
              rules={{
                required: "Please confirm your password",
              }}
              render={({ field }) => (
                <FormItem>
                  <FormLabel className="text-neutral-700 dark:text-neutral-200">Confirm password</FormLabel>
                  <FormControl>
                    <div className="relative">
                      <Lock className="absolute left-4 top-1/2 h-5 w-5 -translate-y-1/2 text-neutral-400" aria-hidden="true" />
                      <Input
                        {...field}
                        type="password"
                        placeholder="Confirm your password"
                        className="h-12 pl-12 text-base"
                        disabled={isLoading}
                        aria-label="Confirm password"
                      />
                    </div>
                  </FormControl>
                  <FormMessage />
                </FormItem>
              )}
            />

            {/* Server Error */}
            {serverError && (
              <p className="rounded-lg bg-red-50 p-4 text-sm text-red-600 dark:bg-red-900/20 dark:text-red-400">
                {serverError}
              </p>
            )}

            {/* Sign Up Button */}
            <Button
              type="submit"
              variant="primary"
              size="lg"
              className="h-12 w-full text-base"
              disabled={isLoading}
            >
              {isLoading ? "Creating account..." : "Create account"}
              {!isLoading && <span className="ml-2">→</span>}
            </Button>
          </form>
        </Form>

        {/* Login Link */}
        <p className="text-center text-sm text-neutral-600 dark:text-neutral-400">
          Already have an account?{" "}
          <Link href="/auth/login" className="font-medium text-primary hover:underline">
            Sign in
          </Link>
        </p>
      </div>
    </div>
  );
}

export default SignupForm;
