"use client";

import { AuthLayout } from "@/components/auth";
import { LoginForm } from "@/components/auth";

import { AuthRedirect } from "@/components/auth/AuthRedirect";
// ...existing imports...

export default function LoginPage() {
  return (
    <AuthRedirect>
      <AuthLayout>
        <LoginForm />
      </AuthLayout>
    </AuthRedirect>
  );
}
