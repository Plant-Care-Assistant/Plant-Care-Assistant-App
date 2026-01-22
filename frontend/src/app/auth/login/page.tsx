"use client";

import { AuthLayout } from "@/components/auth";
import { LoginForm } from "@/components/auth";

export default function LoginPage() {
  return (
    <AuthLayout>
      <LoginForm />
    </AuthLayout>
  );
}
