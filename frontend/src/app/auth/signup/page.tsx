"use client";

import { AuthLayout } from "@/components/auth";
import { SignupForm } from "@/components/auth";

import { AuthRedirect } from "@/components/auth/AuthRedirect";
// ...existing imports...

export default function SignupPage() {
  return (
    <AuthRedirect>
      <AuthLayout>
        <SignupForm />
      </AuthLayout>
    </AuthRedirect>
  );
}
