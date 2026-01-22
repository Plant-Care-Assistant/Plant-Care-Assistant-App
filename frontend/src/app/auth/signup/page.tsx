"use client";

import { AuthLayout } from "@/components/auth";
import { SignupForm } from "@/components/auth";

export default function SignupPage() {
  return (
    <AuthLayout>
      <SignupForm />
    </AuthLayout>
  );
}
