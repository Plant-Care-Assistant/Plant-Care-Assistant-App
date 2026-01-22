import { type NextRequest, NextResponse } from "next/server";

/**
 * Middleware to protect routes.
 * Redirects unauthenticated users to /auth/login
 */
export function middleware(request: NextRequest) {
  const token = request.cookies.get("auth_token");
  const pathname = request.nextUrl.pathname;

  // Public routes that don't require auth
  const publicRoutes = ["/auth/login", "/auth/signup", "/auth/forgot-password"];
  const isPublic = publicRoutes.some((route) => pathname.startsWith(route));

  // Root path should be guarded
  const isProtected = !isPublic && pathname !== "/";

  // If no token and trying to access protected route, redirect to login
  if (!token && isProtected) {
    return NextResponse.redirect(new URL("/auth/login", request.url));
  }

  // If has token and trying to access auth routes, redirect to home
  if (token && isPublic) {
    return NextResponse.redirect(new URL("/", request.url));
  }

  return NextResponse.next();
}

// Configure which routes this middleware applies to
export const config = {
  matcher: [
    /*
     * Match all request paths except for the ones starting with:
     * - api (API routes)
     * - _next/static (static files)
     * - _next/image (image optimization files)
     * - favicon.ico (favicon file)
     * - public folder
     */
    "/((?!api|_next/static|_next/image|favicon.ico|.*\\.png|.*\\.svg).*)",
  ],
};
