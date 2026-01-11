"use client";

import { useQuery } from "@tanstack/react-query";
import { userApi } from "@/lib/api";
import { UserData } from "@/types";

const USER_DATA_KEY = ["user", "stats"];

/** Fetch user gamification/profile data. */
export function useUserDataQuery(enabled = true) {
  return useQuery<UserData>({
    queryKey: USER_DATA_KEY,
    queryFn: () => userApi.getUserData(),
    enabled,
  });
}
