import { createClient } from "@supabase/supabase-js";
import Constants from "expo-constants";
import { Platform } from "react-native";
import type {
  AuthFlowType,
  SupabaseClientOptions,
} from "@supabase/supabase-js";

type AppExtra = {
  SUPABASE_URL: string;
  SUPABASE_ANON_KEY: string;
};

const extra = (Constants.expoConfig?.extra ?? {}) as AppExtra;
const { SUPABASE_URL, SUPABASE_ANON_KEY } = extra;

if (!SUPABASE_URL || !SUPABASE_ANON_KEY) {
  throw new Error(
    "Missing SUPABASE_URL or SUPABASE_ANON_KEY in Constants.expoConfig.extra"
  );
}

let options: SupabaseClientOptions<"public"> = {
  auth: {
    persistSession: true,
    autoRefreshToken: true,
    detectSessionInUrl: false,
    flowType: "pkce" as AuthFlowType,
    debug: false,
  },
};

if (Platform.OS !== "web") {
  const AsyncStorage =
    require("@react-native-async-storage/async-storage").default;
  options.auth = {
    ...options.auth,
    storage: AsyncStorage,
  } as any;
}

export const supabase = createClient(SUPABASE_URL, SUPABASE_ANON_KEY, options);
