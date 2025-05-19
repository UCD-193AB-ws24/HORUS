import "dotenv/config";

export default ({ config }) => ({
  ...config,
  extra: {
    SUPABASE_URL: process.env.SUPABASE_URL,
    SUPABASE_ANON_KEY: process.env.SUPABASE_ANON_KEY,
  },
  expo: {
    ...config.expo,
    extra: {
      ...config.expo?.extra, 
      apiUrl: process.env.EXPO_PUBLIC_API_URL,
      eas: {
        projectId: "50db876c-6aa5-4b1f-8d55-e9c24f2270f2",
        EAS_SKIP_AUTO_FINGERPRINT: "1"
      }
    },
    ios: {
      bundleIdentifier: "com.ecs193.signlanguageapp",
      infoPlist: {
        ITSAppUsesNonExemptEncryption: "false"
      }
    }
  }
});
