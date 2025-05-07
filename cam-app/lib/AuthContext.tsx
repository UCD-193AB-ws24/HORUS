import {
  createContext,
  useContext,
  useEffect,
  useState,
  ReactNode,
} from "react";
import { supabase } from "@/lib/supabaseClient";
import type { Session, User } from "@supabase/supabase-js";
import { saveUserProgress, getUserProgress } from "@/lib/databaseService";

type AuthContextType = {
  user: User | null;
  session: Session | null;
  signIn: (email: string, password: string) => Promise<{ error: Error | null }>;
  signUp: (email: string, password: string) => Promise<{ error: Error | null }>;
  signOut: () => Promise<{ error: Error | null }>;
  saveProgress: (letterIndex: number) => Promise<{ error: Error | null }>;
  getProgress: () => Promise<{ letterIndex: number; error: Error | null }>;
};

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [session, setSession] = useState<Session | null>(null);
  const [user, setUser] = useState<User | null>(null);

  useEffect(() => {
    supabase.auth.getSession().then(({ data: { session } }) => {
      setSession(session);
      setUser(session?.user ?? null);
    });

    const {
      data: { subscription },
    } = supabase.auth.onAuthStateChange((_event, session) => {
      setSession(session);
      setUser(session?.user ?? null);
    });

    return () => subscription.unsubscribe();
  }, []);

  const signIn = (email: string, password: string) =>
    supabase.auth.signInWithPassword({ email, password });
  const signUp = (email: string, password: string) =>
    supabase.auth.signUp({ email, password });
  const signOut = () => supabase.auth.signOut();

  const saveProgress = async (letterIndex: number) => {
    if (!user) return { error: new Error("User not authenticated") };
    return saveUserProgress(user.id, letterIndex);
  };

  const getProgress = async () => {
    if (!user)
      return { letterIndex: 0, error: new Error("User not authenticated") };
    return getUserProgress(user.id);
  };

  return (
    <AuthContext.Provider
      value={{
        user,
        session,
        signIn,
        signUp,
        signOut,
        saveProgress,
        getProgress,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used inside AuthProvider");
  return ctx;
}
