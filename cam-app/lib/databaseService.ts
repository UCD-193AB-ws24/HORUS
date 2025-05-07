import { supabase } from "@/lib/supabaseClient";

/**
 * Saves the user's current progress in learning ASL
 * @param userId
 * @param letterIndex
 * @returns
 */
export async function saveUserProgress(
  userId: string,
  letterIndex: number
): Promise<{ error: Error | null }> {
  try {
    const { error } = await supabase
      .from("user_progress")
      .upsert(
        { user_id: userId, current_letter_index: letterIndex },
        { onConflict: "user_id" }
      );

    return { error: error as Error | null };
  } catch (error) {
    return { error: error as Error };
  }
}

/**
 * Gets the user's current progress in learning ASL
 * @param userId
 * @returns
 */
export async function getUserProgress(
  userId: string
): Promise<{ letterIndex: number; error: Error | null }> {
  try {
    const { data, error } = await supabase
      .from("user_progress")
      .select("current_letter_index")
      .eq("user_id", userId)
      .single();

    if (error) {
      return { letterIndex: 0, error: error as Error };
    }

    return {
      letterIndex: data?.current_letter_index || 0,
      error: null,
    };
  } catch (error) {
    return { letterIndex: 0, error: error as Error };
  }
}
