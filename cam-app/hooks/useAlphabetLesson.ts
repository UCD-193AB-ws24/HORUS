const LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".split("") as readonly string[];

export default function useAlphabetLesson() {
  return { letters: LETTERS };
}
