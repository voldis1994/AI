export type Personality = "assistant" | "friend" | "teacher" | "creative";

export interface UserProfile {
  name: string;
  assistantName: string;
  personality: Personality;
  language: "lv" | "en";
  interests: string[];
  about: string;
}

export interface Memory {
  id: string;
  content: string;
  createdAt: string;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  createdAt: string;
}

export interface ChatSession {
  id: string;
  title: string;
  messages: ChatMessage[];
  createdAt: string;
  updatedAt: string;
}

export const DEFAULT_PROFILE: UserProfile = {
  name: "",
  assistantName: "Mans AI",
  personality: "assistant",
  language: "lv",
  interests: [],
  about: "",
};

export const PERSONALITY_LABELS: Record<
  Personality,
  { lv: string; en: string; description: { lv: string; en: string } }
> = {
  assistant: {
    lv: "Asistents",
    en: "Assistant",
    description: {
      lv: "Profesionāls un efektīvs palīgs ikdienas uzdevumiem",
      en: "Professional and efficient helper for daily tasks",
    },
  },
  friend: {
    lv: "Draugs",
    en: "Friend",
    description: {
      lv: "Siltš un draudzīgs sarunu biedrs",
      en: "Warm and friendly conversation partner",
    },
  },
  teacher: {
    lv: "Skolotājs",
    en: "Teacher",
    description: {
      lv: "Pacietīgs skaidrotājs, kas māca soli pa solim",
      en: "Patient explainer who teaches step by step",
    },
  },
  creative: {
    lv: "Radošais",
    en: "Creative",
    description: {
      lv: "Iedomīgs partneris idejām un radošiem projektiem",
      en: "Imaginative partner for ideas and creative projects",
    },
  },
};

export function buildSystemPrompt(profile: UserProfile, memories: Memory[]): string {
  const personalityMap: Record<Personality, string> = {
    assistant: "Tu esi profesionāls un efektīvs personīgais asistents. Atbildi skaidri, strukturēti un konkrēti.",
    friend: "Tu esi siltš, draudzīgs un empātisks sarunu biedrs. Esi dabīgs un neformāls, bet joprojām noderīgs.",
    teacher: "Tu esi pacietīgs skolotājs. Skaidro jēdzienu soli pa solim, izmanto piemērus un uzdod jautājumus, lai pārbaudītu izpratni.",
    creative: "Tu esi radošs un iedomīgs partneris. Piedāvā neparastas idejas, metaforas un radošus risinājumus.",
  };

  const langInstruction =
    profile.language === "lv"
      ? "Atbildi latviešu valodā, ja vien lietotājs neprasa citā valodā."
      : "Respond in English unless the user asks for another language.";

  const parts = [
    `Tu esi ${profile.assistantName} — personīgais AI asistents.`,
    personalityMap[profile.personality],
    langInstruction,
  ];

  if (profile.name) {
    parts.push(`Lietotāja vārds ir ${profile.name}.`);
  }

  if (profile.about) {
    parts.push(`Par lietotāju: ${profile.about}`);
  }

  if (profile.interests.length > 0) {
    parts.push(`Lietotāja intereses: ${profile.interests.join(", ")}.`);
  }

  if (memories.length > 0) {
    const memoryList = memories.map((m) => `- ${m.content}`).join("\n");
    parts.push(`Atceries šīs svarīgās lietas par lietotāju:\n${memoryList}`);
  }

  parts.push(
    "Esi personīgs, atceries kontekstu un palīdzi lietotājam sasniegt viņa mērķus. Ja nezini atbildi, esi godīgs par to."
  );

  return parts.join("\n\n");
}

export function generateId(): string {
  return `${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;
}
