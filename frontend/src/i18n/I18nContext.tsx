import React, { createContext, useContext, useEffect, useState } from "react";
import { CARDINAL_TRANSLATIONS, ENUM_TRANSLATIONS, TRANSLATIONS, type Locale } from "./translations";

interface I18nContextValue {
  locale: Locale;
  setLocale: (loc: Locale) => void;
  t: (key: string, vars?: Record<string, string | number>) => string;
  tEnum: (value: string | undefined | null) => string;
  tCardinal: (dir: string | undefined | null) => string;
}

const I18nContext = createContext<I18nContextValue | null>(null);

function detectInitialLocale(): Locale {
  try {
    const saved = localStorage.getItem("skyobserver_locale");
    if (saved === "en" || saved === "it") return saved;
    if (typeof navigator !== "undefined" && navigator.language.toLowerCase().startsWith("it")) {
      return "it";
    }
  } catch {
    // ignore
  }
  return "en";
}

export const I18nProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [locale, setLocaleState] = useState<Locale>(detectInitialLocale);

  useEffect(() => {
    document.documentElement.lang = locale;
  }, [locale]);

  const setLocale = (loc: Locale) => {
    setLocaleState(loc);
    try {
      localStorage.setItem("skyobserver_locale", loc);
    } catch {
      // ignore
    }
  };

  const t = (key: string, vars?: Record<string, string | number>): string => {
    let text = TRANSLATIONS[locale][key] ?? TRANSLATIONS.en[key] ?? key;
    if (vars) {
      for (const [k, v] of Object.entries(vars)) {
        text = text.replace(new RegExp(`\\{${k}\\}`, "g"), String(v));
      }
    }
    return text;
  };

  const tEnum = (value: string | undefined | null): string => {
    if (!value) return "";
    return ENUM_TRANSLATIONS[locale][value] ?? ENUM_TRANSLATIONS.en[value] ?? value.replace(/_/g, " ");
  };

  const tCardinal = (dir: string | undefined | null): string => {
    if (!dir) return "";
    return CARDINAL_TRANSLATIONS[locale][dir] ?? dir;
  };

  return (
    <I18nContext.Provider value={{ locale, setLocale, t, tEnum, tCardinal }}>
      {children}
    </I18nContext.Provider>
  );
};

export function useI18n(): I18nContextValue {
  const ctx = useContext(I18nContext);
  if (!ctx) throw new Error("useI18n must be used inside I18nProvider");
  return ctx;
}
