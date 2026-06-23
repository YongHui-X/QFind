import type { SavedChat } from "./types";

const STORAGE_KEY = "clauselens.chats.v1";
const MAX_CHATS = 20;

export function loadChats(): SavedChat[] {
  try {
    const value = JSON.parse(localStorage.getItem(STORAGE_KEY) ?? "[]");
    return Array.isArray(value) ? value.slice(0, MAX_CHATS) : [];
  } catch {
    return [];
  }
}

export function persistChats(chats: SavedChat[]): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(chats.slice(0, MAX_CHATS)));
  } catch {
    // The UI remains usable when storage is unavailable or its quota is full.
  }
}

export function chatTitle(question: string): string {
  const clean = question.replace(/\s+/g, " ").trim();
  return clean.length <= 52 ? clean : `${clean.slice(0, 51).trim()}…`;
}
