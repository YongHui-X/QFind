import type { ChatMessage } from "./types";

const CLAUSE_TYPE_TERMS: Record<string, string[]> = {
  "Anti-Assignment": [
    "anti assignment",
    "assign",
    "assignment",
    "transfer the agreement",
    "transfer this agreement",
    "transferred to another party",
  ],
  "Cap On Liability": [
    "cap on liability",
    "liability cap",
    "limit liability",
    "limitation of liability",
    "damages",
    "consequential loss",
    "consequential damages",
    "indirect damages",
    "lost profits",
    "categories of loss",
    "excluded losses",
    "anticipated savings",
    "prospective profits",
    "special damages",
    "punitive damages",
  ],
  "License Grant": [
    "license",
    "licence",
    "licensed materials",
    "usage rights",
    "right to use",
    "rights to use",
    "use intellectual property",
  ],
  "Audit Rights": ["audit", "inspect records", "inspect books", "books and records"],
  "Termination For Convenience": [
    "termination for convenience",
    "terminate for convenience",
    "terminate without cause",
    "termination without cause",
    "walk away",
  ],
};

function normalized(value: string): string {
  return value.toLowerCase().replaceAll("-", " ").replace(/\s+/g, " ").trim();
}

export function inferClauseType(query: string): string | null {
  const clean = normalized(query);
  const matches: Array<[number, string]> = [];
  for (const [clauseType, terms] of Object.entries(CLAUSE_TYPE_TERMS)) {
    const score = terms.reduce(
      (total, term) => total + (clean.includes(term) ? term.split(" ").length : 0),
      0,
    );
    if (score) matches.push([score, clauseType]);
  }
  if (
    clean.includes("intellectual property") &&
    ["use", "right", "rights", "grant", "granted", "license", "licence"].some((term) =>
      clean.includes(term),
    )
  ) {
    matches.push([4, "License Grant"]);
  }
  matches.sort((a, b) => b[0] - a[0]);
  return matches[0]?.[1] ?? null;
}

export function isReferentialFollowUp(query: string): boolean {
  const clean = normalized(query).replace(/^["']|["']$/g, "");
  const words = clean.split(" ");
  const prefixes = [
    "what about",
    "how about",
    "how long",
    "how much",
    "how often",
    "is it",
    "does it",
    "can it",
    "are they",
    "do they",
    "can they",
    "is that",
    "does that",
    "also",
  ];
  const referenceTokens = new Set(["it", "its", "they", "them", "that", "this", "also"]);
  return (
    ((words.length <= 8 || prefixes.some((prefix) => clean.startsWith(prefix))) &&
      words.some((word) => referenceTokens.has(word))) ||
    ["how long", "how much", "how often", "what about"].some((prefix) =>
      clean.startsWith(prefix),
    )
  );
}

export function resolveConversationClauseType(
  messages: ChatMessage[],
  requested?: string | null,
): string | null {
  if (requested) return requested;
  const latest = messages.at(-1)?.content ?? "";
  const latestType = inferClauseType(latest);
  if (latestType && !isReferentialFollowUp(latest)) return latestType;
  for (const message of messages.slice(0, -1).reverse()) {
    const prior = inferClauseType(message.content);
    if (prior) return prior;
  }
  return latestType;
}

export function contextualizeQuery(messages: ChatMessage[], clauseType: string): string {
  const latest = messages.at(-1)?.content.trim() ?? "";
  const latestType = inferClauseType(latest);
  if (latestType === clauseType && !isReferentialFollowUp(latest)) return latest;
  return normalized(latest).includes(normalized(clauseType))
    ? latest
    : `${latest} ${clauseType}`;
}

export function chooseQueryReranking(query: string, clauseType: string): boolean {
  const clean = normalized(query);
  const ipParaphrase =
    clauseType === "License Grant" &&
    clean.includes("intellectual property") &&
    ["use", "usage", "right", "rights", "grant", "granted", "provision"].some((term) =>
      clean.includes(term),
    ) &&
    !clean.includes("license") &&
    !clean.includes("licence");
  const detailTerms = [
    "affiliate",
    "anniversary",
    "cost",
    "consequence",
    "days",
    "duration",
    "effective",
    "exception",
    "how much",
    "how often",
    "operation of law",
    "percent",
    "perpetual",
    "prior notice",
    "subsidiary",
    "territory",
    "threshold",
    "void",
    "what happens",
    "wholly owned",
    "written notice",
  ];
  return ipParaphrase || detailTerms.some((term) => clean.includes(term));
}
