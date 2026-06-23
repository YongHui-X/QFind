declare global {
  interface Window {
    turnstile?: {
      render(
        element: HTMLElement,
        options: {
          sitekey: string;
          action: string;
          size: "invisible";
          callback(token: string): void;
          "error-callback"(): void;
          "expired-callback"(): void;
        },
      ): string;
      execute(widgetId: string): void;
      reset(widgetId: string): void;
    };
  }
}

let widgetId: string | null = null;
let resolver: ((token: string) => void) | null = null;
let rejecter: ((error: Error) => void) | null = null;

export function initializeTurnstile(element: HTMLElement, siteKey: string): void {
  const attempt = () => {
    if (!window.turnstile) {
      setTimeout(attempt, 100);
      return;
    }
    widgetId = window.turnstile.render(element, {
      sitekey: siteKey,
      action: "chat",
      size: "invisible",
      callback(token) {
        resolver?.(token);
        resolver = null;
        rejecter = null;
      },
      "error-callback"() {
        rejecter?.(new Error("Browser verification failed"));
        resolver = null;
        rejecter = null;
      },
      "expired-callback"() {
        rejecter?.(new Error("Browser verification expired"));
        resolver = null;
        rejecter = null;
      },
    });
  };
  attempt();
}

export function turnstileToken(): Promise<string> {
  return new Promise((resolve, reject) => {
    if (!widgetId || !window.turnstile) {
      reject(new Error("Browser verification is still loading"));
      return;
    }
    resolver = resolve;
    rejecter = reject;
    window.turnstile.reset(widgetId);
    window.turnstile.execute(widgetId);
  });
}
