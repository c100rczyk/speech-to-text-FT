#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────
#  claude-safe.sh — uruchamia Claude Code w sandboxie nono (Landlock)
#
#  Sandbox obejmuje WYŁĄCZNIE folder, w którym leży ten skrypt, czyli
#  folder projektu. Wszystkie procesy potomne (bash, git, npm...) dziedziczą
#  ograniczenia — to egzekwuje jądro Linuksa, nie da się tego cofnąć
#  z wnętrza sesji.
#
#  Użycie:
#    ./claude-safe.sh                  zwykła sesja Claude Code
#    ./claude-safe.sh --local-config   stan Claude (łącznie z logowaniem)
#                                      trzymany w projekcie, nie w ~/.claude
#    ./claude-safe.sh <argumenty>      reszta argumentów leci do "claude"
# ─────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
PROFILE="$SCRIPT_DIR/nono-claude.jsonc"

# 1. Czy nono jest zainstalowane?
if ! command -v nono >/dev/null 2>&1; then
  echo "BŁĄD: brak polecenia 'nono'." >&2
  echo "Instalacja na Ubuntu — paczka .deb z GitHub Releases:" >&2
  echo "  https://nono.sh/docs/cli/getting_started/installation" >&2
  echo "(kroki są też w INSTRUKCJA.md, sekcja 'Instalacja')" >&2
  exit 1
fi

# 2. Czy profil leży obok skryptu?
if [[ ! -f "$PROFILE" ]]; then
  echo "BŁĄD: nie znaleziono profilu: $PROFILE" >&2
  echo "Plik nono-claude.jsonc musi leżeć w tym samym folderze co ten skrypt." >&2
  exit 1
fi

# 3. Czy Landlock jest aktywny w jądrze? (bez niego sandbox nie wystartuje)
if [[ -r /sys/kernel/security/lsm ]] && ! grep -q landlock /sys/kernel/security/lsm; then
  echo "BŁĄD: Landlock nie jest aktywny w jądrze." >&2
  echo "Sprawdź: cat /sys/kernel/security/lsm  (powinno zawierać 'landlock')." >&2
  echo "Naprawa opisana w INSTRUKCJA.md → 'Rozwiązywanie problemów'." >&2
  exit 1
fi

# 4. Bezpiecznik: nie pozwól przypadkiem objąć sandboxem całego HOME ani /.
case "$SCRIPT_DIR" in
  "$HOME"|"/")
    echo "BŁĄD: skrypt leży w '$SCRIPT_DIR'." >&2
    echo "Wrzuć te pliki do folderu KONKRETNEGO projektu i uruchom stamtąd." >&2
    exit 1
    ;;
esac

# 5. Opcjonalnie: cały stan Claude w projekcie (osobne logowanie per projekt).
#    Katalog i jego zamek (.lock) są wewnątrz projektu, więc mieszczą się
#    w istniejących uprawnieniach sandboxa — zero dodatkowych wyjątków.
if [[ "${1:-}" == "--local-config" ]]; then
  shift
  export CLAUDE_CONFIG_DIR="$SCRIPT_DIR/.claude-local"
  mkdir -p "$CLAUDE_CONFIG_DIR"
  echo "· Stan Claude w projekcie: $CLAUDE_CONFIG_DIR"
fi

cd -- "$SCRIPT_DIR"
echo "· Sandbox nono/Landlock: zapis możliwy tylko w $SCRIPT_DIR"
# --allow-cwd jawnie dodaje folder projektu do białej listy (r+w zgodnie
# z "workdir" w profilu). Bez tej flagi nono potrafi pominąć CWD w trybie
# nieinteraktywnym — sprawdzone na v0.67.1.
exec nono run --profile "$PROFILE" --allow-cwd -- claude "$@"