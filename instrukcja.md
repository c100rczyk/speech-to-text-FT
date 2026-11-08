# Claude Code w sandboxie nono (Landlock) — instrukcja

Ten zestaw plików zamyka Claude Code w folderze projektu. Egzekwowanie odbywa się w jądrze Linuksa przez **Landlock LSM**: nono buduje białą listę ścieżek, nakłada ją na proces i od tego momentu jądro odrzuca każdą operację spoza listy. Restrykcje są **nieodwracalne** w obrębie sesji i **dziedziczone przez wszystkie procesy potomne** — także przez `bash`, `git`, `npm` czy skrypty, które Claude uruchomi. Nie ma flagi, promptu ani sztuczki, którą agent mógłby to obejść od środka; nawet samo nono nie umie poszerzyć uprawnień po ich nałożeniu.

## Zawartość zestawu

| Plik | Rola |
|---|---|
| `nono-claude.jsonc` | Profil sandboxa (biała lista Landlock + sieć). Tu dostrajasz uprawnienia. |
| `claude-safe.sh` | Skrypt startowy z kontrolami bezpieczeństwa. |
| `claude-settings-nono.json` | Opcjonalny snippet ustawień Claude Code (patrz sekcja niżej). |
| `INSTRUKCJA.md` | Ten plik. |

## Instalacja nono (jednorazowo, na całą maszynę)

Na Ubuntu najprościej przez paczkę `.deb` z oficjalnych wydań:

```bash
VERSION=$(curl -sIL https://github.com/nolabs-ai/nono/releases/latest | grep -i '^location' | tail -1 | grep -oP 'tag/v\K[0-9a-zA-Z.-]+')
ARCH=$(dpkg --print-architecture)
wget "https://github.com/always-further/nono/releases/download/v${VERSION}/nono-cli_${VERSION}_${ARCH}.deb"
sudo dpkg -i "nono-cli_${VERSION}_${ARCH}.deb"
nono --version
```

Alternatywnie: `brew install nono` (jeśli używasz Homebrew na Linuksie). Pełna lista metod: <https://nono.sh/docs/cli/getting_started/installation>

Następnie zweryfikuj wsparcie Landlocka:

```bash
nono setup
```

Wynik powinien pokazać m.in. `Landlock enabled in LSM list` oraz wersję ABI (na Ubuntu 24.04 z jądrem 6.8 to zwykle V4, na nowszych jądrach V5/V6 — im wyżej, tym więcej funkcji, np. filtrowanie TCP od V4). Ubuntu 22.04+ ma Landlock włączony domyślnie. Szybki test ręczny: `cat /sys/kernel/security/lsm` powinien zawierać `landlock`.

## Użycie w projekcie

1. Wrzuć `nono-claude.jsonc`, `claude-safe.sh` i (opcjonalnie) pozostałe pliki do **katalogu głównego projektu**.
2. Nadaj skryptowi prawo wykonania: `chmod +x claude-safe.sh`.
3. Startuj sesje przez: `./claude-safe.sh`

Przy **pierwszym** uruchomieniu nono zapyta o instalację oficjalnego packa `always-further/claude` (to on dostarcza bazowy profil `claude-code`; pochodzenie weryfikowane kryptograficznie przez Sigstore). Zatwierdź Enterem. Logowanie `claude /login` działa normalnie — na Linuksie nono deleguje otwarcie przeglądarki do nienadzorowanego procesu-rodzica, wyłącznie dla zaufanych adresów OAuth.

Argumenty przekazujesz dalej jak zwykle, np. `./claude-safe.sh --continue`.

## Co wolno, czego nie

| Obszar | Dostęp |
|---|---|
| Folder projektu | odczyt + zapis (jedyny obszar roboczy) |
| `~/.claude`, `~/.claude.json`, `~/.local/share/claude` | odczyt + zapis — stan, ustawienia i poświadczenia samego Claude Code; bez tego nie zaloguje się i nie wystartuje |
| Ścieżki systemowe (`/usr`, `/lib`, …), toolchainy z grup w profilu | tylko odczyt (żeby programy w ogóle się uruchamiały) |
| **Cała reszta**: `~/.ssh`, `~/.aws`, `~/.gnupg`, dokumenty, inne projekty… | **brak** — nie ma ich na białej liście, więc jądro odrzuca dostęp; poświadczenia, configi i historia shella są dodatkowo objęte wbudowanymi grupami blokującymi |
| Sieć | wg profilu bazowego (API Anthropic itp.) + domeny, które sam odkomentujesz w `network.allow_domain` |

Jeśli przeszkadza Ci, że stan Claude żyje w `~/.claude` (czyli poza projektem), uruchamiaj przez `./claude-safe.sh --local-config` — wtedy cała konfiguracja i logowanie lądują w `<projekt>/.claude-local` (logujesz się wtedy osobno w każdym projekcie; dopisz `.claude-local/` do `.gitignore`).

## Jak sprawdzić, że sandbox naprawdę działa

Diagnostyka bez uruchamiania sesji:

```bash
nono why --path ~/.ssh/id_rsa --op read          # powinno pokazać: BLOCKED
nono run --profile ./nono-claude.jsonc --allow-cwd --dry-run -- claude   # podgląd uprawnień bez startu
```

Test bojowy: w sesji poproś Claude, żeby przeczytał `~/.ssh/id_rsa` albo skasował coś w `~/Dokumenty`. Operacja zostanie odrzucona przez jądro, a hook z packa podpowie Claude'owi, że to granica sandboxa nono — agent zrelacjonuje Ci to wprost, zamiast kombinować.

## Dostosowanie

Wszystko dzieje się w `nono-claude.jsonc` (składnia JSONC — komentarze i przecinki końcowe są dozwolone):

- **Domeny sieciowe** (git po HTTPS, npm, pip, cargo): odkomentuj wpisy w `network.allow_domain`. Doraźnie, na jedną sesję: `nono run --profile ./nono-claude.jsonc --allow-cwd --allow-domain github.com -- claude`.
- **Sesja offline** (przegląd kodu bez API): `nono run --profile ./nono-claude.jsonc --allow-cwd --block-net -- claude`.
- **Toolchainy spoza projektu** (cargo, go, pyenv…): odkomentuj ścieżki w `filesystem.read`. To tylko odczyt — wystarcza do wyszukiwania binarek w `PATH`.
- **Sekrety wewnątrz projektu** (`.env` itp.): odkomentuj wpisy w `filesystem.deny` razem z sekcją `security.capability_elevation` na końcu pliku. Uwaga: czysty Landlock nie umie wyciąć dziecka dozwolonego katalogu, dlatego ta blokada wymaga trybu nadzorowanego (seccomp-notify), który nono włącza właśnie tą opcją.

Po każdej zmianie warto puścić `nono profile validate ./nono-claude.jsonc`, a wynik scalania z bazą obejrzeć przez `nono profile show ./nono-claude.jsonc`.

## Ważne: wbudowany sandbox Claude Code

Claude Code ma własne (opcjonalne) sandboxowanie, ale z udokumentowaną furtką: po odmowie może ponowić komendę poza sandboxem (`dangerouslyDisableSandbox`). Skoro warstwą egzekwującą jest nono, **zostaw wbudowany sandbox wyłączony** (stan domyślny). Jeżeli świadomie go używasz, dodaj w ustawieniach Claude Code `"allowUnsandboxedCommands": false`, żeby zamknąć furtkę. Szczegóły: <https://code.claude.com/docs/en/sandboxing>

Plik `claude-settings-nono.json` zawiera natomiast coś innego: wstępną zgodę na **diagnostyczne, wyłącznie odczytowe** komendy `nono why` / `nono profile …`, których Claude używa po odmowie sandboxa, żeby wyjaśnić Ci, co zostało zablokowane. Skopiuj go jako `.claude/settings.json` w projekcie (a jeśli już masz ten plik — dołącz wpisy z sekcji `permissions.allow`).

## Rozwiązywanie problemów

**„Landlock is not available" / skrypt zgłasza brak Landlocka.** Jądro nie ma Landlocka na liście LSM. Sprawdź `cat /sys/kernel/security/lsm`; jeśli brakuje `landlock`, dodaj do parametrów rozruchu `lsm=landlock,lockdown,yama,integrity,apparmor` (w `/etc/default/grub`, potem `sudo update-grub` i restart). Na standardowym Ubuntu 22.04+ to nie występuje. Uwaga: wewnątrz kontenerów (Docker/LXC) Landlock bywa blokowany przez seccomp — sandbox uruchamiaj na hoście.

**`claude` nie startuje w sandboxie.** Najczęściej binarka leży w katalogu, którego nie ma na białej liście. Profil obejmuje już `~/.local/bin` (instalacja natywna) i ścieżki node/nvm/npm; dla innych menedżerów wersji dodaj odpowiedni katalog do `filesystem.read`. Podejrzyj swój `PATH`: `echo "$PATH" | tr ':' '\n'` i dosyp brakujące pozycje.

**Narzędzie w projekcie wymaga `/run`, `/var/run` albo `/sys`.** Profil `default` celowo ich nie obejmuje. Zmień pierwszą linię profilu na `"extends": ["claude-code", "linux-host-compat"]`.

**Coś innego jest blokowane i nie wiesz co.** `nono why --path <ścieżka> --op read` odpowiada wprost, która reguła zablokowała dostęp. Po zakończeniu sesji nono wypisuje też stopkę diagnostyczną z listą odmów i propozycją zapisania wyjątków do profilu — nic nie jest dopisywane bez Twojej zgody.

## Granice rozwiązania (uczciwie)

Landlock izoluje system plików i (od ABI v4) sieć TCP, ale proces nadal współdzieli jądro z resztą systemu — to ochrona przed „agent poszedł tam, gdzie nie powinien", a nie przed exploitem jądra; przy pracy z celowo złośliwym kodem dołóż VM/kontener jako drugą warstwę. Poza tym pamiętaj, że wszystko, co **jest** w folderze projektu, pozostaje dla agenta w pełni dostępne — sekrety trzymaj poza projektem albo użyj `filesystem.deny` + `capability_elevation` jak wyżej. Sam projekt nono jest młody i aktywnie rozwijany (audyt bezpieczeństwa w toku), więc aktualizuj go regularnie: `nono update`.