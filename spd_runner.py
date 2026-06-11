#!/usr/bin/env python3
"""
SPD Seed Finder Automation
Usage: python3 spd_runner.py <pattern> [<pattern> ...] <required_count> [char_class]
                             [--floors N [N ...]] [--preserve-existing]
  pattern: regex(es) matched case-insensitively against seed-finder output;
           a seed counts only if EVERY pattern matches at least one line.
           %tierN (N = 1-5) expands to that tier's weapon names, e.g.
           '%tier4 \\+3' matches longsword/battle axe/.../katana at +3
  required_count: how many matching saves to end up with
  char_class: warrior|mage|rogue|huntress|duelist|cleric (default: duelist)
  --floors: max floor per pattern (one value for all patterns, or one per
            pattern; default 7)
  --preserve-existing: never delete saves present at startup; work in the
            remaining slots (target is capped to the free slot count)

Examples:
  # 1 save with 'scroll of upgrade' somewhere in the first 7 floors
  python3 spd_runner.py 'scroll of upgrade' 1

  # 2 warrior saves with any tier-4 weapon at exactly +3
  python3 spd_runner.py '%tier4 \\+3' 2 warrior

  # BOTH must hit: shortsword +3 by floor 4 AND greatshield +3 by floor 7
  python3 spd_runner.py 'shortsword \\+3' 'greatshield \\+3' 1 --floors 4 7

  # fill free slots only, never touching saves already on the device
  python3 spd_runner.py '%tier5 \\+2' 4 --preserve-existing

  # enchanted weapons print as 'longsword of force +3' — use .* before the +N
  python3 spd_runner.py '%tier4.*\\+3' 1
"""

import argparse, subprocess, time, re
from functools import lru_cache

import numpy as np
from PIL import Image

SEED_FINDER = "/home/tris/pequod/tris/tools/seed-finder-3/seed-finder-3.3.0/seed-finder.jar"
FLOORS      = 7

# Row y-centers by total items on screen (saves + NGB button if saves < 6)
ROW_CENTERS = {
    6: [540, 644, 748, 852, 956, 1060],
    5: [584, 692, 800, 908, 1016],
    4: [636, 744, 852, 960],
    3: [684, 800, 916],
    2: [740, 860],
    1: [800],
}
CHAR_X = dict(warrior=60, mage=180, rogue=300, huntress=420, duelist=540, cleric=660)

# ── Template OCR ──────────────────────────────────────────────────────────────

_FONT_PATH  = "/home/tris/pequod/tris/tools/seed-finder-3/pixel_font.png"
# Characters in sprite order (not ASCII order); each occupies 8×8 px at index*8
_FONT_CHARS = '!"#$%\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{:}`~'
_FONT_W     = 8
_FONT_H     = 8

# Empirical ROI for the seed value in a 720×1560 card screenshot
_SEED_Y0 = 869   # top pixel row of glyphs
_SEED_X0 = 392   # left edge of first character of value


@lru_cache(maxsize=1)
def _seed_templates() -> dict:
    """Binary 8×8 templates for A–Z and hyphen, built once from the font sprite."""
    font = (np.array(Image.open(_FONT_PATH).convert('L')) > 128).astype(np.uint8)
    return {
        ch: font[:, i * _FONT_W:(i + 1) * _FONT_W]
        for i, ch in enumerate(_FONT_CHARS)
        if (ch.isupper() and ch.isalpha()) or ch == '-'
    }


def _best_char(window: np.ndarray, templates: dict) -> tuple:
    """Return (char, score) with highest pixel-agreement against templates."""
    win_bin = (window > 128).astype(np.uint8)
    best_ch, best_sc = '?', -1.0
    for ch, tmpl in templates.items():
        sc = float(np.sum(win_bin == tmpl)) / tmpl.size
        if sc > best_sc:
            best_sc, best_ch = sc, ch
    return best_ch, best_sc


def _detect_stride(gray: np.ndarray, templates: dict) -> int:
    """
    Detect character stride by matching the hyphen at its known positions
    (3 and 7) in the XXX-XXX-XXX pattern across stride candidates 8–20.
    """
    hyphen = templates['-']
    best_stride, best_score = 8, -1.0
    for stride in range(8, 21):
        score = 0.0
        for pos in (3, 7):
            x = _SEED_X0 + pos * stride
            if x + _FONT_W > gray.shape[1]:
                break
            win = gray[_SEED_Y0:_SEED_Y0 + _FONT_H, x:x + _FONT_W]
            score += float(np.sum((win > 128).astype(np.uint8) == hyphen)) / hyphen.size
        if score > best_score:
            best_score, best_stride = score, stride
    return best_stride


def _read_seed_from_card_template(png_path: str) -> str | None:
    """
    Read XXX-XXX-XXX seed via pixel template matching against the game font.
    Returns None on failure; use _read_seed_from_card (Tesseract) as backup.
    """
    templates = _seed_templates()
    gray      = np.array(Image.open(png_path).convert('L'))
    stride    = _detect_stride(gray, templates)

    chars = []
    for i in range(11):
        if i in (3, 7):
            chars.append('-')
            continue
        x   = _SEED_X0 + i * stride
        win = gray[_SEED_Y0:_SEED_Y0 + _FONT_H, x:x + _FONT_W]
        ch, _ = _best_char(win, templates)
        chars.append(ch)

    result = ''.join(chars)
    m = re.search(r'[A-Z?]{3}-[A-Z?]{3}-[A-Z?]{3}', result)
    return m.group(0) if m else None


def _parse_seed_from_text(text):
    m = re.search(r'[A-Z]{3}-[A-Z]{3}-[A-Z]{3}', text)
    return m.group(0) if m else None


def _read_seed_from_card(png_path):
    """Run tesseract on a saved screenshot and extract XXX-XXX-XXX seed."""
    out = subprocess.run(
        ["tesseract", png_path, "stdout",
         "-c", "tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ- "],
        capture_output=True, text=True
    ).stdout
    return _parse_seed_from_text(out)


def _row_activity_score(gray, y, x0=120, x1=620, height=28) -> float:
    """Return a simple activity score for a row-like region."""
    half = height // 2
    top = max(0, y - half)
    bot = min(gray.shape[0], y + half)
    patch = gray[top:bot, x0:x1]
    if patch.size == 0:
        return 0.0
    edges = np.abs(np.diff(patch.astype(np.int16), axis=1)).mean()
    return float(np.std(patch) + edges)


def _crop_row_region(img, row_y, x0=120, x1=700, height=120):
    half = height // 2
    top = max(0, row_y - half)
    bottom = min(img.height, row_y + half)
    return img.crop((x0, top, x1, bottom))


def _row_has_new_game(img, row_y):
    crop = _crop_row_region(img, row_y)
    tmp_path = f"/tmp/detect-newgame-{row_y}.png"
    crop.save(tmp_path)
    out = subprocess.run([
        "tesseract", tmp_path, "stdout",
        "-c", "tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz "],
        capture_output=True, text=True
    ).stdout.lower()
    return bool(re.search(r'new\s*game|newgame', out))


def _detect_save_count_from_newgame(png_path):
    """Estimate save count by finding the New Game button row."""
    img = Image.open(png_path).convert('L')
    for total_items in range(1, 6):
        if _row_has_new_game(img, ROW_CENTERS[total_items][-1]):
            return total_items - 1
    return 6


def _detect_save_count_from_layout(png_path):
    """Estimate save count from known row layout using screenshot pixel structure."""
    gray = np.array(Image.open(png_path).convert('L'))
    best_total = 1
    best_score = -1.0
    for total_items in range(1, 7):
        score = sum(_row_activity_score(gray, y) for y in ROW_CENTERS[total_items])
        if score > best_score:
            best_score = score
            best_total = total_items
    return best_total if best_total == 6 else best_total - 1


def capture_seed_card(row_y, png_path):
    """Tap the save row, screenshot the card, and dismiss the card."""
    tap(280, row_y)
    subprocess.run(f"adb exec-out screencap -p > {png_path}", shell=True, check=True)
    tap(360, 1350)                 # Dismiss


def read_seed(row_y, png_path):
    """Capture a seed card screenshot and return its file path."""
    capture_seed_card(row_y, png_path)
    return png_path


def read_seeds_from_screenshots(paths):
    """OCR multiple seed screenshots in parallel."""
    procs = {
        path: subprocess.Popen(
            ["tesseract", path, "stdout",
             "-c", "tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ- "],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        for path in paths
    }
    results = {}
    for path, p in procs.items():
        out, _ = p.communicate()
        results[path] = _parse_seed_from_text(out)
    return results

# ── ADB helpers ───────────────────────────────────────────────────────────────

def tap(x, y, wait=0.1):
    subprocess.run(f"adb shell input tap {x} {y}", shell=True, check=True)
    time.sleep(wait)

def taps(*points, wait=0.1):
    """Send several taps in one adb invocation, saving a shell roundtrip per tap.
    No sleeps between them — the on-device `input` command is slow enough to
    cover UI transitions."""
    cmd = "; ".join(f"input tap {x} {y}" for x, y in points)
    subprocess.run(f'adb shell "{cmd}"', shell=True, check=True)
    time.sleep(wait)

# ── Game actions ──────────────────────────────────────────────────────────────

def detect_save_count():
    """Count saves by screenshotting the games list and using OCR or button-layout detection."""
    subprocess.run("adb exec-out screencap -p > /tmp/detect.png", shell=True, check=True)
    out = subprocess.run(["tesseract", "/tmp/detect.png", "stdout"],
                         capture_output=True, text=True).stdout
    t = out.lower()
    # Each save shows a timestamp: "X minutes ago", "just now", "moments ago", etc.
    count = t.count(' ago') + t.count('just now') + t.count('moments ago')
    if count == 0:
        print("  OCR save-count detection failed; checking New Game row")
        count = _detect_save_count_from_newgame("/tmp/detect.png")
        if count == 6:
            print("  New Game check also missed; falling back to layout inference")
            count = _detect_save_count_from_layout("/tmp/detect.png")
    print(f"  Detected {count} save(s)")
    return count

def nav_to_games_list():
    """From main menu, navigate to games list."""
    tap(360, 750)
    time.sleep(0.2)

def create_game(char, save_count):
    """Call from games list. Taps NGB, creates game, returns to main menu."""
    total = min(save_count + 1, 6)
    ngy = ROW_CENTERS[total][-1]
    taps((360, ngy),               # New Game button
         (CHAR_X[char], 1430),     # Character icon
         (400, 1340), wait=1.0)    # Class button (loading starts)
    tap(290, 929, wait=1.0)        # Continue (dungeon loads)
    taps((690, 55),                # ≡ menu
         (360, 820))               # Main Menu

def delete_save(row_y):
    taps((280, row_y),
         (525, 940),               # Erase
         (360, 720))               # Yes

SEED_FINDER_TIMEOUT = 30

# Weapon names by tier, as regex fragments. "shortsword" gets a lookbehind so
# tier 2 doesn't match "worn shortsword"; plain "sword" is safe because \bsword
# can't start mid-word in longsword/greatsword/shortsword.
_TIER_WEAPONS = {
    1: ["worn shortsword", "dagger", "studded gloves", "rapier", "cudgel", "mage's staff"],
    2: [r"(?<!worn )shortsword", "hand axe", r"(?<!sacrificial )(?<!throwing )spear",
        "dirk", "quarterstaff", "sickle", "pickaxe"],
    3: ["sword", "mace", "scimitar", "sai", "round shield", "whip"],
    4: ["longsword", "battle axe", "flail", "assassin's blade", "runic blade", "crossbow", "katana"],
    5: ["greatsword", "war hammer", "glaive", "greataxe", "greatshield", "stone gauntlet", "war scythe"],
}


def expand_tier_macros(pattern):
    """Expand %tierN shorthand into an alternation of that tier's weapon names,
    e.g. '%tier4 \\+3' → '(?:\\blongsword\\b|\\bbattle axe\\b|...) \\+3'."""
    def repl(m):
        tier = int(m.group(1))
        names = _TIER_WEAPONS.get(tier)
        if names is None:
            raise ValueError(f"unknown weapon tier {tier} (valid: 1-5)")
        return "(?:" + "|".join(rf"\b{n}\b" for n in names) + ")"
    return re.sub(r'%tier(\d+)', repl, pattern, flags=re.IGNORECASE)


def _parse_floor_sections(output):
    """Split seed-finder output into (floor, lines) sections.
    Lines before the first '--- floor N:' header (trinkets etc.) are floor 0."""
    sections = []
    floor, lines = 0, []
    for line in output.splitlines():
        m = re.match(r'---\s*floor\s+(\d+)', line)
        if m:
            sections.append((floor, lines))
            floor, lines = int(m.group(1)), []
        else:
            lines.append(line)
    sections.append((floor, lines))
    return sections


def _match_criteria(output, criteria):
    """criteria: list of (compiled_regex, max_floor). Return annotated match
    lines if EVERY regex hits a line on floors <= its max_floor (floor 0 =
    seed-wide items, always in range); else None."""
    sections = _parse_floor_sections(output)
    hits = []
    for regex, max_floor in criteria:
        found = [(floor, line.strip())
                 for floor, lines in sections if floor <= max_floor
                 for line in lines if regex.search(line)]
        if not found:
            return None
        hits.extend(found)
    seen, out = set(), []
    for floor, line in hits:
        if (floor, line) in seen:
            continue
        seen.add((floor, line))
        out.append(f"floor {floor}: {line}" if floor else line)
    return out


def check_seed(seed, criteria):
    run_floors = max(fl for _, fl in criteria)
    proc = subprocess.run(
        ["java", "-jar", SEED_FINDER, str(run_floors), seed],
        capture_output=True, text=True, timeout=SEED_FINDER_TIMEOUT
    )
    return "\n".join(_match_criteria(proc.stdout, criteria) or [])

# ── Main loop ─────────────────────────────────────────────────────────────────

def run(criteria, required_count, char="duelist", preserve_existing=False):
    matching = {}   # seed → match_text for preserved saves
    iteration = 0
    run_floors = max(fl for _, fl in criteria)

    # Navigate to games list once — stay there throughout
    nav_to_games_list()
    save_count = detect_save_count()

    # Saves present at startup are the oldest, so they sit on the bottom rows;
    # with --preserve-existing we never tap, OCR, or delete them.
    protected_count = save_count if preserve_existing else 0
    target = required_count
    if preserve_existing:
        available = 6 - protected_count
        if available <= 0:
            print("All 6 slots hold existing saves — nothing to do with --preserve-existing.")
            return
        if target > available:
            print(f"  Target {target} exceeds {available} free slot(s); filling free slots instead.")
            target = available

    while len(matching) < target:
        iteration += 1
        print(f"\n── Iteration {iteration} ──")

        # Fill to 6 saves
        while save_count < 6:
            print(f"  Creating game {save_count+1}/6 ({char})...")
            create_game(char, save_count)   # returns to main menu
            nav_to_games_list()
            save_count += 1

        # Read seeds — already on games list. Bottom-up row order is:
        # protected startup saves (oldest), then preserved matches, then live saves.
        preserved_seeds = list(matching.keys())
        seeds = []
        rows = ROW_CENTERS[6]
        first_protected_row = len(rows) - protected_count
        first_preserved_row = first_protected_row - len(preserved_seeds)
        for i, ry in enumerate(rows):
            if i >= first_protected_row:
                print(f"  Row {i+1}: protected existing save — untouched")
                seeds.append({"row_y": ry, "seed": None, "kind": "protected", "screenshot": None})
            elif i >= first_preserved_row:
                seed = preserved_seeds[i - first_preserved_row]
                print(f"  Row {i+1} preserved seed: {seed}")
                seeds.append({"row_y": ry, "seed": seed, "kind": "matched", "screenshot": None})
            else:
                path = f"/tmp/card-{i}.png"
                read_seed(ry, path)
                print(f"  Row {i+1} screenshot saved: {path}")
                seeds.append({"row_y": ry, "seed": None, "kind": "live", "screenshot": path})

        print("  Running OCR on new saves...")
        screenshot_paths = [entry["screenshot"] for entry in seeds if entry["kind"] == "live"]
        ocr_results = read_seeds_from_screenshots(screenshot_paths)
        for idx, entry in enumerate(seeds):
            if entry["kind"] != "live":
                continue
            entry["seed"] = ocr_results.get(entry["screenshot"])
            if entry["seed"] is None:
                print(f"    Row {idx+1} unknown: OCR failed")
            else:
                print(f"    Row {idx+1} seed: {entry['seed']}")

        # Run seed finder on new/unverified saves only
        print("  Running seed finder...")
        procs = {}
        for entry in seeds:
            if entry["kind"] != "live":
                continue
            seed = entry["seed"]
            if seed is None:
                print(f"    Row unknown: OCR failed — skipping")
                continue
            procs[seed] = subprocess.Popen(
                ["java", "-jar", SEED_FINDER, str(run_floors), seed],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )

        results = {}
        for seed, p in procs.items():
            try:
                out, _ = p.communicate(timeout=SEED_FINDER_TIMEOUT)
            except subprocess.TimeoutExpired:
                p.kill()
                out, _ = p.communicate()
                print(f"    {seed}: timed out after {SEED_FINDER_TIMEOUT}s")
                results[seed] = ""
                continue
            lines = _match_criteria(out, criteria)
            results[seed] = "\n".join(lines) if lines else ""
            print(f"    {seed}: " + ("MATCH (all patterns)" if lines else "no match"))
            for line in lines or []:
                print(f"      {line}")

        # Already on games list after last seed dismiss — no nav tap needed

        # Delete non-matching live saves top-down; matched and protected rows stay
        saves_remaining = 6
        entries = list(seeds)  # top to bottom order

        # Walk top-down; always re-read position from table after each deletion
        i = 0
        while i < len(entries):
            entry = entries[i]
            keep = entry["kind"] != "live" or bool(results.get(entry["seed"]))
            if not keep:
                # total items on screen = saves + NGB (unless saves == 6, no NGB)
                total_items = saves_remaining if saves_remaining == 6 else saves_remaining + 1
                ry = ROW_CENTERS[total_items][i]
                print(f"  Deleting row {i+1} ({entry['seed']})...")
                delete_save(ry)
                saves_remaining -= 1
                entries.pop(i)
                # don't increment i — next entry shifts into position i
            else:
                if entry["kind"] == "live":
                    matching[entry["seed"]] = results[entry["seed"]]
                i += 1

        save_count = saves_remaining
        print(f"  Matching so far: {len(matching)}/{target}: {list(matching.keys())}")

    print(f"\n✓ Done. {len(matching)} matching save(s):")
    for seed, items in matching.items():
        print(f"  {seed}:")
        for line in items.splitlines():
            print(f"    {line}")

def parse_args(argv):
    """Return (criteria, count, char, preserve_existing) from CLI args."""
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="SPD seed finder automation. A seed matches only if EVERY "
                    "pattern hits a line of seed-finder output within its floor limit. "
                    "A character class may be given as the final argument "
                    f"({'|'.join(sorted(CHAR_X))}; default duelist).",
        epilog="""examples:
  # 1 save with 'scroll of upgrade' somewhere in the first 7 floors
  %(prog)s 'scroll of upgrade' 1

  # 2 warrior saves with any tier-4 weapon at exactly +3
  %(prog)s '%%tier4 \\+3' 2 warrior

  # BOTH must hit: shortsword +3 by floor 4 AND greatshield +3 by floor 7
  %(prog)s 'shortsword \\+3' 'greatshield \\+3' 1 --floors 4 7

  # fill free slots only, never touching saves already on the device
  %(prog)s '%%tier5 \\+2' 4 --preserve-existing

  # enchanted weapons print as 'longsword of force +3' — use .* before the +N
  %(prog)s '%%tier4.*\\+3' 1""")
    ap.add_argument("patterns", nargs="+", metavar="pattern",
                    help="case-insensitive regex(es) against seed-finder output; "
                         "%%tierN expands to tier-N weapon names (N = 1-5)")
    ap.add_argument("count", type=int, help="how many matching saves to end up with")
    ap.add_argument("--floors", type=int, nargs="+", metavar="N",
                    help=f"max floor per pattern: one value for all patterns, "
                         f"or one per pattern (default {FLOORS})")
    ap.add_argument("--preserve-existing", action="store_true",
                    help="never delete saves present at startup; work in the free slots")

    # argparse can't disambiguate a trailing optional positional after nargs='+',
    # so peel off a class name (last positional, possibly before flags) ourselves
    argv = list(argv)
    char = "duelist"
    for i in range(len(argv) - 1, -1, -1):
        if argv[i] in CHAR_X and (i == len(argv) - 1 or argv[i + 1].startswith("-")):
            char = argv.pop(i)
            break
    args = ap.parse_args(argv)

    floors = args.floors or [FLOORS]
    if len(floors) == 1:
        floors *= len(args.patterns)
    if len(floors) != len(args.patterns):
        ap.error(f"--floors takes 1 or {len(args.patterns)} values to match the patterns")

    criteria = []
    for pat, fl in zip(args.patterns, floors):
        try:
            criteria.append((re.compile(expand_tier_macros(pat), re.IGNORECASE), fl))
        except ValueError as e:
            ap.error(f"{pat!r}: {e}")
        except re.error as e:
            ap.error(f"bad regex {pat!r}: {e}")

    return criteria, args.count, char, args.preserve_existing


if __name__ == "__main__":
    import sys
    run(*parse_args(sys.argv[1:]))
