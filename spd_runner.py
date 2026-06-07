#!/usr/bin/env python3
"""
SPD Seed Finder Automation
Usage: python3 spd_runner.py <grep_pattern> <required_count> [char_class]
  grep_pattern: passed to grep -iE against seed-finder output
  required_count: how many matching saves to end up with
  char_class: warrior|mage|rogue|huntress|duelist|cleric (default: duelist)
"""

import subprocess, sys, time, re

SEED_FINDER = "/home/tris/pequod/tris/tools/seed-finder-3/seed-finder-3.3.0/seed-finder.jar"
FLOORS      = 4

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

# ── OCR ───────────────────────────────────────────────────────────────────────

def _read_seed_from_card(png_path):
    """Run tesseract on a saved screenshot and extract XXX-XXX-XXX seed."""
    out = subprocess.run(
        ["tesseract", png_path, "stdout",
         "-c", "tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ- "],
        capture_output=True, text=True
    ).stdout
    m = re.search(r'[A-Z]{3}-[A-Z]{3}-[A-Z]{3}', out)
    return m.group(0) if m else None

# ── ADB helpers ───────────────────────────────────────────────────────────────

def tap(x, y, wait=0.3):
    subprocess.run(f"adb shell input tap {x} {y}", shell=True, check=True)
    time.sleep(wait)

# ── Game actions ──────────────────────────────────────────────────────────────

def detect_save_count():
    """Count saves by screenshotting the games list and counting timestamp indicators."""
    subprocess.run("adb exec-out screencap -p > /tmp/detect.png", shell=True, check=True)
    out = subprocess.run(["tesseract", "/tmp/detect.png", "stdout"],
                         capture_output=True, text=True).stdout
    t = out.lower()
    # Each save shows a timestamp: "X minutes ago", "just now", "moments ago", etc.
    count = t.count(' ago') + t.count('just now') + t.count('moments ago')
    print(f"  Detected {count} save(s)")
    return count

def nav_to_games_list():
    """From main menu, navigate to games list."""
    tap(360, 750)
    time.sleep(1.0)

def create_game(char, save_count):
    """Call from games list. Taps NGB, creates game, returns to main menu."""
    total = min(save_count + 1, 6)
    ngy = ROW_CENTERS[total][-1]
    tap(360, ngy)                  # New Game button
    tap(CHAR_X[char], 1430)        # Character icon
    tap(400, 1340, wait=1.0)       # Class button (loading starts)
    tap(290, 929,  wait=1.0)       # Continue (dungeon loads)
    tap(690, 55)                   # ≡ menu
    tap(360, 820)                  # Main Menu

def read_seed(row_y):
    tap(280, row_y)
    subprocess.run("adb exec-out screencap -p > /tmp/card.png", shell=True, check=True)
    seed = _read_seed_from_card("/tmp/card.png")
    tap(360, 1350)                 # Dismiss
    return seed

def delete_save(row_y):
    tap(280, row_y)
    tap(525, 940)                  # Erase
    tap(360, 720, wait=0.3)        # Yes

def check_seed(seed, grep_pattern):
    cmd = (f"java -jar {SEED_FINDER} {FLOORS} {seed} 2>/dev/null"
           f" | grep -iE '{grep_pattern}'")
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return r.stdout.strip()

# ── Main loop ─────────────────────────────────────────────────────────────────

def run(grep_pattern, required_count, char="duelist"):
    matching = {}   # seed → match_text for preserved saves
    iteration = 0

    # Navigate to games list once — stay there throughout
    nav_to_games_list()
    save_count = detect_save_count()

    while len(matching) < required_count:
        iteration += 1
        print(f"\n── Iteration {iteration} ──")

        # Fill to 6 saves
        while save_count < 6:
            print(f"  Creating game {save_count+1}/6 ({char})...")
            create_game(char, save_count)   # returns to main menu
            nav_to_games_list()
            save_count += 1

        # Read seeds — already on games list, with preserved saves shifted to the bottom
        preserved_seeds = list(matching.keys())
        preserved_count = len(preserved_seeds)
        seeds = []
        rows = ROW_CENTERS[6]
        first_live_row = len(rows) - preserved_count
        for i, ry in enumerate(rows):
            if i >= first_live_row:
                seed = preserved_seeds[i - first_live_row]
                print(f"  Row {i+1} preserved seed: {seed}")
                seeds.append((ry, seed, True))
            else:
                seed = read_seed(ry)
                print(f"  Row {i+1} seed: {seed}")
                seeds.append((ry, seed, False))

        # Run seed finder on new/unverified saves only
        print("  Running seed finder...")
        procs = {}
        for ry, seed, preserved in seeds:
            if preserved:
                continue
            if seed is None:
                print(f"    Row unknown: OCR failed — skipping")
                continue
            cmd = (f"java -jar {SEED_FINDER} {FLOORS} {seed} 2>/dev/null"
                   f" | grep -iE '{grep_pattern}'")
            procs[seed] = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE,
                                           stderr=subprocess.PIPE, text=True)

        results = {}
        for seed, p in procs.items():
            out, _ = p.communicate()
            results[seed] = out.strip()
            status = "MATCH" if out.strip() else "no match"
            print(f"    {seed}: {status}" + (f" — {out.strip()[:60]}" if out.strip() else ""))

        # Already on games list after last seed dismiss — no nav tap needed

        # Delete non-matching saves top-down; preserve already-matched rows
        to_keep = set(preserved_seeds) | {seed for seed, out in results.items() if out}
        saves_remaining = 6
        seeds_by_row = [seed for _, seed, _ in seeds]  # top to bottom order

        # Walk top-down; always re-read position from table after each deletion
        i = 0
        while i < len(seeds_by_row):
            seed = seeds_by_row[i]
            if seed not in to_keep:
                # total items on screen = saves + NGB (unless saves == 6, no NGB)
                total_items = saves_remaining if saves_remaining == 6 else saves_remaining + 1
                ry = ROW_CENTERS[total_items][i]
                print(f"  Deleting row {i+1} ({seed})...")
                delete_save(ry)
                saves_remaining -= 1
                seeds_by_row.pop(i)
                # don't increment i — next seed shifts into position i
            else:
                if seed in results:
                    matching[seed] = results[seed]
                i += 1

        save_count = saves_remaining
        print(f"  Matching so far: {len(matching)}/{required_count}: {list(matching.keys())}")

    print(f"\n✓ Done. {len(matching)} matching save(s):")
    for seed, items in matching.items():
        print(f"  {seed}:")
        for line in items.splitlines():
            print(f"    {line}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 spd_runner.py <grep_pattern> <required_count> [char_class]")
        sys.exit(1)
    pattern = sys.argv[1]
    count   = int(sys.argv[2])
    char    = sys.argv[3] if len(sys.argv) > 3 else "duelist"
    run(pattern, count, char)
