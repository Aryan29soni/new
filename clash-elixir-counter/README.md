# Clash Royale Opponent Elixir Counter

An external program that watches your phone screen through a camera and estimates how much
elixir your opponent has, based on the cards they play.

```
 camera ──► flatten phone screen ──► detect new enemy cards ──► elixir model ──► HUD window
            (perspective fix)        (template matching +        (regen, x2/x3,
                                      tracking)                   cap, cycle)
```

It runs on a separate computer and only *looks* at the screen. It never touches the game or the phone.

## How the count works

- Both players start at **5 elixir** and gain **1 elixir every 2.8 s**. This becomes **x2 in the last
  minute** and in overtime. The schedule can be changed with `--schedule`, or press `e` during a match.
- The count stops at **10**. Anything above 10 is shown as **leaked**, meaning the opponent wasted elixir.
- When the opponent plays a card, its cost is subtracted. If they play a card the counter thought
  they couldn't afford, the estimate was too low. The program then **corrects itself** to 0, since they
  must have had at least that much. The HUD shows these corrections as `fix +x`, so the count gets
  better as the match goes on.
- **Mirror** costs the last card + 1.
- **Cycle tracking:** the last 4 cards the opponent played are in their queue, not in their hand.
  Once all 8 of their cards have been seen, the HUD shows exactly which 4 they are holding
  (green) and which card comes back next.

## Setup

```bash
cd clash-elixir-counter
pip install -r requirements.txt        # Python 3.10+
python -m pytest -q                    # optional: run the tests
```

### 1. Point the camera at the phone

- Mount the camera so it looks at the phone screen as straight on as possible, filling most of the frame.
  A tripod or phone holder works well, and the camera must not move after calibration.
- Turn off the room light behind you if there is glare on the screen. Turn screen brightness up.
- The camera can be a USB webcam (`--source 0`, `1`, ...), or an old phone running an IP-camera app
  (`--source http://192.168.1.20:8080/video`).
- If you are testing with a screen recording instead of a camera, add `--no-calibration`.

### 2. Calibrate (once, or whenever the camera moves)

```bash
python -m elixir_counter calibrate --source 0
```

1. Click the 4 corners of the phone's game screen in order: top-left, top-right, bottom-right,
   bottom-left. Press Enter.
2. Drag a box over the **opponent's side of the arena**, where their cards get placed. Press Enter.

This saves `calibration.json`.

### 3. Teach it what the cards look like (for automatic detection)

```bash
python -m elixir_counter teach --source 0
```

Play a match or watch a replay. When an opponent unit appears, press **SPACE** to freeze the frame,
drag a tight box around the unit, press Enter, and type the card name (`hog`, `log`, `musk` are fine).
Crops go to `templates/<card>/`.

Tips:
- Crop right after the unit is placed, and include its **red enemy colours**. This keeps your own blue
  units from matching.
- Save 2–4 crops per card from different moments. More crops catch more plays.
- Start with the cards you meet most often. Cards without templates can still be entered with the keyboard.
- `teach` also works on a recorded video: `--source match.mp4`. Press any key to step through frames.

### 4. Run it

```bash
python -m elixir_counter run --source 0
```

Press **SPACE** the moment the battle starts. If you forget, the clock starts when the first card is seen.

| Key     | Action |
|---------|--------|
| `SPACE` | Battle started (resets to 5 elixir) |
| `1`–`8` | Opponent played deck slot N. Slots fill in as new cards are seen. This is the fastest manual input. |
| `n`     | Type a card name in the terminal |
| `u`     | Undo the last play, for a false detection |
| `+` `-` | Nudge the estimate by 1, e.g. for an Elixir Collector |
| `e`     | Elixir rate x1 → x2 → x3, for when the game announces Double or Triple Elixir and the counter is out of sync |
| `c`     | Sync to the in-game clock: type the time left, e.g. `1:42` |
| `v`     | Vision detection on/off |
| `r`     | Reset for a new match |
| `q`     | Quit |

Useful options:

```
--threshold 0.8     match score needed (raise it if you get false plays, lower it if plays are missed)
--confirm 2         frames a match must last before it counts
--manual            keyboard only, no vision
--schedule 0:1,120:2,240:3   custom elixir phases (start second : multiplier)
--cards my_cards.json        custom card costs
```

`python -m elixir_counter cards` lists every card and its cost. Costs are in
`elixir_counter/data/cards.json`. Supercell changes card costs and adds new cards from time to time,
so check the list and edit it if something is off.

## Accuracy and limits

- **Detection by camera is the weak link.** Glare, motion blur, unit animations, spell effects and
  overlapping units all lower match scores. Expect to adjust `--threshold` and to add more crops
  for some cards. Keep the camera window open: yellow circles are units being tracked, and red
  circles are units that were counted as plays.
- Detection only looks inside the zone you drew. Cards placed elsewhere are not seen
  automatically: spells cast on your side, Miner, Goblin Drill, Graveyard, and placements into your
  side after a princess tower falls. Use the `1`–`8` keys for those. If an enemy unit walks into the zone
  later, the counter may see it too late or count it at the wrong moment.
- Two copies of the same card placed in the same spot within ~3 s count as one play.
- The model assumes standard regeneration. Elixir Collector, Elixir Golem, and game modes with other
  elixir rules need manual nudges or a different `--schedule`.
- Supercell's terms of service may not allow outside tools that help you during matches,
  even ones that never touch the game. Use this at your own discretion: for practice, replays,
  or friendly battles.

## Project layout

```
elixir_counter/
  engine.py   elixir model: regen, phases, cap/leak, corrections, mirror, undo, cycle  (no OpenCV)
  cards.py    card cost database + fuzzy name lookup
  vision.py   camera input, perspective calibration, template detector with tracking
  hud.py      the "Opponent Elixir" window
  app.py      calibrate / teach / run / cards commands
  data/cards.json
templates/    your card crops (templates/<card>/*.png)
tests/        engine and detector tests (synthetic images, no camera needed)
```
