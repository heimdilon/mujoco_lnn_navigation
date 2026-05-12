from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "report" / "weekly_2026_05_11"
FIGURE_DIR = REPORT_DIR / "figures"
CURRENT_METRICS = REPORT_DIR / "weekly_metrics.csv"
OLD_METRICS = ROOT / "results" / "cfc_radius010_custom22_dagger2_eval_all24_dynamic" / "summary.csv"

EPISODES_PER_MAP = 4

BG = (248, 249, 246)
PANEL = (255, 255, 252)
INK = (34, 39, 44)
MUTED = (92, 101, 109)
GRID = (218, 223, 224)
GREEN = (42, 150, 85)
GREEN_DARK = (25, 102, 58)
GREEN_LIGHT = (218, 240, 226)
RED = (202, 70, 57)
RED_DARK = (138, 45, 39)
RED_LIGHT = (247, 223, 220)
ORANGE = (222, 150, 55)
BLUE = (44, 111, 185)
BLUE_LIGHT = (225, 238, 250)
GRAY = (174, 181, 187)


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def number_from_map(name: str) -> str:
    if name.startswith("custom_map_"):
        return name.removeprefix("custom_map_")
    if name == "dynamic_open_single":
        return "dyn-open"
    if name == "dynamic_crossing":
        return "dyn-cross"
    return name


def as_float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def status_for(row: dict[str, str]) -> str:
    if as_float(row, "success_rate") >= 0.999:
        return "success"
    if as_float(row, "collision_rate") >= as_float(row, "timeout_rate"):
        return "collision"
    return "timeout"


def color_for(status: str, light: bool = False) -> tuple[int, int, int]:
    if status == "success":
        return GREEN_LIGHT if light else GREEN
    if status == "collision":
        return RED_LIGHT if light else RED
    if status == "timeout":
        return (250, 234, 206) if light else ORANGE
    return (232, 234, 235) if light else GRAY


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/seguisb.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf"),
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


F_TITLE = font(52, True)
F_H1 = font(38, True)
F_H2 = font(30, True)
F_BODY = font(24)
F_BODY_BOLD = font(24, True)
F_SMALL = font(19)
F_TINY = font(16)


def text_size(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.ImageFont) -> tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=fnt)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def text_center(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    fnt: ImageFont.ImageFont,
    fill: tuple[int, int, int] = INK,
) -> None:
    w, h = text_size(draw, text, fnt)
    x0, y0, x1, y1 = box
    draw.text((x0 + (x1 - x0 - w) / 2, y0 + (y1 - y0 - h) / 2 - 2), text, font=fnt, fill=fill)


def save(image: Image.Image, name: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    image.save(FIGURE_DIR / name, quality=95)


def draw_progress(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], value: float, color: tuple[int, int, int]) -> None:
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(box, radius=12, fill=(232, 235, 235))
    fill_w = int((x1 - x0) * max(0.0, min(1.0, value)))
    if fill_w > 0:
        draw.rounded_rectangle((x0, y0, x0 + fill_w, y1), radius=12, fill=color)


def card(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], title: str, value: str, note: str, color: tuple[int, int, int]) -> None:
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(box, radius=24, fill=PANEL, outline=(224, 228, 229), width=2)
    draw.rounded_rectangle((x0, y0, x1, y0 + 12), radius=10, fill=color)
    draw.text((x0 + 30, y0 + 34), title, font=F_BODY_BOLD, fill=INK)
    draw.text((x0 + 30, y0 + 82), value, font=F_H1, fill=color)
    draw.text((x0 + 30, y0 + 148), note, font=F_SMALL, fill=MUTED)


def build_kpi_dashboard(rows: list[dict[str, str]], old_rows: list[dict[str, str]]) -> None:
    total_maps = len(rows)
    success_maps = sum(status_for(row) == "success" for row in rows)
    collision_maps = sum(status_for(row) == "collision" for row in rows)
    timeout_maps = sum(status_for(row) == "timeout" for row in rows)
    old_success = sum(status_for(row) == "success" for row in old_rows)
    success_episodes = int(sum(as_float(row, "success_rate") * EPISODES_PER_MAP for row in rows))
    collision_episodes = int(sum(as_float(row, "collision_rate") * EPISODES_PER_MAP for row in rows))
    timeout_episodes = int(sum(as_float(row, "timeout_rate") * EPISODES_PER_MAP for row in rows))

    width, height = 1800, 1040
    image = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(image)
    draw.text((70, 54), "Haftalik PDF Dashboard", font=F_TITLE, fill=INK)
    draw.text(
        (72, 120),
        "Saf cfc_deep192 politika | 24 harita | 96 episode | safety filter yok",
        font=F_BODY,
        fill=MUTED,
    )

    gap = 28
    card_w = (width - 140 - 3 * gap) // 4
    y = 188
    cards = [
        ("Harita basarisi", f"{success_maps}/{total_maps}", "Basari orani: %.1f%%" % (100 * success_maps / total_maps), GREEN),
        ("Episode sonucu", f"{success_episodes}/96", f"{collision_episodes} collision, {timeout_episodes} timeout", BLUE),
        ("Kalan hata", f"{collision_maps}+{timeout_maps}", "3 harita collision, timeout yok", RED),
        ("Model ilerlemesi", f"+{success_maps - old_success}", f"{old_success}/24 -> {success_maps}/24", ORANGE),
    ]
    for idx, data in enumerate(cards):
        x = 70 + idx * (card_w + gap)
        card(draw, (x, y, x + card_w, y + 210), *data)

    draw.rounded_rectangle((70, 455, width - 70, 952), radius=28, fill=PANEL, outline=(224, 228, 229), width=2)
    draw.text((105, 494), "Kirilim bazli sonuc", font=F_H2, fill=INK)
    draw.text((105, 536), "Train/holdout ve custom/dynamic ayrimi tek bakista okunur.", font=F_BODY, fill=MUTED)

    groups = [
        ("Custom train", [row for row in rows if row["map_type"] == "custom" and row["split"] == "train"]),
        ("Custom holdout", [row for row in rows if row["map_type"] == "custom" and row["split"] == "holdout"]),
        ("Dynamic train", [row for row in rows if row["map_type"] == "dynamic" and row["split"] == "train"]),
        ("Dynamic holdout", [row for row in rows if row["map_type"] == "dynamic" and row["split"] == "holdout"]),
    ]
    bar_x0, bar_x1 = 460, width - 190
    bar_y = 610
    for idx, (name, group_rows) in enumerate(groups):
        gy = bar_y + idx * 82
        group_total = max(1, len(group_rows))
        group_success = sum(status_for(row) == "success" for row in group_rows)
        group_collision = sum(status_for(row) == "collision" for row in group_rows)
        draw.text((105, gy + 6), name, font=F_BODY_BOLD, fill=INK)
        draw.text((335, gy + 7), f"{group_success}/{group_total}", font=F_BODY, fill=MUTED)
        draw.rounded_rectangle((bar_x0, gy, bar_x1, gy + 38), radius=18, fill=(234, 237, 237))
        success_w = int((bar_x1 - bar_x0) * group_success / group_total)
        collision_w = int((bar_x1 - bar_x0) * group_collision / group_total)
        if success_w:
            draw.rounded_rectangle((bar_x0, gy, bar_x0 + success_w, gy + 38), radius=18, fill=GREEN)
        if collision_w:
            draw.rounded_rectangle((bar_x0 + success_w, gy, bar_x0 + success_w + collision_w, gy + 38), radius=18, fill=RED)
        draw.text((bar_x1 + 20, gy + 4), f"{100 * group_success / group_total:.0f}%", font=F_BODY_BOLD, fill=GREEN if group_success else RED)

    legend_y = 898
    for x, label, color in [(105, "success", GREEN), (230, "collision", RED), (365, "timeout", ORANGE)]:
        draw.rounded_rectangle((x, legend_y, x + 30, legend_y + 30), radius=8, fill=color)
        draw.text((x + 42, legend_y - 1), label, font=F_SMALL, fill=MUTED)
    save(image, "kpi_dashboard.png")


def build_map_matrix(rows: list[dict[str, str]]) -> None:
    width, height = 1800, 1120
    image = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(image)
    draw.text((70, 54), "Harita Risk Matrisi", font=F_TITLE, fill=INK)
    draw.text((72, 120), "Her kutu bir harita: sonuc, ortalama adim ve hedefe kalan mesafe.", font=F_BODY, fill=MUTED)

    sections = [
        ("CUSTOM TRAIN", [row for row in rows if row["map_type"] == "custom" and row["split"] == "train"], 2),
        ("CUSTOM HOLDOUT", [row for row in rows if row["map_type"] == "custom" and row["split"] == "holdout"], 1),
        ("DYNAMIC", [row for row in rows if row["map_type"] == "dynamic"], 1),
    ]
    tile_w, tile_h = 124, 126
    gap = 16
    left = 350
    y = 215
    for label, group_rows, section_rows in sections:
        draw.text((70, y + 28), label, font=F_BODY_BOLD, fill=MUTED)
        for idx, row in enumerate(group_rows):
            local_row = idx // 9
            col = idx % 9
            x0 = left + col * (tile_w + gap)
            y0 = y + local_row * (tile_h + gap)
            x1, y1 = x0 + tile_w, y0 + tile_h
            status = status_for(row)
            draw.rounded_rectangle((x0, y0, x1, y1), radius=18, fill=color_for(status, True), outline=color_for(status), width=3)
            text_center(draw, (x0 + 8, y0 + 8, x1 - 8, y0 + 42), number_from_map(row["map"]), F_BODY_BOLD, color_for(status))
            text_center(draw, (x0 + 8, y0 + 45, x1 - 8, y0 + 76), "OK" if status == "success" else "COL", F_SMALL, INK)
            draw.text((x0 + 16, y0 + 82), f"{as_float(row, 'mean_steps'):.0f} st", font=F_TINY, fill=MUTED)
            draw.text((x0 + 16, y0 + 104), f"{as_float(row, 'mean_final_distance'):.2f} m", font=F_TINY, fill=MUTED)
        y += section_rows * (tile_h + gap) + 90

    draw.rounded_rectangle((70, height - 130, width - 70, height - 64), radius=18, fill=PANEL, outline=(224, 228, 229), width=2)
    draw.text((105, height - 112), "Okuma:", font=F_BODY_BOLD, fill=INK)
    draw.text((200, height - 112), "yesil hedefe ulasma, kirmizi collision, metinler ortalama adim ve final mesafeyi gosterir.", font=F_BODY, fill=MUTED)
    save(image, "map_risk_matrix.png")


def build_steps_distance(rows: list[dict[str, str]]) -> None:
    width, height = 1800, 1000
    image = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(image)
    draw.text((70, 54), "Adim - Final Mesafe Dagilimi", font=F_TITLE, fill=INK)
    draw.text((72, 120), "Basarili haritalar hedef bandinda toparlaniyor; collision haritalari final mesafede ayrisiyor.", font=F_BODY, fill=MUTED)

    x0, y0, x1, y1 = 190, 210, width - 120, height - 150
    draw.rounded_rectangle((x0 - 65, y0 - 42, x1 + 42, y1 + 72), radius=24, fill=PANEL, outline=(224, 228, 229), width=2)
    draw.rectangle((x0, y0, x1, y1), fill=(252, 253, 250))

    max_steps = 900.0
    max_dist = max(as_float(row, "mean_final_distance") for row in rows) * 1.08

    def sx(value: float) -> int:
        return int(x0 + (value / max_steps) * (x1 - x0))

    def sy(value: float) -> int:
        return int(y1 - (value / max_dist) * (y1 - y0))

    goal_y = sy(0.30)
    draw.rectangle((x0, goal_y, x1, y1), fill=(226, 243, 233))
    draw.text((x0 + 16, goal_y + 12), "goal band <= 0.30 m", font=F_SMALL, fill=GREEN_DARK)

    for tick in range(0, 901, 150):
        x = sx(tick)
        draw.line((x, y0, x, y1), fill=GRID, width=1)
        text_center(draw, (x - 35, y1 + 16, x + 35, y1 + 48), str(tick), F_SMALL, MUTED)
    dist_tick = 0.0
    while dist_tick <= max_dist + 0.01:
        y = sy(dist_tick)
        draw.line((x0, y, x1, y), fill=GRID, width=1)
        draw.text((x0 - 72, y - 12), f"{dist_tick:.1f}", font=F_SMALL, fill=MUTED)
        dist_tick += 1.0

    draw.line((x0, y1, x1, y1), fill=INK, width=2)
    draw.line((x0, y0, x0, y1), fill=INK, width=2)
    text_center(draw, (x0 + 420, y1 + 52, x1 - 420, y1 + 86), "Ortalama episode adimi", F_BODY_BOLD, INK)
    draw.text((42, y0 + 250), "Final mesafe (m)", font=F_BODY_BOLD, fill=INK)

    for row in rows:
        x = sx(as_float(row, "mean_steps"))
        y = sy(as_float(row, "mean_final_distance"))
        status = status_for(row)
        radius = 15 if row["map_type"] == "dynamic" else 12
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color_for(status), outline=(255, 255, 255), width=3)
        label = number_from_map(row["map"])
        if status != "success" or row["map_type"] == "dynamic":
            draw.text((x + 16, y - 13), label, font=F_SMALL, fill=INK)

    legend = [(GREEN, "success"), (RED, "collision"), (BLUE, "dynamic label")]
    lx = x1 - 410
    ly = y0 + 28
    draw.rounded_rectangle((lx - 24, ly - 20, lx + 360, ly + 122), radius=18, fill=(255, 255, 252), outline=(224, 228, 229))
    for idx, (color, label) in enumerate(legend):
        yy = ly + idx * 42
        draw.ellipse((lx, yy, lx + 24, yy + 24), fill=color)
        draw.text((lx + 38, yy - 2), label, font=F_SMALL, fill=MUTED)
    save(image, "steps_distance_scatter.png")


def build_model_delta(rows: list[dict[str, str]], old_rows: list[dict[str, str]]) -> None:
    old_by_map = {row["map"]: row for row in old_rows}
    width, height = 1800, 960
    image = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(image)

    old_success = sum(status_for(row) == "success" for row in old_rows)
    new_success = sum(status_for(row) == "success" for row in rows)
    recovered = [
        row["map"]
        for row in rows
        if row["map"] in old_by_map and status_for(old_by_map[row["map"]]) != "success" and status_for(row) == "success"
    ]
    regressions = [
        row["map"]
        for row in rows
        if row["map"] in old_by_map and status_for(old_by_map[row["map"]]) == "success" and status_for(row) != "success"
    ]

    draw.text((70, 54), "Onceki Modele Gore Harita Kazanimi", font=F_TITLE, fill=INK)
    draw.text((72, 120), f"{old_success}/24 -> {new_success}/24 | kazanilan harita: {len(recovered)} | regresyon: {len(regressions)}", font=F_BODY, fill=MUTED)

    tile_w, tile_h = 126, 120
    gap = 14
    left, top = 80, 220
    for idx, row in enumerate(rows):
        col, rr = idx % 12, idx // 12
        x0 = left + col * (tile_w + gap)
        y0 = top + rr * (tile_h + gap)
        x1, y1 = x0 + tile_w, y0 + tile_h
        old_status = status_for(old_by_map.get(row["map"], row))
        new_status = status_for(row)
        draw.rounded_rectangle((x0, y0, x1, y1), radius=18, fill=PANEL, outline=(224, 228, 229), width=2)
        text_center(draw, (x0 + 6, y0 + 9, x1 - 6, y0 + 36), number_from_map(row["map"]), F_SMALL, INK)
        draw.rounded_rectangle((x0 + 14, y0 + 46, x1 - 14, y0 + 70), radius=12, fill=color_for(old_status, True), outline=color_for(old_status), width=2)
        draw.rounded_rectangle((x0 + 14, y0 + 82, x1 - 14, y0 + 106), radius=12, fill=color_for(new_status), outline=color_for(new_status), width=2)
        draw.text((x0 + 19, y0 + 48), "old", font=F_TINY, fill=MUTED)
        draw.text((x0 + 19, y0 + 83), "new", font=F_TINY, fill=(255, 255, 255) if new_status != "timeout" else INK)

    panel_y = 535
    draw.rounded_rectangle((80, panel_y, width - 80, height - 70), radius=24, fill=PANEL, outline=(224, 228, 229), width=2)
    draw.text((115, panel_y + 34), "Kazanilan haritalar", font=F_H2, fill=INK)
    draw.text((115, panel_y + 82), ", ".join(number_from_map(name) for name in recovered), font=F_BODY, fill=GREEN_DARK)
    draw.text((115, panel_y + 140), "Degismeyen kritik haritalar", font=F_H2, fill=INK)
    still_failed = [row["map"] for row in rows if status_for(row) != "success"]
    draw.text((115, panel_y + 188), ", ".join(number_from_map(name) for name in still_failed), font=F_BODY, fill=RED_DARK)
    draw.text((115, panel_y + 258), "Yorum", font=F_H2, fill=INK)
    draw.text(
        (115, panel_y + 306),
        "Deep CfC kapasitesi timeout problemini kaldirdi; kalan risk dar gecis ve dinamik crossing carpismasi.",
        font=F_BODY,
        fill=MUTED,
    )
    save(image, "model_delta_grid.png")


def build_failure_panel(rows: list[dict[str, str]]) -> None:
    failures = [row for row in rows if status_for(row) != "success"]
    width, height = 1800, 900
    image = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(image)
    draw.text((70, 54), "Failure Case Paneli", font=F_TITLE, fill=INK)
    draw.text((72, 120), "Uc kalan hata modunun rollout goruntusu ve sayisal izi.", font=F_BODY, fill=MUTED)

    card_w = 520
    gap = 52
    top = 205
    for idx, row in enumerate(failures):
        x0 = 70 + idx * (card_w + gap)
        y0 = top
        x1 = x0 + card_w
        y1 = height - 78
        draw.rounded_rectangle((x0, y0, x1, y1), radius=24, fill=PANEL, outline=(224, 228, 229), width=2)
        draw.rounded_rectangle((x0, y0, x1, y0 + 12), radius=10, fill=RED)
        draw.text((x0 + 26, y0 + 34), number_from_map(row["map"]), font=F_H2, fill=INK)
        draw.text((x0 + 26, y0 + 76), "collision 4/4", font=F_BODY_BOLD, fill=RED_DARK)

        figure_path = FIGURE_DIR / f"{row['map']}_rollout.png"
        if figure_path.exists():
            thumb = Image.open(figure_path).convert("RGB")
            thumb.thumbnail((card_w - 94, 300), Image.Resampling.LANCZOS)
            tx = x0 + (card_w - thumb.width) // 2
            ty = y0 + 128
            draw.rounded_rectangle((tx - 6, ty - 6, tx + thumb.width + 6, ty + thumb.height + 6), radius=16, fill=(241, 243, 241))
            image.paste(thumb, (tx, ty))
        else:
            draw.rounded_rectangle((x0 + 26, y0 + 128, x1 - 26, y0 + 518), radius=18, fill=(236, 238, 238))
            text_center(draw, (x0 + 26, y0 + 128, x1 - 26, y0 + 518), "rollout missing", F_BODY, MUTED)

        metric_y = y0 + 458
        draw.text((x0 + 28, metric_y), f"mean steps: {as_float(row, 'mean_steps'):.0f}", font=F_BODY, fill=INK)
        draw.text((x0 + 28, metric_y + 42), f"final distance: {as_float(row, 'mean_final_distance'):.2f} m", font=F_BODY, fill=INK)
        if row["map"] == "dynamic_crossing":
            note = "dinamik holdout stresi"
        elif row["map"] == "custom_map_03":
            note = "holdout statik failure"
        else:
            note = "train icinde kalan collision"
        draw.text((x0 + 28, metric_y + 88), note, font=F_SMALL, fill=MUTED)

    save(image, "failure_cases_panel.png")


def main() -> None:
    rows = load_rows(CURRENT_METRICS)
    old_rows = load_rows(OLD_METRICS)
    build_kpi_dashboard(rows, old_rows)
    build_map_matrix(rows)
    build_steps_distance(rows)
    build_model_delta(rows, old_rows)
    build_failure_panel(rows)
    print(f"wrote weekly figures to {FIGURE_DIR}")


if __name__ == "__main__":
    main()
