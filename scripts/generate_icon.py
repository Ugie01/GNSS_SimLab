"""
Generate a 4x-supersampled (1024x1024 -> Lanczos downsampled) multi-resolution Windows icon
(`assets/app.ico`, 16x16 to 256x256) and high-DPI sidebar logo (`assets/app_logo.png`)
with an aerospace/defense GNSS satellite, 3D Earth grid, orbital ring, CRPA wavefronts,
and EGI navigation reticle theme using Pillow.
"""

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter


def draw_satellite_icon(size: int = 256) -> Image.Image:
    """
    Render a high-contrast, 4x-supersampled anti-aliased GNSS satellite + Earth horizon
    + orbital ring + CRPA RF wavefronts icon.
    """
    ss = 4  # 4x supersampling factor for razor-sharp anti-aliasing
    W = size * ss
    img = Image.new("RGBA", (W, W), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # 1. Rounded deep-space navy badge background with subtle inner glow layers
    pad = int(W * 0.045)
    radius = int(W * 0.22)

    # Outer dark slate shadow/border
    draw.rounded_rectangle(
        [pad, pad, W - pad, W - pad],
        radius=radius,
        fill=(10, 17, 34, 255),
        outline=(56, 189, 248, 255),
        width=max(4, int(W * 0.022)),
    )

    # Inner deep-navy gradient steps
    for k in range(12):
        inset = pad + int(W * 0.018) + k * int(W * 0.015)
        r_in = max(8, radius - k * int(W * 0.012))
        col = (15 + k, 25 + k * 2, 52 + k * 3, 255)
        draw.rounded_rectangle([inset, inset, W - inset, W - inset], radius=r_in, fill=col)

    # Subtle background radar/navigation reticle grid lines
    cx_mid, cy_mid = W // 2, W // 2
    grid_col = (51, 65, 85, 95)
    draw.line([pad + int(W * 0.08), cy_mid, W - pad - int(W * 0.08), cy_mid], fill=grid_col, width=max(2, ss))
    draw.line([cx_mid, pad + int(W * 0.08), cx_mid, W - pad - int(W * 0.08)], fill=grid_col, width=max(2, ss))

    # 2. Glowing Earth horizon & latitude/longitude arcs in lower-left corner
    earth_r = int(W * 0.43)
    earth_cx = int(W * 0.25)
    earth_cy = int(W * 0.78)

    # Atmosphere outer halo
    for h_idx in range(5):
        hr = earth_r + (5 - h_idx) * int(W * 0.012)
        draw.ellipse(
            [earth_cx - hr, earth_cy - hr, earth_cx + hr, earth_cy + hr],
            fill=(56, 189, 248, 22 + h_idx * 12),
        )

    # Earth body
    draw.ellipse(
        [earth_cx - earth_r, earth_cy - earth_r, earth_cx + earth_r, earth_cy + earth_r],
        fill=(23, 62, 168, 245),
        outline=(96, 165, 250, 255),
        width=max(4, int(W * 0.016)),
    )

    # Earth latitude/longitude wireframe curves
    for lat_scale in [0.35, 0.65, 0.88]:
        lr = int(earth_r * lat_scale)
        draw.arc(
            [earth_cx - lr, earth_cy - lr, earth_cx + lr, earth_cy + lr],
            start=260,
            end=20,
            fill=(147, 197, 253, 125),
            width=max(2, int(W * 0.008)),
        )

    # Local ENU target marker on Earth surface
    target_x = int(W * 0.34)
    target_y = int(W * 0.62)
    tr = int(W * 0.025)
    draw.ellipse(
        [target_x - tr, target_y - tr, target_x + tr, target_y + tr],
        fill=(34, 197, 94, 255),
        outline=(220, 252, 231, 255),
        width=max(2, int(W * 0.008)),
    )

    # 3. Dynamic golden/cyan orbital ellipse ring
    orbit_bbox = [int(W * 0.13), int(W * 0.16), int(W * 0.89), int(W * 0.84)]
    draw.arc(
        orbit_bbox,
        start=180,
        end=42,
        fill=(148, 163, 184, 210),
        width=max(3, int(W * 0.014)),
    )

    # 4. Concentric RF navigation signal wavefronts radiating from satellite toward Earth target
    sat_x = int(W * 0.67)
    sat_y = int(W * 0.32)
    for ring_idx, r_scale in enumerate([0.15, 0.24, 0.33]):
        r_px = int(W * r_scale)
        alpha = 245 - ring_idx * 45
        col_wave = (56, 189, 248, alpha) if ring_idx < 2 else (34, 197, 94, alpha)
        draw.arc(
            [sat_x - r_px, sat_y - r_px, sat_x + r_px, sat_y + r_px],
            start=102,
            end=190,
            fill=col_wave,
            width=max(4, int(W * 0.020)),
        )

    # 5. Satellite solar array wings, cell grid, antenna dish, and golden bus body
    angle = math.radians(-35)
    cos_a, sin_a = math.cos(angle), math.sin(angle)

    def rot_pt(dx: float, dy: float) -> tuple[int, int]:
        rx = sat_x + dx * cos_a - dy * sin_a
        ry = sat_y + dx * sin_a + dy * cos_a
        return int(rx), int(ry)

    def draw_rot_rect(x0: float, y0: float, x1: float, y1: float, fill_col, out_col, width_px: int = 3) -> None:
        pts = [rot_pt(x0, y0), rot_pt(x1, y0), rot_pt(x1, y1), rot_pt(x0, y1)]
        draw.polygon(pts, fill=fill_col, outline=out_col)
        for i in range(4):
            draw.line([pts[i], pts[(i + 1) % 4]], fill=out_col, width=width_px)

    s = W / 256.0
    border_w = max(2, int(2.5 * s))

    # Left & Right Solar Wings
    draw_rot_rect(-68 * s, -15 * s, -22 * s, 15 * s, (29, 78, 216, 255), (147, 197, 253, 255), border_w)
    draw_rot_rect(22 * s, -15 * s, 68 * s, 15 * s, (29, 78, 216, 255), (147, 197, 253, 255), border_w)

    # Solar cell divider lines on wings
    for x_div in [-53 * s, -38 * s, 38 * s, 53 * s]:
        draw.line([rot_pt(x_div, -14 * s), rot_pt(x_div, 14 * s)], fill=(96, 165, 250, 220), width=max(2, int(1.5 * s)))
    draw.line([rot_pt(-67 * s, 0), rot_pt(-23 * s, 0)], fill=(96, 165, 250, 200), width=max(2, int(1.5 * s)))
    draw.line([rot_pt(23 * s, 0), rot_pt(67 * s, 0)], fill=(96, 165, 250, 200), width=max(2, int(1.5 * s)))

    # Connecting truss bar
    draw.line([rot_pt(-23 * s, 0), rot_pt(23 * s, 0)], fill=(226, 232, 240, 255), width=max(3, int(4.5 * s)))

    # Directional L-band antenna feed pointing toward Earth
    ant_pt = rot_pt(0, 24 * s)
    ant_r = int(7 * s)
    draw.ellipse(
        [ant_pt[0] - ant_r, ant_pt[1] - ant_r, ant_pt[0] + ant_r, ant_pt[1] + ant_r],
        fill=(56, 189, 248, 255),
        outline=(255, 255, 255, 255),
        width=max(2, int(2 * s)),
    )

    # Central satellite gold foil bus body
    draw_rot_rect(-17 * s, -19 * s, 17 * s, 19 * s, (245, 158, 11, 255), (254, 243, 199, 255), max(2, int(3 * s)))
    draw_rot_rect(-10 * s, -11 * s, 10 * s, 11 * s, (217, 119, 6, 255), (253, 230, 138, 255), max(1, int(1.5 * s)))

    # Downsample with high-quality Lanczos filter
    final_img = img.resize((size, size), resample=Image.Resampling.LANCZOS)
    return final_img


def generate_multi_res_ico(output_path: Path) -> Path:
    """Create and save a multi-resolution Windows .ico file (16..256 px) and PNG logo (`assets/app_logo.png`)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    base_img = draw_satellite_icon(256)

    # Save crisp PNG logo for in-app sidebar branding
    png_path = output_path.parent / "app_logo.png"
    base_img.save(png_path, format="PNG")

    icon_sizes = [(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]
    base_img.save(output_path, format="ICO", sizes=icon_sizes)
    return output_path


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    ico_file = project_root / "assets" / "app.ico"
    generate_multi_res_ico(ico_file)
    png_file = project_root / "assets" / "app_logo.png"
    print(f"[OK] Generated 4x-supersampled icon: {ico_file} ({ico_file.stat().st_size} bytes)")
    print(f"[OK] Generated high-DPI sidebar PNG logo: {png_file} ({png_file.stat().st_size} bytes)")
