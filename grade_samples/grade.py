import sys, os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

LUMA = np.array([0.2126, 0.7152, 0.0722])

def luma(x): return (x * LUMA).sum(-1, keepdims=True)
def sat(x, s):
    y = luma(x); return y + (x - y) * s
def scurve(x, amt, pivot=0.45):
    # smooth contrast around pivot
    return np.clip(pivot + (x - pivot) * (1 + amt) - amt * (x - pivot) ** 3 * 2.2, 0, 1)
def lgg(x, lift, gamma, gain):
    lift, gamma, gain = map(np.array, (lift, gamma, gain))
    x = np.clip(x * gain + lift * (1 - x), 0, 1)
    return x ** (1 / gamma)
def split(x, sh, hi, amt_s, amt_h):
    y = luma(x)
    ws = np.clip(1 - y / 0.5, 0, 1) ** 1.5
    wh = np.clip((y - 0.5) / 0.5, 0, 1) ** 1.2
    return x + ws * amt_s * (np.array(sh) - 0.5) + wh * amt_h * (np.array(hi) - 0.5)
def skin_protect(orig, graded, amt=0.5):
    # pull orange hues back toward a natural-ish tone
    r, g, b = orig[..., 0], orig[..., 1], orig[..., 2]
    m = np.clip((r - b) * 4, 0, 1) * np.clip((r - g) * 6, 0, 1) * np.clip(1 - np.abs(g - (r + b) / 2) * 3, 0, 1)
    m = m[..., None] * amt
    keep = sat(orig, 0.75) * 0.85 + graded * 0.15
    return graded * (1 - m) + keep * m
def matte(x, black=0.04, white=0.96): return black + x * (white - black)

LOOKS = {
 "01_cold_concrete": lambda x: matte(skin_protect(x, sat(split(scurve(lgg(x,(-.03,-.01,.02),(.9,.95,1.0),(.92,.98,1.02)),.6),(.12,.45,.6),(.56,.52,.46),.28,.1),.6),.45),.02,.94),
 "02_bleach_bypass": lambda x: matte(np.clip(0.6*sat(lgg(x,(-.01,0,.015),(1,1,1),(1,1,1)),.45)+0.4*scurve(luma(x).repeat(3,-1)*np.array([.97,1,1.04]),1.0),0,1)**1.12,.015,.96),
 "03_sodium_sick":   lambda x: matte(skin_protect(x, sat(split(scurve(lgg(x,(-.03,-.005,-.02),(.9,.97,.9),(.97,1.0,.9)),.6),(.3,.5,.45),(.57,.56,.42),.22,.1),.6),.45),.02,.94),
 "04_bruise":        lambda x: matte(skin_protect(x, sat(split(scurve(lgg(x,(-.01,-.03,.02),(.93,.9,.98),(.96,.95,1.0)),.6),(.52,.3,.62),(.5,.5,.52),.26,.06),.6),.5),.02,.94),
 "05_ember_dark":    lambda x: matte(sat(split(scurve(lgg(x,(-.035,-.01,.015),(.88,.9,.95),(1.02,.97,.88)),.6),(.18,.42,.5),(.66,.53,.38),.24,.14),.75),.015,.95),
}

def vignette(img, strength=0.45):
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    d = np.sqrt(((xx - w/2)/(w/2))**2 + ((yy - h/2)/(h/2))**2) / np.sqrt(2)
    return img * (1 - strength * np.clip(d*1.25-0.3, 0, 1)[..., None]**1.6)

def write_cube(fn, path, n=33):
    g = np.linspace(0, 1, n)
    b, gg, r = np.meshgrid(g, g, g, indexing="ij")
    rgb = np.stack([r, gg, b], -1).reshape(-1, 3)
    out = np.clip(fn(rgb), 0, 1)
    with open(path, "w") as f:
        f.write(f'TITLE "{os.path.basename(path)[:-5]}"\nLUT_3D_SIZE {n}\n')
        for v in out: f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

if __name__ == "__main__":
    src, outdir = sys.argv[1], sys.argv[2]
    os.makedirs(outdir, exist_ok=True)
    x = np.asarray(Image.open(src).convert("RGB")).astype(np.float32) / 255
    tiles = [("00_original", x)]
    for name, fn in LOOKS.items():
        y = vignette(np.clip(fn(x), 0, 1))
        Image.fromarray((y*255).round().astype(np.uint8)).save(f"{outdir}/{name}.jpg", quality=93)
        write_cube(fn, f"{outdir}/{name}.cube")
        tiles.append((name, y))
    # contact sheet 2x3
    h, w = x.shape[:2]; tw, th = w//2, h//2
    sheet = Image.new("RGB", (tw*2, th*3 + 3*30), (12, 12, 12))
    d = ImageDraw.Draw(sheet)
    try: font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except Exception: font = ImageFont.load_default()
    for i, (name, im) in enumerate(tiles):
        c, r = i % 2, i // 2
        t = Image.fromarray((np.clip(im,0,1)*255).astype(np.uint8)).resize((tw, th), Image.LANCZOS)
        sheet.paste(t, (c*tw, r*(th+30)+30))
        d.text((c*tw+10, r*(th+30)+4), name[3:].replace("_", " ").upper(), fill=(230,230,230), font=font)
    sheet.save(f"{outdir}/contact_sheet.jpg", quality=92)
