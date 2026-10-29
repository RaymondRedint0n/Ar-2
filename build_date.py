"""Собирает сайт AR-фото: вписывает сегодняшнюю дату в табличку и собирает standee.glb в повышенном качестве (2K).

Запуск:  python build_date.py                 -> папка _site с index.html и standee.glb
         DATE=2026-12-31 python build_date.py  -> собрать на конкретную дату (для проверки)
"""
import datetime as dt
import hashlib
import io
import json
import os
import shutil
import struct
from zoneinfo import ZoneInfo
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from scipy import ndimage

TZ = ZoneInfo('Asia/Yekaterinburg')      # дата меняется в полночь по Екатеринбургу
GAP = 0.006                              # половина толщины «картона», м (12 мм между передом и спиной против Z-fighting на iPhone)
ROOT = os.path.dirname(os.path.abspath(__file__))
A = os.path.join(ROOT, 'assets')         # здесь же лежат шрифты
OUT = os.path.join(ROOT, '_site')

RESAMPLE_BICUBIC = getattr(Image, 'Resampling', Image).BICUBIC
RESAMPLE_LANCZOS = getattr(Image, 'Resampling', Image).LANCZOS
TRANSFORM_PERSPECTIVE = getattr(Image, 'Transform', Image).PERSPECTIVE

# Коэффициент повышения разрешения текстур модели (1.5x -> ~2560..2880px по высоте для максимальной чёткости лица и таблички)
UPSCALE = 1.5
MAX_FIG_HEIGHT = 2400

# Девушки календаря. Новая девушка = папка в assets (front.png, back.png, sign.json) + строка здесь.
#   assets — папка с картинками, out — папка на сайте ('' = главная страница),
#   height — высота силуэта от верхней точки (макушка или поднятая рука/мольберт) до подошвы, м
#   live — True, если в папке есть «живое» видео (live.mp4 + live.json)
FIGURES = [
    {'assets': 'assets',            'out': '',           'height': 1.70},
    {'assets': 'assets/balerina',   'out': 'balerina',   'height': 1.98, 'live': True},   # рука поднята, на пуантах; живое видео
    {'assets': 'assets/skripachka', 'out': 'skripachka', 'height': 1.74, 'live': True},   # скрипачка с табличкой; живое видео 2K
    {'assets': 'assets/xydochnica', 'out': 'xydochnica', 'height': 1.76, 'live': True},   # художница у мольберта; живое видео 2K
]

MONTHS = ['января', 'февраля', 'марта', 'апреля', 'мая', 'июня', 'июля',
          'августа', 'сентября', 'октября', 'ноября', 'декабря']
WEEKDAYS = ['понедельник', 'вторник', 'среда', 'четверг', 'пятница', 'суббота', 'воскресенье']
ORANGE = (242, 107, 20); INK = (24, 24, 26); GREY = (120, 120, 124)


def today():
    if os.environ.get('DATE'):
        return dt.date.fromisoformat(os.environ['DATE'])
    return dt.datetime.now(TZ).date()


def font(bold, size):
    return ImageFont.truetype(os.path.join(A, 'font-bold.ttf' if bold else 'font.ttf'), size)


def fit(draw, text, bold, max_w, size):
    """Подбирает кегль, чтобы строка влезла по ширине."""
    while size > 10:
        f = font(bold, size)
        if draw.textlength(text, font=f) <= max_w:
            return f
        size -= 2
    return font(bold, size)


def sign_card(d, W=1200, H=1660):
    """Макет таблички в 2x разрешении (1200x1660): белый лист, тёмный текст, оранжевый акцент."""
    k = W / 600.0
    im = Image.new('RGB', (W, H), 'white'); g = ImageDraw.Draw(im)
    pad = round(46 * k); mw = W - 2 * pad

    def center(text, f, y, fill, spacing=0):
        if spacing:
            w = sum(g.textlength(ch, font=f) for ch in text) + spacing * (len(text) - 1)
            x = (W - w) / 2
            for ch in text:
                g.text((x, y), ch, font=f, fill=fill); x += g.textlength(ch, font=f) + spacing
        else:
            g.text(((W - g.textlength(text, font=f)) / 2, y), text, font=f, fill=fill)

    f_top = font(True, round(40 * k))
    center('СЕГОДНЯ', f_top, round(52 * k), GREY, spacing=round(10 * k))
    g.rectangle([W / 2 - round(60 * k), round(116 * k), W / 2 + round(60 * k), round(124 * k)], fill=ORANGE)

    day = str(d.day)
    f_day = fit(g, day, True, mw, round(360 * k))
    bb = g.textbbox((0, 0), day, font=f_day)
    g.text(((W - (bb[2] - bb[0])) / 2 - bb[0], round(150 * k) - bb[1]), day, font=f_day, fill=INK)
    y = round(150 * k) + (bb[3] - bb[1]) + round(40 * k)

    month = MONTHS[d.month - 1].upper()
    f_m = fit(g, month, True, mw, round(92 * k))
    center(month, f_m, y, ORANGE); y += f_m.size + round(34 * k)

    g.line([pad + round(40 * k), y, W - pad - round(40 * k), y], fill=(205, 205, 208), width=round(4 * k))
    y += round(30 * k)
    wd = WEEKDAYS[d.weekday()]
    f_w = fit(g, wd, False, mw, round(66 * k))
    center(wd, f_w, y, INK); y += f_w.size + round(22 * k)
    center(str(d.year), font(False, round(40 * k)), y, GREY)
    return im


def perspective_coeffs(dst, src):
    """Коэффициенты PIL PERSPECTIVE: точка результата -> точка исходника."""
    M = []; B = []
    for (x, y), (u, v) in zip(dst, src):
        M.append([x, y, 1, 0, 0, 0, -u * x, -u * y]); B.append(u)
        M.append([0, 0, 0, x, y, 1, -v * x, -v * y]); B.append(v)
    return np.linalg.solve(np.array(M, float), np.array(B, float)).tolist()


def enhance_and_decontaminate(im, scale=UPSCALE):
    """Повышает детализацию текстуры (Lanczos + UnsharpMask) и очищает полупрозрачные края от белого ореола."""
    if scale and abs(scale - 1.0) > 1e-3:
        nw, nh = round(im.width * scale), round(im.height * scale)
        rgb_im = im.convert('RGB').resize((nw, nh), RESAMPLE_LANCZOS)
        rgb_im = rgb_im.filter(ImageFilter.UnsharpMask(radius=1.2, percent=60, threshold=3))
        alpha_im = im.split()[3].resize((nw, nh), RESAMPLE_LANCZOS)
    else:
        rgb_im = im.convert('RGB').filter(ImageFilter.UnsharpMask(radius=1.0, percent=45, threshold=3))
        alpha_im = im.split()[3]

    c = np.array(rgb_im, dtype=np.uint8)
    a = np.array(alpha_im, dtype=np.uint8)
    # Заполняем внешние и полупрозрачные краевые пиксели цветом ближайшего плотного пикселя внутри фигуры (убирает светлую кайму)
    core = a > 210
    if np.any(core) and not np.all(core):
        _, (iy, ix) = ndimage.distance_transform_edt(~core, return_indices=True)
        edge_zone = ~core
        c[edge_zone] = c[iy[edge_zone], ix[edge_zone]]
    return Image.fromarray(np.dstack([c, a]), 'RGBA')


def front_with_date(d, meta, folder=A, scale=UPSCALE):
    with Image.open(os.path.join(folder, 'front.png')) as raw_front:
        front = enhance_and_decontaminate(raw_front.convert('RGBA'), scale=scale)
    card = sign_card(d, W=1200, H=1660)
    q = [(x * scale, y * scale) for x, y in meta['front_sign_inner']]
    src = [(0, 0), (card.width, 0), (card.width, card.height), (0, card.height)]
    warped = card.transform(front.size, TRANSFORM_PERSPECTIVE, perspective_coeffs(q, src), RESAMPLE_BICUBIC, fillcolor='white')
    # Суперсэмплинг маски таблички (2x) для гладких краёв рамки
    mw, mh = front.width * 2, front.height * 2
    mask_hi = Image.new('L', (mw, mh), 0)
    ImageDraw.Draw(mask_hi).polygon([(x * 2, y * 2) for x, y in q], fill=255)
    mask = mask_hi.resize(front.size, RESAMPLE_LANCZOS)

    rgb = np.array(front.convert('RGB')).astype(float)
    ink = np.array(warped).astype(float) / 255.0
    # Вписываем дату только по белой бумаге холста/таблички, чтобы рука, кисть или палитра поверх холста оставались нетронутыми
    is_paper = ((rgb.min(axis=2) > 165) & ((rgb.max(axis=2) - rgb.min(axis=2)) < 30)).astype(float)
    paper_w = ndimage.gaussian_filter(is_paper, sigma=0.8)[..., None]
    m = (np.array(mask).astype(float) / 255.0)[..., None] * paper_w
    out = rgb * (1.0 - m) + (rgb * ink) * m   # мягкое смешивание по сглаженной границе таблички
    res = Image.fromarray(out.clip(0, 255).astype('uint8'))
    res.putalpha(front.split()[3])
    return res


def silhouette_info(im, scale=UPSCALE):
    a = np.array(im)[:, :, 3] > 128
    ys, xs = np.nonzero(a)
    top, bot = int(ys.min()), int(ys.max())
    band = max(1, round(240 * scale))
    feet_x = float(np.nonzero(a[max(0, bot - band):bot + 1])[1].mean())
    center_x = 0.5 * float(xs.min() + xs.max())
    return top, bot, feet_x, center_x


def export_standee_glb(front_im, back_im, height_m, out_path, target_center_3d=None):
    """Собирает компактный высококачественный glTF 2.0 (.glb) с идеальным совпадением силуэтов спереди и сзади."""
    ft, fb, ffx, fcx = silhouette_info(front_im)
    bt, bb, _, bcx = silhouette_info(back_im)
    S_f = height_m / (fb - ft)
    S_b = height_m / (bb - bt)

    # Центр переднего силуэта в 3D (по умолчанию относительно стоп ffx, либо синхронизированный с live.mp4)
    if target_center_3d is None:
        target_center_3d = (fcx - ffx) * S_f
    f_origin_x = fcx - target_center_3d / S_f
    # Для задней стороны (зеркальной по X: x_3d = -(x_px - b_origin_x) * S_b) совмещаем центр силуэта с передней стороной:
    b_origin_x = bcx + target_center_3d / S_b

    def quad_vecs(im, S, origin_x, bot, is_front):
        w, h = im.size
        x0 = -origin_x * S; x1 = (w - origin_x) * S
        y0 = -(h - 1 - bot) * S; y1 = y0 + h * S
        z = GAP if is_front else -GAP
        if is_front:
            v = [[x0, y0, z], [x1, y0, z], [x1, y1, z], [x0, y1, z]]
            n = [[0.0, 0.0, 1.0]] * 4
        else:
            v = [[-x0, y0, z], [-x1, y0, z], [-x1, y1, z], [-x0, y1, z]]
            n = [[0.0, 0.0, -1.0]] * 4
        return np.array(v, dtype=np.float32), np.array(n, dtype=np.float32)

    def encode_png(im):
        buf = io.BytesIO()
        im.save(buf, format='PNG', optimize=True)
        return buf.getvalue()

    indices = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)
    uv = np.array([[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    v_front, n_front = quad_vecs(front_im, S_f, f_origin_x, fb, True)
    v_back, n_back = quad_vecs(back_im, S_b, b_origin_x, bb, False)
    png_front = encode_png(front_im)
    png_back = encode_png(back_im)

    chunks = []
    views = []

    def add_view(raw_bytes, target=None):
        offset = sum(len(c) for c in chunks)
        length = len(raw_bytes)
        pad = (4 - (length % 4)) % 4
        chunks.append(raw_bytes + (b'\x00' * pad))
        bv = {'buffer': 0, 'byteOffset': offset, 'byteLength': length}
        if target is not None:
            bv['target'] = target
        views.append(bv)
        return len(views) - 1

    bv_idx = add_view(indices.tobytes(), 34963)
    bv_uv = add_view(uv.tobytes(), 34962)
    bv_vf = add_view(v_front.tobytes(), 34962)
    bv_nf = add_view(n_front.tobytes(), 34962)
    bv_vb = add_view(v_back.tobytes(), 34962)
    bv_nb = add_view(n_back.tobytes(), 34962)
    bv_img_f = add_view(png_front)
    bv_img_b = add_view(png_back)

    bin_blob = b''.join(chunks)

    gltf = {
        'asset': {'version': '2.0', 'generator': 'ar-foto-hq-builder'},
        'scene': 0,
        'scenes': [{'nodes': [0, 1]}],
        'nodes': [
            {'name': 'front', 'mesh': 0},
            {'name': 'back', 'mesh': 1},
        ],
        'meshes': [
            {
                'name': 'front',
                'primitives': [{
                    'attributes': {'POSITION': 1, 'NORMAL': 2, 'TEXCOORD_0': 5},
                    'indices': 0,
                    'mode': 4,
                    'material': 0,
                }],
            },
            {
                'name': 'back',
                'primitives': [{
                    'attributes': {'POSITION': 3, 'NORMAL': 4, 'TEXCOORD_0': 5},
                    'indices': 0,
                    'mode': 4,
                    'material': 1,
                }],
            },
        ],
        'materials': [
            {
                'name': 'front_mat',
                'pbrMetallicRoughness': {
                    'baseColorTexture': {'index': 0},
                    'metallicFactor': 0.0,
                    'roughnessFactor': 1.0,
                },
                'emissiveTexture': {'index': 0},
                'emissiveFactor': [0.22, 0.22, 0.22],
                'alphaMode': 'MASK',
                'alphaCutoff': 0.5,
                'doubleSided': False,
            },
            {
                'name': 'back_mat',
                'pbrMetallicRoughness': {
                    'baseColorTexture': {'index': 1},
                    'metallicFactor': 0.0,
                    'roughnessFactor': 1.0,
                },
                'emissiveTexture': {'index': 1},
                'emissiveFactor': [0.22, 0.22, 0.22],
                'alphaMode': 'MASK',
                'alphaCutoff': 0.5,
                'doubleSided': False,
            },
        ],
        'samplers': [{
            'magFilter': 9729,   # LINEAR
            'minFilter': 9987,   # LINEAR_MIPMAP_LINEAR
            'wrapS': 33071,      # CLAMP_TO_EDGE
            'wrapT': 33071,      # CLAMP_TO_EDGE
        }],
        'textures': [
            {'sampler': 0, 'source': 0},
            {'sampler': 0, 'source': 1},
        ],
        'images': [
            {'bufferView': bv_img_f, 'mimeType': 'image/png'},
            {'bufferView': bv_img_b, 'mimeType': 'image/png'},
        ],
        'accessors': [
            {
                'bufferView': bv_idx,
                'componentType': 5125,
                'count': 6,
                'type': 'SCALAR',
                'min': [0],
                'max': [3],
            },
            {
                'bufferView': bv_vf,
                'componentType': 5126,
                'count': 4,
                'type': 'VEC3',
                'min': v_front.min(axis=0).tolist(),
                'max': v_front.max(axis=0).tolist(),
            },
            {
                'bufferView': bv_nf,
                'componentType': 5126,
                'count': 4,
                'type': 'VEC3',
                'min': [0.0, 0.0, 1.0],
                'max': [0.0, 0.0, 1.0],
            },
            {
                'bufferView': bv_vb,
                'componentType': 5126,
                'count': 4,
                'type': 'VEC3',
                'min': v_back.min(axis=0).tolist(),
                'max': v_back.max(axis=0).tolist(),
            },
            {
                'bufferView': bv_nb,
                'componentType': 5126,
                'count': 4,
                'type': 'VEC3',
                'min': [0.0, 0.0, -1.0],
                'max': [0.0, 0.0, -1.0],
            },
            {
                'bufferView': bv_uv,
                'componentType': 5126,
                'count': 4,
                'type': 'VEC2',
                'min': [0.0, 0.0],
                'max': [1.0, 1.0],
            },
        ],
        'bufferViews': views,
        'buffers': [{'byteLength': len(bin_blob)}],
    }

    json_bytes = json.dumps(gltf, separators=(',', ':')).encode('utf-8')
    json_pad = (4 - (len(json_bytes) % 4)) % 4
    json_chunk = json_bytes + (b' ' * json_pad)

    total_len = 12 + 8 + len(json_chunk) + 8 + len(bin_blob)
    with open(out_path, 'wb') as fh:
        fh.write(struct.pack('<4sII', b'glTF', 2, total_len))
        fh.write(struct.pack('<I4s', len(json_chunk), b'JSON'))
        fh.write(json_chunk)
        fh.write(struct.pack('<I4s', len(bin_blob), b'BIN\x00'))
        fh.write(bin_blob)


def file_md5_tag(path):
    with open(path, 'rb') as fh:
        return hashlib.md5(fh.read()).hexdigest()[:10]


def build_figure(d, cfg):
    folder = os.path.join(ROOT, cfg['assets'])
    if not os.path.exists(os.path.join(folder, 'front.png')):
        return
    out = os.path.join(OUT, cfg['out'])
    root_target = os.path.join(ROOT, cfg['out']) if cfg['out'] else ROOT
    with open(os.path.join(folder, 'sign.json'), encoding='utf-8') as fh:
        meta = json.load(fh)
    front = front_with_date(d, meta, folder, scale=UPSCALE)
    with Image.open(os.path.join(folder, 'back.png')) as raw_back:
        back = enhance_and_decontaminate(raw_back.convert('RGBA'), scale=UPSCALE)
    t, b, _, _ = silhouette_info(front, scale=UPSCALE)
    S = cfg['height'] / (b - t)

    has_live = os.path.exists(os.path.join(folder, 'live.mp4')) and os.path.exists(os.path.join(folder, 'live.json'))
    # Если есть живое видео, синхронизируем 3D-центр мешей standee.glb с силуэтом в live.mp4
    target_center_3d = None
    if has_live:
        with open(os.path.join(folder, 'live.json'), encoding='utf-8') as lfh:
            lm0 = json.load(lfh)
        if 'centerX' in lm0:
            s_live = lm0.get('heightM', cfg['height']) / (lm0['bottom'] - lm0['top'])
            target_center_3d = (lm0['centerX'] - lm0['feetX']) * s_live
        else:
            target_center_3d = 0.0468

    os.makedirs(out, exist_ok=True)
    os.makedirs(root_target, exist_ok=True)
    glb_path = os.path.join(out, 'standee.glb')
    export_standee_glb(front, back, cfg['height'], glb_path, target_center_3d=target_center_3d)
    shutil.copy2(glb_path, os.path.join(root_target, 'standee.glb'))

    # картинки для режима «Фото с фигурой» (камера телефона + фигура поверх) в повышенном разрешении
    for im, name in [(front, 'figure.png'), (back, 'figure-back.png')]:
        cropped = im.crop(im.getbbox())
        if cropped.height > MAX_FIG_HEIGHT:
            cropped = cropped.resize((round(cropped.width * MAX_FIG_HEIGHT / cropped.height), MAX_FIG_HEIGHT), RESAMPLE_LANCZOS)
        out_img = os.path.join(out, name)
        cropped.save(out_img, optimize=True)
        shutil.copy2(out_img, os.path.join(root_target, name))

    with open(os.path.join(ROOT, 'index.html'), encoding='utf-8') as fh:
        page = fh.read()
    if has_live:   # «живое фото»
        live_mp4_src = os.path.join(folder, 'live.mp4')
        shutil.copy2(live_mp4_src, os.path.join(out, 'live.mp4'))
        if os.path.abspath(root_target) != os.path.abspath(folder):
            shutil.copy2(live_mp4_src, os.path.join(root_target, 'live.mp4'))
        with open(os.path.join(folder, 'live.json'), encoding='utf-8') as fh:
            lm = json.load(fh)
        lm['v'] = file_md5_tag(live_mp4_src)      # телефон не возьмёт старое видео из кэша
        for dst_dir in [out, root_target]:
            with open(os.path.join(dst_dir, 'live.json'), 'w', encoding='utf-8') as fh:
                json.dump(lm, fh, separators=(',', ':'))
        try:
            build_live_usdz(d, folder, out, back, S)
            usdz_tag = file_md5_tag(os.path.join(out, 'live.usdz'))
            if os.path.exists(os.path.join(out, 'live.usdz')):
                shutil.copy2(os.path.join(out, 'live.usdz'), os.path.join(root_target, 'live.usdz'))
            # AR от Apple на iPhone открывает живую модель вместо статичной; адрес новый при каждом изменении модели
            page = page.replace('ar ar-modes="quick-look"', f'ios-src="live.usdz?v={usdz_tag}" ar ar-modes="quick-look"', 1)
        except Exception as e:   # без модели для iPhone страница всё равно работает (статичная фигура)
            print('   live.usdz пропущен:', e)
    with open(os.path.join(out, 'index.html'), 'w', encoding='utf-8') as fh:
        fh.write(page)
    if cfg['out']:
        with open(os.path.join(root_target, 'index.html'), 'w', encoding='utf-8') as fh:
            fh.write(page)
    print('  ', cfg['out'] or '(главная)', '->', out, f'({front.size[0]}x{front.size[1]})')


def build_live_usdz(d, folder, out, back, S_static, fps=12, scale=None):
    """Живая фигура для AR от Apple (iPhone):
    - fps=12 кратна и 24fps исходного видео (step=2), и экранам iPhone 60Hz (5x) / 120Hz ProMotion (10x);
    - повышенное разрешение текстур (768x1152 вместо 512x1024) даёт 2x чёткость лица и платья в AR Quick Look;
    - TPS=2400 (0.4 мс на тик) и перекрытие ключей [a, b] на 1 тик полностью исключают мерцание/провал масштаба между кадрами."""
    import subprocess, tempfile
    from pxr import Usd, UsdGeom, UsdShade, UsdUtils, Sdf, Gf, Vt
    with open(os.path.join(folder, 'live.json'), encoding='utf-8') as fh:
        meta = json.load(fh)
    if scale is None:
        scale = min(1.0, 1080 / meta['h'])
    step = max(1, round(meta['fps'] / fps))
    W, H = meta['w'], meta['h']
    tw, th = int(W * scale) // 2 * 2, int(H * scale) // 2 * 2
    tmp = tempfile.mkdtemp(prefix='usdz_')
    subprocess.run(['ffmpeg', '-v', 'error', '-i', os.path.join(folder, 'live.mp4'),
                    '-vf', f"select='not(mod(n\\,{step}))',scale={tw * 2}:{th}:flags=lanczos", '-fps_mode', 'passthrough',
                    os.path.join(tmp, 'v%03d.png')], check=True)
    vframes = sorted(f for f in os.listdir(tmp) if f.startswith('v'))
    pp = meta.get('pingpong')            # видео «вперёд-назад»: текстуры берём только из прямого хода
    if pp:
        vframes = [f for i, f in enumerate(vframes) if i * step < pp]
    n = len(vframes)
    seq = list(range(n)) + (list(range(n - 2, 0, -1)) if pp else [])   # порядок показа кадров
    card = sign_card(d, W=1200, H=1660)
    src = [(0, 0), (card.width, 0), (card.width, card.height), (0, card.height)]
    tex = os.path.join(tmp, 'tex'); os.makedirs(tex)
    TEXW, TEXH = 768, 1152
    k = tw / W
    for i, name in enumerate(vframes):
        with Image.open(os.path.join(tmp, name)) as raw_vf:
            im = np.array(raw_vf.convert('RGB'))
        col = Image.fromarray(im[:, :tw]); al = Image.fromarray(im[:, tw:tw * 2, 0])
        q = [(x * k, y * k) for x, y in meta['sign'][min(i * step, meta['frames'] - 1)]]
        warped = card.transform(col.size, TRANSFORM_PERSPECTIVE, perspective_coeffs(q, src), RESAMPLE_BICUBIC, fillcolor='white')
        mask = Image.new('L', col.size, 0); ImageDraw.Draw(mask).polygon(q, fill=255)
        c = np.array(col).astype(float); ink = np.array(warped).astype(float) / 255
        # Вписываем дату только по белой бумаге холста/таблички, чтобы рука, кисть или палитра поверх холста оставались нетронутыми
        is_paper = ((c.min(axis=2) > 165) & ((c.max(axis=2) - c.min(axis=2)) < 30)).astype(float)
        m_paper = (ndimage.gaussian_filter(is_paper, sigma=0.8) * (np.array(mask) > 0).astype(float))[..., None]
        c = c * (1.0 - m_paper) + (c * ink) * m_paper
        an = np.array(al)
        inside = an > 127
        c = c.clip(0, 255).astype('uint8')
        _, (iy, ix) = ndimage.distance_transform_edt(~inside, return_indices=True)
        c = c[iy, ix]                                   # снаружи — цвет ближайшего края фигуры, без тёмной каймы
        rgba = Image.fromarray(np.dstack([c, np.where(inside, 255, 0).astype('uint8')]), 'RGBA')
        fit_k = min(TEXW / rgba.width, TEXH / rgba.height)
        rgba = rgba.resize((round(rgba.width * fit_k), round(rgba.height * fit_k)), RESAMPLE_LANCZOS)
        a2 = np.array(rgba)[:, :, 3]; rgba.putalpha(Image.fromarray(np.where(a2 > 127, 255, 0).astype('uint8')))
        canvas = Image.new('RGBA', (TEXW, TEXH), (0, 0, 0, 0))
        canvas.paste(rgba, (0, 0))
        cv = np.array(canvas); inside2 = cv[:, :, 3] > 0
        dist_out, (jy, jx) = ndimage.distance_transform_edt(~inside2, return_indices=True)
        # Заливаем цветом края только 3-пиксельную кайму вокруг силуэта (для линейной фильтрации GPU), а дальний фон обнуляем
        cv[:, :, :3] = np.where((dist_out <= 3)[..., None], cv[:, :, :3][jy, jx], 0)
        rgba = Image.fromarray(cv, 'RGBA'); img_w, img_h = round(tw * fit_k), round(th * fit_k)
        rgba.quantize(256, method=Image.Quantize.FASTOCTREE, dither=Image.Dither.NONE) \
            .save(os.path.join(tex, f'f{i:03d}.png'), optimize=True)

    def make(variant):
        usda = os.path.join(tmp, f'{variant}.usdc')
        st = Usd.Stage.CreateNew(usda)
        UsdGeom.SetStageUpAxis(st, UsdGeom.Tokens.y); UsdGeom.SetStageMetersPerUnit(st, 1.0)
        # Высокая частота таймкодов (2400 делений в секунду = 0.4 мс на деление) + перекрытие соседних кадров на границе b:
        # на экранах iPhone 60Hz / 120Hz ProMotion предыдущий кадр держит полный размер (1,1,1) до полного включения следующего.
        TPS = 2400; span = TPS // fps; end = len(seq) * span
        if variant != 'test-static':
            st.SetStartTimeCode(0); st.SetEndTimeCode(end); st.SetTimeCodesPerSecond(TPS); st.SetFramesPerSecond(TPS)
        root = UsdGeom.Xform.Define(st, '/Root'); st.SetDefaultPrim(root.GetPrim())
        UsdGeom.Scope.Define(st, '/Root/Materials')

        def material(name, image_file):
            m = UsdShade.Material.Define(st, f'/Root/Materials/{name}')
            sh = UsdShade.Shader.Define(st, f'/Root/Materials/{name}/PBR'); sh.CreateIdAttr('UsdPreviewSurface')
            sh.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(1.0)
            sh.CreateInput('metallic', Sdf.ValueTypeNames.Float).Set(0.0)
            rd = UsdShade.Shader.Define(st, f'/Root/Materials/{name}/st'); rd.CreateIdAttr('UsdPrimvarReader_float2')
            rd.CreateInput('varname', Sdf.ValueTypeNames.String).Set('st')
            def texture(tname, f):
                t = UsdShade.Shader.Define(st, f'/Root/Materials/{name}/{tname}'); t.CreateIdAttr('UsdUVTexture')
                t.CreateInput('file', Sdf.ValueTypeNames.Asset).Set(f'tex/{f}')
                t.CreateInput('st', Sdf.ValueTypeNames.Float2).ConnectToSource(rd.ConnectableAPI(), 'result')
                t.CreateInput('wrapS', Sdf.ValueTypeNames.Token).Set('clamp'); t.CreateInput('wrapT', Sdf.ValueTypeNames.Token).Set('clamp')
                return t
            tc = texture('img', image_file)
            tc.CreateInput('sourceColorSpace', Sdf.ValueTypeNames.Token).Set('sRGB')
            tc.CreateOutput('rgb', Sdf.ValueTypeNames.Float3); tc.CreateOutput('a', Sdf.ValueTypeNames.Float)
            sh.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).ConnectToSource(tc.ConnectableAPI(), 'rgb')
            sh.CreateInput('opacity', Sdf.ValueTypeNames.Float).ConnectToSource(tc.ConnectableAPI(), 'a')
            sh.CreateInput('opacityThreshold', Sdf.ValueTypeNames.Float).Set(0.5)
            m.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), 'surface')
            return m

        def plane(path, pts, mat):
            mesh = UsdGeom.Mesh.Define(st, path)
            mesh.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*p) for p in pts]))
            mesh.CreateFaceVertexCountsAttr([4]); mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
            mesh.CreateDoubleSidedAttr(False)
            mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
            mesh.CreateNormalsAttr(Vt.Vec3fArray([Gf.Vec3f(0, 0, 1)] * 4)); mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
            mesh.CreateExtentAttr(UsdGeom.Mesh.ComputeExtent(mesh.GetPointsAttr().Get()))
            pv = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar('st', Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
            pv.Set(Vt.Vec2fArray([Gf.Vec2f(0, 0), Gf.Vec2f(1, 0), Gf.Vec2f(1, 1), Gf.Vec2f(0, 1)]))
            UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat)

        S = meta['heightM'] / (meta['bottom'] - meta['top'])
        xl, xr = (0 - meta['feetX']) * S, (W - meta['feetX']) * S
        yt, yb = meta['bottom'] * S, (meta['bottom'] - H) * S
        xr = xl + (xr - xl) * TEXW / img_w
        yb = yt - (yt - yb) * TEXH / img_h
        cx, cy = (xl + xr) / 2, (yt + yb) / 2
        pts = [(xl - cx, yb - cy, 0), (xr - cx, yb - cy, 0), (xr - cx, yt - cy, 0), (xl - cx, yt - cy, 0)]
        UsdGeom.Xform.Define(st, '/Root/Live')
        tiny = Gf.Vec3f(0.0001, 0.0001, 0.0001)
        for i in (range(1) if variant == 'test-static' else range(n)):
            x = UsdGeom.Xform.Define(st, f'/Root/Live/F{i:03d}')
            # Чередуем микро-ступеньку по Z (0.3 мм) между чётными и нечётными кадрами, чтобы на 0.4 мс перекрытии не было Z-fighting
            z_frame = 0.01 + (i % 2) * 0.0003
            x.AddTranslateOp().Set(Gf.Vec3d(cx, cy, z_frame))
            if variant == 'test-static':
                plane(f'/Root/Live/F{i:03d}/Quad', pts, material(f'M{i:03d}', f'f{i:03d}.png'))
                continue
            sop = x.AddScaleOp() if variant == 'live' else None
            one = Gf.Vec3f(1, 1, 1)
            keys = {}
            for pos in [p for p, v in enumerate(seq) if v == i]:   # кадр держит полный размер [a, b] включительно (без провала на границе!)
                a, b = pos * span, (pos + 1) * span
                if a > 0: keys[a - 1] = tiny
                keys[a] = one; keys[b] = one
                if b + 1 <= end: keys[b + 1] = tiny
            keys.setdefault(0, tiny)
            last = max(p for p, v in enumerate(seq) if v == i)
            keys[end] = one if (last + 1) * span == end else tiny
            if variant == 'live':
                for t in sorted(keys):
                    sop.Set(keys[t], t)
            else:
                va = x.CreateVisibilityAttr()
                for t in sorted(keys):
                    va.Set(UsdGeom.Tokens.inherited if keys[t] == one else UsdGeom.Tokens.invisible, t)
            plane(f'/Root/Live/F{i:03d}/Quad', pts, material(f'M{i:03d}', f'f{i:03d}.png'))
        st.GetRootLayer().Save()
        cwd = os.getcwd(); os.chdir(tmp)
        try:
            ok = UsdUtils.CreateNewARKitUsdzPackage(Sdf.AssetPath(f'{variant}.usdc'), f'{variant}.usdz')
        finally:
            os.chdir(cwd)
        if not ok:
            raise RuntimeError('USDZ не собрался')
        shutil.copy(os.path.join(tmp, f'{variant}.usdz'), os.path.join(out, f'{variant}.usdz'))

    make('live')
    shutil.rmtree(tmp, ignore_errors=True)
    print('   live.usdz:', n, 'кадров,', len(seq), 'показов за круг,', os.path.getsize(os.path.join(out, 'live.usdz')) // 1024, 'КБ')


def main():
    d = today()
    print('built for', d.isoformat())
    for cfg in FIGURES:
        build_figure(d, cfg)
    with open(os.path.join(ROOT, 'last_build.txt'), 'w', encoding='utf-8') as fh:
        fh.write(d.isoformat() + '\n')


if __name__ == '__main__':
    main()