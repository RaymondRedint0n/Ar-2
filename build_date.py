"""Собирает сайт AR-фото: вписывает сегодняшнюю дату в табличку и собирает standee.glb.

Запуск:  python build_date.py                 -> папка _site с index.html и standee.glb
         DATE=2026-12-31 python build_date.py  -> собрать на конкретную дату (для проверки)
"""
import datetime as dt
import hashlib
import json
import os
import shutil
from zoneinfo import ZoneInfo
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import trimesh
from trimesh.visual import TextureVisuals
from trimesh.visual.material import PBRMaterial

TZ = ZoneInfo('Asia/Yekaterinburg')      # дата меняется в полночь по Екатеринбургу
GAP = 0.004                              # половина толщины «картона», м
ROOT = os.path.dirname(os.path.abspath(__file__))
A = os.path.join(ROOT, 'assets')         # здесь же лежат шрифты
OUT = os.path.join(ROOT, '_site')

RESAMPLE_BICUBIC = getattr(Image, 'Resampling', Image).BICUBIC
RESAMPLE_LANCZOS = getattr(Image, 'Resampling', Image).LANCZOS
TRANSFORM_PERSPECTIVE = getattr(Image, 'Transform', Image).PERSPECTIVE

# Девушки календаря. Новая девушка = папка в assets (front.png, back.png, sign.json) + строка здесь.
#   assets — папка с картинками, out — папка на сайте ('' = главная страница),
#   height — высота силуэта от верхней точки (макушка или поднятая рука) до подошвы, м
#   live — True, если в папке есть «живое» видео (live.mp4 + live.json)
FIGURES = [
    {'assets': 'assets',          'out': '',         'height': 1.70},
    {'assets': 'assets/balerina', 'out': 'balerina', 'height': 1.98, 'live': True},   # рука поднята, на пуантах; живое видео
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


def sign_card(d, W=600, H=830):
    """Макет таблички: белый лист, тёмный текст, оранжевый акцент."""
    im = Image.new('RGB', (W, H), 'white'); g = ImageDraw.Draw(im)
    pad = 46; mw = W - 2 * pad

    def center(text, f, y, fill, spacing=0):
        if spacing:
            w = sum(g.textlength(ch, font=f) for ch in text) + spacing * (len(text) - 1)
            x = (W - w) / 2
            for ch in text:
                g.text((x, y), ch, font=f, fill=fill); x += g.textlength(ch, font=f) + spacing
        else:
            g.text(((W - g.textlength(text, font=f)) / 2, y), text, font=f, fill=fill)

    f_top = font(True, 40)
    center('СЕГОДНЯ', f_top, 52, GREY, spacing=10)
    g.rectangle([W / 2 - 60, 116, W / 2 + 60, 124], fill=ORANGE)

    day = str(d.day)
    f_day = fit(g, day, True, mw, 360)
    bb = g.textbbox((0, 0), day, font=f_day)
    g.text(((W - (bb[2] - bb[0])) / 2 - bb[0], 150 - bb[1]), day, font=f_day, fill=INK)
    y = 150 + (bb[3] - bb[1]) + 40

    month = MONTHS[d.month - 1].upper()
    f_m = fit(g, month, True, mw, 92)
    center(month, f_m, y, ORANGE); y += f_m.size + 34

    g.line([pad + 40, y, W - pad - 40, y], fill=(205, 205, 208), width=4); y += 30
    wd = WEEKDAYS[d.weekday()]
    f_w = fit(g, wd, False, mw, 66)
    center(wd, f_w, y, INK); y += f_w.size + 22
    center(str(d.year), font(False, 40), y, GREY)
    return im


def perspective_coeffs(dst, src):
    """Коэффициенты PIL PERSPECTIVE: точка результата -> точка исходника."""
    M = []; B = []
    for (x, y), (u, v) in zip(dst, src):
        M.append([x, y, 1, 0, 0, 0, -u * x, -u * y]); B.append(u)
        M.append([0, 0, 0, x, y, 1, -v * x, -v * y]); B.append(v)
    return np.linalg.solve(np.array(M, float), np.array(B, float)).tolist()


def front_with_date(d, meta, folder=A):
    with Image.open(os.path.join(folder, 'front.png')) as raw_front:
        front = raw_front.convert('RGBA')
    card = sign_card(d)
    q = meta['front_sign_inner']
    src = [(0, 0), (card.width, 0), (card.width, card.height), (0, card.height)]
    warped = card.transform(front.size, TRANSFORM_PERSPECTIVE, perspective_coeffs(q, src), RESAMPLE_BICUBIC, fillcolor='white')
    mask = Image.new('L', front.size, 0); ImageDraw.Draw(mask).polygon([tuple(p) for p in q], fill=255)
    rgb = np.array(front.convert('RGB')).astype(float)
    ink = np.array(warped).astype(float) / 255
    m = (np.array(mask) > 0)[..., None]
    out = np.where(m, rgb * ink, rgb)        # «умножение»: свет и тень на листе сохраняются
    res = Image.fromarray(out.clip(0, 255).astype('uint8')); res.putalpha(front.split()[3])
    return res


def silhouette_info(im):
    a = np.array(im)[:, :, 3] > 128
    ys, xs = np.nonzero(a)
    top, bot = ys.min(), ys.max()
    feet_x = np.nonzero(a[bot - 240:bot + 1])[1].mean()
    return top, bot, feet_x


def quad(im, S, front):
    w, h = im.size
    top, bot, fx = silhouette_info(im)
    x0 = -fx * S; x1 = (w - fx) * S
    y0 = -(h - 1 - bot) * S; y1 = y0 + h * S
    z = GAP if front else -GAP
    if front:
        v = [[x0, y0, z], [x1, y0, z], [x1, y1, z], [x0, y1, z]]
    else:   # вид сзади: левый край картинки — слева у зрителя за спиной фигуры
        v = [[-x0, y0, z], [-x1, y0, z], [-x1, y1, z], [-x0, y1, z]]
    f = [[0, 1, 2], [0, 2, 3]]
    uv = [[0, 0], [1, 0], [1, 1], [0, 1]]
    mat = PBRMaterial(baseColorTexture=im, metallicFactor=0.0, roughnessFactor=1.0,
                      alphaMode='MASK', alphaCutoff=0.5, doubleSided=False,
                      emissiveTexture=im.convert('RGB'), emissiveFactor=[0.22] * 3)
    return trimesh.Trimesh(np.array(v, float), f, visual=TextureVisuals(uv=np.array(uv, float), material=mat), process=False)


def file_md5_tag(path):
    with open(path, 'rb') as fh:
        return hashlib.md5(fh.read()).hexdigest()[:10]


def build_figure(d, cfg):
    folder = os.path.join(ROOT, cfg['assets'])
    out = os.path.join(OUT, cfg['out'])
    with open(os.path.join(folder, 'sign.json'), encoding='utf-8') as fh:
        meta = json.load(fh)
    front = front_with_date(d, meta, folder)
    with Image.open(os.path.join(folder, 'back.png')) as raw_back:
        back = raw_back.convert('RGBA')
    t, b, _ = silhouette_info(front)
    S = cfg['height'] / (b - t)
    sc = trimesh.Scene()
    sc.add_geometry(quad(front, S, True), node_name='front', geom_name='front')
    sc.add_geometry(quad(back, S, False), node_name='back', geom_name='back')
    os.makedirs(out, exist_ok=True)
    glb_path = os.path.join(out, 'standee.glb')
    sc.export(glb_path)
    # картинки для режима «Фото с фигурой» (камера телефона + фигура поверх)
    for im, name in [(front, 'figure.png'), (back, 'figure-back.png')]:
        cropped = im.crop(im.getbbox())
        if cropped.height > 1600:
            cropped = cropped.resize((round(cropped.width * 1600 / cropped.height), 1600), RESAMPLE_LANCZOS)
        out_img = os.path.join(out, name)
        cropped.save(out_img, optimize=True)
        if not cfg['out']:
            shutil.copy2(out_img, os.path.join(ROOT, name))
    if not cfg['out']:
        shutil.copy2(glb_path, os.path.join(ROOT, 'standee.glb'))

    with open(os.path.join(ROOT, 'index.html'), encoding='utf-8') as fh:
        page = fh.read()
    if cfg.get('live') and os.path.exists(os.path.join(folder, 'live.mp4')):   # «живое фото»
        live_mp4_src = os.path.join(folder, 'live.mp4')
        shutil.copy(live_mp4_src, os.path.join(out, 'live.mp4'))
        with open(os.path.join(folder, 'live.json'), encoding='utf-8') as fh:
            lm = json.load(fh)
        lm['v'] = file_md5_tag(live_mp4_src)      # телефон не возьмёт старое видео из кэша
        with open(os.path.join(out, 'live.json'), 'w', encoding='utf-8') as fh:
            json.dump(lm, fh, separators=(',', ':'))
        try:
            build_live_usdz(d, folder, out, back, S)
            usdz_tag = file_md5_tag(os.path.join(out, 'live.usdz'))
            # AR от Apple на iPhone открывает живую модель вместо статичной; адрес новый при каждом изменении модели
            page = page.replace('ar ar-modes="quick-look"', f'ios-src="live.usdz?v={usdz_tag}" ar ar-modes="quick-look"', 1)
        except Exception as e:   # без модели для iPhone страница всё равно работает (статичная фигура)
            print('   live.usdz пропущен:', e)
    with open(os.path.join(out, 'index.html'), 'w', encoding='utf-8') as fh:
        fh.write(page)
    print('  ', cfg['out'] or '(главная)', '->', out)


def build_live_usdz(d, folder, out, back, S_static, fps=12, scale=None):
    """Живая фигура для AR от Apple (iPhone): кадры видео — плоскости, которые показываются по очереди.
    Apple не проигрывает видео внутри модели, но проигрывает анимацию — получается «мультфильм» по кругу."""
    import subprocess, tempfile
    from scipy import ndimage
    from pxr import Usd, UsdGeom, UsdShade, UsdUtils, Sdf, Gf, Vt
    with open(os.path.join(folder, 'live.json'), encoding='utf-8') as fh:
        meta = json.load(fh)
    if scale is None:   # текстура ~640 px по высоте, сколько бы ни занимала фигура в кадре видео
        scale = min(1.0, 640 / meta['h'])
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
    card = sign_card(d)
    src = [(0, 0), (card.width, 0), (card.width, card.height), (0, card.height)]
    tex = os.path.join(tmp, 'tex'); os.makedirs(tex)
    TEXW, TEXH = 512, 1024
    k = tw / W
    for i, name in enumerate(vframes):
        with Image.open(os.path.join(tmp, name)) as raw_vf:
            im = np.array(raw_vf.convert('RGB'))
        col = Image.fromarray(im[:, :tw]); al = Image.fromarray(im[:, tw:tw * 2, 0])
        q = [(x * k, y * k) for x, y in meta['sign'][min(i * step, meta['frames'] - 1)]]
        warped = card.transform(col.size, TRANSFORM_PERSPECTIVE, perspective_coeffs(q, src), RESAMPLE_BICUBIC, fillcolor='white')
        mask = Image.new('L', col.size, 0); ImageDraw.Draw(mask).polygon(q, fill=255)
        c = np.array(col).astype(float); ink = np.array(warped).astype(float) / 255
        c = np.where((np.array(mask) > 0)[..., None], c * ink, c)
        # одна картинка на кадр: цвет + прозрачность вместе (с отдельными картинками iPhone сдвигал прозрачность)
        an = np.array(al)
        inside = an > 127
        c = c.clip(0, 255).astype('uint8')
        _, (iy, ix) = ndimage.distance_transform_edt(~inside, return_indices=True)
        c = c[iy, ix]                                   # снаружи — цвет ближайшего края фигуры, без тёмной каймы
        rgba = Image.fromarray(np.dstack([c, np.where(inside, 255, 0).astype('uint8')]), 'RGBA')
        # размер картинки — степень двойки (512×1024): iPhone обрезал картинки с «неровной» шириной справа
        fit_k = min(TEXW / rgba.width, TEXH / rgba.height)
        rgba = rgba.resize((round(rgba.width * fit_k), round(rgba.height * fit_k)), RESAMPLE_LANCZOS)
        a2 = np.array(rgba)[:, :, 3]; rgba.putalpha(Image.fromarray(np.where(a2 > 127, 255, 0).astype('uint8')))
        canvas = Image.new('RGBA', (TEXW, TEXH), (0, 0, 0, 0))
        canvas.paste(rgba, (0, 0))
        cv = np.array(canvas); inside2 = cv[:, :, 3] > 0
        _, (jy, jx) = ndimage.distance_transform_edt(~inside2, return_indices=True)
        cv[:, :, :3] = cv[:, :, :3][jy, jx]          # поля тоже залиты цветом края (прозрачные)
        rgba = Image.fromarray(cv, 'RGBA'); img_w, img_h = round(tw * fit_k), round(th * fit_k)
        rgba.quantize(256, method=Image.Quantize.FASTOCTREE, dither=Image.Dither.FLOYDSTEINBERG) \
            .save(os.path.join(tex, f'f{i:03d}.png'), optimize=True)

    def make(variant):
        usda = os.path.join(tmp, f'{variant}.usdc')
        st = Usd.Stage.CreateNew(usda)
        UsdGeom.SetStageUpAxis(st, UsdGeom.Tokens.y); UsdGeom.SetStageMetersPerUnit(st, 1.0)
        # Apple сглаживает анимацию между ключами — поэтому шкала времени мелкая (240 делений в секунду),
        # а кадр включается и выключается за одно деление (~4 мс): плавных «переходов» между кадрами нет.
        TPS = 240; span = TPS // fps; end = len(seq) * span
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
            tc = texture('img', image_file)                 # цвет и прозрачность из одной картинки
            tc.CreateInput('sourceColorSpace', Sdf.ValueTypeNames.Token).Set('sRGB')
            tc.CreateOutput('rgb', Sdf.ValueTypeNames.Float3); tc.CreateOutput('a', Sdf.ValueTypeNames.Float)
            sh.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).ConnectToSource(tc.ConnectableAPI(), 'rgb')
            sh.CreateInput('opacity', Sdf.ValueTypeNames.Float).ConnectToSource(tc.ConnectableAPI(), 'a')
            sh.CreateInput('opacityThreshold', Sdf.ValueTypeNames.Float).Set(0.5)   # чёткое вырезание по маске
            m.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), 'surface')
            return m

        def plane(path, pts, mat):
            mesh = UsdGeom.Mesh.Define(st, path)
            mesh.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*p) for p in pts]))
            mesh.CreateFaceVertexCountsAttr([4]); mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
            mesh.CreateDoubleSidedAttr(False)
            # без сглаживания: по умолчанию iPhone скругляет плоскость почти в овал и срезает углы (табличку)
            mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
            mesh.CreateNormalsAttr(Vt.Vec3fArray([Gf.Vec3f(0, 0, 1)] * 4)); mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
            mesh.CreateExtentAttr(UsdGeom.Mesh.ComputeExtent(mesh.GetPointsAttr().Get()))
            pv = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar('st', Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
            pv.Set(Vt.Vec2fArray([Gf.Vec2f(0, 0), Gf.Vec2f(1, 0), Gf.Vec2f(1, 1), Gf.Vec2f(0, 1)]))
            UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat)

        # перед: кадры видео (масштаб — как в проигрывателе на странице)
        S = meta['heightM'] / (meta['bottom'] - meta['top'])
        xl, xr = (0 - meta['feetX']) * S, (W - meta['feetX']) * S
        yt, yb = meta['bottom'] * S, (meta['bottom'] - H) * S
        # картинка кадра занимает левый верхний угол текстуры 512×1024 — плоскость расширяем на поля
        xr = xl + (xr - xl) * TEXW / img_w
        yb = yt - (yt - yb) * TEXH / img_h
        cx, cy, z = (xl + xr) / 2, (yt + yb) / 2, 0.01
        pts = [(xl - cx, yb - cy, 0), (xr - cx, yb - cy, 0), (xr - cx, yt - cy, 0), (xl - cx, yt - cy, 0)]
        UsdGeom.Xform.Define(st, '/Root/Live')
        tiny = Gf.Vec3f(0.0001, 0.0001, 0.0001)
        for i in (range(1) if variant == 'test-static' else range(n)):
            x = UsdGeom.Xform.Define(st, f'/Root/Live/F{i:03d}')
            x.AddTranslateOp().Set(Gf.Vec3d(cx, cy, z))
            if variant == 'test-static':
                plane(f'/Root/Live/F{i:03d}/Quad', pts, material(f'M{i:03d}', f'f{i:03d}.png'))
                continue
            sop = x.AddScaleOp() if variant == 'live' else None
            one = Gf.Vec3f(1, 1, 1)
            keys = {}
            for pos in [p for p, v in enumerate(seq) if v == i]:   # кадр виден на [a, b) — один или два раза за круг
                a, b = pos * span, (pos + 1) * span
                if a > 0: keys[a - 1] = tiny
                keys[a] = one; keys[b - 1] = one
                if b < end: keys[b] = tiny
            keys.setdefault(0, tiny)
            last = max(p for p, v in enumerate(seq) if v == i)
            keys[end] = one if (last + 1) * span == end else tiny   # последний показанный кадр держится до конца круга
            if variant == 'live':
                for t in sorted(keys):
                    sop.Set(keys[t], t)
            else:   # test-vis: кадры переключаются атрибутом видимости
                va = x.CreateVisibilityAttr()
                for t in sorted(keys):
                    va.Set(UsdGeom.Tokens.inherited if keys[t] == one else UsdGeom.Tokens.invisible, t)
            plane(f'/Root/Live/F{i:03d}/Quad', pts, material(f'M{i:03d}', f'f{i:03d}.png'))
        # задней стороны в живой модели нет: на iPhone она перекрывала переднюю (обрез таблички и руки, дрожание)
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
