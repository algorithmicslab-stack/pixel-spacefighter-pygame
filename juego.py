# =====================================================================
#  PIXEL SPACEFIGHTER (versión Pygame) — juego de plataformas y peleas en 3 sectores.
#  Copia de juego.html pasada a Python + Pygame.
#  El piloto se lee de assets/piloto.json (exportado del creador web); si no hay, se usa una figura simple.
#  Música y efectos se sintetizan con numpy (sin archivos de audio).
# =====================================================================
import asyncio, json, math, os, random, base64, io, sys
WEB = sys.platform == 'emscripten'  # corriendo en el navegador (pygbag)
import numpy as np
import pygame

HERE = os.path.dirname(os.path.abspath(__file__))
def P(*parts): return os.path.join(HERE, *parts)

VW, VH, GROUND, TILE = 480, 270, 222, 16
SAVE = P('partida.json')

pygame.mixer.pre_init(22050, -16, 1, 512)
pygame.init()
info = pygame.display.Info()
S = 2 if WEB else max(1, min(int(info.current_w * 0.9 // VW), int(info.current_h * 0.88 // VH)))  # escala entera
screen = pygame.display.set_mode((VW * S, VH * S))
pygame.display.set_caption('Pixel Spacefighter')
ctx = pygame.Surface((VW, VH))
clock = pygame.time.Clock()

# ---------- Utilidades ----------
class O:
    """Objeto con atributos libres; un atributo que no existe vale None (como undefined en JS)."""
    def __init__(self, **k): self.__dict__.update(k)
    def __getattr__(self, n): return None

def R(v): return int(math.floor(v + 0.5))  # Math.round
def rnd(a, b): return a + random.random() * (b - a)
def clamp(v, a, b): return max(a, min(b, v))
def sign(v): return (v > 0) - (v < 0)
def overlap(a, b): return a.x < b.x + b.w and a.x + a.w > b.x and a.y < b.y + b.h and a.y + a.h > b.y

def mulberry(seed):
    st = [seed & 0xffffffff]
    M = 0xffffffff
    def r():
        st[0] = (st[0] + 0x6D2B79F5) & M
        t = st[0]
        t = ((t ^ (t >> 15)) * (1 | t)) & M
        t = ((t + (((t ^ (t >> 7)) * (61 | t)) & M)) & M) ^ t
        return ((t ^ (t >> 14)) & M) / 4294967296
    return r

_colors = {}
def C(c):
    if isinstance(c, tuple): return c if len(c) == 4 else (*c, 1.0)
    if c in _colors: return _colors[c]
    if c.startswith('#'): v = (int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16), 1.0)
    else:
        n = [float(x) for x in c[c.index('(') + 1:-1].split(',')]
        v = (int(n[0]), int(n[1]), int(n[2]), n[3] if len(n) > 3 else 1.0)
    _colors[c] = v
    return v

# estado de dibujo: transparencia global y desplazamiento (temblor de cámara)
ALPHA = 1.0
OX = OY = 0

def rect(x, y, w, h, c):
    r, g, b, a = C(c); a *= ALPHA
    x, y, w, h = R(x), R(y), R(w), R(h)
    if w < 0: x += w; w = -w
    if h < 0: y += h; h = -h
    if w <= 0 or h <= 0 or a <= 0: return
    if a >= 0.999: ctx.fill((r, g, b), (x + OX, y + OY, w, h))
    else:
        s = pygame.Surface((w, h)); s.fill((r, g, b)); s.set_alpha(int(a * 255)); ctx.blit(s, (x + OX, y + OY))

def _rows(rows, c):
    r, g, b, a = C(c); a *= ALPHA
    if a <= 0 or not rows: return
    if a >= 0.999:
        for x, y, w in rows: ctx.fill((r, g, b), (x + OX, y + OY, w, 1))
        return
    x0 = min(q[0] for q in rows); y0 = min(q[1] for q in rows)
    x1 = max(q[0] + q[2] for q in rows); y1 = max(q[1] for q in rows) + 1
    s = pygame.Surface((x1 - x0, y1 - y0), pygame.SRCALPHA)
    for x, y, w in rows: s.fill((r, g, b, 255), (x - x0, y - y0, w, 1))
    s.set_alpha(int(a * 255)); ctx.blit(s, (x0 + OX, y0 + OY))

# círculo de pixeles (sin antialias)
def pcircle(x, y, r, c):
    rows = []; dy = -r
    while dy <= r:
        w = R(math.sqrt(max(0, r * r - dy * dy))); rows.append((R(x - w), R(y + dy), w * 2 + 1)); dy += 1
    _rows(rows, c)

def pellipse(x, y, rx, ry, c):
    rows = []; dy = -ry
    while dy <= ry:
        w = R(rx * math.sqrt(max(0, 1 - (dy * dy) / (ry * ry)))); rows.append((R(x - w), R(y + dy), w * 2 + 1)); dy += 1
    _rows(rows, c)

# ---------- Imágenes ----------
def load_img(path):
    try: return pygame.image.load(P('assets', *path.split('/'))).convert_alpha()
    except Exception: return None

def ok(i): return i is not None

def _color_filter(surf, hue=0, bright=1.0, gray=False):
    s = surf.copy()
    arr = pygame.surfarray.pixels3d(s)
    f = arr.astype(np.float32)
    if gray:
        l = f[..., 0] * 0.2126 + f[..., 1] * 0.7152 + f[..., 2] * 0.0722
        f = np.stack([l, l, l], -1)
    if hue:
        a = math.radians(hue); c, sn = math.cos(a), math.sin(a)
        m = np.array([
            [0.213 + c * 0.787 - sn * 0.213, 0.715 - c * 0.715 - sn * 0.715, 0.072 - c * 0.072 + sn * 0.928],
            [0.213 - c * 0.213 + sn * 0.143, 0.715 + c * 0.285 + sn * 0.140, 0.072 - c * 0.072 - sn * 0.283],
            [0.213 - c * 0.213 - sn * 0.787, 0.715 - c * 0.715 + sn * 0.715, 0.072 + c * 0.928 + sn * 0.072]], np.float32)
        f = f @ m.T
    f *= bright
    arr[:] = np.clip(f, 0, 255).astype(np.uint8)
    del arr
    return s

_tcache = {}
def transformed(im, area=None, w=None, h=None, flip=False, hue=0, bright=1.0, gray=False):
    key = (id(im), area, w, h, flip, hue, bright, gray)
    s = _tcache.get(key)
    if s is None:
        s = im.subsurface(area) if area else im
        if w is not None and (w, h) != s.get_size(): s = pygame.transform.scale(s, (max(1, w), max(1, h)))
        if flip: s = pygame.transform.flip(s, True, False)
        if hue or bright != 1.0 or gray: s = _color_filter(s, hue, bright, gray)
        _tcache[key] = s
    return s

def draw_img(im, x, y, w=None, h=None, area=None, flip=False, hue=0, bright=1.0):
    if im is None: return
    if w is not None: w, h = R(w), R(h)
    s = transformed(im, area, w, h, flip, hue, bright)
    if ALPHA < 0.999:
        s = s.copy(); s.set_alpha(int(ALPHA * 255))
    ctx.blit(s, (R(x) + OX, R(y) + OY))

def draw_local(im, ox, lx, y, d=1, sc=1.0, **kw):
    """Equivale a translate(ox, 0); scale(d*sc, sc); drawImage(im, lx, ...) del canvas."""
    if im is None: return
    w, h = im.get_width() * sc, im.get_height() * sc
    left = ox + lx * sc if d > 0 else ox - (lx * sc + w)
    draw_img(im, left, y, w, h, flip=d < 0, **kw)

# ---------- Texto nítido (se dibuja al final, en la resolución de la ventana) ----------
_fonts = {}
def font(px):
    px = max(4, int(px))
    if px not in _fonts:
        f = pygame.font.SysFont('segoeui', px, bold=True)
        _fonts[px] = f
    return _fonts[px]

_SUBST = {'►': '>', '▶': '>', '▼': 'v'}
def safe(s, f):
    out = []
    for ch in s:
        m = f.metrics(ch)
        out.append(_SUBST.get(ch, '?') if (not m or m[0] is None) and ch in _SUBST else ch)
    return ''.join(out)

texts = []
def text(s, x, y, size=8, color='#e8ebff', align='left'):
    texts.append((s, x + OX, y + OY, size, color, align))

def measure(s, size=8):
    f = font(size * S); return f.size(safe(s, f))[0] / S

_tsurf = {}
def flush_text():
    for s, x, y, size, color, align in texts:
        f = font(size * S); s = safe(s, f)
        key = (s, size, color)
        pair = _tsurf.get(key)
        if pair is None:
            r, g, b, a = C(color)
            pair = (f.render(s, True, (0, 0, 0)), f.render(s, True, (r, g, b)))
            pair[0].set_alpha(217)
            if len(_tsurf) > 600: _tsurf.clear()
            _tsurf[key] = pair
        w = pair[1].get_width()
        px = x * S - (w / 2 if align == 'center' else w if align == 'right' else 0)
        # el texto de Segoe UI trae espacio arriba; se compensa un poco para parecerse al canvas
        py = y * S - size * S * 0.18
        screen.blit(pair[0], (px + 0.6 * S, py + 0.6 * S)); screen.blit(pair[1], (px, py))
    texts.clear()

# ---------- Sonido (sintetizado con numpy, sin archivos) ----------
SR = 22050
AUDIO = pygame.mixer.get_init() is not None
MCH = pygame.mixer.get_init()[2] if AUDIO else 1
if AUDIO:
    pygame.mixer.set_num_channels(32); pygame.mixer.set_reserved(2)
muted = False

def _wave(kind, ph):
    if kind == 'square': return np.where((ph % 1) < 0.5, 1.0, -1.0)
    if kind == 'sawtooth': return 2 * (ph % 1) - 1
    if kind == 'triangle': return 2 * np.abs(2 * (ph % 1) - 1) - 1
    return np.sin(2 * np.pi * ph)

def _osc(kind, f0, dur, f1=None, total=None):
    n = max(1, int((total or dur) * SR)); t = np.arange(n) / SR
    if f1 is None: freq = np.full(n, float(f0))
    else: freq = f0 * (f1 / f0) ** (np.minimum(t, dur) / dur)
    return _wave(kind, np.cumsum(freq) / SR)

def make_sound(buf):
    if not AUDIO: return None
    a = (np.clip(buf * 0.9, -1, 1) * 32767).astype(np.int16)
    if MCH > 1: a = np.repeat(a[:, None], MCH, 1)
    return pygame.sndarray.make_sound(np.ascontiguousarray(a))

def beep_buf(f, d=0.08, kind='square', v=0.05, slide=0):
    w = _osc(kind, f, d, max(20, f + slide) if slide else None)
    n = len(w)
    return w * v * (0.0001 / v) ** (np.arange(n) / n)

_beeps = {}
def beep(f, d=0.08, kind='square', v=0.05, slide=0):
    if muted or not AUDIO: return
    key = (round(f), d, kind, v, slide)
    if key not in _beeps: _beeps[key] = make_sound(beep_buf(f, d, kind, v, slide))
    _beeps[key].play()

def _pick_buf():
    a = beep_buf(660, 0.08, 'triangle', 0.05); b = beep_buf(990, 0.1, 'triangle', 0.05)
    off = int(0.07 * SR); out = np.zeros(off + len(b)); out[:len(a)] += a; out[off:] += b
    return out

_sfx_bufs = {
    'shoot': lambda: beep_buf(880, 0.06, 'square', 0.03, -400),
    'slash': lambda: beep_buf(300, 0.1, 'sawtooth', 0.04, 500),
    'jump': lambda: beep_buf(300, 0.12, 'square', 0.03, 300),
    'hit': lambda: beep_buf(160, 0.08, 'square', 0.04, -80),
    'hurt': lambda: beep_buf(120, 0.25, 'sawtooth', 0.06, -60),
    'kill': lambda: beep_buf(500, 0.15, 'triangle', 0.05, -350),
    'pick': _pick_buf,
    'boom': lambda: beep_buf(80, 0.5, 'sawtooth', 0.08, -50),
    'enemyShot': lambda: beep_buf(420, 0.07, 'triangle', 0.025, -150),
}
_sfx = {k: make_sound(fn()) for k, fn in _sfx_bufs.items()} if AUDIO else {}
class _SFX:
    def __getattr__(self, n):
        def play():
            if not muted and _sfx.get(n): _sfx[n].play()
        return play
SFX = _SFX()
_typing = [make_sound(beep_buf(500 + i * 50, 0.02, 'square', 0.01)) for i in range(7)] if AUDIO else []

# ---------- Música chiptune: cada canción se sintetiza una vez y se repite en bucle ----------
MINOR, HARM = [0, 2, 3, 5, 7, 8, 10], [0, 2, 3, 5, 7, 8, 11]
SONGS = {
    'lobby': dict(bpm=92, root=57, scale=MINOR, chords=[0, 5, 2, 6], wave='triangle', arpWave='square', arpVol=0.018,
                  bass='x...x...x.o.x...', arp='x.x.x.x.x.x.x.x.', kick='x.......x.......', snare='....x.......x...', hat='..x...x...x...x.',
                  lead='7---9-8-7---5---4---5-7-8-------7---9-a-b---a-9-8---7-------....', leadVol=0.035),
    'sector0': dict(bpm=122, root=52, scale=MINOR, chords=[0, 5, 6, 4], wave='square', arpWave='square', arpVol=0.016,
                    bass='x.xox.xox.xox.xo', arp='xxxxxxxxxxxxxxxx', kick='x...x...x...x...', snare='....x.......x...', hat='x.x.x.x.x.x.x.x.',
                    lead='7.7.9.a-9.7.5---4.5.7.8-7.5.4---7.7.9.a-c.a.9---8.7.5.4-5-------', leadVol=0.03),
    'sector1': dict(bpm=108, root=50, scale=HARM, chords=[0, 5, 3, 4], wave='triangle', arpWave='triangle', arpVol=0.03,
                    bass='x..x..x.x..x..o.', arp='x.xx.xx.x.xx.xx.', kick='x.....x...x.....', snare='....x.......x..x', hat='xxx.xxx.xxx.xxx.',
                    lead='9---8---7-6-7---4---------------9---a---b-a-9---6-----------....', leadVol=0.032),
    'sector2': dict(bpm=136, root=48, scale=MINOR, chords=[0, 5, 6, 4], wave='sawtooth', arpWave='square', arpVol=0.015,
                    bass='xoxoxoxoxoxoxoxo', arp='xxxxxxxxxxxxxxxx', kick='x..x..x.x..x..x.', snare='....x.......x...', hat='.x.x.x.x.x.x.x.x',
                    lead='7-7-a-9-7-5-4-5-7---------------7-7-a-9-c-a-9-8-7-------7-8-9-a-', leadVol=0.026),
    'boss': dict(bpm=156, root=45, scale=HARM, chords=[0, 0, 5, 4], wave='sawtooth', arpWave='square', arpVol=0.016,
                 bass='xxoxxxoxxxoxxoxo', arp='xxxxxxxxxxxxxxxx', kick='x.x.x.x.x.x.x.x.', snare='....x..x....x.xx', hat='xxxxxxxxxxxxxxxx',
                 lead='7-6-7-9-a---9-7-6-7-4---------..7-6-7-9-a-c-d-c-a-9-7---6---7---', leadVol=0.03),
}
JINGLES = {'clear': dict(notes=[64, 67, 71, 76, 74, 76, 79, 83], step=0.11, wave='square'),
           'over': dict(notes=[64, 63, 62, 61, 60, 55], step=0.22, wave='triangle')}

def midi(n): return 440 * 2 ** ((n - 69) / 12)

def _tone(buf, at, f, dur, wave, vol, slide=0):
    i0 = int(at * SR)
    w = _osc(wave, f, dur, max(30, f * slide) if slide else None, total=dur + 0.02)
    n = len(w); att = max(1, int(0.005 * SR)); dn = max(att + 1, int(dur * SR))
    env = np.full(n, 0.0001)
    env[:att] = np.linspace(0.0001, vol, att)
    k = min(n, dn) - att
    env[att:att + k] = vol * (0.0001 / vol) ** (np.arange(k) / max(1, dn - att))
    buf[i0:i0 + n] += (w * env)[:len(buf) - i0]

_noise = np.random.uniform(-1, 1, SR // 2)
_hat_noise = np.diff(_noise, prepend=0) * 0.6
_sn_noise = np.convolve(_noise, np.ones(3) / 3, 'same') - np.convolve(_noise, np.ones(14) / 14, 'same')
def _hit(buf, at, dur, vol, hp):
    i0 = int(at * SR); n = int(dur * SR)
    src = (_hat_noise if hp else _sn_noise * 1.6)[:n]
    buf[i0:i0 + n] += (src * vol * (0.0001 / vol) ** (np.arange(n) / n))[:len(buf) - i0]

def render_song(Sg):
    L = Sg['lead']; period = math.lcm(16 * len(Sg['chords']), len(L))
    ln = 60 / Sg['bpm'] / 4; total = int(period * ln * SR); tail = 3 * SR
    buf = np.zeros(total + tail)
    deg = lambda i: Sg['scale'][i % 7] + 12 * (i // 7)
    for step in range(period):
        t = step * ln; st = step % 16; ch = Sg['chords'][(step // 16) % len(Sg['chords'])]
        b = Sg['bass'][st]
        if b != '.': _tone(buf, t, midi(Sg['root'] - 12 + deg(ch) + (12 if b == 'o' else 0)), ln * 0.9, 'sawtooth' if Sg['wave'] == 'sawtooth' else 'triangle', 0.07)
        if Sg['arp'][st] == 'x':
            tri = [ch, ch + 2, ch + 4, ch + 7][step % 4]; _tone(buf, t, midi(Sg['root'] + 12 + deg(tri)), ln * 0.7, Sg['arpWave'], Sg['arpVol'])
        if Sg['kick'][st] == 'x': _tone(buf, t, 150, 0.16, 'sine', 0.22, 0.25)
        if Sg['snare'][st] == 'x': _hit(buf, t, 0.12, 0.09, False)
        if Sg['hat'][st] == 'x': _hit(buf, t, 0.03, 0.035, True)
        li = step % len(L); c = L[li]
        if c not in '.-':
            n = 1
            while L[(li + n) % len(L)] == '-' and n < 16: n += 1
            _tone(buf, t, midi(Sg['root'] + 12 + deg(int(c, 36))), ln * n * 0.95, Sg['wave'], Sg['leadVol'])
    out = buf[:total].copy(); out[:tail] += buf[total:total + tail]  # la cola vuelve al inicio para que el bucle no se corte
    return out

class _Music:
    def __init__(self):
        self.cur = None; self.songs = {}; self.jingles = {}; self.side = 0
        if not AUDIO: return
        for k, Sg in SONGS.items(): self.songs[k] = make_sound(render_song(Sg))
        for k, J in JINGLES.items():
            n = len(J['notes']); buf = np.zeros(int((n * J['step'] + J['step'] * 4 + 0.3) * SR))
            for i, note in enumerate(J['notes']): _tone(buf, 0.1 + i * J['step'], midi(note), J['step'] * (4 if i == n - 1 else 1.1), J['wave'], 0.06)
            self.jingles[k] = make_sound(buf)
    def _chan(self): return pygame.mixer.Channel(self.side)
    def play(self, name):
        if not AUDIO or self.cur == name: return
        self._chan().fadeout(500)
        self.cur = name
        if not name or name not in self.songs: return
        self.side ^= 1; ch = self._chan()
        ch.set_volume(0 if muted else 1); ch.play(self.songs[name], loops=-1, fade_ms=600)
    def jingle(self, name):
        self.play(None)
        if not AUDIO: return
        self.side ^= 1; ch = self._chan(); ch.set_volume(0 if muted else 1); ch.play(self.jingles[name])
    def set_muted(self, m):
        if AUDIO:
            for i in (0, 1): pygame.mixer.Channel(i).set_volume(0 if m else 1)
Music = _Music()

# ---------- Guardado (récord y tutorial) ----------
def load_save():
    try:
        with open(SAVE, encoding='utf-8') as f: return json.load(f)
    except Exception: return {}
def write_save():
    try:
        with open(SAVE, 'w', encoding='utf-8') as f: json.dump({'hiscore': hiscore, 'tuto': tutoDone}, f)
    except Exception: pass

# ---------- Datos del piloto (exportados del creador web a assets/piloto.json) ----------
sheet = None; sheetImg = None; pilot = {'colors': {}, 'parts': {}}
try:
    with open(P('assets', 'piloto.json'), encoding='utf-8') as f: _pj = json.load(f)
    sheet = _pj.get('sheet'); pilot = _pj.get('character') or pilot
    if sheet:
        try:  # PNG normal (funciona también en el navegador con pygbag)
            sheetImg = pygame.image.load(P('assets', 'piloto.png')).convert_alpha()
        except Exception:
            sheetImg = pygame.image.load(io.BytesIO(base64.b64decode(sheet['src'].split(',', 1)[1])), 'piloto.png').convert_alpha()
except Exception as err:
    print('No se pudo cargar el piloto:', err)
    sheet = None
ACCENT = pilot.get('colors', {}).get('accent') or '#ff5c5c'
VISOR = pilot.get('colors', {}).get('visor') or '#3fd8ff'
WEAPON = pilot.get('parts', {}).get('weapon', 1)  # 0 nada, 1 bláster, 2 espada, 3 escudo
if WEAPON is None: WEAPON = 1
MELEE = WEAPON == 2

def sheet_frame(row, f):
    return (f * sheet['frameWidth'], row * sheet['frameHeight'], sheet['frameWidth'], sheet['frameHeight'])

# ---------- Entrada ----------
keys, pressed = {}, {}
KEYMAP = {pygame.K_LEFT: 'left', pygame.K_a: 'left', pygame.K_RIGHT: 'right', pygame.K_d: 'right',
          pygame.K_UP: 'jump', pygame.K_w: 'jump', pygame.K_SPACE: 'jump', pygame.K_z: 'jump',
          pygame.K_j: 'shoot', pygame.K_x: 'shoot', pygame.K_k: 'shoot',
          pygame.K_RETURN: 'start', pygame.K_KP_ENTER: 'start', pygame.K_p: 'pause', pygame.K_ESCAPE: 'pause',
          pygame.K_m: 'mute', pygame.K_DOWN: 'down', pygame.K_s: 'down', pygame.K_t: 'skip'}
mouse = O(x=VW / 2, y=VH / 2, down=False)

# ---------- Tiles del piso ----------
def makeTileThemed(top, Pc):
    c = pygame.Surface((TILE, TILE), pygame.SRCALPHA)
    f = lambda col, r: c.fill(C(col)[:3], r)
    f(Pc['base'], (0, 0, 16, 16))
    f(Pc['dark'], (0, 15, 16, 1)); f(Pc['dark'], (15, 0, 1, 16))
    f(Pc['mid'], (1, 1, 14, 1)); f(Pc['mid'], (1, 1, 1, 14))
    for r in [(3, 3, 2, 2), (11, 11, 2, 2), (11, 3, 2, 2), (3, 11, 2, 2)]: f(Pc['dot'], r)
    if top:
        f(Pc['top'], (0, 0, 16, 3)); f(Pc['hi'], (0, 0, 16, 1)); f(Pc['glow'], (6, 1, 4, 1))
    return c
_BASE_TILE = dict(base='#232a55', dark='#1a1f40', mid='#2f3870', dot='#3e478a', top='#6a74c9', hi='#9aa3ff', glow='#5cf2ff')
TILE_TOP, TILE_IN = makeTileThemed(True, _BASE_TILE), makeTileThemed(False, _BASE_TILE)

# ---------- Fondos espaciales con parallax (port de assets/fondos.js) ----------
def radial(w, h, cx, cy, r, c0, c1):
    yy, xx = np.mgrid[0:h, 0:w]
    t = np.clip(np.hypot(xx - cx, yy - cy) / r, 0, 1)[..., None]
    a0, a1 = np.array(C(c0), np.float32), np.array(C(c1), np.float32)
    a0[3] *= 255; a1[3] *= 255
    px = (a0 * (1 - t) + a1 * t).astype(np.uint8)
    s = pygame.Surface((w, h), pygame.SRCALPHA)
    rgb = pygame.surfarray.pixels3d(s); rgb[:] = px[..., :3].transpose(1, 0, 2); del rgb
    al = pygame.surfarray.pixels_alpha(s); al[:] = px[..., 3].T; del al
    return s

class _Fondos:
    def __init__(self):
        L = lambda src, speed, **o: dict(src='fondos/' + src, speed=speed, **o)
        STARS = [L('parallax-largo/Parallax60.png', 10), L('parallax-largo/BackdropBlackLittleSparkTransparent.png', 24)]
        self.list = {
            'estrellas': dict(procedural=True, layers=[]),
            'planetas': dict(color='#1b0f22', layers=[
                L('parallax-capas/parallax-space-backgound.png', 3, fit=True),
                L('parallax-capas/parallax-space-stars.png', 8, fit=True),
                L('parallax-capas/parallax-space-far-planets.png', 14, fit=True),
                L('parallax-capas/parallax-space-ring-planet.png', 24, single=True, x=0.72, y=0.12, fit=True, ref=160),
                L('parallax-capas/parallax-space-big-planet.png', 36, single=True, x=0.22, y=0.4, fit=True, ref=160)]),
            'espacio': dict(color='#000000', layers=[L('parallax-largo/Parallax100.png', 3), *[dict(s) for s in STARS]]),
            'espacioRojo': dict(color='#000000', layers=[L('parallax-largo/Parallax80.png', 3), *[dict(s) for s in STARS]]),
            'nebulosaAzul': dict(layers=[L('nebulosas/512x512_blue_nebula_2.png', 3), *[dict(s) for s in STARS]]),
            'nebulosaVerde': dict(layers=[L('nebulosas/512x512_green_nebula_1.png', 3), *[dict(s) for s in STARS]]),
            'nebulosaMorada': dict(layers=[L('nebulosas/512x512_purple_nebula_1.png', 3), *[dict(s) for s in STARS]]),
        }
        self.loaded = {}; self.stars = None; self.grad = None
    def load(self, key):
        bg = self.list.get(key)
        if not bg or key in self.loaded: return
        for l in bg['layers']: l['img'] = load_img(l['src'])
        self.loaded[key] = all(l['img'] for l in bg['layers'])
    def procedural(self, t, w, h):
        if self.stars is None:
            self.stars = [dict(x=random.random() * w, y=random.random() * h, v=12 + random.random() * 70, s=2 if random.random() < 0.2 else 1) for _ in range(round(w * h / 1200))]
            self.grad = radial(w, h, w * 0.3, h * 0.2, max(w, h) * 0.8, '#23285a', '#0b0d1a')
        ctx.blit(self.grad, (0, 0))
        for st in self.stars:
            x = (st['x'] - t * st['v']) % w
            v = int(255 * min(1, 0.3 + st['v'] / 140))
            ctx.fill((v, v, v), (int(x), int(st['y']), st['s'], st['s']))
    def draw(self, key, t, w, h):
        bg = self.list.get(key) or self.list['estrellas']
        if bg.get('procedural') or not self.loaded.get(key): self.procedural(t, w, h); return
        ctx.fill(C(bg.get('color', '#000000'))[:3])
        for l in bg['layers']:
            im = l['img']
            sc = h / (l.get('ref') or im.get_height()) if l.get('fit') else 1
            iw, ih = im.get_width() * sc, im.get_height() * sc
            if l.get('single'):
                span = w + iw
                x = ((l['x'] * w - t * l['speed']) % span) - iw
                draw_img(im, x, l['y'] * h, iw, ih); continue
            ox = -((t * l['speed']) % iw)
            y = 0.0
            while y < h:
                x = ox
                while x < w:
                    draw_img(im, x, y, math.ceil(iw), math.ceil(ih)); x += iw
                y += ih
FONDOS = _Fondos()

# ---------- Niveles ----------
LEVELS = [
    O(name='SECTOR 1 · CINTURÓN DE PLANETAS', bg='planetas', len=2600, seed=7, enemies=0.55, boss=0),
    O(name='SECTOR 2 · NEBULOSA MORADA', bg='nebulosaMorada', len=3200, seed=21, enemies=0.75, boss=1),
    O(name='SECTOR 3 · NÚCLEO ROJO', bg='espacioRojo', len=3800, seed=99, enemies=0.95, boss=2),
]
ARENA = 480

level = solids = player = cam = boss = hub = None
enemies, bullets, ebullets, pickups, parts, fx, timers = [], [], [], [], [], [], []
levelIdx = 0
state, stateT, score, levelScore, shake, gtime, flashT = 'title', 0.0, 0, 0, 0.0, 0.0, 0.0
pausedFrom = 'play'
_save = load_save()
tutoDone = bool(_save.get('tuto')); hiscore = int(_save.get('hiscore') or 0)
fade = O(a=0.0, dir=0, cb=None)

def buildLevel(n):
    global level, solids, enemies, bullets, ebullets, pickups, parts, boss, fx
    level = LEVELS[n]
    r = mulberry(level.seed)
    solids = []; enemies = []; bullets = []; ebullets = []; pickups = []; parts = []; boss = None
    groundSegs = [(0, 420)]; x = 420
    while x < level.len:
        gap = 36 + r() * (28 + n * 10)
        x += gap
        # el último tramo se recorta, pero nunca a menos de 140 px
        w = max(140, min(180 + r() * 320, level.len - x + 40))
        groundSegs.append((x, w)); x += w
    arenaX = max(x, level.len) + 40
    groundSegs.append((arenaX - 40, ARENA + 80))
    level.arenaX = arenaX; level.end = arenaX + ARENA; level.camMax = arenaX; fx = []
    for sx, w in groundSegs: solids.append(O(x=sx, y=GROUND, w=w, h=VH - GROUND + 20, ground=True))
    solids.append(O(x=level.end, y=0, w=40, h=VH, wall=True))
    solids.append(O(x=-40, y=0, w=40, h=VH, wall=True))
    # plataformas flotantes, enemigos y objetos
    for sx, w in groundSegs[1:-1]:
        if r() < 0.7:
            pw = 48 + math.floor(r() * 4) * 16; px = sx + r() * max(10, w - pw); py = GROUND - 50 - math.floor(r() * 3) * 18
            solids.append(O(x=R(px), y=py, w=pw, h=12, plat=True))
            if r() < 0.45: pickups.append(O(x=px + pw / 2 - 5, y=py - 16, w=10, h=10, type='vida' if r() < 0.6 else 'poder', t=r() * 6))
            if r() < level.enemies * 0.5: enemies.append(makeEnemy('slime' if r() < 0.5 else 'crab', px + pw / 2, py))
        count = math.floor(w / 180 * level.enemies + r() * 0.8)
        for _ in range(count):
            ex = sx + 30 + r() * (w - 60); t = r()
            enemies.append(makeEnemy('slime', ex, GROUND) if t < 0.45 else makeEnemy('crab', ex, GROUND) if t < 0.75 else makeEnemy('drone', ex, GROUND - 70 - r() * 50))
    if n > 0: pickups.append(O(x=260, y=GROUND - 16, w=10, h=10, type='poder', t=0))
    FONDOS.load(level.bg)
    buildDecor(n)

# ---------- Jugador ----------
def resetPlayer(full):
    global player, cam
    hp = 5 if full or not player else player.hp
    player = O(x=40, y=GROUND - 28, w=12, h=28, vx=0, vy=0, dir=1, ground=False, coyote=0, buffer=0,
               jumps=0, hp=hp, inv=0, cool=0, power=0, anim='quieto', frame=0, ft=0, land=0, safeX=40, slash=0, dead=False, drawFrame=0)
    cam = O(x=0, lock=False)

def startLevel(n, full=True):
    global levelIdx, levelScore, state, stateT
    levelIdx = n; buildLevel(n); resetPlayer(full); levelScore = score
    state = 'intro'; stateT = 0

# ---------- Enemigos ----------
ENEMY_T = dict(slime=dict(w=16, h=11, hp=2, score=100, speed=28), crab=dict(w=18, h=12, hp=3, score=150, speed=50),
               drone=dict(w=14, h=14, hp=2, score=200, speed=40), target=dict(w=25, h=23, hp=2, score=50, speed=0))
def makeEnemy(type_, x, groundY):
    T = ENEMY_T[type_]
    return O(type=type_, x=x - T['w'] / 2, y=groundY - T['h'], w=T['w'], h=T['h'], vx=(-1 if random.random() < 0.5 else 1) * T['speed'], vy=0,
             hp=T['hp'] + levelIdx // 2, score=T['score'], speed=T['speed'], t=random.random() * 10, hit=0,
             homeY=groundY - T['h'], cool=1 + random.random() * 2, active=False, ground=False)

def solidAt(x, y): return any(x >= s.x and x < s.x + s.w and y >= s.y and y < s.y + s.h for s in solids)

def moveBody(b, dt, bounce=False):
    b.x += b.vx * dt
    for s in solids:
        if s.plat or not overlap(b, s): continue
        if b.vx > 0: b.x = s.x - b.w
        elif b.vx < 0: b.x = s.x + s.w
        b.hitWall = True
        if bounce: b.vx = -b.vx
    prevBottom = b.y + b.h
    b.y += b.vy * dt; b.ground = False
    for s in solids:
        if not overlap(b, s): continue
        if s.plat:  # plataformas: solo se pisan desde arriba
            if b.vy >= 0 and prevBottom <= s.y + 2 and not (b is player and keys.get('down')):
                b.y = s.y - b.h; b.vy = 0; b.ground = True
            continue
        if b.vy > 0: b.y = s.y - b.h; b.vy = 0; b.ground = True
        elif b.vy < 0: b.y = s.y + s.h; b.vy = 0

def hurtPlayer(dmg, fromX):
    global shake, flashT, state, stateT
    if player.inv > 0 or player.dead or state not in ('play', 'hub'): return
    if level.hub:
        player.inv = 0.8; player.vx = (-1 if player.x + player.w / 2 < fromX else 1) * 160; player.vy = -160; SFX.hurt(); return
    player.hp -= dmg; player.inv = 1.4; shake = 6; flashT = 0.15; SFX.hurt()
    player.vx = (-1 if player.x + player.w / 2 < fromX else 1) * 160; player.vy = -180
    burst(player.x + 6, player.y + 14, '#ff5c5c', 10)
    if player.hp <= 0:
        player.dead = True; state = 'gameover'; stateT = 0; SFX.boom(); Music.jingle('over'); burst(player.x + 6, player.y + 14, VISOR, 30)

def damageEnemy(e, dmg, kx):
    global score
    e.hp -= dmg; e.hit = 0.12; SFX.hit()
    e.x += kx * 4
    burst(e.x + e.w / 2, e.y + e.h / 2, '#ffffff', 4)
    if e.hp <= 0:
        e.dead = True; score += e.score; SFX.kill(); explode(e.x + e.w / 2, e.y + e.h / 2, 0.7 if e.type == 'target' else 0.6)
        burst(e.x + e.w / 2, e.y + e.h / 2, '#7dff6a' if e.type == 'slime' else '#ff9a3c' if e.type == 'crab' else '#ff4d6d', 14)
        if not level.hub and random.random() < 0.12:
            pickups.append(O(x=e.x + e.w / 2 - 5, y=e.y, w=10, h=10, type='vida' if random.random() < 0.6 else 'poder', t=0, vy=-120))

def updateEnemy(e, dt):
    e.t += dt
    if e.hit > 0: e.hit -= dt
    if e.type == 'target': return  # blanco de práctica
    px, py = player.x + player.w / 2, player.y + player.h / 2
    dx = px - (e.x + e.w / 2); dist = abs(dx)
    if not e.active:
        if abs(e.x - cam.x - VW / 2) < VW: e.active = True
        else: return
    if e.type in ('slime', 'crab'):
        e.vy += 900 * dt
        ahead = e.x + e.w + 2 if e.vx > 0 else e.x - 2
        if e.ground and not solidAt(ahead, e.y + e.h + 3): e.vx = -e.vx
        e.hitWall = False
        if e.type == 'crab' and dist < 140 and e.ground: e.vx = sign(dx) * e.speed * 1.5
        if e.type == 'slime' and not e.noHop and e.ground and e.t > 2.2: e.t = 0; e.vy = -220
        moveBody(e, dt)
        if e.hitWall: e.vx = -e.vx
        if e.y > VH + 40: e.dead = True
    elif e.type == 'drone':
        tx = sign(dx) * e.speed if dist < 260 else math.sin(e.t) * e.speed
        e.vx += (tx - e.vx) * dt * 2
        e.x += e.vx * dt; e.y = e.homeY + math.sin(e.t * 2) * 10
        e.cool -= dt
        if e.cool <= 0 and dist < 240:
            e.cool = 2.4 - levelIdx * 0.4
            a = math.atan2(py - (e.y + 7), px - (e.x + 7))
            ebullets.append(O(x=e.x + 5, y=e.y + 5, w=4, h=4, vx=math.cos(a) * 110, vy=math.sin(a) * 110, color='#ff4d6d', life=4))
            SFX.enemyShot()
    # contacto con el jugador (pisotón = daño al enemigo, excepto cangrejo con púas)
    if overlap(player, e) and not player.dead:
        stomp = player.vy > 60 and player.y + player.h - e.y < 10
        if stomp and e.type != 'crab': damageEnemy(e, 2, 0); player.vy = -260; player.jumps = 1; SFX.jump()
        else: hurtPlayer(1, e.x + e.w / 2)

def drawEnemy(e):
    global ALPHA
    x, y, flash = R(e.x - cam.x), R(e.y), e.hit > 0
    W = '#ffffff' if flash else None
    lookR = player.x > e.x
    if e.type == 'target':
        if ok(CITY['turret']):
            if flash: ALPHA = 0.5
            draw_img(CITY['turret'], x, y); ALPHA = 1
        rect(x + 6, y - 5, 13, 2, '#3a1020'); rect(x + 6, y - 5, 13 * e.hp / 2, 2, '#7dff6a'); return
    if e.type == 'slime':
        sq = math.sin(e.t * 8) * 1 if e.ground else -1.5
        cx, cy = x + 8, y + 7
        pellipse(cx, cy + 1, 9 - sq, 6 + sq, '#0d0f1c')
        pellipse(cx, cy + 1, 8 - sq, 5 + sq, W or '#3fbf4a')
        pellipse(cx - 2, cy - 1, 4, 2, W or '#7dff6a')
        rect(cx - 4, cy, 3, 3, '#ffffff'); rect(cx + 2, cy, 3, 3, '#ffffff')
        rect(cx - 3 + (1 if lookR else 0), cy + 1, 1, 2, '#000000'); rect(cx + 3 + (1 if lookR else 0), cy + 1, 1, 2, '#000000')
        rect(x + 1, y + 10, 14, 1, 'rgba(0,0,0,.3)')
    elif e.type == 'crab':
        leg = math.floor(e.t * 10) % 2
        for i in range(3): rect(x + 2 + i * 6, y + 9 + ((i + leg) % 2), 2, 3, '#7a3a12')
        pellipse(x + 9, y + 7, 9, 4, '#0d0f1c')
        pellipse(x + 9, y + 7, 8, 3, W or '#ff9a3c')
        for i in range(4): rect(x + 3 + i * 4, y + 1, 2, 3, W or '#ffe14d'); rect(x + 3 + i * 4, y, 1, 1, '#ffffff')
        rect(x + (12 if lookR else 3), y + 5, 3, 2, '#ffffff'); rect(x + (14 if lookR else 3), y + 5, 1, 2, '#000000')
    else:
        blink = math.floor(e.t * 12) % 2
        rect(x + 2, y - 2, 10, 1, '#9aa3b8'); rect(x + 6, y - 1, 2, 2, '#9aa3b8')
        rect(x + 1 + blink * 6, y - 3, 5, 1, '#c8d0e0')
        pcircle(x + 7, y + 7, 7, '#0d0f1c')
        pcircle(x + 7, y + 7, 6, W or '#5a6380')
        pcircle(x + 7, y + 7, 4, '#1b1b2a')
        ex = clamp((player.x - e.x) / 40, -1.5, 1.5)
        pcircle(x + 7 + ex, y + 7, 2, '#ff4d6d')
        rect(x + 6 + ex, y + 6, 1, 1, '#ffffff')
        rect(x + 5, y + 14, 4, 1 + blink, '#ff8a1f')

# ---------- Jefes ----------
BOSSES = [dict(name='OJO MADRE', hp=45, color='#8a4dff'), dict(name='MECHA-CANGREJO', hp=70, color='#ff9a3c'),
          dict(name='CEREBRO SUPREMO', hp=100, color='#ff6ad5')]
def spawnBoss():
    global boss
    B = BOSSES[level.boss]; ax = level.arenaX
    boss = O(type=level.boss, name=B['name'], hp=B['hp'], max=B['hp'], color=B['color'], t=0, hit=0, cool=2, cool2=5,
             x=ax + ARENA - 120, y=70, w=56, h=56, vx=0, vy=0, phase=1, dead=False, intro=2, ang=0, ground=False)
    if boss.type == 1: boss.__dict__.update(w=72, h=40, y=GROUND - 40)
    beep(110, 0.6, 'sawtooth', 0.06, 200)

def bossShoot(x, y, a, sp, color='#ff4d6d', size=5):
    ebullets.append(O(x=x - size / 2, y=y - size / 2, w=size, h=size, vx=math.cos(a) * sp, vy=math.sin(a) * sp, color=color, life=6))

def later(sec, fn): timers.append([sec, fn])  # equivalente a setTimeout

def updateBoss(dt):
    global shake
    b = boss; b.t += dt
    if b.hit > 0: b.hit -= dt
    if b.dead:
        b.deadT = (b.deadT or 0) + dt
        if random.random() < 0.4: burst(b.x + rnd(0, b.w), b.y + rnd(0, b.h), '#ffe14d' if random.random() < 0.5 else b.color, 6); shake = 4
        if random.random() < 0.15: explode(b.x + rnd(0, b.w), b.y + rnd(0, b.h), rnd(0.6, 1.2))
        if random.random() < 0.08: SFX.boom()
        return
    if b.intro > 0: b.intro -= dt; return
    if b.hp < b.max / 2 and b.phase == 1: b.phase = 2; shake = 10; beep(90, 0.5, 'sawtooth', 0.06, 150)
    fast = 1.5 if b.phase == 2 else 1
    cx, cy = b.x + b.w / 2, b.y + b.h / 2
    pcx, pcy = player.x + player.w / 2, player.y + player.h / 2
    aim = math.atan2(pcy - cy, pcx - cx)
    b.cool -= dt * fast; b.cool2 -= dt * fast
    ax = level.arenaX
    if b.type == 0:  # OJO MADRE: flota, abanicos de disparos, invoca slimes
        b.x = ax + ARENA / 2 - b.w / 2 + math.sin(b.t * 0.6) * 150
        b.y = 100 + math.sin(b.t * 1.1) * 55
        if b.cool <= 0:
            b.cool = 1.7
            for i in range(-2, 3): bossShoot(cx, cy, aim + i * 0.22, 100, '#c58cff')
            SFX.enemyShot()
        if b.cool2 <= 0:
            b.cool2 = 6
            for i in range(2):
                s = makeEnemy('slime', cx + (30 if i else -30), b.y + b.h); s.vy = -100; s.active = True; enemies.append(s)
    elif b.type == 1:  # MECHA-CANGREJO: camina, salta y suelta ondas, dispara tenazas
        b.vy += 900 * dt
        if b.ground and b.cool2 <= 0: b.cool2 = 3.5; b.vy = -380; b.vx = clamp((pcx - cx) * 1.1, -220, 220); b.jumping = True
        elif b.ground and not b.jumping: b.vx = sign(pcx - cx) * 40 * fast
        wasAir = not b.ground
        moveBody(b, dt)
        if b.ground and wasAir and b.jumping:
            b.jumping = False; b.vx = 0; shake = 8; SFX.boom()
            for d in (-1, 1): ebullets.append(O(x=cx - 4, y=GROUND - 10, w=8, h=10, vx=d * 150, vy=0, color='#ffe14d', life=3, wave=True))
        if b.cool <= 0 and b.ground: b.cool = 1.4; bossShoot(cx + sign(pcx - cx) * 30, b.y + 14, aim, 150, '#ff9a3c', 6); SFX.enemyShot()
    else:  # CEREBRO SUPREMO: espirales, ráfagas y drones
        b.x = ax + ARENA / 2 - b.w / 2 + math.sin(b.t * 0.5) * 140
        b.y = 95 + math.sin(b.t * 0.9) * 58
        b.ang += dt * 1.8 * fast
        if b.cool <= 0:
            b.cool = 0.35; n = 4 if b.phase == 2 else 3
            for i in range(n): bossShoot(cx, cy, b.ang + i * math.pi * 2 / n, 85, '#ff6ad5', 4)
        if b.cool2 <= 0:
            b.cool2 = 5
            for i in range(3):
                def shot(cx=cx, cy=cy):
                    if boss and not boss.dead: bossShoot(cx, cy, math.atan2(player.y - cy, player.x - cx), 170, '#ffffff', 6); SFX.enemyShot()
                later(i * 0.18, shot)
            if len([e for e in enemies if not e.dead and e.type == 'drone']) < 3:
                d = makeEnemy('drone', cx, cy + 40); d.active = True; enemies.append(d)
    if overlap(player, b): hurtPlayer(1, cx)

def drawBoss():
    b = boss; x, y = R(b.x - cam.x), R(b.y); fl = '#ffffff' if b.hit > 0 else None
    cx, cy = x + b.w / 2, y + b.h / 2
    look = math.atan2(player.y + 14 - (b.y + b.h / 2), player.x + 6 - (b.x + b.w / 2))
    if b.intro > 0 and math.floor(b.t * 10) % 2: return
    if b.type == 0:
        for i in range(6):  # tentáculos
            tx, ty = cx - 20 + i * 8, cy + 18
            for s in range(8):
                tx += math.sin(b.t * 3 + i + s * 0.6) * 1.6; ty += 3; rect(tx - 1, ty, 3 - s / 4, 3, '#5b2aa8' if s % 2 else '#7a3fd6')
        pcircle(cx, cy, 29, '#0d0f1c'); pcircle(cx, cy, 28, fl or b.color); pcircle(cx - 8, cy - 10, 10, fl or '#a874ff')
        pcircle(cx, cy, 17, '#f4f0ff')
        ix, iy = cx + math.cos(look) * 7, cy + math.sin(look) * 7
        pcircle(ix, iy, 8, '#ff2a2a' if b.phase == 2 else '#d6283c'); pcircle(ix, iy, 4, '#000000'); rect(ix - 3, iy - 4, 2, 2, '#ffffff')
        rect(cx - 17, cy - 18, 35, 34 if math.sin(b.t * 0.7) > 0.97 else 2, fl or b.color)  # párpado
    elif b.type == 1:
        leg = math.floor(b.t * 8) % 2
        for i in range(4): rect(x + 6 + i * 18, y + 30 + ((i + leg) % 2) * 2, 4, 10, '#7a3a12')
        pellipse(cx, cy + 2, 37, 17, '#0d0f1c'); pellipse(cx, cy + 2, 36, 16, fl or b.color); pellipse(cx - 8, cy - 4, 20, 6, fl or '#ffc07a')
        for i in range(6): rect(x + 12 + i * 9, y - 2, 4, 6, fl or '#ffe14d')
        dr = 1 if player.x > b.x + b.w / 2 else -1
        for d in (-1, 1):  # tenazas
            clx, cly = cx + d * 40, cy + 2 + math.sin(b.t * 4 + d) * 3
            pcircle(clx, cly, 9, '#0d0f1c'); pcircle(clx, cly, 8, fl or '#ff7a1a')
            rect(clx + d * 2, cly - 2, 9 * d, 4, '#0d0f1c')
        for d in (-10, 10):
            rect(cx + d, y - 12, 2, 10, '#7a3a12'); pcircle(cx + d + 1, y - 13, 4, '#ffffff')
            pcircle(cx + d + 1 + dr * 2, y - 13, 2, '#ff2a2a' if b.phase == 2 else '#000000')
    else:
        for i in range(5):
            tx, ty = cx - 18 + i * 9, cy + 16
            for s in range(10):
                tx += math.sin(b.t * 2.5 + i * 1.3 + s * 0.5) * 2; ty += 3; rect(tx - 1, ty, 3, 3, '#a83a8f' if s % 2 else '#d65abd')
        pulse = math.sin(b.t * 6) * 1.5
        pellipse(cx, cy, 31 + pulse, 25 + pulse, '#0d0f1c'); pellipse(cx, cy, 30 + pulse, 24 + pulse, fl or b.color)
        for i in range(7):
            yy = cy - 18 + i * 6
            for xx in range(-24, 24, 2):
                if abs(xx) < 28 - abs(yy - cy) * 0.6: rect(cx + xx, yy + R(math.sin(xx * 0.4 + i) * 2), 2, 1, '#c24aa6')
        rect(cx - 1, cy - 24, 2, 40, '#c24aa6')
        pcircle(cx, cy + 6, 8, '#1b0f22'); pcircle(cx + math.cos(look) * 3, cy + 6 + math.sin(look) * 3, 5, '#ff2a2a' if b.phase == 2 else '#5cf2ff')

# ---------- Partículas ----------
def burst(x, y, color, n):
    for _ in range(n): parts.append(O(x=x, y=y, vx=rnd(-90, 90), vy=rnd(-140, 30), life=rnd(0.3, 0.7), color=color, s=2 if random.random() < 0.3 else 1))

# ---------- Actualización ----------
def updatePlayer(dt):
    p = player
    if p.dead: return
    acc, mx = (1100 if p.ground else 700), 130
    dirIn = (1 if keys.get('right') else 0) - (1 if keys.get('left') else 0)
    if dirIn: p.vx = clamp(p.vx + dirIn * acc * dt, -mx, mx); p.dir = dirIn
    else:
        f = (900 if p.ground else 300) * dt; p.vx = 0 if abs(p.vx) <= f else p.vx - sign(p.vx) * f
    # salto con "coyote time", buffer y doble salto (jetpack)
    p.coyote = 0.1 if p.ground else p.coyote - dt
    p.buffer = 0.12 if pressed.get('jump') else p.buffer - dt
    if p.buffer > 0:
        if p.coyote > 0: p.vy = -330; p.coyote = 0; p.buffer = 0; p.jumps = 1; SFX.jump(); burst(p.x + 6, p.y + p.h, '#9aa3ff', 5)
        elif p.jumps < 2:
            p.vy = -290; p.buffer = 0; p.jumps = 2; SFX.jump()
            for _ in range(8): parts.append(O(x=p.x + 6, y=p.y + p.h, vx=rnd(-30, 30), vy=rnd(40, 120), life=0.35, color='#ff8a1f' if random.random() < 0.5 else '#ffe14d', s=2))
    if not keys.get('jump') and p.vy < -120: p.vy = -120  # salto variable
    p.vy = min(p.vy + 900 * dt, 480)
    wasAir = not p.ground
    moveBody(p, dt)
    if p.ground:
        p.jumps = 0
        if wasAir: p.land = 0.12; burst(p.x + 6, p.y + p.h, '#6a74c9', 3)
    if p.ground and solidAt(p.x + p.w / 2, p.y + p.h + 2): p.safeX = p.x
    # caer al vacío
    if p.y > VH + 30:
        hurtPlayer(1, p.x)
        if not p.dead: p.x = p.safeX - p.dir * 20; p.y = 60; p.vx = 0; p.vy = 0
    if cam.lock: p.x = clamp(p.x, level.arenaX, level.end - p.w)
    # disparar / espadazo
    p.cool -= dt
    if p.inv > 0: p.inv -= dt
    if p.power > 0: p.power -= dt
    if p.land > 0: p.land -= dt
    if p.slash > 0: p.slash -= dt
    mouseAim = mouse.down and state in ('play', 'hub')
    if mouseAim: p.dir = 1 if mouse.x + cam.x >= p.x + p.w / 2 else -1
    if (keys.get('shoot') or mouseAim) and p.cool <= 0:
        if MELEE:
            p.cool = 0.28; p.slash = 0.14; SFX.slash()
            hb = O(x=p.x + p.w if p.dir > 0 else p.x - 26, y=p.y - 4, w=26, h=34)
            for e in enemies:
                if not e.dead and overlap(hb, e): damageEnemy(e, 4 if p.power > 0 else 2, p.dir)
            if boss and not boss.dead and boss.intro <= 0 and overlap(hb, boss): hitBoss(4 if p.power > 0 else 2)
            for b in ebullets:
                if overlap(hb, b) and not b.wave: b.life = 0  # la espada corta balas
        else:
            p.cool = 0.14 if p.power > 0 else 0.2; SFX.shoot()
            bx, by = (p.x + p.w + 4 if p.dir > 0 else p.x - 10), p.y + 13
            # asistencia de puntería: apunta al enemigo o jefe más cercano que esté enfrente
            aim, best = 0, 1e9
            for t in enemies + ([boss] if boss and not boss.dead and boss.intro <= 0 else []):
                if t.dead: continue
                dx = (t.x + t.w / 2 - bx) * p.dir; dy = t.y + t.h / 2 - by; d = math.hypot(dx, dy)
                a = math.atan2(dy, dx)
                if dx > 0 and d < 300 and abs(a) < 0.9 and d < best: best = d; aim = a
            ang = aim if p.dir > 0 else math.pi - aim
            if mouseAim: ang = math.atan2(mouse.y - by, mouse.x + cam.x - bx)
            for s in ([-0.18, 0, 0.18] if p.power > 0 else [0]):
                bullets.append(O(x=bx, y=by, w=7, h=3, vx=math.cos(ang + s) * 340, vy=math.sin(ang + s) * 340, life=0.9))
            parts.append(O(x=bx + (0 if p.dir > 0 else 6), y=by + 1, vx=0, vy=0, life=0.05, color='#ffffff', s=4))
    # animación según el estado
    anim, frame = 'quieto', 0
    if not p.ground:
        anim = 'saltar'
        frame = 3 if p.vy < -150 else 4 if p.vy < -40 else 5 if p.vy < 40 else 6 if p.vy < 150 else 7
    elif p.land > 0: anim = 'saltar'; frame = 8
    elif abs(p.vx) > 15: anim = 'correr'
    if anim != 'saltar':
        if anim != p.anim: p.frame = 0; p.ft = 0
        A = sheet and sheet['animations'].get(anim)
        if A:
            p.ft += dt * 1000 * (abs(p.vx) / 130 if anim == 'correr' else 1)
            d = (A.get('durations') or [None] * 99)[p.frame] or 1000 / A['fps']
            if p.ft >= d: p.ft -= d; p.frame = (p.frame + 1) % A['frames']
        frame = p.frame
    p.anim = anim; p.drawFrame = frame

def hitBoss(dmg):
    global score, ebullets
    boss.hp -= dmg; boss.hit = 0.08; SFX.hit(); score += 10
    if boss.hp <= 0:
        boss.dead = True; boss.deadT = 0; score += 5000; ebullets = []
        for e in enemies: e.dead = True

def update(dt):
    global gtime, stateT, muted, state, pausedFrom, score, level, cam, hiscore
    gtime += dt; stateT += dt
    if pressed.get('mute'): muted = not muted; Music.set_muted(muted)
    # música según la pantalla
    if state in ('title', 'hub', 'win'): Music.play('lobby')
    elif state in ('intro', 'play'): Music.play('boss' if boss and not boss.dead else 'sector%d' % levelIdx)
    for t in timers[:]:
        t[0] -= dt
        if t[0] <= 0: timers.remove(t); t[1]()
    updateFade(dt)
    if fade.dir == 1: return  # durante el fundido no se lee la entrada
    go = pressed.get('start') or pressed.get('jump') or pressed.get('click')
    if state == 'title':
        if go: fadeTo(startHub)
        elif pressed.get('skip') and tutoDone: fadeTo(lambda: (_reset_score(), startLevel(0)))
        return
    if state == 'hub':
        if pressed.get('pause'): state = 'paused'; pausedFrom = 'hub'; return
        if not updateHub(dt): simulate(dt)
        return
    if state == 'intro':
        if stateT > 2.2 or pressed.get('start') or pressed.get('click'): state = 'play'
        return
    if state == 'gameover':
        updateParts(dt)
        if stateT > 1 and go: score = levelScore; startLevel(levelIdx)
        return
    if state == 'win':
        animDance(dt)
        if stateT > 2 and go:
            def back():
                global state, level, cam
                state = 'title'; level = None; cam = None
            fadeTo(back)
        return
    if state == 'paused':
        if pressed.get('pause') or pressed.get('start'): state = pausedFrom
        return
    if state == 'clear':
        animDance(dt); updateParts(dt)
        if stateT > 3.5 and not fade.dir:
            if levelIdx + 1 < len(LEVELS): fadeTo(lambda: startLevel(levelIdx + 1))
            else:
                state = 'win'; stateT = 0
                if score > hiscore: hiscore = score; write_save()
        return
    if pressed.get('pause'): state = 'paused'; pausedFrom = 'play'; return
    simulate(dt)

def _reset_score():
    global score
    score = 0

# física, enemigos, balas y objetos (se usa en los sectores y en el pueblo)
def simulate(dt):
    global enemies, bullets, ebullets, pickups, state, stateT, score, shake, flashT
    updatePlayer(dt)
    updateFx(dt)
    if not level.hub and not cam.lock and player.x > level.arenaX + 40: cam.lock = True; spawnBoss()
    target = level.arenaX if cam.lock else clamp(player.x - VW * 0.4, 0, level.camMax)
    cam.x += (target - cam.x) * min(1, dt * (3 if cam.lock else 8))
    for e in enemies:
        if not e.dead: updateEnemy(e, dt)
    enemies = [e for e in enemies if not e.dead]
    if boss:
        updateBoss(dt)
        if boss.dead and boss.deadT > 2.2 and state != 'clear':
            state = 'clear'; stateT = 0; player.frame = 0; player.ft = 0; Music.jingle('clear')
    for b in bullets:
        b.x += b.vx * dt; b.y += b.vy * dt; b.life -= dt
        if any(not s.plat and overlap(b, s) for s in solids): b.life = 0; burst(b.x, b.y, '#9aa3ff', 2)
        for e in enemies:
            if not e.dead and overlap(b, e): damageEnemy(e, 2 if player.power > 0 else 1, sign(b.vx)); b.life = 0; break
        if b.life > 0 and boss and not boss.dead and boss.intro <= 0 and overlap(b, boss):
            hitBoss(2 if player.power > 0 else 1); b.life = 0; burst(b.x, b.y, '#ffffff', 3)
    bullets = [b for b in bullets if b.life > 0]
    for b in ebullets:
        b.x += b.vx * dt; b.y += b.vy * dt; b.life -= dt
        if not b.wave and any(not s.plat and overlap(b, s) for s in solids): b.life = 0
        if overlap(b, player): hurtPlayer(1, b.x); b.life = 0
    ebullets = [b for b in ebullets if b.life > 0 and cam.x - 60 < b.x < cam.x + VW + 60 and -60 < b.y < VH + 20]
    for k in pickups:
        k.t += dt
        if k.vy is not None:
            k.vy += 600 * dt; k.y += k.vy * dt
            if solidAt(k.x + 5, k.y + k.h): k.y = math.floor(k.y + k.h) - k.h; k.vy = 0
            for s in solids:
                if overlap(k, s): k.y = s.y - k.h; k.vy = 0
        if overlap(k, player):
            k.got = True; SFX.pick()
            if k.tuto: hub.items += 1
            else: score += 50
            if k.type == 'vida': player.hp = min(5, player.hp + 1)
            else: player.power = 10
            burst(k.x + 5, k.y + 5, '#ff5c5c' if k.type == 'vida' else '#ffe14d', 10)
    pickups = [k for k in pickups if not k.got and k.y < VH + 40]
    updateParts(dt)
    if shake > 0: shake = max(0, shake - dt * 20)
    if flashT > 0: flashT -= dt

def updateParts(dt):
    global parts
    for q in parts: q.x += q.vx * dt; q.y += q.vy * dt; q.vy += 400 * dt; q.life -= dt
    parts = [q for q in parts if q.life > 0]

# baile de victoria (usa la animación "bailar" del creador)
def animDance(dt):
    p = player; A = sheet and sheet['animations'].get('bailar')
    if p.anim != 'bailar': p.anim = 'bailar'; p.frame = 0; p.ft = 0
    if A:
        p.ft += dt * 1000; d = (A.get('durations') or [None] * 99)[p.frame] or 100
        if p.ft >= d: p.ft -= d; p.frame = (p.frame + 1) % A['frames']
    p.drawFrame = p.frame

# ---------- Dibujo ----------
def drawPlayer():
    global ALPHA
    p = player
    if p.dead: return
    if p.inv > 0 and math.floor(p.inv * 20) % 2: return
    x, y = R(p.x - cam.x + p.w / 2), R(p.y + p.h)
    if sheet and sheetImg:
        A = sheet['animations'].get(p.anim) or sheet['animations']['quieto']
        f = min(p.drawFrame or 0, A['frames'] - 1)
        jy = (A.get('jy') or [0] * 99)[f] or 0
        fw = sheet['frameWidth']
        left = x - 16 if p.dir > 0 else x + 16 - fw
        draw_img(sheetImg, left, y - 39 + jy, area=sheet_frame(A['row'], f), flip=p.dir < 0)
    else:  # sin piloto guardado: figura simple
        rect(x - 6, y - 28, 12, 10, '#e9ecf5'); rect(x - 2 + p.dir * 2, y - 25, 6, 4, VISOR)
        rect(x - 5, y - 18, 10, 11, '#3b5bdb'); rect(x - 5, y - 7, 4, 7, '#2b2f45'); rect(x + 1, y - 7, 4, 7, '#2b2f45')
    if p.power > 0:
        ALPHA = 0.25 + math.sin(gtime * 20) * 0.1; pcircle(x, y - 14, 18, '#ffe14d'); ALPHA = 1
    if p.slash > 0:  # espadazo
        a0, a1 = (-1.2, 1.2) if p.dir > 0 else (math.pi + 1.2, math.pi - 1.2)
        lo, hi = min(a0, a1), max(a0, a1)
        for rr, col, wd in ((20, '#ffffff', 2), (17, ACCENT, 2)):
            cxx, cyy = x + p.dir * 4 + OX, y - 14 + OY
            pygame.draw.arc(ctx, C(col)[:3], (cxx - rr, cyy - rr, rr * 2, rr * 2), -hi, -lo, wd)

def drawLevel():
    for s in solids:
        if s.wall: continue
        x0 = R(s.x - cam.x)
        if x0 > VW or x0 + s.w < 0: continue
        if s.plat:
            x = 0
            while x < s.w:
                draw_img(level.decor.tiles['top'], x0 + x, s.y, area=(0, 0, int(min(TILE, s.w - x)), 12)); x += TILE
            rect(x0, s.y + 12, s.w, 2, 'rgba(0,0,0,.35)')
            continue
        y = 0
        while y < s.h:
            x = 0
            while x < s.w:
                draw_img(level.decor.tiles['top'] if y == 0 else level.decor.tiles['in'], x0 + x, s.y + y, area=(0, 0, max(1, int(min(TILE, s.w - x))), TILE)); x += TILE
            y += TILE
    # portal de la arena del jefe
    gx = R(level.arenaX - cam.x)
    if -20 < gx < VW + 20:
        for i in range(6): rect(gx - 2, GROUND - 60 + i * 10 + ((gtime * 40) % 10), 4, 5, '#ff4d6d' if cam.lock else '#5cf2ff')
        rect(gx - 4, GROUND - 64, 8, 4, '#9aa3b8')

def drawPickup(k):
    x, y = R(k.x - cam.x), R(k.y + math.sin(k.t * 4) * 2)
    if k.type == 'vida': heart(x, y)
    else:
        pcircle(x + 5, y + 5, 5, '#0d0f1c'); pcircle(x + 5, y + 5, 4, '#ffe14d'); rect(x + 4, y + 2, 2, 6, '#ff8a1f'); rect(x + 2, y + 4, 6, 2, '#ff8a1f')

# Corazón: "Heart" de shinzohotoroki (OpenGameArt, CC0), reducido a 10×9; el vacío es la misma silueta oscura
HEART_IMG = load_img('opengameart/heart.png')
heartFull = heartEmpty = None
if HEART_IMG:
    heartFull = pygame.transform.smoothscale(HEART_IMG.subsurface((5, 7, 21, 19)), (10, 9))
    heartEmpty = _color_filter(heartFull, gray=True, bright=0.35)
def heart(x, y, empty=False):
    if heartFull: ctx.blit(heartEmpty if empty else heartFull, (R(x) + OX, R(y) + OY))

def drawHUD():
    for i in range(5): heart(8 + i * 11, 6, i >= player.hp)
    text('PUNTOS ' + str(score).zfill(6), VW - 8, 7, 8, '#e8ebff', 'right')
    text(level.name, 8, 18, 7, '#8a92c9')
    if player.power > 0: text(('ESPADA x2 ' if MELEE else 'DISPARO TRIPLE ') + str(math.ceil(player.power)), 8, 28, 7, '#ffe14d')
    if boss and not boss.dead:
        w = 200; x = VW / 2 - w / 2; y = VH - 16
        text(boss.name + (' · ¡FURIOSO!' if boss.phase == 2 else ''), VW / 2, y - 10, 7, '#ff5c5c' if boss.phase == 2 else '#e8ebff', 'center')
        rect(x - 1, y - 1, w + 2, 7, '#000000'); rect(x, y, w, 5, '#3a1020'); rect(x, y, w * max(0, boss.hp / boss.max), 5, boss.color)

def overlayBox(lines, sub=None):
    rect(0, 0, VW, VH, 'rgba(5,6,13,.55)')
    for i, (s, size, c) in enumerate(lines): text(s, VW / 2, 70 + i * (size + 8), size, c, 'center')
    if sub and math.floor(gtime * 2) % 2: text(sub, VW / 2, 200, 8, '#5cf2ff', 'center')

def draw():
    global OX, OY
    inHub = level and level.hub
    if state == 'title' or inHub: drawCity(gtime * 40 if state == 'title' else cam.x)
    else: FONDOS.draw(level.bg, cam.x / 40 + gtime * 0.3, VW, VH); drawFar(); drawMid()
    if state == 'title': drawTitle(); return
    if shake > 0: OX, OY = R(rnd(-shake, shake)), R(rnd(-shake, shake))
    if inHub:
        drawHubWorld(); drawHubSolids()
        for n in hub.npcs: drawNPC(n)
    else: drawGroundDecor(); drawLevel(); drawLevelDetails()
    for k in pickups: drawPickup(k)
    for e in enemies: drawEnemy(e)
    if boss: drawBoss()
    drawPlayer()
    if inHub: drawBolt()
    for b in bullets:  # láser con estela según su dirección
        n = math.hypot(b.vx, b.vy); ux, uy = b.vx / n, b.vy / n
        for i in range(7): rect(b.x - cam.x - ux * i, b.y + 1 - uy * i, 2, 2, '#ffffff' if i < 2 else ACCENT)
    for b in ebullets:
        if b.wave: rect(b.x - cam.x, b.y, b.w, b.h, b.color); rect(b.x - cam.x + 2, b.y - 4, 4, 4, '#ff8a1f'); continue
        pcircle(b.x - cam.x + b.w / 2, b.y + b.h / 2, b.w / 2 + 1, '#0d0f1c'); pcircle(b.x - cam.x + b.w / 2, b.y + b.h / 2, b.w / 2, b.color)
    drawFx()
    for q in parts: rect(q.x - cam.x, q.y, q.s, q.s, q.color)
    if not inHub: drawDust()
    else:  # lluvia de neón sobre Puerto Neón
        for d in hub.rain:
            d.y += d.v / 60; d.x -= d.v / 300
            if d.y > GROUND + 4:
                d.y = -10; d.x = random.random() * (VW + 40)
                if random.random() < 0.3: parts.append(O(x=d.x + cam.x, y=GROUND, vx=rnd(-20, 20), vy=rnd(-40, -10), life=0.15, color='rgba(160,220,255,.6)', s=1))
            rect(d.x, d.y, 1, 5, 'rgba(150,200,255,.28)')
    OX = OY = 0
    if flashT > 0: rect(0, 0, VW, VH, 'rgba(255,60,60,.25)')
    if inHub: drawHubHUD(); drawDialog()
    else: drawHUD()
    if state == 'intro': overlayBox([[level.name, 12, '#5cf2ff'], ['¡Llega al final y vence al jefe!', 8, '#e8ebff']])
    if state == 'paused': overlayBox([['PAUSA', 20, '#e8ebff']], 'P PARA SEGUIR')
    if state == 'gameover': overlayBox([['GAME OVER', 24, '#ff5c5c'], ['Puntos: %d' % score, 10, '#e8ebff']], 'ENTER PARA REINTENTAR EL SECTOR')
    if state == 'clear': overlayBox([['¡SECTOR COMPLETADO!', 16, '#ffe14d'], [boss.name + ' DERROTADO' if boss else '', 9, '#e8ebff']])
    if state == 'win': overlayBox([['¡GANASTE!', 26, '#ffe14d'], ['La galaxia está a salvo', 10, '#e8ebff'], ['Puntos: %d   Récord: %d' % (score, hiscore), 9, '#5cf2ff']], 'ENTER PARA VOLVER AL INICIO')

# pantalla de título: la ciudad de fondo, la calle y tu piloto bailando
def drawTitle():
    rect(0, GROUND, VW, VH - GROUND, '#0b1220'); rect(0, GROUND, VW, 2, '#2a3d5c'); rect(0, GROUND + 2, VW, 1, '#ff5ca8')
    i = -((gtime * 40) % 24)
    while i < VW: rect(i, GROUND + 10, 12, 1, '#ffd84d'); i += 24
    if sheet and sheetImg:
        A = sheet['animations'].get('bailar') or sheet['animations']['quieto']
        durs = A.get('durations') or [1000 / A['fps']] * A['frames']
        t = (gtime * 1000) % sum(durs); f = 0
        while f < len(durs) - 1 and t > durs[f]: t -= durs[f]; f += 1
        fw, fh = sheet['frameWidth'], sheet['frameHeight']
        draw_img(sheetImg, VW / 2 - 32, GROUND - 78 + ((A.get('jy') or [0] * 99)[f] or 0) * 2, fw * 2, fh * 2, area=sheet_frame(A['row'], f))
    rect(0, 22, VW, 74, 'rgba(5,6,13,.55)')
    text('PIXEL', VW / 2, 26, 20, '#e8ebff', 'center')
    text('SPACEFIGHTER', VW / 2, 50, 26, '#5cf2ff', 'center')
    text('Puerto Neón · 3 sectores · 3 jefes', VW / 2, 82, 8, '#ffd84d', 'center')
    if not sheet: text('Tip: crea tu piloto en el Creador web y expórtalo a assets/piloto.json', VW / 2, 104, 7, '#ffe14d', 'center')
    if math.floor(gtime * 2) % 2: text('ENTER / CLIC: EMPEZAR EN PUERTO NEÓN', VW / 2, 236, 8, '#5cf2ff', 'center')
    if tutoDone: text('T: saltar el entrenamiento e ir al Sector 1', VW / 2, 248, 7, '#8a92c9', 'center')
    if hiscore: text('RÉCORD %d' % hiscore, VW - 8, 8, 8, '#ffe14d', 'right')

# =====================================================================
#  PUERTO NEÓN — pueblo espacial con el tutorial (assets/pueblo, Warped City de ansimuz, CC0)
# =====================================================================
IMGS = {}
def img(path):
    if path not in IMGS: IMGS[path] = load_img('pueblo/' + path)
    return IMGS[path]
def anim(base, n, fps): return dict(frames=[img('%s-%d.png' % (base, i + 1)) for i in range(n)], fps=fps)
def animFrame(a, t): return a['frames'][math.floor(t * a['fps']) % len(a['frames'])]

CITY = dict(
    sky=img('ENVIRONMENT/background/skyline-a.png'), far=img('ENVIRONMENT/background/buildings-bg.png'),
    near=img('ENVIRONMENT/background/near-buildings-bg.png'), tiles=img('ENVIRONMENT/tileset.png'),
    cars=[img('SPRITES/vehicles/v-red.png'), img('SPRITES/vehicles/v-yellow.png'), img('SPRITES/vehicles/v-police.png')],
    npc=anim('SPRITES/player/idle/idle', 4, 6), walk=anim('SPRITES/player/walk/walk', 16, 14),
    drone=anim('SPRITES/misc/drone/drone', 4, 10), turret=img('SPRITES/misc/turret/turret-1.png'),
    boom=anim('SPRITES/misc/enemy-explosion/enemy-explosion', 6, 16))
# piezas del tileset [x, y, w, h]
PIECE = dict(lamp=(112, 100, 32, 60), shop=(224, 96, 48, 64), shutter=(224, 16, 48, 64), window=(48, 16, 48, 64),
             door=(288, 16, 32, 64), door2=(144, 16, 32, 64), portal=(224, 176, 48, 64), hover=(105, 176, 46, 16), fan=(176, 144, 32, 32), pipes=(160, 96, 48, 32))
PROPS = dict(bigBanner=anim('ENVIRONMENT/props/banner-big/banner-big', 4, 4), neon=anim('ENVIRONMENT/props/banner-neon/banner-neon', 4, 6),
             coke=anim('ENVIRONMENT/props/banner-coke/banner-coke', 3, 4), side=anim('ENVIRONMENT/props/banner-side/banner-side', 4, 5),
             sushi=anim('ENVIRONMENT/props/banner-sushi/banner-sushi', 3, 3), monitor=anim('ENVIRONMENT/props/monitorface/monitor-face', 4, 3),
             scroll=anim('ENVIRONMENT/props/banner-scroll/banner-scroll', 4, 8))
STILL = dict(hotel=img('ENVIRONMENT/props/hotel-sign.png'), antenna=img('ENVIRONMENT/props/antenna.png'),
             box=img('ENVIRONMENT/props/control-box-1.png'), arrow=img('ENVIRONMENT/props/banner-arrow.png'))

# cielo y ciudad con parallax (también para la pantalla de título)
skyColor = None; flyers = []
def drawCity(camX):
    global skyColor
    if not ok(CITY['sky']): rect(0, 0, VW, VH, '#0e0b2a'); return
    if skyColor is None: skyColor = tuple(CITY['sky'].get_at((0, 0)))[:3]
    ctx.fill(skyColor)
    def tile(im, speed, y):
        if not ok(im): return
        w = im.get_width(); x = -((camX * speed) % w)
        while x < VW: draw_img(im, x, y); x += w
    tile(CITY['sky'], 0.05, VH - 240)
    tile(CITY['far'], 0.15, GROUND - 124 + 16)
    # autos voladores entre las capas
    if not flyers:
        for i in range(4): flyers.append(O(x=rnd(0, VW), y=rnd(20, 110), v=rnd(25, 60) * (-1 if random.random() < 0.5 else 1), car=i % 3))
    for f in flyers:
        f.x += f.v / 60
        if f.x > VW + 90: f.x = -90
        if f.x < -90: f.x = VW + 90
        im = CITY['cars'][f.car]
        if not ok(im): continue
        w, h = im.get_width(), im.get_height()
        draw_img(im, R(f.x) - w / 4, R(f.y), w / 2, h / 2, flip=f.v < 0)
    tile(CITY['near'], 0.4, GROUND - 209 + 6)

def piece(name, x, y):
    if ok(CITY['tiles']): draw_img(CITY['tiles'], x, y, area=PIECE[name])

# ---------- Construcción del pueblo ----------
HUB_END = 1720
def buildHub():
    global level, solids, enemies, bullets, ebullets, pickups, parts, fx, boss, hub
    level = O(name='PUERTO NEÓN', hub=True, bg='pueblo', arenaX=1e9, end=HUB_END, camMax=HUB_END - VW)
    solids = []; enemies = []; bullets = []; ebullets = []; pickups = []; parts = []; fx = []; boss = None
    solids.append(O(x=0, y=GROUND, w=HUB_END, h=60, ground=True, street=True))
    solids.extend([O(x=-40, y=0, w=40, h=VH, wall=True), O(x=HUB_END, y=0, w=40, h=VH, wall=True)])
    solids.append(O(x=380, y=GROUND - 30, w=32, h=30, crate=True))              # salto
    solids.append(O(x=560, y=GROUND - 76, w=48, h=76, block=True))              # doble salto
    solids.append(O(x=872, y=GROUND - 52, w=46, h=8, plat=True, hover=True))    # plataforma flotante
    solids.append(O(x=965, y=GROUND - 16, w=6, h=16, fence=True))               # corral del slime
    def gate(x):
        g = O(x=x, y=0, w=8, h=GROUND, gate=True); solids.append(g); return g
    hub = O(step=0, talked=-1, dialog=None, items=0, launched=False, bolt=O(x=60, y=150), atDoor=False, lastChar=-1,
            gates=[gate(935), gate(1150), gate(1330)],
            walkers=[O(x=x, dir=d, hue=h, v=v) for x, d, h, v in [(200, 1, 120, 26), (700, -1, 40, 22), (1150, 1, 250, 30), (1500, -1, 180, 24), (420, -1, 320, 20)]],
            rain=[O(x=random.random() * VW, y=random.random() * VH, v=180 + random.random() * 120) for _ in range(90)],
            npcs=[O(name='CAPITANA NOVA', x=110, hue=0, color='#ffd84d'), O(name='SARGENTO REX', x=760, hue=200, color='#7dd3ff'),
                  O(name='MECÁNICA KAI', x=1440, hue=290, color='#ff8ad8')])
    for x, y in [(800, GROUND), (840, GROUND), (884, GROUND - 52)]: enemies.append(makeEnemy('target', x, y))
    s = makeEnemy('slime', 1060, GROUND); s.noHop = True; s.hp = 2; enemies.append(s)
    pickups.append(O(x=1200, y=GROUND - 16, w=10, h=10, type='vida', t=0, tuto=True))
    pickups.append(O(x=1255, y=GROUND - 16, w=10, h=10, type='poder', t=0, tuto=True))

SHOOT_TXT = (['Hora de practicar. Destruye los 3 blancos con tu espada.', 'Ataca con X, J o clic del mouse (te volteas hacia el cursor). ¡La espada también corta las balas enemigas!']
             if MELEE else
             ['Hora de practicar puntería. Destruye los 3 blancos.', 'Dispara con X o J: tus disparos apuntan solos al enemigo más cercano que tengas enfrente.',
              'O usa el MOUSE: apunta con el cursor y haz clic (o mantenlo presionado) para disparar hacia donde quieras.'])
# pasos del tutorial: quién habla, dónde empieza y cuándo se cumple
TUTO = [
    dict(at=-1, who=0, goal='Camina con ← → (o A / D)', lines=['¡Bienvenido a Puerto Neón, piloto!', 'Antes de salir al espacio vamos a repasar lo básico.', 'Camina con ← → (o A / D). BOLT, mi robot, te va a acompañar. ¡Síguelo!'], done=lambda: player.x > 300),
    dict(at=300, who='bolt', goal='Salta el contenedor: ↑, W o ESPACIO', lines=['Bip-bup. Hola, soy BOLT.', 'Ese contenedor estorba. Salta con ↑, W o ESPACIO.', 'Si mantienes presionado el botón, saltas más alto.'], done=lambda: player.x > 430),
    dict(at=470, who='bolt', goal='Doble salto: salta y vuelve a saltar en el aire', lines=['Ese muro es más alto. Usa tu jetpack:', 'salta y, en el aire, presiona salto OTRA VEZ. ¡Doble salto!'], done=lambda: player.x > 625),
    dict(at=690, who=1, goal=('Ataca' if MELEE else 'Dispara') + ' con X / J o clic del mouse · destruye los 3 blancos', lines=['¡Firmes, recluta! Soy el Sargento Rex.', *SHOOT_TXT, 'Uno está en la plataforma flotante. ¡Súbete o apunta desde abajo!'], done=lambda: not any(e.type == 'target' for e in enemies)),
    dict(at=945, who='bolt', goal='Aplasta al slime cayéndole encima', lines=['¡Alerta! Se escapó un slime del laboratorio.', 'Aplástalo cayéndole encima... o dispárale.', 'Ojo: a los cangrejos con púas NUNCA los pises, ¡duelen!'], done=lambda: not any(e.type == 'slime' for e in enemies)),
    dict(at=1160, who='bolt', goal='Recoge el corazón y la estrella de poder', lines=['Los corazones te devuelven vida.', 'La estrella da ' + ('espada doble' if MELEE else 'disparo triple') + ' por 10 segundos. ¡Agarra los dos!'], done=lambda: hub.items >= 2),
    dict(at=1350, who=2, goal='Entra a la puerta brillante y presiona ↓ o ENTER', lines=['¡Así que tú eres el nuevo piloto! Soy Kai, la mecánica.', 'Tu nave ya tiene combustible. Está detrás de la puerta brillante del hangar.', 'Hay 3 sectores y un jefe al final de cada uno. Si caes al vacío pierdes un corazón.', 'Párate en la puerta y presiona ↓ o ENTER para despegar. ¡Buena suerte!'], done=lambda: False),
]
DOOR_X = 1580

def speaker(who):
    if who == 'bolt': return O(name='BOLT', color='#5cf2ff', bolt=True)
    return hub.npcs[who]

def openDialog(step):
    T = TUTO[step]
    hub.dialog = O(who=speaker(T['who']), lines=T['lines'], i=0, chars=0)
    hub.talked = step; beep(700, 0.05, 'triangle', 0.04)

def updateHub(dt):
    global solids, tutoDone
    B = hub.bolt  # BOLT sigue al jugador
    tx, ty = player.x + player.w / 2 - player.dir * 22, player.y - 22 + math.sin(gtime * 3) * 3
    B.x += (tx - B.x) * min(1, dt * 3); B.y += (ty - B.y) * min(1, dt * 3)
    if hub.dialog:
        d = hub.dialog; line = d.lines[d.i]
        d.chars = min(len(line), d.chars + dt * 45)
        c = math.floor(d.chars)
        if c != hub.lastChar and c % 3 == 0 and d.chars < len(line) and not muted and _typing: random.choice(_typing).play()
        hub.lastChar = c
        if pressed.get('start') or pressed.get('jump') or pressed.get('shoot') or pressed.get('down') or pressed.get('click'):
            if d.chars < len(line): d.chars = len(line)
            else:
                d.i += 1
                if d.i >= len(d.lines): hub.dialog = None
                else: d.chars = 0
        updateParts(dt); updateFx(dt)
        return True  # mientras hay diálogo, el mundo se pausa
    T = TUTO[hub.step]
    if hub.talked < hub.step and player.x >= T['at']: openDialog(hub.step); return True
    if hub.talked == hub.step and T['done']():
        gi = {3: 0, 4: 1, 5: 2}.get(hub.step)
        if gi is not None:
            g = hub.gates[gi]; solids = [s for s in solids if s is not g]; burst(g.x + 4, GROUND - 40, '#5cf2ff', 20)
        SFX.pick(); hub.step += 1
    # puerta del hangar
    atDoor = player.x + player.w > DOOR_X + 8 and player.x < DOOR_X + 40 and player.ground
    hub.atDoor = atDoor and hub.step >= 6
    if hub.atDoor and (pressed.get('down') or pressed.get('start')) and not hub.launched:
        hub.launched = True; tutoDone = True; write_save()
        beep(200, 0.8, 'sawtooth', 0.05, 600)
        fadeTo(lambda: (_reset_score(), startLevel(0)))
    return False

def drawHubWorld():
    cx = R(cam.x)
    rect(0, GROUND - 74, VW, 74, '#101a28')  # fachada de las tiendas
    x = -(cx % 32)
    while x < VW: rect(x, GROUND - 74, 1, 74, '#16243a'); x += 32
    rect(0, GROUND - 76, VW, 2, '#22344f'); rect(0, GROUND - 74, VW, 1, '#0a1019')
    def at(wx, fn):
        sx = wx - cx
        if -120 < sx < VW + 120: fn(sx)
    t = gtime
    for x, p in [(30, 'lamp'), (520, 'lamp'), (1010, 'lamp'), (1390, 'lamp'), (1660, 'lamp')]: at(x, lambda sx, p=p: piece(p, sx, GROUND - 60))
    for x, p, yy in [(160, 'shop', 64), (230, 'door', 64), (460, 'window', 64), (660, 'shutter', 64), (1090, 'window', 64), (1270, 'door2', 64),
                     (1480, 'shutter', 64), (DOOR_X, 'portal', 64), (1530, 'pipes', 106), (330, 'fan', 106)]:
        at(x, lambda sx, p=p, yy=yy: piece(p, sx, GROUND - yy))
    # letreros animados
    def A(p, x, y): at(x, lambda sx: draw_img(animFrame(PROPS[p], t), sx, y))
    A('bigBanner', 200, GROUND - 168); A('neon', 290, GROUND - 130); A('sushi', 165, GROUND - 82); A('coke', 610, GROUND - 150)
    A('monitor', 470, GROUND - 92); A('side', 1000, GROUND - 150); A('scroll', 1120, GROUND - 124); A('monitor', 1300, GROUND - 92)
    A('neon', 1460, GROUND - 132); A('bigBanner', 1620, GROUND - 172)
    at(700, lambda sx: draw_img(STILL['hotel'], sx, GROUND - 112))
    at(1550, lambda sx: draw_img(STILL['antenna'], sx, GROUND - 76 - 96))
    # transeúntes (detrás del jugador, un poco más oscuros)
    for w in hub.walkers:
        w.x += w.dir * w.v / 60
        if w.x > HUB_END - 20: w.dir = -1
        if w.x < 20: w.dir = 1
        at(w.x, lambda sx, w=w: draw_local(animFrame(CITY['walk'], gtime + w.hue), sx, -36, GROUND - 67, w.dir, hue=w.hue, bright=0.7))
    if hub.step >= 6: at(DOOR_X + 12, lambda sx: text('▼', sx + 12, GROUND - 92 + math.sin(t * 5) * 3, 10, '#5cf2ff', 'center'))
    text('HANGAR', DOOR_X + 24 - cx, GROUND - 80, 7, '#7dff6a', 'center')

def drawHubSolids():
    cx = R(cam.x)
    for s in solids:
        x = R(s.x - cx)
        if x > VW + 10 or x + s.w < -10: continue
        if s.street:  # calle con línea de neón
            rect(x, GROUND, s.w, VH - GROUND, '#0b1220'); rect(x, GROUND, s.w, 2, '#2a3d5c')
            i = (-cx) % 24
            while i < VW: rect(i, GROUND + 10, 12, 1, '#ffd84d'); i += 24
            rect(x, GROUND + 2, s.w, 1, '#ff5ca8')
        elif s.crate: draw_img(STILL['box'], x, s.y)
        elif s.block:
            y = 0
            while y < s.h:
                for i in range(0, s.w, TILE): draw_img(TILE_TOP if y == 0 else TILE_IN, x + i, s.y + y, area=(0, 0, TILE, int(min(TILE, s.h - y))))
                y += TILE
        elif s.hover: piece('hover', x, s.y - 4)
        elif s.fence: rect(x, s.y, s.w, s.h, '#3a4a66'); rect(x, s.y, s.w, 2, '#ff5ca8')
        elif s.gate:  # barrera holográfica
            for y in range(20, GROUND, 6): rect(x + 2 + R(math.sin(gtime * 6 + y) * 1.5), y + ((gtime * 30) % 6), 4, 3, 'rgba(92,242,255,.75)')
            rect(x, GROUND - 4, s.w, 4, '#5cf2ff'); rect(x, 16, s.w, 4, '#5cf2ff')

def drawNPC(n):
    x = R(n.x - cam.x); f = animFrame(CITY['npc'], gtime + n.hue)
    if x < -80 or x > VW + 80 or not ok(f): return
    draw_local(f, x, -36, GROUND - 67, -1 if player.x < n.x else 1, hue=n.hue)
    text(n.name, x, GROUND - 58, 6, n.color, 'center')

def drawBolt():
    f = animFrame(CITY['drone'], gtime); B = hub.bolt
    if not ok(f): return
    draw_local(f, R(B.x - cam.x), -27, R(B.y) - 13, player.dir, sc=0.5)

def wrap(s, maxW):
    out, cur = [], ''
    for w in s.split(' '):
        t = cur + ' ' + w if cur else w
        if measure(t, 8) > maxW: out.append(cur); cur = w
        else: cur = t
    out.append(cur); return out

def drawDialog():
    d = hub.dialog
    if not d: return
    x, y, w, h = 12, VH - 66, VW - 24, 56
    rect(x - 2, y - 2, w + 4, h + 4, '#000000'); rect(x, y, w, h, 'rgba(14,18,40,.95)'); rect(x, y, w, 1, d.who.color)
    rect(x + 6, y + 6, 44, 44, '#05060d')  # retrato
    ctx.set_clip(pygame.Rect(x + 6, y + 6, 44, 44))
    if d.who.bolt: draw_img(animFrame(CITY['drone'], gtime), x + 1, y + 2)
    else:
        f = animFrame(CITY['npc'], gtime)
        if ok(f): draw_img(f, x + 6 - 14, y + 6 - 6, 71, 67, hue=d.who.hue or 0)
    ctx.set_clip(None)
    text(d.who.name, x + 58, y + 6, 8, d.who.color)
    shown = d.lines[d.i][:math.floor(d.chars)]
    for i, l in enumerate(wrap(shown, w - 70)): text(l, x + 58, y + 18 + i * 10, 8, '#e8ebff')
    if d.chars >= len(d.lines[d.i]) and math.floor(gtime * 3) % 2: text('▶ ENTER', x + w - 6, y + h - 11, 7, '#5cf2ff', 'right')
    text('%d/%d' % (d.i + 1, len(d.lines)), x + 58, y + h - 11, 6, '#8a92c9')

def drawHubHUD():
    text('PUERTO NEÓN · ENTRENAMIENTO', 8, 7, 8, '#ffd84d')
    T = TUTO[hub.step] if hub.step < len(TUTO) else None
    if T and not hub.dialog:
        s = '► ' + T['goal']; w = measure(s, 8) + 12
        rect(8, 19, w, 13, 'rgba(5,6,13,.75)'); rect(8, 19, 2, 13, '#5cf2ff')
        text(s, 14, 22, 8, '#e8ebff')
    for i in range(5): heart(VW - 60 + i * 11, 6, i >= player.hp)
    if hub.atDoor and math.floor(gtime * 3) % 2: text('↓ / ENTER  PARA DESPEGAR', VW / 2, 120, 10, '#7dff6a', 'center')

# ---------- Explosiones (sprite de Warped City) ----------
def explode(x, y, s=1): fx.append(O(x=x, y=y, t=0, s=s))
def updateFx(dt):
    global fx
    for e in fx: e.t += dt
    fx = [e for e in fx if e.t < 6 / 16]
def drawFx():
    for e in fx:
        f = CITY['boom']['frames'][min(5, math.floor(e.t * 16))]
        if ok(f): draw_img(f, R(e.x - cam.x - 27 * e.s), R(e.y - 26 * e.s), 55 * e.s, 52 * e.s)

# ---------- Fundido entre pantallas ----------
def fadeTo(cb):
    global fade
    fade = O(a=fade.a, dir=1, cb=cb)
def updateFade(dt):
    if fade.dir == 1:
        fade.a += dt * 3
        if fade.a >= 1:
            fade.a = 1; fade.dir = -1
            if fade.cb: fade.cb()
    elif fade.dir == -1:
        fade.a -= dt * 2.5
        if fade.a <= 0: fade.a = 0; fade.dir = 0

def startHub():
    global state, stateT
    buildHub(); resetPlayer(True); player.x = 40; state = 'hub'; stateT = 0

# =====================================================================
#  DECORACIÓN DE LOS SECTORES — capas de fondo, objetos en el piso y polvo espacial
# =====================================================================
THEMES = [
    dict(tile=dict(base='#232a55', dark='#1a1f40', mid='#2f3870', dot='#3e478a', top='#6a74c9', hi='#9aa3ff', glow='#5cf2ff'),
         glow=['rgba(255,92,168,.18)', 'rgba(140,80,255,.14)'], planetHue=0, sil='#24132e', silLight='#ff5ca8',
         mid=['tower', 'tower', 'rock', 'station'], ground=['lamp', 'antenna', 'box', 'crystal', 'beacon', 'sign'], crystal='#ff5ca8'),
    dict(tile=dict(base='#173a40', dark='#0f2a2f', mid='#1f4d55', dot='#2c6670', top='#3fb8a6', hi='#8ff5e0', glow='#c58cff'),
         glow=['rgba(120,90,255,.2)', 'rgba(60,220,200,.12)'], planetHue=160, sil='#141a33', silLight='#5cf2ff',
         mid=['spire', 'jelly', 'spire', 'rock'], ground=['crystal', 'plant', 'crystal', 'plant', 'beacon', 'pipe'], crystal='#5cf2ff'),
    dict(tile=dict(base='#3a1f1f', dark='#2a1515', mid='#4d2a26', dot='#6a3a30', top='#c2552f', hi='#ff9a5c', glow='#ffe14d'),
         glow=['rgba(255,80,40,.2)', 'rgba(255,180,60,.1)'], planetHue=340, sil='#1f0c0c', silLight='#ff3b3b',
         mid=['ruin', 'rock', 'ruin', 'rock'], ground=['vent', 'pipe', 'beacon', 'antenna', 'vent', 'box'], crystal='#ff7a3c'),
]
THEME_TILES = [dict(top=makeTileThemed(True, T['tile']), **{'in': makeTileThemed(False, T['tile'])}) for T in THEMES]
_glow_cache = {}

def buildDecor(n):
    T = THEMES[n]; r = mulberry(LEVELS[n].seed * 31 + 5)
    D = O(theme=T, tiles=THEME_TILES[n], mid=[], ground=[], dust=[], edges=[])
    x = -100
    while x < level.end * 0.4 + VW:  # capa media (parallax 0.35)
        D.mid.append(O(x=x, kind=T['mid'][math.floor(r() * len(T['mid']))], h=50 + r() * 90, y=30 + r() * 90, s=r(), seed=r() * 100))
        x += 60 + r() * 70
    for g in [s for s in solids if s.ground]:
        D.edges.extend([g.x, g.x + g.w])  # bordes con franjas de peligro
        x = g.x + 24
        while x < g.x + g.w - 24:
            if not (level.arenaX - 30 < x < level.arenaX + 30):
                D.ground.append(O(x=x, kind=T['ground'][math.floor(r() * len(T['ground']))], s=r(), seed=r() * 100))
            x += 46 + r() * 70
    for _ in range(40): D.dust.append(O(x=r() * VW, y=r() * VH, v=0.2 + r() * 0.6, s=2 if r() < 0.2 else 1, a=0.2 + r() * 0.5))
    level.decor = D
    if n not in _glow_cache:
        g = radial(VW, VH, VW * 0.25, VH * 0.3, 220, T['glow'][0], 'rgba(0,0,0,0)')
        g.blit(radial(VW, VH, VW * 0.8, VH * 0.6, 200, T['glow'][1], 'rgba(0,0,0,0)'), (0, 0))
        _glow_cache[n] = g

# --- capa lejana: brillos de nebulosa y un planeta grande que se mueve muy lento
def drawFar():
    T = level.decor.theme
    ctx.blit(_glow_cache[levelIdx], (0, 0))
    L = FONDOS.list['planetas']['layers']; big = L[4].get('img'); ring = L[3].get('img')
    if big: draw_img(big, R(VW * 0.62 - cam.x * 0.03), 26, big.get_width() * 1.5, big.get_height() * 1.5, hue=T['planetHue'])
    if ring: draw_img(ring, R(VW * 0.18 - cam.x * 0.05), 14, hue=T['planetHue'])

# --- capa media: siluetas con luces
def drawMid():
    D = level.decor; T = D.theme; px = cam.x * 0.35
    for m in D.mid:
        x = R(m.x - px)
        if x < -80 or x > VW + 80: continue
        blink = math.sin(gtime * 3 + m.seed) > 0.6
        if m.kind == 'tower':
            w = 14 + R(m.s * 16); top = GROUND - m.h
            rect(x, top, w, m.h + 10, T['sil']); rect(x - 3, GROUND - 14, w + 6, 24, T['sil'])
            rect(x + w / 2 - 1, top - 14, 2, 14, T['sil']); rect(x + w / 2 - 1, top - 16, 2, 2, '#ff3b3b' if blink else '#5a1020')
            y = top + 6
            while y < GROUND - 16:
                for i in range(3, w - 3, 5):
                    if math.floor(m.seed + y * 0.3 + i) % 3 == 0: rect(x + i, y, 2, 2, T['silLight'])
                y += 8
        elif m.kind == 'station':
            y = m.y; pellipse(x, y, 26, 7, T['sil']); pellipse(x, y, 20, 4, '#000000')
            rect(x - 4, y - 12, 8, 24, T['sil']); rect(x - 1, y - 18, 2, 6, T['sil'])
            for i in range(-18, 19, 9): rect(x + i, y + 1, 2, 1, '#ffffff' if blink and i == 0 else T['silLight'])
        elif m.kind == 'rock':
            y = m.y + math.sin(gtime * 0.6 + m.seed) * 4; r = 5 + R(m.s * 9)
            pcircle(x, y, r, T['sil']); pcircle(x - r * 0.3, y - r * 0.3, max(1, r * 0.4), 'rgba(255,255,255,.06)')
            rect(x + r * 0.2, y + r * 0.1, 2, 2, 'rgba(0,0,0,.4)')
        elif m.kind == 'spire':
            top = GROUND - m.h; w = 10 + R(m.s * 10); i = 0
            while i < m.h:
                ww = max(2, w * (i / m.h)); rect(x - ww / 2, top + i, ww, 4, T['sil']); i += 4
            pcircle(x, top + 4, 2, T['silLight'] if blink else 'rgba(92,242,255,.4)')
        elif m.kind == 'jelly':
            y = m.y + math.sin(gtime + m.seed) * 8
            pellipse(x, y, 7, 5, 'rgba(197,140,255,.35)'); pellipse(x, y - 1, 5, 3, 'rgba(230,200,255,.5)')
            for i in (-4, 0, 4):
                for s in range(5): rect(x + i + math.sin(gtime * 2 + s + i) * 1.5, y + 5 + s * 3, 1, 2, 'rgba(197,140,255,.4)')
        elif m.kind == 'ruin':
            w = 18 + R(m.s * 14); top = GROUND - m.h
            rect(x, top + 10, w, m.h, T['sil'])
            for i in range(0, w, 4): rect(x + i, top + 10 - ((i * 7 + m.seed) % 11), 4, 12, T['sil'])  # borde roto
            if blink: rect(x + w / 2, top + 20, 3, 3, T['silLight'])
            rect(x + 4, top + 30, w - 8, 2, 'rgba(255,90,40,.25)')

# --- objetos del piso (detrás del jugador)
def drawGroundDecor():
    global ALPHA
    D = level.decor; T = D.theme
    for o in D.ground:
        x = R(o.x - cam.x)
        if x < -40 or x > VW + 40: continue
        y = GROUND; k = o.kind
        if k == 'lamp': piece('lamp', x - 16, y - 60)
        elif k == 'antenna':
            draw_img(STILL['antenna'], x - 11, y - 96); rect(x - 1, y - 98, 2, 2, '#ff3b3b' if math.sin(gtime * 4 + o.seed) > 0 else '#4a1010')
        elif k == 'box': draw_img(STILL['box'], x - 16, y - 30)
        elif k == 'sign':
            rect(x - 1, y - 30, 2, 30, '#2a2f4a'); draw_img(animFrame(PROPS['neon'], gtime + o.seed), x - 9, y - 78)
        elif k == 'crystal':
            h = 8 + R(o.s * 12); glow = 0.5 + math.sin(gtime * 2 + o.seed) * 0.3
            for dx, hh, w in ((-5, h * 0.6, 3), (0, h, 4), (5, h * 0.75, 3)):
                i = 0
                while i < hh:
                    ww = max(1, R(w * (1 - i / hh) + 1)); rect(x + dx - ww / 2, y - i - 1, ww, 1, T['crystal']); i += 1
                rect(x + dx - 1, y - hh * 0.7, 1, hh * 0.4, '#ffffff')
            ALPHA = glow * 0.35; pcircle(x, y - h / 2, h * 0.8, T['crystal']); ALPHA = 1
        elif k == 'plant':
            sway = math.sin(gtime * 1.5 + o.seed) * 2
            for i in (-1, 0, 1):
                hh = 10 + R(o.s * 8) - abs(i) * 3
                for s in range(hh): rect(x + i * 4 + sway * (s / hh), y - s, 1, 1, '#2f8f6a')
                pcircle(x + i * 4 + sway, y - hh, 2, '#c58cff' if i else '#7dff6a')
        elif k == 'pipe':
            rect(x - 18, y - 9, 36, 6, '#4a5068'); rect(x - 18, y - 9, 36, 1, '#7a82a0'); rect(x - 3, y - 12, 6, 12, '#3a3f55'); rect(x - 2, y - 11, 4, 2, T['tile']['glow'])
        elif k == 'beacon':
            rect(x - 1, y - 18, 2, 18, '#3a3f55'); on = math.sin(gtime * 5 + o.seed) > 0
            rect(x - 2, y - 21, 4, 3, T['tile']['glow'] if on else '#222222')
            if on: ALPHA = 0.25; pcircle(x, y - 20, 6, T['tile']['glow']); ALPHA = 1
        elif k == 'vent':
            rect(x - 10, y - 4, 20, 4, '#2a1515')
            for i in range(-8, 10, 4): rect(x + i, y - 3, 2, 3, '#ff7a3c')
            ALPHA = 0.3 + math.sin(gtime * 6 + o.seed) * 0.15; rect(x - 10, y - 8, 20, 4, '#ff5a28'); ALPHA = 1
            if random.random() < 0.05: parts.append(O(x=o.x + rnd(-8, 8), y=y - 6, vx=rnd(-5, 5), vy=rnd(-40, -20), life=0.8, color='#ff7a3c' if random.random() < 0.5 else '#5a3a3a', s=1))

# franjas de peligro en las orillas de los huecos y propulsores bajo las plataformas
def drawLevelDetails():
    D = level.decor
    for idx, ex in enumerate(D.edges):
        x = R(ex - cam.x)
        if x < -10 or x > VW + 10: continue
        for i in range(4): rect(x + (-3 if idx % 2 else 0), GROUND + 3 + i * 4, 3, 2, '#ffd84d' if i % 2 else '#111111')
    for s in solids:
        if not s.plat: continue
        x = R(s.x - cam.x)
        if x < -60 or x > VW + 10: continue
        for fx2 in (x + 6, x + s.w - 10):
            fl = 3 + R(random.random() * 2)
            rect(fx2, s.y + 12, 4, 2, '#4a5068'); rect(fx2 + 1, s.y + 14, 2, fl, '#ff8a1f'); rect(fx2 + 1, s.y + 14, 2, 1, '#ffe14d')
        rect(x + 2, s.y + 3, s.w - 4, 1, D.theme['tile']['glow'])

# polvo espacial en primer plano (parallax más rápido que el mundo)
def drawDust():
    global ALPHA
    for d in level.decor.dust:
        x = (d.x - cam.x * 1.3 * d.v - gtime * 8 * d.v) % VW
        ALPHA = d.a; rect(x, d.y, d.s, d.s, '#cfd6ff')
    ALPHA = 1

# ---------- Bucle principal ----------
async def main():
    FONDOS.load('planetas')
    fade_s = pygame.Surface((VW * S, VH * S)); fade_s.fill((0, 0, 0))
    while True:
        dt = min(0.033, clock.tick(60) / 1000)
        for e in pygame.event.get():
            if e.type == pygame.QUIT: pygame.quit(); return
            if e.type == pygame.KEYDOWN:
                k = KEYMAP.get(e.key)
                if k:
                    if not keys.get(k): pressed[k] = True
                    keys[k] = True
            elif e.type == pygame.KEYUP:
                k = KEYMAP.get(e.key)
                if k: keys[k] = False
            elif e.type == pygame.MOUSEMOTION:
                mouse.x, mouse.y = e.pos[0] / S, e.pos[1] / S
            elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                mouse.x, mouse.y = e.pos[0] / S, e.pos[1] / S; mouse.down = True; pressed['click'] = True
            elif e.type == pygame.MOUSEBUTTONUP and e.button == 1:
                mouse.down = False
            elif e.type == pygame.WINDOWFOCUSLOST:
                mouse.down = False
                for k in keys: keys[k] = False
        update(dt)
        pressed.clear()
        draw()
        pygame.transform.scale(ctx, screen.get_size(), screen)
        flush_text()
        if fade.a > 0: fade_s.set_alpha(int(fade.a * 255)); screen.blit(fade_s, (0, 0))
        pygame.display.flip()
        await asyncio.sleep(0)  # cede el control al navegador en la versión web

if __name__ == '__main__':
    asyncio.run(main())
