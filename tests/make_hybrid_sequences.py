# writes tests/seq_hybrid_*.txt, see tests/README.md
import itertools, os, shutil

TESTS = os.path.dirname(os.path.abspath(__file__))

CLOSE = '{0.539, 1.860, 0.845}, {-0.337, 1.203, -0.031}, {0, 1, 0}, {60}'
PATH = ('"smooth 1 loop 1 dur 4 ; '
        '{0.185, 1.860, 1.199}, {-0.691, 1.203, 0.323}, {0, 1, 0}, {60} ; '
        '{0.893, 1.860, 0.491}, {0.017, 1.203, -0.385}, {0, 1, 0}, {60} ; '
        '{0.185, 1.860, 1.199}, {-0.691, 1.203, 0.323}, {0, 1, 0}, {60}"')

COMMON_HEADER = """#
# Every configuration runs twice: from a static close view into the grid, where most
# instances are hidden behind the nearest bunnies, and as `_motion` along a sideways
# camera path, which is what reaches the second raster pass (disocclusion). The static
# view is camera path 1, a held path overrides --camerastring.
# --hybridmatchcheck traces the primary ray as well and logs an error when more pixels
# disagree with the raster than edge rounding explains.
"""

BASE = ['--sequenceframes 32', '--sequenceresetframes 8', '--renderer 3', '--hybridmatchcheck 1',
        '--dlss 0', '--culling 1', '--streaming 1', '--gridunique 0',
        '--blassharing 1', '--blasmerging 1', '--blascaching 1', '--hitstreaming 0',
        '--addcamerapath ' + PATH, '--addcamerapath STILL']


def write(name, header, configs, base=BASE, close=CLOSE, path=PATH, extra=()):
    """configs: list of (sequence name, [params]); each becomes static + motion, plus one
    sequence per (suffix, camera path) in extra."""
    still = '"smooth 0 loop 0 dur 1 ; ' + close + ' ; ' + close + '"'
    base = [('--addcamerapath ' + (still if 'STILL' in b else path)) if b.startswith('--addcamerapath') else b for b in base]
    base += ['--addcamerapath ' + p for _, p in extra]
    lines = [header.rstrip() + '\n' + COMMON_HEADER]
    first = True
    for seq, params in configs:
        for motion in (False, True):
            lines.append(f'SEQUENCE "{seq}{"_motion" if motion else ""}"')
            if first:
                lines += base
                first = False
            if not motion:
                lines += params
                lines.append('--runcamerapath 1 1')
            else:
                lines.append('--runcamerapath 0 32')
            lines.append('')
        for i, (suffix, _) in enumerate(extra):
            lines += [f'SEQUENCE "{seq}_{suffix}"', f'--runcamerapath {2 + i} 32', '']
    with open(os.path.join(TESTS, name), 'w', newline='\n') as f:
        f.write('\n'.join(lines))
    print(name, (2 + len(extra)) * len(configs))


def onoff(v):
    return '1' if v else '0'


# blas reuse, as seq_raytrace_blas.txt
for unique in (0, 1):
    configs = []
    for share, merge, cache in itertools.product((0, 1), repeat=3):
        configs.append((f'hy_unique{unique}_share{share}_merge{merge}_cache{cache}',
                        [f'--gridunique {unique}', f'--blassharing {share}', f'--blasmerging {merge}',
                         f'--blascaching {cache}', '--blasdedup 0', '--streaming 1']))
    configs.append((f'hy_unique{unique}_dedup', ['--blassharing 1', '--blasmerging 1', '--blascaching 1', '--blasdedup 1']))
    configs.append((f'hy_unique{unique}_preloaded', ['--blasdedup 0', '--streaming 0']))
    write(f'seq_hybrid_blas{unique}.txt', f"""# Hybrid path: --blassharing x --blasmerging x --blascaching with --gridunique {unique},
# plus --blasdedup and preloaded geometry. Visible instances must match the raster whatever
# blas reuse picks for the hidden ones.""", configs)

# culling
configs = []
for culling in (0, 1):
    for scale in (1, 8):
        for loderror in (1, 0.1):
            configs.append((f'hy_cull{culling}_scale{scale}_err{str(loderror).replace(".", "")}',
                            [f'--culling {culling}', f'--cullederrorscale {scale}', f'--loderror {loderror}']))
# raster only flags, the hybrid path always runs two passes: same images as hy_cull1_scale8_err1
for twopass, reject in itertools.product((0, 1), repeat=2):
    configs.append((f'hy_twopass{twopass}_reject{reject}',
                    ['--culling 1', '--cullederrorscale 8', '--loderror 1',
                     f'--twopassculling {twopass}', f'--twopassrejectlists {reject}']))
# hiz handed over from the raster renderer's two passes
configs.append(('ra_twopass1', ['--renderer 0']))
configs.append(('hy_after_raster', ['--renderer 3']))
# primitive culling in the raster pass, ignored without --culling
configs.append(('hy_primcull1', ['--culling 1', '--primitiveculling 1']))
configs.append(('hy_primcull1_cull0', ['--culling 0', '--primitiveculling 1']))
write('seq_hybrid_culling.txt', """# Hybrid path: --culling x --cullederrorscale x --loderror, --primitiveculling.
# Culling off has no hidden instances and a single raster pass. --twopassculling and
# --twopassrejectlists are raster only, then a switch from the raster renderer.""", configs)

# traversal
configs = []
for persistent, discrete, sorting in itertools.product((0, 1), repeat=3):
    configs.append((f'hy_persistent{persistent}_discrete{discrete}_sort{sorting}',
                    [f'--persistenttraversal {persistent}', f'--hybriddiscretelod {discrete}', f'--instancesorting {sorting}']))
write('seq_hybrid_traversal.txt', """# Hybrid path: --persistenttraversal x --hybriddiscretelod x --instancesorting.""", configs)

# cached blas
configs = []
for unique, discrete, levels, tolerance in itertools.product((0, 1), (0, 1), (1, 3), (-1, 1)):
    configs.append((f'hy_unique{unique}_discrete{discrete}_levels{levels}_tol{max(tolerance, 0)}',
                    [f'--gridunique {unique}', f'--hybriddiscretelod {discrete}', f'--hybridcachinglevels {levels}',
                     f'--hybridcachedtolerance {tolerance}', f'--cachingpushlevels {max(tolerance, 0)}',
                     f'--hybridmatchcheck {onoff(tolerance < 0)}']))
write('seq_hybrid_cached.txt', """# Hybrid path: --hybriddiscretelod x --hybridcachinglevels x --hybridcachedtolerance. Discrete lod draws
# the cached level, so its blas matches. The tolerance is a known mismatch, no match check.""", configs)

# streaming
configs = []
for unique in (0, 1):
    for hit, hitlod in ((0, 0), (1, 0), (1, 1)):
        configs.append((f'hy_unique{unique}_hit{hit}_hitlod{hitlod}',
                        [f'--gridunique {unique}', f'--hitstreaming {hit}', f'--hitstreaminglod {hitlod}', '--maxresidentgroups 65536']))
configs.append(('hy_hitlod_tightbudget', ['--gridunique 1', '--hitstreaming 1', '--hitstreaminglod 1', '--maxresidentgroups 2048']))
configs.append(('hy_hitlod_lens', ['--gridunique 0', '--maxresidentgroups 65536', '--opticalmode 3']))
write('seq_hybrid_streaming.txt', """# Hybrid path: --hitstreaming / --hitstreaminglod over both geometry layouts, a residency budget
# that cannot hold the grid, and the magnifying lens. The rasterized primary hit registers like a
# traced one.""", configs)

# ptlas
configs = []
for partitions, ptlas, earlyout in ((0, 0, 0), (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)):
    configs.append((f'hy_partitions{partitions}_ptlas{ptlas}_earlyout{earlyout}',
                    [f'--partitions {partitions}', f'--ptlas {ptlas}', f'--partitionearlyout {earlyout}']))
# only the hybrid frustum test keeps visible partitions awake
for ptlas in (0, 1):
    configs.append((f'hy_ptlas{ptlas}_earlyout_frustumonly',
                    ['--partitions 1', f'--ptlas {ptlas}', '--partitionearlyout 1', '--partitionfrustumrevalidate 0',
                     '--partitionrevalidateframes 4096']))
# above the grid centre, far partitions visible: held, then a +-60 degree yaw sweep moving them in and out of the frustum
EYE = '{-100, 10, -77}'
OVERVIEW = f'"smooth 0 loop 0 dur 1 ; {EYE}, {{-130, 2, -100}}, {{0, 1, 0}}, {{60}} ; {EYE}, {{-130, 2, -100}}, {{0, 1, 0}}, {{60}}"'
PAN = ('"smooth 1 loop 1 dur 4 ; '
       + ' ; '.join(f'{EYE}, {{{-100 + dx}, 2, {-77 + dz}}}, {{0, 1, 0}}, {{60}}'
                    for dx, dz in ((-30, -23), (-35, 14.5), (-30, -23), (4.9, -37.5), (-30, -23)))
       + '"')
def copy_cfg(src, dst):
    # the scene of the raster/ray tracing file, its header names the hybrid sequence file
    with open(os.path.join(TESTS, src + '.cfg'), newline='', encoding='utf-8') as f:
        text = f.read()
    with open(os.path.join(TESTS, dst + '.cfg'), 'w', newline='', encoding='utf-8') as f:
        f.write(text.replace(src + '.txt', dst + '.txt', 1))


write('seq_hybrid_ptlas.txt', """# Hybrid path: --partitions x --ptlas x --partitionearlyout on 4096 instances. The early-out
# processes every partition in the frustum, `_frustumonly` turns the other wake-ups off.
# `_overview` looks over the grid, `_pan` turns so partitions enter and leave the frustum.""", configs,
      extra=[('overview', OVERVIEW), ('pan', PAN)])
copy_cfg('seq_ptlas', 'seq_hybrid_ptlas')

# materials
configs = [
    ('hy_mat_default', ['--forcetwosided 0', '--flipwinding 0', '--extmeshshader 0']),
    ('hy_mat_forcetwosided', ['--forcetwosided 1', '--flipwinding 0', '--extmeshshader 0']),
    ('hy_mat_flipwinding', ['--forcetwosided 0', '--flipwinding 1', '--extmeshshader 0']),
    ('hy_mat_extmeshshader', ['--forcetwosided 0', '--flipwinding 0', '--extmeshshader 1']),
    ('hy_mat_rayquery', ['--forcetwosided 0', '--flipwinding 0', '--extmeshshader 0', '--rayquery 1']),
    ('hy_mat_rayquery_forcetwosided', ['--forcetwosided 1', '--rayquery 1']),
    # alpha masked clusters stay on the hardware path
    ('hy_mat_swraster', ['--forcetwosided 0', '--rayquery 0', '--swraster 1', '--swrasterthreshold 64']),
]
write('seq_hybrid_materials.txt', """# Hybrid path on the material fixture: alpha mask and blend, two sided, textures, and the
# raster state the hybrid raster pass shares with the raster renderer.""", configs,
      close='{-14, 5, 14}, {0, 0, 0}, {0, 1, 0}, {60}',
      path=('"smooth 1 loop 1 dur 4 ; {-20, 6, 20}, {0, 0, 0}, {0, 1, 0}, {60} ; '
            '{20, 6, 20}, {0, 0, 0}, {0, 1, 0}, {60} ; {-20, 6, 20}, {0, 0, 0}, {0, 1, 0}, {60}"'))
copy_cfg('seq_materials', 'seq_hybrid_materials')

# ray query
configs = [(f'hy_rq_vis{v}', ['--rayquery 1', f'--visualize {v}', '--dlss 0', '--opticalmode 0']) for v in (0, 2, 5, 10)]
configs += [('hy_rq_dlss', ['--rayquery 1', '--visualize 0', '--dlss 1', '--dlssquality 2', '--opticalmode 0'])]
configs += [(f'hy_rq_optical{o}', ['--rayquery 1', '--visualize 0', '--dlss 0', f'--opticalmode {o}']) for o in (1, 2, 3)]
configs += [(f'hy_rq_bounces{b}', ['--rayquery 1', '--opticalmode 0', f'--pathtracebounces {b}']) for b in (0, 3)]
configs += [('hy_rq_ambient', ['--rayquery 1', '--pathtracebounces 1', '--pathtraceambient 1'])]
configs += [(f'hy_rq_unique{unique}_hit{hit}_hitlod{hitlod}',
             ['--rayquery 1', '--pathtraceambient 0', f'--gridunique {unique}', f'--hitstreaming {hit}', f'--hitstreaminglod {hitlod}'])
            for unique, hit, hitlod in ((0, 1, 0), (1, 1, 1))]
write('seq_hybrid_rayquery.txt', """# Hybrid path with --rayquery: the path tracer as a compute shader with ray queries over
# visualizations, DLSS, the optical modes, bounce counts and hit streaming. The materials
# fixture runs it in seq_hybrid_materials.txt.""", configs)

# visualize
configs = [(f'hy_vis{v}', [f'--visualize {v}', '--dlss 0', '--opticalmode 0']) for v in range(12)]
configs += [(f'hy_dlss_q{q}', ['--visualize 0', '--dlss 1', f'--dlssquality {q}', '--opticalmode 0']) for q in (0, 2, 5)]
configs += [(f'hy_optical{o}', ['--visualize 0', '--dlss 0', f'--opticalmode {o}']) for o in (1, 2, 3)]
configs += [(f'hy_bounces{b}', ['--visualize 0', '--dlss 0', '--opticalmode 0', f'--pathtracebounces {b}']) for b in (0, 3)]
write('seq_hybrid_visualize.txt', """# Hybrid path: every visualization, DLSS, the optical modes and bounce counts.""", configs)

# compute rasterization, every entry from the same settings so none inherits the previous one's
SW = {'--swraster': '1', '--swrasterthreshold': '64', '--culling': '1', '--rayquery': '0', '--dlss': '0',
      '--visualize': '0', '--loderror': '1', '--hybridmatchcheck': '1', '--renderclusterbits': '20'}


def sw(name, **overrides):
    opts = dict(SW, **{'--' + k: v for k, v in overrides.items()})
    return (name, [f'{k} {v}' for k, v in opts.items()])


configs = [
    sw('hy_sw_t8', swrasterthreshold='8'),
    sw('hy_sw_t8_err01', swrasterthreshold='8', loderror='0.1'),
    sw('hy_sw_t64'),
    sw('hy_sw_t64_rayquery', rayquery='1'),
    sw('hy_sw_t64_dlss', dlss='1'),
] + [sw(f'hy_sw_t64_vis{v}', visualize=str(v)) for v in (2, 4, 10)] + [
    sw('hy_sw_t64_limit', hybridmatchcheck='0', renderclusterbits='12'),
    # without culling it falls back to the hardware path
    sw('hy_sw_t64_cull0', culling='0'),
]
write('seq_hybrid_swraster.txt', """# Hybrid path: --swraster x --swrasterthreshold, small opaque clusters of both raster passes
# go to the compute raster. A large threshold sends nearly all clusters there.""", configs)

# limits
configs = [
    ('hy_limit_renderclusters', ['--hybridmatchcheck 0', '--renderclusterbits 12', '--rendertraversalbits 20']),
    # fits (~1300 tasks) but the persistent kernel runs out of read slots early
    ('hy_limit_traversal', ['--hybridmatchcheck 1', '--renderclusterbits 20', '--rendertraversalbits 11']),
    ('hy_limit_traversal_overflow', ['--hybridmatchcheck 0', '--renderclusterbits 20', '--rendertraversalbits 8']),
    ('hy_limit_both', ['--hybridmatchcheck 0', '--renderclusterbits 10', '--rendertraversalbits 10']),
]
write('seq_hybrid_limits.txt', """# Hybrid path: the render list (and with it the hybrid and reject lists) and the traversal
# queue overflowing. A degraded image is expected, nothing else, so no match check where
# a list overflows.""", configs)
