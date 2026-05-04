project('marshal-compositor', 'c',
  version : '0.1.0',
  default_options : [
    'c_std=c11',
    'warning_level=1',
  ],
)

add_project_arguments('-DWLR_USE_UNSTABLE', language : 'c')

wlroots = dependency('wlroots-0.18')
wayland_server = dependency('wayland-server')
wayland_protocols = dependency('wayland-protocols')
xkbcommon = dependency('xkbcommon')
cairo = dependency('cairo')
pangocairo = dependency('pangocairo')
libcurl = dependency('libcurl')
cjson = dependency('libcjson')
drm = dependency('libdrm')
jpeg = dependency('libjpeg')
rsvg = dependency('librsvg-2.0', required : false)
xcb = dependency('xcb')
xcb_icccm = dependency('xcb-icccm')
alsa = dependency('alsa')
libsystemd = dependency('libsystemd')
math = meson.get_compiler('c').find_library('m')
threads = dependency('threads')

# Generate xdg-shell protocol header (needed by wlr_xdg_shell.h)
wayland_scanner = find_program('wayland-scanner')
wl_proto_dir = wayland_protocols.get_variable('pkgdatadir')

xdg_shell_xml = wl_proto_dir / 'stable/xdg-shell/xdg-shell.xml'
xdg_shell_header = custom_target('xdg-shell-protocol-header',
  input : xdg_shell_xml,
  output : 'xdg-shell-protocol.h',
  command : [wayland_scanner, 'server-header', '@INPUT@', '@OUTPUT@'],
)

# Generate wlr-layer-shell protocol header (enum definitions for layer shell)
layer_shell_xml = meson.current_source_dir() / 'protocols/wlr-layer-shell-unstable-v1.xml'
layer_shell_header = custom_target('wlr-layer-shell-protocol-header',
  input : layer_shell_xml,
  output : 'wlr-layer-shell-unstable-v1-protocol.h',
  command : [wayland_scanner, 'server-header', '@INPUT@', '@OUTPUT@'],
)

executable('marshal-compositor',
  'compositor.c',
  'renderer.c',
  'feed.c',
  'input.c',
  'spring.c',
  'status.c',
  xdg_shell_header,
  layer_shell_header,
  dependencies : [
    wlroots,
    wayland_server,
    xkbcommon,
    cairo,
    pangocairo,
    libcurl,
    cjson,
    drm,
    jpeg,
    xcb,
    xcb_icccm,
    alsa,
    libsystemd,
    math,
    threads,
    rsvg,
  ],
  c_args : rsvg.found() ? ['-DMARSHAL_HAS_RSVG=1'] : [],
  install : true,
)
