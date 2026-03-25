project('leaves-compositor', 'c',
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

executable('leaves-compositor',
  'compositor.c',
  'renderer.c',
  'feed.c',
  'input.c',
  'spring.c',
  xdg_shell_header,
  dependencies : [
    wlroots,
    wayland_server,
    xkbcommon,
    cairo,
    pangocairo,
    libcurl,
    cjson,
    drm,
    math,
    threads,
  ],
  install : true,
)
