project('marshal-terminal', 'c',
  version : '0.1.0',
  default_options : [
    'c_std=c11',
    'warning_level=2',
  ],
)

wayland_client = dependency('wayland-client')
wayland_protocols = dependency('wayland-protocols')
cairo = dependency('cairo')
pangocairo = dependency('pangocairo')
xkbcommon = dependency('xkbcommon')
tsm = dependency('libtsm')
util = meson.get_compiler('c').find_library('util')  # for openpty
math = meson.get_compiler('c').find_library('m')

# Generate xdg-shell client protocol
wayland_scanner = find_program('wayland-scanner')
wl_proto_dir = wayland_protocols.get_variable('pkgdatadir')

xdg_shell_xml = wl_proto_dir / 'stable/xdg-shell/xdg-shell.xml'

xdg_shell_header = custom_target('xdg-shell-client-header',
  input : xdg_shell_xml,
  output : 'xdg-shell-client-protocol.h',
  command : [wayland_scanner, 'client-header', '@INPUT@', '@OUTPUT@'],
)

xdg_shell_code = custom_target('xdg-shell-client-code',
  input : xdg_shell_xml,
  output : 'xdg-shell-protocol.c',
  command : [wayland_scanner, 'private-code', '@INPUT@', '@OUTPUT@'],
)

executable('marshal-terminal',
  'terminal.c',
  xdg_shell_header,
  xdg_shell_code,
  dependencies : [
    wayland_client,
    cairo,
    pangocairo,
    xkbcommon,
    tsm,
    util,
    math,
  ],
  install : true,
)
