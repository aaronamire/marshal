project('marshal-notifyd', 'c',
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
sdbus = dependency('libsystemd')
math = meson.get_compiler('c').find_library('m')

wayland_scanner = find_program('wayland-scanner')
wl_proto_dir = wayland_protocols.get_variable('pkgdatadir')

# xdg-shell client protocol (needed by layer-shell which references xdg_popup)
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

# Layer shell protocol (reuse from compositor)
layer_shell_xml = meson.current_source_dir() / '../compositor/protocols/wlr-layer-shell-unstable-v1.xml'

layer_shell_header = custom_target('layer-shell-client-header',
  input : layer_shell_xml,
  output : 'wlr-layer-shell-unstable-v1-client-protocol.h',
  command : [wayland_scanner, 'client-header', '@INPUT@', '@OUTPUT@'],
)

layer_shell_code = custom_target('layer-shell-client-code',
  input : layer_shell_xml,
  output : 'wlr-layer-shell-unstable-v1-protocol.c',
  command : [wayland_scanner, 'private-code', '@INPUT@', '@OUTPUT@'],
)

executable('marshal-notifyd',
  'notifyd.c',
  xdg_shell_header,
  xdg_shell_code,
  layer_shell_header,
  layer_shell_code,
  dependencies : [
    wayland_client,
    cairo,
    pangocairo,
    sdbus,
    math,
  ],
  install : true,
)
