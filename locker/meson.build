project('marshal-locker', 'c',
  version : '0.2.0',
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
pam = meson.get_compiler('c').find_library('pam')
math = meson.get_compiler('c').find_library('m')

wayland_scanner = find_program('wayland-scanner')
wl_proto_dir = wayland_protocols.get_variable('pkgdatadir')

# ext-session-lock-v1 (replaces layer-shell + input-inhibitor)
ext_lock_xml = wl_proto_dir / 'staging/ext-session-lock/ext-session-lock-v1.xml'

ext_lock_header = custom_target('ext-session-lock-client-header',
  input : ext_lock_xml,
  output : 'ext-session-lock-v1-client-protocol.h',
  command : [wayland_scanner, 'client-header', '@INPUT@', '@OUTPUT@'],
)

ext_lock_code = custom_target('ext-session-lock-client-code',
  input : ext_lock_xml,
  output : 'ext-session-lock-v1-protocol.c',
  command : [wayland_scanner, 'private-code', '@INPUT@', '@OUTPUT@'],
)

executable('marshal-locker',
  'locker.c',
  ext_lock_header,
  ext_lock_code,
  dependencies : [
    wayland_client,
    cairo,
    pangocairo,
    xkbcommon,
    pam,
    math,
  ],
  install : true,
)
