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
xkbcommon = dependency('xkbcommon')
cairo = dependency('cairo')
pangocairo = dependency('pangocairo')
libcurl = dependency('libcurl')
cjson = dependency('libcjson')
drm = dependency('libdrm')
math = meson.get_compiler('c').find_library('m')
threads = dependency('threads')

executable('leaves-compositor',
  'compositor.c',
  'renderer.c',
  'feed.c',
  'input.c',
  'spring.c',
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
