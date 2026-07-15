project('cdarts','c','cpp','cython',default_options: ['cpp_std=c++17'],version:'1.0.0')
enable_static=get_option('link-static')
enable_python=get_option('build-python')
enable_tests=get_option('build-tests')
onnxruntime_dir=get_option('onnxruntime-dir')
python_include_dir=get_option('python-include-dir')
# Validate the c++ Standard
if get_option('cpp_std') != 'c++17'
    error('This project requires c++17 support')
endif
extra_compile_args = ['-Wno-sign-compare','-Wno-deprecated-declarations']
extra_link_args=['-w']

if get_option('debug-memory')
    message('compile with dmalloc!')
    extra_compile_args+=['-g','-ddmalloc','-ldmallocxx','-ddmalloc_func_check' ]
    extra_link_args+=['-ldmalloc']
else
    extra_compile_args+=['-O2']
endif

if build_machine.system()=='darwin'
    if meson.get_compiler('c').get_id() == 'clang'
        extra_compile_args += ['-mmacosx-version-min=11.0','-std=c++17','-stdlib=libc++']
    endif
endif

message('check the deps libray.....')
cc = meson.get_compiler('cpp')
fmt_pkg_dep=dependency('fmt', method: 'pkg-config', required:false,version:'>=9.1.0')
if enable_static and fmt_pkg_dep.found()
    fmt_dep = declare_dependency(dependencies: [fmt_pkg_dep], compile_args: ['-DFMT_HEADER_ONLY'])
elif fmt_pkg_dep.found()
    fmt_dep = fmt_pkg_dep
else
    fmt_dep = cc.find_library(
        'fmt', required: true,
        has_headers: ['fmt/core.h'],
        static: enable_static
    )
endif

onnxrt_dep=dependency('onnxruntime', method: 'pkg-config', version:'>=1.12.1',required:false)
if not onnxrt_dep.found()
    if onnxruntime_dir == ''
        error('onnxruntime dependency not found. Install pkg-config metadata or pass -Donnxruntime-dir=/path/to/onnxruntime')
    endif
    onnxrt_dep = declare_dependency(
        link_args : ['-L' + join_paths(onnxruntime_dir, 'lib'), '-lonnxruntime'],
        compile_args : ['-I' + join_paths(onnxruntime_dir, 'include')],
        version:'1.12.1'
    )
endif

zlib_dep = dependency('zlib', version : '>=1.2.11', required : true,static: false)
jsoncpp_dep = dependency('jsoncpp', version : '>=1.7.4',required : true,static: false)
proto_dep = dependency('protobuf', version : '>=3.17.3',
    method: 'pkg-config', required : true,
    static: false,modules: ['protobuf::libprotobuf-lite']
)
base_deps=[zlib_dep,jsoncpp_dep,proto_dep,onnxrt_dep,fmt_dep]

message('building proto source.....')
protoc = find_program('protoc', required : false)
if not protoc.found()
    protoc = find_program(
        join_paths(meson.project_build_root(), '..', 'deps', 'sysroot', 'usr', 'bin', 'protoc'),
        required : false,
    )
endif
if not protoc.found()
  error('MESON_SKIP_TEST: protoc tool and/or protobuf pkg-config dependency not found')
endif
proto_sources=[]
proto_headers=[]
fs=import('fs')
foreach pfile : files('src/utils/darts.proto') 
    proto_path=fs.parent(pfile)
    base_name=fs.stem(pfile)
    proto_list = custom_target(
        base_name,
        output : ['@BASENAME@.pb.cc', '@BASENAME@.pb.h'],
        input : pfile,
        command : [protoc, '--proto_path=@CURRENT_SOURCE_DIR@/'+proto_path, '--cpp_out=@OUTDIR@','@PLAINNAME@'],
    )
    proto_sources += proto_list[0]
    proto_headers += proto_list[1]
endforeach

my_proto_deb=declare_dependency(
    link_with : [
        static_library('proto',
            proto_sources,dependencies:proto_dep,
            cpp_args : extra_compile_args,
            c_args : extra_compile_args, pic:true)
    ],
    link_args : ['-ldl' ],
    sources:[proto_headers]
)
base_deps+=my_proto_deb

message('buiding darts lib....')
incdir = include_directories('src')
darts_lib=both_libraries('darts', 'src/main/darts.cxx', 
    version : '1.2.3', soversion : '0',
    dependencies:base_deps,include_directories : incdir,
    cpp_args : extra_compile_args,c_args : extra_compile_args,
    pic:true
)
darts_dict_repack = executable(
    'darts-dict-repack',
    'src/main/dict_repack.cxx',
    dependencies : [zlib_dep, proto_dep, my_proto_deb],
    include_directories : incdir,
    cpp_args : extra_compile_args,
    install : false,
)
if enable_tests
    message('buiding darts test....')
    test_rpath = ''
    if onnxruntime_dir != ''
        test_rpath = join_paths(onnxruntime_dir, 'lib')
    endif
    darts_test_exe = executable('darts_test','test/test_func.cc',
        dependencies:base_deps,include_directories : incdir,
        cpp_args : extra_compile_args,c_args : extra_compile_args,
        build_rpath: test_rpath
    )
    test(
        'darts-core',
        darts_test_exe,
        workdir : meson.project_source_root(),
        timeout : 120,
    )
endif

if enable_python
    message('building cython so...')
    py = import('python').find_installation('python3')
    python_compile_args = []
    if python_include_dir != ''
        python_compile_args += ['-I' + python_include_dir]
    endif
    pylib=py.extension_module(
        'cdarts',
        'python/darts/cdarts.pyx',
        override_options : ['cython_language=cpp'],
        include_directories : [include_directories('src/main/')],
        link_with : darts_lib.get_static_lib(),
        install : true, install_dir : join_paths(py.get_install_dir(), 'darts'),
        cpp_args : extra_compile_args + python_compile_args,
        c_args : extra_compile_args + python_compile_args,
        link_args: extra_link_args,
        install_rpath:'$ORIGIN'
    )
endif
