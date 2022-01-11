project('cdarts','c','cpp','cython',default_options: ['cpp_std=c++17'],version:'1.0.0')
#useage:like blow
#CC=gcc CXX=g++ meson setup directory -Dcmake_prefix_path=<Path1> -Dpkg_config_path=<Path1> -Dbuild-python=false
#CC=clang CXX=clang++ meson setup directory
#meson compile -C directory
#meson install -C directory
enable_static=get_option('link-static')
# Validate the c++ Standard
if get_option('cpp_std') != 'c++17'
    error('This project requires c++17 support')
endif
extra_args = []
if meson.get_compiler('c').get_id() == 'clang'
    if build_machine.system()=='darwin'
        extra_args += ['-mmacosx-version-min=11.0','-std=c++17','-stdlib=libc++']
    endif
endif

message('check the deps libray.....')
zlib_dep = dependency('zlib', version : '>=1.2.11',method: 'cmake', required : true,static: enable_static)
hdf5_dep=dependency('hdf5', version : '>=1.10.3', required : true,static: enable_static)
jsoncpp_dep = dependency('jsoncpp', version : '>=1.7.4',required : true,static: enable_static)
eigen_dep = dependency('eigen3', version : '>=3.3.7', required : true,static: enable_static)
proto_dep = dependency('protobuf', version : '>=3.17.3',
    method: 'pkg-config', required : true,
    static: enable_static,modules: ['protobuf::libprotobuf']
)
base_deps=[zlib_dep,hdf5_dep,jsoncpp_dep,jsoncpp_dep,eigen_dep,proto_dep]

message('set gen the proto source.....')
protoc = find_program('protoc', required : false)
if not protoc.found()
  error('MESON_SKIP_TEST: protoc tool and/or protobuf pkg-config dependency not found')
endif

fs=import('fs')
proto_sources=[]
proto_headers=[]

foreach pfile : files('src/utils/darts.proto') 
    proto_path=fs.parent(pfile)
    base_name=fs.stem(pfile)
    proto_list = custom_target(
        base_name,
        output : ['@BASENAME@.pb.cc', '@BASENAME@.pb.h'],
        input : pfile,
        command : [protoc, '--proto_path=@CURRENT_SOURCE_DIR@/'+proto_path, '--cpp_out=@OUTDIR@','@PLAINNAME@'],
    )
    # gen = generator(protoc,
    #     output    : ['@BASENAME@.pb.cc', '@BASENAME@.pb.h'],
    #     arguments : ['--proto_path=@CURRENT_SOURCE_DIR@/'+proto_path, '--cpp_out=@BUILD_DIR@',proto_name])
    # proto_list=gen.process(pfile)
    proto_headers+=proto_list[1]
    proto_sources+=proto_list
endforeach

proto_lib=static_library('proto',proto_sources,dependencies:proto_dep,
    cpp_args : extra_args,c_args : extra_args
)
my_proto_deb=declare_dependency(
    link_with : [proto_lib],
    link_args : ['-ldl' ],
    sources:[proto_headers]
)
base_deps+=my_proto_deb

message('set gen darts main exe....')

cur_dir=meson.current_source_dir()
incdir = include_directories('src')
darts_lib=both_libraries('darts', 'src/main/darts.cxx', 
    version : '1.2.3', soversion : '0',
    dependencies:base_deps,include_directories : incdir,
    cpp_args : extra_args,c_args : extra_args
)
darts_exe = executable('darts.bin', 'src/main/main.cpp', 
    dependencies:base_deps, include_directories : incdir,
    install : true, install_dir : cur_dir/'python/darts/bin',
    cpp_args : extra_args,c_args : extra_args
)
darts_test_exe = executable('darts_test','test/test_func.cc',
    dependencies:base_deps,include_directories : incdir,
    cpp_args : extra_args,c_args : extra_args
)
if get_option('build-python')
    message('gen cython so...')
    py = import('python').find_installation('python3.9')
    dep_py = py.dependency(required:true)
    pylib=py.extension_module(
        'cdarts',
        'python/darts/cdarts.pyx',
        override_options : ['cython_language=cpp'],
        dependencies : [dep_py],
        include_directories : [include_directories('src/main/')],
        link_with : darts_lib.get_static_lib(),
        install : true, install_dir : cur_dir/'python/darts',
        cpp_args : extra_args,c_args : extra_args
    )
endif
