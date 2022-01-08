project('cdarts','c','cpp','cython',default_options: ['cpp_std=c++17'],version:'1.0.0')
# Validate the c++ Standard
if get_option('cpp_std') != 'c++17'
    error('This project requires c++17 support')
endif
#-Dcmake_prefix_path=<Path1> -Dpkg_config_path=<Path1>
message('check the deps libray.....')
zlib_dep = dependency('zlib', version : '>=1.2.11',method: 'cmake', required : true,static: true)
hdf5_dep=dependency('hdf5', version : '>=1.10.6', required : true,static: true)
jsoncpp_dep = dependency('jsoncpp', version : '>=1.9.4',required : true,static: true)
eigen_dep = dependency('eigen3', version : '>=3.3.9', required : true,static: true)
proto_dep = dependency('protobuf', version : '>=3.17.3',method: 'pkg-config', required : true,static: true,modules: ['protobuf::libprotobuf'])
base_deps=[zlib_dep,hdf5_dep,jsoncpp_dep,jsoncpp_dep,eigen_dep,proto_dep]

message('set gen the proto source.....')

protoc = find_program('protoc', required : false)
if not protoc.found()
  error('MESON_SKIP_TEST: protoc tool and/or protobuf pkg-config dependency not found')
endif
fs=import('fs')
proto_gens=[]
foreach pfile : files('src/utils/darts.proto') 
    proto_path=fs.parent(pfile)
    proto_name=fs.name(pfile)
    gen = generator(protoc,
        output    : ['@BASENAME@.pb.cc', '@BASENAME@.pb.h'],
        arguments : ['--proto_path=@CURRENT_SOURCE_DIR@/'+proto_path, '--cpp_out=@BUILD_DIR@',proto_name])
    proto_gens+=gen.process(pfile)
endforeach
message('set gen darts main exe....')
# blaspp_dep = declare_dependency(
#     link_args : ['-L' + blaspp_lib_dir, '-l' + blaspp_lib],
#     include_directories : include_directories(blaspp_header_dir))

incdir = include_directories('src')
darts_lib=both_libraries('darts', 'src/main/darts.cxx',proto_gens, 
        version : '1.2.3', soversion : '0',
        dependencies:base_deps,include_directories : incdir,
)
darts_exe = executable('darts', 'src/main/main.cpp', proto_gens, dependencies:base_deps, include_directories : incdir)
darts_test_exe = executable('darts_test','test/test_func.cc',proto_gens,dependencies:base_deps,include_directories : incdir)
if get_option('build-python')
    message('gen cython so...')
    py = import('python').find_installation('python3')
    dep_py = py.dependency(required:true)
    pylib=py.extension_module(
        'cdarts',
        'python/darts/cdarts.pyx',
        override_options : ['cython_language=cpp'],
        dependencies : [dep_py],
        include_directories : [include_directories('src/main/')],
        link_with : darts_lib.get_static_lib(),
        install : true, install_dir : 'python/darts'
    )
endif