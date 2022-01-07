project('cdarts','c','cpp','cython',default_options: ['cpp_std=c++17'],version:'1.0.0')

message('check the deps libray.....')
zlib_dep = dependency('zlib', version : '>=1.2.11', required : true,static: true)
hdf5_dep=dependency('hdf5', version : '>=1.12.1', required : true,static: true)
jsoncpp_dep = dependency('jsoncpp', version : '>=1.9.4',required : true,static: true)
eigen_dep = dependency('eigen3', version : '>=3.4.0', required : true,static: true)
proto_dep = dependency('protobuf', version : '>=3.19.1', required : false,static: true)
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
incdir = include_directories('src')
darts_lib=both_libraries('cdarts', 'src/main/darts.cxx',proto_gens, version : '1.2.3', soversion : '0',dependencies:base_deps,include_directories : incdir)
darts_exe = executable('darts', 'src/main/main.cpp', proto_gens, dependencies:base_deps, include_directories : incdir)
darts_test_exe = executable('darts_test','test/test_func.cc',proto_gens,dependencies:base_deps,include_directories : incdir)
if get_option('build-python')
    message('gen cython so...')
    py = import('python').find_installation('python3.9')
    dep_py = py.dependency(disabler:true)
    py.extension_module(
        'cdarts',
        'python/darts/cdarts.pyx',
        override_options : ['cython_language=cpp'],
        dependencies : [dep_py],
        include_directories : [include_directories('src/main/')],
        link_with : darts_lib.get_static_lib(),
    )
endif