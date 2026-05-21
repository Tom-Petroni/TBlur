use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-env-changed=NUKE_SOURCE_PATH");
    println!("cargo:rerun-if-env-changed=PLATFORM_NAME");
    println!("cargo:rerun-if-env-changed=CPP_VERSION");
    println!("cargo:rerun-if-changed=src/tblur_base.cpp");

    let nuke_root = if let Ok(sources) = std::env::var("NUKE_SOURCE_PATH") {
        PathBuf::from(sources)
    } else {
        return;
    };
    let nuke_path = nuke_root.join("include");

    let platform_name = if let Ok(name) = std::env::var("PLATFORM_NAME") {
        name
    } else {
        return;
    };

    let cpp_version = std::env::var("CPP_VERSION").unwrap_or_else(|_| "17".to_string());

    // ----------------------------------------------------------------
    // CUDA backend: compile tblur_cuda.cu with nvcc
    // ----------------------------------------------------------------
    #[cfg(feature = "cuda_backend")]
    {
        println!("cargo:rerun-if-changed=src/tblur_cuda.cu");
        println!("cargo:rerun-if-changed=src/tblur_cuda.h");
        println!("cargo:rerun-if-env-changed=CUDA_PATH");

        let cuda_path = std::env::var("CUDA_PATH")
            .expect("CUDA_PATH environment variable must be set for cuda_backend");
        let cuda_root = PathBuf::from(&cuda_path);
        let nvcc = cuda_root.join("bin").join(if platform_name == "windows" {
            "nvcc.exe"
        } else {
            "nvcc"
        });

        let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
        let cuda_obj = out_dir.join(if platform_name == "windows" {
            "tblur_cuda.obj"
        } else {
            "tblur_cuda.o"
        });

        let src_dir = PathBuf::from("src");

        let mut cmd = std::process::Command::new(&nvcc);
        cmd.arg("-c")
            .arg(src_dir.join("tblur_cuda.cu"))
            .arg("-o")
            .arg(&cuda_obj)
            .arg("-O2")
            .arg("--use_fast_math")
            .arg("-Xptxas").arg("-O3,-dlcm=ca")
            .arg("--std=c++17")
            .arg("-gencode").arg("arch=compute_75,code=sm_75")
            .arg("-gencode").arg("arch=compute_86,code=sm_86")
            .arg("-gencode").arg("arch=compute_89,code=sm_89")
            .arg("-gencode").arg("arch=compute_90,code=sm_90")
            .arg("-gencode").arg("arch=compute_90,code=compute_90")
            .arg(format!("-I{}", src_dir.display()));

        if platform_name == "windows" {
            cmd.arg("--compiler-options").arg("/EHsc,/O2,/DWIN32,/DNOMINMAX,/MD");
        } else {
            cmd.arg("--compiler-options").arg("-fPIC,-O2");
        }

        let status = cmd
            .status()
            .expect("Failed to run nvcc. Is the CUDA toolkit installed?");
        if !status.success() {
            panic!("nvcc compilation failed with exit code: {}", status);
        }

        // Create a static library from the CUDA object using cc
        let mut cuda_lib = cc::Build::new();
        cuda_lib.object(&cuda_obj);
        cuda_lib.compile("tblur-cuda");

        // Link CUDA runtime (static)
        let cuda_lib_dir = if platform_name == "windows" {
            cuda_root.join("lib").join("x64")
        } else {
            cuda_root.join("lib64")
        };
        println!("cargo:rustc-link-search=native={}", cuda_lib_dir.display());
        println!("cargo:rustc-link-lib=static=cudart_static");

        // Windows system libraries required by cudart_static
        if platform_name == "windows" {
            println!("cargo:rustc-link-lib=dylib=user32");
            println!("cargo:rustc-link-lib=dylib=advapi32");
        }
    }

    // ----------------------------------------------------------------
    // C++ Nuke node compilation
    // ----------------------------------------------------------------
    let mut builder = cc::Build::new();
    builder
        .cpp(true)
        .std(&format!("c++{cpp_version}"))
        .file("src/tblur_base.cpp")
        .include(&nuke_path)
        .flag_if_supported("-DGLEW_NO_GLU");

    #[cfg(feature = "cuda_backend")]
    {
        builder.define("TBLUR_CUDA", "1");
        builder.include("src"); // for tblur_cuda.h
    }

    if platform_name == "linux" {
        builder
            .flag("-fPIC")
            .flag_if_supported("-Wno-deprecated-copy-with-user-provided-copy")
            .flag_if_supported("-Wno-ignored-qualifiers")
            .flag_if_supported("-Wno-date-time")
            .flag_if_supported("-Wno-unused-parameter");

        if std::env::var("USE_CXX11_ABI").is_ok() {
            builder.flag("-D_GLIBCXX_USE_CXX11_ABI=1");
        }

        if std::env::var("USING_ZIG").is_ok() {
            builder.define("__gnu_cxx", "std");
        }
    } else if platform_name == "macos" {
        builder
            .flag_if_supported("-Wno-deprecated-copy-with-user-provided-copy")
            .flag_if_supported("-Wno-ignored-qualifiers")
            .flag_if_supported("-Wno-date-time")
            .flag_if_supported("-Wno-unused-parameter");
    } else if platform_name == "windows" {
        builder
            .define("_CPPUNWIND", "1")
            .define("NOMINMAX", "1")
            .define("_USE_MATH_DEFINES", "1")
            .flag("/EHsc");
    }

    builder.compile("tblur-nuke");

    println!("cargo:rustc-link-search=all={}", nuke_root.display());
    println!("cargo:rustc-link-lib=dylib=DDImage");
}
