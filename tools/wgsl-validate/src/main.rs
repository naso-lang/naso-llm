//! Validate a WGSL file with naga: parse then validate.
//!
//! This is the layer a GPU driver rejects first. naga is wgpu's own WGSL
//! front-end, so a shader that passes here is one a WebGPU implementation can
//! accept; a shader that fails is one no browser would have run. It is strictly
//! stronger than parsing and needs no GPU.
//!
//! Usage: wgsl-validate <file.wgsl>   (exit 0 valid, 1 invalid)

use std::process::ExitCode;

fn main() -> ExitCode {
    let path = match std::env::args().nth(1) {
        Some(p) => p,
        None => {
            eprintln!("usage: wgsl-validate <file.wgsl>");
            return ExitCode::from(2);
        }
    };
    let src = match std::fs::read_to_string(&path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: cannot read {path}: {e}");
            return ExitCode::from(1);
        }
    };

    let module = match naga::front::wgsl::parse_str(&src) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("parse failed: {}", e.emit_to_string(&src));
            return ExitCode::from(1);
        }
    };

    match naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    )
    .validate(&module)
    {
        Ok(_) => {
            println!("valid: {path}");
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("validation failed: {}", e.emit_to_string(&src));
            ExitCode::from(1)
        }
    }
}
