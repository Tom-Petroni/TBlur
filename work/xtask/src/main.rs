mod nuke;
mod util;

use core::fmt;

use anyhow::Result;
use clap::{ArgAction, Parser};

use crate::nuke::{compile_nuke, create_package, get_sources};

#[derive(clap::ValueEnum, Clone, Copy, Debug, PartialEq)]
pub enum TargetPlatform {
    Windows,
    Linux,
    MacosX86_64,
    MacosAarch64,
}

impl fmt::Display for TargetPlatform {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            TargetPlatform::Windows => write!(f, "x86_64-windows"),
            TargetPlatform::Linux => write!(f, "x86_64-linux"),
            TargetPlatform::MacosX86_64 => write!(f, "x86_64-macos"),
            TargetPlatform::MacosAarch64 => write!(f, "aarch64-macos"),
        }
    }
}

#[derive(Parser, Debug)]
#[command(
    version,
    about = "Build helper for TBlur Nuke plugin",
    arg_required_else_help = true
)]
struct Args {
    #[clap(short, long, action = ArgAction::SetTrue)]
    compile: bool,

    #[clap(long, action = ArgAction::SetTrue)]
    use_zig: bool,

    #[clap(short, long)]
    target_platform: Option<TargetPlatform>,

    #[clap(short, long, value_delimiter = ',')]
    nuke_versions: Vec<String>,

    #[clap(short, long, action = ArgAction::SetTrue)]
    fetch_nuke: bool,

    #[clap(long, action = ArgAction::SetTrue)]
    output_to_package: bool,

    #[clap(long, action = ArgAction::SetTrue)]
    limit_threads: bool,

    #[clap(long, action = ArgAction::SetTrue)]
    cuda_backend: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .compact()
        .with_line_number(false)
        .with_file(false)
        .init();

    let args = Args::parse();

    if args.fetch_nuke
        && let Some(target_platform) = args.target_platform
    {
        get_sources(
            vec![target_platform],
            args.nuke_versions.clone(),
            args.limit_threads,
        )
        .await?;
    }

    if args.compile
        && let Some(target_platform) = args.target_platform
    {
        compile_nuke(
            args.nuke_versions.clone(),
            target_platform,
            args.limit_threads,
            args.use_zig,
            args.cuda_backend,
        )
        .await?;
    }

    if args.output_to_package
        && let Some(target_platform) = args.target_platform
    {
        create_package(target_platform, args.nuke_versions).await?;
    }

    Ok(())
}

