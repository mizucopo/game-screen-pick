use std::{
    fs, io,
    path::{Component, Path, PathBuf},
    sync::atomic::{AtomicBool, Ordering},
};

use anyhow::{Context, Result, bail};
use clap::{Args, ColorChoice, Parser, Subcommand};
use serde::Serialize;

use crate::{
    config::Config,
    media::{MediaTools, discover_inputs},
};

#[derive(Parser)]
#[command(version, about = "Pick blog images from game recordings", color = ColorChoice::Never,
    subcommand_negates_reqs = true, args_conflicts_with_subcommands = true,
    after_help = "Selection is not implemented yet. Use validate for input/config checks or extract for an original full-resolution frame.")]
pub struct Cli {
    #[command(subcommand)]
    command: Option<Command>,
    #[command(flatten)]
    request: Request,
}

#[derive(Subcommand)]
enum Command {
    /// Validate arguments, configuration and paths without external commands or writes
    Validate(Request),
    /// Extract one full-resolution PNG (and optional before/after PNGs) into a new directory
    Extract(ExtractArgs),
}

#[derive(Args)]
#[group(skip)]
struct Request {
    /// New TOML configuration file
    #[arg(long, required = true)]
    config: Option<PathBuf>,
    /// Number of selected images, 1..999
    #[arg(long, required = true, value_parser = clap::value_parser!(u16).range(1..=999))]
    count: Option<u16>,
    /// Game title for context generation
    #[arg(long, value_parser = nonblank, required_unless_present = "game_context", conflicts_with = "game_context")]
    game_title: Option<String>,
    /// Direct game context; no search is needed
    #[arg(long, value_parser = nonblank, required_unless_present = "game_title")]
    game_context: Option<String>,
    /// Directory containing regular .mp4/.mkv/.mov/.webm files
    #[arg(required = true)]
    input_directory: Option<PathBuf>,
    /// Empty or new output directory, separate from inputs/config
    #[arg(required = true)]
    output_directory: Option<PathBuf>,
}

#[derive(Args)]
struct ExtractArgs {
    /// Seconds from the first valid frame PTS; chooses the first frame at or after this time
    #[arg(long, value_parser = finite_nonnegative)]
    at: f64,
    /// Also extract context at time minus/plus this positive interval, clipped to endpoints
    #[arg(long, value_parser = finite_positive)]
    context_seconds: Option<f64>,
    /// One regular input video
    input_video: PathBuf,
    /// A new directory under an existing parent; existing directories are rejected
    output_directory: PathBuf,
}

pub struct CliError {
    pub code: i32,
    pub message: String,
}

impl CliError {
    fn input(error: anyhow::Error) -> Self {
        Self {
            code: 2,
            message: safe(&error.to_string()),
        }
    }
    fn processing(error: anyhow::Error) -> Self {
        Self {
            code: 1,
            message: safe(&error.to_string()),
        }
    }
}

impl Cli {
    pub fn run(self, cancelled: &AtomicBool) -> std::result::Result<(), CliError> {
        match self.command {
            Some(Command::Extract(args)) => extract(args, cancelled),
            Some(Command::Validate(request)) => {
                let (_, inputs) = validate(request).map_err(CliError::input)?;
                eprintln!(
                    "validated: configuration, arguments and {} input video(s); no processing performed",
                    inputs.len()
                );
                Ok(())
            }
            None => {
                let (config, inputs) = validate(self.request).map_err(CliError::input)?;
                eprintln!(
                    "{}: preflight complete ({} input video(s))",
                    config.selection.method,
                    inputs.len()
                );
                Err(CliError::processing(anyhow::anyhow!(
                    "selection, AI assessment and artifact publication are not implemented yet; use extract to inspect original frames"
                )))
            }
        }
    }
}

fn nonblank(value: &str) -> std::result::Result<String, String> {
    if value.trim().is_empty() {
        Err("must be nonempty".into())
    } else {
        Ok(value.to_owned())
    }
}

fn finite_nonnegative(value: &str) -> std::result::Result<f64, String> {
    match value.parse::<f64>() {
        Ok(value) if value.is_finite() && value >= 0.0 => Ok(value),
        _ => Err("must be a finite nonnegative number of seconds".into()),
    }
}

fn finite_positive(value: &str) -> std::result::Result<f64, String> {
    let value = finite_nonnegative(value)?;
    if value > 0.0 {
        Ok(value)
    } else {
        Err("must be greater than zero".into())
    }
}

fn safe(value: &str) -> String {
    let mut text = String::new();
    for character in value.chars() {
        if character.is_control() {
            text.extend(character.escape_default());
        } else {
            text.push(character);
        }
    }
    text
}

pub fn sanitize_parse_error(error: &mut clap::Error) {
    use clap::error::ContextValue;
    // Escape dynamic contexts before Clap adds its trusted help/usage layout.
    let contexts: Vec<_> = error
        .context()
        .filter_map(|(kind, value)| {
            let escaped = match value {
                ContextValue::String(value) => ContextValue::String(safe(value)),
                ContextValue::Strings(values) => {
                    ContextValue::Strings(values.iter().map(|value| safe(value)).collect())
                }
                _ => return None,
            };
            Some((kind, escaped))
        })
        .collect();
    for (kind, value) in contexts {
        error.insert(kind, value);
    }
}

/// Reject symlinks in every existing path component, traversal, and special files.
fn checked_path(path: &Path) -> Result<PathBuf> {
    let absolute = if path.is_absolute() {
        path.to_owned()
    } else {
        std::env::current_dir()?.join(path)
    };
    let mut current = PathBuf::new();
    for component in absolute.components() {
        if matches!(component, Component::ParentDir) {
            bail!("parent traversal is not allowed in paths");
        }
        current.push(component.as_os_str());
        match fs::symlink_metadata(&current) {
            Ok(metadata) if metadata.file_type().is_symlink() => {
                bail!("symlink path components are not allowed")
            }
            Ok(metadata) if !metadata.is_dir() && !metadata.is_file() => {
                bail!("special files are not allowed")
            }
            Ok(_) => {}
            Err(error) if error.kind() == io::ErrorKind::NotFound => {}
            Err(_) => bail!("cannot inspect path"),
        }
    }
    // Normalize the existing ancestor so lexical aliases cannot bypass overlap checks.
    let mut ancestor = absolute.as_path();
    let mut suffix = Vec::new();
    while !ancestor.exists() {
        suffix.push(ancestor.file_name().context("invalid path")?.to_owned());
        ancestor = ancestor.parent().context("invalid path")?;
    }
    if !suffix.is_empty() && !ancestor.is_dir() {
        bail!("path parent must be an ordinary directory");
    }
    let mut normalized = ancestor.canonicalize().context("cannot resolve path")?;
    for part in suffix.into_iter().rev() {
        normalized.push(part);
    }
    Ok(normalized)
}

fn validate(request: Request) -> Result<(Config, Vec<PathBuf>)> {
    crate::domain::GameInput::new(request.game_title, request.game_context)?;
    if !request
        .count
        .is_some_and(|count| (1..=999).contains(&count))
    {
        bail!("--count must be from 1 to 999");
    }
    let config_path = checked_path(request.config.as_deref().context("--config is required")?)?;
    let input = checked_path(
        request
            .input_directory
            .as_deref()
            .context("input directory is required")?,
    )?;
    let output = checked_path(
        request
            .output_directory
            .as_deref()
            .context("output directory is required")?,
    )?;
    if output.starts_with(&input) || input.starts_with(&output) || config_path.starts_with(&output)
    {
        bail!("input, output and configuration paths must not overlap");
    }
    if output.exists() && (!output.is_dir() || fs::read_dir(&output)?.next().is_some()) {
        bail!(
            "output is not an empty directory; preserve it and choose a new empty output directory"
        );
    }
    let config = Config::load(&config_path)?;
    let inputs = discover_inputs(&input)?;
    // Authentication is resolved only by the future common client; it is never printed here.
    Ok((config, inputs))
}

#[derive(Serialize)]
struct ExtractionRecord {
    source: String,
    stream_index: usize,
    frames: Vec<ExtractionFrame>,
}

#[derive(Serialize)]
struct ExtractionFrame {
    file: String,
    requested_seconds: f64,
    actual_seconds: f64,
    width: u32,
    height: u32,
}

struct NewOutput {
    directory: PathBuf,
    files: Vec<PathBuf>,
    committed: bool,
}

impl Drop for NewOutput {
    fn drop(&mut self) {
        if !self.committed {
            for path in &self.files {
                let _ = fs::remove_file(path);
            }
            let _ = fs::remove_dir(&self.directory);
        }
    }
}

fn extract(args: ExtractArgs, cancelled: &AtomicBool) -> std::result::Result<(), CliError> {
    let input = checked_path(&args.input_video).map_err(CliError::input)?;
    let output = checked_path(&args.output_directory).map_err(CliError::input)?;
    if !input.is_file() || input.file_name().and_then(|name| name.to_str()).is_none() {
        return Err(CliError::input(anyhow::anyhow!(
            "extract requires an existing regular video with a UTF-8 filename"
        )));
    }
    if output.exists() || output.parent().is_none_or(|parent| !parent.is_dir()) {
        return Err(CliError::input(anyhow::anyhow!(
            "extract requires a new output directory with an existing parent"
        )));
    }
    let tools = MediaTools::default();
    let video = tools
        .probe(&input, cancelled)
        .map_err(CliError::processing)?;
    if args.at > video.duration_seconds {
        return Err(CliError::input(anyhow::anyhow!(
            "requested extraction time exceeds the video's valid duration"
        )));
    }
    let result = (|| -> Result<()> {
        let frames = match args.context_seconds {
            Some(interval) => tools
                .extract_context(&video, args.at, interval, cancelled)?
                .into_iter()
                .collect::<Vec<_>>(),
            None => vec![tools.extract(&video, args.at, cancelled)?],
        };
        if cancelled.load(Ordering::Relaxed) {
            bail!("extraction interrupted");
        }
        fs::create_dir(&output).context("cannot exclusively create extraction output directory")?;
        let mut owned = NewOutput {
            directory: output.clone(),
            files: Vec::new(),
            committed: false,
        };
        let names = if frames.len() == 3 {
            vec!["before.png", "frame.png", "after.png"]
        } else {
            vec!["frame.png"]
        };
        let mut record = ExtractionRecord {
            source: input
                .file_name()
                .and_then(|n| n.to_str())
                .context("UTF-8 video filename required")?
                .to_owned(),
            stream_index: video.stream_index as usize,
            frames: Vec::new(),
        };
        for (frame, name) in frames.iter().zip(names) {
            if cancelled.load(Ordering::Relaxed) {
                bail!("extraction interrupted");
            }
            let path = output.join(name);
            let mut file = fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&path)?;
            owned.files.push(path);
            io::copy(&mut fs::File::open(frame.path())?, &mut file)?;
            record.frames.push(ExtractionFrame {
                file: name.to_owned(),
                requested_seconds: frame.requested_seconds,
                actual_seconds: frame.actual_seconds,
                width: frame.width,
                height: frame.height,
            });
        }
        let path = output.join("extraction.json");
        let file = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)?;
        owned.files.push(path);
        serde_json::to_writer_pretty(file, &record)?;
        if cancelled.load(Ordering::Relaxed) {
            bail!("extraction interrupted");
        }
        owned.committed = true;
        eprintln!(
            "extract: complete {}/{} original frame(s); timings recorded in extraction.json",
            frames.len(),
            frames.len()
        );
        Ok(())
    })();
    result.map_err(CliError::processing)
}
