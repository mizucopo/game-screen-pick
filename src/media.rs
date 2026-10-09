//! Input discovery and owned, full-resolution FFmpeg intermediates.
use anyhow::{Context, Result, bail, ensure};
use serde_json::Value;
use std::ffi::OsStr;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::thread;
use std::time::Duration;
use tempfile::{Builder, TempDir};

const STREAM_LIMIT: usize = 1024 * 1024;
const FRAME_LIMIT: usize = 64 * 1024 * 1024;
const STDERR_LIMIT: usize = 64 * 1024;

#[derive(Debug, Clone)]
pub struct MediaTools {
    pub ffmpeg: PathBuf,
    pub ffprobe: PathBuf,
}

impl Default for MediaTools {
    fn default() -> Self {
        Self {
            ffmpeg: "ffmpeg".into(),
            ffprobe: "ffprobe".into(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct FrameTime {
    /// Decode ordinal, including frames without a usable timestamp.
    pub decode_index: usize,
    /// Seconds relative to this stream's first valid PTS.
    pub seconds: f64,
    pub duration_seconds: f64,
}

#[derive(Debug, Clone)]
pub struct Video {
    pub source: PathBuf,
    pub stream_index: u32,
    pub start_seconds: f64,
    pub duration_seconds: f64,
    /// Display dimensions, after FFmpeg's display-matrix rotation.
    pub width: u32,
    pub height: u32,
    pub frames: Vec<FrameTime>,
}

#[derive(Debug)]
pub struct FrameArtifact {
    directory: TempDir,
    pub source: PathBuf,
    pub requested_seconds: f64,
    pub actual_seconds: f64,
    pub width: u32,
    pub height: u32,
}

impl FrameArtifact {
    pub fn path(&self) -> PathBuf {
        self.directory.path().join("frame.png")
    }
}

#[derive(Debug)]
pub struct ChunkArtifact {
    directory: TempDir,
    pub source: PathBuf,
    pub requested_start_seconds: f64,
    pub requested_end_seconds: f64,
    pub actual_start_seconds: f64,
    pub actual_end_seconds: f64,
    pub width: u32,
    pub height: u32,
}

impl ChunkArtifact {
    pub fn path(&self) -> PathBuf {
        self.directory.path().join("chunk.mov")
    }
}

#[derive(Debug)]
pub struct Interrupted;

impl std::fmt::Display for Interrupted {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("media processing interrupted")
    }
}

impl std::error::Error for Interrupted {}

/// Discover only ordinary supported files, sorted by UTF-8 filename bytes.
pub fn discover_inputs(directory: &Path) -> Result<Vec<PathBuf>> {
    let metadata = fs::symlink_metadata(directory)
        .with_context(|| format!("cannot inspect input directory {directory:?}"))?;
    ensure!(
        metadata.is_dir(),
        "input path must be an ordinary directory: {directory:?}"
    );
    let mut files = Vec::new();
    for entry in fs::read_dir(directory).context("cannot read input directory")? {
        let entry = entry.context("cannot read input directory entry")?;
        let name = entry.file_name();
        let name = name
            .to_str()
            .context("input directory contains a non-UTF-8 filename")?;
        let extension = Path::new(name).extension().and_then(OsStr::to_str);
        if !extension.is_some_and(|extension| {
            ["mp4", "mkv", "mov", "webm"]
                .iter()
                .any(|supported| extension.eq_ignore_ascii_case(supported))
        }) {
            continue;
        }
        ensure!(
            entry.file_type()?.is_file(),
            "supported video must be an ordinary file, without symlinks: {:?}",
            entry.path()
        );
        files.push(entry.path());
    }
    files.sort_by(|left, right| left.file_name().cmp(&right.file_name()));
    ensure!(
        !files.is_empty(),
        "input directory contains no supported ordinary videos (.mp4/.mkv/.mov/.webm)"
    );
    Ok(files)
}

impl MediaTools {
    /// Probe one stream and its valid frame PTS. Cold probing decodes the stream;
    /// #341 must reuse this metadata rather than repeating it on warm runs.
    pub fn probe(&self, source: &Path, cancelled: &AtomicBool) -> Result<Video> {
        check_cancelled(cancelled)?;
        ensure!(
            fs::symlink_metadata(source)?.is_file(),
            "video must be an ordinary file, without symlinks: {source:?}"
        );
        ensure!(
            source.file_name().and_then(OsStr::to_str).is_some(),
            "video filename must be UTF-8"
        );
        let source = source
            .canonicalize()
            .with_context(|| format!("cannot resolve video {source:?}"))?;
        let mut command = Command::new(&self.ffprobe);
        command.args(["-v", "error", "-show_streams", "-show_entries", "stream=index,codec_type,width,height,start_time,duration:stream_disposition=attached_pic:stream_side_data=rotation", "-of", "json"]);
        command.arg(&source);
        let bytes = run(command, cancelled, STREAM_LIMIT, "FFprobe stream metadata")?;
        let metadata: Value =
            serde_json::from_slice(&bytes).context("FFprobe returned invalid stream JSON")?;
        let streams = metadata["streams"]
            .as_array()
            .context("FFprobe metadata has no streams array")?;
        let stream = streams
            .iter()
            .filter(|stream| {
                stream["codec_type"].as_str() == Some("video")
                    && stream["disposition"]["attached_pic"].as_u64() == Some(0)
            })
            .min_by_key(|stream| stream["index"].as_u64().unwrap_or(u64::MAX))
            .context("video has no ordinary video stream (attached pictures are excluded)")?;
        let stream_index = u32::try_from(
            stream["index"]
                .as_u64()
                .context("invalid video stream index")?,
        )
        .context("video stream index is too large")?;
        let mut width = dimension(&stream["width"], "width")?;
        let mut height = dimension(&stream["height"], "height")?;
        if let Some(side_data) = stream["side_data_list"].as_array() {
            for data in side_data {
                if let Some(rotation) = optional_number(&data["rotation"], "display rotation")? {
                    let quarter_turns = rotation.rem_euclid(360.0) / 90.0;
                    ensure!(
                        (quarter_turns - quarter_turns.round()).abs() < 0.0001,
                        "unsupported display rotation; expected a multiple of 90 degrees"
                    );
                    if (quarter_turns.round() as i64).rem_euclid(2) == 1 {
                        std::mem::swap(&mut width, &mut height);
                    }
                    break;
                }
            }
        }
        // Reject malformed declared values, but timestamps are based on decoded PTS.
        let declared_start = optional_number(&stream["start_time"], "stream start time")?;
        let declared_duration = optional_number(&stream["duration"], "stream duration")?;
        if let Some(duration) = declared_duration {
            ensure!(duration > 0.0, "stream duration must be positive");
        }
        let mut command = Command::new(&self.ffprobe);
        command.args(["-v", "error", "-select_streams"]);
        command.arg(stream_index.to_string());
        command.args([
            "-show_frames",
            "-show_entries",
            "frame=pts_time,duration_time,pkt_duration_time",
            "-of",
            "json",
        ]);
        command.arg(&source);
        let bytes = run(command, cancelled, FRAME_LIMIT, "FFprobe frame metadata")?;
        let metadata: Value =
            serde_json::from_slice(&bytes).context("FFprobe returned invalid frame JSON")?;
        let raw_frames = metadata["frames"]
            .as_array()
            .context("FFprobe metadata has no frames array")?;
        let mut frames = Vec::new();
        let mut start = None;
        for (decode_index, frame) in raw_frames.iter().enumerate() {
            let pts = optional_number(&frame["pts_time"], "frame PTS")?;
            let Some(pts) = pts else { continue };
            let start_seconds = *start.get_or_insert(pts);
            let seconds = probe_seconds(pts - start_seconds);
            ensure!(
                seconds.is_finite() && seconds >= 0.0,
                "invalid frame time relative to first PTS"
            );
            ensure!(
                frames
                    .last()
                    .is_none_or(|previous: &FrameTime| previous.seconds < seconds),
                "video frame PTS must be strictly increasing"
            );
            let duration_seconds = optional_number(&frame["duration_time"], "frame duration")?
                .or(optional_number(
                    &frame["pkt_duration_time"],
                    "packet duration",
                )?)
                .unwrap_or(0.0);
            ensure!(
                duration_seconds >= 0.0,
                "frame duration must not be negative"
            );
            frames.push(FrameTime {
                decode_index,
                seconds,
                duration_seconds,
            });
        }
        let start_seconds = start.context("video has no frames with a valid PTS")?;
        let last = frames.last().context("video has no frames")?;
        let last_seconds = last.seconds;
        let duration_seconds = if last.duration_seconds > 0.0 {
            probe_seconds(last.seconds + last.duration_seconds)
        } else if let Some(duration) = declared_duration {
            probe_seconds(declared_start.unwrap_or(start_seconds) + duration - start_seconds)
        } else {
            bail!("video has no usable final frame duration or stream duration");
        };
        ensure!(
            duration_seconds.is_finite() && duration_seconds > last.seconds,
            "video duration does not include its final valid frame"
        );
        for index in 0..frames.len().saturating_sub(1) {
            frames[index].duration_seconds =
                probe_seconds(frames[index + 1].seconds - frames[index].seconds);
        }
        frames
            .last_mut()
            .expect("validated nonempty frames")
            .duration_seconds = probe_seconds(duration_seconds - last_seconds);
        Ok(Video {
            source,
            stream_index,
            start_seconds,
            duration_seconds,
            width,
            height,
            frames,
        })
    }

    /// Select the first valid PTS at or after the request. The duration endpoint
    /// selects the final frame. PNG retains the decoded full-resolution pixels.
    pub fn extract(
        &self,
        video: &Video,
        requested_seconds: f64,
        cancelled: &AtomicBool,
    ) -> Result<FrameArtifact> {
        let frame = frame_at(video, requested_seconds)?;
        check_cancelled(cancelled)?;
        let directory = Builder::new()
            .prefix("game-screen-pick-frame-")
            .tempdir()
            .context("cannot create private frame workspace")?;
        let path = directory.path().join("frame.png");
        let mut command = self.decoder(video);
        command.args([
            "-vf",
            &format!("select=eq(n\\,{})", frame.decode_index),
            "-frames:v",
            "1",
            "-fps_mode",
            "passthrough",
            "-c:v",
            "png",
            "-pix_fmt",
            "rgb24",
            "-update",
            "1",
        ]);
        command.arg(&path);
        run(command, cancelled, STREAM_LIMIT, "FFmpeg frame extraction")?;
        ensure!(
            fs::metadata(&path).is_ok_and(|metadata| metadata.len() > 0),
            "FFmpeg did not produce a frame"
        );
        let mut header = [0; 24];
        fs::File::open(&path)?
            .read_exact(&mut header)
            .context("FFmpeg produced a truncated PNG frame")?;
        ensure!(
            header[..8] == [137, 80, 78, 71, 13, 10, 26, 10] && header[12..16] == *b"IHDR",
            "FFmpeg did not produce a PNG frame"
        );
        let width = u32::from_be_bytes(header[16..20].try_into()?);
        let height = u32::from_be_bytes(header[20..24].try_into()?);
        ensure!(
            (width, height) == (video.width, video.height),
            "extracted frame dimensions differ from the probed display dimensions; changing-resolution media is unsupported"
        );
        Ok(FrameArtifact {
            directory,
            source: video.source.clone(),
            requested_seconds,
            actual_seconds: frame.seconds,
            width: video.width,
            height: video.height,
        })
    }

    /// Before / center / after frames, with boundary requests clipped to the video.
    pub fn extract_context(
        &self,
        video: &Video,
        seconds: f64,
        delta_seconds: f64,
        cancelled: &AtomicBool,
    ) -> Result<[FrameArtifact; 3]> {
        frame_at(video, seconds)?;
        ensure!(
            delta_seconds.is_finite() && delta_seconds > 0.0,
            "context interval must be finite and positive"
        );
        Ok([
            self.extract(video, (seconds - delta_seconds).max(0.0), cancelled)?,
            self.extract(video, seconds, cancelled)?,
            self.extract(
                video,
                (seconds + delta_seconds).min(video.duration_seconds),
                cancelled,
            )?,
        ])
    }

    /// Lossless video intermediate for later semantic requests. Frames retain
    /// their VFR spacing and full final display interval, start at zero, and use
    /// the same selected source stream. PNG-in-MOV stores per-frame duration;
    /// setts restores the final hold which encoders otherwise replace with a
    /// nominal frame interval. The microsecond timebase matches probed PTS precision.
    pub fn extract_chunk(
        &self,
        video: &Video,
        start_seconds: f64,
        end_seconds: f64,
        cancelled: &AtomicBool,
    ) -> Result<ChunkArtifact> {
        let first = frame_at(video, start_seconds)?;
        frame_at(video, end_seconds)?;
        ensure!(start_seconds < end_seconds, "chunk start must precede end");
        let last = video
            .frames
            .iter()
            .rev()
            .find(|frame| frame.seconds < end_seconds && frame.seconds >= first.seconds)
            .context("chunk interval contains no valid frame PTS")?;
        ensure!(
            last.decode_index - first.decode_index + 1
                == video
                    .frames
                    .iter()
                    .filter(|frame| frame.seconds >= first.seconds && frame.seconds <= last.seconds)
                    .count(),
            "semantic chunk contains frames without valid PTS; this media is unsupported"
        );
        check_cancelled(cancelled)?;
        let directory = Builder::new()
            .prefix("game-screen-pick-chunk-")
            .tempdir()
            .context("cannot create private chunk workspace")?;
        let path = directory.path().join("chunk.mov");
        let mut command = self.decoder(video);
        command.args([
            "-vf",
            &format!(
                "select=between(n\\,{}\\,{}),setpts=PTS-STARTPTS",
                first.decode_index, last.decode_index
            ),
            "-fps_mode",
            "passthrough",
            "-c:v",
            "png",
            "-pix_fmt",
            "rgb24",
            "-enc_time_base",
            "1:1000000",
            "-video_track_timescale",
            "1000000",
            "-bsf:v",
            &format!(
                "setts=duration=if(eq(N\\,{})\\,round({}/TB)\\,NEXT_PTS-PTS)",
                last.decode_index - first.decode_index,
                last.duration_seconds
            ),
        ]);
        command.arg(&path);
        run(
            command,
            cancelled,
            STREAM_LIMIT,
            "FFmpeg semantic chunk extraction",
        )?;
        ensure!(
            fs::metadata(&path).is_ok_and(|metadata| metadata.len() > 0),
            "FFmpeg did not produce a chunk"
        );
        Ok(ChunkArtifact {
            directory,
            source: video.source.clone(),
            requested_start_seconds: start_seconds,
            requested_end_seconds: end_seconds,
            actual_start_seconds: first.seconds,
            actual_end_seconds: probe_seconds(last.seconds + last.duration_seconds),
            width: video.width,
            height: video.height,
        })
    }

    fn decoder(&self, video: &Video) -> Command {
        let mut command = Command::new(&self.ffmpeg);
        command.args(["-hide_banner", "-loglevel", "error", "-nostdin", "-n", "-i"]);
        command.arg(&video.source);
        command.args([
            "-map",
            &format!("0:{}", video.stream_index),
            "-an",
            "-sn",
            "-dn",
        ]);
        command
    }
}

fn frame_at(video: &Video, seconds: f64) -> Result<&FrameTime> {
    ensure!(
        seconds.is_finite() && seconds >= 0.0 && seconds <= video.duration_seconds,
        "requested frame time must be within 0..={} seconds",
        video.duration_seconds
    );
    video
        .frames
        .iter()
        .find(|frame| frame.seconds >= seconds)
        .or_else(|| video.frames.last())
        .context("video has no valid frame times")
}

fn dimension(value: &Value, name: &str) -> Result<u32> {
    let dimension = value
        .as_u64()
        .with_context(|| format!("invalid video {name}"))?;
    ensure!(
        dimension > 0 && dimension <= 65_535,
        "video {name} is outside 1..=65535"
    );
    Ok(dimension as u32)
}

// FFprobe's *_time fields use microsecond precision. Keep subtraction and
// duration arithmetic on that same grid so a literal PTS remains an endpoint,
// rather than falling just after it through binary floating-point roundoff.
fn probe_seconds(seconds: f64) -> f64 {
    (seconds * 1_000_000.0).round() / 1_000_000.0
}

fn optional_number(value: &Value, name: &str) -> Result<Option<f64>> {
    if value.is_null() || value.as_str() == Some("N/A") {
        return Ok(None);
    }
    let number = if let Some(text) = value.as_str() {
        text.parse::<f64>()
            .with_context(|| format!("invalid {name} in FFprobe metadata"))?
    } else {
        value
            .as_f64()
            .with_context(|| format!("invalid {name} type in FFprobe metadata"))?
    };
    ensure!(number.is_finite(), "non-finite {name} in FFprobe metadata");
    Ok(Some(number))
}

fn check_cancelled(cancelled: &AtomicBool) -> Result<()> {
    if cancelled.load(Ordering::Relaxed) {
        return Err(Interrupted.into());
    }
    Ok(())
}

struct OwnedChild(Child);

impl OwnedChild {
    fn kill_group(&self) {
        #[cfg(unix)]
        unsafe {
            libc::kill(-(self.0.id() as libc::pid_t), libc::SIGKILL);
        }
    }
}

impl Drop for OwnedChild {
    fn drop(&mut self) {
        // Each command has its own process group, so interruption also closes
        // inherited pipes held by descendants. We never signal the caller's group.
        self.kill_group();
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn bounded_reader(mut reader: impl Read, limit: usize) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    let mut buffer = [0; 8192];
    loop {
        let count = reader
            .read(&mut buffer)
            .context("cannot read media command output")?;
        if count == 0 {
            return Ok(bytes);
        }
        ensure!(
            bytes.len() + count <= limit,
            "media command output exceeded {limit} byte limit"
        );
        bytes.extend_from_slice(&buffer[..count]);
    }
}

fn run(
    mut command: Command,
    cancelled: &AtomicBool,
    limit: usize,
    operation: &str,
) -> Result<Vec<u8>> {
    check_cancelled(cancelled)?;
    command
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        command.process_group(0);
    }
    let program = command.get_program().to_os_string();
    let mut child = OwnedChild(command.spawn().with_context(|| format!("cannot start {operation} command {program:?}; install FFmpeg/FFprobe or correct its command path"))?);
    let stdout = child.0.stdout.take().context("missing command stdout")?;
    let stderr = child.0.stderr.take().context("missing command stderr")?;
    let (sender, receiver) = mpsc::channel();
    let error_sender = sender.clone();
    let stdout_thread = thread::spawn(move || {
        let result = bounded_reader(stdout, limit);
        let _ = sender.send((true, result));
    });
    let stderr_thread = thread::spawn(move || {
        let result = bounded_reader(stderr, STDERR_LIMIT);
        let _ = error_sender.send((false, result));
    });
    let result = (|| {
        let mut stdout = None;
        let mut stderr = None;
        let mut status = None;
        loop {
            check_cancelled(cancelled)?;
            while let Ok((is_stdout, bytes)) = receiver.try_recv() {
                let bytes = bytes.with_context(|| operation.to_owned())?;
                if is_stdout {
                    stdout = Some(bytes);
                } else {
                    stderr = Some(bytes);
                }
            }
            if status.is_none() {
                status = child
                    .0
                    .try_wait()
                    .context("cannot wait for media command")?;
                if status.is_some() {
                    // A descendant may still hold a pipe after its leader exits.
                    // Stop the entire owned group before waiting for EOF.
                    child.kill_group();
                }
            }
            if let Some(status) = status
                .as_ref()
                .filter(|_| stdout.is_some() && stderr.is_some())
            {
                if !status.success() {
                    bail!(
                        "{operation} failed ({status}); inspect the video and FFmpeg installation (child diagnostics are withheld to protect secrets)"
                    );
                }
                return stdout.context("media command stdout unavailable");
            }
            thread::sleep(Duration::from_millis(10));
        }
    })();
    drop(child);
    let _ = stdout_thread.join();
    let _ = stderr_thread.join();
    result
}
