//! Non-secret measurements; timings describe work performed, not model quality.
use std::cell::Cell;
use std::time::Instant;

use serde::Serialize;

/// Cumulative wall-clock seconds for synchronous operations since preparation.
/// They omit parts of orchestration, rendering, selection and publication,
/// so they are not an exhaustive partition of the session elapsed time. Caller
/// callbacks can themselves probe/validate inputs; their intervals may overlap.
#[derive(Clone, Debug, Default, Serialize)]
pub struct StageSeconds {
    /// FFmpeg frame/context/chunk extraction, including selected-frame decoding.
    pub extraction: f64,
    /// Local candidate image decoding and mechanical quality analysis only.
    pub machine_quality: f64,
    /// Caller assessment, semantic and context callbacks, including their retry,
    /// transport, search or runtime preparation. This is not model-only latency.
    pub inference_callbacks: f64,
    /// Input identity lookup/hash, checkpoint/domain validation, assessment-key
    /// image hashing, Session input consistency checks and output ownership
    /// inspections (including staged/published output and recovery).
    pub cache_verification: f64,
    /// Actual FFprobe invocations only, excluding cache lookup and validation.
    pub media_probe: f64,
}

/// Current-process lifetime peak resident set size as reported by the OS.
/// This includes work before Session::prepare and other threads in this process;
/// it excludes child processes (FFmpeg/FFprobe) and remote inference servers.
#[derive(Clone, Debug, Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum PeakMemory {
    Measured {
        bytes: u64,
        scope: &'static str,
    },
    Unavailable {
        reason: &'static str,
        scope: &'static str,
    },
    Unsupported {
        scope: &'static str,
    },
}

const MEMORY_SCOPE: &str =
    "current_process_lifetime_peak_rss_excluding_children_and_remote_servers";

/// A failed measurement is explicit and never turns an otherwise valid run into
/// a failure. Linux reports KiB while macOS reports bytes.
pub(crate) fn peak_memory() -> PeakMemory {
    #[cfg(any(target_os = "macos", target_os = "linux"))]
    {
        let mut usage = std::mem::MaybeUninit::<libc::rusage>::zeroed();
        // SAFETY: getrusage writes the complete structure on success, and the
        // initialized value is read only after that success has been checked.
        if unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) } != 0 {
            return PeakMemory::Unavailable {
                reason: "getrusage_failed",
                scope: MEMORY_SCOPE,
            };
        }
        // SAFETY: the successful getrusage call initialized this structure.
        let usage = unsafe { usage.assume_init() };
        let bytes = rss_bytes(i128::from(usage.ru_maxrss), cfg!(target_os = "linux"));
        match bytes {
            Some(bytes) => PeakMemory::Measured {
                bytes,
                scope: MEMORY_SCOPE,
            },
            None => PeakMemory::Unavailable {
                reason: "invalid_or_overflowing_peak_rss",
                scope: MEMORY_SCOPE,
            },
        }
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        PeakMemory::Unsupported {
            scope: MEMORY_SCOPE,
        }
    }
}

#[cfg(any(test, target_os = "macos", target_os = "linux"))]
fn rss_bytes(value: i128, kibibytes: bool) -> Option<u64> {
    let value = u64::try_from(value).ok().filter(|value| *value > 0)?;
    value.checked_mul(if kibibytes { 1024 } else { 1 })
}

/// Measure successes and returned errors equally. A callback panic is outside
/// the normal workflow result and does not produce completed run metrics.
pub(crate) fn timed<T>(seconds: &mut f64, operation: impl FnOnce() -> T) -> T {
    let started = Instant::now();
    let result = operation();
    *seconds += started.elapsed().as_secs_f64();
    result
}

pub(crate) fn timed_cell<T>(seconds: &Cell<f64>, operation: impl FnOnce() -> T) -> T {
    let started = Instant::now();
    let result = operation();
    seconds.set(seconds.get() + started.elapsed().as_secs_f64());
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn peak_rss_units_and_invalid_values_are_explicit() {
        assert_eq!(rss_bytes(2048, false), Some(2048));
        assert_eq!(rss_bytes(2048, true), Some(2 * 1024 * 1024));
        assert_eq!(rss_bytes(0, true), None);
        assert_eq!(rss_bytes(-1, false), None);
        assert_eq!(rss_bytes(i128::from(i64::MAX), true), None);
    }

    #[test]
    fn peak_memory_serializes_its_measurement_scope_and_status() {
        let value = serde_json::to_value(peak_memory()).unwrap();
        assert_eq!(value["scope"], MEMORY_SCOPE);
        #[cfg(any(target_os = "macos", target_os = "linux"))]
        assert!(value["status"] == "measured" || value["status"] == "unavailable");
        #[cfg(not(any(target_os = "macos", target_os = "linux")))]
        assert_eq!(value["status"], "unsupported");
        if value["status"] == "measured" {
            assert!(value["bytes"].as_u64().unwrap() > 0);
        }
        let unavailable = serde_json::to_value(PeakMemory::Unavailable {
            reason: "getrusage_failed",
            scope: MEMORY_SCOPE,
        })
        .unwrap();
        assert_eq!(unavailable["status"], "unavailable");
        assert!(unavailable.get("bytes").is_none());
    }
}
