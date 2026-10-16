use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};

use clap::Parser;
use game_screen_pick::cli::Cli;

fn main() {
    let cli = match Cli::try_parse() {
        Ok(cli) => cli,
        Err(mut error) => {
            let code = error.exit_code();
            game_screen_pick::cli::sanitize_parse_error(&mut error);
            let _ = error.print();
            std::process::exit(code);
        }
    };
    let cancelled = Arc::new(AtomicBool::new(false));
    let received = Arc::new(AtomicUsize::new(0));
    if install_interruption_handler(cancelled.clone(), received.clone()).is_err() {
        eprintln!("error: cannot install interruption handler");
        std::process::exit(1);
    }
    let code = match cli.run(&cancelled) {
        Ok(()) => 0,
        Err(error) => {
            eprintln!("error: {}", error.message);
            error.code
        }
    };
    #[cfg(unix)]
    let signal = received.load(Ordering::Relaxed);
    // Windows has one interruption kind. Derive its exit status from the same
    // flag the workers observe instead of publishing two independent atomics.
    #[cfg(windows)]
    let signal: usize = if cancelled.load(Ordering::Relaxed) {
        2
    } else {
        0
    };
    if signal != 0 {
        eprintln!("interrupted; temporary extraction files and child processes cleaned up");
        std::process::exit(128 + signal as i32);
    }
    std::process::exit(code);
}

#[cfg(unix)]
fn install_interruption_handler(
    cancelled: Arc<AtomicBool>,
    received: Arc<AtomicUsize>,
) -> std::io::Result<()> {
    for signal in [libc::SIGINT, libc::SIGTERM, libc::SIGHUP] {
        signal_hook::flag::register_usize(signal, received.clone(), signal as usize)?;
        signal_hook::flag::register(signal, cancelled.clone())?;
    }
    Ok(())
}

#[cfg(windows)]
fn install_interruption_handler(
    cancelled: Arc<AtomicBool>,
    _received: Arc<AtomicUsize>,
) -> std::io::Result<()> {
    use std::sync::OnceLock;
    use windows_sys::Win32::System::Console::{
        CTRL_BREAK_EVENT, CTRL_C_EVENT, SetConsoleCtrlHandler,
    };

    static CANCELLED: OnceLock<Arc<AtomicBool>> = OnceLock::new();

    unsafe extern "system" fn handle(event: u32) -> i32 {
        if matches!(event, CTRL_C_EVENT | CTRL_BREAK_EVENT)
            && let Some(cancelled) = CANCELLED.get()
        {
            // Only set flags here. The normal execution path cancels HTTP and
            // media work, releases locks, and stops its private process jobs.
            cancelled.store(true, Ordering::Relaxed);
            return 1;
        }
        // Close/logoff/shutdown retain Windows' default handling; they cannot
        // become a successful completion or an unbounded handler-thread wait.
        0
    }

    CANCELLED
        .set(cancelled)
        .map_err(|_| std::io::Error::other("interruption handler already initialized"))?;
    // A launcher may have inherited CTRL_C_IGNORE into this process. Restore
    // normal delivery before installing the handler so it can perform cleanup.
    if unsafe { SetConsoleCtrlHandler(None, 0) } == 0
        || unsafe { SetConsoleCtrlHandler(Some(handle), 1) } == 0
    {
        return Err(std::io::Error::last_os_error());
    }
    Ok(())
}
