use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};

use clap::Parser;
use game_screen_pick::cli::Cli;

fn main() {
    let cli = match Cli::try_parse() {
        Ok(cli) => cli,
        Err(error) => {
            let code = error.exit_code();
            let _ = error.print();
            std::process::exit(code);
        }
    };
    let cancelled = Arc::new(AtomicBool::new(false));
    let received = Arc::new(AtomicUsize::new(0));
    for signal in [libc::SIGINT, libc::SIGTERM, libc::SIGHUP] {
        if signal_hook::flag::register_usize(signal, received.clone(), signal as usize).is_err()
            || signal_hook::flag::register(signal, cancelled.clone()).is_err()
        {
            eprintln!("error: cannot install interruption handler");
            std::process::exit(1);
        }
    }
    let code = match cli.run(&cancelled) {
        Ok(()) => 0,
        Err(error) => {
            eprintln!("error: {}", error.message);
            error.code
        }
    };
    let signal = received.load(Ordering::Relaxed);
    if signal != 0 {
        eprintln!("interrupted; temporary extraction files and child processes cleaned up");
        std::process::exit(128 + signal as i32);
    }
    std::process::exit(code);
}
