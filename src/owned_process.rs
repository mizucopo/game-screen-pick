//! Ownership of one launched command and its descendants, including wrappers
//! that exit while descendants are still running or holding captured pipes.
use std::io;
use std::process::{ChildStderr, ChildStdout, Command, ExitStatus};

#[cfg(unix)]
type Child = std::process::Child;
#[cfg(windows)]
type Child = Box<dyn process_wrap::std::ChildWrapper>;

pub(crate) struct OwnedChild {
    child: Child,
}

impl OwnedChild {
    pub(crate) fn spawn(command: Command) -> io::Result<Self> {
        #[cfg(unix)]
        let child = {
            use std::os::unix::process::CommandExt;
            let mut command = command;
            command.process_group(0).spawn()?
        };
        #[cfg(windows)]
        let child = {
            use process_wrap::std::{CommandWrap, JobObject};
            // JobObject suspends the child before it can launch descendants,
            // assigns it to a private job, then resumes it. Failed assignment
            // or resume terminates and reaps the suspended child. No PID scan,
            // global endpoint stop, or child-spawn/assignment race is involved.
            CommandWrap::from(command).wrap(JobObject).spawn()?
        };
        Ok(Self { child })
    }

    pub(crate) fn id(&self) -> u32 {
        self.child.id()
    }

    pub(crate) fn stdout(&mut self) -> Option<ChildStdout> {
        #[cfg(unix)]
        return self.child.stdout.take();
        #[cfg(windows)]
        return self.child.stdout().take();
    }

    pub(crate) fn stderr(&mut self) -> Option<ChildStderr> {
        #[cfg(unix)]
        return self.child.stderr.take();
        #[cfg(windows)]
        return self.child.stderr().take();
    }

    pub(crate) fn try_wait(&mut self) -> io::Result<Option<ExitStatus>> {
        self.child.try_wait()
    }

    pub(crate) fn exit_status(&mut self) -> io::Result<Option<ExitStatus>> {
        #[cfg(unix)]
        {
            use std::os::unix::process::ExitStatusExt;
            // Reserve the leader PID until the whole owned group is stopped.
            // Reaping an exited wrapper earlier could allow PID/group reuse.
            let mut info: libc::siginfo_t = unsafe { std::mem::zeroed() };
            let result = unsafe {
                libc::waitid(
                    libc::P_PID,
                    self.child.id() as libc::id_t,
                    &mut info,
                    libc::WEXITED | libc::WNOHANG | libc::WNOWAIT,
                )
            };
            if result != 0 {
                return Err(io::Error::last_os_error());
            }
            if unsafe { info.si_pid() } == 0 {
                return Ok(None);
            }
            let status = unsafe { info.si_status() };
            let raw = match info.si_code {
                libc::CLD_EXITED => status << 8,
                libc::CLD_KILLED => status,
                libc::CLD_DUMPED => status | 0x80,
                _ => return Err(io::Error::other("invalid command exit status")),
            };
            Ok(Some(ExitStatus::from_raw(raw)))
        }
        #[cfg(windows)]
        {
            // The private job handle, rather than the reusable PID, continues
            // to own descendants even after Windows reports the leader exit.
            self.child.try_wait()
        }
    }

    pub(crate) fn has_exited(&mut self) -> io::Result<bool> {
        Ok(self.exit_status()?.is_some())
    }

    pub(crate) fn terminate(&mut self) {
        #[cfg(unix)]
        unsafe {
            libc::kill(-(self.child.id() as libc::pid_t), libc::SIGKILL);
        }
        #[cfg(windows)]
        {
            let _ = self.child.start_kill();
        }
    }
}

impl Drop for OwnedChild {
    fn drop(&mut self) {
        self.terminate();
        #[cfg(unix)]
        {
            let _ = self.child.kill();
            let _ = self.child.wait();
        }
        #[cfg(windows)]
        {
            // try_wait reaps the native leader without a blocking receive from
            // the job's completion port. Earlier status polls may have consumed
            // its final notification, including when a wrapper already exited.
            // The job remains owned until all handles are dropped below.
            while matches!(self.child.try_wait(), Ok(None)) {
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
        }
    }
}
