//! Synchronous callers with cancellable async I/O and an absolute deadline.
//! Each response owns its reactor: dropping a request/response also stops its
//! connection tasks. No detached blocking HTTP request continues after return.
use std::future::{Future, poll_fn};
use std::sync::atomic::{AtomicBool, Ordering};
use std::task::Poll;
use std::time::{Duration, Instant};

use anyhow::{Result, anyhow, ensure};
use reqwest::header::HeaderMap;
use reqwest::{RequestBuilder, Response, StatusCode};
use tokio::runtime::{Builder, Runtime};
use url::Url;

use crate::media::Interrupted;

const CANCELLATION_POLL: Duration = Duration::from_millis(20);

pub(crate) struct HttpClient {
    client: reqwest::Client,
}

/// Headers are available before body consumption, so callers can reject a
/// redirect or exhausted quota without reading an untrusted/slow body.
pub(crate) struct HttpResponse {
    response: Response,
    runtime: Reactor,
}

struct Reactor(Option<Runtime>);

impl Reactor {
    fn block_on<F: Future>(&self, future: F) -> F::Output {
        self.0
            .as_ref()
            .expect("owned HTTP reactor")
            .block_on(future)
    }
}

impl Drop for Reactor {
    fn drop(&mut self) {
        // Abort every async HTTP connection task without waiting for Tokio's
        // blocking system-DNS lookup to finish. An already-running OS resolver
        // may finish in the background; no detached HTTP request, credentials
        // or payload are sent by it. Waiting for it would block SIGINT cleanup.
        if let Some(runtime) = self.0.take() {
            runtime.shutdown_timeout(Duration::ZERO);
        }
    }
}

impl HttpResponse {
    pub fn status(&self) -> StatusCode {
        self.response.status()
    }

    pub fn headers(&self) -> &HeaderMap {
        self.response.headers()
    }

    pub fn content_length(&self) -> Option<u64> {
        self.response.content_length()
    }
}

impl HttpClient {
    pub fn new() -> Result<Self> {
        let client = reqwest::Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .no_proxy()
            // A response's runtime is private and short-lived. Connections
            // must not remain pooled for a different response's reactor.
            .pool_max_idle_per_host(0)
            .build()
            .map_err(|_| anyhow!("cannot initialize HTTP client"))?;
        Ok(Self { client })
    }

    pub fn get(&self, url: Url) -> RequestBuilder {
        self.client.get(url)
    }

    pub fn post(&self, url: Url) -> RequestBuilder {
        self.client.post(url)
    }

    pub fn send(
        &self,
        request: RequestBuilder,
        deadline: Instant,
        cancelled: &AtomicBool,
        operation: &'static str,
    ) -> Result<HttpResponse> {
        check_cancelled(cancelled)?;
        let runtime = Reactor(Some(
            Builder::new_current_thread()
                .enable_all()
                .build()
                .map_err(|_| anyhow!("cannot initialize HTTP runtime"))?,
        ));
        let response = runtime.block_on(async {
            // reqwest builds its timeout timer when send() is invoked, so the
            // invocation itself must occur inside the owned reactor context.
            cancellable(request.send(), deadline, cancelled, operation).await
        })?;
        check_cancelled(cancelled)?;
        Ok(HttpResponse { response, runtime })
    }

    pub fn read_body(
        &self,
        response: HttpResponse,
        limit: u64,
        deadline: Instant,
        cancelled: &AtomicBool,
        operation: &'static str,
    ) -> Result<Vec<u8>> {
        let HttpResponse {
            mut response,
            runtime,
        } = response;
        ensure!(
            response
                .content_length()
                .is_none_or(|length| length <= limit),
            "{operation} response exceeds byte limit"
        );
        let mut bytes = Vec::new();
        runtime.block_on(async {
            while let Some(chunk) =
                cancellable(response.chunk(), deadline, cancelled, operation).await?
            {
                ensure!(
                    chunk.len() as u64 <= limit.saturating_sub(bytes.len() as u64),
                    "{operation} response exceeds byte limit"
                );
                bytes.extend_from_slice(&chunk);
            }
            check_cancelled(cancelled)?;
            Ok(bytes)
        })
    }
}

pub(crate) fn check_cancelled(cancelled: &AtomicBool) -> Result<()> {
    if cancelled.load(Ordering::Relaxed) {
        return Err(Interrupted.into());
    }
    Ok(())
}

#[derive(Debug)]
struct ConnectionRefused(&'static str);

impl std::fmt::Display for ConnectionRefused {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{} HTTP connection refused", self.0)
    }
}

impl std::error::Error for ConnectionRefused {}

pub(crate) fn connection_refused(error: &anyhow::Error) -> bool {
    error.downcast_ref::<ConnectionRefused>().is_some()
}

fn request_error(error: reqwest::Error, operation: &'static str) -> anyhow::Error {
    if error.is_timeout() {
        return anyhow!("{operation} HTTP deadline exceeded");
    }
    let mut cause: Option<&(dyn std::error::Error + 'static)> = Some(&error);
    while let Some(error) = cause {
        if error
            .downcast_ref::<std::io::Error>()
            .is_some_and(|error| error.kind() == std::io::ErrorKind::ConnectionRefused)
        {
            return ConnectionRefused(operation).into();
        }
        cause = error.source();
    }
    // Do not retain reqwest's error chain: it can include authenticated URLs,
    // headers or provider diagnostics when printed with anyhow's alternate fmt.
    anyhow!("{operation} HTTP request or response body failed")
}

async fn cancellable<T>(
    future: impl Future<Output = std::result::Result<T, reqwest::Error>>,
    deadline: Instant,
    cancelled: &AtomicBool,
    operation: &'static str,
) -> Result<T> {
    let mut future = std::pin::pin!(future);
    let mut tick = std::pin::pin!(tokio::time::sleep(Duration::ZERO));
    // Select the owned I/O future against cancellation and the same absolute
    // deadline. poll_fn avoids an extra Tokio macros dependency.
    poll_fn(|context| {
        if let Err(error) = check_cancelled(cancelled) {
            return Poll::Ready(Err(error));
        }
        if Instant::now() >= deadline {
            return Poll::Ready(Err(anyhow!("{operation} HTTP deadline exceeded")));
        }
        if let Poll::Ready(result) = future.as_mut().poll(context) {
            return Poll::Ready(result.map_err(|error| request_error(error, operation)));
        }
        if tick.as_mut().poll(context).is_ready() {
            tick.as_mut().reset(tokio::time::Instant::from_std(
                deadline.min(Instant::now() + CANCELLATION_POLL),
            ));
            // Register this task's waker for the next cancellation/deadline poll.
            let _ = tick.as_mut().poll(context);
        }
        Poll::Pending
    })
    .await
}
