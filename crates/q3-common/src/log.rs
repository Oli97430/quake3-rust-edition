//! Initialisation du logger (tracing-subscriber) avec filtrage par env var.

use tracing_subscriber::{fmt, prelude::*, EnvFilter};

/// Directive de filtrage par défaut.
///
/// * `info` pour nos crates et le reste du monde
/// * `warn` pour `wgpu_core` / `wgpu_hal` / `naga` : en INFO, `wgpu_core`
///   log une ligne `Device::maintain: waiting for submission index N` par
///   frame, ce qui noie totalement nos propres traces. On garde quand même
///   les warnings (swapchain out-of-date, device lost, etc.).
/// * `warn` pour `winit` : génère du bruit sur certaines plateformes.
const DEFAULT_FILTER: &str =
    "info,wgpu_core=warn,wgpu_hal=warn,wgpu=warn,naga=warn,winit=warn";

/// Initialise le logger global. Idempotent (à appeler une seule fois au
/// démarrage).
///
/// Par défaut : niveau `info` (sauf wgpu/naga/winit, voir [`DEFAULT_FILTER`]).
/// Override avec `RUST_LOG=debug` ou `RUST_LOG=q3_renderer=trace,q3_bsp=debug`.
pub fn init() {
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new(DEFAULT_FILTER));

    let _ = tracing_subscriber::registry()
        .with(filter)
        .with(
            fmt::layer()
                .with_target(true)
                .with_thread_names(false)
                .with_level(true)
                .compact(),
        )
        .try_init();
}
