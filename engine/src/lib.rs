#[cfg(feature = "window")]
pub mod desktop;
pub mod fast;
pub mod games;
pub mod host;
pub mod lichao;
pub mod model;

#[cfg(target_arch = "wasm32")]
pub mod wasm;
