use pyo3::prelude::*;

pub mod job;
pub mod host;
pub mod event;
pub mod base_env;
pub mod bucket_adaptive_env;

use bucket_adaptive_env::BucketAdaptiveEnv;

#[pymodule]
fn lsf_env_rust(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<BucketAdaptiveEnv>()?;
    Ok(())
}
