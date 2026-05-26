use std::ops::{Deref, DerefMut};

use pyo3::prelude::*;
use pyo3::types::PyDict;
use numpy::{PyArray1, PyReadonlyArray1};

use crate::base_env::BaseClusterEnv;

#[pyclass]
pub struct BucketAdaptiveEnv {
    base: BaseClusterEnv,
    max_buckets: usize,
    cached_state: Vec<f32>,
    macro_levels: Vec<f32>,

    last_macro_jobs_scheduled: usize,
    last_macro_level: usize,
    total_macro_actions: usize,
    avg_jobs_per_macro: f32,

    macro_level_counts: Vec<usize>,

    reward_interval: u64,
    accumulated_util_sum: f64,
    accumulated_avg_waiting_sum: f64,
    accumulated_max_waiting_sum: f64,
    accumulated_sample_count: u64,
    last_reward_time: u64,

    wait_penalty_coef: f32,
}

impl Deref for BucketAdaptiveEnv {
    type Target = BaseClusterEnv;

    fn deref(&self) -> &Self::Target {
        &self.base
    }
}

impl DerefMut for BucketAdaptiveEnv {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.base
    }
}

#[pymethods]
impl BucketAdaptiveEnv {
    #[new]
    #[pyo3(signature = (
        num_hosts = 1000,
        max_queue_length = None,
        host_cores_range = (32, 128),
        host_memory_range = (131072, 524288),
        job_cores_range = (1, 32),
        job_memory_range = (2048, 65536),
        job_duration_range = (1, 60),
        max_jobs_per_step = 50,
        max_time = 4096,
        use_skewed_arrivals = false,
        seed = None,
        max_buckets = 100,
        reward_interval = 1,
        macro_levels = None,
        wait_penalty_coef = 0.1
    ))]
    pub fn new(
        num_hosts: usize,
        max_queue_length: Option<usize>,
        host_cores_range: (u32, u32),
        host_memory_range: (u32, u32),
        job_cores_range: (u32, u32),
        job_memory_range: (u32, u32),
        job_duration_range: (u32, u32),
        max_jobs_per_step: usize,
        max_time: usize,
        use_skewed_arrivals: bool,
        seed: Option<u64>,
        max_buckets: usize,
        reward_interval: u64,
        macro_levels: Option<Vec<f32>>,
        wait_penalty_coef: f32,
    ) -> Self {
        let config = crate::base_env::ClusterConfig::new(
            num_hosts,
            max_queue_length,
            host_cores_range,
            host_memory_range,
            job_cores_range,
            job_memory_range,
            job_duration_range,
            max_jobs_per_step,
            max_time,
            use_skewed_arrivals,
            seed,
        );

        let base = BaseClusterEnv::from_config(&config);
        let state_size = max_buckets * 4 + 2 + 1;
        let cached_state = vec![0.0; state_size];

        let macro_levels = macro_levels.unwrap_or_else(|| vec![0.2, 0.5, 1.0]);
        let num_levels = macro_levels.len();

        BucketAdaptiveEnv {
            base,
            max_buckets,
            cached_state,
            macro_levels,
            last_macro_jobs_scheduled: 0,
            last_macro_level: 0,
            total_macro_actions: 0,
            avg_jobs_per_macro: 0.0,
            macro_level_counts: vec![0; num_levels],
            reward_interval: reward_interval.max(1),
            accumulated_util_sum: 0.0,
            accumulated_avg_waiting_sum: 0.0,
            accumulated_max_waiting_sum: 0.0,
            accumulated_sample_count: 0,
            last_reward_time: 0,
            wait_penalty_coef,
        }
    }

    pub fn reset(&mut self, py: Python) -> PyResult<Py<PyArray1<f32>>> {
        self.reset_base();
        self.add_new_jobs_to_arrival_buffer();

        self.last_macro_jobs_scheduled = 0;
        self.last_macro_level = 0;
        self.total_macro_actions = 0;
        self.avg_jobs_per_macro = 0.0;
        for count in &mut self.macro_level_counts {
            *count = 0;
        }
        self.accumulated_util_sum = 0.0;
        self.accumulated_avg_waiting_sum = 0.0;
        self.accumulated_max_waiting_sum = 0.0;
        self.accumulated_sample_count = 0;
        self.last_reward_time = 0;

        self.get_state(py)
    }

    pub fn step(&mut self, py: Python, action: &PyAny) -> PyResult<(Py<PyArray1<f32>>, f32, bool, PyObject)> {
        if self.try_start_batch() {
            for i in 0..self.job_buckets.len() {
                self.set_bucket_priorities(i, None);
            }
        }

        let (bucket_idx, macro_level) = self.parse_action(action)?;

        let can_select = bucket_idx < self.job_buckets.len()
            && !self.job_buckets[bucket_idx].jobs.is_empty();

        let mut jobs_scheduled_this_macro = 0;

        if can_select {
            let host_priorities = self.job_buckets[bucket_idx].host_priorities.clone();
            let bucket_key = self.job_buckets[bucket_idx].bucket_key.clone();

            if let Some(priorities) = host_priorities {
                let bucket_size = self.job_buckets[bucket_idx].jobs.len();
                let target_jobs = self.calculate_target_jobs(bucket_size, macro_level);

                let mut current_bucket_idx = bucket_idx;

                loop {
                    if jobs_scheduled_this_macro >= target_jobs {
                        break;
                    }

                    if current_bucket_idx >= self.job_buckets.len() || self.job_buckets[current_bucket_idx].jobs.is_empty() {
                        break;
                    }

                    let job = self.job_buckets[current_bucket_idx].jobs.front().cloned();

                    if let Some(job) = job {
                        let scheduled = self.schedule_job_with_host_priorities(job.clone(), &priorities);

                        if scheduled > 0 {
                            jobs_scheduled_this_macro += 1;
                            self.job_buckets[current_bucket_idx].jobs.pop_front();
                            self.scheduling_attempts_this_batch += 1;

                            if self.job_buckets[current_bucket_idx].jobs.is_empty() {
                                self.job_buckets.remove(current_bucket_idx);
                                break;
                            } else {
                                self.sort_buckets_by_arrival_time();
                                if let Some(new_idx) = self.job_buckets.iter().position(|b| b.bucket_key == bucket_key) {
                                    current_bucket_idx = new_idx;
                                } else {
                                    break;
                                }
                            }
                        } else {
                            if jobs_scheduled_this_macro == 0 {
                                self.defer_bucket(current_bucket_idx);
                            }
                            break;
                        }
                    } else {
                        break;
                    }
                }
            }
        }

        self.last_macro_jobs_scheduled = jobs_scheduled_this_macro;
        self.last_macro_level = macro_level;
        self.total_macro_actions += 1;
        if self.total_macro_actions > 0 {
            self.avg_jobs_per_macro = (self.avg_jobs_per_macro * (self.total_macro_actions - 1) as f32
                                       + jobs_scheduled_this_macro as f32) / self.total_macro_actions as f32;
        }

        if macro_level < self.macro_level_counts.len() {
            self.macro_level_counts[macro_level] += 1;
        }

        let finishing_batch_now = self.should_finish_batch();

        self.update_environment_state(finishing_batch_now);

        let reward = self.calculate_step_reward(finishing_batch_now);

        if finishing_batch_now {
            self.advance_cycle();
            if self.try_start_batch() {
                for i in 0..self.job_buckets.len() {
                    self.set_bucket_priorities(i, None);
                }
            }
        }

        let done = self.check_episode_done();

        let info = PyDict::new(py);
        info.set_item("num_buckets", self.job_buckets.len())?;
        info.set_item("max_buckets_in_episode", self.max_buckets_in_episode)?;
        info.set_item("macro_jobs_scheduled", self.last_macro_jobs_scheduled)?;
        info.set_item("macro_level", self.last_macro_level)?;
        info.set_item("avg_jobs_per_macro", self.avg_jobs_per_macro)?;
        info.set_item("total_macro_actions", self.total_macro_actions)?;
        info.set_item("num_macro_levels", self.macro_levels.len())?;
        info.set_item("macro_level_counts", self.macro_level_counts.clone())?;
        info.set_item("queue_length", self.job_queue.len())?;
        info.set_item("active_jobs", self.active_jobs.len())?;
        info.set_item("total_jobs_generated", self.total_jobs_generated)?;
        info.set_item("total_jobs_completed", self.total_jobs_completed)?;

        Ok((self.get_state(py)?, reward, done, info.into()))
    }

    pub fn get_state(&mut self, py: Python) -> PyResult<Py<PyArray1<f32>>> {
        self.cached_state.fill(0.0);

        if self.job_buckets.len() > self.max_buckets {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!(
                    "Number of buckets ({}) exceeds max_buckets ({}). Increase --max-buckets parameter.",
                    self.job_buckets.len(),
                    self.max_buckets
                )
            ));
        }

        let num_buckets = self.job_buckets.len();
        for i in 0..num_buckets {
            let start_idx = i * 4;
            self.fill_bucket_features(i, start_idx);
        }

        let global_start_idx = self.max_buckets * 4;
        self.fill_global_features(global_start_idx);

        let num_valid_idx = self.max_buckets * 4 + 2;
        self.cached_state[num_valid_idx] = self.job_buckets.len() as f32;

        Ok(PyArray1::from_slice(py, &self.cached_state).to_owned())
    }

    pub fn needs_decision(&self) -> bool {
        self.job_buckets.iter().any(|b| !b.jobs.is_empty())
    }

    pub fn get_step_info(&self, py: Python) -> PyResult<PyObject> {
        self.base.get_step_info(py)
    }

    pub fn get_metrics(&self, py: Python) -> PyResult<PyObject> {
        self.base.get_metrics(py)
    }

    pub fn get_host_configs(&self, py: Python) -> PyResult<PyObject> {
        self.base.get_host_configs(py)
    }

    pub fn get_job_schedule(&self, py: Python) -> PyResult<PyObject> {
        self.base.get_job_schedule(py)
    }

    pub fn get_cluster_info(&self, py: Python) -> PyResult<PyObject> {
        self.base.get_cluster_info(py)
    }

    pub fn set_random_seed(&mut self, seed: Option<u64>) {
        self.base.set_random_seed(seed);
    }

    pub fn get_max_buckets(&self) -> usize {
        self.max_buckets
    }

    pub fn get_num_macro_levels(&self) -> usize {
        self.macro_levels.len()
    }

    pub fn get_macro_levels(&self) -> Vec<f32> {
        self.macro_levels.clone()
    }

    fn parse_action(&self, action: &PyAny) -> PyResult<(usize, usize)> {
        let action_arr = if let Ok(arr32) = action.extract::<PyReadonlyArray1<i64>>() {
            let slice = arr32.as_slice()?;
            if slice.len() != 2 {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    format!("Action array must have 2 elements [bucket_idx, macro_level], got {}", slice.len())
                ));
            }
            vec![slice[0] as usize, slice[1] as usize]
        } else if let Ok(arr64) = action.extract::<PyReadonlyArray1<f32>>() {
            let slice = arr64.as_slice()?;
            if slice.len() != 2 {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    format!("Action array must have 2 elements [bucket_idx, macro_level], got {}", slice.len())
                ));
            }
            vec![slice[0] as usize, slice[1] as usize]
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "`action` must be a 2-element array [bucket_idx, macro_level]",
            ));
        };

        let bucket_idx = action_arr[0];
        let macro_level = action_arr[1];

        if bucket_idx >= self.max_buckets {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Bucket index {} out of range [0, {})", bucket_idx, self.max_buckets)
            ));
        }

        if macro_level > self.macro_levels.len() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Macro level {} out of range [0, {}] for configured levels {:?} (level {} is single-job)",
                    macro_level, self.macro_levels.len(), self.macro_levels, self.macro_levels.len())
            ));
        }

        Ok((bucket_idx, macro_level))
    }

    fn calculate_target_jobs(&self, bucket_size: usize, macro_level: usize) -> usize {
        if macro_level >= self.macro_levels.len() {
            return 1;
        }

        let percentage = self.macro_levels.get(macro_level).copied().unwrap_or(1.0);
        if percentage >= 1.0 {
            bucket_size  // 100% (all jobs)
        } else {
            ((bucket_size as f32 * percentage).ceil() as usize).max(1)  // At least 1
        }
    }

    fn should_finish_batch(&self) -> bool {
        self.job_buckets.is_empty()
    }

    fn fill_global_features(&mut self, start_idx: usize) {
        let avail_cores: u32 = self.hosts.iter().map(|h| h.available_cores).sum();
        self.cached_state[start_idx] = avail_cores as f32 / self.total_cluster_cores.max(1) as f32;

        let avail_memory: u32 = self.hosts.iter().map(|h| h.available_memory).sum();
        self.cached_state[start_idx + 1] = avail_memory as f32 / self.total_cluster_memory.max(1) as f32;
    }

    fn fill_bucket_features(&mut self, bucket_idx: usize, start_idx: usize) {
        if bucket_idx >= self.job_buckets.len() {
            return;
        }

        let current_time = self.current_time as f64;
        let (bucket_cores, bucket_memory, job_count, waiting_time) = {
            let bucket = &self.job_buckets[bucket_idx];
            if let Some(first_job) = bucket.jobs.front() {
                let wait = current_time - first_job.submission_time;
                (
                    first_job.cores_required as f32,
                    first_job.memory_required as f32,
                    bucket.jobs.len() as f32,
                    wait as f32,
                )
            } else {
                (0.0, 0.0, 0.0, 0.0)
            }
        };

        self.cached_state[start_idx] = bucket_cores / (self.job_cores_range.1.max(1) as f32);
        self.cached_state[start_idx + 1] = bucket_memory / (self.job_memory_range.1.max(1) as f32);
        self.cached_state[start_idx + 2] = job_count / 100.0;
        self.cached_state[start_idx + 3] = (waiting_time / 300.0).min(1.0);
    }

    fn calculate_step_reward(&mut self, batch_complete: bool) -> f32 {
        if batch_complete {
            let next_sample_time = self.last_reward_time + self.accumulated_sample_count;
            if self.current_time >= next_sample_time {
                let utilization = self.calculate_pure_resource_utilization_reward();
                let (avg_waiting_penalty, max_waiting_penalty) = self.calculate_pending_waiting_penalties();
                self.accumulated_util_sum += utilization as f64;
                self.accumulated_avg_waiting_sum += avg_waiting_penalty as f64;
                self.accumulated_max_waiting_sum += max_waiting_penalty as f64;
                self.accumulated_sample_count += 1;
            }

            if self.current_time - self.last_reward_time >= self.reward_interval {
                let avg_util = if self.accumulated_sample_count > 0 {
                    (self.accumulated_util_sum / self.accumulated_sample_count as f64) as f32
                } else {
                    self.calculate_pure_resource_utilization_reward()
                };

                let avg_waiting_penalty = if self.accumulated_sample_count > 0 {
                    (self.accumulated_avg_waiting_sum / self.accumulated_sample_count as f64) as f32
                } else {
                    0.0
                };

                let max_waiting_penalty = if self.accumulated_sample_count > 0 {
                    (self.accumulated_max_waiting_sum / self.accumulated_sample_count as f64) as f32
                } else {
                    0.0
                };

                let total_penalty = avg_waiting_penalty + max_waiting_penalty;
                let reward = avg_util - total_penalty;

                self.accumulated_util_sum = 0.0;
                self.accumulated_avg_waiting_sum = 0.0;
                self.accumulated_max_waiting_sum = 0.0;
                self.accumulated_sample_count = 0;
                self.last_reward_time = self.current_time;

                reward
            } else {
                0.0
            }
        } else {
            0.0
        }
    }

    fn calculate_pure_resource_utilization_reward(&self) -> f32 {
        let mut used_cores: u32 = 0;
        let mut used_memory: u32 = 0;
        for host in &self.hosts {
            used_cores += host.total_cores - host.available_cores;
            used_memory += host.total_memory - host.available_memory;
        }
        let core_util = used_cores as f64 / self.total_cluster_cores.max(1) as f64;
        let mem_util = used_memory as f64 / self.total_cluster_memory.max(1) as f64;
        core_util.min(mem_util) as f32
    }

    fn calculate_pending_waiting_penalties(&self) -> (f32, f32) {
        let current_time = self.current_time as f64;
        let mut total_weighted_waiting = 0.0_f64;
        let mut max_weighted_waiting = 0.0_f64;
        let mut total_weight = 0.0_f64;

        let max_cores = self.job_cores_range.1 as f64;
        let max_memory = self.job_memory_range.1 as f64;

        for bucket in &self.job_buckets {
            for job in &bucket.jobs {
                let waiting_time = current_time - job.submission_time;
                let core_ratio = job.cores_required as f64 / max_cores;
                let mem_ratio = job.memory_required as f64 / max_memory;
                let job_size_weight = (core_ratio + mem_ratio) / 2.0;
                let weight = 1.0 + job_size_weight;
                let weighted_wait = waiting_time * weight;

                total_weighted_waiting += weighted_wait;
                max_weighted_waiting = max_weighted_waiting.max(weighted_wait);
                total_weight += weight;
            }
        }

        for bucket in &self.deferred_buckets {
            for job in &bucket.jobs {
                let waiting_time = current_time - job.submission_time;
                let core_ratio = job.cores_required as f64 / max_cores;
                let mem_ratio = job.memory_required as f64 / max_memory;
                let job_size_weight = (core_ratio + mem_ratio) / 2.0;
                let weight = 1.0 + job_size_weight;
                let weighted_wait = waiting_time * weight;

                total_weighted_waiting += weighted_wait;
                max_weighted_waiting = max_weighted_waiting.max(weighted_wait);
                total_weight += weight;
            }
        }

        if total_weight == 0.0 {
            return (0.0, 0.0);
        }

        let avg_weighted_waiting = total_weighted_waiting / total_weight;

        let avg_normalized = (avg_weighted_waiting / 300.0).min(1.0) as f32;
        let max_normalized = (max_weighted_waiting / 300.0).min(1.0) as f32;

        let avg_penalty = self.wait_penalty_coef * avg_normalized;
        let max_penalty = self.wait_penalty_coef * max_normalized;

        (avg_penalty, max_penalty)
    }

}
