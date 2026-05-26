use std::collections::{BinaryHeap, HashMap, VecDeque};

use pyo3::prelude::*;
use pyo3::types::PyDict;
use rand::prelude::*;

use crate::job::{Job, JobStatus};
use crate::host::Host;
use crate::event::CompletionEvent;

#[derive(Clone)]
pub struct ClusterConfig {
    pub num_hosts: usize,
    pub max_queue_length: Option<usize>,
    pub host_cores_range: (u32, u32),
    pub host_memory_range: (u32, u32),
    pub job_cores_range: (u32, u32),
    pub job_memory_range: (u32, u32),
    pub job_duration_range: (u32, u32),
    pub max_jobs_per_step: usize,
    pub max_time: usize,
    pub use_skewed_arrivals: bool,
    pub seed: Option<u64>,
}

impl ClusterConfig {
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
    ) -> Self {
        ClusterConfig {
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
        }
    }
}

#[derive(Debug, Clone)]
pub struct JobBucket {
    pub bucket_key: String,
    pub jobs: VecDeque<Job>,
    pub host_priorities: Option<Vec<f32>>,
    pub dispatched_count: usize,
}

pub struct BaseClusterEnv {
    pub num_hosts: usize,
    pub max_queue_length: usize,
    pub host_cores_range: (u32, u32),
    pub host_memory_range: (u32, u32),
    pub job_cores_range: (u32, u32),
    pub job_memory_range: (u32, u32),
    pub job_duration_range: (u32, u32),
    pub max_jobs_per_step: usize,
    pub max_time: usize,
    pub use_skewed_arrivals: bool,

    pub hosts: Vec<Host>,

    pub core_util_sum: f64,
    pub memory_util_sum: f64,
    pub last_stats_update_time: u64,

    pub job_queue: VecDeque<Job>,
    pub arrival_buffer: VecDeque<Job>,
    pub total_jobs_in_current_batch: usize,
    pub scheduling_attempts_this_batch: usize,
    pub job_buckets: Vec<JobBucket>,
    pub deferred_buckets: Vec<JobBucket>,
    pub active_jobs: HashMap<u32, Job>,
    pub completion_heap: BinaryHeap<CompletionEvent>,

    pub current_time: u64,
    pub current_step: usize,
    pub next_job_id: u32,

    pub job_arrival_schedule: Vec<usize>,
    pub job_cores_schedule: Vec<u32>,
    pub job_memory_schedule: Vec<u32>,
    pub job_duration_schedule: Vec<u32>,
    pub total_jobs_in_pool: usize,
    pub jobs_moved_to_queue: usize,

    pub total_jobs_generated: u32,
    pub total_jobs_completed: u32,
    pub total_jobs_deferred: u32,
    pub total_waiting_time_all_jobs: f64,
    pub makespan: Option<u64>,
    pub max_buckets_in_episode: usize,

    pub rng: StdRng,
    pub original_seed: Option<u64>,

    pub total_cluster_cores: u32,
    pub total_cluster_memory: u32,

}

impl BaseClusterEnv {
    pub fn from_config(config: &ClusterConfig) -> Self {
        Self::new(
            config.num_hosts,
            config.max_queue_length,
            config.host_cores_range,
            config.host_memory_range,
            config.job_cores_range,
            config.job_memory_range,
            config.job_duration_range,
            config.max_jobs_per_step,
            config.max_time,
            config.use_skewed_arrivals,
            config.seed,
        )
    }

    fn new(
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
    ) -> Self {
        let actual_max_queue_length = max_queue_length.unwrap_or(max_time * max_jobs_per_step);
        let original_seed = seed;
        let mut rng = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };

        let (job_arrival_schedule, job_cores_schedule, job_memory_schedule, job_duration_schedule, total_jobs_in_pool) =
            Self::generate_deterministic_job_schedule(
                max_time,
                max_jobs_per_step,
                job_cores_range,
                job_memory_range,
                job_duration_range,
                use_skewed_arrivals,
                &mut rng
            );

        let mut hosts = Vec::with_capacity(num_hosts);
        for i in 0..num_hosts {
            let (cores, memory) = Self::generate_realistic_host_config(host_cores_range, host_memory_range, &mut rng);
            hosts.push(Host::new(i, cores, memory));
        }

        let total_cluster_cores: u32 = hosts.iter().map(|h| h.total_cores).sum();
        let total_cluster_memory: u32 = hosts.iter().map(|h| h.total_memory).sum();

        BaseClusterEnv {
            num_hosts,
            max_queue_length: actual_max_queue_length,
            host_cores_range,
            host_memory_range,
            job_cores_range,
            job_memory_range,
            job_duration_range,
            max_jobs_per_step,
            max_time,
            use_skewed_arrivals,
            hosts,
            core_util_sum: 0.0,
            memory_util_sum: 0.0,
            last_stats_update_time: 0,
            job_queue: VecDeque::new(),
            arrival_buffer: VecDeque::new(),
            total_jobs_in_current_batch: 0,
            scheduling_attempts_this_batch: 0,
            job_buckets: Vec::new(),
            deferred_buckets: Vec::new(),
            active_jobs: HashMap::new(),
            completion_heap: BinaryHeap::new(),
            current_time: 0,
            current_step: 0,
            next_job_id: 0,
            job_arrival_schedule,
            job_cores_schedule,
            job_memory_schedule,
            job_duration_schedule,
            total_jobs_in_pool,
            jobs_moved_to_queue: 0,
            total_jobs_generated: 0,
            total_jobs_completed: 0,
            total_jobs_deferred: 0,
            total_waiting_time_all_jobs: 0.0,
            makespan: None,
            max_buckets_in_episode: 0,
            rng,
            original_seed,
            total_cluster_cores,
            total_cluster_memory,
        }
    }

    pub fn reset_base(&mut self) {
        let mut host_configs = Vec::with_capacity(self.num_hosts);

        if let Some(seed) = self.original_seed {
            let mut cluster_rng = StdRng::seed_from_u64(seed);
            for _i in 0..self.num_hosts {
                let (cores, memory) = Self::generate_realistic_host_config(
                    self.host_cores_range,
                    self.host_memory_range,
                    &mut cluster_rng
                );
                host_configs.push((cores, memory));
            }
        } else {
            for _i in 0..self.num_hosts {
                let (cores, memory) = Self::generate_realistic_host_config(
                    self.host_cores_range,
                    self.host_memory_range,
                    &mut self.rng
                );
                host_configs.push((cores, memory));
            }
        }
        for (i, (cores, memory)) in host_configs.into_iter().enumerate() {
            let host = &mut self.hosts[i];
            host.total_cores = cores;
            host.total_memory = memory;
            host.available_cores = cores;
            host.available_memory = memory;
            host.running_job_ids.clear();
        }

        self.total_cluster_cores = self.hosts.iter().map(|h| h.total_cores).sum();
        self.total_cluster_memory = self.hosts.iter().map(|h| h.total_memory).sum();

        self.job_queue.clear();
        self.arrival_buffer.clear();
        self.total_jobs_in_current_batch = 0;
        self.scheduling_attempts_this_batch = 0;
        self.job_buckets.clear();
        self.deferred_buckets.clear();
        self.active_jobs.clear();
        self.completion_heap.clear();

        self.current_time = 0;
        self.current_step = 0;
        self.next_job_id = 0;
        self.jobs_moved_to_queue = 0;

        self.total_jobs_generated = 0;
        self.total_jobs_completed = 0;
        self.total_jobs_deferred = 0;
        self.total_waiting_time_all_jobs = 0.0;
        self.makespan = None;
        self.max_buckets_in_episode = 0;

        self.core_util_sum = 0.0;
        self.memory_util_sum = 0.0;
        self.last_stats_update_time = 0;
        if let Some(seed) = self.original_seed {
            let mut schedule_rng = StdRng::seed_from_u64(seed);
            let (job_arrival_schedule, job_cores_schedule, job_memory_schedule, job_duration_schedule, total_jobs_in_pool) =
                Self::generate_deterministic_job_schedule(
                    self.max_time,
                    self.max_jobs_per_step,
                    self.job_cores_range,
                    self.job_memory_range,
                    self.job_duration_range,
                    self.use_skewed_arrivals,
                    &mut schedule_rng
                );

            self.job_arrival_schedule = job_arrival_schedule;
            self.job_cores_schedule = job_cores_schedule;
            self.job_memory_schedule = job_memory_schedule;
            self.job_duration_schedule = job_duration_schedule;
            self.total_jobs_in_pool = total_jobs_in_pool;
        } else {
            let (job_arrival_schedule, job_cores_schedule, job_memory_schedule, job_duration_schedule, total_jobs_in_pool) =
                Self::generate_deterministic_job_schedule(
                    self.max_time,
                    self.max_jobs_per_step,
                    self.job_cores_range,
                    self.job_memory_range,
                    self.job_duration_range,
                    self.use_skewed_arrivals,
                    &mut self.rng
                );

            self.job_arrival_schedule = job_arrival_schedule;
            self.job_cores_schedule = job_cores_schedule;
            self.job_memory_schedule = job_memory_schedule;
            self.job_duration_schedule = job_duration_schedule;
            self.total_jobs_in_pool = total_jobs_in_pool;
        }
    }

    pub fn generate_deterministic_job_schedule(
        max_time: usize,
        max_jobs_per_step: usize,
        job_cores_range: (u32, u32),
        job_memory_range: (u32, u32),
        job_duration_range: (u32, u32),
        use_skewed_arrivals: bool,
        rng: &mut StdRng,
    ) -> (Vec<usize>, Vec<u32>, Vec<u32>, Vec<u32>, usize) {
        let mut job_arrival_schedule = Vec::with_capacity(max_time);
        let mut job_cores_schedule = Vec::new();
        let mut job_memory_schedule = Vec::new();
        let mut job_duration_schedule = Vec::new();

        for _timestep in 0..max_time {
            let num_jobs = Self::generate_job_arrivals(max_jobs_per_step, use_skewed_arrivals, rng);
            job_arrival_schedule.push(num_jobs);

            for _job in 0..num_jobs {
                let (cores, memory, duration) = Self::generate_eda_job(
                    job_cores_range, job_memory_range, job_duration_range, rng
                );
                job_cores_schedule.push(cores);
                job_memory_schedule.push(memory);
                job_duration_schedule.push(duration);
            }
        }
        let total_jobs_in_pool = job_cores_schedule.len();

        (job_arrival_schedule, job_cores_schedule, job_memory_schedule, job_duration_schedule, total_jobs_in_pool)
    }

    fn generate_job_arrivals(max_jobs_per_step: usize, use_skewed: bool, rng: &mut StdRng) -> usize {
        if use_skewed {
            // Create a skewed distribution favoring higher values
            // Target: For max=50, average should be around 35 (70% of max)
            // Use Beta distribution transformed to desired range

            // Beta(2, 1) gives us right-skewed distribution (more values near 1.0)
            // For stronger skew toward high values, use Beta(3, 1) or even Beta(4, 1)
            let alpha = 2.0;
            let beta = 1.0;

            // Generate two uniform samples to create Beta distribution using acceptance-rejection
            let sample = loop {
                let u1: f64 = rng.gen();
                let u2: f64 = rng.gen();

                // Simple Beta(alpha, beta) using ratio of uniforms method
                let x = u1.powf(1.0 / alpha);
                let y = u2.powf(1.0 / beta);
                let sum = x + y;

                if sum <= 1.0 {
                    break x / sum;
                }
            };

            // Transform to range [1, max_jobs_per_step]
            let scaled = 1.0 + sample * (max_jobs_per_step - 1) as f64;

            scaled.round().max(1.0).min(max_jobs_per_step as f64) as usize
        } else {
            // Uniform distribution over [1, max_jobs_per_step]
            rng.gen_range(1..=max_jobs_per_step)
        }
    }

    pub fn generate_realistic_host_config(
        cores_range: (u32, u32),
        memory_range: (u32, u32),
        rng: &mut StdRng,
    ) -> (u32, u32) {
        // Common core configurations for EDA clusters
        const COMMON_CORES: &[u32] = &[8, 16, 20, 24, 28, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128];

        // Common memory configurations (in MB - matching env units)
        const COMMON_MEMORY_MB: &[u32] = &[
            32 * 1024,   // 32GB
            48 * 1024,   // 48GB
            64 * 1024,   // 64GB
            96 * 1024,   // 96GB
            128 * 1024,  // 128GB
            192 * 1024,  // 192GB
            256 * 1024,  // 256GB
            384 * 1024,  // 384GB
            512 * 1024,  // 512GB
            768 * 1024,  // 768GB
            1024 * 1024, // 1024GB
        ];

        let valid_cores: Vec<u32> = COMMON_CORES
            .iter()
            .filter(|&&c| c >= cores_range.0 && c <= cores_range.1)
            .cloned()
            .collect();

        // Filter memory within range (memory_range already in MB)
        let valid_memory_mb: Vec<u32> = COMMON_MEMORY_MB
            .iter()
            .filter(|&&m| m >= memory_range.0 && m <= memory_range.1)
            .cloned()
            .collect();

        let cores = if valid_cores.is_empty() {
            cores_range.0 + (cores_range.1 - cores_range.0) / 2
        } else {
            valid_cores[rng.gen_range(0..valid_cores.len())]
        };

        let memory_mb = if valid_memory_mb.is_empty() {
            memory_range.0 + (memory_range.1 - memory_range.0) / 2
        } else {
            valid_memory_mb[rng.gen_range(0..valid_memory_mb.len())]
        };

        (cores, memory_mb)
    }

    /// Generate EDA job with cores, memory, and duration based on common job archetypes.
    /// Uses discrete standard values (like real LSF jobs) to limit bucket count.
    /// Weighted heavily toward small jobs (realistic production workload).
    ///
    /// Duration is correlated with job size: larger jobs (more cores/memory) run longer.
    /// Jobs with the same (cores, memory) will have similar durations (±20% noise).
    ///
    /// Job Types (cores/memory distribution):
    /// - 50% Tiny:   Compile, lint        (smallest 1/3 of range)
    /// - 25% Small:  DRC, LVS, unit tests (lower-mid range)
    /// - 15% Medium: Block synthesis, sim (mid-upper range)
    /// - 10% Large:  P&R, timing analysis (largest 1/2 of range)
    fn generate_eda_job(
        cores_range: (u32, u32),
        memory_range: (u32, u32),
        duration_range: (u32, u32),
        rng: &mut StdRng
    ) -> (u32, u32, u32) {
        // Standard EDA core counts (common LSF request values)
        const COMMON_CORES: &[u32] = &[1, 2, 4, 8, 16, 32, 64];
        // Standard EDA memory sizes in GB
        const COMMON_MEMORY_GB: &[u32] = &[1, 2, 4, 8, 16, 32, 64];

        let valid_cores: Vec<u32> = COMMON_CORES.iter()
            .filter(|&&c| c >= cores_range.0 && c <= cores_range.1)
            .cloned()
            .collect();
        let valid_memory_gb: Vec<u32> = COMMON_MEMORY_GB.iter()
            .filter(|&&m| m >= memory_range.0 / 1024 && m <= memory_range.1 / 1024)
            .cloned()
            .collect();

        let valid_cores = if valid_cores.is_empty() {
            vec![cores_range.0, (cores_range.0 + cores_range.1) / 2, cores_range.1]
        } else {
            valid_cores
        };
        let valid_memory_gb = if valid_memory_gb.is_empty() {
            vec![memory_range.0 / 1024, memory_range.1 / 1024]
        } else {
            valid_memory_gb
        };

        let roll: f32 = rng.gen();

        let n_cores = valid_cores.len();
        let n_mem = valid_memory_gb.len();

        let (cores_idx_range, mem_idx_range) = if roll < 0.50 {
            // 50% Tiny: smallest cores/memory
            (0..((n_cores + 2) / 3).max(1), 0..((n_mem + 2) / 3).max(1))
        } else if roll < 0.75 {
            // 25% Small: lower-mid range
            let c_lo = n_cores / 4;
            let c_hi = (n_cores * 2 / 3).max(c_lo + 1);
            let m_lo = n_mem / 4;
            let m_hi = (n_mem * 2 / 3).max(m_lo + 1);
            (c_lo..c_hi, m_lo..m_hi)
        } else if roll < 0.90 {
            // 15% Medium: mid-upper range
            let c_lo = n_cores / 3;
            let c_hi = (n_cores * 3 / 4).max(c_lo + 1);
            let m_lo = n_mem / 3;
            let m_hi = (n_mem * 3 / 4).max(m_lo + 1);
            (c_lo..c_hi, m_lo..m_hi)
        } else {
            // 10% Large: largest cores/memory
            let c_lo = n_cores / 2;
            let m_lo = n_mem / 2;
            (c_lo..n_cores, m_lo..n_mem)
        };

        let cores_idx = rng.gen_range(cores_idx_range.start..=cores_idx_range.end.saturating_sub(1).max(cores_idx_range.start));
        let mem_idx = rng.gen_range(mem_idx_range.start..=mem_idx_range.end.saturating_sub(1).max(mem_idx_range.start));

        let cores = valid_cores[cores_idx.min(n_cores - 1)];
        let memory_gb = valid_memory_gb[mem_idx.min(n_mem - 1)];

        // Duration based on actual job size (cores + memory indices), not job type
        // This ensures jobs with same (cores, memory) have similar durations
        let size_factor = (cores_idx as f32 / (n_cores - 1).max(1) as f32
                        + mem_idx as f32 / (n_mem - 1).max(1) as f32) / 2.0;

        // Add ±20% noise for variation
        let noise = rng.gen_range(0.8_f32..1.2_f32);
        let dur_span = (duration_range.1 - duration_range.0) as f32;
        let duration = (duration_range.0 as f32 + size_factor * dur_span * noise)
            .clamp(duration_range.0 as f32, duration_range.1 as f32) as u32;

        (cores, memory_gb * 1024, duration)
    }

    pub fn add_new_jobs_to_arrival_buffer(&mut self) {
        let timestep = self.current_time as usize;

        if timestep >= self.max_time {
            return;
        }

        if timestep >= self.job_arrival_schedule.len() {
            panic!("Timestep {} exceeds job arrival schedule length {}. This indicates a design error.",
                   timestep, self.job_arrival_schedule.len());
        }

        if self.jobs_moved_to_queue >= self.total_jobs_in_pool {
            // This could happen near episode end, just return instead of panicking
            return;
        }

        let num_jobs_to_add = self.job_arrival_schedule[timestep];
        let total_pending = self.job_queue.len() + self.arrival_buffer.len();
        let jobs_to_add = num_jobs_to_add.min(self.max_queue_length - total_pending);

        // Each bsub command costs 1/max_jobs_per_step seconds,
        // so jobs within a second get staggered fractional arrival times
        let submission_interval = 1.0 / self.max_jobs_per_step as f64;

        for i in 0..jobs_to_add {
            if self.jobs_moved_to_queue >= self.total_jobs_in_pool {
                panic!("Job pool exhausted during job addition. Pool size: {}, Jobs moved: {}",
                       self.total_jobs_in_pool, self.jobs_moved_to_queue);
            }

            let cores = self.job_cores_schedule[self.jobs_moved_to_queue];
            let memory = self.job_memory_schedule[self.jobs_moved_to_queue];
            let duration = self.job_duration_schedule[self.jobs_moved_to_queue];

            let submission_time = self.current_time as f64 + i as f64 * submission_interval;

            let job = Job::new(
                self.next_job_id,
                cores,
                memory,
                duration,
                submission_time,
            );

            self.next_job_id += 1;
            self.total_jobs_generated += 1;
            self.jobs_moved_to_queue += 1;
            self.arrival_buffer.push_back(job);
        }
    }

    pub fn flush_arrival_buffer(&mut self) {
        while let Some(job) = self.arrival_buffer.pop_front() {
            self.job_queue.push_back(job);
        }
    }

    pub fn start_batch_processing(&mut self) {
        self.restore_deferred_buckets();

        while let Some(job) = self.job_queue.pop_front() {
            self.add_job_to_bucket(job);
        }

        self.sort_buckets_by_arrival_time();

        self.total_jobs_in_current_batch = self.job_buckets.iter()
            .map(|b| b.jobs.len())
            .sum();

        for bucket in &mut self.job_buckets {
            bucket.dispatched_count = 0;
        }

        self.scheduling_attempts_this_batch = 0;
    }

    pub fn finish_batch_processing(&mut self) {
        for bucket in &mut self.job_buckets {
            bucket.dispatched_count = 0;
            bucket.host_priorities = None;
        }

        self.job_buckets.retain(|bucket| !bucket.jobs.is_empty());

        self.total_jobs_in_current_batch = 0;
        self.scheduling_attempts_this_batch = 0;
    }

    pub fn generate_bucket_key(_job_id: u32, _cores: u32, _memory: u32) -> String {
        format!("c_{}_m_{}", _cores, _memory)
    }

    pub fn add_job_to_bucket(&mut self, job: Job) {
        let key = Self::generate_bucket_key(job.id, job.cores_required, job.memory_required);
        let bucket_index = self.job_buckets.iter().position(|b| b.bucket_key == key);

        match bucket_index {
            Some(idx) => {
                self.job_buckets[idx].jobs.push_back(job);
            }
            None => {
                let mut new_bucket = JobBucket {
                    bucket_key: key,
                    jobs: VecDeque::new(),
                    host_priorities: None,
                    dispatched_count: 0,
                };
                new_bucket.jobs.push_back(job);
                self.job_buckets.push(new_bucket);

                if self.job_buckets.len() > self.max_buckets_in_episode {
                    self.max_buckets_in_episode = self.job_buckets.len();
                }
            }
        }
    }

    pub fn sort_buckets_by_arrival_time(&mut self) {
        self.job_buckets.sort_by(|a, b| {
            let a_time = a.jobs.front().map(|j| j.submission_time).unwrap_or(f64::MAX);
            let b_time = b.jobs.front().map(|j| j.submission_time).unwrap_or(f64::MAX);
            a_time.partial_cmp(&b_time).unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    pub fn schedule_job_with_host_priorities(&mut self, job: Job, host_priorities: &[f32]) -> usize {
        let can_be_scheduled = self.hosts.iter().any(|host| {
            host.total_cores >= job.cores_required && host.total_memory >= job.memory_required
        });

        if !can_be_scheduled {
            println!("WARNING: Job {} cannot be scheduled on any host", job.id);
            return 0;
        }

        let mut sorted_host_priorities: Vec<(usize, f32)> = host_priorities.iter()
            .enumerate()
            .map(|(i, &p)| (i, p))
            .collect();
        sorted_host_priorities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        for &(host_idx, _) in &sorted_host_priorities {
            if self.try_single_host_scheduling(&job, host_idx) {
                return 1;
            }
        }

        0
    }

    pub fn defer_bucket(&mut self, bucket_idx: usize) {
        if bucket_idx >= self.job_buckets.len() {
            return;
        }

        let mut bucket = self.job_buckets.remove(bucket_idx);
        bucket.host_priorities = None;

        for job in &mut bucket.jobs {
            job.deferred_count += 1;
        }
        self.total_jobs_deferred += bucket.jobs.len() as u32;
        self.deferred_buckets.push(bucket);
    }

    pub fn restore_deferred_buckets(&mut self) {
        self.job_buckets.append(&mut self.deferred_buckets);
        self.deferred_buckets.clear();
    }

    pub fn try_single_host_scheduling(&mut self, job: &Job, host_idx: usize) -> bool {
        let host = &mut self.hosts[host_idx];

        if host.can_accommodate(job) {
            let mut scheduled_job = job.clone();

            if host.allocate_job(&mut scheduled_job) {
                scheduled_job.status = JobStatus::Running;
                scheduled_job.start_time = Some(self.current_time as f64);

                let waiting_time = self.current_time as f64 - scheduled_job.submission_time;
                self.total_waiting_time_all_jobs += waiting_time;

                let release_noise: f64 = self.rng.gen_range(0.0..2.0);
                let completion_time = (self.current_time + scheduled_job.duration as u64) as f64 + release_noise;
                self.completion_heap.push(CompletionEvent {
                    completion_time,
                    job_id: scheduled_job.id,
                });

                self.active_jobs.insert(scheduled_job.id, scheduled_job);

                return true;
            }
        }
        false
    }

    pub fn process_completions(&mut self) {
        while let Some(event) = self.completion_heap.peek() {
            if event.completion_time > self.current_time as f64 {
                break;
            }

            let event = self.completion_heap.pop().unwrap();

            if let Some(mut completed_job) = self.active_jobs.remove(&event.job_id) {
                completed_job.status = JobStatus::Completed;
                completed_job.end_time = Some(self.current_time as f64);

                if let Some(host_id) = completed_job.assigned_host {
                    self.hosts[host_id].release_job(&completed_job);
                }

                self.total_jobs_completed += 1;
            }
        }
    }

    pub fn update_host_utilization(&mut self) {
        if self.current_time > self.last_stats_update_time {
            for host in self.hosts.iter_mut() {
                host.update_utilization_history();
            }

            self.update_utilization_statistics_optimized(self.current_time);
        }
    }

    fn update_utilization_statistics_optimized(&mut self, current_time: u64) {
        self.last_stats_update_time = current_time;

        let mut used_cores: u32 = 0;
        let mut used_memory: u32 = 0;
        for host in &self.hosts {
            used_cores += host.total_cores - host.available_cores;
            used_memory += host.total_memory - host.available_memory;
        }
        let core_util = used_cores as f64 / self.total_cluster_cores.max(1) as f64;
        let mem_util = used_memory as f64 / self.total_cluster_memory.max(1) as f64;

        self.core_util_sum += core_util;
        self.memory_util_sum += mem_util;
    }

    pub fn try_start_batch(&mut self) -> bool {
        if self.total_jobs_in_current_batch == 0
           && (!self.job_queue.is_empty() || !self.deferred_buckets.is_empty())
        {
            self.start_batch_processing();
            true
        } else {
            false
        }
    }

    pub fn set_bucket_priorities(&mut self, bucket_idx: usize, priorities: Option<&[f32]>) {
        if bucket_idx >= self.job_buckets.len() {
            return;
        }

        let priorities_vec = if let Some(p) = priorities {
            p.to_vec()
        } else {
            self.get_default_host_priorities()
        };

        self.job_buckets[bucket_idx].host_priorities = Some(priorities_vec);
    }

    fn get_default_host_priorities(&self) -> Vec<f32> {
        self.hosts.iter()
            .map(|host| {
                let core_avail = host.available_cores as f32 / host.total_cores.max(1) as f32;
                let mem_avail = host.available_memory as f32 / host.total_memory.max(1) as f32;
                (core_avail + mem_avail) / 2.0
            })
            .collect()
    }

    pub fn advance_cycle(&mut self) {
        self.current_time += 1;
        self.flush_arrival_buffer();

        let all_jobs_generated = self.jobs_moved_to_queue >= self.total_jobs_in_pool;
        if !all_jobs_generated {
            self.add_new_jobs_to_arrival_buffer();
        }

        self.current_step += 1;
    }

    pub fn update_environment_state(&mut self, finishing_batch_now: bool) {
        if finishing_batch_now {
            self.finish_batch_processing();
        }

        self.process_completions();
        self.update_host_utilization();
    }

    pub fn check_episode_done(&mut self) -> bool {
        let all_jobs_generated = self.jobs_moved_to_queue >= self.total_jobs_in_pool;
        let all_jobs_scheduled = self.job_queue.is_empty()
            && self.job_buckets.is_empty()
            && self.deferred_buckets.is_empty();

        let done = all_jobs_generated && all_jobs_scheduled;

        if done && self.makespan.is_none() {
            self.makespan = Some(self.current_time);
        }

        done
    }

    pub fn get_step_info(&self, py: Python) -> PyResult<PyObject> {
        let info = PyDict::new(py);
        info.set_item("queue_length", self.job_queue.len())?;

        let bucket_jobs_count: usize = self.job_buckets.iter().map(|b| b.jobs.len()).sum();
        info.set_item("bucket_jobs_count", bucket_jobs_count)?;
        info.set_item("num_buckets", self.job_buckets.len())?;
        info.set_item("max_buckets_in_episode", self.max_buckets_in_episode)?;

        info.set_item("active_jobs", self.active_jobs.len())?;
        info.set_item("needs_decision", self.job_buckets.iter().any(|b| !b.jobs.is_empty()))?;
        info.set_item("total_jobs_generated", self.total_jobs_generated)?;
        info.set_item("total_jobs_completed", self.total_jobs_completed)?;
        info.set_item("current_time", self.current_time)?;
        info.set_item("current_step", self.current_step)?;
        info.set_item("total_jobs_deferred", self.total_jobs_deferred)?;

        if let Some(makespan_time) = self.makespan {
            info.set_item("makespan", makespan_time)?;
        }

        Ok(info.into())
    }

    pub fn get_metrics(&self, py: Python) -> PyResult<PyObject> {
        let episode_duration = self.current_time.max(1) as f64;
        let avg_core_util = (self.core_util_sum / episode_duration) as f32;
        let avg_memory_util = (self.memory_util_sum / episode_duration) as f32;

        let metrics = PyDict::new(py);

        let avg_waiting_time = if self.total_jobs_completed > 0 {
            self.total_waiting_time_all_jobs / self.total_jobs_completed as f64
        } else {
            0.0
        };

        metrics.set_item("total_jobs_completed", self.total_jobs_completed)?;
        metrics.set_item("avg_waiting_time", avg_waiting_time)?;

        if let Some(makespan_time) = self.makespan {
            metrics.set_item("makespan", makespan_time as f64)?;
        } else {
            metrics.set_item("makespan", py.None())?;
        }

        metrics.set_item("avg_host_core_utilization", avg_core_util)?;
        metrics.set_item("avg_host_memory_utilization", avg_memory_util)?;
        metrics.set_item("max_buckets_in_episode", self.max_buckets_in_episode)?;

        Ok(metrics.into())
    }

    pub fn get_host_configs(&self, py: Python) -> PyResult<PyObject> {
        let hosts_list = pyo3::types::PyList::empty(py);

        for (i, host) in self.hosts.iter().enumerate() {
            let host_dict = pyo3::types::PyDict::new(py);
            host_dict.set_item("host_id", i)?;
            host_dict.set_item("total_cores", host.total_cores)?;
            host_dict.set_item("total_memory", host.total_memory)?;
            hosts_list.append(host_dict)?;
        }

        Ok(hosts_list.to_object(py))
    }

    pub fn get_job_schedule(&self, py: Python) -> PyResult<PyObject> {
        let schedule_dict = pyo3::types::PyDict::new(py);

        schedule_dict.set_item("job_arrival_schedule", self.job_arrival_schedule.clone())?;
        schedule_dict.set_item("job_cores_schedule", self.job_cores_schedule.clone())?;
        schedule_dict.set_item("job_memory_schedule", self.job_memory_schedule.clone())?;
        schedule_dict.set_item("job_duration_schedule", self.job_duration_schedule.clone())?;
        schedule_dict.set_item("total_jobs_in_pool", self.total_jobs_in_pool)?;
        schedule_dict.set_item("max_time", self.max_time)?;
        schedule_dict.set_item("max_jobs_per_step", self.max_jobs_per_step)?;
        schedule_dict.set_item("num_hosts", self.num_hosts)?;
        schedule_dict.set_item("host_cores_range", self.host_cores_range)?;
        schedule_dict.set_item("host_memory_range", self.host_memory_range)?;
        schedule_dict.set_item("job_cores_range", self.job_cores_range)?;
        schedule_dict.set_item("job_memory_range", self.job_memory_range)?;
        schedule_dict.set_item("job_duration_range", self.job_duration_range)?;

        Ok(schedule_dict.to_object(py))
    }

    pub fn get_cluster_info(&self, py: Python) -> PyResult<PyObject> {
        let info_dict = pyo3::types::PyDict::new(py);
        info_dict.set_item("total_cluster_cores", self.total_cluster_cores)?;
        info_dict.set_item("total_cluster_memory", self.total_cluster_memory)?;
        info_dict.set_item("num_hosts", self.num_hosts)?;
        info_dict.set_item("host_cores_range", self.host_cores_range)?;
        info_dict.set_item("host_memory_range", self.host_memory_range)?;
        Ok(info_dict.to_object(py))
    }

    pub fn set_random_seed(&mut self, seed: Option<u64>) {
        self.original_seed = seed;
        if let Some(s) = seed {
            self.rng = rand::SeedableRng::seed_from_u64(s);
        } else {
            self.rng = rand::SeedableRng::from_entropy();
        }
    }
}