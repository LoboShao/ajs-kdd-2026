#[derive(Debug, Clone, PartialEq)]
pub enum JobStatus {
    Pending,
    Running,
    Completed,
    Failed,
}

#[derive(Debug, Clone)]
pub struct Job {
    pub id: u32,
    pub cores_required: u32,
    pub memory_required: u32,
    pub duration: u32,
    pub submission_time: f64,
    pub start_time: Option<f64>,
    pub end_time: Option<f64>,
    pub assigned_host: Option<usize>,
    pub status: JobStatus,
    pub deferred_count: u32,  // Number of times this job has been deferred
    pub bucket_key: String,    // Key for bucket grouping (e.g., "c_4_m_512")
}

impl Job {
    pub fn new(
        id: u32,
        cores_required: u32,
        memory_required: u32,
        duration: u32,
        submission_time: f64,
    ) -> Self {
        // Generate bucket key - using unique job ID for per-job scheduling
        // This ensures each job is in its own bucket
        let bucket_key = format!("job_{}", id);
        
        Job {
            id,
            cores_required,
            memory_required,
            duration,
            submission_time,
            start_time: None,
            end_time: None,
            assigned_host: None,
            status: JobStatus::Pending,
            deferred_count: 0,
            bucket_key,
        }
    }
}