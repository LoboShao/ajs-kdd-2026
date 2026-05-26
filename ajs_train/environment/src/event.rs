use std::cmp::Ordering;

#[derive(Debug, Clone, PartialEq)]
pub struct CompletionEvent {
    pub completion_time: f64,
    pub job_id: u32,
}

impl Eq for CompletionEvent {}

impl PartialOrd for CompletionEvent {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for CompletionEvent {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse ordering for BinaryHeap (min-heap behavior)
        other.completion_time.partial_cmp(&self.completion_time)
            .unwrap_or(Ordering::Equal)
            .then_with(|| other.job_id.cmp(&self.job_id))
    }
}