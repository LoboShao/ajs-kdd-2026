/***************************************************************************
 * AJS Plugin for IBM Spectrum LSF
 *
 * Purpose: Implements machine learning-based job ordering for LSF scheduler
 * Version: 1.0
 * Date: 2025
 ***************************************************************************/

#include <stdlib.h>
#include <sys/types.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <curl/curl.h>
#include "lssched.h"
#include "lsf.h"
#include "lsbatch.h"

/***************************************************************************
 * CONSTANTS AND CONFIGURATION
 ***************************************************************************/

/* Plugin Configuration */
static const int HANDLER_ID = 118;

/* ML Server configuration */
#define ML_SERVER_URL "http://localhost:5002/select_bucket"
#define JSON_BUFFER_SIZE 65536
#define ML_SERVER_TIMEOUT 5L

/* Local hostname - change this when deploying to a different host */
#define LOCAL_HOST "yiming-dev1.fyre.ibm.com"

/* System Constants */
#ifndef MAXLINELEN
#define MAXLINELEN 512
#endif

#ifndef FREEUP
#define FREEUP(p) { if ( p ) { free(p); (p) = NULL; } }
#endif

/* Host Tracking */
#define MAX_HOSTS 50                    /* Maximum number of tracked hosts */

/* ML Model Configuration */
#define MAX_BUCKETS 10                  /* Maximum number of job buckets */
#define JOB_MAX_SLOTS 4.0              /* Maximum slots for job normalization */
#define JOB_MAX_MEM (1024.0 * 8.0)      /* Maximum memory for job normalization (8GB in MB) */
/* Cluster totals - computed once from first sort when cluster is fully idle */
static double total_cluster_slots = 0.0;   /* Sum of all host max slots */
static double total_cluster_memory = 0.0;  /* Sum of all host max memory in MB */
static int total_cluster_initialized = 0;  /* 1 after first sort sets totals */

/***************************************************************************
 * GLOBAL STATE MANAGEMENT
 ***************************************************************************/

/* Remaining resource ratios - computed from host_table
 * Matches bucket_adaptive_env.rs: remaining_cores_ratio / remaining_memory_ratio */
static double remaining_cores_ratio = 1.0;
static double remaining_memory_ratio = 1.0;

/* Per-host resource tracking table
 * Initialized by ajs_sort, updated by notifyAllocFn on allocate/deallocate */
typedef struct {
    char hostname[256];
    double remaining_slots;
    double remaining_memory;  /* in MB */
    int active;
} host_resource_entry;

static host_resource_entry host_table[MAX_HOSTS];
static int host_table_count = 0;
static int host_table_initialized = 0;  /* 1 after first sort populates table this cycle */

/* Simple call counter for first-call detection (not used in ML vector) */
static int scheduling_calls_this_cycle = 0;

/* Macro action state tracking
 * The ML model outputs (bucket_index, macro_action) where:
 *   - bucket_index: which bucket to dispatch from
 *   - macro_action: 0=20%, 1=50%, 2=100% of jobs in that bucket
 * We track state across job_ordering calls to execute the macro action.
 * Note: Bucket pointer changes after each dispatch, so we identify by (slots, memory).
 */
static int remaining_dispatch_attempts = 0;    /* Jobs left to try from current bucket */
static double target_bucket_slots = 0.0;       /* Slots of target bucket for identification */
static double target_bucket_memory = 0.0;      /* Memory of target bucket for identification */

/* Cycle tracking */
static int current_cycle_number = 0;

/* Dispatch/notify counters per cycle (for verifying notify matches order) */
static int order_dispatch_count = 0;
static int notify_alloc_count = 0;
static int notify_dealloc_count = 0;

/***************************************************************************
 * DATA STRUCTURES
 ***************************************************************************/

/* Memory buffer for CURL responses */
struct MemoryStruct {
    char *memory;
    size_t size;
};

/* Plugin data structure for AJS requests */
typedef struct {
    int useAJS;
    double job_slots;           /* Slots for this job/bucket */
    double job_memory;          /* Memory for this job/bucket in MB */
} ajs_data;

/***************************************************************************
 * FUNCTION DECLARATIONS
 ***************************************************************************/

/* === Resource Request Handler Callbacks === */
static int ajs_new(void *resreq);
static int ajs_sort(ajs_data *data, void *candGroupList, void *reasonTb);
static void ajs_free(ajs_data *data);
static int ajs_checkAlloc(ajs_data *data, void *job, void *alloc, void *allocLimitList);
static int ajs_notifyAlloc(ajs_data *data, void *job, void *alloc, void *allocLimitList, int flag);

/* === Job Ordering === */
static INT_JobBlock *ajs_job_ordering(char *queueName, INT_JobList *jobList);

/* === Data Management === */
static ajs_data *create_ajs_data(void);
static void destroy_ajs_data(void *p);

/* === Resource Extraction === */
static int extract_job_slots_from_resreq(void *resreq);
static int extract_job_memory_from_resreq(void *resreq);
static int extract_job_slots_from_jobblock(INT_JobBlock *jobBlock);
static double extract_job_memory_from_jobblock(INT_JobBlock *jobBlock);
static double extract_job_waiting_time(INT_JobBlock *jobBlock);

/* === ML Vector Generation === */
static double *build_ml_input_vector(INT_JobList *jobList, int *vector_length);

/* === ML Inference === */
static int ml_inference(double *input_vector, int vector_length,
                        int num_buckets, int *out_bucket_index, int *out_macro_action);
static INT_JobBlock *find_bucket_by_index(INT_JobList *jobList, int bucket_index);

/* === CURL Functions === */
static size_t WriteMemoryCallback(void *contents, size_t size, size_t nmemb, void *userp);
static char *callMLServer(const char *jsonData);
static int parseMLResponse(const char *jsonResponse, int *bucket_index, int *macro_action);

/* === Utility Functions === */
static int is_localhost(const char *hostname);
static void free_host_ids(char *hostname, char *clustername);

/* === Host Table Functions === */
static int host_table_find(const char *hostname);
static void host_table_add_or_update(const char *hostname, double slots, double memory);
static void host_table_compute_ratios(void);

/***************************************************************************
 * HOST TABLE FUNCTIONS
 ***************************************************************************/

static int host_table_find(const char *hostname)
{
    if (!hostname) return -1;
    for (int i = 0; i < host_table_count; i++) {
        if (host_table[i].active && strcmp(host_table[i].hostname, hostname) == 0) {
            return i;
        }
    }
    return -1;
}

static void host_table_add_or_update(const char *hostname, double slots, double memory)
{
    if (!hostname) return;

    int idx = host_table_find(hostname);
    if (idx >= 0) {
        host_table[idx].remaining_slots = slots;
        host_table[idx].remaining_memory = memory;
        return;
    }

    if (host_table_count >= MAX_HOSTS) {
        ls_syslog(LOG_WARNING, "host_table_add_or_update: host table full (%d), ignoring %s",
                  MAX_HOSTS, hostname);
        return;
    }

    strncpy(host_table[host_table_count].hostname, hostname, sizeof(host_table[0].hostname) - 1);
    host_table[host_table_count].hostname[sizeof(host_table[0].hostname) - 1] = '\0';
    host_table[host_table_count].remaining_slots = slots;
    host_table[host_table_count].remaining_memory = memory;
    host_table[host_table_count].active = 1;
    host_table_count++;
}

static void host_table_compute_ratios(void)
{
    double total_slots = 0.0;
    double total_memory = 0.0;

    for (int i = 0; i < host_table_count; i++) {
        if (host_table[i].active) {
            total_slots += host_table[i].remaining_slots;
            total_memory += host_table[i].remaining_memory;
        }
    }

    remaining_cores_ratio = (total_cluster_slots > 0.0) ? total_slots / total_cluster_slots : 0.0;
    remaining_memory_ratio = (total_cluster_memory > 0.0) ? total_memory / total_cluster_memory : 0.0;
}

/***************************************************************************
 * PLUGIN FRAMEWORK IMPLEMENTATION (MANDATORY)
 ***************************************************************************/

/**
 * sched_version - Report plugin version compatibility
 */
int sched_version(void *param)
{
    return (0);
}

/**
 * sched_init - Initialize the AJS plugin
 * Registers handlers and job ordering function with LSF
 */
int sched_init(void *param)
{
    static char fname[] = "ajs_sched_init";
    RsrcReqHandlerType *handler = NULL;

    /* Allocate handler structure */
    handler = (RsrcReqHandlerType *)calloc(1, sizeof(RsrcReqHandlerType));
    if (handler == NULL) {
        ls_syslog(LOG_ERR, "%s: calloc() failed", fname);
        return (-1);
    }

    /* Register callback functions */
    handler->newFn = (RsrcReqHandler_NewFn) ajs_new;
    handler->freeFn = (RsrcReqHandler_FreeFn) ajs_free;
    handler->matchFn = (RsrcReqHandler_MatchFn) NULL;
    handler->sortFn = (RsrcReqHandler_SortFn) ajs_sort;
    handler->notifyAllocFn = (RsrcReqHandler_NotifyAllocFn) ajs_notifyAlloc;
    handler->checkAllocFn = (RsrcReqHandler_CheckAllocFn) ajs_checkAlloc;

    /* Register handler with LSF */
    extsched_resreq_registerhandler(HANDLER_ID, handler);
    FREEUP(handler);

    /* Register the job ordering function for all queues */
    extsched_order_registerOrderFn4AllQueues(ajs_job_ordering);

    /* Initialize CURL globally */
    curl_global_init(CURL_GLOBAL_ALL);

    ls_syslog(LOG_INFO, "AJS plugin initialized");
    return (0);
}

/**
 * sched_pre_proc - Called at the START of each scheduling cycle
 */
int sched_pre_proc(void *param)
{
    /* Log previous cycle's dispatch vs notify counts (if any dispatches happened) */
    if (order_dispatch_count > 0 || notify_alloc_count > 0 || notify_dealloc_count > 0) {
        ls_syslog(LOG_INFO, "AJS_CYCLE_SUMMARY: cycle=%d dispatched=%d alloc_notify=%d dealloc_notify=%d %s",
                  current_cycle_number, order_dispatch_count, notify_alloc_count, notify_dealloc_count,
                  (order_dispatch_count == notify_alloc_count) ? "MATCH" : "MISMATCH");
    }

    current_cycle_number++;

    scheduling_calls_this_cycle = 0;

    /* Reset dispatch/notify counters */
    order_dispatch_count = 0;
    notify_alloc_count = 0;
    notify_dealloc_count = 0;

    /* Reset macro action state */
    remaining_dispatch_attempts = 0;
    target_bucket_slots = 0.0;
    target_bucket_memory = 0.0;

    return (0);
}

int sched_match_limit(void *param)
{
    return (0);
}

int sched_order_alloc(void *param)
{
    return (0);
}

/**
 * sched_post_proc - Called at the END of each scheduling cycle
 */
int sched_post_proc(void *param)
{
    return (0);
}

int sched_finalize(void *param)
{
    /* Cleanup CURL */
    curl_global_cleanup();
    return (0);
}

/***************************************************************************
 * RESOURCE REQUEST HANDLER CALLBACKS
 ***************************************************************************/

/**
 * ajs_new - Register handler data for every job
 * Since AJS is the only scheduler, always register so that
 * notifyAllocFn/checkAllocFn are called for all jobs.
 */
static int ajs_new(void *resreq)
{
    ajs_data *data;
    char key[MAXLINELEN];
    int memMB = 0;
    int numCores = 0;

    if (resreq == NULL) {
        return (0);
    }

    {
        data = create_ajs_data();
        if (data == NULL) {
            return (-1);
        }

        data->useAJS = 1;

        /* Extract job slots from resreq using proper API */
        numCores = extract_job_slots_from_resreq(resreq);
        if (numCores <= 0) {
            numCores = 1;  /* Default 1 slot */
        }
        data->job_slots = (double)numCores;

        /* Extract job memory from resreq using proper API */
        memMB = extract_job_memory_from_resreq(resreq);
        if (memMB <= 0) {
            memMB = 256;  /* Default 256MB */
        }
        data->job_memory = (double)memMB;

        snprintf(key, sizeof(key), "#AJS#m%dc%d#", memMB, numCores);

        extsched_resreq_setobject(resreq, HANDLER_ID, key, data);
    }

    return (0);
}

/**
 * ajs_sort - Sort candidate hosts and populate per-host resource table
 *
 * Populates host_table once (on the very first call).
 * All subsequent updates are handled by notifyAllocFn.
 */
static int ajs_sort(ajs_data *data, void *candGroupList, void *reasonTb)
{
    struct candHostGroup *candGroupEntry = NULL;
    int num_hosts = 0;

    if (data == NULL) {
        return 0;
    }

    /* Only populate host table once */
    if (host_table_initialized) {
        return 0;
    }

    if (candGroupList == NULL) {
        return 0;
    }

    candGroupEntry = lsb_cand_getnextgroup(candGroupList);
    if (candGroupEntry == NULL) {
        return 0;
    }

    /* Clear host table and rebuild from candidate hosts */
    host_table_count = 0;
    memset(host_table, 0, sizeof(host_table));

    num_hosts = candGroupEntry->numOfMembers;

    for (int i = 0; i < num_hosts; i++) {
        struct candHost *host = &candGroupEntry->candHost[i];

        char *hostname = NULL;
        char *clustername = NULL;
        int skip_host = 0;

        if (extsched_getHostID(host->hostPtr, &hostname, &clustername) == 0) {
            if (is_localhost(hostname)) {
                skip_host = 1;
            }
        }

        if (!skip_host && hostname) {
            double host_slots = 0.0;
            double host_memory = 0.0;

            struct hostResources *hostRes = extsched_host_resources(host->hostPtr);
            if (hostRes != NULL) {
                for (int j = 0; j < hostRes->nres; j++) {
                    struct resources *res = &hostRes->res[j];
                    if (res->cType != FLOAT_VAL) continue;

                    if (strcmp(res->resName, "slots") == 0) {
                        host_slots = res->val.fval;
                    } else if (strcmp(res->resName, "mem") == 0) {
                        host_memory = res->val.fval;
                    }
                }
            }

            host_table_add_or_update(hostname, host_slots, host_memory);
        }

        free_host_ids(hostname, clustername);
    }

    /* Compute cluster totals once (first sort when cluster is fully idle) */
    if (!total_cluster_initialized) {
        total_cluster_slots = 0.0;
        total_cluster_memory = 0.0;
        for (int i = 0; i < host_table_count; i++) {
            if (host_table[i].active) {
                total_cluster_slots += host_table[i].remaining_slots;
                total_cluster_memory += host_table[i].remaining_memory;
            }
        }
        if (total_cluster_slots < 1.0) total_cluster_slots = 1.0;
        if (total_cluster_memory < 1.0) total_cluster_memory = 1.0;
        total_cluster_initialized = 1;
        ls_syslog(LOG_INFO, "AJS_SORT: cycle=%d hosts=%d total_slots=%.0f total_memory=%.0f (initialized)",
                  current_cycle_number, host_table_count, total_cluster_slots, total_cluster_memory);
    }

    /* Compute global ratios from host table */
    host_table_compute_ratios();
    host_table_initialized = 1;

    return 0;
}

static void ajs_free(ajs_data *data)
{
    destroy_ajs_data((void *)data);
}

static int ajs_checkAlloc(ajs_data *data, void *job, void *alloc, void *allocLimitList)
{
    if (data == NULL || !data->useAJS) {
        return (0);
    }


    return (0);
}

static int ajs_notifyAlloc(ajs_data *data, void *job, void *alloc, void *allocLimitList, int flag)
{
    if (data == NULL || !data->useAJS) {
        return (0);
    }

    int event = extsched_determineEvent((INT_Job *)job, (INT_Alloc *)alloc,
                                         (INT_AllocLimitList *)allocLimitList, flag);

    /* Get per-host allocation details */
    extsched_Alloc *allocInfo = NULL;
    int updated = 0;

    if (extsched_getAlloc((INT_Alloc *)alloc, &allocInfo) == 0 && allocInfo != NULL) {
        for (int i = 0; i < allocInfo->nInst; i++) {
            extsched_AllocInst *inst = allocInfo->inst[i];
            if (!inst || !inst->host || !inst->rsrc) continue;

            int idx = host_table_find(inst->host);
            if (idx < 0) continue;

            if (event & SCH_FM_EVE_ALLOCATE) {
                /* Job dispatched - subtract allocated resources */
                if (strcmp(inst->rsrc, "slots") == 0) {
                    host_table[idx].remaining_slots -= (double)inst->amount;
                    if (host_table[idx].remaining_slots < 0.0)
                        host_table[idx].remaining_slots = 0.0;
                } else if (strcmp(inst->rsrc, "mem") == 0) {
                    host_table[idx].remaining_memory -= (double)inst->amount;
                    if (host_table[idx].remaining_memory < 0.0)
                        host_table[idx].remaining_memory = 0.0;
                }
                updated = 1;
            } else if (event & SCH_FM_EVE_DEALLOCATE) {
                /* Job finished - add resources back */
                if (strcmp(inst->rsrc, "slots") == 0) {
                    host_table[idx].remaining_slots += (double)inst->amount;
                } else if (strcmp(inst->rsrc, "mem") == 0) {
                    host_table[idx].remaining_memory += (double)inst->amount;
                }
                updated = 1;
            }
        }
        extsched_freeAlloc(&allocInfo);
    }

    if (updated) {
        host_table_compute_ratios();
    } else {
        /* Fallback: simple update using job-level data */
        if (event & SCH_FM_EVE_ALLOCATE) {
            remaining_cores_ratio -= data->job_slots / total_cluster_slots;
            remaining_memory_ratio -= data->job_memory / total_cluster_memory;
        } else if (event & SCH_FM_EVE_DEALLOCATE) {
            remaining_cores_ratio += data->job_slots / total_cluster_slots;
            remaining_memory_ratio += data->job_memory / total_cluster_memory;
        }
    }

    if (event & SCH_FM_EVE_ALLOCATE) {
        notify_alloc_count++;
    } else if (event & SCH_FM_EVE_DEALLOCATE) {
        notify_dealloc_count++;
    }
    ls_syslog(LOG_INFO, "AJS_NOTIFY: cycle=%d alloc=%d dealloc=%d event=0x%x slots=%.0f mem=%.0f cores_ratio=%.4f mem_ratio=%.4f",
              current_cycle_number, notify_alloc_count, notify_dealloc_count, event, data->job_slots, data->job_memory,
              remaining_cores_ratio, remaining_memory_ratio);

    return (0);
}

/***************************************************************************
 * DATA MANAGEMENT FUNCTIONS
 ***************************************************************************/

/**
 * create_ajs_data - Allocate and initialize ML order data structure
 */
static ajs_data *create_ajs_data(void)
{
    ajs_data *data = (ajs_data *)calloc(1, sizeof(ajs_data));
    if (data == NULL) {
        return NULL;
    }
    data->useAJS = 0;
    data->job_slots = 0.0;
    data->job_memory = 0.0;
    return data;
}

static void destroy_ajs_data(void *p)
{
    ajs_data *data = (ajs_data *)p;
    if (data) {
        FREEUP(data);
    }
}

/**
 * extract_job_slots_from_resreq - Extract CPU slots from resource requirement
 */
static int extract_job_slots_from_resreq(void *resreq)
{
    if (resreq == NULL) {
        return 0;
    }

    /* Direct extraction from known structure layout */
    int *int_ptr = (int *)resreq;
    return int_ptr[2];  /* Slots at position 2 */
}

/**
 * extract_job_memory_from_resreq - Extract memory from resource requirement
 * Uses LSF API to properly extract memory requirements
 */
static int extract_job_memory_from_resreq(void *resreq)
{
    extsched_rsrcReqInfo *rsrcReqInfo = NULL;
    int memoryMB = 0;

    if (!resreq) {
        return 0;
    }

    /* Get resource requirement info using proper API */
    rsrcReqInfo = extsched_getRsrcReqInfo((INT_RsrcReq *)resreq);
    if (!rsrcReqInfo) {
        return 0;
    }

    /* Iterate through resource consumption requirements */
    for (int i = 0; i < rsrcReqInfo->nRsrcConsump; i++) {
        if (rsrcReqInfo->rsrcConsump[i].rsrcName) {
            /* Check for memory resource - 'mem' is used in job requirements */
            if (strcmp(rsrcReqInfo->rsrcConsump[i].rsrcName, "mem") == 0) {
                memoryMB = (int)rsrcReqInfo->rsrcConsump[i].amount;
                break;
            }
        }
    }

    /* Clean up allocated memory */
    extsched_freeRsrcReqInfo(&rsrcReqInfo);

    return memoryMB;
}

/***************************************************************************
 * JOB ORDERING IMPLEMENTATION
 ***************************************************************************/

/**
 * ajs_job_ordering - Select next job to schedule using ML approach
 *
 * Uses a macro action model with two outputs:
 *   - bucket_index: which bucket to dispatch from
 *   - macro_action: 0=20%, 1=50%, 2=100% of jobs in that bucket
 *
 * State machine:
 *   1. If we have remaining dispatch attempts for current bucket:
 *      - Check if target bucket still exists (by pointer comparison)
 *      - If yes: continue dispatching, decrement counter
 *      - If no: bucket gone (empty/failed), need new inference
 *   2. If no remaining attempts: run inference for new bucket selection
 *
 * @queueName: Name of the queue being processed
 * @jobList: List of pending jobs to order
 * @return: Selected job block or NULL
 */
static INT_JobBlock *ajs_job_ordering(char *queueName, INT_JobList *jobList)
{
    INT_JobBlock *job = NULL;
    INT_JobBlock *selected_job = NULL;
    int vector_length = 0;
    double *ml_input_vector = NULL;

    if (jobList == NULL) {
        return NULL;
    }

    /* Check if list is empty */
    if (extsched_order_isJobListEmpty(jobList)) {
        return NULL;
    }

    /* Increment scheduling attempts counter */
    scheduling_calls_this_cycle++;

    /* On first call of cycle, check for long wait buckets */
    if (scheduling_calls_this_cycle == 1) {
        INT_JobBlock *logJob = extsched_order_getFirstJobOfList(jobList);

        /* Check for long wait buckets */
        logJob = extsched_order_getFirstJobOfList(jobList);
        int has_long_wait = 0;
        int bucket_idx = 0;
        while (logJob != NULL) {
            double wait = extract_job_waiting_time(logJob);

            /* Track if any bucket has waiting time > 150 seconds */
            if (wait > 150.0) {
                has_long_wait = 1;
            }

            bucket_idx++;
            logJob = extsched_order_getNextJobOfList(logJob, jobList);
        }

        /* If any bucket has long wait time, log all buckets with detailed info */
        if (has_long_wait) {
            ls_syslog(LOG_INFO, "AJS_LONG_WAIT: cycle=%d detected bucket(s) with waiting_time > 150s, dumping all %d buckets:",
                      current_cycle_number, bucket_idx);
            logJob = extsched_order_getFirstJobOfList(jobList);
            int idx = 0;
            while (logJob != NULL && idx < bucket_idx) {
                int bucket_size = extsched_order_getNumPendJobsInBucket(logJob);
                int cores = extract_job_slots_from_jobblock(logJob);
                double mem = extract_job_memory_from_jobblock(logJob);
                double wait = extract_job_waiting_time(logJob);
                ls_syslog(LOG_INFO, "AJS_LONG_WAIT: cycle=%d bucket[%d] slots=%d mem=%.0f job_count=%d waiting_time=%.0f%s",
                          current_cycle_number, idx, cores, mem, bucket_size, wait,
                          (wait > 150.0) ? " [LONG_WAIT]" : "");
                idx++;
                logJob = extsched_order_getNextJobOfList(logJob, jobList);
            }
        }
    }

    /* === STATE MACHINE: Check if we're in the middle of a macro action === */
    if (remaining_dispatch_attempts > 0) {
        /* Search for target bucket by (slots, memory) */
        job = extsched_order_getFirstJobOfList(jobList);
        while (job != NULL) {
            double job_slots = (double)extract_job_slots_from_jobblock(job);
            double job_memory = extract_job_memory_from_jobblock(job);

            if (job_slots == target_bucket_slots && job_memory == target_bucket_memory) {
                /* Found matching bucket - continue dispatching */
                remaining_dispatch_attempts--;
                selected_job = job;

                order_dispatch_count++;
                ls_syslog(LOG_INFO, "AJS_DISPATCH: cycle=%d dispatch=%d (macro_continue) slots=%.0f mem=%.0f remaining=%d",
                          current_cycle_number, order_dispatch_count, job_slots, job_memory, remaining_dispatch_attempts);

                return selected_job;
            }
            job = extsched_order_getNextJobOfList(job, jobList);
        }

        /* Target bucket gone (empty or dispatch failed) - need new inference */
        remaining_dispatch_attempts = 0;
    }

    /* === NEW INFERENCE NEEDED === */

    /* Count buckets and collect their sizes */
    int num_buckets = 0;
    int bucket_sizes[MAX_BUCKETS] = {0};

    job = extsched_order_getFirstJobOfList(jobList);
    while (job != NULL && num_buckets < MAX_BUCKETS) {
        bucket_sizes[num_buckets] = extsched_order_getNumPendJobsInBucket(job);
        num_buckets++;
        job = extsched_order_getNextJobOfList(job, jobList);
    }

    /* Check if there are more buckets than MAX_BUCKETS */
    if (job != NULL) {
        int skipped_buckets = 0;
        while (job != NULL) {
            skipped_buckets++;
            job = extsched_order_getNextJobOfList(job, jobList);
        }
        ls_syslog(LOG_ERR, "ajs_job_ordering: total_buckets=%d exceeds MAX_BUCKETS(%d), %d buckets will be ignored",
                  num_buckets + skipped_buckets, MAX_BUCKETS, skipped_buckets);
    }

    if (num_buckets == 0) {
        return NULL;
    }

    /* Build ML input vector */
    ml_input_vector = build_ml_input_vector(jobList, &vector_length);

    /* Run ML inference */
    int selected_bucket_index = 0;
    int macro_action = 0;

    int rc = ml_inference(ml_input_vector, vector_length,
                          num_buckets, &selected_bucket_index, &macro_action);
    if (rc != 0) {
        /* Fallback to first bucket with 100% on ML failure */
        selected_bucket_index = 0;
        macro_action = 2;
    }

    if (ml_input_vector != NULL) {
        free(ml_input_vector);
    }

    /* Find selected bucket */
    selected_job = find_bucket_by_index(jobList, selected_bucket_index);
    if (selected_job == NULL) {
        /* Fallback to first bucket */
        selected_job = extsched_order_getFirstJobOfList(jobList);
        selected_bucket_index = 0;
    }

    /* Calculate remaining dispatch attempts based on macro action */
    int bucket_size = bucket_sizes[selected_bucket_index];
    int target_jobs = 0;

    switch (macro_action) {
        case 0:  /* 20% */
            target_jobs = (bucket_size * 20 + 99) / 100;  /* Ceiling */
            if (target_jobs < 1) target_jobs = 1;
            break;
        case 1:  /* 50% */
            target_jobs = (bucket_size * 50 + 99) / 100;  /* Ceiling */
            if (target_jobs < 1) target_jobs = 1;
            break;
        case 2:  /* 100% */
        default:
            target_jobs = bucket_size;
            break;
    }

    /* Ablation: force single-job dispatch per inference (override macro) */
    // target_jobs =bucket_size;

    /* Set state for subsequent calls (minus 1 for current dispatch) */
    target_bucket_slots = (double)extract_job_slots_from_jobblock(selected_job);
    target_bucket_memory = extract_job_memory_from_jobblock(selected_job);
    remaining_dispatch_attempts = target_jobs - 1;

    order_dispatch_count++;
    ls_syslog(LOG_INFO, "AJS_DISPATCH: cycle=%d dispatch=%d (new_inference) bucket=%d macro=%d target_jobs=%d slots=%.0f mem=%.0f",
              current_cycle_number, order_dispatch_count, selected_bucket_index, macro_action, target_jobs,
              target_bucket_slots, target_bucket_memory);

    return selected_job;
}

/***************************************************************************
 * RESOURCE EXTRACTION FUNCTIONS
 ***************************************************************************/

/**
 * extract_job_slots_from_jobblock - Extract CPU slots from job block
 */
static int extract_job_slots_from_jobblock(INT_JobBlock *jobBlock)
{
    int slots = 0;

    if (jobBlock == NULL) {
        return 0;
    }

    /* Use LSF API to get asked slots */
    slots = extsched_job_getaskedslot(jobBlock);
    if (slots <= 0) {
        slots = 1;  /* Default 1 slot if not specified */
    }

    return slots;
}

/*
 * Extract job memory from a JobBlock
 */
static double extract_job_memory_from_jobblock(INT_JobBlock *jobBlock)
{
    INT_Job *job = NULL;
    INT_RsrcReq *resreq = NULL;
    int memoryMB = 0;

    if (jobBlock == NULL) {
        return 0.0;
    }

    /* Get INT_Job from JobBlock */
    job = extsched_get_INT_Job(jobBlock);
    if (job == NULL) {
        return 0.0;
    }

    /* Get resource requirements from job */
    resreq = extsched_getRsrcReqForJob(job);
    if (resreq != NULL) {
        /* Use existing function to extract memory */
        memoryMB = extract_job_memory_from_resreq(resreq);
    }

    if (memoryMB <= 0) {
        memoryMB = 256;  /* Default 256MB if not specified in job requirements */
    }

    return (double)memoryMB;
}

/**
 * extract_job_waiting_time - Extract waiting time of first job in bucket
 * Waiting time = current_time - submit_time (in seconds)
 *
 * @jobBlock: Job block representing the bucket
 * @return: Waiting time in seconds, or 0.0 on failure
 */
static double extract_job_waiting_time(INT_JobBlock *jobBlock)
{
    time_t submit_time = 0;
    time_t current_time = 0;
    double waiting_time = 0.0;

    if (jobBlock == NULL) {
        return 0.0;
    }

    /* Get submit time from job block */
    submit_time = extsched_order_getJobSubmitTime(jobBlock);
    if (submit_time <= 0) {
        return 0.0;
    }

    /* Calculate waiting time */
    current_time = time(NULL);
    waiting_time = difftime(current_time, submit_time);

    /* Ensure non-negative */
    if (waiting_time < 0.0) {
        waiting_time = 0.0;
    }

    return waiting_time;
}

/***************************************************************************
 * ML VECTOR GENERATION
 ***************************************************************************/

/**
 * build_ml_input_vector - Generate normalized input vector for ML model
 *
 * Vector Structure (total size: MAX_BUCKETS * 4 + 2 + 1):
 * =======================================================
 *
 * PART 1: Bucket Features (MAX_BUCKETS * 4 values)
 * ------------------------------------------------
 * For each bucket i (0 to MAX_BUCKETS-1), 4 features:
 *   [i*4 + 0] = cores_norm   : first_job_cores / JOB_MAX_SLOTS
 *   [i*4 + 1] = memory_norm  : first_job_memory / JOB_MAX_MEM
 *   [i*4 + 2] = count_norm   : bucket_job_count / 100.0
 *   [i*4 + 3] = wait_norm    : waiting_time / 300.0
 *
 * Note: If fewer than MAX_BUCKETS exist, remaining slots are zero-padded.
 *       waiting_time is the waiting time of the first job in the bucket (in seconds)
 *
 * PART 2: Cluster Global Features (2 values)
 * ------------------------------------------
 * Matches bucket_adaptive_env.rs: remaining resource ratios
 *   [base + 0] = remaining_cores_ratio : starts at 1.0, decremented per attempt, clipped to 0
 *   [base + 1] = remaining_memory_ratio: starts at 1.0, decremented per attempt, clipped to 0
 *
 * PART 3: num_valid (1 value)
 * ---------------------------
 *   [base + 2] = num_valid : number of valid buckets (used by model for masking)
 *
 * @jobList: List of pending job buckets
 * @vector_length: Output parameter for vector size
 * @return: Allocated vector (caller must free) or NULL on failure
 */
static double *build_ml_input_vector(INT_JobList *jobList, int *vector_length)
{
    double *vector = NULL;
    INT_JobBlock *jobBlock = NULL;
    int bucket_count = 0;
    int vector_size;
    int idx = 0;
    int global_base;

    if (jobList == NULL || vector_length == NULL) {
        return NULL;
    }

    /* Vector size: bucket features (MAX_BUCKETS * 4) + global features (2) + num_valid (1) */
    vector_size = MAX_BUCKETS * 4 + 2 + 1;
    *vector_length = vector_size;
    global_base = MAX_BUCKETS * 4;  /* Start index for global features */

    /* Allocate vector and initialize with zeros (handles padding automatically) */
    vector = (double *)calloc(vector_size, sizeof(double));
    if (vector == NULL) {
        ls_syslog(LOG_ERR, "build_ml_input_vector: Failed to allocate memory for ML vector");
        return NULL;
    }

    /* === PART 1: Populate bucket features === */
    jobBlock = extsched_order_getFirstJobOfList(jobList);
    while (jobBlock != NULL && bucket_count < MAX_BUCKETS) {
        double job_slots, job_memory, waiting_time;
        int bucket_size;

        /* Extract job resource requirements */
        job_slots = (double)extract_job_slots_from_jobblock(jobBlock);
        job_memory = extract_job_memory_from_jobblock(jobBlock);
        bucket_size = extsched_order_getNumPendJobsInBucket(jobBlock);
        waiting_time = extract_job_waiting_time(jobBlock);

        /* Populate normalized features for this bucket */
        idx = bucket_count * 4;
        vector[idx + 0] = job_slots / JOB_MAX_SLOTS;              /* Normalized cores */
        vector[idx + 1] = job_memory / JOB_MAX_MEM;               /* Normalized memory */
        vector[idx + 2] = (double)bucket_size / 100.0;            /* Normalized job count */
        vector[idx + 3] = waiting_time / 300.0;                   /* Normalized waiting time */

        bucket_count++;
        jobBlock = extsched_order_getNextJobOfList(jobBlock, jobList);
    }
    /* Remaining bucket slots are already zero-padded by calloc */

    /* === PART 2: Populate cluster global features (2 values) === */
    /* Remaining resource ratios: start at 1.0, decremented per attempt, clipped to 0 */
    vector[global_base + 0] = remaining_cores_ratio > 0.0 ? remaining_cores_ratio : 0.0;
    vector[global_base + 1] = remaining_memory_ratio > 0.0 ? remaining_memory_ratio : 0.0;

    /* === PART 3: num_valid === */
    vector[global_base + 2] = (double)bucket_count;                                 /* Number of valid buckets */

    return vector;
}

/***************************************************************************
 * CURL FUNCTIONS
 ***************************************************************************/

/**
 * WriteMemoryCallback - CURL callback for receiving data
 */
static size_t WriteMemoryCallback(void *contents, size_t size, size_t nmemb, void *userp)
{
    size_t realsize = size * nmemb;
    struct MemoryStruct *mem = (struct MemoryStruct *)userp;

    char *ptr = realloc(mem->memory, mem->size + realsize + 1);
    if (ptr == NULL) {
        ls_syslog(LOG_ERR, "WriteMemoryCallback: realloc failed");
        return 0;
    }

    mem->memory = ptr;
    memcpy(&(mem->memory[mem->size]), contents, realsize);
    mem->size += realsize;
    mem->memory[mem->size] = 0;

    return realsize;
}

/**
 * callMLServer - Send JSON request to ML server and get response
 */
static char *callMLServer(const char *jsonData)
{
    static char fname[] = "callMLServer";
    CURL *curl;
    CURLcode res;
    struct MemoryStruct chunk;
    struct curl_slist *headers = NULL;

    chunk.memory = malloc(1);
    chunk.size = 0;

    curl = curl_easy_init();
    if (!curl) {
        ls_syslog(LOG_ERR, "%s: Failed to initialize CURL", fname);
        free(chunk.memory);
        return NULL;
    }

    headers = curl_slist_append(headers, "Content-Type: application/json");

    curl_easy_setopt(curl, CURLOPT_URL, ML_SERVER_URL);
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, jsonData);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteMemoryCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, (void *)&chunk);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, ML_SERVER_TIMEOUT);

    res = curl_easy_perform(curl);

    if (res != CURLE_OK) {
        FREEUP(chunk.memory);
        chunk.memory = NULL;
    }

    curl_easy_cleanup(curl);
    curl_slist_free_all(headers);

    return chunk.memory;
}

/**
 * parseMLResponse - Parse JSON response from ML server
 * Expected format: {"bucket_index": N, "macro_action": M}
 */
static int parseMLResponse(const char *jsonResponse, int *bucket_index, int *macro_action)
{
    char *bucket_start = strstr(jsonResponse, "\"bucket_index\":");
    char *macro_start = strstr(jsonResponse, "\"macro_action\":");

    if (!bucket_start || !macro_start) {
        return -1;
    }

    bucket_start += 15;  /* Skip "bucket_index": */
    macro_start += 15;   /* Skip "macro_action": */

    *bucket_index = (int)strtol(bucket_start, NULL, 10);
    *macro_action = (int)strtol(macro_start, NULL, 10);

    return 0;
}

/***************************************************************************
 * ML INFERENCE
 ***************************************************************************/

/**
 * ml_inference - Call ML server for bucket selection
 *
 * @input_vector: ML input vector
 * @vector_length: Length of input vector
 * @num_buckets: Number of active buckets
 * @out_bucket_index: Output - selected bucket index
 * @out_macro_action: Output - macro action (0=20%, 1=50%, 2=100%)
 * @return: 0 on success, -1 on failure
 */
static int ml_inference(double *input_vector, int vector_length,
                        int num_buckets, int *out_bucket_index, int *out_macro_action)
{
    char jsonBuffer[JSON_BUFFER_SIZE];
    int pos;
    char *jsonResponse;

    /* Build JSON request */
    pos = snprintf(jsonBuffer, sizeof(jsonBuffer),
                   "{\"input_vector\": [");

    for (int i = 0; i < vector_length; i++) {
        if (i > 0) {
            pos += snprintf(jsonBuffer + pos, sizeof(jsonBuffer) - pos, ", ");
        }
        pos += snprintf(jsonBuffer + pos, sizeof(jsonBuffer) - pos, "%.6f", input_vector[i]);
    }

    pos += snprintf(jsonBuffer + pos, sizeof(jsonBuffer) - pos,
                    "], \"num_buckets\": %d}", num_buckets);

    /* Call ML server */
    jsonResponse = callMLServer(jsonBuffer);
    if (!jsonResponse) {
        return -1;
    }

    /* Parse response */
    if (parseMLResponse(jsonResponse, out_bucket_index, out_macro_action) != 0) {
        FREEUP(jsonResponse);
        return -1;
    }

    FREEUP(jsonResponse);
    return 0;
}

/**
 * find_bucket_by_index - Get bucket pointer by index
 *
 * @jobList: List of job buckets
 * @bucket_index: Index of bucket to find (0-based)
 * @return: Pointer to bucket or NULL if not found
 */
static INT_JobBlock *find_bucket_by_index(INT_JobList *jobList, int bucket_index)
{
    INT_JobBlock *job = NULL;
    int current_index = 0;

    if (jobList == NULL || bucket_index < 0) {
        return NULL;
    }

    job = extsched_order_getFirstJobOfList(jobList);
    while (job != NULL && current_index < bucket_index) {
        current_index++;
        job = extsched_order_getNextJobOfList(job, jobList);
    }

    return job;
}

/***************************************************************************
 * UTILITY FUNCTIONS
 ***************************************************************************/

static int is_localhost(const char *hostname)
{
    if (!hostname) {
        return 0;
    }

    if (strcasecmp(hostname, LOCAL_HOST) == 0) {
        return 1;
    }

    if (strncasecmp(hostname, LOCAL_HOST, strlen(LOCAL_HOST)) == 0) {
        const char *suffix = hostname + strlen(LOCAL_HOST);
        if (*suffix == '.' || *suffix == '\0') {
            return 1;
        }
    }

    return 0;
}

static void free_host_ids(char *hostname, char *clustername)
{
    if (hostname) {
        free(hostname);
    }
    if (clustername) {
        free(clustername);
    }
}

/* End of ajs.c */