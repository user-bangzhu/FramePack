import threading
import time
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, Optional
import queue as queue_module  # Renamed to avoid conflicts

from diffusers_helper.thread_utils import AsyncStream


# Simple LIFO queue implementation to avoid dependency on queue.LifoQueue
class SimpleLifoQueue:
    def __init__(self):
        self._queue = []
        self._mutex = threading.Lock()
        self._not_empty = threading.Condition(self._mutex)
    
    def put(self, item):
        with self._mutex:
            self._queue.append(item)
            self._not_empty.notify()
    
    def get(self):
        with self._not_empty:
            while not self._queue:
                self._not_empty.wait()
            return self._queue.pop()
    
    def task_done(self):
        pass  # For compatibility with queue.Queue


class JobStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Job:
    id: str
    params: Dict[str, Any]
    status: JobStatus
    created_at: float
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Optional[str] = None  # Path to output video
    error: Optional[str] = None
    progress_data: Dict[str, Any] = None  # Store latest progress data
    stream: Optional[AsyncStream] = None  # Stream for this job


class VideoJobQueue:
    def __init__(self):
        self.queue = queue_module.Queue()  # Using standard Queue instead of LifoQueue
        self.jobs = {}
        self.current_job = None
        self.lock = threading.Lock()
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        self.worker_function = None  # Will be set from outside
    
    def set_worker_function(self, worker_function):
        """Set the worker function to use for processing jobs"""
        self.worker_function = worker_function
    
    def add_job(self, params):
        """Add a job to the queue and return its ID"""
        job_id = str(uuid.uuid4())
        job = Job(
            id=job_id,
            params=params,
            status=JobStatus.PENDING,
            created_at=time.time(),
            progress_data={},
            stream=AsyncStream()
        )
        
        with self.lock:
            self.jobs[job_id] = job
            self.queue.put(job_id)
        
        return job_id
    
    def get_job(self, job_id):
        """Get job by ID"""
        with self.lock:
            return self.jobs.get(job_id)
    
    def get_all_jobs(self):
        """Get all jobs"""
        with self.lock:
            return list(self.jobs.values())
    
    def cancel_job(self, job_id):
        """Cancel a pending job"""
        with self.lock:
            job = self.jobs.get(job_id)
            if job and job.status == JobStatus.PENDING:
                job.status = JobStatus.CANCELLED
                return True
            elif job and job.status == JobStatus.RUNNING and job.stream:
                job.stream.input_queue.push('end')
                return True
            return False
    
    def get_queue_position(self, job_id):
        """Get position in queue (0 = currently running)"""
        with self.lock:
            job = self.jobs.get(job_id)
            if not job:
                return None
                
            if job.status == JobStatus.RUNNING:
                return 0
                
            if job.status != JobStatus.PENDING:
                return None
                
            # Count pending jobs ahead in queue
            position = 1  # Start at 1 because 0 means running
            for j in self.jobs.values():
                if (j.status == JobStatus.PENDING and 
                    j.created_at < job.created_at):
                    position += 1
            return position
    
    def update_job_progress(self, job_id, progress_data):
        """Update job progress data"""
        with self.lock:
            job = self.jobs.get(job_id)
            if job:
                job.progress_data = progress_data
    
    def _worker_loop(self):
        """Worker thread that processes jobs from the queue"""
        while True:
            try:
                job_id = self.queue.get()
                
                with self.lock:
                    job = self.jobs.get(job_id)
                    if not job or job.status == JobStatus.CANCELLED:
                        self.queue.task_done()
                        continue
                    
                    job.status = JobStatus.RUNNING
                    job.started_at = time.time()
                    self.current_job = job
                
                try:
                    if self.worker_function is None:
                        raise ValueError("Worker function not set. Call set_worker_function() first.")
                    
                    # Start the worker function with the job parameters
                    from diffusers_helper.thread_utils import async_run
                    async_run(
                        self.worker_function,
                        **job.params,
                        job_stream=job.stream
                    )
                    
                    # Process the results from the stream
                    output_filename = None
                    while True:
                        flag, data = job.stream.output_queue.next()
                        
                        if flag == 'file':
                            output_filename = data
                            with self.lock:
                                job.result = output_filename
                        
                        elif flag == 'progress':
                            preview, desc, html = data
                            with self.lock:
                                job.progress_data = {
                                    'preview': preview,
                                    'desc': desc,
                                    'html': html
                                }
                        
                        elif flag == 'end':
                            break
                    
                    with self.lock:
                        job.status = JobStatus.COMPLETED
                        job.completed_at = time.time()
                
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    with self.lock:
                        job.status = JobStatus.FAILED
                        job.error = str(e)
                        job.completed_at = time.time()
                
                finally:
                    with self.lock:
                        self.current_job = None
                    self.queue.task_done()
            
            except Exception as e:
                print(f"Error in worker loop: {e}")
                # Continue processing other jobs
