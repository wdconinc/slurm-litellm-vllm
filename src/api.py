import os
import subprocess
from typing import Optional
from bin.start import submit_job, monitor_job

class SlurmLLMFleet:
    """
    Programmatic API for researchers to orchestrate multi-node LLM backends 
    directly from Jupyter Notebooks or Python scripts.
    
    Example:
        fleet = SlurmLLMFleet(model="smollm-cpu")
        fleet.start()
        
        client = fleet.get_openai_client()
        print(client.chat.completions.create(
            model="my-local-model",
            messages=[{"role": "user", "content": "Hello!"}]
        ))
        
        fleet.stop()
    """
    def __init__(self, model: str = "mistral"):
        self.model = model
        self.job_id: Optional[str] = None
        self.endpoint_file: Optional[str] = None
        self.api_base: Optional[str] = None
        self.api_key: Optional[str] = None
        
    def start(self, wait: bool = True):
        """Submits the Slurm job and optionally blocks until the Ray cluster and vLLM are online."""
        print(f"🚀 Starting SlurmLLM backend for {self.model}...")
        self.job_id = submit_job(self.model)
        
        if wait:
            self.endpoint_file = monitor_job(self.job_id)
            self._parse_endpoint()
            print(f"✅ Backend online at {self.api_base}")
            
    def _parse_endpoint(self):
        """Reads the dynamic cluster IP from the generated environment file."""
        if not self.endpoint_file or not os.path.exists(self.endpoint_file):
            return
            
        with open(self.endpoint_file, "r") as f:
            for line in f:
                if "OPENAI_API_BASE" in line:
                    self.api_base = line.split("=", 1)[1].strip('"')
                elif "OPENAI_API_KEY" in line:
                    self.api_key = line.split("=", 1)[1].strip('"')
                    
    def get_openai_client(self):
        """Returns a configured OpenAI Python client connected to the Slurm proxy."""
        if not self.api_base:
            raise ValueError("Backend not ready. Call start(wait=True) first.")
            
        try:
            from openai import OpenAI
            return OpenAI(base_url=self.api_base, api_key=self.api_key)
        except ImportError:
            raise ImportError("The 'openai' package is required. Run: pip install openai")
            
    def stop(self):
        """Gracefully terminates the Slurm job and shuts down the Ray cluster."""
        if self.job_id:
            print(f"🛑 Stopping Slurm job {self.job_id}...")
            subprocess.run(["scancel", self.job_id])
            self.job_id = None
            self.api_base = None
            if self.endpoint_file and os.path.exists(self.endpoint_file):
                os.remove(self.endpoint_file)
