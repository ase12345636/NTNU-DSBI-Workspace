import numpy as np
import pandas as pd
import scanpy as sc
import os
import subprocess
import threading
import time as time_module
from sklearn.metrics import normalized_mutual_info_score


class MemoryTracker:
    """
    Track RAM and VRAM usage for the current process and its children.
    Uses background thread sampling to capture true peak memory usage.
    
    Usage:
        tracker = MemoryTracker()
        tracker.start()
        # ... your code ...
        metrics = tracker.stop()
        # metrics = {'peak_ram_mb': ..., 'peak_vram_mb': ..., 'used_ram_mb': ..., 'used_vram_mb': ...}
    """
    
    def __init__(self, track_gpu=True, track_children=True, sample_interval=1.0):
        """
        Parameters
        ----------
        track_gpu : bool
            Whether to track GPU memory using nvidia-smi.
        track_children : bool
            Whether to track memory of child processes (default True).
        sample_interval : float
            Interval in seconds for sampling memory (default: 1.0).
        """
        self.track_gpu = track_gpu
        self.track_children = track_children
        self.sample_interval = sample_interval
        self.root_pid = os.getpid()
        self._started = False
        self._monitoring_thread = None
        self._stop_monitoring = False
        self._ram_samples = []
        self._vram_samples = []
        self._cached_pids = None
        self._cache_time = 0
        self._cache_duration = 10.0  # Cache process tree for 10 seconds
    
    def _get_process_tree_pids(self):
        """Get all PIDs in process tree with caching."""
        current_time = time_module.time()
        
        # Use cached result if still valid
        if (self._cached_pids is not None and 
            current_time - self._cache_time < self._cache_duration):
            return self._cached_pids
        
        pids = {self.root_pid}
        
        if self.track_children:
            try:
                children_path = f"/proc/{self.root_pid}/task/{self.root_pid}/children"
                if os.path.exists(children_path):
                    with open(children_path, 'r') as f:
                        child_pids = f.read().split()
                    for child_pid_str in child_pids:
                        try:
                            child_pid = int(child_pid_str)
                            pids.add(child_pid)
                            # Recursively get children (1 level deep)
                            child_children_path = f"/proc/{child_pid}/task/{child_pid}/children"
                            if os.path.exists(child_children_path):
                                with open(child_children_path, 'r') as f:
                                    grandchild_pids = f.read().split()
                                for gchild in grandchild_pids:
                                    try:
                                        pids.add(int(gchild))
                                    except (ValueError, OSError):
                                        pass
                        except (ValueError, OSError):
                            continue
            except (OSError, IOError):
                pass
        
        self._cached_pids = pids
        self._cache_time = current_time
        return pids
    
    def _get_ram_mb(self, pids):
        """Get total RSS in MB for all PIDs using /proc/pid/status."""
        total_kb = 0
        for pid in pids:
            status_path = f"/proc/{pid}/status"
            try:
                with open(status_path, "r") as f:
                    for line in f:
                        if line.startswith("VmRSS:"):
                            total_kb += int(line.split()[1])
                            break
            except (OSError, ValueError, IOError):
                continue
        return total_kb / 1024.0
    
    def _get_vram_mb(self, pids):
        """Get total VRAM in MB for all PIDs using nvidia-smi."""
        if not self.track_gpu:
            return None
        
        cmd = ["nvidia-smi", "--query-compute-apps=pid,used_memory", 
               "--format=csv,noheader,nounits"]
        try:
            output = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL)
        except (OSError, subprocess.CalledProcessError, FileNotFoundError):
            return None
        
        total_mb = 0.0
        for line in output.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 2:
                continue
            try:
                pid = int(parts[0])
                used_mb = float(parts[1])
                if pid in pids:
                    total_mb += used_mb
            except ValueError:
                continue
        return total_mb
    
    def _monitor_loop(self):
        """Background monitoring loop."""
        while not self._stop_monitoring:
            try:
                pids = self._get_process_tree_pids()
                ram_mb = self._get_ram_mb(pids)
                vram_mb = self._get_vram_mb(pids)
                
                self._ram_samples.append(ram_mb)
                if vram_mb is not None:
                    self._vram_samples.append(vram_mb)
            except Exception:
                pass
            
            time_module.sleep(self.sample_interval)
    
    def start(self):
        """Start tracking memory usage."""
        if self._started:
            return
        
        self._started = True
        self._ram_samples = []
        self._vram_samples = []
        self._cached_pids = None
        
        # Record baseline
        pids = self._get_process_tree_pids()
        self._ram_baseline = self._get_ram_mb(pids)
        self._vram_baseline = self._get_vram_mb(pids) or 0.0
        
        # Always start background monitoring thread
        self._stop_monitoring = False
        self._monitoring_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitoring_thread.start()
    
    def stop(self):
        """
        Stop tracking and return memory metrics.
        
        Returns
        -------
        dict
            Dictionary containing:
            - 'peak_ram_mb': Peak RAM usage during execution (MB)
            - 'used_ram_mb': Net RAM increase (current - baseline, MB)
            - 'peak_vram_mb': Peak VRAM usage during execution (MB)
            - 'used_vram_mb': Net VRAM increase (current - baseline, MB)
        """
        if not self._started:
            raise RuntimeError("MemoryTracker.start() must be called before stop()")
        
        # Stop monitoring thread
        self._stop_monitoring = True
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=2.0)
        
        # Get final sample
        pids = self._get_process_tree_pids()
        current_ram = self._get_ram_mb(pids)
        current_vram = self._get_vram_mb(pids) or 0.0
        
        # Calculate RAM metrics
        if self._ram_samples:
            peak_ram = max(self._ram_samples)
            avg_ram = sum(self._ram_samples) / len(self._ram_samples)
        else:
            peak_ram = current_ram
            avg_ram = current_ram
        
        used_ram = max(0, current_ram - self._ram_baseline)
        
        # Calculate VRAM metrics
        if self._vram_samples:
            peak_vram = max(self._vram_samples)
            avg_vram = sum(self._vram_samples) / len(self._vram_samples)
        else:
            peak_vram = current_vram
            avg_vram = current_vram
        
        used_vram = max(0, current_vram - self._vram_baseline)
        
        self._started = False
        
        return {
            'peak_ram_mb': round(peak_ram, 2),
            'used_ram_mb': round(used_ram, 2),
            'peak_vram_mb': round(peak_vram, 2),
            'used_vram_mb': round(used_vram, 2)
        }


def find_best_leiden_resolution(adata, celltype_key, seed=42, 
                                 res_min=0.1, res_max=2.0, res_step=0.1,
                                 save_path=None):
    """
    Test Leiden clustering across a range of resolutions and select the one with highest NMI.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with neighbors already computed
    celltype_key : str
        Column name in adata.obs containing cell type labels
    seed : int, optional
        Random seed for reproducibility (default: 42)
    res_min : float, optional
        Minimum resolution to test (default: 0.1)
    res_max : float, optional
        Maximum resolution to test (default: 2.0)
    res_step : float, optional
        Step size for resolution increment (default: 0.1)
    save_path : str, optional
        Path to save resolution search results CSV. If None, results are not saved.
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'best_resolution': Best Leiden resolution
        - 'best_nmi': NMI score at best resolution
        - 'all_resolutions': Array of all tested resolutions
        - 'all_nmi_scores': Array of all NMI scores
        - 'resolution_df': DataFrame with resolution search results
    """
    
    print(f"\n{'='*60}")
    print(f"Testing Leiden clustering resolutions from {res_min} to {res_max}...")
    print(f"{'='*60}\n")
    
    resolutions = np.arange(res_min, res_max + res_step/2, res_step)
    nmi_scores = []
    celltype_labels = adata.obs[celltype_key].astype(str).to_numpy()
    
    for res in resolutions:
        sc.tl.leiden(adata, resolution=res, random_state=seed, 
                     key_added=f"leiden_res_{res:.1f}")
        cluster_labels = adata.obs[f"leiden_res_{res:.1f}"].astype(str).to_numpy()
        nmi = normalized_mutual_info_score(celltype_labels, cluster_labels)
        nmi_scores.append(nmi)
        print(f"Resolution {res:.1f}: NMI = {nmi:.4f}")
    
    # Find best resolution
    best_idx = np.argmax(nmi_scores)
    best_resolution = resolutions[best_idx]
    best_nmi = nmi_scores[best_idx]
    
    print(f"\n{'='*60}")
    print(f"Best resolution: {best_resolution:.1f} with NMI = {best_nmi:.4f}")
    print(f"{'='*60}\n")
    
    # Create results dataframe
    resolution_df = pd.DataFrame({
        'resolution': resolutions,
        'NMI': nmi_scores
    })
    
    # Save if path provided
    if save_path is not None:
        resolution_df.to_csv(save_path, index=False)
        print(f"Resolution search results saved to {save_path}")
    
    # Set the best resolution as the default leiden clustering
    adata.obs['leiden'] = adata.obs[f"leiden_res_{best_resolution:.1f}"]
    
    return {
        'best_resolution': best_resolution,
        'best_nmi': best_nmi,
        'all_resolutions': resolutions,
        'all_nmi_scores': np.array(nmi_scores),
        'resolution_df': resolution_df
    }