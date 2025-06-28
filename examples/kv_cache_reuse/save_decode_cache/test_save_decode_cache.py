"""
Test for save_decode_cache functionality with warmup mechanism
"""

import os
import time
import getpass
import hashlib
import yaml
import re
import sys
from typing import Dict, List, Tuple

def generate_unique_rpc_port():
    """Generate a unique RPC port to avoid conflicts"""
    username = getpass.getuser()
    pid = os.getpid()
    unique_id = hashlib.md5(f"{username}_{pid}".encode()).hexdigest()[:8]
    rpc_port = int(unique_id, 16) % 10000
    print(f"Using RPC port: {rpc_port}")
    return rpc_port

def create_lmcache_config(save_decode_cache: bool, config_suffix: str) -> str:
    """Create LMCache configuration file"""
    config = {
        'chunk_size': 256,
        'local_cpu': True,
        'max_local_cpu_size': 5.0,
        'save_decode_cache': save_decode_cache,
        'remote_serde': 'cachegen'
    }
    
    config_file = f"lmcache_config_{config_suffix}.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    return config_file

def analyze_logs_for_decode_cache(stdout: str, stderr: str) -> Dict:
    """Analyze logs for save_decode_cache evidence"""
    all_logs = stdout + "\n" + stderr
    
    results = {
        'prefill_stores': [],
        'decode_stores': [],
        'cache_hits': [],
        'config_loaded': False,
        'save_decode_cache_enabled': None
    }
    
    # Check if save_decode_cache config was loaded
    config_pattern = r"'save_decode_cache': (True|False)"
    config_matches = re.findall(config_pattern, all_logs)
    if config_matches:
        results['config_loaded'] = True
        results['save_decode_cache_enabled'] = config_matches[-1] == 'True'
    
    # Find prefill cache stores (skip_leading_tokens=0)
    prefill_pattern = r"Storing KV cache for (\d+) out of (\d+) tokens \(skip_leading_tokens=0\)"
    results['prefill_stores'] = re.findall(prefill_pattern, all_logs)
    
    # Find decode cache stores (skip_leading_tokens>0) - KEY EVIDENCE
    decode_pattern = r"Storing KV cache for (\d+) out of (\d+) tokens \(skip_leading_tokens=(\d+)\)"
    all_stores = re.findall(decode_pattern, all_logs)
    results['decode_stores'] = [(stored, total, skip) for stored, total, skip in all_stores if int(skip) > 0]
    
    # Find cache hit messages
    hit_pattern = r"Reqid: \d+, Total tokens (\d+), LMCache hit tokens: (\d+), need to load: (-?\d+)"
    results['cache_hits'] = re.findall(hit_pattern, all_logs)
    
    return results

def run_warmup_inference(rpc_port: int):
    """Run warmup inference to initialize system state"""
    print("System warmup...")
    
    import subprocess
    import tempfile
    
    warmup_script = f"""
import os
import sys
import time
sys.path.insert(0, '{os.getcwd()}')

from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig
from lmcache.v1.cache_engine import LMCacheEngineBuilder
from lmcache.integration.vllm.utils import ENGINE_NAME

# Configure KV cache transfer
ktc = KVTransferConfig(
    kv_connector="LMCacheConnectorV1",
    kv_role="kv_both",
    kv_connector_extra_config={{"lmcache_rpc_port": {rpc_port}}}
)

# Initialize LLM for warmup
llm = LLM(
    model="facebook/opt-1.3b",
    kv_transfer_config=ktc,
    max_model_len=2048,
    gpu_memory_utilization=0.6
)

# Short warmup inference
sampling_params = SamplingParams(temperature=0, max_tokens=5)
_ = llm.generate(["Hello"], sampling_params)

# Cleanup
try:
    LMCacheEngineBuilder.destroy(ENGINE_NAME)
except:
    pass
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(warmup_script)
        temp_script = f.name
    
    try:
        result = subprocess.run([sys.executable, temp_script], 
                              capture_output=True, text=True, timeout=300)
        print("Warmup completed")
    finally:
        os.unlink(temp_script)

def test_decode_cache_directly(config_file: str, test_name: str, rpc_port: int):
    """Test decode cache functionality"""
    
    import subprocess
    import tempfile
    
    test_script = f"""
import os
import sys
import time
sys.path.insert(0, '{os.getcwd()}')

os.environ["LMCACHE_CONFIG_FILE"] = "{config_file}"

try:
    from vllm import LLM, SamplingParams
    from vllm.config import KVTransferConfig
    from lmcache.v1.cache_engine import LMCacheEngineBuilder
    from lmcache.integration.vllm.utils import ENGINE_NAME
    print("Successfully imported all modules")
except Exception as e:
    print(f"Import error: {{e}}")
    sys.exit(1)

try:
    ktc = KVTransferConfig(
        kv_connector="LMCacheConnectorV1",
        kv_role="kv_both",
        kv_connector_extra_config={{"lmcache_rpc_port": {rpc_port}}}
    )
    print("KVTransferConfig created successfully")
except Exception as e:
    print(f"KVTransferConfig error: {{e}}")
    sys.exit(1)

try:
    llm = LLM(
        model="facebook/opt-1.3b",
        kv_transfer_config=ktc,
        max_model_len=2048,
        gpu_memory_utilization=0.6
    )
    print("LLM initialized successfully")
except Exception as e:
    print(f"LLM initialization error: {{e}}")
    sys.exit(1)

base_prompt = "Tell me about the future of technology. " * 40
decode_prompt = base_prompt + " What do you think will happen next?"

sampling_params = SamplingParams(
    temperature=0,
    top_p=0.95,
    max_tokens=30
)

# First inference
try:
    print("Starting first inference...")
    start_time = time.time()
    outputs1 = llm.generate([base_prompt], sampling_params)
    first_time = time.time() - start_time
    print(f"TIMING: First run: {{first_time:.3f}}s")
    print(f"First inference completed, generated {{len(outputs1[0].outputs[0].text)}} characters")
except Exception as e:
    print(f"First inference error: {{e}}")
    sys.exit(1)

# Clear cache after first inference
print("Clearing cache...")
try:
    # Get the cache engine and clear it
    engine = LMCacheEngineBuilder.get(ENGINE_NAME)
    if hasattr(engine, 'clear'):
        engine.clear()
    elif hasattr(engine, 'reset'):
        engine.reset()
    elif hasattr(engine, 'cache_engine'):
        if hasattr(engine.cache_engine, 'clear'):
            engine.cache_engine.clear()
        elif hasattr(engine.cache_engine, 'reset'):
            engine.cache_engine.reset()
    print("Cache cleared")
except Exception as e:
    print(f"Cache clear attempt: {{e}}")

time.sleep(1)

# Second inference (should start fresh)
try:
    print("Starting second inference...")
    start_time = time.time()
    outputs2 = llm.generate([decode_prompt], sampling_params)
    second_time = time.time() - start_time
    print(f"TIMING: Second run: {{second_time:.3f}}s")
    print(f"Second inference completed, generated {{len(outputs2[0].outputs[0].text)}} characters")
except Exception as e:
    print(f"Second inference error: {{e}}")
    sys.exit(1)

try:
    LMCacheEngineBuilder.destroy(ENGINE_NAME)
    print("Engine destroyed successfully")
except Exception as e:
    print(f"Engine destroy error: {{e}}")
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(test_script)
        temp_script = f.name
    
    try:
        print(f"Running {test_name}...")
        result = subprocess.run([sys.executable, temp_script], 
                              capture_output=True, text=True, timeout=600)
        
        stdout = result.stdout
        stderr = result.stderr
        
        # Check for errors
        if result.returncode != 0:
            print(f"Error: Process exited with code {result.returncode}")
            print("STDOUT:", stdout[-500:] if len(stdout) > 500 else stdout)
            print("STDERR:", stderr[-500:] if len(stderr) > 500 else stderr)
        
        # Extract timing information
        timing_pattern = r"TIMING: (\w+) run: ([\d.]+)s"
        timings = dict(re.findall(timing_pattern, stdout))
        first_time = float(timings.get('First', 0))
        second_time = float(timings.get('Second', 0))
        
        # If timing is zero, show debug info
        if first_time == 0 or second_time == 0:
            print(f"Warning: Zero timing detected for {test_name}")
            print("STDOUT output:")
            print(stdout)
            print("STDERR output:")
            print(stderr)
        
        # Analyze logs for decode cache evidence
        log_analysis = analyze_logs_for_decode_cache(stdout, stderr)
        
        return {
            'first_time': first_time,
            'second_time': second_time,
            'speedup': first_time / second_time if second_time > 0 else 1.0,
            'stdout': stdout,
            'stderr': stderr,
            'log_analysis': log_analysis
        }
        
    finally:
        os.unlink(temp_script)

def check_environment():
    """Check if the environment is properly set up"""
    print("Checking environment...")
    
    try:
        import vllm
        print(f"✓ vLLM version: {vllm.__version__}")
    except ImportError as e:
        print(f"✗ vLLM import failed: {e}")
        return False
    
    try:
        import lmcache
        print("✓ LMCache imported successfully")
    except ImportError as e:
        print(f"✗ LMCache import failed: {e}")
        return False
    
    try:
        import torch
        print(f"✓ PyTorch version: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"✓ CUDA available, devices: {torch.cuda.device_count()}")
        else:
            print("⚠ CUDA not available")
    except ImportError as e:
        print(f"✗ PyTorch import failed: {e}")
        return False
    
    return True

def main():
    """Main test function with warmup mechanism"""
    print("save_decode_cache Detection Test")
    print("=" * 40)
    
    # Check environment first
    if not check_environment():
        print("Environment check failed. Please ensure all dependencies are installed.")
        return
    
    rpc_port = generate_unique_rpc_port()
    
    # Test 1: save_decode_cache = True
    print("\nTesting save_decode_cache = True")
    config_enabled = create_lmcache_config(save_decode_cache=True, config_suffix="enabled")
    results_enabled = test_decode_cache_directly(config_enabled, "save_decode_cache=True", rpc_port + 100)
    
    time.sleep(2)
    
    # Test 2: save_decode_cache = False  
    print("\nTesting save_decode_cache = False")
    config_disabled = create_lmcache_config(save_decode_cache=False, config_suffix="disabled")
    results_disabled = test_decode_cache_directly(config_disabled, "save_decode_cache=False", rpc_port + 200)
    
    # Analysis
    print("\n" + "=" * 40)
    print("RESULTS")
    print("=" * 40)
    
    enabled_analysis = results_enabled['log_analysis']
    disabled_analysis = results_disabled['log_analysis']
    
    print(f"\nsave_decode_cache=True:")
    print(f"  Performance: {results_enabled['first_time']:.3f}s → {results_enabled['second_time']:.3f}s (speedup: {results_enabled['speedup']:.2f}x)")
    print(f"  Config loaded: {enabled_analysis['config_loaded']}")
    print(f"  Decode stores: {len(enabled_analysis['decode_stores'])}")
    print(f"  Prefill stores: {len(enabled_analysis['prefill_stores'])}")
    
    print(f"\nsave_decode_cache=False:")
    print(f"  Performance: {results_disabled['first_time']:.3f}s → {results_disabled['second_time']:.3f}s (speedup: {results_disabled['speedup']:.2f}x)")
    print(f"  Config loaded: {disabled_analysis['config_loaded']}")
    print(f"  Decode stores: {len(disabled_analysis['decode_stores'])}")
    print(f"  Prefill stores: {len(disabled_analysis['prefill_stores'])}")
    
    # Evidence analysis
    enabled_decode_stores = len(enabled_analysis['decode_stores'])
    disabled_decode_stores = len(disabled_analysis['decode_stores'])
    
    print(f"\n" + "=" * 40)
    print("VERDICT")
    print("=" * 40)
    
    if enabled_decode_stores > 0 and enabled_decode_stores > disabled_decode_stores:
        print("RESULT: save_decode_cache is WORKING")
        print(f"Evidence: {enabled_decode_stores} decode cache operations when enabled vs {disabled_decode_stores} when disabled")
        
        # Show sample decode operations
        if enabled_analysis['decode_stores']:
            print("Sample decode operations:")
            for i, (stored, total, skip) in enumerate(enabled_analysis['decode_stores'][:3]):
                print(f"  {stored}/{total} tokens (skip_leading_tokens={skip})")
            if len(enabled_analysis['decode_stores']) > 3:
                print(f"  ... and {len(enabled_analysis['decode_stores']) - 3} more")
                
    elif enabled_analysis['config_loaded'] and disabled_analysis['config_loaded']:
        enabled_setting = enabled_analysis['save_decode_cache_enabled']
        disabled_setting = disabled_analysis['save_decode_cache_enabled']
        
        if enabled_setting == True and disabled_setting == False:
            print("RESULT: Configuration loaded correctly but no decode operations detected")
        else:
            print("RESULT: Configuration loading issue")
            print(f"Expected: True/False, Got: {enabled_setting}/{disabled_setting}")
    else:
        print("RESULT: Configuration not detected")
    
    # Cleanup
    try:
        os.remove(config_enabled)
        os.remove(config_disabled)
    except:
        pass

if __name__ == "__main__":
    main() 