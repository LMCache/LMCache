import torch
from nixl._api import nixl_agent as NixlAgent
import uuid

PAGE_SIZE = 4096  # adjust to your system page size

def register_n_agents(n: int,
                      num_bytes: int,
                      device_id: int = 0,
                      dtype=torch.uint8,
                      mem_type: str = "cuda",
                      page_size: int = PAGE_SIZE):
    """
    Allocate n CPU buffers of size `num_bytes` each,
    create n NixlAgents, register one buffer per agent,
    and prepare XFER handlers for each buffer.

    Returns two lists: [agent0, agent1, …], [buf0, buf1, …], and [xfer0, xfer1, …].
    You must keep the buffers and handlers alive.
    """
    agents = []
    buffers = []
    xfer_handlers = []
    for i in range(n):
        buf = torch.empty(num_bytes, dtype=dtype, device="cuda")

        buffers.append(buf)
        
        # 2. descriptor tuple for this buffer
        ptr = buf.data_ptr()
        size = buf.numel() * buf.element_size()
        desc = [(ptr, size, device_id, "")]
        print(f"device_id is {device_id}")
        
        # 3. new agent + register this single region
        agent_name = str(uuid.uuid4())
        agent = NixlAgent(agent_name)
        reg_descs = agent.get_reg_descs(desc, mem_type=mem_type)
        agent.register_memory(reg_descs)
        
        # 4. prepare transfer descriptors and handler
        xfer_desc_list = []
        for base in range(ptr, ptr + size, page_size):
            xfer_desc_list.append((base, page_size, device_id))
        xfer_descs = agent.get_xfer_descs(xfer_desc_list, mem_type=mem_type)
        xfer_handler = agent.prep_xfer_dlist(agent_name, xfer_descs, mem_type=mem_type)
        xfer_handlers.append(xfer_handler)
        
        agents.append(agent)
    
    return agents, buffers, xfer_handlers

# -----------------------
# Example usage:

if __name__ == "__main__":
    num_agents = 2             # how many NixlAgent instances
    buffer_size = PAGE_SIZE * 100    # size in bytes per buffer
    
    agents, bufs, handlers = register_n_agents(
        n=num_agents,
        num_bytes=buffer_size,
        device_id=0,
        dtype=torch.uint8,
        mem_type="cuda",
        page_size=PAGE_SIZE
    )
    
    print(f"Created {num_agents} agents, each with a {buffer_size}‑byte buffer and XFER handler.")
