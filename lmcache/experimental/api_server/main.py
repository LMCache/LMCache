from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import argparse
import uvicorn


from lmcache.experimental.cache_controller import LMCacheClusterExecutor
from lmcache.logging import init_logger


logger = init_logger(__name__)



def create_app(lmcache_instance_ids: List[str]) -> FastAPI:
    """
    Create a FastAPI application with endpoints for LMCache operations.
    """
    app = FastAPI()
    lmcache_cluster_executor = LMCacheClusterExecutor(lmcache_instance_ids)

    class LookupRequest(BaseModel):
        instance_id: str
        token_ids: List[int]
        worker_ids: List[int] = []

    @app.post("/lookup")
    async def lookup(req: LookupRequest):
        try:
            kwargs = {
                "instance_id": req.instance_id,
                "token_ids": req.token_ids, 
                "worker_ids": req.worker_ids
            }
            return await lmcache_cluster_executor.execute("lookup", **kwargs)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--lmcache-instance-ids", type=List[str], default=["lmcache-instance"])
    args = parser.parse_args()

    try:
        app = create_app(args.lmcache_instance_ids)
        
        logger.info(f"Starting LMCache controller at {args.host}:{args.port}")
        logger.info(f"LMCache instance ids: {args.lmcache_instance_ids}")
        
        uvicorn.run(app, host=args.host, port=args.port, reload=True)
    except TimeoutError as e:
        logger.error(e)